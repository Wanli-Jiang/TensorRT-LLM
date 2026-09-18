# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash-recoverable PlanDrafter to PlanReviewer controller loop.

This module is deterministic controller code, not an agent role.  A tick
performs at most one externally meaningful reconciliation step, and every
scheduler submission is fenced by a durable ``SUBMITTING`` attempt record.
Role processes receive immutable inputs and can only publish a response file;
they never write authoritative workflow state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Literal, Mapping, cast

from agent_flow.config import CODEX_DEFAULT_MODEL

from ..common.artifacts import (
    INPUT_FILENAME,
    WorkerInputManifest,
    digest_file,
    load_input_manifest,
    worker_output_directory,
    write_input_manifest,
)
from ..common.credentials import (
    ALL_BACKEND_CREDENTIAL_NAMES,
    CredentialBinding,
    CredentialBroker,
    CredentialDescriptor,
    CredentialProvision,
    CredentialRevocationCause,
    CredentialState,
    descriptor_from_public_dict,
    locate_credential_provision,
    validate_unique_credential_handles,
)
from ..common.isolation import (
    WorkerIsolation,
    attach_agent_credential_provision,
    build_worker_isolation,
)
from ..common.outcomes import (
    OutcomeError,
    PlanDraftBlockedOutcome,
    PlanDraftOutcome,
    PlanReviewDecision,
    PlanReviewOutcome,
    parse_plan_draft,
    parse_plan_review,
    planning_outcome_instruction,
)
from ..common.policy import WorkItemKind as PolicyWorkItemKind
from ..common.runners import BackendKind, RoleProcessResult, RoleProcessSpec, validate_role_spec
from ..common.slurm import JobIdentity, JobStatus, Mount, ResourceRequest, Scheduler, SchedulerError
from ..common.submission_recovery import Clock as SubmissionClock
from ..common.submission_recovery import (
    SubmissionCancellationAction,
    SubmissionRecoveryAction,
    SubmissionRecoveryError,
    SubmissionRecoveryPolicy,
    decide_submission_cancellation,
    decide_submission_recovery,
    load_submission_cancellation,
    record_manual_recovery,
    record_probe_error,
    record_submission_outcome,
    submission_intent_digest,
    submission_recovery_lock,
)
from ..prompts.plan_drafter import SYSTEM_PROMPT as PLAN_DRAFTER_SYSTEM_PROMPT
from ..prompts.plan_reviewer import SYSTEM_PROMPT as PLAN_REVIEWER_SYSTEM_PROMPT
from ..state import (
    PLANNING_ITEM_ID,
    STATE_FILENAME,
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    GoalRecord,
    HierarchyStatus,
    JobReference,
    Role,
    RunState,
    RunTerminalStatus,
    StageRecord,
    StageStatus,
    WorkflowMode,
    WorkItemKind,
    WorkItemRecord,
    load_state,
    save_state,
)
from ..task_schema import NormalizedTask, normalized_task_json
from .attempts import AttemptEngineError, cancel_owned_attempt
from .role_policy import ResolvedRolePolicy, RolePolicyError, resolve_role_policy

_PRODUCTION_WORK_ITEM_KINDS = frozenset(
    {
        PolicyWorkItemKind.CATALOG_VERIFY,
        PolicyWorkItemKind.CATALOG_ONBOARD,
        PolicyWorkItemKind.ASSEMBLE_CORE,
        PolicyWorkItemKind.ASSEMBLE_FEATURE,
        PolicyWorkItemKind.ROUTING,
        PolicyWorkItemKind.TUNE_HYPOTHESIS,
    }
)

_ROLE_INPUT_FILENAME = "role-input.json"
_ROLE_RESULT_FILENAME = "role-result.json"
_PLANNING_PROFILE = "planner"
# RoleProcessSpec IDs must start with an alphanumeric character.  Authoritative
# state retains its reserved ``__planning__`` item identity; this is only the
# isolated process transport identity and is checked together with attempt ID.
_ROLE_ITEM_ID = "planning"


class PlanningError(RuntimeError):
    """Raised when the planning controller cannot safely make progress."""


class PlanningPhase(str, Enum):
    """Durable high-level planning phase derived from authoritative state."""

    DRAFTING = "drafting"
    REVIEWING = "reviewing"
    ADMITTED = "admitted"
    BLOCKED_INPUT = "blocked_input"
    EXHAUSTED = "exhausted"
    CANCELLED = "cancelled"
    TERMINAL = "terminal"


class PlanningEvent(str, Enum):
    """Typed event completed by one controller tick."""

    DRAFT_PREPARED = "draft_prepared"
    REVIEW_PREPARED = "review_prepared"
    SUBMITTED = "submitted"
    ADOPTED = "adopted"
    SCHEDULER_UPDATED = "scheduler_updated"
    TERMINAL_OBSERVED = "terminal_observed"
    RESULT_VALIDATED = "result_validated"
    RESULT_REJECTED = "result_rejected"
    PLAN_ADMITTED = "plan_admitted"
    RUN_BLOCKED = "run_blocked"
    MANUAL_RECOVERY_REQUIRED = "manual_recovery_required"
    RUN_EXHAUSTED = "run_exhausted"
    RUN_CANCELLED = "run_cancelled"
    NO_CHANGE = "no_change"


class PlanningAction(str, Enum):
    """Typed next action for the outer deterministic controller."""

    TICK = "tick"
    RECONCILE = "reconcile"
    WAIT_FOR_SCHEDULER = "wait_for_scheduler"
    START_WORKFLOW = "start_workflow"
    STOP = "stop"


@dataclass(frozen=True, slots=True)
class PlanningConfig:
    """Bounded role and retry policy for the planning loop."""

    backend_kind: BackendKind = "codex"
    model: str = CODEX_DEFAULT_MODEL
    drafter_resource_class: str = "coder_analysis"
    reviewer_resource_class: str = "reviewer_analysis"
    max_draft_attempts: int = 3
    max_review_attempts_per_draft: int = 2

    def __post_init__(self) -> None:
        if self.backend_kind not in {"claude-code", "codex"}:
            raise ValueError(f"unsupported planning backend {self.backend_kind!r}")
        if not self.model.strip():
            raise ValueError("planning model must be non-empty")
        if not self.drafter_resource_class.strip() or not self.reviewer_resource_class.strip():
            raise ValueError("planning resource-class names must be non-empty")
        if self.max_draft_attempts < 1:
            raise ValueError("max_draft_attempts must be positive")
        if self.max_review_attempts_per_draft < 1:
            raise ValueError("max_review_attempts_per_draft must be positive")


@dataclass(frozen=True, slots=True)
class PlanningTickResult:
    """Typed result of exactly one planning-controller tick."""

    phase: PlanningPhase
    event: PlanningEvent
    next_action: PlanningAction
    revision: int
    attempt_id: str | None = None
    job: JobIdentity | None = None
    scheduler_status: JobStatus | None = None
    review_decision: PlanReviewDecision | None = None


class PlanningEngine:
    """Single-writer, tick-based durable planning controller.

    The caller must already own the controller lease for ``generation``.  This
    class independently enforces the same generation and revision fences on
    every state write so a stale controller cannot admit a plan.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        task: NormalizedTask,
        scheduler: Scheduler,
        generation: int,
        config: PlanningConfig | None = None,
        credential_broker: CredentialBroker | None = None,
        backend_environment: Mapping[str, str] | None = None,
        submission_clock: SubmissionClock | None = None,
        submission_recovery_policy: SubmissionRecoveryPolicy | None = None,
    ) -> None:
        self._workspace = workspace.resolve(strict=True)
        self._task = task
        self._scheduler = scheduler
        self._generation = generation
        task_agent = task.execution.agent
        self._config = config or PlanningConfig(
            backend_kind=task_agent.backend_kind,
            model=task_agent.model,
        )
        if (
            self._config.backend_kind != task_agent.backend_kind
            or self._config.model != task_agent.model
        ):
            raise PlanningError(
                "PlanningConfig backend/model must equal frozen task.execution.agent identity"
            )
        if credential_broker is None:
            raise PlanningError(
                "planning requires an explicit CredentialBroker; pass NoCredentialBroker only "
                "for task policies with no allowed credential names"
            )
        self._credential_broker = credential_broker
        if backend_environment is not None:
            leaked_names = sorted(
                ALL_BACKEND_CREDENTIAL_NAMES.intersection(backend_environment.keys())
            )
            if leaked_names:
                raise PlanningError(
                    "deterministic planning controller cannot receive backend credential values; "
                    f"configure a trusted CredentialBroker handle instead: {leaked_names!r}"
                )
        self._backend_environment = {} if backend_environment is None else backend_environment
        self._submission_clock = submission_clock or (lambda: datetime.now(UTC))
        self._submission_recovery_policy = submission_recovery_policy or SubmissionRecoveryPolicy()
        self._state_path = self._workspace / STATE_FILENAME
        repository_root = task.repository.root.resolve(strict=True)
        self._repository_root = repository_root
        try:
            self._workspace.relative_to(task.repository.workspace_root.resolve(strict=True))
        except ValueError as error:
            raise PlanningError(
                "planning workspace is outside repository.workspace_root"
            ) from error
        configured_classes = {
            resource.name for resource in task.execution.slurm.smith.resource_classes
        }
        required_classes = {
            self._config.drafter_resource_class,
            self._config.reviewer_resource_class,
        }
        missing_classes = required_classes - configured_classes
        if missing_classes:
            raise PlanningError(
                f"planning resource classes are not configured: {sorted(missing_classes)!r}"
            )

    def tick(
        self,
        *,
        cancel_requested: bool = False,
        cancellation_reason: str = "planning cancelled by controller",
    ) -> PlanningTickResult:
        """Reconcile and advance the planning loop by one deterministic step.

        Args:
            cancel_requested: Whether the owning controller is cancelling this run.
            cancellation_reason: Durable non-empty cancellation reason.

        Returns:
            A typed event and next action; no control decision is parsed from prose.
        """
        state = self._load_owned_state()
        if cancel_requested:
            return self._cancel(state, cancellation_reason)
        if state.terminal_status is not RunTerminalStatus.ACTIVE:
            self._revoke_planning_credentials(state)
        terminal = self._terminal_result(state)
        if terminal is not None:
            return terminal
        if state.items:
            return self._result(
                state,
                PlanningPhase.ADMITTED,
                PlanningEvent.NO_CHANGE,
                PlanningAction.START_WORKFLOW,
            )
        if not state.planning_attempts:
            return self._prepare_drafter(state, corrections=())

        latest = state.planning_attempts[-1]
        if latest.scheduler_state == "SUBMISSION_MANUAL_RECOVERY":
            self._revoke_attempt_credentials(state, latest)
            return self._result(
                state,
                self._phase_for_role(latest.role),
                PlanningEvent.MANUAL_RECOVERY_REQUIRED,
                PlanningAction.STOP,
                attempt=latest,
            )
        if latest.status not in {
            AttemptStatus.VALIDATED,
            AttemptStatus.RETRYABLE_FAILED,
            AttemptStatus.FAILED,
            AttemptStatus.PREEMPTED,
            AttemptStatus.CANCELLED,
            AttemptStatus.LOST,
        }:
            return self._drive_attempt(state, latest)
        self._revoke_attempt_credentials(state, latest)
        if latest.status is AttemptStatus.CANCELLED:
            return self._result(
                state,
                self._phase_for_role(latest.role),
                PlanningEvent.NO_CHANGE,
                PlanningAction.STOP,
                attempt=latest,
            )
        if latest.status is not AttemptStatus.VALIDATED:
            return self._retry_or_exhaust(state, latest)
        if latest.role is Role.PLAN_DRAFTER:
            draft = self._load_draft(latest, workflow_mode=state.workflow_mode.value)
            if isinstance(draft, PlanDraftBlockedOutcome):
                return self._block_input(state, f"PlanDrafter: {draft.reason}")
            return self._prepare_reviewer(state, latest, draft)

        review = self._load_review(latest)
        if review.outcome is PlanReviewDecision.ACCEPT:
            draft_attempt = self._planning_attempt(state, cast(str, latest.review_of_attempt_id))
            draft = self._load_draft(draft_attempt, workflow_mode=state.workflow_mode.value)
            if isinstance(draft, PlanDraftBlockedOutcome):
                raise PlanningError("PlanReviewer accepted a blocked PlanDrafter outcome")
            return self._admit(state, latest, draft)
        if review.outcome is PlanReviewDecision.BLOCK:
            return self._block_input(
                state,
                f"PlanReviewer: {'; '.join(review.corrections)}",
            )
        if self._draft_attempt_count(state) >= self._config.max_draft_attempts:
            return self._exhaust(state, "plan_revision_limit_exhausted")
        return self._prepare_drafter(state, corrections=review.corrections)

    def _load_owned_state(self) -> RunState:
        state = load_state(self._state_path)
        if state.task_digest != self._task.digest:
            raise PlanningError("normalized task digest does not match authoritative state")
        if state.generation != self._generation:
            raise PlanningError(
                f"controller generation fence failed: state={state.generation}, "
                f"writer={self._generation}"
            )
        return state

    def _terminal_result(self, state: RunState) -> PlanningTickResult | None:
        if state.terminal_status is RunTerminalStatus.ACTIVE:
            return None
        if state.terminal_status is RunTerminalStatus.EXHAUSTED:
            phase = PlanningPhase.EXHAUSTED
        elif state.terminal_status is RunTerminalStatus.BLOCKED_INPUT:
            phase = PlanningPhase.BLOCKED_INPUT
        elif state.terminal_status is RunTerminalStatus.CANCELLED:
            phase = PlanningPhase.CANCELLED
        else:
            phase = PlanningPhase.TERMINAL
        return self._result(state, phase, PlanningEvent.NO_CHANGE, PlanningAction.STOP)

    def _prepare_drafter(
        self, state: RunState, *, corrections: tuple[str, ...]
    ) -> PlanningTickResult:
        if self._draft_attempt_count(state) >= self._config.max_draft_attempts:
            return self._exhaust(state, "plan_draft_attempt_limit_exhausted")
        sequence = len(state.planning_attempts) + 1
        attempt = AttemptRecord(
            attempt_id=f"plan-draft-{sequence:04d}",
            item_id=PLANNING_ITEM_ID,
            sequence=sequence,
            role=Role.PLAN_DRAFTER,
            kind=AttemptKind.ROLE,
            generation=state.generation,
            resource_class=self._config.drafter_resource_class,
        )
        prompt = self._drafter_prompt(corrections, workflow_mode=state.workflow_mode.value)
        self._ensure_attempt_inputs(attempt, PLAN_DRAFTER_SYSTEM_PROMPT, prompt)
        updated = replace(
            state,
            planning_attempts=(*state.planning_attempts, attempt),
            revision=state.revision + 1,
        )
        self._save(state, updated)
        return self._result(
            updated,
            PlanningPhase.DRAFTING,
            PlanningEvent.DRAFT_PREPARED,
            PlanningAction.TICK,
            attempt=attempt,
        )

    def _prepare_reviewer(
        self,
        state: RunState,
        draft_attempt: AttemptRecord,
        draft: PlanDraftOutcome,
    ) -> PlanningTickResult:
        self._validate_production_plan(draft, workflow_mode=state.workflow_mode)
        sequence = len(state.planning_attempts) + 1
        attempt = AttemptRecord(
            attempt_id=f"plan-review-{sequence:04d}",
            item_id=PLANNING_ITEM_ID,
            sequence=sequence,
            role=Role.PLAN_REVIEWER,
            kind=AttemptKind.REVIEWER_ANALYSIS,
            generation=state.generation,
            resource_class=self._config.reviewer_resource_class,
            candidate_digest=draft.digest,
            review_of_attempt_id=draft_attempt.attempt_id,
            reviewed_candidate_digest=draft.digest,
        )
        prompt = self._reviewer_prompt(
            draft_attempt, draft, workflow_mode=state.workflow_mode.value
        )
        self._ensure_attempt_inputs(attempt, PLAN_REVIEWER_SYSTEM_PROMPT, prompt)
        updated = replace(
            state,
            planning_attempts=(*state.planning_attempts, attempt),
            revision=state.revision + 1,
        )
        self._save(state, updated)
        return self._result(
            updated,
            PlanningPhase.REVIEWING,
            PlanningEvent.REVIEW_PREPARED,
            PlanningAction.TICK,
            attempt=attempt,
        )

    def _drive_attempt(self, state: RunState, attempt: AttemptRecord) -> PlanningTickResult:
        if attempt.status is AttemptStatus.PREPARED:
            return self._submit(state, attempt)
        if attempt.status is AttemptStatus.SUBMITTING:
            return self._adopt(state, attempt)
        if attempt.status in {
            AttemptStatus.SUBMITTED,
            AttemptStatus.PENDING,
            AttemptStatus.RUNNING,
        }:
            return self._observe(state, attempt)
        if attempt.status is AttemptStatus.TERMINAL_OBSERVED:
            return self._handle_terminal(state, attempt)
        if attempt.status is AttemptStatus.COLLECTING:
            return self._collect(state, attempt)
        raise PlanningError(f"unsupported non-terminal planning status {attempt.status.value!r}")

    def _submit(self, state: RunState, attempt: AttemptRecord) -> PlanningTickResult:
        policy = self._role_policy(attempt)
        descriptor = self._credential_descriptor(attempt, policy, require_live=True)
        provision = self._credential_broker.inject_after_admission(
            workspace=self._workspace,
            descriptor=descriptor,
        )
        if provision.descriptor != descriptor:
            raise PlanningError("credential broker changed the admitted public handle")
        self._worker_isolation(policy, attempt, provision=provision)
        token = self._submission_token(state, attempt)
        submitting = attempt.transition(AttemptStatus.SUBMITTING, submission_token=token)
        state = self._replace_attempt(state, attempt, submitting)
        return self._submit_or_adopt(state, submitting)

    def _submit_or_adopt(
        self,
        state: RunState,
        attempt: AttemptRecord,
    ) -> PlanningTickResult:
        token = cast(str, attempt.submission_token)
        journal_dir = self._workspace / "receipts" / "submissions" / "planning" / attempt.attempt_id
        # Adoption must not depend on rematerializing an already-running
        # worker's credential bundle.  Probe under the same token journal lock
        # before reconstructing a possible retry submission.
        with submission_recovery_lock(journal_dir):
            try:
                visible = self._scheduler.probe_submission(token)
            except SchedulerError:
                visible = None
            if visible is not None and visible.identity is not None:
                ownership = self._scheduler.verify_ownership(visible.identity, token)
                if ownership.matched:
                    adopted = attempt.transition(
                        AttemptStatus.SUBMITTED,
                        job=self._state_job(visible.identity),
                        scheduler_state=visible.status.value,
                        terminal_reason=None,
                    )
                    updated = self._replace_attempt(state, attempt, adopted)
                    return self._result(
                        updated,
                        self._phase_for_role(attempt.role),
                        PlanningEvent.ADOPTED,
                        PlanningAction.RECONCILE,
                        attempt=adopted,
                        job=visible.identity,
                        scheduler_status=visible.status,
                    )
        policy = self._role_policy(attempt)
        descriptor = self._credential_descriptor(attempt, policy, require_live=False)
        provision = locate_credential_provision(
            workspace=self._workspace,
            descriptor=descriptor,
        )
        if provision.descriptor != descriptor:
            raise PlanningError("credential broker changed the admitted public handle")
        isolation = self._worker_isolation(policy, attempt, provision=provision)
        command = isolation.worker_command(self._attempt_dir(attempt) / _ROLE_INPUT_FILENAME)
        resources = self._resources(attempt, isolation, policy)
        environment = isolation.environment.mapping
        intent_revision = f"{attempt.generation}:{attempt.sequence}:{attempt.attempt_id}"
        intent_digest = submission_intent_digest(
            intent_revision,
            digest_file(self._attempt_dir(attempt) / _ROLE_INPUT_FILENAME),
            resources,
            command,
            environment,
        )
        with submission_recovery_lock(journal_dir):
            try:
                probe = self._scheduler.probe_submission(token)
            except SchedulerError as error:
                decision = record_probe_error(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=str(error),
                    clock=self._submission_clock,
                    policy=self._submission_recovery_policy,
                )
                if decision.action is SubmissionRecoveryAction.MANUAL_RECOVERY:
                    return self._manual_submission_recovery(state, attempt, decision.reason)
                waiting = replace(
                    attempt,
                    scheduler_state="SUBMISSION_PROBE_ERROR",
                    terminal_reason=decision.reason,
                )
                updated = (
                    state if waiting == attempt else self._replace_attempt(state, attempt, waiting)
                )
                return self._result(
                    updated,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.NO_CHANGE,
                    PlanningAction.WAIT_FOR_SCHEDULER,
                    attempt=waiting,
                )
            if len(probe.matches) > 1:
                reason = f"immutable planning submission token matched {len(probe.matches)} jobs"
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=reason,
                    clock=self._submission_clock,
                )
                return self._manual_submission_recovery(state, attempt, reason)
            if probe.matches and probe.identity is None:
                reason = probe.reason or "planning scheduler evidence is untrustworthy"
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=reason,
                    clock=self._submission_clock,
                )
                return self._manual_submission_recovery(state, attempt, reason)
            if probe.identity is not None:
                ownership = self._scheduler.verify_ownership(probe.identity, token)
                if not ownership.matched:
                    reason = f"planning submission ownership is ambiguous: {ownership.reason}"
                    record_manual_recovery(
                        journal_dir=journal_dir,
                        submission_token=token,
                        intent_revision=intent_revision,
                        intent_digest=intent_digest,
                        reason=reason,
                        clock=self._submission_clock,
                    )
                    return self._manual_submission_recovery(state, attempt, reason)
                adopted = attempt.transition(
                    AttemptStatus.SUBMITTED,
                    job=self._state_job(probe.identity),
                    scheduler_state=probe.status.value,
                    terminal_reason=None,
                )
                updated = self._replace_attempt(state, attempt, adopted)
                return self._result(
                    updated,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.ADOPTED,
                    PlanningAction.RECONCILE,
                    attempt=adopted,
                    job=probe.identity,
                    scheduler_status=probe.status,
                )
            try:
                decision = decide_submission_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    probe=probe,
                    clock=self._submission_clock,
                    policy=self._submission_recovery_policy,
                )
            except SubmissionRecoveryError as error:
                return self._manual_submission_recovery(
                    state,
                    attempt,
                    f"invalid recovery journal: {error}",
                )
            if decision.action is SubmissionRecoveryAction.MANUAL_RECOVERY:
                return self._manual_submission_recovery(state, attempt, decision.reason)
            if decision.action is SubmissionRecoveryAction.WAIT:
                waiting = replace(
                    attempt,
                    scheduler_state="SUBMISSION_RECOVERY_WAIT",
                    terminal_reason=decision.reason,
                )
                updated = (
                    state if waiting == attempt else self._replace_attempt(state, attempt, waiting)
                )
                return self._result(
                    updated,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.NO_CHANGE,
                    PlanningAction.WAIT_FOR_SCHEDULER,
                    attempt=waiting,
                )
            claim_sequence = decision.claim_sequence
            if claim_sequence is None:
                raise PlanningError("submission recovery omitted its scheduler-call claim")
            if descriptor.handle is not None and descriptor.handle.expires_at_epoch_seconds <= int(
                self._submission_clock().timestamp()
            ):
                reason = "planning credential expired before the scheduler launch boundary"
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=reason,
                    clock=self._submission_clock,
                )
                return self._manual_submission_recovery(state, attempt, reason)
            try:
                identity = self._scheduler.submit(
                    resources,
                    command,
                    token,
                    environment=environment,
                )
            except SchedulerError as error:
                record_submission_outcome(
                    journal_dir=journal_dir,
                    claim_sequence=claim_sequence,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    outcome="uncertain",
                    clock=self._submission_clock,
                )
                waiting = replace(
                    attempt,
                    scheduler_state="SUBMISSION_UNCERTAIN",
                    terminal_reason=(
                        "scheduler submission returned without a durable identity; "
                        f"bounded token reconciliation is required: {error}"
                    ),
                )
                updated = self._replace_attempt(state, attempt, waiting)
                return self._result(
                    updated,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.NO_CHANGE,
                    PlanningAction.WAIT_FOR_SCHEDULER,
                    attempt=waiting,
                )
            record_submission_outcome(
                journal_dir=journal_dir,
                claim_sequence=claim_sequence,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                outcome="accepted",
                clock=self._submission_clock,
                scheduler_id=identity.scheduler_id,
            )
            ownership = self._scheduler.verify_ownership(identity, token)
            if not ownership.matched:
                reason = f"submitted planning ownership is ambiguous: {ownership.reason}"
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=reason,
                    clock=self._submission_clock,
                )
                return self._manual_submission_recovery(state, attempt, reason)
            submitted = attempt.transition(
                AttemptStatus.SUBMITTED,
                job=self._state_job(identity),
                scheduler_state=JobStatus.PENDING.value,
                terminal_reason=None,
            )
            updated = self._replace_attempt(state, attempt, submitted)
            return self._result(
                updated,
                self._phase_for_role(attempt.role),
                PlanningEvent.SUBMITTED,
                PlanningAction.RECONCILE,
                attempt=submitted,
                job=identity,
            )

    def _adopt(self, state: RunState, attempt: AttemptRecord) -> PlanningTickResult:
        return self._submit_or_adopt(state, attempt)

    def _manual_submission_recovery(
        self,
        state: RunState,
        attempt: AttemptRecord,
        reason: str,
    ) -> PlanningTickResult:
        failed = attempt.transition(
            AttemptStatus.FAILED,
            scheduler_state="SUBMISSION_MANUAL_RECOVERY",
            terminal_reason=f"manual scheduler reconciliation required: {reason}",
        )
        updated = self._replace_attempt(state, attempt, failed)
        self._revoke_attempt_credentials(updated, failed)
        return self._result(
            updated,
            self._phase_for_role(attempt.role),
            PlanningEvent.MANUAL_RECOVERY_REQUIRED,
            PlanningAction.STOP,
            attempt=failed,
        )

    def _observe(self, state: RunState, attempt: AttemptRecord) -> PlanningTickResult:
        identity = self._scheduler_job(cast(JobReference, attempt.job))
        if attempt.submission_token is None:
            raise PlanningError("submitted planning attempt has no immutable token")
        try:
            observation = self._scheduler.observe_owned(identity, attempt.submission_token)
        except SchedulerError:
            return self._result(
                state,
                self._phase_for_role(attempt.role),
                PlanningEvent.NO_CHANGE,
                PlanningAction.WAIT_FOR_SCHEDULER,
                attempt=attempt,
                job=identity,
                scheduler_status=JobStatus.UNKNOWN,
            )
        if observation.status is JobStatus.UNKNOWN:
            return self._result(
                state,
                self._phase_for_role(attempt.role),
                PlanningEvent.NO_CHANGE,
                PlanningAction.WAIT_FOR_SCHEDULER,
                attempt=attempt,
                job=identity,
                scheduler_status=observation.status,
            )
        if observation.status is JobStatus.PENDING:
            if attempt.status is not AttemptStatus.SUBMITTED:
                return self._result(
                    state,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.NO_CHANGE,
                    PlanningAction.WAIT_FOR_SCHEDULER,
                    attempt=attempt,
                    job=identity,
                    scheduler_status=observation.status,
                )
            next_attempt = attempt.transition(
                AttemptStatus.PENDING, scheduler_state=observation.status.value
            )
            action = PlanningAction.WAIT_FOR_SCHEDULER
            event = PlanningEvent.SCHEDULER_UPDATED
        elif observation.status is JobStatus.RUNNING:
            if attempt.status is AttemptStatus.RUNNING:
                return self._result(
                    state,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.NO_CHANGE,
                    PlanningAction.WAIT_FOR_SCHEDULER,
                    attempt=attempt,
                    job=identity,
                    scheduler_status=observation.status,
                )
            next_attempt = attempt.transition(
                AttemptStatus.RUNNING, scheduler_state=observation.status.value
            )
            action = PlanningAction.WAIT_FOR_SCHEDULER
            event = PlanningEvent.SCHEDULER_UPDATED
        else:
            next_attempt = attempt.transition(
                AttemptStatus.TERMINAL_OBSERVED,
                scheduler_state=observation.status.value,
            )
            action = PlanningAction.TICK
            event = PlanningEvent.TERMINAL_OBSERVED
        updated = self._replace_attempt(state, attempt, next_attempt)
        return self._result(
            updated,
            self._phase_for_role(attempt.role),
            event,
            action,
            attempt=next_attempt,
            job=identity,
            scheduler_status=observation.status,
        )

    def _handle_terminal(self, state: RunState, attempt: AttemptRecord) -> PlanningTickResult:
        scheduler_status = JobStatus(cast(str, attempt.scheduler_state))
        if scheduler_status is JobStatus.COMPLETED:
            collecting = attempt.transition(AttemptStatus.COLLECTING)
            updated = self._replace_attempt(state, attempt, collecting)
            return self._collect(updated, collecting)
        terminal_status = {
            JobStatus.CANCELLED: AttemptStatus.CANCELLED,
            JobStatus.PREEMPTED: AttemptStatus.PREEMPTED,
            JobStatus.NODE_FAIL: AttemptStatus.RETRYABLE_FAILED,
        }.get(scheduler_status, AttemptStatus.FAILED)
        failed = attempt.transition(
            terminal_status,
            terminal_reason=f"scheduler_{scheduler_status.value.lower()}",
        )
        updated = self._replace_attempt(state, attempt, failed)
        self._revoke_attempt_credentials(updated, failed)
        return self._result(
            updated,
            self._phase_for_role(attempt.role),
            PlanningEvent.RESULT_REJECTED,
            PlanningAction.TICK,
            attempt=failed,
            scheduler_status=scheduler_status,
        )

    def _collect(self, state: RunState, attempt: AttemptRecord) -> PlanningTickResult:
        try:
            role_result, result_digest = self._load_role_result(attempt)
            if attempt.role is Role.PLAN_DRAFTER:
                outcome = parse_plan_draft(
                    role_result.response,
                    allowed_resource_classes=self._resource_class_names(),
                    allowed_path_roots=self._allowed_path_roots(),
                    workflow_mode=state.workflow_mode.value,
                    target_features=self._task.target.features,
                )
                if isinstance(outcome, PlanDraftOutcome):
                    self._validate_production_plan(
                        outcome,
                        workflow_mode=state.workflow_mode,
                    )
                candidate_digest = outcome.digest if isinstance(outcome, PlanDraftOutcome) else None
                decision = None
            else:
                outcome = parse_plan_review(
                    role_result.response,
                    expected_plan_digest=cast(str, attempt.reviewed_candidate_digest),
                )
                candidate_digest = attempt.candidate_digest
                decision = outcome.outcome
        except (OSError, json.JSONDecodeError, ValueError, OutcomeError, PlanningError) as error:
            failed = attempt.transition(
                AttemptStatus.FAILED,
                terminal_reason=f"invalid_role_result:{type(error).__name__}",
            )
            updated = self._replace_attempt(state, attempt, failed)
            self._revoke_attempt_credentials(updated, failed)
            return self._result(
                updated,
                self._phase_for_role(attempt.role),
                PlanningEvent.RESULT_REJECTED,
                PlanningAction.TICK,
                attempt=failed,
            )
        validated = attempt.transition(
            AttemptStatus.VALIDATED,
            result_digest=result_digest,
            candidate_digest=candidate_digest,
        )
        updated = self._replace_attempt(state, attempt, validated)
        self._revoke_attempt_credentials(updated, validated)
        return self._result(
            updated,
            self._phase_for_role(attempt.role),
            PlanningEvent.RESULT_VALIDATED,
            PlanningAction.TICK,
            attempt=validated,
            review_decision=decision,
        )

    def _retry_or_exhaust(self, state: RunState, attempt: AttemptRecord) -> PlanningTickResult:
        if attempt.role is Role.PLAN_DRAFTER:
            if self._draft_attempt_count(state) >= self._config.max_draft_attempts:
                return self._exhaust(state, "plan_draft_attempt_limit_exhausted")
            return self._prepare_drafter(state, corrections=())
        draft_attempt = self._planning_attempt(state, cast(str, attempt.review_of_attempt_id))
        draft = self._load_draft(draft_attempt, workflow_mode=state.workflow_mode.value)
        if isinstance(draft, PlanDraftBlockedOutcome):
            return self._block_input(state, f"PlanDrafter: {draft.reason}")
        reviewer_count = sum(
            entry.role is Role.PLAN_REVIEWER
            and entry.review_of_attempt_id == draft_attempt.attempt_id
            for entry in state.planning_attempts
        )
        if reviewer_count >= self._config.max_review_attempts_per_draft:
            return self._exhaust(state, "plan_reviewer_attempt_limit_exhausted")
        return self._prepare_reviewer(state, draft_attempt, draft)

    def _admit(
        self,
        state: RunState,
        review_attempt: AttemptRecord,
        draft: PlanDraftOutcome,
    ) -> PlanningTickResult:
        if review_attempt.reviewed_candidate_digest != draft.digest:
            raise PlanningError("approved reviewer is not pinned to the admitted plan digest")
        self._validate_production_plan(draft, workflow_mode=state.workflow_mode)
        goals_by_id = {goal.goal_id: goal for goal in draft.goals}
        stages = tuple(
            StageRecord(stage.stage_id, stage.goal_ids, StageStatus.PLANNED)
            for stage in draft.stages
        )
        goals = tuple(
            GoalRecord(goal.goal_id, goal.stage_id, goal.item_ids, HierarchyStatus.PLANNED)
            for goal in draft.goals
        )
        items = tuple(
            WorkItemRecord(
                item_id=item.item_id,
                stage_id=goals_by_id[item.goal_id].stage_id,
                goal_id=item.goal_id,
                kind=WorkItemKind(item.kind.value),
                profile=self._profile_for_kind(item.kind.value),
                dependencies=item.dependencies,
            )
            for item in draft.items
        )
        admitted = replace(
            state,
            stages=stages,
            goals=goals,
            items=items,
            revision=state.revision + 1,
        )
        self._save(state, admitted)
        return self._result(
            admitted,
            PlanningPhase.ADMITTED,
            PlanningEvent.PLAN_ADMITTED,
            PlanningAction.START_WORKFLOW,
            attempt=review_attempt,
            review_decision=PlanReviewDecision.ACCEPT,
        )

    def _cancel(self, state: RunState, reason: str) -> PlanningTickResult:
        if not reason.strip():
            raise ValueError("cancellation_reason must be non-empty")
        if state.items:
            raise PlanningError("planning cancellation cannot close an admitted workflow")
        manual = next(
            (
                attempt
                for attempt in state.planning_attempts
                if attempt.scheduler_state == "SUBMISSION_MANUAL_RECOVERY"
            ),
            None,
        )
        if manual is not None:
            return self._result(
                state,
                self._phase_for_role(manual.role),
                PlanningEvent.MANUAL_RECOVERY_REQUIRED,
                PlanningAction.STOP,
                attempt=manual,
            )
        attempts: list[AttemptRecord] = []
        for attempt in state.planning_attempts:
            if attempt.terminal:
                attempts.append(attempt)
                continue
            current = attempt
            if current.status is AttemptStatus.SUBMITTING and current.job is None:
                return self._cancel_submitting_without_job(state, current, reason)
            identity = None if current.job is None else self._scheduler_job(current.job)
            if identity is not None:
                if current.job is None:
                    current = replace(current, job=self._state_job(identity))
                try:
                    cancel_owned_attempt(current, self._scheduler)
                except AttemptEngineError as error:
                    raise PlanningError(
                        f"planning cancellation ownership verification failed closed: {error}"
                    ) from error
            attempts.append(
                current.transition(AttemptStatus.CANCELLED, terminal_reason=reason.strip())
            )
        cancelled = replace(
            state,
            planning_attempts=tuple(attempts),
            terminal_status=RunTerminalStatus.CANCELLED,
            terminal_reason=reason.strip(),
            revision=state.revision + 1,
        )
        self._save(state, cancelled)
        self._revoke_planning_credentials(cancelled)
        return self._result(
            cancelled,
            PlanningPhase.CANCELLED,
            PlanningEvent.RUN_CANCELLED,
            PlanningAction.STOP,
        )

    def _cancel_submitting_without_job(
        self,
        state: RunState,
        attempt: AttemptRecord,
        reason: str,
    ) -> PlanningTickResult:
        token = cast(str, attempt.submission_token)
        policy = self._role_policy(attempt)
        descriptor = self._credential_descriptor(attempt, policy, require_live=False)
        provision = locate_credential_provision(
            workspace=self._workspace,
            descriptor=descriptor,
        )
        isolation = self._worker_isolation(policy, attempt, provision=provision)
        command = isolation.worker_command(self._attempt_dir(attempt) / _ROLE_INPUT_FILENAME)
        resources = self._resources(attempt, isolation, policy)
        environment = isolation.environment.mapping
        intent_revision = f"{attempt.generation}:{attempt.sequence}:{attempt.attempt_id}"
        intent_digest = submission_intent_digest(
            intent_revision,
            digest_file(self._attempt_dir(attempt) / _ROLE_INPUT_FILENAME),
            resources,
            command,
            environment,
        )
        journal_dir = self._workspace / "receipts" / "submissions" / "planning" / attempt.attempt_id
        with submission_recovery_lock(journal_dir):
            recorded = load_submission_cancellation(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
            )
            if recorded is not None:
                if recorded.action is SubmissionCancellationAction.MANUAL_RECOVERY:
                    return self._manual_submission_recovery(state, attempt, recorded.reason)
                cancelled = attempt.transition(
                    AttemptStatus.CANCELLED,
                    scheduler_state="SUBMISSION_INTENT_CANCELLED",
                    terminal_reason=reason.strip(),
                )
                updated = self._replace_attempt(state, attempt, cancelled)
                self._revoke_attempt_credentials(updated, cancelled)
                return self._result(
                    updated,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.RUN_CANCELLED,
                    PlanningAction.TICK,
                    attempt=cancelled,
                )
            try:
                probe = self._scheduler.probe_submission(token)
            except SchedulerError as error:
                failure = f"planning cancellation probe failed: {error}"
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=failure,
                    clock=self._submission_clock,
                )
                return self._manual_submission_recovery(state, attempt, failure)
            if len(probe.matches) > 1 or (probe.matches and probe.identity is None):
                failure = probe.reason or "planning cancellation evidence is ambiguous"
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=failure,
                    clock=self._submission_clock,
                )
                return self._manual_submission_recovery(state, attempt, failure)
            if probe.identity is not None:
                ownership = self._scheduler.verify_ownership(probe.identity, token)
                if not ownership.matched:
                    failure = f"planning cancellation ownership is ambiguous: {ownership.reason}"
                    record_manual_recovery(
                        journal_dir=journal_dir,
                        submission_token=token,
                        intent_revision=intent_revision,
                        intent_digest=intent_digest,
                        reason=failure,
                        clock=self._submission_clock,
                    )
                    return self._manual_submission_recovery(state, attempt, failure)
                adopted = attempt.transition(
                    AttemptStatus.SUBMITTED,
                    job=self._state_job(probe.identity),
                    scheduler_state=probe.status.value,
                    terminal_reason=None,
                )
                updated = self._replace_attempt(state, attempt, adopted)
                return self._result(
                    updated,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.ADOPTED,
                    PlanningAction.RECONCILE,
                    attempt=adopted,
                    job=probe.identity,
                    scheduler_status=probe.status,
                )
            try:
                decision = decide_submission_cancellation(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    probe=probe,
                    clock=self._submission_clock,
                    policy=self._submission_recovery_policy,
                )
            except SubmissionRecoveryError as error:
                return self._manual_submission_recovery(
                    state,
                    attempt,
                    f"invalid cancellation recovery journal: {error}",
                )
            if decision.action is SubmissionCancellationAction.MANUAL_RECOVERY:
                return self._manual_submission_recovery(state, attempt, decision.reason)
            if decision.action is SubmissionCancellationAction.WAIT:
                waiting = replace(
                    attempt,
                    scheduler_state="SUBMISSION_CANCELLATION_WAIT",
                    terminal_reason=decision.reason,
                )
                updated = (
                    state if waiting == attempt else self._replace_attempt(state, attempt, waiting)
                )
                return self._result(
                    updated,
                    self._phase_for_role(attempt.role),
                    PlanningEvent.NO_CHANGE,
                    PlanningAction.WAIT_FOR_SCHEDULER,
                    attempt=waiting,
                )
            cancelled = attempt.transition(
                AttemptStatus.CANCELLED,
                scheduler_state="SUBMISSION_INTENT_CANCELLED",
                terminal_reason=reason.strip(),
            )
            updated = self._replace_attempt(state, attempt, cancelled)
            self._revoke_attempt_credentials(updated, cancelled)
            return self._result(
                updated,
                self._phase_for_role(attempt.role),
                PlanningEvent.RUN_CANCELLED,
                PlanningAction.TICK,
                attempt=cancelled,
            )

    def _exhaust(self, state: RunState, reason: str) -> PlanningTickResult:
        exhausted = state.finish(RunTerminalStatus.EXHAUSTED, reason)
        self._save(state, exhausted)
        self._revoke_planning_credentials(exhausted)
        return self._result(
            exhausted,
            PlanningPhase.EXHAUSTED,
            PlanningEvent.RUN_EXHAUSTED,
            PlanningAction.STOP,
        )

    def _block_input(self, state: RunState, reason: str) -> PlanningTickResult:
        blocked = state.finish(RunTerminalStatus.BLOCKED_INPUT, reason)
        self._save(state, blocked)
        self._revoke_planning_credentials(blocked)
        return self._result(
            blocked,
            PlanningPhase.BLOCKED_INPUT,
            PlanningEvent.RUN_BLOCKED,
            PlanningAction.STOP,
        )

    def _replace_attempt(
        self,
        state: RunState,
        previous: AttemptRecord,
        desired: AttemptRecord,
    ) -> RunState:
        attempts = tuple(
            desired if entry.attempt_id == previous.attempt_id else entry
            for entry in state.planning_attempts
        )
        if attempts == state.planning_attempts:
            raise PlanningError(f"unknown planning attempt {previous.attempt_id!r}")
        updated = replace(state, planning_attempts=attempts, revision=state.revision + 1)
        self._save(state, updated)
        return updated

    def _save(self, previous: RunState, desired: RunState) -> None:
        save_state(
            self._state_path,
            desired,
            expected_revision=previous.revision,
            expected_generation=self._generation,
        )

    def _ensure_attempt_inputs(
        self,
        attempt: AttemptRecord,
        system_prompt: str,
        prompt: str,
    ) -> None:
        attempt_dir = self._attempt_dir(attempt)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        worker_output_directory(attempt_dir).mkdir(parents=True, exist_ok=True)
        policy = self._role_policy(attempt)
        descriptor = self._credential_descriptor(attempt, policy)
        isolation = self._worker_isolation(
            policy,
            attempt,
            descriptor=descriptor,
            input_bundle_prepublication=True,
        )
        host_role_spec = validate_role_spec(
            RoleProcessSpec(
                schema_version=1,
                run_id=self._load_owned_state().run_id,
                task_digest=self._task.digest,
                generation=attempt.generation,
                item_id=_ROLE_ITEM_ID,
                attempt_id=attempt.attempt_id,
                prompt_id=attempt.attempt_id,
                role=cast(Literal["plan_drafter", "plan_reviewer"], attempt.role.value),
                profile=_PLANNING_PROFILE,
                backend_kind=self._config.backend_kind,
                model=self._config.model,
                source_root=str(self._repository_root),
                cwd=str(self._repository_root),
                result_path=str(worker_output_directory(attempt_dir) / _ROLE_RESULT_FILENAME),
                system_prompt=system_prompt,
                prompt=prompt,
                credential_descriptor=descriptor.to_public_dict(),
                launch_policy=policy.launch_policy.to_public_dict(),
            )
        )
        role_spec = isolation.to_worker_role_spec(host_role_spec)
        role_input = attempt_dir / _ROLE_INPUT_FILENAME
        self._ensure_json(role_input, cast(dict[str, object], asdict(role_spec)))
        worker_input = WorkerInputManifest(
            run_id=role_spec.run_id,
            item_id=PLANNING_ITEM_ID,
            attempt_id=role_spec.attempt_id,
            task_digest=role_spec.task_digest,
            generation=attempt.generation,
            role=attempt.role,
            profile=None,
            worktree=role_spec.cwd,
            payload={
                "role_input": _ROLE_INPUT_FILENAME,
                "prompt_id": role_spec.prompt_id,
            },
        )
        artifact_input = attempt_dir / INPUT_FILENAME
        if artifact_input.exists():
            persisted, _digest = load_input_manifest(artifact_input)
            if persisted != worker_input:
                raise PlanningError(f"immutable worker input drift for {attempt.attempt_id!r}")
        else:
            write_input_manifest(artifact_input, worker_input)

    def _load_role_result(self, attempt: AttemptRecord) -> tuple[RoleProcessResult, str]:
        attempt_dir = self._attempt_dir(attempt)
        result_path = worker_output_directory(attempt_dir) / _ROLE_RESULT_FILENAME
        if result_path.is_symlink() or not result_path.is_file():
            raise PlanningError("scheduler completed without a regular role result")
        value = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise PlanningError("role result must be a JSON object")
        expected_keys = set(RoleProcessResult.__dataclass_fields__)
        if set(value) != expected_keys:
            raise PlanningError(
                "role result keys invalid; "
                f"missing={sorted(expected_keys - set(value))}, "
                f"unknown={sorted(set(value) - expected_keys)}"
            )
        result = RoleProcessResult(**value)
        spec = self._load_host_role_spec(attempt)
        expected_identity = (
            spec.schema_version,
            spec.run_id,
            spec.task_digest,
            spec.generation,
            spec.item_id,
            spec.attempt_id,
            spec.prompt_id,
            spec.role,
            spec.profile,
        )
        actual_identity = (
            result.schema_version,
            result.run_id,
            result.task_digest,
            result.generation,
            result.item_id,
            result.attempt_id,
            result.prompt_id,
            result.role,
            result.profile,
        )
        if actual_identity != expected_identity:
            raise PlanningError("role result identity does not match immutable input")
        response_digest = hashlib.sha256(result.response.encode("utf-8")).hexdigest()
        if result.response_digest != response_digest:
            raise PlanningError("role response digest does not match response bytes")
        return result, digest_file(result_path)

    def _load_draft(
        self,
        attempt: AttemptRecord,
        *,
        workflow_mode: str,
    ) -> PlanDraftOutcome | PlanDraftBlockedOutcome:
        if attempt.role is not Role.PLAN_DRAFTER or attempt.status is not AttemptStatus.VALIDATED:
            raise PlanningError("plan draft source is not a validated PlanDrafter attempt")
        result, _digest = self._load_role_result(attempt)
        outcome = parse_plan_draft(
            result.response,
            allowed_resource_classes=self._resource_class_names(),
            allowed_path_roots=self._allowed_path_roots(),
            workflow_mode=workflow_mode,
            target_features=self._task.target.features,
        )
        if isinstance(outcome, PlanDraftOutcome):
            if outcome.digest != attempt.candidate_digest:
                raise PlanningError("persisted draft digest differs from the validated attempt")
        elif attempt.candidate_digest is not None:
            raise PlanningError("blocked PlanDrafter outcome cannot carry a candidate digest")
        return outcome

    def _load_review(self, attempt: AttemptRecord) -> PlanReviewOutcome:
        if attempt.role is not Role.PLAN_REVIEWER or attempt.status is not AttemptStatus.VALIDATED:
            raise PlanningError("plan review source is not a validated PlanReviewer attempt")
        result, _digest = self._load_role_result(attempt)
        return parse_plan_review(
            result.response,
            expected_plan_digest=cast(str, attempt.reviewed_candidate_digest),
        )

    def _drafter_prompt(self, corrections: tuple[str, ...], *, workflow_mode: str) -> str:
        correction_payload = json.dumps(
            list(corrections), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return "\n\n".join(
            (
                "Create the typed execution plan for this immutable normalized task:",
                normalized_task_json(self._task),
                f"Frozen workflow mode: {workflow_mode}",
                (
                    "Production WorkItem adapter invariant: kind must be one of "
                    "catalog_verify, catalog_onboard, assemble_core, assemble_feature, "
                    "routing, or tune_hypothesis. Do not emit feasibility, search, or "
                    "standalone gate WorkItems; deterministic gates are controller-owned "
                    "subrequests of an admitted candidate."
                ),
                f"Prior PlanReviewer corrections: {correction_payload}",
                planning_outcome_instruction(Role.PLAN_DRAFTER.value),
            )
        )

    def _reviewer_prompt(
        self,
        draft_attempt: AttemptRecord,
        draft: PlanDraftOutcome,
        *,
        workflow_mode: str,
    ) -> str:
        draft_result, _digest = self._load_role_result(draft_attempt)
        return "\n\n".join(
            (
                "Review this immutable normalized task:",
                normalized_task_json(self._task),
                f"Frozen workflow mode: {workflow_mode}",
                (
                    "Reject any WorkItem kind outside the production adapter set: "
                    "catalog_verify, catalog_onboard, assemble_core, assemble_feature, "
                    "routing, and tune_hypothesis. feasibility, search, and standalone "
                    "gate WorkItems have no production dispatch adapter."
                ),
                f"Frozen plan digest: {draft.digest}",
                "Frozen PlanDrafter JSON:",
                draft_result.response,
                planning_outcome_instruction(Role.PLAN_REVIEWER.value, plan_digest=draft.digest),
            )
        )

    def _resources(
        self,
        attempt: AttemptRecord,
        isolation: WorkerIsolation,
        policy: ResolvedRolePolicy,
    ) -> ResourceRequest:
        resource = policy.resource
        controller = self._task.execution.slurm.controller
        log_root = self._workspace / "slurm" / "planning" / attempt.attempt_id
        log_root.mkdir(parents=True, exist_ok=True)
        return ResourceRequest(
            label=attempt.attempt_id,
            account=controller.account,
            partition=resource.partition or controller.partition,
            qos=resource.qos if resource.qos is not None else controller.qos,
            reservation=controller.reservation,
            time_limit=self._slurm_time(resource.time_limit_seconds),
            output_path=log_root / "%j.out",
            error_path=log_root / "%j.err",
            nodes=resource.nodes,
            tasks_per_node=resource.tasks_per_node,
            cpus_per_task=resource.cpus_per_task,
            gpus_per_node=resource.gpus_per_node,
            memory_mb=resource.memory_mib,
            requeue=False,
            container_image=policy.launch_policy.image,
            mounts=isolation.mounts,
            container_launch_mode=controller.container_launch_mode.value,
            agent_policy_digest=policy.launch_policy.digest,
        )

    def _worker_isolation(
        self,
        policy: ResolvedRolePolicy,
        attempt: AttemptRecord,
        *,
        descriptor: CredentialDescriptor | None = None,
        provision: CredentialProvision | None = None,
        input_bundle_prepublication: bool = False,
    ) -> WorkerIsolation:
        controller = self._task.execution.slurm.controller
        base = build_worker_isolation(
            controller_mounts=tuple(
                Mount(mount.host_path, Path(mount.container_path), mount.read_only)
                for mount in controller.mounts
            ),
            workspace=self._workspace,
            repository=self._repository_root,
            mailbox=worker_output_directory(self._attempt_dir(attempt)),
            input_bundle=self._attempt_dir(attempt) / _ROLE_INPUT_FILENAME,
            checkpoint=self._task.reference.checkpoint,
            reference_sources=self._task.reference.additional_sources,
            environment=policy.environment,
            input_bundle_prepublication=input_bundle_prepublication,
        )
        if provision is not None:
            return attach_agent_credential_provision(base, provision)
        if descriptor is None:
            raise PlanningError("planning isolation lacks a credential descriptor")
        return replace(base, credential_descriptor=descriptor)

    def _role_policy(self, attempt: AttemptRecord) -> ResolvedRolePolicy:
        try:
            return resolve_role_policy(
                self._task,
                role=attempt.role,
                profile=attempt.profile,
                resource_class=attempt.resource_class,
                ambient_environment=self._backend_environment,
            )
        except RolePolicyError as error:
            raise PlanningError(f"planning role policy failed closed: {error}") from error

    def _credential_binding(self, attempt: AttemptRecord) -> CredentialBinding:
        state = self._load_owned_state()
        return CredentialBinding(
            run_id=state.run_id,
            item_id=_ROLE_ITEM_ID,
            attempt_id=attempt.attempt_id,
            task_digest=self._task.digest,
            generation=attempt.generation,
            backend_kind=self._task.execution.agent.backend_kind,
        )

    @staticmethod
    def _validate_credential_descriptor(
        descriptor: CredentialDescriptor,
        binding: CredentialBinding,
        policy: ResolvedRolePolicy,
        *,
        require_live: bool,
    ) -> None:
        if descriptor.binding != binding:
            raise PlanningError("credential broker handle is not bound to the exact attempt")
        broker_policy = policy.credential_broker_policy
        if descriptor.credential_names != broker_policy.allowed_credential_names:
            raise PlanningError("credential descriptor names differ from task role policy")
        if not broker_policy.allowed_credential_names:
            if descriptor.state is not CredentialState.NONE or descriptor.handle is not None:
                raise PlanningError(
                    "preauthenticated role policy requires an explicit no-credential descriptor"
                )
            return
        if descriptor.state is not CredentialState.BUNDLE or descriptor.handle is None:
            raise PlanningError("credential-bearing role policy requires a broker handle")
        if descriptor.handle.broker_id != broker_policy.broker_id:
            raise PlanningError("credential descriptor broker differs from task role policy")
        now = int(time.time())
        expiry = descriptor.handle.expires_at_epoch_seconds
        if require_live and expiry <= now:
            raise PlanningError("credential descriptor is already expired")
        if expiry > now + broker_policy.per_attempt_ttl_seconds:
            raise PlanningError("credential descriptor expiry exceeds task role policy TTL")

    def _credential_descriptor(
        self,
        attempt: AttemptRecord,
        policy: ResolvedRolePolicy,
        *,
        require_live: bool = True,
    ) -> CredentialDescriptor:
        """Recover the immutable handle, or mint it before input publication."""
        role_input = self._attempt_dir(attempt) / _ROLE_INPUT_FILENAME
        if role_input.is_file() and not role_input.is_symlink():
            value = json.loads(role_input.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise PlanningError("role input must be a JSON object")
            descriptor = descriptor_from_public_dict(value.get("credential_descriptor"))
        else:
            descriptor = self._credential_broker.describe(self._credential_binding(attempt))
        self._validate_credential_descriptor(
            descriptor,
            self._credential_binding(attempt),
            policy,
            require_live=require_live,
        )
        persisted: list[CredentialDescriptor] = []
        for candidate in self._load_owned_state().planning_attempts:
            path = self._attempt_dir(candidate) / _ROLE_INPUT_FILENAME
            if not path.is_file() or path.is_symlink():
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise PlanningError("role input must be a JSON object")
            candidate_descriptor = descriptor_from_public_dict(value.get("credential_descriptor"))
            if candidate_descriptor.binding != descriptor.binding:
                persisted.append(candidate_descriptor)
        validate_unique_credential_handles((*persisted, descriptor))
        return descriptor

    def _load_host_role_spec(
        self,
        attempt: AttemptRecord,
        *,
        policy: ResolvedRolePolicy | None = None,
    ) -> RoleProcessSpec:
        path = self._attempt_dir(attempt) / _ROLE_INPUT_FILENAME
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise PlanningError("role input must be a JSON object")
        expected_keys = set(RoleProcessSpec.__dataclass_fields__)
        if set(value) != expected_keys:
            raise PlanningError(
                "role input keys invalid; "
                f"missing={sorted(expected_keys - set(value))}, "
                f"unknown={sorted(set(value) - expected_keys)}"
            )
        try:
            worker_spec = RoleProcessSpec(**value)
            resolved = policy or self._role_policy(attempt)
            descriptor = self._credential_descriptor(attempt, resolved, require_live=False)
            return self._worker_isolation(
                resolved,
                attempt,
                descriptor=descriptor,
            ).to_host_role_spec(worker_spec)
        except (TypeError, ValueError) as error:
            raise PlanningError(f"invalid isolated role input: {error}") from error

    def _revoke_attempt_credentials(
        self,
        state: RunState,
        attempt: AttemptRecord,
        *,
        cause: CredentialRevocationCause | None = None,
    ) -> None:
        """Revoke one exact durable terminal role bundle and publish a receipt."""
        policy = self._role_policy(attempt)
        spec = self._load_host_role_spec(attempt, policy=policy)
        descriptor = descriptor_from_public_dict(spec.credential_descriptor)
        binding = self._credential_binding(attempt)
        if binding.run_id != state.run_id:
            raise PlanningError("credential binding differs from authoritative run")
        self._validate_credential_descriptor(
            descriptor,
            binding,
            policy,
            require_live=False,
        )
        revocation_cause = cause or self._credential_revocation_cause(attempt)
        receipt = self._credential_broker.revoke(
            workspace=self._workspace,
            expected_binding=binding,
            descriptor=descriptor,
            cause=revocation_cause,
        )
        if receipt.descriptor != descriptor or receipt.cause is not revocation_cause:
            raise PlanningError("credential broker returned a mismatched revocation receipt")

    @staticmethod
    def _credential_revocation_cause(
        attempt: AttemptRecord,
    ) -> CredentialRevocationCause:
        if attempt.status is AttemptStatus.CANCELLED:
            return CredentialRevocationCause.CANCELLED
        if attempt.status in {
            AttemptStatus.PREEMPTED,
            AttemptStatus.RETRYABLE_FAILED,
            AttemptStatus.LOST,
        }:
            return CredentialRevocationCause.REPLACED
        return CredentialRevocationCause.TERMINAL

    def _revoke_planning_credentials(self, state: RunState) -> None:
        """Idempotently revoke every explicit provision in a terminal run."""
        for attempt in state.planning_attempts:
            self._revoke_attempt_credentials(state, attempt)

    def _allowed_path_roots(self) -> tuple[str, ...]:
        family = self._task.target.family
        return (
            "tensorrt_llm/_torch/modeling_v2/catalog",
            f"tensorrt_llm/_torch/modeling_v2/models/{family}",
            "tensorrt_llm/_torch/modeling_v2/_router_index.py",
            "tests/unittest/_torch/modeling_v2",
            "tests/integration/defs/accuracy",
            "tests/integration/test_lists/test-db",
        )

    def _resource_class_names(self) -> tuple[str, ...]:
        return tuple(
            resource.name for resource in self._task.execution.slurm.smith.resource_classes
        )

    def _attempt_dir(self, attempt: AttemptRecord) -> Path:
        return self._workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id

    def _planning_attempt(self, state: RunState, attempt_id: str) -> AttemptRecord:
        for attempt in state.planning_attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise PlanningError(f"unknown linked planning attempt {attempt_id!r}")

    @staticmethod
    def _draft_attempt_count(state: RunState) -> int:
        return sum(attempt.role is Role.PLAN_DRAFTER for attempt in state.planning_attempts)

    @staticmethod
    def _profile_for_kind(kind: str) -> DomainProfile:
        if kind == WorkItemKind.TUNE_HYPOTHESIS.value:
            return DomainProfile.TUNER
        if kind in {
            WorkItemKind.ASSEMBLE_CORE.value,
            WorkItemKind.ASSEMBLE_FEATURE.value,
            WorkItemKind.ROUTING.value,
            WorkItemKind.GATE.value,
        }:
            return DomainProfile.ASSEMBLER
        return DomainProfile.SMITH

    @staticmethod
    def _validate_production_plan(
        draft: PlanDraftOutcome,
        *,
        workflow_mode: WorkflowMode,
    ) -> None:
        """Require complete production adapters before review or admission."""
        unsupported = tuple(
            sorted(
                {
                    item.kind.value
                    for item in draft.items
                    if item.kind not in _PRODUCTION_WORK_ITEM_KINDS
                }
            )
        )
        if unsupported:
            raise PlanningError(
                "production WorkItem adapter invariant rejects unsupported kinds before "
                f"admission: {list(unsupported)!r}; deterministic gates are candidate "
                "subrequests, not standalone WorkItems"
            )
        if workflow_mode is WorkflowMode.TUNE:
            assembler = tuple(
                sorted(
                    item.item_id
                    for item in draft.items
                    if item.kind
                    in {
                        PolicyWorkItemKind.ASSEMBLE_CORE,
                        PolicyWorkItemKind.ASSEMBLE_FEATURE,
                        PolicyWorkItemKind.ROUTING,
                    }
                )
            )
            if assembler:
                raise PlanningError(
                    "production WorkItem adapter invariant rejects Assembler items in tune "
                    f"mode before admission: {list(assembler)!r}"
                )

    @staticmethod
    def _phase_for_role(role: Role) -> PlanningPhase:
        return PlanningPhase.DRAFTING if role is Role.PLAN_DRAFTER else PlanningPhase.REVIEWING

    @staticmethod
    def _submission_token(state: RunState, attempt: AttemptRecord) -> str:
        material = f"{state.run_id}:{attempt.attempt_id}:{state.task_digest}".encode("utf-8")
        return f"planning-{hashlib.sha256(material).hexdigest()[:32]}"

    @staticmethod
    def _state_job(identity: JobIdentity) -> JobReference:
        return JobReference(
            identity.job_id,
            array_task_id=identity.array_task_id,
            cluster=identity.cluster,
        )

    @staticmethod
    def _scheduler_job(reference: JobReference) -> JobIdentity:
        return JobIdentity(
            reference.job_id,
            array_task_id=reference.array_task_id,
            cluster=reference.cluster,
        )

    @staticmethod
    def _slurm_time(seconds: int) -> str:
        days, remainder = divmod(seconds, 86_400)
        hours, remainder = divmod(remainder, 3_600)
        minutes, final_seconds = divmod(remainder, 60)
        prefix = f"{days}-" if days else ""
        return f"{prefix}{hours:02d}:{minutes:02d}:{final_seconds:02d}"

    @staticmethod
    def _ensure_json(path: Path, payload: dict[str, object]) -> None:
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise PlanningError(f"immutable controller artifact drift at {path}")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(payload, output, allow_nan=False, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing != payload:
                    raise PlanningError(f"immutable controller artifact drift at {path}")
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _result(
        state: RunState,
        phase: PlanningPhase,
        event: PlanningEvent,
        action: PlanningAction,
        *,
        attempt: AttemptRecord | None = None,
        job: JobIdentity | None = None,
        scheduler_status: JobStatus | None = None,
        review_decision: PlanReviewDecision | None = None,
    ) -> PlanningTickResult:
        return PlanningTickResult(
            phase=phase,
            event=event,
            next_action=action,
            revision=state.revision,
            attempt_id=attempt.attempt_id if attempt is not None else None,
            job=job,
            scheduler_status=scheduler_status,
            review_decision=review_decision,
        )


__all__ = [
    "PlanningAction",
    "PlanningConfig",
    "PlanningEngine",
    "PlanningError",
    "PlanningEvent",
    "PlanningPhase",
    "PlanningTickResult",
]
