# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic submission and reconciliation for one Staircase attempt.

The functions in this module advance at most one durable boundary per call.
They never sleep and never invoke a shell. In particular, a ``PREPARED`` tick
only returns the ``SUBMITTING`` record that the controller must persist; the
next tick may call the scheduler. This makes submission intent durable before
``sbatch`` and lets a restarted controller adopt a job by its immutable token.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from agent_flow.workflows.staircase.common.artifacts import (
    COMPLETE_FILENAME,
    ArtifactError,
    DuplicateResultError,
    IncompleteResultError,
    ResultExpectation,
    StaleResultError,
    WorkerResultManifest,
    WorkerResultStatus,
    ingest_result,
    load_result_manifest,
    quarantine_result,
)
from agent_flow.workflows.staircase.common.slurm import (
    InternalCommand,
    JobIdentity,
    JobObservation,
    JobStatus,
    ResourceRequest,
    Scheduler,
    SchedulerError,
)
from agent_flow.workflows.staircase.common.submission_recovery import Clock as SubmissionClock
from agent_flow.workflows.staircase.common.submission_recovery import (
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
from agent_flow.workflows.staircase.state import (
    AttemptRecord,
    AttemptStatus,
    JobReference,
    Role,
    WorkItemRecord,
)


class AttemptEngineError(RuntimeError):
    """Base error for invalid deterministic attempt-controller operations."""


class RetryLimitExceeded(AttemptEngineError):
    """Raised when an infrastructure retry would exceed its configured bound."""


class InfrastructureRetryCategory(str, Enum):
    """The only scheduler failures eligible for automatic retry."""

    PREEMPTED = "preempted"
    NODE_FAIL = "node_fail"


class AttemptAction(str, Enum):
    """Observable controller action produced by one attempt tick."""

    PERSIST_SUBMITTING = "persist_submitting"
    SUBMITTED = "submitted"
    ADOPTED = "adopted"
    OBSERVED = "observed"
    WAITING_FOR_ACCOUNTING = "waiting_for_accounting"
    TERMINAL_OBSERVED = "terminal_observed"
    COLLECTING = "collecting"
    VALIDATED = "validated"
    TERMINAL_FAILURE = "terminal_failure"
    DUPLICATE_SUBMISSION = "duplicate_submission"
    QUARANTINED = "quarantined"
    CANCELLED_INTENT = "cancelled_intent"
    NOOP = "noop"


@dataclass(frozen=True)
class AttemptPolicy:
    """Bounded reconciliation and infrastructure-retry policy."""

    accounting_unknown_grace_ticks: int = 2
    max_preempted_retries: int = 2
    max_node_fail_retries: int = 2

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            self.accounting_unknown_grace_ticks,
            "accounting_unknown_grace_ticks",
        )
        _require_nonnegative_int(self.max_preempted_retries, "max_preempted_retries")
        _require_nonnegative_int(self.max_node_fail_retries, "max_node_fail_retries")

    def retry_budget(self, category: InfrastructureRetryCategory) -> int:
        """Return the independent task-policy budget for one category."""
        if category is InfrastructureRetryCategory.PREEMPTED:
            return self.max_preempted_retries
        return self.max_node_fail_retries


@dataclass(frozen=True)
class InfrastructureRetryPlan:
    """Pure durable plan for one idempotent infrastructure retry append."""

    category: InfrastructureRetryCategory
    predecessor_attempt_id: str
    retries_used: int
    retry_budget: int
    attempt: AttemptRecord
    already_appended: bool = False


@dataclass(frozen=True)
class AttemptExecution:
    """Controller-owned scheduler and mailbox inputs for one attempt."""

    resources: ResourceRequest
    command: InternalCommand
    submission_token: str
    attempt_dir: Path
    expectation: ResultExpectation
    receipt_root: Path
    quarantine_root: Path
    environment: tuple[tuple[str, str], ...] = ()
    credential_expires_at_epoch_seconds: int | None = None

    def __post_init__(self) -> None:
        if not self.submission_token.strip():
            raise ValueError("submission_token must be non-empty")
        names = [name for name, _ in self.environment]
        if len(set(names)) != len(names):
            raise ValueError("attempt environment contains duplicate names")
        if names != sorted(names):
            raise ValueError("attempt environment must be sorted by name")
        expiry = self.credential_expires_at_epoch_seconds
        if expiry is not None and (
            isinstance(expiry, bool) or not isinstance(expiry, int) or expiry < 1
        ):
            raise ValueError("credential expiry must be a positive epoch second")

    @property
    def environment_mapping(self) -> Mapping[str, str]:
        """Return the scheduler environment without exposing mutable state."""
        return dict(self.environment)


@dataclass(frozen=True)
class AttemptTickResult:
    """Result of one non-blocking deterministic attempt tick."""

    attempt: AttemptRecord
    action: AttemptAction
    unknown_observations: int = 0
    worker_result: WorkerResultManifest | None = None
    duplicate_jobs: tuple[JobIdentity, ...] = ()
    quarantined_path: Path | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.unknown_observations, "unknown_observations")


def tick_attempt(
    attempt: AttemptRecord,
    *,
    scheduler: Scheduler,
    execution: AttemptExecution,
    policy: AttemptPolicy | None = None,
    unknown_observations: int = 0,
    submission_clock: SubmissionClock | None = None,
    submission_recovery_policy: SubmissionRecoveryPolicy | None = None,
) -> AttemptTickResult:
    """Advance one attempt by at most one durable lifecycle boundary.

    Args:
        attempt: Latest persisted attempt record.
        scheduler: Typed scheduler boundary owned by the controller.
        execution: Immutable scheduler and mailbox inputs for this attempt.
        policy: Bounded accounting-grace and retry settings, or the defaults.
        unknown_observations: Consecutive ``UNKNOWN`` observations retained by
            the caller between ticks. A controller restart safely resets this
            value and therefore extends, rather than shortens, the grace period.

    Returns:
        The next record, action, and cursor data to persist before another tick.
    """
    effective_policy = policy or AttemptPolicy()
    _validate_execution_identity(attempt, execution)
    _require_nonnegative_int(unknown_observations, "unknown_observations")

    if attempt.terminal:
        return _handle_terminal_attempt(attempt, execution, unknown_observations)
    if attempt.status is AttemptStatus.PREPARED:
        submitting = attempt.transition(
            AttemptStatus.SUBMITTING,
            submission_token=execution.submission_token,
            scheduler_state="SUBMISSION_INTENT",
        )
        return AttemptTickResult(submitting, AttemptAction.PERSIST_SUBMITTING)
    if attempt.submission_token != execution.submission_token:
        raise AttemptEngineError("execution token differs from persisted submission token")
    if attempt.status is AttemptStatus.SUBMITTING:
        return _submit_or_adopt(
            attempt,
            scheduler,
            execution,
            clock=submission_clock or (lambda: datetime.now(UTC)),
            recovery_policy=submission_recovery_policy or SubmissionRecoveryPolicy(),
        )
    if attempt.status in {
        AttemptStatus.SUBMITTED,
        AttemptStatus.PENDING,
        AttemptStatus.RUNNING,
    }:
        return _observe_known_job(
            attempt,
            scheduler,
            policy=effective_policy,
            unknown_observations=unknown_observations,
        )
    if attempt.status is AttemptStatus.TERMINAL_OBSERVED:
        return _map_terminal_observation(attempt)
    if attempt.status is AttemptStatus.COLLECTING:
        return _collect_completed_result(attempt, execution)
    raise AttemptEngineError(f"unsupported non-terminal attempt state: {attempt.status.value}")


def create_infrastructure_retry(
    item: WorkItemRecord,
    failed_attempt_id: str,
    *,
    new_attempt_id: str,
    generation: int,
    policy: AttemptPolicy | None = None,
) -> WorkItemRecord:
    """Append a fresh attempt after a bounded infrastructure failure.

    Historical attempts are retained unchanged. The retry starts at
    ``PREPARED`` with no token or scheduler identity; a later tick must persist
    ``SUBMITTING`` before it can submit. Worker/application failure is not
    eligible for this automatic policy.

    Args:
        item: Work item containing the failed attempt history.
        failed_attempt_id: Exact failed attempt selected for retry.
        new_attempt_id: Controller-allocated immutable identity.
        generation: Active controller generation for the new attempt.
        policy: Retry bound, or the defaults.

    Returns:
        A replacement work item with the new attempt appended.
    """
    plan = plan_infrastructure_retry(
        item,
        failed_attempt_id,
        new_attempt_id=new_attempt_id,
        generation=generation,
        policy=policy,
    )
    if plan.already_appended:
        return item
    return item.add_attempt(plan.attempt)


def cancel_submitting_intent(
    attempt: AttemptRecord,
    *,
    scheduler: Scheduler,
    execution: AttemptExecution,
    cancellation_reason: str,
    submission_clock: SubmissionClock | None = None,
    submission_recovery_policy: SubmissionRecoveryPolicy | None = None,
) -> AttemptTickResult:
    """Adopt a visible job or durably cancel one SUBMITTING intent without submitting."""
    if attempt.status is not AttemptStatus.SUBMITTING or attempt.job is not None:
        raise AttemptEngineError("intent cancellation requires SUBMITTING without a job")
    if not cancellation_reason.strip():
        raise AttemptEngineError("intent cancellation reason must be non-empty")
    _validate_execution_identity(attempt, execution)
    if attempt.submission_token != execution.submission_token:
        raise AttemptEngineError("execution token differs from persisted submission token")
    intent_revision = f"{attempt.generation}:{attempt.sequence}:{attempt.attempt_id}"
    intent_digest = submission_intent_digest(
        intent_revision,
        execution.resources,
        execution.command,
        execution.environment,
        execution.expectation,
    )
    journal_dir = execution.receipt_root / "submissions" / attempt.attempt_id
    return cancel_submitting_intent_from_journal(
        attempt,
        scheduler=scheduler,
        journal_dir=journal_dir,
        intent_digest=intent_digest,
        cancellation_reason=cancellation_reason,
        submission_clock=submission_clock,
        submission_recovery_policy=submission_recovery_policy,
    )


def cancel_submitting_intent_from_journal(
    attempt: AttemptRecord,
    *,
    scheduler: Scheduler,
    journal_dir: Path,
    intent_digest: str,
    cancellation_reason: str,
    submission_clock: SubmissionClock | None = None,
    submission_recovery_policy: SubmissionRecoveryPolicy | None = None,
    recover_unique_after_manual: bool = False,
) -> AttemptTickResult:
    """Reconcile one token-only intent against its exact append-only journal."""
    if attempt.status is not AttemptStatus.SUBMITTING or attempt.job is not None:
        raise AttemptEngineError("intent cancellation requires SUBMITTING without a job")
    if attempt.submission_token is None:
        raise AttemptEngineError("intent cancellation requires a durable submission token")
    if not cancellation_reason.strip():
        raise AttemptEngineError("cancellation reason must be non-empty")
    if len(intent_digest) != 64 or any(char not in "0123456789abcdef" for char in intent_digest):
        raise AttemptEngineError("intent cancellation requires a SHA-256 intent digest")
    token = attempt.submission_token
    intent_revision = f"{attempt.generation}:{attempt.sequence}:{attempt.attempt_id}"
    clock = submission_clock or (lambda: datetime.now(UTC))
    policy = submission_recovery_policy or SubmissionRecoveryPolicy()
    with submission_recovery_lock(journal_dir):
        recorded = load_submission_cancellation(
            journal_dir=journal_dir,
            submission_token=token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
        )
        if recorded is not None:
            if recorded.action is SubmissionCancellationAction.CANCEL_INTENT:
                cancelled = attempt.transition(
                    AttemptStatus.CANCELLED,
                    scheduler_state="SUBMISSION_INTENT_CANCELLED",
                    terminal_reason=cancellation_reason.strip(),
                )
                return AttemptTickResult(cancelled, AttemptAction.CANCELLED_INTENT)
            if not recover_unique_after_manual:
                return _manual_recovery_attempt(attempt, recorded.reason)
        try:
            probe = scheduler.probe_submission(token)
        except SchedulerError as error:
            reason = f"submission cancellation probe failed: {error}"
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=reason,
                clock=clock,
            )
            return _manual_recovery_attempt(attempt, reason)
        if len(probe.matches) > 1 or (probe.matches and probe.identity is None):
            reason = probe.reason or "submission cancellation evidence is ambiguous"
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=reason,
                clock=clock,
            )
            return _manual_recovery_attempt(attempt, reason)
        if probe.identity is not None:
            ownership = scheduler.verify_ownership(probe.identity, token)
            if not ownership.matched:
                reason = f"submission cancellation ownership is ambiguous: {ownership.reason}"
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=reason,
                    clock=clock,
                )
                return _manual_recovery_attempt(attempt, reason)
            adopted = attempt.transition(
                AttemptStatus.SUBMITTED,
                job=_state_job(probe.identity),
                scheduler_state=probe.status.value,
                terminal_reason=None,
            )
            return AttemptTickResult(adopted, AttemptAction.ADOPTED)
        if recorded is not None:
            return _manual_recovery_attempt(attempt, recorded.reason)
        try:
            decision = decide_submission_cancellation(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                probe=probe,
                clock=clock,
                policy=policy,
            )
        except SubmissionRecoveryError as error:
            return _manual_recovery_attempt(
                attempt,
                f"invalid cancellation recovery journal: {error}",
            )
        if decision.action is SubmissionCancellationAction.MANUAL_RECOVERY:
            return _manual_recovery_attempt(attempt, decision.reason)
        if decision.action is SubmissionCancellationAction.WAIT:
            waiting = replace(
                attempt,
                scheduler_state="SUBMISSION_CANCELLATION_WAIT",
                terminal_reason=decision.reason,
            )
            return AttemptTickResult(
                waiting,
                AttemptAction.WAITING_FOR_ACCOUNTING,
                unknown_observations=1,
            )
        cancelled = attempt.transition(
            AttemptStatus.CANCELLED,
            scheduler_state="SUBMISSION_INTENT_CANCELLED",
            terminal_reason=cancellation_reason.strip(),
        )
        return AttemptTickResult(cancelled, AttemptAction.CANCELLED_INTENT)


def plan_infrastructure_retry(
    item: WorkItemRecord,
    failed_attempt_id: str,
    *,
    new_attempt_id: str,
    generation: int,
    policy: AttemptPolicy | None = None,
) -> InfrastructureRetryPlan:
    """Plan exactly one PREEMPTED or NODE_FAIL retry without side effects.

    LOST, submission errors, out-of-memory, timeouts, and worker-produced
    retryable results are intentionally excluded.  Replaying the same plan
    against a state that already contains its exact lineage child returns that
    child with ``already_appended=True``.
    """
    effective_policy = policy or AttemptPolicy()
    failed = _find_attempt(item, failed_attempt_id)
    category = _retry_category(failed)
    if category is None:
        raise AttemptEngineError(_automatic_retry_rejection_reason(failed))
    if failed.resource_class is None:
        raise AttemptEngineError("automatic retry requires a persisted selected resource_class")

    existing_children = tuple(
        attempt for attempt in item.attempts if attempt.predecessor_attempt_id == failed.attempt_id
    )
    if existing_children:
        child = existing_children[0]
        if (
            child.attempt_id != new_attempt_id
            or child.generation != generation
            or child.resource_class != failed.resource_class
            or child.resource_escalation_request_id != failed.resource_escalation_request_id
        ):
            raise AttemptEngineError(
                "retry replay conflicts with the already appended lineage child"
            )
        return InfrastructureRetryPlan(
            category=category,
            predecessor_attempt_id=failed.attempt_id,
            retries_used=_lineage_failure_count(item, failed, category),
            retry_budget=effective_policy.retry_budget(category),
            attempt=child,
            already_appended=True,
        )

    retries_used = _lineage_failure_count(item, failed, category)
    retry_budget = effective_policy.retry_budget(category)
    if retries_used > retry_budget:
        raise RetryLimitExceeded(
            f"attempt {failed_attempt_id!r} exhausted {retry_budget} {category.value} retries"
        )
    retry = AttemptRecord(
        attempt_id=new_attempt_id,
        item_id=item.item_id,
        sequence=len(item.attempts) + 1,
        role=failed.role,
        kind=failed.kind,
        generation=generation,
        profile=failed.profile,
        resource_class=failed.resource_class,
        predecessor_attempt_id=failed.attempt_id,
        resource_escalation_request_id=failed.resource_escalation_request_id,
        candidate_digest=failed.candidate_digest,
        review_of_attempt_id=failed.review_of_attempt_id,
        reviewed_candidate_digest=failed.reviewed_candidate_digest,
    )
    return InfrastructureRetryPlan(
        category=category,
        predecessor_attempt_id=failed.attempt_id,
        retries_used=retries_used,
        retry_budget=retry_budget,
        attempt=retry,
    )


def cancel_owned_attempt(attempt: AttemptRecord, scheduler: Scheduler) -> JobIdentity:
    """Request cancellation of exactly one persisted scheduler identity.

    This helper does not mark the attempt cancelled; a later reconciliation
    must observe Slurm's terminal ``CANCELLED`` state. It accepts no job-name,
    token, range, wildcard, or arbitrary scheduler target.

    Args:
        attempt: Active attempt with an exact persisted job reference.
        scheduler: Typed scheduler boundary.

    Returns:
        The exact identity passed to the scheduler.
    """
    if attempt.terminal:
        raise AttemptEngineError("cannot cancel a terminal attempt")
    if attempt.job is None:
        raise AttemptEngineError("cannot cancel an attempt without a persisted job identity")
    if attempt.submission_token is None:
        raise AttemptEngineError("cannot cancel an attempt without a persisted submission token")
    identity = _scheduler_identity(attempt.job)
    ownership = scheduler.verify_ownership(identity, attempt.submission_token)
    if not ownership.matched:
        raise AttemptEngineError(
            f"refusing to cancel job without exact scheduler ownership: {ownership.reason}"
        )
    scheduler.cancel(identity)
    return identity


def logical_attempt_root(item: WorkItemRecord, attempt_id: str) -> AttemptRecord:
    """Return the immutable lineage root for an attempt retry/escalation chain."""
    by_id = {attempt.attempt_id: attempt for attempt in item.attempts}
    attempt = _find_attempt(item, attempt_id)
    seen: set[str] = set()
    while attempt.predecessor_attempt_id is not None:
        if attempt.attempt_id in seen:
            raise AttemptEngineError("attempt lineage contains a cycle")
        seen.add(attempt.attempt_id)
        predecessor = by_id.get(attempt.predecessor_attempt_id)
        if predecessor is None:
            raise AttemptEngineError("attempt lineage contains an unknown predecessor")
        attempt = predecessor
    return attempt


def logical_attempt_ordinal(item: WorkItemRecord, attempt_id: str) -> int:
    """Return a one-based logical ordinal with retries collapsed to their root."""
    root = logical_attempt_root(item, attempt_id)
    roots = tuple(
        attempt
        for attempt in item.attempts
        if attempt.predecessor_attempt_id is None
        and attempt.role is root.role
        and attempt.kind is root.kind
        and attempt.profile is root.profile
    )
    try:
        return tuple(attempt.attempt_id for attempt in roots).index(root.attempt_id) + 1
    except ValueError as error:
        raise AttemptEngineError("logical attempt root is absent from item history") from error


def cancel_owned_attempts(
    attempts: Sequence[AttemptRecord], scheduler: Scheduler
) -> tuple[JobIdentity, ...]:
    """Cancel active owned jobs one exact identity at a time.

    Args:
        attempts: Attempt records selected by authoritative controller state.
        scheduler: Typed scheduler boundary.

    Returns:
        Exact scheduler identities cancelled in input order.
    """
    return tuple(cancel_owned_attempt(attempt, scheduler) for attempt in attempts)


def _submit_or_adopt(
    attempt: AttemptRecord,
    scheduler: Scheduler,
    execution: AttemptExecution,
    *,
    clock: SubmissionClock,
    recovery_policy: SubmissionRecoveryPolicy,
) -> AttemptTickResult:
    if execution.resources.requeue:
        raise AttemptEngineError("automatic Slurm requeue is disabled for Staircase attempts")
    token = execution.submission_token
    intent_revision = f"{attempt.generation}:{attempt.sequence}:{attempt.attempt_id}"
    intent_digest = submission_intent_digest(
        intent_revision,
        execution.resources,
        execution.command,
        execution.environment,
        execution.expectation,
    )
    journal_dir = execution.receipt_root / "submissions" / attempt.attempt_id
    with submission_recovery_lock(journal_dir):
        legacy_permit = execution.attempt_dir / "submission.permit.json"
        if legacy_permit.exists():
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason="legacy attempt claim has no trusted timestamp",
                clock=clock,
            )
            return _manual_recovery_attempt(
                attempt,
                "legacy attempt submission claim requires operator reconciliation",
            )
        try:
            probe = scheduler.probe_submission(token)
        except SchedulerError as error:
            decision = record_probe_error(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=str(error),
                clock=clock,
                policy=recovery_policy,
            )
            if decision.action is SubmissionRecoveryAction.MANUAL_RECOVERY:
                return _manual_recovery_attempt(attempt, decision.reason)
            uncertain = replace(
                attempt,
                scheduler_state="SUBMISSION_PROBE_ERROR",
                terminal_reason=decision.reason,
            )
            return AttemptTickResult(
                uncertain,
                AttemptAction.WAITING_FOR_ACCOUNTING,
                unknown_observations=1,
            )
        if len(probe.matches) > 1:
            quarantine_path = _quarantine_duplicate_submission(
                execution,
                probe.matches,
                source=probe.source.value,
                reason=probe.reason or "ambiguous immutable submission token",
            )
            failed = attempt.transition(
                AttemptStatus.FAILED,
                scheduler_state="DUPLICATE_SUBMISSION",
                terminal_reason=(
                    f"immutable submission token matched {len(probe.matches)} exact jobs"
                ),
            )
            return AttemptTickResult(
                failed,
                AttemptAction.DUPLICATE_SUBMISSION,
                duplicate_jobs=probe.matches,
                quarantined_path=quarantine_path,
            )
        if probe.matches and probe.identity is None:
            reason = probe.reason or "attempt scheduler evidence is untrustworthy"
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=reason,
                clock=clock,
            )
            return _manual_recovery_attempt(attempt, reason)
        if probe.identity is not None:
            ownership = scheduler.verify_ownership(probe.identity, token)
            if not ownership.matched:
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=f"attempt ownership is ambiguous: {ownership.reason}",
                    clock=clock,
                )
                return _manual_recovery_attempt(attempt, ownership.reason)
            adopted = attempt.transition(
                AttemptStatus.SUBMITTED,
                job=_state_job(probe.identity),
                scheduler_state=probe.status.value,
                terminal_reason=None,
            )
            return AttemptTickResult(adopted, AttemptAction.ADOPTED)
        try:
            decision = decide_submission_recovery(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                probe=probe,
                clock=clock,
                policy=recovery_policy,
            )
        except SubmissionRecoveryError as error:
            return _manual_recovery_attempt(attempt, f"invalid recovery journal: {error}")
        if decision.action is SubmissionRecoveryAction.MANUAL_RECOVERY:
            return _manual_recovery_attempt(attempt, decision.reason)
        if decision.action is SubmissionRecoveryAction.WAIT:
            waiting = replace(
                attempt,
                scheduler_state="SUBMISSION_RECOVERY_WAIT",
                terminal_reason=decision.reason,
            )
            return AttemptTickResult(
                waiting,
                AttemptAction.WAITING_FOR_ACCOUNTING,
                unknown_observations=1,
            )
        claim_sequence = decision.claim_sequence
        if claim_sequence is None:
            raise AttemptEngineError("submission recovery omitted its scheduler-call claim")
        expiry = execution.credential_expires_at_epoch_seconds
        if expiry is not None and expiry <= int(clock().timestamp()):
            reason = "attempt credential expired before the scheduler launch boundary"
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=reason,
                clock=clock,
            )
            return _manual_recovery_attempt(attempt, reason)
        try:
            identity = scheduler.submit(
                execution.resources,
                execution.command,
                token,
                execution.environment_mapping,
            )
        except SchedulerError as error:
            record_submission_outcome(
                journal_dir=journal_dir,
                claim_sequence=claim_sequence,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                outcome="uncertain",
                clock=clock,
            )
            uncertain = replace(
                attempt,
                scheduler_state="SUBMISSION_UNCERTAIN",
                terminal_reason=(
                    "scheduler submission returned without a durable identity; "
                    f"bounded token reconciliation is required: {error}"
                ),
            )
            return AttemptTickResult(
                uncertain,
                AttemptAction.WAITING_FOR_ACCOUNTING,
                unknown_observations=1,
            )
        record_submission_outcome(
            journal_dir=journal_dir,
            claim_sequence=claim_sequence,
            submission_token=token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
            outcome="accepted",
            clock=clock,
            scheduler_id=identity.scheduler_id,
        )
        ownership = scheduler.verify_ownership(identity, token)
        if not ownership.matched:
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=f"submitted attempt ownership is ambiguous: {ownership.reason}",
                clock=clock,
            )
            return _manual_recovery_attempt(attempt, ownership.reason)
        submitted = attempt.transition(
            AttemptStatus.SUBMITTED,
            job=_state_job(identity),
            scheduler_state=JobStatus.PENDING.value,
            terminal_reason=None,
        )
        return AttemptTickResult(submitted, AttemptAction.SUBMITTED)


def _observe_known_job(
    attempt: AttemptRecord,
    scheduler: Scheduler,
    *,
    policy: AttemptPolicy,
    unknown_observations: int,
) -> AttemptTickResult:
    if attempt.job is None:
        raise AttemptEngineError("submitted attempt has no persisted scheduler identity")
    if attempt.submission_token is None:
        raise AttemptEngineError("submitted attempt has no persisted submission token")
    observation = scheduler.observe_owned(
        _scheduler_identity(attempt.job),
        attempt.submission_token,
    )
    if observation.status is JobStatus.UNKNOWN:
        unknown_count = unknown_observations + 1
        if unknown_count <= policy.accounting_unknown_grace_ticks:
            return AttemptTickResult(
                attempt,
                AttemptAction.WAITING_FOR_ACCOUNTING,
                unknown_observations=unknown_count,
            )
        lost = attempt.transition(
            AttemptStatus.LOST,
            scheduler_state=JobStatus.UNKNOWN.value,
            terminal_reason=_observation_reason(
                observation,
                "job absent from both queue and accounting after bounded grace",
            ),
        )
        return AttemptTickResult(lost, AttemptAction.TERMINAL_FAILURE)
    if observation.status is JobStatus.PENDING:
        pending = _advance_visible(attempt, AttemptStatus.PENDING, observation)
        return AttemptTickResult(pending, AttemptAction.OBSERVED)
    if observation.status is JobStatus.RUNNING:
        running = _advance_visible(attempt, AttemptStatus.RUNNING, observation)
        return AttemptTickResult(running, AttemptAction.OBSERVED)
    if not observation.status.terminal:
        raise AttemptEngineError(f"scheduler returned unsupported state {observation.status.value}")
    observed = attempt.transition(
        AttemptStatus.TERMINAL_OBSERVED,
        scheduler_state=observation.status.value,
        terminal_reason=(
            None
            if observation.status is JobStatus.COMPLETED
            else _observation_reason(observation, "scheduler terminal state")
        ),
    )
    return AttemptTickResult(observed, AttemptAction.TERMINAL_OBSERVED)


def _manual_recovery_attempt(
    attempt: AttemptRecord,
    reason: str,
) -> AttemptTickResult:
    failed = attempt.transition(
        AttemptStatus.FAILED,
        scheduler_state="SUBMISSION_MANUAL_RECOVERY",
        terminal_reason=f"manual scheduler reconciliation required: {reason}",
    )
    return AttemptTickResult(failed, AttemptAction.TERMINAL_FAILURE)


def _advance_visible(
    attempt: AttemptRecord,
    status: AttemptStatus,
    observation: JobObservation,
) -> AttemptRecord:
    if attempt.status is status:
        return replace(attempt, scheduler_state=observation.status.value, terminal_reason=None)
    return attempt.transition(
        status,
        scheduler_state=observation.status.value,
        terminal_reason=None,
    )


def _map_terminal_observation(attempt: AttemptRecord) -> AttemptTickResult:
    try:
        scheduler_status = JobStatus(attempt.scheduler_state)
    except (TypeError, ValueError) as error:
        raise AttemptEngineError(
            f"terminal observation has invalid scheduler state {attempt.scheduler_state!r}"
        ) from error
    if scheduler_status is JobStatus.COMPLETED:
        collecting = attempt.transition(AttemptStatus.COLLECTING)
        return AttemptTickResult(collecting, AttemptAction.COLLECTING)
    if scheduler_status is JobStatus.CANCELLED:
        cancelled = attempt.transition(
            AttemptStatus.CANCELLED,
            terminal_reason=attempt.terminal_reason or "scheduler reported CANCELLED",
        )
        return AttemptTickResult(cancelled, AttemptAction.TERMINAL_FAILURE)
    if scheduler_status is JobStatus.PREEMPTED:
        preempted = attempt.transition(
            AttemptStatus.PREEMPTED,
            terminal_reason=attempt.terminal_reason or "scheduler reported PREEMPTED",
        )
        return AttemptTickResult(preempted, AttemptAction.TERMINAL_FAILURE)
    if scheduler_status in {
        JobStatus.NODE_FAIL,
        JobStatus.OUT_OF_MEMORY,
        JobStatus.TIMEOUT,
    }:
        retryable = attempt.transition(
            AttemptStatus.RETRYABLE_FAILED,
            terminal_reason=(
                attempt.terminal_reason
                or f"scheduler infrastructure failure: {scheduler_status.value}"
            ),
        )
        return AttemptTickResult(retryable, AttemptAction.TERMINAL_FAILURE)
    if scheduler_status is JobStatus.FAILED:
        failed = attempt.transition(
            AttemptStatus.FAILED,
            terminal_reason=attempt.terminal_reason or "scheduler application job failed",
        )
        return AttemptTickResult(failed, AttemptAction.TERMINAL_FAILURE)
    raise AttemptEngineError(
        f"non-terminal scheduler state persisted as terminal: {scheduler_status.value}"
    )


def _collect_completed_result(
    attempt: AttemptRecord,
    execution: AttemptExecution,
) -> AttemptTickResult:
    try:
        ingested = ingest_result(
            execution.attempt_dir,
            execution.expectation,
            receipt_root=execution.receipt_root,
            quarantine_root=execution.quarantine_root,
        )
        manifest = ingested.manifest
        result_digest = ingested.result_digest
    except DuplicateResultError:
        manifest, result_digest = load_result_manifest(execution.attempt_dir)
        _validate_result_identity(manifest, execution.expectation)
    except (IncompleteResultError, StaleResultError, ArtifactError) as error:
        failed = attempt.transition(
            AttemptStatus.FAILED,
            terminal_reason=(
                "application failure: scheduler completed without a valid matching "
                f"COMPLETE/result manifest: {error}"
            ),
        )
        return AttemptTickResult(failed, AttemptAction.TERMINAL_FAILURE)
    return _map_worker_result(attempt, manifest, result_digest)


def _map_worker_result(
    attempt: AttemptRecord,
    manifest: WorkerResultManifest,
    result_digest: str,
) -> AttemptTickResult:
    if manifest.status in {
        WorkerResultStatus.SUCCEEDED,
        WorkerResultStatus.BLOCKED,
        WorkerResultStatus.REJECTED,
        WorkerResultStatus.RESOURCE_ESCALATION,
    }:
        validated = attempt.transition(
            AttemptStatus.VALIDATED,
            result_digest=result_digest,
            candidate_digest=(None if attempt.role is Role.CODER else manifest.candidate_digest),
        )
        return AttemptTickResult(
            validated,
            AttemptAction.VALIDATED,
            worker_result=manifest,
        )
    reason = f"worker reported {manifest.status.value}: {manifest.summary}"
    if manifest.status is WorkerResultStatus.CANCELLED:
        terminal = attempt.transition(
            AttemptStatus.CANCELLED,
            result_digest=result_digest,
            terminal_reason=reason,
        )
    elif manifest.status is WorkerResultStatus.RETRYABLE_FAILED:
        terminal = attempt.transition(
            AttemptStatus.RETRYABLE_FAILED,
            result_digest=result_digest,
            terminal_reason=reason,
        )
    else:
        terminal = attempt.transition(
            AttemptStatus.FAILED,
            result_digest=result_digest,
            terminal_reason=f"application failure: {reason}",
        )
    return AttemptTickResult(
        terminal,
        AttemptAction.TERMINAL_FAILURE,
        worker_result=manifest,
    )


def _handle_terminal_attempt(
    attempt: AttemptRecord,
    execution: AttemptExecution,
    unknown_observations: int,
) -> AttemptTickResult:
    if (
        attempt.result_digest is not None
        or not (execution.attempt_dir / COMPLETE_FILENAME).is_file()
    ):
        return AttemptTickResult(
            attempt,
            AttemptAction.NOOP,
            unknown_observations=unknown_observations,
        )
    reason = (
        f"late result for terminal attempt {attempt.attempt_id!r} in state {attempt.status.value!r}"
    )
    destination = quarantine_result(
        execution.attempt_dir,
        execution.quarantine_root,
        reason,
    )
    return AttemptTickResult(
        attempt,
        AttemptAction.QUARANTINED,
        unknown_observations=unknown_observations,
        quarantined_path=destination,
    )


def _retry_category(
    attempt: AttemptRecord,
) -> InfrastructureRetryCategory | None:
    if attempt.status is AttemptStatus.PREEMPTED:
        return InfrastructureRetryCategory.PREEMPTED
    if (
        attempt.status is AttemptStatus.RETRYABLE_FAILED
        and attempt.scheduler_state == JobStatus.NODE_FAIL.value
        and attempt.result_digest is None
    ):
        return InfrastructureRetryCategory.NODE_FAIL
    return None


def _automatic_retry_rejection_reason(attempt: AttemptRecord) -> str:
    if attempt.status is AttemptStatus.LOST:
        category = "LOST"
    elif attempt.scheduler_state == "SUBMIT_ERROR":
        category = "SUBMIT_ERROR"
    elif attempt.scheduler_state == JobStatus.OUT_OF_MEMORY.value:
        category = "OUT_OF_MEMORY"
    elif attempt.scheduler_state == JobStatus.TIMEOUT.value:
        category = "TIMEOUT"
    elif attempt.status is AttemptStatus.RETRYABLE_FAILED and attempt.result_digest is not None:
        category = "worker RETRYABLE_FAILED"
    else:
        category = attempt.status.value
    return (
        f"attempt {attempt.attempt_id!r} category {category} is not eligible "
        "for automatic retry; only PREEMPTED and scheduler NODE_FAIL are eligible"
    )


def _lineage_failure_count(
    item: WorkItemRecord,
    failed: AttemptRecord,
    category: InfrastructureRetryCategory,
) -> int:
    by_id = {attempt.attempt_id: attempt for attempt in item.attempts}
    count = 0
    cursor: AttemptRecord | None = failed
    seen: set[str] = set()
    while cursor is not None:
        if cursor.attempt_id in seen:
            raise AttemptEngineError("attempt lineage contains a cycle")
        seen.add(cursor.attempt_id)
        if _retry_category(cursor) is category:
            count += 1
        predecessor_id = cursor.predecessor_attempt_id
        if predecessor_id is None:
            break
        cursor = by_id.get(predecessor_id)
        if cursor is None:
            raise AttemptEngineError("attempt lineage contains an unknown predecessor")
    return count


def _validate_execution_identity(
    attempt: AttemptRecord,
    execution: AttemptExecution,
) -> None:
    expectation = execution.expectation
    if expectation.item_id != attempt.item_id or expectation.attempt_id != attempt.attempt_id:
        raise AttemptEngineError("result expectation does not match attempt identity")
    if expectation.generation != attempt.generation:
        raise AttemptEngineError("result expectation does not match attempt generation")


def _quarantine_duplicate_submission(
    execution: AttemptExecution,
    matches: tuple[JobIdentity, ...],
    *,
    source: str,
    reason: str,
) -> Path:
    """Persist exact duplicate identities without choosing or cancelling one."""
    destination = (
        execution.quarantine_root / "submissions" / f"{execution.expectation.attempt_id}.json"
    )
    payload = {
        "schema_version": 1,
        "attempt_id": execution.expectation.attempt_id,
        "generation": execution.expectation.generation,
        "submission_token": execution.submission_token,
        "source": source,
        "reason": reason,
        "jobs": [
            {
                "job_id": identity.job_id,
                "array_task_id": identity.array_task_id,
                "cluster": identity.cluster,
            }
            for identity in matches
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if destination.read_text(encoding="utf-8") != encoded:
            raise AttemptEngineError("duplicate-submission quarantine receipt conflicts")
        return destination
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(destination.parent)
    return destination


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_result_identity(
    result: WorkerResultManifest,
    expectation: ResultExpectation,
) -> None:
    actual = (
        result.run_id,
        result.item_id,
        result.attempt_id,
        result.task_digest,
        result.generation,
        result.input_digest,
    )
    expected = (
        expectation.run_id,
        expectation.item_id,
        expectation.attempt_id,
        expectation.task_digest,
        expectation.generation,
        expectation.input_digest,
    )
    if actual != expected:
        raise AttemptEngineError("previously ingested result no longer matches expectation")


def _find_attempt(item: WorkItemRecord, attempt_id: str) -> AttemptRecord:
    for attempt in item.attempts:
        if attempt.attempt_id == attempt_id:
            return attempt
    raise AttemptEngineError(f"work item {item.item_id!r} has no attempt {attempt_id!r}")


def _observation_reason(observation: JobObservation, fallback: str) -> str:
    detail = observation.reason.strip() if observation.reason else fallback
    return f"{observation.status.value}: {detail}"


def _state_job(identity: JobIdentity) -> JobReference:
    return JobReference(
        identity.job_id,
        array_task_id=identity.array_task_id,
        cluster=identity.cluster,
    )


def _scheduler_identity(reference: JobReference) -> JobIdentity:
    return JobIdentity(
        reference.job_id,
        array_task_id=reference.array_task_id,
        cluster=reference.cluster,
    )


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
