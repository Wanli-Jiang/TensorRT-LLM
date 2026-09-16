# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic Staircase bootstrap, observation, and controller ownership."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import signal
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Callable, Iterator, Literal, Mapping, Protocol, cast

from .common.slurm import (
    DependencyType,
    InternalCommand,
    InternalEntrypoint,
    JobDependency,
    JobIdentity,
    JobObservation,
    JobStatus,
    Mount,
    ObservationSource,
    ResourceRequest,
    Scheduler,
    SchedulerError,
    SlurmScheduler,
)
from .common.submission_recovery import Clock as SubmissionClock
from .common.submission_recovery import (
    SubmissionRecoveryAction,
    SubmissionRecoveryError,
    SubmissionRecoveryPolicy,
    decide_submission_recovery,
    record_manual_recovery,
    record_probe_error,
    record_submission_outcome,
    submission_intent_digest,
    submission_recovery_lock,
)
from .state import (
    ATTEMPT_TERMINAL_STATUSES,
    LEASE_FILENAME,
    PLANNING_ITEM_ID,
    STATE_FILENAME,
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    ControllerBootstrapStatus,
    ControllerCheckpointSignal,
    ControllerLease,
    ControllerLifecycle,
    JobReference,
    OrphanReceiptAction,
    Role,
    RunState,
    RunTerminalStatus,
    StateConflictError,
    WorkflowMode,
    initialize_state,
    load_lease_record,
    load_state,
    save_state,
)
from .task_schema import NormalizedTask, load_normalized_task, write_normalized_task

if TYPE_CHECKING:
    from .common.credentials import (
        CredentialBinding,
        CredentialBroker,
        CredentialDescriptor,
        CredentialProvision,
        CredentialRevocationCause,
        CredentialRevocationReceipt,
    )
    from .common.discovery import ObservedExecutionFacts
    from .common.reporting import PlanningEvidence
    from .controller.runtime import RuntimeCallbacks, RuntimeTickResult

NORMALIZED_TASK_FILENAME = "task.normalized.json"
LAUNCHER_DIRECTORY = "launcher"
INITIAL_LAUNCH_INTENT_FILENAME = "launcher/initial.intent.json"
INITIAL_LAUNCH_RECEIPT_FILENAME = "launcher/initial.receipt.json"
CANCEL_REQUEST_FILENAME = "requests/cancel.json"
CANCEL_RECEIPT_FILENAME = "requests/cancel.receipt.json"
HUMAN_REQUESTS_DIRECTORY = "requests/human"
PREFLIGHT_FACTS_FILENAME = "preflight/observed-facts.json"
STATUS_FILENAME = "status.md"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MIN_POLL_INTERVAL_SECONDS = 0.01
_MAX_POLL_INTERVAL_SECONDS = 30.0
_MAX_REQUEST_BYTES = 1_048_576
_MAX_STATUS_LAUNCHERS = 256
_MAX_STATUS_ERROR_CHARS = 512


class RuntimeDriver(Protocol):
    """Bounded deterministic runtime consumed by the outer process."""

    def tick(self) -> RuntimeTickResult:
        """Complete at most one non-sleeping runtime transition."""


class RuntimeFactory(Protocol):
    """Construction seam for the production runtime and CPU tests."""

    def __call__(
        self,
        *,
        workspace: Path,
        task: NormalizedTask,
        scheduler: Scheduler,
        generation: int,
        callbacks: RuntimeCallbacks,
    ) -> RuntimeDriver:
        """Build one runtime bound to the active controller generation."""


class PreflightFactsProvider(Protocol):
    """Provide launcher-observed facts without copying expected task values."""

    def __call__(
        self,
        task: NormalizedTask,
        phase: Literal["login", "controller"],
    ) -> ObservedExecutionFacts:
        """Return independently observed facts for the named boundary."""


class PreflightCheck(Protocol):
    """Explicit fake/developer preflight seam; production uses observed facts."""

    def __call__(
        self,
        task: NormalizedTask,
        phase: Literal["login", "controller"],
    ) -> None:
        """Validate the named boundary without scheduler mutation."""


@dataclass(slots=True)
class _ControllerSignalFlags:
    """Signal-safe flags sampled only at deterministic outer-loop boundaries."""

    advance_requested: bool = False
    termination_requested: bool = False

    @property
    def stop_dispatch(self) -> bool:
        """Whether no new worker dispatch may begin."""
        return self.advance_requested or self.termination_requested

    @property
    def checkpoint_signal(self) -> ControllerCheckpointSignal | None:
        """Return the highest-priority durable checkpoint reason."""
        if self.termination_requested:
            return ControllerCheckpointSignal.PREEMPTION
        if self.advance_requested:
            return ControllerCheckpointSignal.ADVANCE
        return None

    def handle_advance(self, _signum: int, _frame: FrameType | None) -> None:
        """Set the bounded advance flag without performing I/O."""
        self.advance_requested = True

    def handle_termination(self, _signum: int, _frame: FrameType | None) -> None:
        """Set the bounded termination flag without performing I/O."""
        self.termination_requested = True


class WorkflowError(RuntimeError):
    """Raised when bootstrap or controller ownership cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class _EmergencyIntentReconciliation:
    """One dead-controller resolution for a durable submission intent."""

    original: AttemptRecord
    resolved: AttemptRecord
    action: str


@dataclass(frozen=True, slots=True)
class _WorkspaceRecoveryCredentialBroker:
    """Recovery-only adapter for the existing workspace credential store."""

    def describe(self, binding: CredentialBinding) -> CredentialDescriptor:
        raise WorkflowError("the recovery credential adapter cannot mint a credential descriptor")

    def inject_after_admission(
        self,
        *,
        workspace: Path,
        descriptor: CredentialDescriptor,
    ) -> CredentialProvision:
        raise WorkflowError("the recovery credential adapter cannot inject credentials")

    def revoke(
        self,
        *,
        workspace: Path,
        expected_binding: CredentialBinding,
        descriptor: CredentialDescriptor,
        cause: CredentialRevocationCause,
    ) -> CredentialRevocationReceipt:
        from .common.credentials import (
            CredentialState,
            locate_credential_provision,
            revoke_terminal_attempt_credentials,
        )

        if descriptor.state is not CredentialState.NONE:
            raise WorkflowError(
                "credentialed controller recovery requires an explicit credential broker"
            )
        return revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=expected_binding,
            provision=locate_credential_provision(
                workspace=workspace,
                descriptor=descriptor,
            ),
            cause=cause,
        )


@dataclass(frozen=True)
class StartResult:
    """Public result of starting or safely adopting one run."""

    run_id: str
    workspace: Path
    task_digest: str
    execution_mode: str
    controller_job: JobIdentity | None
    adopted: bool


def _require_production_preflight(
    task: NormalizedTask,
    phase: Literal["login", "controller"],
    facts_provider: PreflightFactsProvider | None,
    preflight_check: PreflightCheck | None,
) -> ObservedExecutionFacts | None:
    """Collect and require the phase-appropriate production contract."""
    from .common.discovery import ObservedExecutionFacts, collect_preflight_snapshot
    from .common.preflight import PreflightPhase, validate_production_preflight

    if preflight_check is not None:
        preflight_check(task, phase)
        return None
    if facts_provider is None:
        raise WorkflowError(
            f"{phase} production preflight requires independently observed launcher/build facts"
        )
    facts = facts_provider(task, phase)
    if not isinstance(facts, ObservedExecutionFacts):
        raise WorkflowError("production preflight facts provider returned an invalid value")
    snapshot = collect_preflight_snapshot(task, PreflightPhase(phase), facts)
    validate_production_preflight(task, snapshot).require_ok()
    return facts


def load_observed_preflight_facts(path: Path) -> ObservedExecutionFacts:
    """Load one strict launcher-observed fact manifest for production preflight."""
    from .common.discovery import ObservedExecutionFacts
    from .common.preflight import ObservedMount

    value = _load_json_mapping(path.resolve(strict=True), "preflight facts")
    expected = {
        "schema_version",
        "repository_root",
        "workspace_root",
        "container_image",
        "build_identity",
        "mounts",
    }
    if set(value) != expected or value.get("schema_version") != 1:
        raise WorkflowError("preflight facts have unknown fields or unsupported schema_version")
    mounts_value = value["mounts"]
    if not isinstance(mounts_value, list):
        raise WorkflowError("preflight facts mounts must be a JSON array")
    mounts = []
    for index, raw_mount in enumerate(mounts_value):
        if not isinstance(raw_mount, dict) or set(raw_mount) != {
            "host_path",
            "container_path",
            "read_only",
        }:
            raise WorkflowError(f"preflight facts mount {index} has invalid fields")
        host_path = raw_mount["host_path"]
        container_path = raw_mount["container_path"]
        read_only = raw_mount["read_only"]
        if not isinstance(host_path, str) or not isinstance(container_path, str):
            raise WorkflowError(f"preflight facts mount {index} paths must be strings")
        if not isinstance(read_only, bool):
            raise WorkflowError(f"preflight facts mount {index} read_only must be a bool")
        mounts.append(ObservedMount(Path(host_path), Path(container_path), read_only))

    path_fields = {}
    for name in ("repository_root", "workspace_root", "container_image"):
        raw_value = value[name]
        if not isinstance(raw_value, str):
            raise WorkflowError(f"preflight facts {name} must be a string")
        path_fields[name] = Path(raw_value)
    build_identity = value["build_identity"]
    if not isinstance(build_identity, str):
        raise WorkflowError("preflight facts build_identity must be a string")
    return ObservedExecutionFacts(
        repository_root=path_fields["repository_root"],
        workspace_root=path_fields["workspace_root"],
        container_image=path_fields["container_image"],
        build_identity=build_identity,
        mounts=tuple(mounts),
    )


def _persist_preflight_facts(workspace: Path, facts: ObservedExecutionFacts) -> None:
    payload = {
        "schema_version": 1,
        "repository_root": str(facts.repository_root),
        "workspace_root": str(facts.workspace_root),
        "container_image": str(facts.container_image),
        "build_identity": facts.build_identity,
        "mounts": [
            {
                "host_path": str(mount.host_path),
                "container_path": str(mount.container_path),
                "read_only": mount.read_only,
            }
            for mount in facts.mounts
        ],
    }
    _write_once_json(workspace / PREFLIGHT_FACTS_FILENAME, payload)


def _persisted_preflight_provider(workspace: Path) -> PreflightFactsProvider | None:
    path = workspace / PREFLIGHT_FACTS_FILENAME
    if not path.is_file():
        return None
    facts = load_observed_preflight_facts(path)

    def provide(
        _task: NormalizedTask,
        _phase: Literal["login", "controller"],
    ) -> ObservedExecutionFacts:
        return facts

    return provide


def start_run(
    *,
    mode: str,
    task: NormalizedTask,
    workspace: Path,
    scheduler: Scheduler | None = None,
    preflight_facts_provider: PreflightFactsProvider | None = None,
    preflight_check: PreflightCheck | None = None,
    submission_clock: SubmissionClock | None = None,
    submission_recovery_policy: SubmissionRecoveryPolicy | None = None,
) -> StartResult:
    """Create or safely adopt a Staircase run and its controller submission."""
    if mode not in {"onboard", "tune"}:
        raise ValueError(f"unsupported Staircase mode {mode!r}")
    workspace = _canonical_workspace(task, workspace)
    state_path = workspace / STATE_FILENAME
    normalized_path = workspace / NORMALIZED_TASK_FILENAME
    if state_path.exists():
        effective_task = load_normalized_task(normalized_path)
        if effective_task.digest != task.digest:
            raise WorkflowError("workspace task digest differs from the supplied task")
        if effective_task.execution.mode == "slurm":
            provider = preflight_facts_provider or _persisted_preflight_provider(workspace)
            _require_production_preflight(
                effective_task,
                "login",
                provider,
                preflight_check,
            )
        state = load_state(state_path)
        return _adopt_existing_run(
            state,
            mode=mode,
            task=effective_task,
            workspace=workspace,
            scheduler=scheduler,
        )
    if normalized_path.exists():
        effective_task = load_normalized_task(normalized_path)
        if effective_task.digest != task.digest:
            raise WorkflowError("workspace task digest differs from the supplied task")
        if effective_task.execution.mode != "slurm":
            raise WorkflowError("local launcher workspace has no authoritative state")
        provider = preflight_facts_provider or _persisted_preflight_provider(workspace)
        _require_production_preflight(
            effective_task,
            "login",
            provider,
            preflight_check,
        )
        return _resume_initial_launcher(
            mode=mode,
            task=effective_task,
            workspace=workspace,
            scheduler=scheduler or SlurmScheduler(),
            submission_clock=submission_clock,
            submission_recovery_policy=submission_recovery_policy,
        )

    observed_facts = None
    if task.execution.mode == "slurm":
        observed_facts = _require_production_preflight(
            task,
            "login",
            preflight_facts_provider,
            preflight_check,
        )
    workspace.mkdir(parents=True, exist_ok=True)
    for relative in (
        "locks",
        "prompts",
        "quarantine",
        "receipts",
        "reports",
        "slurm/controller",
        "items",
        LAUNCHER_DIRECTORY,
        "worktrees",
        "requests/human",
    ):
        (workspace / relative).mkdir(parents=True, exist_ok=True)
    if observed_facts is not None:
        _persist_preflight_facts(workspace, observed_facts)
    write_normalized_task(normalized_path, task)
    run_id = f"staircase-{secrets.token_hex(12)}"
    if task.execution.mode == "local":
        state = RunState(
            run_id=run_id,
            task_digest=task.digest,
            base_commit=task.repository.base_commit,
            generation=1,
            workflow_mode=WorkflowMode(mode),
        )
        initialize_state(state_path, state)
        _write_status(workspace, state, observation=None)
        return StartResult(run_id, workspace, task.digest, "local", None, False)

    submission_token = f"controller-{secrets.token_hex(12)}"
    owner_nonce = f"owner-{secrets.token_hex(12)}"
    intent = {
        "schema_version": 1,
        "kind": "initial",
        "run_id": run_id,
        "task_digest": task.digest,
        "workflow_mode": mode,
        "generation": 1,
        "submission_token": submission_token,
        "owner_nonce": owner_nonce,
    }
    intent_path = workspace / INITIAL_LAUNCH_INTENT_FILENAME
    _write_once_json(intent_path, intent)
    active_scheduler = scheduler or SlurmScheduler()
    identity = _submit_launcher_intent(
        intent_path=intent_path,
        receipt_path=workspace / INITIAL_LAUNCH_RECEIPT_FILENAME,
        scheduler=active_scheduler,
        resources=_controller_resources(task, workspace),
        command=InternalCommand(
            InternalEntrypoint.CONTROLLER,
            workspace=workspace,
            generation=1,
            owner_nonce=owner_nonce,
        ),
        submission_token=submission_token,
        environment=dict(task.execution.slurm.controller.environment),
        clock=submission_clock,
        recovery_policy=submission_recovery_policy,
    )
    return StartResult(run_id, workspace, task.digest, "slurm", identity, False)


def _resume_initial_launcher(
    *,
    mode: str,
    task: NormalizedTask,
    workspace: Path,
    scheduler: Scheduler,
    submission_clock: SubmissionClock | None,
    submission_recovery_policy: SubmissionRecoveryPolicy | None,
) -> StartResult:
    intent_path = workspace / INITIAL_LAUNCH_INTENT_FILENAME
    if not intent_path.is_file():
        # The normalized task is written before any scheduler side effect. If
        # the launcher died at that boundary, creating the still-missing
        # envelope is safe because no submission token existed yet.
        _write_once_json(
            intent_path,
            {
                "schema_version": 1,
                "kind": "initial",
                "run_id": f"staircase-{secrets.token_hex(12)}",
                "task_digest": task.digest,
                "workflow_mode": mode,
                "generation": 1,
                "submission_token": f"controller-{secrets.token_hex(12)}",
                "owner_nonce": f"owner-{secrets.token_hex(12)}",
            },
        )
    intent = _load_json_mapping(intent_path, "initial launcher intent")
    _require_launcher_intent(
        intent,
        kind="initial",
        run_id=cast(str, intent.get("run_id")),
        task_digest=task.digest,
        generation=1,
    )
    if intent.get("workflow_mode") != mode:
        raise WorkflowError(f"workspace launcher is a {intent.get('workflow_mode')!r} run")
    run_id = cast(str, intent["run_id"])
    submission_token = cast(str, intent["submission_token"])
    owner_nonce = cast(str, intent["owner_nonce"])
    identity = _submit_launcher_intent(
        intent_path=intent_path,
        receipt_path=workspace / INITIAL_LAUNCH_RECEIPT_FILENAME,
        scheduler=scheduler,
        resources=_controller_resources(task, workspace),
        command=InternalCommand(
            InternalEntrypoint.CONTROLLER,
            workspace=workspace,
            generation=1,
            owner_nonce=owner_nonce,
        ),
        submission_token=submission_token,
        environment=dict(task.execution.slurm.controller.environment),
        clock=submission_clock,
        recovery_policy=submission_recovery_policy,
    )
    return StartResult(run_id, workspace, task.digest, "slurm", identity, True)


def _submit_launcher_intent(
    *,
    intent_path: Path,
    receipt_path: Path,
    scheduler: Scheduler,
    resources: ResourceRequest,
    command: InternalCommand,
    submission_token: str,
    environment: Mapping[str, str],
    dependency: JobDependency | None = None,
    clock: SubmissionClock | None = None,
    recovery_policy: SubmissionRecoveryPolicy | None = None,
) -> JobIdentity:
    """Submit or recover one immutable launcher envelope without duplication."""
    if receipt_path.is_file():
        receipt = _load_json_mapping(receipt_path, "launcher receipt")
        if receipt.get("submission_token") != submission_token:
            raise WorkflowError("launcher receipt token differs from its immutable intent")
        return _job_from_payload(receipt.get("job"), "launcher receipt job")

    effective_clock = clock or (lambda: datetime.now(UTC))
    policy = recovery_policy or SubmissionRecoveryPolicy()
    intent_revision = hashlib.sha256(intent_path.read_bytes()).hexdigest()
    intent_digest = submission_intent_digest(
        intent_revision,
        resources,
        command,
        tuple(sorted(environment.items())),
        dependency,
    )
    journal_dir = intent_path.parent / "submission-journals" / submission_token
    legacy_claim = intent_path.with_name(f"{intent_path.name}.submit-claimed.json")
    with submission_recovery_lock(journal_dir):
        if legacy_claim.exists():
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason="legacy launcher claim has no trusted timestamp",
                clock=effective_clock,
            )
            raise WorkflowError(
                "launcher submission requires manual recovery because a legacy claim exists"
            )
        try:
            probe = scheduler.probe_submission(submission_token)
        except SchedulerError as error:
            decision = record_probe_error(
                journal_dir=journal_dir,
                submission_token=submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=str(error),
                clock=effective_clock,
                policy=policy,
            )
            raise WorkflowError(f"launcher recovery probe is blocked: {decision.reason}") from error
        if len(probe.matches) > 1:
            quarantine = _write_duplicate_submission_quarantine(
                cast(Path, command.workspace),
                submission_token,
                probe.matches,
                kind="launcher",
                reason=probe.reason or "ambiguous launcher submission token",
            )
            raise WorkflowError(f"launcher duplicates are quarantined in {quarantine}")
        if probe.matches and probe.identity is None:
            reason = probe.reason or "launcher scheduler evidence is untrustworthy"
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=reason,
                clock=effective_clock,
            )
            raise WorkflowError(f"launcher scheduler evidence requires manual recovery: {reason}")
        if probe.identity is not None:
            identity = probe.identity
            if dependency is not None and identity.cluster != dependency.job.cluster:
                raise WorkflowError("adopted launcher job is on a different cluster")
            ownership = scheduler.verify_ownership(identity, submission_token)
            if not ownership.matched:
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=submission_token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=f"launcher ownership is ambiguous: {ownership.reason}",
                    clock=effective_clock,
                )
                raise WorkflowError(
                    f"launcher scheduler ownership could not be proven: {ownership.reason}"
                )
            _write_launcher_receipt(receipt_path, intent_path, submission_token, identity)
            return identity
        try:
            decision = decide_submission_recovery(
                journal_dir=journal_dir,
                submission_token=submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                probe=probe,
                clock=effective_clock,
                policy=policy,
            )
        except SubmissionRecoveryError as error:
            raise WorkflowError(f"launcher recovery journal is invalid: {error}") from error
        if decision.action is not SubmissionRecoveryAction.SUBMIT:
            raise WorkflowError(f"launcher submission recovery: {decision.reason}")
        claim_sequence = cast(int, decision.claim_sequence)
        try:
            identity = scheduler.submit(
                resources,
                command,
                submission_token,
                environment,
                dependency,
            )
        except SchedulerError as error:
            record_submission_outcome(
                journal_dir=journal_dir,
                claim_sequence=claim_sequence,
                submission_token=submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                outcome="uncertain",
                clock=effective_clock,
            )
            raise WorkflowError(
                "launcher submit returned without an adoptable exact identity; "
                "bounded token reconciliation is required"
            ) from error
        record_submission_outcome(
            journal_dir=journal_dir,
            claim_sequence=claim_sequence,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
            outcome="accepted",
            clock=effective_clock,
            scheduler_id=identity.scheduler_id,
        )
        if dependency is not None and identity.cluster != dependency.job.cluster:
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason="scheduler returned a launcher job on the wrong cluster",
                clock=effective_clock,
            )
            raise WorkflowError("scheduler returned a launcher job on the wrong cluster")
        ownership = scheduler.verify_ownership(identity, submission_token)
        if not ownership.matched:
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=f"submitted launcher ownership is ambiguous: {ownership.reason}",
                clock=effective_clock,
            )
            raise WorkflowError(
                f"submitted launcher ownership could not be proven: {ownership.reason}"
            )
        _write_launcher_receipt(receipt_path, intent_path, submission_token, identity)
        return identity


def _write_launcher_receipt(
    receipt_path: Path,
    intent_path: Path,
    submission_token: str,
    identity: JobIdentity,
) -> None:
    _write_once_json(
        receipt_path,
        {
            "schema_version": 1,
            "intent_digest": hashlib.sha256(intent_path.read_bytes()).hexdigest(),
            "submission_token": submission_token,
            "job": _job_payload(identity),
        },
    )


def _adopt_existing_run(
    state: RunState,
    *,
    mode: str,
    task: NormalizedTask,
    workspace: Path,
    scheduler: Scheduler | None,
) -> StartResult:
    if state.task_digest != task.digest:
        raise WorkflowError("authoritative state task digest differs from normalized task")
    if state.workflow_mode is not WorkflowMode(mode):
        raise WorkflowError(f"workspace is a {state.workflow_mode.value!r} run, not a {mode!r} run")
    if task.execution.mode == "local":
        return StartResult(state.run_id, workspace, task.digest, "local", None, True)
    active_scheduler = scheduler or SlurmScheduler(
        cluster=state.controller_job.cluster if state.controller_job is not None else None
    )
    if state.controller_bootstrap_status is ControllerBootstrapStatus.SUBMITTING:
        token = cast(str, state.controller_submission_token)
        lookup = active_scheduler.lookup_submission(token)
        if lookup.identity is None:
            raise WorkflowError(
                "controller submission is unresolved; accounting may be delayed or multiple jobs match; "
                "refusing to submit a duplicate"
            )
        return StartResult(
            state.run_id,
            workspace,
            task.digest,
            "slurm",
            lookup.identity,
            True,
        )
    if state.controller_job is None:
        raise WorkflowError("Slurm run has no exact controller job identity")
    if state.controller_submission_token is None:
        raise WorkflowError("Slurm run has no immutable controller submission token")
    identity = _scheduler_job(state.controller_job)
    observation = active_scheduler.observe_owned(
        identity,
        state.controller_submission_token,
    )
    if (
        state.terminal_status is RunTerminalStatus.ACTIVE
        and state.controller_lifecycle
        in {ControllerLifecycle.RUNNING, ControllerLifecycle.CHECKPOINTED}
        and observation.status.terminal
        and observation.source is ObservationSource.ACCOUNTING
    ):
        ownership = active_scheduler.verify_ownership(
            identity,
            cast(str, state.controller_submission_token),
        )
        if not ownership.matched:
            raise WorkflowError(
                f"dead-controller ownership could not be proven: {ownership.reason}"
            )
        if _controller_lease_is_live(workspace / LEASE_FILENAME):
            raise WorkflowError("dead-controller recovery is fenced by a live controller lease")
        recovery_identity = _launch_recovery_successor(
            state=state,
            task=task,
            workspace=workspace,
            scheduler=active_scheduler,
            predecessor=identity,
        )
        return StartResult(
            state.run_id,
            workspace,
            task.digest,
            "slurm",
            recovery_identity,
            True,
        )
    return StartResult(state.run_id, workspace, task.digest, "slurm", identity, True)


def _launch_recovery_successor(
    *,
    state: RunState,
    task: NormalizedTask,
    workspace: Path,
    scheduler: Scheduler,
    predecessor: JobIdentity,
) -> JobIdentity:
    material = f"{state.run_id}\0{state.generation}\0{predecessor.scheduler_id}\0recovery"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    submission_token = f"controller-recovery-{digest[:30]}"
    owner_nonce = f"owner-{digest[30:54]}"
    generation = state.generation + 1
    grace_started = datetime.now(UTC)
    grace_deadline = grace_started + timedelta(
        seconds=task.execution.slurm.controller.orphan_grace_seconds
    )
    intent_path = workspace / LAUNCHER_DIRECTORY / f"recovery-g{generation}.intent.json"
    receipt_path = workspace / LAUNCHER_DIRECTORY / f"recovery-g{generation}.receipt.json"
    _write_once_json(
        intent_path,
        {
            "schema_version": 1,
            "kind": "unexpected_recovery",
            "run_id": state.run_id,
            "task_digest": state.task_digest,
            "predecessor_generation": state.generation,
            "predecessor_job": _job_payload(predecessor),
            "generation": generation,
            "submission_token": submission_token,
            "owner_nonce": owner_nonce,
            "reason": f"unexpected controller death in generation {state.generation}",
            "orphan_grace_started_at": grace_started.isoformat(),
            "orphan_grace_deadline_at": grace_deadline.isoformat(),
        },
    )
    return _submit_launcher_intent(
        intent_path=intent_path,
        receipt_path=receipt_path,
        scheduler=scheduler,
        resources=replace(_controller_resources(task, workspace), requeue=False),
        command=InternalCommand(
            InternalEntrypoint.CONTROLLER,
            workspace=workspace,
            generation=generation,
            owner_nonce=owner_nonce,
        ),
        submission_token=submission_token,
        environment=dict(task.execution.slurm.controller.environment),
        dependency=JobDependency(DependencyType.AFTERANY, predecessor),
    )


def _require_launcher_intent(
    intent: Mapping[str, object],
    *,
    kind: str,
    run_id: str,
    task_digest: str,
    generation: int,
) -> None:
    required = {
        "schema_version",
        "kind",
        "run_id",
        "task_digest",
        "generation",
        "submission_token",
        "owner_nonce",
    }
    if not required.issubset(intent):
        raise WorkflowError("launcher intent is missing required fields")
    if (
        intent.get("schema_version") != 1
        or intent.get("kind") != kind
        or intent.get("run_id") != run_id
        or intent.get("task_digest") != task_digest
        or intent.get("generation") != generation
    ):
        raise WorkflowError("launcher intent identity does not match this run boundary")
    for field in ("run_id", "submission_token", "owner_nonce"):
        value = intent.get(field)
        if not isinstance(value, str) or not value.strip():
            raise WorkflowError(f"launcher intent {field} must be a non-empty string")


def _job_from_payload(value: object, label: str) -> JobIdentity:
    if not isinstance(value, dict) or set(value) != {"job_id", "array_task_id", "cluster"}:
        raise WorkflowError(f"{label} has invalid fields")
    job_id = value.get("job_id")
    array_task_id = value.get("array_task_id")
    cluster = value.get("cluster")
    if not isinstance(job_id, str):
        raise WorkflowError(f"{label} job_id must be a string")
    if array_task_id is not None and not isinstance(array_task_id, str):
        raise WorkflowError(f"{label} array_task_id must be a string or null")
    if cluster is not None and not isinstance(cluster, str):
        raise WorkflowError(f"{label} cluster must be a string or null")
    try:
        return JobIdentity(job_id, array_task_id=array_task_id, cluster=cluster)
    except ValueError as error:
        raise WorkflowError(f"{label} is invalid: {error}") from error


def _write_duplicate_submission_quarantine(
    workspace: Path,
    submission_token: str,
    matches: tuple[JobIdentity, ...],
    *,
    kind: str,
    reason: str,
) -> Path:
    destination = workspace / "quarantine" / "submissions" / f"{kind}-{submission_token}.json"
    _write_once_json(
        destination,
        {
            "schema_version": 1,
            "kind": kind,
            "submission_token": submission_token,
            "reason": reason,
            "jobs": [_job_payload(identity) for identity in matches],
        },
    )
    return destination


def render_status(
    workspace: Path,
    *,
    as_json: bool = False,
    scheduler: Scheduler | None = None,
) -> str:
    """Render state plus a read-only scheduler reconciliation."""
    workspace = workspace.resolve(strict=True)
    state_path = workspace / STATE_FILENAME
    if not state_path.is_file():
        intent = _load_json_mapping(
            workspace / INITIAL_LAUNCH_INTENT_FILENAME,
            "initial launcher intent",
        )
        receipt_path = workspace / INITIAL_LAUNCH_RECEIPT_FILENAME
        observation = None
        controller_job = None
        read_errors: list[dict[str, object]] = []
        if receipt_path.is_file():
            try:
                receipt = _load_json_mapping(receipt_path, "initial launcher receipt")
                controller_job = _job_from_payload(receipt.get("job"), "initial launcher job")
            except (OSError, TypeError, ValueError, WorkflowError) as error:
                _append_status_error(read_errors, "launcher.initial_receipt", error)
            if controller_job is not None:
                active_scheduler = scheduler or SlurmScheduler(cluster=controller_job.cluster)
                try:
                    observation = active_scheduler.observe_owned(
                        controller_job,
                        cast(str, intent.get("submission_token")),
                    )
                except (OSError, SchedulerError, ValueError) as error:
                    _append_status_error(read_errors, "controller.scheduler", error)
        payload = {
            "run_id": intent.get("run_id"),
            "task_digest": intent.get("task_digest"),
            "workflow_mode": intent.get("workflow_mode"),
            "generation": intent.get("generation"),
            "authoritative_state": "not_bootstrapped",
            "controller_job": (
                _job_payload(controller_job) if controller_job is not None else None
            ),
            "scheduler_observation": (
                _json_safe(asdict(observation)) if observation is not None else None
            ),
            "projection": {
                "launchers": _project_launcher_status(workspace, read_errors),
                "read_errors": read_errors,
            },
        }
        if as_json:
            return json.dumps(payload, indent=2, sort_keys=True) + "\n"
        scheduler_status = observation.status.value if observation is not None else "UNKNOWN"
        return "\n".join(
            (
                f"run: {intent.get('run_id')}",
                f"task_digest: {intent.get('task_digest')}",
                f"workflow_mode: {intent.get('workflow_mode')}",
                "authoritative_state: not_bootstrapped",
                f"scheduler_status: {scheduler_status}",
                f"projection_read_errors: {len(read_errors)}",
                "",
            )
        )
    state = load_state(state_path)
    active_scheduler = scheduler or SlurmScheduler(
        cluster=state.controller_job.cluster if state.controller_job is not None else None
    )
    payload = _status_payload(workspace, state, active_scheduler)
    if as_json:
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    projection = cast(dict[str, object], payload["projection"])
    controller = cast(dict[str, object], projection["controller"])
    lease = cast(dict[str, object], projection["lease"])
    attempts = cast(list[dict[str, object]], projection["attempts"])
    drift = cast(list[dict[str, object]], projection["drift"])
    read_errors = cast(list[dict[str, object]], projection["read_errors"])
    lines = [
        f"run: {state.run_id}",
        f"task_digest: {state.task_digest}",
        f"workflow_mode: {state.workflow_mode.value}",
        f"generation: {state.generation}",
        f"revision: {state.revision}",
        f"outcome: {state.terminal_status.value}",
        f"controller_bootstrap: {state.controller_bootstrap_status.value}",
        f"controller_scheduler_status: {controller.get('scheduler_status', 'UNKNOWN')}",
        f"controller_scheduler_source: {controller.get('scheduler_source', 'UNKNOWN')}",
        f"lease_lock: {lease.get('lock_status', 'unknown')}",
        f"lease_consistent: {lease.get('consistent', False)}",
        f"projected_attempts: {len(attempts)}",
        f"projected_drift: {len(drift)}",
        f"projection_read_errors: {len(read_errors)}",
    ]
    lines.extend(
        (
            f"stages: {len(state.stages)}",
            f"goals: {len(state.goals)}",
            f"items: {len(state.items)}",
        )
    )
    return "\n".join(lines) + "\n"


def request_cancellation(
    workspace: Path,
    *,
    reason: str,
    scheduler: Scheduler | None = None,
) -> None:
    """Durably request cancellation and clean up a dead controller's exact jobs."""
    workspace = workspace.resolve(strict=True)
    if not reason.strip():
        raise ValueError("cancellation reason must be non-empty")
    state_path = workspace / STATE_FILENAME
    if not state_path.is_file():
        intent = _load_json_mapping(
            workspace / INITIAL_LAUNCH_INTENT_FILENAME,
            "initial launcher intent",
        )
        run_id = intent.get("run_id")
        task_digest = intent.get("task_digest")
        token = intent.get("submission_token")
        if not all(isinstance(value, str) for value in (run_id, task_digest, token)):
            raise WorkflowError("initial launcher intent is incomplete")
        _write_once_json(
            workspace / CANCEL_REQUEST_FILENAME,
            {
                "run_id": run_id,
                "task_digest": task_digest,
                "reason": reason.strip(),
            },
        )
        if state_path.is_file():
            return request_cancellation(workspace, reason=reason, scheduler=scheduler)
        receipt_path = workspace / INITIAL_LAUNCH_RECEIPT_FILENAME
        if not receipt_path.is_file():
            return
        launch_receipt = _load_json_mapping(receipt_path, "initial launcher receipt")
        controller_job = _job_from_payload(
            launch_receipt.get("job"),
            "initial launcher job",
        )
        active_scheduler = scheduler or SlurmScheduler(cluster=controller_job.cluster)
        ownership = active_scheduler.verify_ownership(controller_job, cast(str, token))
        if not ownership.matched:
            raise WorkflowError(
                f"controller cancellation ownership could not be proven: {ownership.reason}"
            )
        observation = active_scheduler.observe_owned(controller_job, cast(str, token))
        if observation.status.terminal and observation.source is ObservationSource.ACCOUNTING:
            _write_once_json(
                workspace / CANCEL_RECEIPT_FILENAME,
                {
                    "run_id": run_id,
                    "task_digest": task_digest,
                    "reason": reason.strip(),
                    "controller_job": _job_payload(controller_job),
                    "controller_status": observation.status.value,
                    "controller_observation_source": observation.source.value,
                    "controller_submission_token": token,
                    "controller_ownership_user": ownership.username,
                    "child_actions": [],
                },
            )
        return

    state = load_state(state_path)
    if state.terminal_status is not RunTerminalStatus.ACTIVE:
        raise WorkflowError(f"run is already terminal: {state.terminal_status.value}")
    path = workspace / CANCEL_REQUEST_FILENAME
    payload = {"run_id": state.run_id, "task_digest": state.task_digest, "reason": reason.strip()}
    _write_once_json(path, payload)
    if state.controller_job is None:
        return

    lease_path = workspace / LEASE_FILENAME
    with _held_controller_lease_lock(lease_path) as lock_acquired:
        if not lock_acquired:
            return
        locked_state = load_state(state_path)
        if (
            locked_state.run_id != state.run_id
            or locked_state.task_digest != state.task_digest
            or locked_state.generation != state.generation
            or locked_state.controller_job != state.controller_job
            or locked_state.controller_submission_token != state.controller_submission_token
        ):
            raise WorkflowError("authoritative controller identity changed before cancellation")
        if lease_path.is_file():
            lease_record = load_lease_record(lease_path)
            if (
                lease_record.run_id != locked_state.run_id
                or lease_record.generation != locked_state.generation
                or lease_record.controller_job != locked_state.controller_job
            ):
                raise WorkflowError(
                    "durable controller lease does not match authoritative ownership"
                )
        _cancel_dead_controller_jobs(
            workspace=workspace,
            state=locked_state,
            reason=reason.strip(),
            scheduler=scheduler,
        )


def _cancel_dead_controller_jobs(
    *,
    workspace: Path,
    state: RunState,
    reason: str,
    scheduler: Scheduler | None,
) -> None:
    from .common.credentials import CredentialError, CredentialRevocationCause, CredentialState

    if state.controller_job is None or state.controller_submission_token is None:
        raise WorkflowError("controller cancellation identity is incomplete")
    receipt_path = workspace / CANCEL_RECEIPT_FILENAME
    if receipt_path.exists():
        existing = _load_json_mapping(receipt_path, "emergency cancellation receipt")
        if (
            existing.get("run_id") != state.run_id
            or existing.get("task_digest") != state.task_digest
            or existing.get("reason") != reason
            or existing.get("controller_job") != _job_payload(_scheduler_job(state.controller_job))
            or existing.get("controller_submission_token") != state.controller_submission_token
            or not isinstance(existing.get("credential_actions"), list)
            or not isinstance(existing.get("child_actions"), list)
        ):
            raise WorkflowError("emergency cancellation receipt differs from authoritative state")
        return
    active_scheduler = scheduler or SlurmScheduler(cluster=state.controller_job.cluster)
    controller_job = _scheduler_job(state.controller_job)
    controller_token = state.controller_submission_token
    controller_ownership = active_scheduler.verify_ownership(controller_job, controller_token)
    if not controller_ownership.matched:
        raise WorkflowError(
            f"controller cancellation ownership could not be proven: {controller_ownership.reason}"
        )
    observation = active_scheduler.observe_owned(controller_job, controller_token)
    if not observation.status.terminal or observation.source is not ObservationSource.ACCOUNTING:
        return

    task = load_normalized_task(workspace / NORMALIZED_TASK_FILENAME)
    if task.digest != state.task_digest:
        raise WorkflowError("emergency cancellation task differs from authoritative state")
    task_uses_credentials = any(
        role_class.credential_broker.allowed_credential_names
        for role_class in task.execution.slurm.role_classes
    )
    descriptor_attempts = (
        _active_credential_descriptors(workspace, state) if task_uses_credentials else ()
    )
    credentialed = tuple(
        (attempt, descriptor)
        for attempt, descriptor in descriptor_attempts
        if descriptor.state is not CredentialState.NONE
    )

    # Resolve every identity and every controller-private credential descriptor
    # before the first external mutation.  A partial ownership view must not
    # revoke credentials or cancel a prefix of the active jobs.
    intent_reconciliations = _reconcile_emergency_submission_intents(
        workspace=workspace,
        state=state,
        scheduler=active_scheduler,
        cancellation_reason=reason,
    )
    known_intent_ids = {entry.original.attempt_id for entry in intent_reconciliations}
    historical_attempts = [*state.planning_attempts]
    historical_attempts.extend(attempt for item in state.items for attempt in item.attempts)
    intent_reconciliations = (
        *intent_reconciliations,
        *(
            _EmergencyIntentReconciliation(attempt, attempt, "cancelled_intent")
            for attempt in historical_attempts
            if attempt.attempt_id not in known_intent_ids
            and attempt.status is AttemptStatus.CANCELLED
            and attempt.scheduler_state == "SUBMISSION_INTENT_CANCELLED"
            and attempt.terminal_reason == reason
        ),
    )
    adopted = {
        entry.original.attempt_id: entry.resolved
        for entry in intent_reconciliations
        if entry.resolved.status is AttemptStatus.SUBMITTED
    }
    if adopted:
        desired = _replace_emergency_attempts(state, adopted)
        save_state(
            workspace / STATE_FILENAME,
            desired,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )
        state = desired
    cancelled_intent_ids = {
        entry.original.attempt_id
        for entry in intent_reconciliations
        if entry.resolved.status is AttemptStatus.CANCELLED
    }
    planned_actions: list[tuple[AttemptRecord, JobIdentity, JobObservation, str | None]] = []
    for attempt in _active_attempts(state):
        if attempt.attempt_id in cancelled_intent_ids:
            continue
        if attempt.job is None:
            if attempt.status is AttemptStatus.PREPARED:
                continue
            raise WorkflowError(
                f"active attempt {attempt.attempt_id!r} has no reconciled scheduler identity"
            )
        if attempt.submission_token is None:
            raise WorkflowError(
                f"active attempt {attempt.attempt_id!r} has no durable submission token"
            )
        identity = _scheduler_job(attempt.job)
        ownership = active_scheduler.verify_ownership(identity, attempt.submission_token)
        if not ownership.matched:
            raise WorkflowError(
                "emergency cancellation stopped before cleanup at an ownership mismatch: "
                f"{ownership.reason}"
            )
        child_observation = active_scheduler.observe_owned(
            identity,
            attempt.submission_token,
        )
        planned_actions.append((attempt, identity, child_observation, ownership.username))

    credential_actions: list[dict[str, object]] = []
    if credentialed:
        try:
            broker = _build_emergency_credential_broker(task, workspace=workspace)
            for attempt, descriptor in credentialed:
                receipt = broker.revoke(
                    workspace=workspace,
                    expected_binding=descriptor.binding,
                    descriptor=descriptor,
                    cause=CredentialRevocationCause.CANCELLED,
                )
                if (
                    receipt.descriptor != descriptor
                    or receipt.cause is not CredentialRevocationCause.CANCELLED
                    or receipt.receipt_path.is_symlink()
                    or not receipt.receipt_path.is_file()
                ):
                    raise WorkflowError(
                        "credential broker returned invalid emergency cleanup evidence"
                    )
                credential_actions.append(
                    {
                        "attempt_id": attempt.attempt_id,
                        "descriptor_digest": descriptor.descriptor_digest,
                        "cause": receipt.cause.value,
                        "receipt_path": str(receipt.receipt_path.relative_to(workspace)),
                    }
                )
        except (CredentialError, OSError, RuntimeError, ValueError) as error:
            raise WorkflowError(
                "emergency credential cleanup requires manual reconciliation: "
                f"{type(error).__name__}"
            ) from error

    actions: list[dict[str, object]] = []
    for attempt, identity, child_observation, ownership_username in planned_actions:
        if (
            child_observation.status.terminal
            and child_observation.source is ObservationSource.ACCOUNTING
        ):
            action = "already_terminal"
        else:
            active_scheduler.cancel(identity)
            action = "cancelled"
        actions.append(
            {
                "job": _job_payload(identity),
                "action": action,
                "observed_status": child_observation.status.value,
                "observation_source": child_observation.source.value,
                "submission_token": attempt.submission_token,
                "ownership_user": ownership_username,
            }
        )
    cancelled_updates = {
        entry.original.attempt_id: entry.resolved
        for entry in intent_reconciliations
        if entry.resolved.status is AttemptStatus.CANCELLED and entry.original != entry.resolved
    }
    if cancelled_updates:
        desired = _replace_emergency_attempts(state, cancelled_updates)
        save_state(
            workspace / STATE_FILENAME,
            desired,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )
        state = desired
    actions.extend(
        {
            "attempt_id": entry.original.attempt_id,
            "job": None,
            "action": entry.action,
            "observed_status": JobStatus.UNKNOWN.value,
            "observation_source": ObservationSource.UNKNOWN.value,
            "submission_token": entry.original.submission_token,
            "ownership_user": None,
        }
        for entry in intent_reconciliations
        if entry.resolved.status is AttemptStatus.CANCELLED
    )
    receipt = {
        "run_id": state.run_id,
        "task_digest": state.task_digest,
        "reason": reason,
        "controller_job": _job_payload(controller_job),
        "controller_status": observation.status.value,
        "controller_observation_source": observation.source.value,
        "controller_submission_token": controller_token,
        "controller_ownership_user": controller_ownership.username,
        "credential_actions": credential_actions,
        "child_actions": actions,
    }
    _write_once_json(receipt_path, receipt)


def _reconcile_emergency_submission_intents(
    *,
    workspace: Path,
    state: RunState,
    scheduler: Scheduler,
    cancellation_reason: str,
) -> tuple[_EmergencyIntentReconciliation, ...]:
    """Resolve every active token-only intent without making a scheduler submission."""
    from .controller.attempts import (
        AttemptAction,
        AttemptEngineError,
        cancel_submitting_intent_from_journal,
    )

    reconciled: list[_EmergencyIntentReconciliation] = []
    for attempt in _active_attempts(state):
        if attempt.status is not AttemptStatus.SUBMITTING or attempt.job is not None:
            continue
        token = attempt.submission_token
        if token is None:
            raise WorkflowError(
                f"SUBMITTING attempt {attempt.attempt_id!r} has no durable submission token"
            )
        intent_revision = f"{attempt.generation}:{attempt.sequence}:{attempt.attempt_id}"
        journal_dir = _emergency_submission_journal(workspace, attempt)
        fallback_digest = submission_intent_digest(
            "emergency-cancellation",
            state.run_id,
            state.task_digest,
            intent_revision,
            token,
        )
        intent_digest = _emergency_submission_intent_digest(
            journal_dir,
            submission_token=token,
            intent_revision=intent_revision,
            fallback_digest=fallback_digest,
        )
        try:
            result = cancel_submitting_intent_from_journal(
                attempt,
                scheduler=scheduler,
                journal_dir=journal_dir,
                intent_digest=intent_digest,
                cancellation_reason=cancellation_reason,
                submission_clock=_emergency_cancellation_clock,
                recover_unique_after_manual=True,
            )
        except AttemptEngineError as error:
            raise WorkflowError(
                f"emergency submission cancellation failed closed: {error}"
            ) from error
        if result.action is AttemptAction.ADOPTED:
            action = "adopted_and_cancelled"
        elif result.action is AttemptAction.CANCELLED_INTENT:
            action = "cancelled_intent"
        elif result.action is AttemptAction.WAITING_FOR_ACCOUNTING:
            raise WorkflowError(
                "emergency submission cancellation is waiting for bounded evidence: "
                f"{result.attempt.terminal_reason}"
            )
        else:
            raise WorkflowError(
                "emergency submission cancellation requires manual reconciliation: "
                f"{result.attempt.terminal_reason}"
            )
        reconciled.append(_EmergencyIntentReconciliation(attempt, result.attempt, action))
    return tuple(reconciled)


def _emergency_submission_journal(workspace: Path, attempt: AttemptRecord) -> Path:
    root = workspace / "receipts" / "submissions"
    if attempt.item_id == PLANNING_ITEM_ID:
        return root / "planning" / attempt.attempt_id
    return root / attempt.attempt_id


def _emergency_submission_intent_digest(
    journal_dir: Path,
    *,
    submission_token: str,
    intent_revision: str,
    fallback_digest: str,
) -> str:
    """Recover the exact journal digest, or bind an intent that was never claimed."""
    identities: set[tuple[str, str, str]] = set()
    for path in sorted(journal_dir.glob("*.json")):
        record = _load_json_mapping(path, "submission recovery record")
        identity = (
            record.get("submission_token"),
            record.get("intent_revision"),
            record.get("intent_digest"),
        )
        if not all(isinstance(value, str) for value in identity):
            raise WorkflowError("submission recovery record has an invalid intent identity")
        identities.add(cast(tuple[str, str, str], identity))
    if not identities:
        return fallback_digest
    if len(identities) != 1:
        raise WorkflowError("submission recovery journal contains conflicting intent identities")
    token, revision, digest = identities.pop()
    if token != submission_token or revision != intent_revision:
        raise WorkflowError("submission recovery journal differs from authoritative intent")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise WorkflowError("submission recovery journal has an invalid intent digest")
    return digest


def _replace_emergency_attempts(
    state: RunState,
    updates: Mapping[str, AttemptRecord],
) -> RunState:
    """Persist exact emergency lifecycle updates in one authoritative revision."""
    remaining = dict(updates)
    planning_attempts = tuple(
        remaining.pop(attempt.attempt_id, attempt) for attempt in state.planning_attempts
    )
    items = []
    for item in state.items:
        updated_item = item
        for attempt in item.attempts:
            replacement = remaining.pop(attempt.attempt_id, None)
            if replacement is not None:
                updated_item = updated_item.replace_attempt(replacement)
        items.append(updated_item)
    if remaining:
        raise WorkflowError(f"emergency attempt updates are not authoritative: {sorted(remaining)}")
    return replace(
        state,
        planning_attempts=planning_attempts,
        items=tuple(items),
        revision=state.revision + 1,
    )


def _emergency_cancellation_clock() -> datetime:
    """Return the trusted wall clock used by emergency submission recovery."""
    return datetime.now(UTC)


def _build_emergency_credential_broker(
    task: NormalizedTask,
    *,
    workspace: Path,
) -> CredentialBroker:
    """Construct the production metadata-only broker for emergency cleanup."""
    from .internal import _build_credential_broker

    return _build_credential_broker(task, workspace=workspace)


def record_human_response(
    workspace: Path,
    *,
    request_id: str,
    response: str,
    scheduler: Scheduler | None = None,
    preflight_facts_provider: PreflightFactsProvider | None = None,
    preflight_check: PreflightCheck | None = None,
) -> JobIdentity | None:
    """Publish response/launcher mailboxes without writing authoritative state."""
    workspace = workspace.resolve(strict=True)
    if not _SAFE_ID.fullmatch(request_id):
        raise ValueError("request_id must be a safe identifier")
    if not response.strip():
        raise ValueError("response must be non-empty")
    request_path = workspace / HUMAN_REQUESTS_DIRECTORY / f"{request_id}.request.json"
    if not request_path.is_file():
        raise WorkflowError(f"unknown human-input request {request_id!r}")
    request = _load_json_mapping(request_path, "human-input request")
    task = load_normalized_task(workspace / NORMALIZED_TASK_FILENAME)
    state = load_state(workspace / STATE_FILENAME)
    if request.get("run_id") != state.run_id or request.get("task_digest") != state.task_digest:
        raise WorkflowError("human-input request does not belong to this run and task")
    if request.get("request_id") != request_id:
        raise WorkflowError("human-input request identity does not match its filename")
    human_request = next(
        (entry for entry in reversed(state.human_requests) if entry.request_id == request_id),
        None,
    )
    if human_request is None:
        raise WorkflowError("human-input request is absent from authoritative state")
    response_text = response.strip()
    response_digest = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    response_payload = {
        "run_id": state.run_id,
        "task_digest": state.task_digest,
        "request_id": request_id,
        "response": response_text,
        "response_digest": response_digest,
    }
    _write_once_json(
        workspace / HUMAN_REQUESTS_DIRECTORY / f"{request_id}.response.json",
        response_payload,
    )

    if (
        human_request.response_digest is not None
        and human_request.response_digest != response_digest
    ):
        raise WorkflowError("human-input response digest differs from the frozen response")
    if not human_request.controller_requeue:
        return None
    if task.execution.mode == "local":
        if scheduler is None:
            raise WorkflowError("local detached response requires an explicitly injected scheduler")
        raise WorkflowError("local execution does not submit controller successors")
    provider = preflight_facts_provider or _persisted_preflight_provider(workspace)
    _require_production_preflight(
        task,
        "login",
        provider,
        preflight_check,
    )
    intent_path = workspace / HUMAN_REQUESTS_DIRECTORY / f"{request_id}.resume.intent.json"
    receipt_path = workspace / HUMAN_REQUESTS_DIRECTORY / f"{request_id}.resume.receipt.json"
    if intent_path.is_file():
        resume_intent = _load_json_mapping(intent_path, "human resume launcher intent")
        if (
            resume_intent.get("run_id") != state.run_id
            or resume_intent.get("task_digest") != state.task_digest
            or resume_intent.get("request_id") != request_id
            or resume_intent.get("response_digest") != response_digest
        ):
            raise WorkflowError("human resume launcher intent conflicts with this response")
        predecessor = _job_from_payload(
            resume_intent.get("predecessor_job"),
            "human resume predecessor",
        )
        generation_value = resume_intent.get("generation")
        submission_token_value = resume_intent.get("submission_token")
        owner_nonce_value = resume_intent.get("owner_nonce")
        if (
            isinstance(generation_value, bool)
            or not isinstance(generation_value, int)
            or not isinstance(submission_token_value, str)
            or not isinstance(owner_nonce_value, str)
        ):
            raise WorkflowError("human resume launcher intent has invalid command fields")
        successor_generation = generation_value
        submission_token = submission_token_value
        owner_nonce = owner_nonce_value
    else:
        if state.controller_job is None:
            raise WorkflowError("human-input resume requires an exact predecessor job")
        if human_request.generation != state.generation:
            raise WorkflowError("new human resume intent requires the requesting generation")
        predecessor = _scheduler_job(state.controller_job)
        material = f"{state.run_id}\0{request_id}\0{response_digest}"
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        submission_token = f"controller-resume-{digest[:32]}"
        owner_nonce = f"owner-{digest[32:56]}"
        successor_generation = state.generation + 1
        _write_once_json(
            intent_path,
            {
                "schema_version": 1,
                "kind": "human_response",
                "run_id": state.run_id,
                "task_digest": state.task_digest,
                "request_id": request_id,
                "response_digest": response_digest,
                "predecessor_generation": state.generation,
                "predecessor_job": _job_payload(predecessor),
                "generation": successor_generation,
                "submission_token": submission_token,
                "owner_nonce": owner_nonce,
                "reason": f"human response {request_id}",
            },
        )
    authoritative_successor = next(
        (
            entry
            for entry in reversed(state.controller_successors)
            if entry.human_request_id == request_id
            and entry.submission_token == submission_token
            and entry.successor_job is not None
        ),
        None,
    )
    if authoritative_successor is not None:
        identity = _scheduler_job(cast(JobReference, authoritative_successor.successor_job))
        _write_launcher_receipt(receipt_path, intent_path, submission_token, identity)
        return identity
    active_scheduler = scheduler or SlurmScheduler(cluster=predecessor.cluster)
    return _submit_launcher_intent(
        intent_path=intent_path,
        receipt_path=receipt_path,
        scheduler=active_scheduler,
        resources=replace(_controller_resources(task, workspace), requeue=False),
        command=InternalCommand(
            InternalEntrypoint.CONTROLLER,
            workspace=workspace,
            generation=successor_generation,
            owner_nonce=owner_nonce,
        ),
        submission_token=submission_token,
        environment=dict(task.execution.slurm.controller.environment),
        dependency=JobDependency(DependencyType.AFTERANY, predecessor),
    )


def run_controller(
    workspace: Path,
    *,
    generation: int,
    owner_nonce: str,
    scheduler: Scheduler | None = None,
    runtime_factory: RuntimeFactory | None = None,
    credential_broker: CredentialBroker | None = None,
    poll_interval_seconds: float = 1.0,
    sleeper: Callable[[float], None] = time.sleep,
    signal_flags: _ControllerSignalFlags | None = None,
    max_ticks: int | None = None,
    preflight_facts_provider: PreflightFactsProvider | None = None,
    preflight_check: PreflightCheck | None = None,
) -> None:
    """Run one signal-aware, generation-fenced controller process."""
    from .controller.runtime import RuntimeCallbacks, RuntimeDisposition

    _validate_poll_interval(poll_interval_seconds)
    if max_ticks is not None and max_ticks < 1:
        raise ValueError("max_ticks must be positive when supplied")
    workspace = workspace.resolve(strict=True)
    task = load_normalized_task(workspace / NORMALIZED_TASK_FILENAME)
    state_path = workspace / STATE_FILENAME
    current_job = _current_controller_job(task)
    active_scheduler = _controller_scheduler(task, scheduler, current_job=current_job)
    if task.execution.mode == "slurm":
        provider = preflight_facts_provider or _persisted_preflight_provider(workspace)
        _require_production_preflight(
            task,
            "controller",
            provider,
            preflight_check,
        )
    if state_path.is_file():
        state = load_state(state_path)
    else:
        state = _bootstrap_initial_controller(
            workspace=workspace,
            state_path=state_path,
            task=task,
            generation=generation,
            current_job=current_job,
            scheduler=active_scheduler,
        )
    if state.task_digest != task.digest:
        raise WorkflowError("controller task digest does not match authoritative state")
    if state.generation != generation:
        state = _self_bootstrap_successor(
            workspace=workspace,
            state=state,
            generation=generation,
            current_job=current_job,
            scheduler=active_scheduler,
        )
    if state.generation != generation:
        raise WorkflowError("controller generation does not match authoritative state")
    state = _adopt_controller_from_environment(state_path, state, current_job=current_job)
    _validate_current_controller_identity(task, state, current_job)
    if state.terminal_status is not RunTerminalStatus.ACTIVE:
        _publish_terminal(workspace, state, task, active_scheduler)
        return
    if runtime_factory is None:
        raise WorkflowError(
            "production runtime construction is unavailable without frozen RuntimeDomainInputs; "
            "refusing to start a partially configured controller"
        )

    lease = _activate_and_acquire_lease(
        workspace=workspace,
        state_path=state_path,
        state=state,
        scheduler=active_scheduler,
        current_job=current_job,
        generation=generation,
        owner_nonce=owner_nonce,
        poll_interval_seconds=poll_interval_seconds,
        sleeper=sleeper,
    )
    retain_lease_record = True
    try:
        state = _adopt_orphaned_attempts(
            workspace=workspace,
            state=load_state(state_path),
            task=task,
            scheduler=active_scheduler,
            credential_broker=credential_broker or _WorkspaceRecoveryCredentialBroker(),
        )
        flags = signal_flags or _ControllerSignalFlags()
        callbacks = RuntimeCallbacks(
            heartbeat=lease.heartbeat,
            cancellation_reason=lambda: _cancellation_reason(
                workspace,
                task.digest,
                state.run_id,
            ),
            stop_dispatch_requested=lambda: flags.stop_dispatch,
            requeue_requested=lambda: False,
            waiting_input_reason=lambda: None,
        )
        factory = runtime_factory
        runtime = factory(
            workspace=workspace,
            task=task,
            scheduler=active_scheduler,
            generation=generation,
            callbacks=callbacks,
        )
        ticks = 0
        with _installed_signal_handlers(flags, enabled=signal_flags is None):
            while True:
                current = load_state(state_path)
                _validate_runtime_state(current, task, generation, current_job)
                cancel_reason = _cancellation_reason(workspace, task.digest, current.run_id)
                checkpoint_signal = flags.checkpoint_signal
                if checkpoint_signal is not None and cancel_reason is None:
                    if task.execution.mode == "local":
                        _checkpoint_controller(
                            state_path,
                            current,
                            checkpoint_signal,
                            requeue_requested=False,
                        )
                        _write_status(workspace, load_state(state_path), observation=None)
                        return
                    if not task.execution.slurm.controller.requeue:
                        _checkpoint_controller(
                            state_path,
                            current,
                            checkpoint_signal,
                            requeue_requested=False,
                        )
                        _write_status(workspace, load_state(state_path), observation=None)
                        return
                    checkpointed = _checkpoint_controller(
                        state_path,
                        current,
                        checkpoint_signal,
                        requeue_requested=True,
                    )
                    _write_status(workspace, checkpointed, observation=None)
                    _drive_successor_submission(
                        workspace=workspace,
                        task=task,
                        scheduler=active_scheduler,
                        expected_generation=generation,
                        reason=checkpointed.controller_checkpoints[-1].reason,
                    )
                    return

                lease.heartbeat()
                result = runtime.tick()
                ticks += 1
                current = load_state(state_path)
                _validate_runtime_result(result, current, generation)
                _write_status(workspace, current, observation=None)
                if result.disposition is RuntimeDisposition.TERMINAL:
                    _publish_terminal(workspace, current, task, active_scheduler)
                    retain_lease_record = False
                    return
                if result.disposition is RuntimeDisposition.WAITING_INPUT:
                    waiting = _checkpoint_waiting_for_input(
                        workspace,
                        state_path,
                        current,
                        reason=result.reason,
                        controller_requeue=task.execution.slurm.controller.requeue,
                    )
                    _write_status(workspace, waiting, observation=None)
                    return
                if result.disposition is RuntimeDisposition.REQUEUE:
                    raise WorkflowError(
                        "runtime requested requeue without a controller signal boundary"
                    )
                if max_ticks is not None and ticks >= max_ticks:
                    raise WorkflowError("controller test tick bound was exhausted")
                sleeper(poll_interval_seconds)
    finally:
        lease.release(remove_record=not retain_lease_record)


def _bootstrap_initial_controller(
    *,
    workspace: Path,
    state_path: Path,
    task: NormalizedTask,
    generation: int,
    current_job: JobIdentity | None,
    scheduler: Scheduler,
) -> RunState:
    if task.execution.mode != "slurm" or generation != 1 or current_job is None:
        raise WorkflowError("only an exact initial Slurm controller may create run state")
    intent = _load_json_mapping(
        workspace / INITIAL_LAUNCH_INTENT_FILENAME,
        "initial launcher intent",
    )
    run_id = intent.get("run_id")
    if not isinstance(run_id, str):
        raise WorkflowError("initial launcher intent has no valid run_id")
    _require_launcher_intent(
        intent,
        kind="initial",
        run_id=run_id,
        task_digest=task.digest,
        generation=1,
    )
    token = cast(str, intent["submission_token"])
    ownership = scheduler.verify_ownership(current_job, token)
    if not ownership.matched:
        raise WorkflowError(
            f"initial controller scheduler ownership could not be proven: {ownership.reason}"
        )
    lookup = scheduler.lookup_submission(token)
    if len(lookup.matches) > 1:
        quarantine = _write_duplicate_submission_quarantine(
            workspace,
            token,
            lookup.matches,
            kind="initial-controller",
            reason=lookup.reason or "ambiguous initial controller token",
        )
        raise WorkflowError(f"initial controller duplicates are quarantined in {quarantine}")
    if lookup.identity != current_job:
        raise WorkflowError("initial controller token does not resolve to this exact process")
    state = RunState(
        run_id=run_id,
        task_digest=task.digest,
        base_commit=task.repository.base_commit,
        generation=1,
        workflow_mode=WorkflowMode(cast(str, intent["workflow_mode"])),
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token=token,
        controller_job=_state_job(current_job),
    )
    try:
        initialize_state(state_path, state)
        return state
    except StateConflictError:
        persisted = load_state(state_path)
        if persisted != state:
            raise
        return persisted


def _self_bootstrap_successor(
    *,
    workspace: Path,
    state: RunState,
    generation: int,
    current_job: JobIdentity | None,
    scheduler: Scheduler,
) -> RunState:
    if current_job is None:
        raise WorkflowError("successor self-bootstrap requires an exact Slurm identity")
    from .controller.successor import SuccessorProtocolError, bootstrap_successor_process

    token: str | None = None
    reason: str | None = None
    request_id: str | None = None
    response_digest: str | None = None
    pending = state.controller_successors[-1] if state.controller_successors else None
    if pending is None or pending.predecessor_generation != state.generation:
        envelope = _load_successor_launcher_envelope(workspace, state, generation)
        token = cast(str, envelope["submission_token"])
        reason = cast(str, envelope["reason"])
        if envelope["kind"] == "human_response":
            request_id = cast(str, envelope["request_id"])
            response_digest = cast(str, envelope["response_digest"])
    try:
        return bootstrap_successor_process(
            state_path=workspace / STATE_FILENAME,
            scheduler=scheduler,
            generation=generation,
            current_job=current_job,
            submission_token=token,
            reason=reason,
            human_request_id=request_id,
            response_digest=response_digest,
        )
    except SuccessorProtocolError as error:
        raise WorkflowError(f"successor self-bootstrap failed: {error}") from error


def _load_successor_launcher_envelope(
    workspace: Path,
    state: RunState,
    generation: int,
) -> dict[str, object]:
    candidates: list[Path] = []
    if state.human_requests:
        request_id = state.human_requests[-1].request_id
        candidates.append(workspace / HUMAN_REQUESTS_DIRECTORY / f"{request_id}.resume.intent.json")
    candidates.append(workspace / LAUNCHER_DIRECTORY / f"recovery-g{generation}.intent.json")
    envelope = None
    intent_path = None
    for candidate in candidates:
        if not candidate.is_file():
            continue
        candidate_envelope = _load_json_mapping(candidate, "successor launcher intent")
        if candidate_envelope.get("generation") == generation:
            intent_path = candidate
            envelope = candidate_envelope
            break
    if intent_path is None or envelope is None:
        raise WorkflowError("successor process has no frozen launcher envelope")
    kind = envelope.get("kind")
    if kind not in {"human_response", "unexpected_recovery"}:
        raise WorkflowError("successor launcher intent has an unsupported kind")
    if (
        envelope.get("schema_version") != 1
        or envelope.get("run_id") != state.run_id
        or envelope.get("task_digest") != state.task_digest
        or envelope.get("predecessor_generation") != state.generation
        or envelope.get("generation") != generation
    ):
        raise WorkflowError("successor launcher intent crossed a run or generation fence")
    if state.controller_job is None:
        raise WorkflowError("successor launcher predecessor identity is missing")
    if _job_from_payload(envelope.get("predecessor_job"), "successor predecessor") != (
        _scheduler_job(state.controller_job)
    ):
        raise WorkflowError("successor launcher predecessor does not match state")
    for field in ("submission_token", "owner_nonce", "reason"):
        if not isinstance(envelope.get(field), str) or not cast(str, envelope[field]).strip():
            raise WorkflowError(f"successor launcher intent {field} is invalid")
    if kind == "human_response":
        for field in ("request_id", "response_digest"):
            if not isinstance(envelope.get(field), str):
                raise WorkflowError(f"human successor launcher intent {field} is invalid")
        response_path = (
            workspace
            / HUMAN_REQUESTS_DIRECTORY
            / f"{cast(str, envelope['request_id'])}.response.json"
        )
        response = _load_json_mapping(response_path, "human response")
        if response.get("response_digest") != envelope["response_digest"]:
            raise WorkflowError("human response mailbox digest differs from launcher intent")
    return envelope


def _adopt_controller_from_environment(
    state_path: Path,
    state: RunState,
    *,
    current_job: JobIdentity | None,
) -> RunState:
    if state.controller_bootstrap_status is not ControllerBootstrapStatus.SUBMITTING:
        return state
    if current_job is None:
        raise WorkflowError("SUBMITTING controller has no exact Slurm process identity")
    submitted = state.record_controller_submitted(_state_job(current_job))
    try:
        save_state(
            state_path,
            submitted,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )
        return submitted
    except StateConflictError:
        current = load_state(state_path)
        if current.controller_job != submitted.controller_job:
            raise
        return current


def _controller_scheduler(
    task: NormalizedTask,
    scheduler: Scheduler | None,
    *,
    current_job: JobIdentity | None,
) -> Scheduler:
    if task.execution.mode == "local":
        if scheduler is None:
            raise WorkflowError(
                "local controller execution requires an explicitly injected fake/developer scheduler"
            )
        return scheduler
    cluster = (
        current_job.cluster if current_job is not None else os.environ.get("SLURM_CLUSTER_NAME")
    )
    return scheduler or SlurmScheduler(cluster=cluster)


def _current_controller_job(task: NormalizedTask) -> JobIdentity | None:
    if task.execution.mode == "local":
        return None
    raw_job_id = os.environ.get("SLURM_JOB_ID")
    if not raw_job_id:
        raise WorkflowError("Slurm controller requires exact SLURM_JOB_ID")
    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        raise WorkflowError("controller jobs cannot be Slurm array elements")
    try:
        identity = JobIdentity.parse(
            raw_job_id,
            cluster=os.environ.get("SLURM_CLUSTER_NAME"),
        )
    except ValueError as error:
        raise WorkflowError(f"invalid Slurm controller identity: {error}") from error
    if identity.array_task_id is not None:
        raise WorkflowError("controller jobs cannot be Slurm array elements")
    return identity


def _validate_current_controller_identity(
    task: NormalizedTask,
    state: RunState,
    current_job: JobIdentity | None,
) -> None:
    if task.execution.mode == "local":
        if state.controller_bootstrap_status is not ControllerBootstrapStatus.PREPARED:
            raise WorkflowError("local controller bootstrap must remain PREPARED")
        return
    if state.controller_bootstrap_status is not ControllerBootstrapStatus.SUBMITTED:
        raise WorkflowError("Slurm controller bootstrap is not durably submitted")
    if state.controller_job is None or current_job is None:
        raise WorkflowError("Slurm controller exact job identity is incomplete")
    if _scheduler_job(state.controller_job) != current_job:
        raise WorkflowError("process Slurm identity differs from the fenced controller job")


def _validate_runtime_state(
    state: RunState,
    task: NormalizedTask,
    generation: int,
    current_job: JobIdentity | None,
) -> None:
    if state.task_digest != task.digest or state.generation != generation:
        raise WorkflowError("runtime state crossed its task or generation fence")
    if current_job is None:
        if state.controller_job is not None:
            raise WorkflowError("local runtime unexpectedly acquired a Slurm identity")
    elif state.controller_job is None or _scheduler_job(state.controller_job) != current_job:
        raise WorkflowError("runtime state changed the exact current controller job")


def _validate_runtime_result(
    result: RuntimeTickResult,
    state: RunState,
    generation: int,
) -> None:
    from .controller.runtime import RuntimeDisposition

    if result.revision != state.revision:
        raise WorkflowError("runtime result revision differs from authoritative state")
    if state.generation != generation:
        raise WorkflowError("runtime result crossed the controller generation fence")
    if result.disposition is RuntimeDisposition.TERMINAL:
        if state.terminal_status is RunTerminalStatus.ACTIVE:
            raise WorkflowError("runtime reported terminal without a durable terminal state")
        if result.terminal_status is not state.terminal_status:
            raise WorkflowError("runtime terminal result disagrees with authoritative state")


def _activate_and_acquire_lease(
    *,
    workspace: Path,
    state_path: Path,
    state: RunState,
    scheduler: Scheduler,
    current_job: JobIdentity | None,
    generation: int,
    owner_nonce: str,
    poll_interval_seconds: float,
    sleeper: Callable[[float], None],
) -> ControllerLease:
    lease_path = workspace / LEASE_FILENAME
    if generation == 1:
        if state.controller_lifecycle is not ControllerLifecycle.RUNNING:
            raise WorkflowError("initial controller lifecycle is not RUNNING")
        return ControllerLease.acquire(
            lease_path,
            run_id=state.run_id,
            generation=generation,
            controller_job=state.controller_job,
            owner_nonce=owner_nonce,
        )
    if current_job is None:
        raise WorkflowError("a successor controller requires an exact Slurm job identity")

    from .controller.successor import (
        SuccessorDisposition,
        activate_successor,
        recover_successor_lease,
    )

    while True:
        activation = activate_successor(
            state_path=state_path,
            scheduler=scheduler,
            generation=generation,
            current_job=current_job,
        )
        if activation.disposition is SuccessorDisposition.READY:
            break
        if activation.disposition is SuccessorDisposition.BLOCKED:
            raise WorkflowError(f"successor activation blocked: {activation.reason}")
        sleeper(poll_interval_seconds)
    try:
        return recover_successor_lease(
            lease_path=lease_path,
            state_path=state_path,
            generation=generation,
            current_job=current_job,
            owner_nonce=owner_nonce,
        )
    except RuntimeError as error:
        raise WorkflowError(f"successor lease recovery failed: {error}") from error


def _adopt_orphaned_attempts(
    *,
    workspace: Path,
    state: RunState,
    task: NormalizedTask,
    scheduler: Scheduler,
    credential_broker: CredentialBroker,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> RunState:
    """Adopt exact old-generation attempts before permitting new dispatch."""
    if state.generation == 1:
        return state
    from .controller.recovery import OrphanAction, plan_attempt_adoption, plan_orphan_recovery

    plan = plan_attempt_adoption(state)
    window = next(
        (entry for entry in state.orphan_grace_windows if entry.generation == state.generation),
        None,
    )
    if plan.pending and window is None:
        started, deadline = _orphan_window_times(workspace, state, task)
        desired = state.start_orphan_grace(
            started_at=started.isoformat(),
            deadline_at=deadline.isoformat(),
        )
        save_state(
            workspace / STATE_FILENAME,
            desired,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )
        state = desired
        plan = plan_attempt_adoption(state)
        window = state.orphan_grace_windows[-1]

    if plan.pending:
        if window is None:
            raise WorkflowError("orphan adoption has no durable grace window")
        started = datetime.fromisoformat(window.started_at)
        deadline = datetime.fromisoformat(window.deadline_at)
        now = clock()
        if now.tzinfo is None:
            raise WorkflowError("orphan recovery clock must be timezone-aware")
        state = _bind_late_orphan_submissions(
            workspace=workspace,
            state=state,
            scheduler=scheduler,
        )
        plan = plan_attempt_adoption(state)
        elapsed_seconds = max(0, int((now - started).total_seconds()))
        successor_was_timely = now < deadline
        observations = {}
        for target in plan.pending:
            if target.job is None:
                continue
            if target.submission_token is None:
                raise WorkflowError("orphaned submitted attempt has no durable token")
            identity = _scheduler_job(target.job)
            ownership = scheduler.verify_ownership(identity, target.submission_token)
            if not ownership.matched:
                raise WorkflowError(
                    f"orphan adoption ownership could not be proven: {ownership.reason}"
                )
            observations[target.job] = scheduler.observe_owned(
                identity,
                target.submission_token,
            )
        orphan_plan = plan_orphan_recovery(
            tuple(observations.values()),
            successor_present=successor_was_timely,
            elapsed_seconds=elapsed_seconds,
            grace_seconds=task.execution.slurm.controller.orphan_grace_seconds,
        )
        if orphan_plan.action is OrphanAction.WAIT_FOR_SUCCESSOR:
            raise WorkflowError("active successor produced an invalid orphan-wait action")

        existing_actions = {
            receipt.job
            for receipt in state.orphan_action_receipts
            if receipt.generation == state.generation
        }
        cancel_identities = set(orphan_plan.cancel)
        for target in plan.pending:
            if target.job is not None:
                observation = observations[target.job]
                if orphan_plan.action is OrphanAction.CLEAN_UP:
                    identity = _scheduler_job(target.job)
                    if identity in cancel_identities:
                        ownership = scheduler.verify_ownership(
                            identity,
                            cast(str, target.submission_token),
                        )
                        if not ownership.matched:
                            raise WorkflowError(
                                f"orphan cleanup ownership could not be proven: {ownership.reason}"
                            )
                        scheduler.cancel(identity)
                    if target.job not in existing_actions:
                        desired = state.record_orphan_action(
                            action=OrphanReceiptAction.CLEANED_UP,
                            job=target.job,
                            scheduler_state=observation.status.value,
                        )
                        save_state(
                            workspace / STATE_FILENAME,
                            desired,
                            expected_revision=state.revision,
                            expected_generation=state.generation,
                        )
                        state = desired
                        existing_actions.add(target.job)
                elif not observation.status.terminal and target.job not in existing_actions:
                    desired = state.record_orphan_action(
                        action=OrphanReceiptAction.ADOPTED,
                        job=target.job,
                        scheduler_state=observation.status.value,
                    )
                    save_state(
                        workspace / STATE_FILENAME,
                        desired,
                        expected_revision=state.revision,
                        expected_generation=state.generation,
                    )
                    state = desired
                    existing_actions.add(target.job)
        if orphan_plan.action is OrphanAction.CLEAN_UP:
            _revoke_expired_orphan_credentials(
                workspace=workspace,
                state=state,
                attempt_ids={target.attempt_id for target in plan.pending},
                credential_broker=credential_broker,
            )
            for target in plan.pending:
                if target.job is not None:
                    continue
                desired = state.record_expired_attempt_cleanup(
                    attempt_id=target.attempt_id,
                    attempt_generation=target.attempt_generation,
                    status=target.status,
                    submission_token=target.submission_token,
                    job=target.job,
                )
                save_state(
                    workspace / STATE_FILENAME,
                    desired,
                    expected_revision=state.revision,
                    expected_generation=state.generation,
                )
                state = desired
        for target in plan.pending:
            if orphan_plan.action is OrphanAction.CLEAN_UP and target.job is None:
                continue
            desired = state.record_attempt_adoption(
                attempt_id=target.attempt_id,
                attempt_generation=target.attempt_generation,
                status=target.status,
                submission_token=target.submission_token,
                job=target.job,
            )
            if desired is state:
                continue
            save_state(
                workspace / STATE_FILENAME,
                desired,
                expected_revision=state.revision,
                expected_generation=state.generation,
            )
            state = desired
    plan_attempt_adoption(state).require_new_work_allowed()
    _fence_unadopted_orphan_credentials(
        workspace=workspace,
        state=state,
        credential_broker=credential_broker,
    )
    return state


def _bind_late_orphan_submissions(
    *,
    workspace: Path,
    state: RunState,
    scheduler: Scheduler,
) -> RunState:
    """Adopt exact jobs accepted before an older controller persisted their IDs."""
    from .controller.recovery import plan_attempt_adoption

    for target in plan_attempt_adoption(state).pending:
        if (
            target.job is not None
            or target.status is not AttemptStatus.SUBMITTING
            or target.submission_token is None
        ):
            continue
        try:
            probe = scheduler.probe_submission(target.submission_token)
        except SchedulerError as error:
            raise WorkflowError(
                f"old-generation submission probe failed for {target.attempt_id}: {error}"
            ) from error
        if len(probe.matches) > 1:
            raise WorkflowError(
                f"old-generation token for {target.attempt_id} is ambiguous across "
                f"{len(probe.matches)} jobs"
            )
        if probe.identity is None:
            if not probe.absence_proven:
                raise WorkflowError(
                    f"old-generation token absence is not provable for {target.attempt_id}: "
                    f"{probe.reason}"
                )
            continue
        ownership = scheduler.verify_ownership(probe.identity, target.submission_token)
        if not ownership.matched:
            raise WorkflowError(
                f"old-generation submission ownership is ambiguous: {ownership.reason}"
            )
        state = _persist_late_orphan_identity(
            workspace=workspace,
            state=state,
            attempt_id=target.attempt_id,
            identity=probe.identity,
            scheduler_state=probe.status.value,
        )
    return state


def _persist_late_orphan_identity(
    *,
    workspace: Path,
    state: RunState,
    attempt_id: str,
    identity: JobIdentity,
    scheduler_state: str,
) -> RunState:
    attempt = next(
        (
            entry
            for entry in (
                *state.planning_attempts,
                *(candidate for item in state.items for candidate in item.attempts),
            )
            if entry.attempt_id == attempt_id
        ),
        None,
    )
    if attempt is None:
        raise WorkflowError("late orphan identity references an unknown attempt")
    submitted = attempt.transition(
        AttemptStatus.SUBMITTED,
        job=_state_job(identity),
        scheduler_state=scheduler_state,
        terminal_reason=None,
    )
    if attempt.item_id == PLANNING_ITEM_ID:
        desired = replace(
            state,
            planning_attempts=tuple(
                submitted if entry.attempt_id == attempt_id else entry
                for entry in state.planning_attempts
            ),
            revision=state.revision + 1,
        )
    else:
        item = state.item(attempt.item_id)
        updated = item.replace_attempt(submitted)
        desired = replace(
            state,
            items=tuple(
                updated if entry.item_id == item.item_id else entry for entry in state.items
            ),
            revision=state.revision + 1,
        )
    save_state(
        workspace / STATE_FILENAME,
        desired,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    return desired


def _revoke_expired_orphan_credentials(
    *,
    workspace: Path,
    state: RunState,
    attempt_ids: set[str],
    credential_broker: CredentialBroker,
) -> None:
    """Revoke the exact credential handles whose orphan grace expired."""
    from .common.credentials import CredentialRevocationCause, CredentialRevocationReceipt

    descriptors = _old_generation_credential_descriptors(workspace, state)
    for attempt, descriptor in descriptors:
        if attempt.attempt_id not in attempt_ids:
            continue
        receipt = credential_broker.revoke(
            workspace=workspace,
            expected_binding=descriptor.binding,
            descriptor=descriptor,
            cause=CredentialRevocationCause.ORPHAN_EXPIRED,
        )
        if (
            not isinstance(receipt, CredentialRevocationReceipt)
            or receipt.descriptor != descriptor
            or receipt.cause is not CredentialRevocationCause.ORPHAN_EXPIRED
        ):
            raise WorkflowError("credential broker returned a mismatched orphan-expiry receipt")
        path = receipt.receipt_path
        if path.is_symlink() or not path.is_file():
            raise WorkflowError("orphan-expiry credential receipt is not a regular file")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkflowError("orphan-expiry credential receipt is unreadable") from error
        if value != receipt.to_public_dict():
            raise WorkflowError("orphan-expiry credential receipt bytes do not match its type")


def _fence_unadopted_orphan_credentials(
    *,
    workspace: Path,
    state: RunState,
    credential_broker: CredentialBroker,
) -> None:
    """Revoke only old-generation provisions excluded from exact adoption."""
    from .controller.recovery import revoke_reconciled_generation_credentials

    descriptor_attempts = _old_generation_credential_descriptors(workspace, state)
    if not descriptor_attempts:
        return
    adopted_jobs = {
        receipt.job
        for receipt in state.orphan_action_receipts
        if receipt.generation == state.generation and receipt.action is OrphanReceiptAction.ADOPTED
    }
    cleaned_jobs = {
        receipt.job
        for receipt in state.orphan_action_receipts
        if receipt.generation == state.generation
        and receipt.action is OrphanReceiptAction.CLEANED_UP
    }
    adopted_bindings = tuple(
        descriptor.binding
        for attempt, descriptor in descriptor_attempts
        if attempt.job is None or attempt.job in adopted_jobs or attempt.job in cleaned_jobs
    )
    revoke_reconciled_generation_credentials(
        state,
        broker=credential_broker,
        workspace=workspace,
        descriptors=tuple(descriptor for _attempt, descriptor in descriptor_attempts),
        adopted_bindings=adopted_bindings,
    )


def _old_generation_credential_descriptors(
    workspace: Path,
    state: RunState,
) -> tuple[tuple[AttemptRecord, CredentialDescriptor], ...]:
    """Load a bounded credential set from immutable old-attempt inputs."""
    from .common.artifacts import INPUT_FILENAME, ArtifactError, load_input_manifest
    from .common.credentials import CredentialError, descriptor_from_public_dict
    from .common.runners import RoleProcessError, load_role_spec

    attempts = tuple(
        sorted(
            (
                attempt
                for attempt in (
                    *state.planning_attempts,
                    *(attempt for item in state.items for attempt in item.attempts),
                )
                if attempt.generation < state.generation
                and attempt.status not in ATTEMPT_TERMINAL_STATUSES
            ),
            key=lambda attempt: (
                attempt.generation,
                attempt.item_id,
                attempt.sequence,
                attempt.attempt_id,
            ),
        )
    )
    result: list[tuple[AttemptRecord, CredentialDescriptor]] = []
    for attempt in attempts:
        raw_descriptor: object | None
        if attempt.item_id == PLANNING_ITEM_ID:
            input_path = (
                workspace
                / "items"
                / PLANNING_ITEM_ID
                / "attempts"
                / attempt.attempt_id
                / "role-input.json"
            )
            if not input_path.exists():
                continue
            try:
                role_input = load_role_spec(input_path)
            except (OSError, RoleProcessError, TypeError, ValueError) as error:
                raise WorkflowError(
                    f"old planning attempt credential manifest is invalid: {error}"
                ) from error
            raw_descriptor = role_input.credential_descriptor
        else:
            input_path = (
                workspace
                / "items"
                / attempt.item_id
                / "attempts"
                / f"{attempt.sequence:04d}"
                / INPUT_FILENAME
            )
            if not input_path.exists():
                continue
            try:
                manifest, _digest = load_input_manifest(input_path)
            except (ArtifactError, OSError, TypeError, ValueError) as error:
                raise WorkflowError(
                    f"old attempt credential manifest is invalid: {error}"
                ) from error
            if (
                manifest.run_id != state.run_id
                or manifest.item_id != attempt.item_id
                or manifest.attempt_id != attempt.attempt_id
                or manifest.task_digest != state.task_digest
                or manifest.generation != attempt.generation
            ):
                raise WorkflowError(
                    "old attempt credential manifest differs from authoritative state"
                )
            runtime = manifest.payload.get("runtime")
            if not isinstance(runtime, dict):
                raise WorkflowError("old attempt credential manifest lacks a runtime object")
            raw_descriptor = runtime.get("credential_descriptor")
        if raw_descriptor is None:
            continue
        try:
            descriptor = descriptor_from_public_dict(raw_descriptor)
        except CredentialError as error:
            raise WorkflowError(f"old attempt credential descriptor is invalid: {error}") from error
        binding = descriptor.binding
        expected_item_id = "planning" if attempt.item_id == PLANNING_ITEM_ID else attempt.item_id
        if (
            binding.run_id != state.run_id
            or binding.item_id != expected_item_id
            or binding.attempt_id != attempt.attempt_id
            or binding.task_digest != state.task_digest
            or binding.generation != attempt.generation
        ):
            raise WorkflowError(
                "old attempt credential descriptor differs from authoritative state"
            )
        result.append((attempt, descriptor))
    return tuple(result)


def _active_credential_descriptors(
    workspace: Path,
    state: RunState,
) -> tuple[tuple[AttemptRecord, CredentialDescriptor], ...]:
    """Load exact active descriptors from controller-private immutable inputs."""
    from .common.artifacts import INPUT_FILENAME, ArtifactError, load_input_manifest
    from .common.credentials import (
        CredentialError,
        descriptor_from_public_dict,
        validate_unique_credential_handles,
    )

    supported_roles = {
        Role.PLAN_DRAFTER,
        Role.PLAN_REVIEWER,
        Role.CODER,
        Role.REVIEWER,
        Role.QA,
    }
    result: list[tuple[AttemptRecord, CredentialDescriptor]] = []
    for attempt in _active_attempts(state):
        if attempt.role not in supported_roles:
            continue
        if attempt.item_id == PLANNING_ITEM_ID:
            input_path = (
                workspace
                / "items"
                / PLANNING_ITEM_ID
                / "attempts"
                / attempt.attempt_id
                / "role-input.json"
            )
            if not input_path.exists():
                if attempt.status is AttemptStatus.PREPARED:
                    continue
                raise WorkflowError(
                    "active planning credential descriptor is unavailable; "
                    "manual reconciliation is required"
                )
            role_input = _load_json_mapping(input_path, "active planning role input")
            raw_descriptor = role_input.get("credential_descriptor")
        else:
            input_path = (
                workspace
                / "items"
                / attempt.item_id
                / "attempts"
                / f"{attempt.sequence:04d}"
                / INPUT_FILENAME
            )
            if not input_path.exists():
                if attempt.status is AttemptStatus.PREPARED:
                    continue
                raise WorkflowError(
                    "active attempt credential descriptor is unavailable; "
                    "manual reconciliation is required"
                )
            try:
                manifest, _digest = load_input_manifest(input_path)
            except (ArtifactError, OSError, TypeError, ValueError) as error:
                raise WorkflowError(
                    "active attempt credential manifest is invalid; "
                    "manual reconciliation is required"
                ) from error
            if (
                manifest.run_id != state.run_id
                or manifest.item_id != attempt.item_id
                or manifest.attempt_id != attempt.attempt_id
                or manifest.task_digest != state.task_digest
                or manifest.generation != attempt.generation
            ):
                raise WorkflowError(
                    "active attempt credential manifest differs from authoritative state"
                )
            runtime = manifest.payload.get("runtime")
            if not isinstance(runtime, dict) or runtime.get("kind") != "agent":
                raise WorkflowError("active credentialed attempt lacks an agent runtime")
            raw_descriptor = runtime.get("credential_descriptor")
        if raw_descriptor is None:
            raise WorkflowError(
                "active attempt credential descriptor is unavailable; "
                "manual reconciliation is required"
            )
        try:
            descriptor = descriptor_from_public_dict(raw_descriptor)
        except CredentialError as error:
            raise WorkflowError("active attempt credential descriptor is invalid") from error
        expected_item_id = "planning" if attempt.item_id == PLANNING_ITEM_ID else attempt.item_id
        binding = descriptor.binding
        if (
            binding.run_id != state.run_id
            or binding.item_id != expected_item_id
            or binding.attempt_id != attempt.attempt_id
            or binding.task_digest != state.task_digest
            or binding.generation != attempt.generation
        ):
            raise WorkflowError(
                "active attempt credential descriptor differs from authoritative state"
            )
        result.append((attempt, descriptor))
    try:
        validate_unique_credential_handles(tuple(descriptor for _attempt, descriptor in result))
    except CredentialError as error:
        raise WorkflowError("active credential handles are not uniquely bound") from error
    return tuple(result)


def _orphan_window_times(
    workspace: Path,
    state: RunState,
    task: NormalizedTask,
) -> tuple[datetime, datetime]:
    recovery_intent = workspace / LAUNCHER_DIRECTORY / f"recovery-g{state.generation}.intent.json"
    if recovery_intent.is_file():
        envelope = _load_json_mapping(recovery_intent, "recovery launcher intent")
        started_raw = envelope.get("orphan_grace_started_at")
        deadline_raw = envelope.get("orphan_grace_deadline_at")
        if isinstance(started_raw, str) and isinstance(deadline_raw, str):
            try:
                started = datetime.fromisoformat(started_raw)
                deadline = datetime.fromisoformat(deadline_raw)
            except ValueError as error:
                raise WorkflowError("recovery orphan-grace timestamps are invalid") from error
            if started.tzinfo is None or deadline.tzinfo is None or deadline <= started:
                raise WorkflowError("recovery orphan-grace timestamps are not bounded")
            return started, deadline
    started = datetime.now(UTC)
    deadline = started + timedelta(seconds=task.execution.slurm.controller.orphan_grace_seconds)
    return started, deadline


def _checkpoint_controller(
    state_path: Path,
    state: RunState,
    checkpoint_signal: ControllerCheckpointSignal,
    *,
    requeue_requested: bool,
) -> RunState:
    reason = f"controller {checkpoint_signal.value} checkpoint for generation {state.generation}"
    desired = state.checkpoint_controller(
        reason=reason,
        requeue_requested=requeue_requested,
        signal=checkpoint_signal,
    )
    save_state(
        state_path,
        desired,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    return desired


def _checkpoint_waiting_for_input(
    workspace: Path,
    state_path: Path,
    state: RunState,
    *,
    reason: str,
    controller_requeue: bool,
) -> RunState:
    prompt_digest = hashlib.sha256(reason.encode("utf-8")).hexdigest()
    request_id = f"request-g{state.generation}-{prompt_digest[:12]}"
    request = {
        "run_id": state.run_id,
        "task_digest": state.task_digest,
        "request_id": request_id,
        "generation": state.generation,
        "prompt_digest": prompt_digest,
        "reason": reason,
        "controller_requeue": controller_requeue,
    }
    _write_once_json(
        workspace / HUMAN_REQUESTS_DIRECTORY / f"{request_id}.request.json",
        request,
    )
    desired = state.wait_for_human_input(
        request_id=request_id,
        prompt_digest=prompt_digest,
        controller_requeue=controller_requeue,
        reason=reason,
    )
    save_state(
        state_path,
        desired,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    return desired


def _drive_successor_submission(
    *,
    workspace: Path,
    task: NormalizedTask,
    scheduler: Scheduler,
    expected_generation: int,
    reason: str,
    human_request_id: str | None = None,
) -> JobIdentity:
    from .controller.successor import (
        SuccessorDisposition,
        SuccessorSubmissionPermit,
        tick_successor_submission,
    )

    state_path = workspace / STATE_FILENAME
    state = load_state(state_path)
    permit: SuccessorSubmissionPermit | None = None
    nonce_material = "\0".join(
        (
            state.run_id,
            str(expected_generation),
            reason,
            human_request_id or "",
        )
    )
    owner_nonce = f"owner-{hashlib.sha256(nonce_material.encode('utf-8')).hexdigest()[:24]}"
    command = InternalCommand(
        InternalEntrypoint.CONTROLLER,
        workspace=workspace,
        generation=expected_generation + 1,
        owner_nonce=owner_nonce,
    )
    for _step in range(4):
        result = tick_successor_submission(
            state_path=state_path,
            scheduler=scheduler,
            resources=replace(_controller_resources(task, workspace), requeue=False),
            command=command,
            expected_generation=expected_generation,
            reason=reason,
            environment=dict(task.execution.slurm.controller.environment),
            human_request_id=human_request_id,
            permit=permit,
        )
        if result.disposition is SuccessorDisposition.BLOCKED:
            raise WorkflowError(f"successor submission blocked: {result.reason}")
        if result.disposition is SuccessorDisposition.EXIT:
            if result.job is None:
                raise WorkflowError("fenced successor result has no exact job identity")
            return result.job
        permit = result.permit
    raise WorkflowError("successor protocol exceeded its bounded transition count")


def _cancellation_reason(workspace: Path, task_digest: str, run_id: str) -> str | None:
    cancel_path = workspace / CANCEL_REQUEST_FILENAME
    if not cancel_path.is_file():
        return None
    request = _load_json_mapping(cancel_path, "cancellation request")
    if request.get("run_id") != run_id or request.get("task_digest") != task_digest:
        raise WorkflowError("cancellation request belongs to another run or task")
    reason = request.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise WorkflowError("cancellation request has no non-empty reason")
    return reason.strip()


def _publish_terminal(
    workspace: Path,
    state: RunState,
    task: NormalizedTask,
    scheduler: Scheduler,
) -> None:
    from .common.artifacts import IngestedResult, load_result_manifest
    from .common.gates import (
        AccuracyCriteria,
        ClaimScope,
        EvidenceScope,
        GatePurpose,
        build_gate_suite,
        validate_receipt_claim,
    )
    from .common.reporting import GateEvidence, build_terminal_report, write_terminal_report
    from .controller.results import (
        GateResult,
        QaResult,
        decode_gate_result,
        decode_qa_result,
        load_candidate_receipt,
    )

    ingested_results: list[IngestedResult] = []
    gate_evidence: list[GateEvidence] = []
    item_by_id = {item.item_id: item for item in state.items}
    gates = build_gate_suite(
        entry_gpu_tests=task.gates.component_tests,
        collective_tests=task.gates.collective_tests,
        boot_tests=task.gates.boot_tests,
        accuracy=AccuracyCriteria(
            selector=task.gates.accuracy.selector,
            reference=task.gates.accuracy.reference,
            protocol=task.gates.accuracy.protocol,
            tolerance=task.gates.accuracy.tolerance,
        ),
        feature_signal_tests=task.gates.feature_signal_tests,
        base_environment=dict(task.execution.slurm.controller.environment),
    )
    for attempt, attempt_dir in _attempt_directories(workspace, state):
        if attempt.result_digest is None:
            continue
        manifest, result_digest = load_result_manifest(attempt_dir)
        if result_digest != attempt.result_digest:
            raise WorkflowError(
                f"terminal result digest differs from state for {attempt.attempt_id!r}"
            )
        receipt_path = (
            workspace / "receipts" / state.run_id / attempt.item_id / f"{attempt.attempt_id}.json"
        )
        ingested_results.append(IngestedResult(manifest, result_digest, receipt_path))
        if attempt.kind not in {AttemptKind.DETERMINISTIC_GATE, AttemptKind.QA}:
            continue
        item = item_by_id.get(attempt.item_id)
        if item is None or item.candidate_attempt_id is None:
            raise WorkflowError(
                f"terminal gate attempt {attempt.attempt_id!r} has no frozen candidate"
            )
        candidate_attempt = next(
            (entry for entry in item.attempts if entry.attempt_id == item.candidate_attempt_id),
            None,
        )
        if candidate_attempt is None:
            raise WorkflowError("terminal gate candidate attempt is absent from state")
        candidate_path = (
            workspace
            / "items"
            / item.item_id
            / "attempts"
            / f"{candidate_attempt.sequence:04d}"
            / "candidate-receipt.json"
        )
        candidate = load_candidate_receipt(candidate_path)
        if attempt.kind is AttemptKind.DETERMINISTIC_GATE:
            gate = _gate_spec_for_terminal_attempt(item, attempt, gates)
            decoded = decode_gate_result(
                manifest,
                attempt,
                candidate,
                expected_spec=gate,
            )
            if not isinstance(decoded, GateResult):
                raise WorkflowError("deterministic gate decoder returned another result kind")
            if decoded.receipt.purpose is GatePurpose.ACCURACY:
                claim = ClaimScope.ACCURACY
            elif decoded.receipt.purpose is GatePurpose.PERFORMANCE:
                claim = ClaimScope.PERFORMANCE
            elif decoded.receipt.scope is EvidenceScope.CPU_STATIC:
                claim = ClaimScope.STRUCTURAL
            elif decoded.receipt.scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
                claim = ClaimScope.LOCAL_FOUR_GPU_PRODUCT
            elif decoded.receipt.scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER:
                claim = ClaimScope.SYNTHETIC_RUNNER
            elif decoded.receipt.scope is EvidenceScope.REAL_MULTI_NODE_PRODUCT:
                claim = ClaimScope.MULTI_NODE_PRODUCT
            else:
                claim = ClaimScope.PRODUCT_CORRECTNESS
            validate_receipt_claim(decoded.receipt, claim, gate_spec=gate)
            receipts = (decoded.receipt,)
        else:
            expected_gate_results = {
                entry.attempt_id: entry.result_digest
                for entry in item.attempts
                if entry.kind is AttemptKind.DETERMINISTIC_GATE and entry.result_digest is not None
            }
            decoded = decode_qa_result(
                manifest,
                attempt,
                candidate,
                expected_gate_results=expected_gate_results,
            )
            if not isinstance(decoded, QaResult):
                raise WorkflowError("QA decoder returned another result kind")
            receipts = ()
        gate_evidence.extend(
            GateEvidence(attempt.attempt_id, result_digest, receipt) for receipt in receipts
        )

    observations = tuple(
        scheduler.observe_owned(identity, token) for identity, token in _owned_scheduler_jobs(state)
    )
    report = build_terminal_report(
        state,
        task,
        workspace=workspace,
        ingested_results=tuple(ingested_results),
        planning_evidence=_accepted_planning_evidence(workspace, state, task),
        gate_evidence=tuple(gate_evidence),
        scheduler_observations=observations,
    )
    write_terminal_report(workspace, report)
    _write_status(workspace, state, observation=None)


def _gate_spec_for_terminal_attempt(
    item: object,
    attempt: AttemptRecord,
    gates: tuple[object, ...],
) -> object:
    """Resolve one physical gate retry to its persisted logical gate identity."""
    from .controller.attempts import (
        AttemptEngineError,
        logical_attempt_ordinal,
        logical_attempt_root,
    )
    from .state import WorkItemRecord

    if not isinstance(item, WorkItemRecord) or attempt.kind is not AttemptKind.DETERMINISTIC_GATE:
        raise WorkflowError("terminal gate mapping requires a deterministic-gate attempt")
    gate_attempts = tuple(
        entry for entry in item.attempts if entry.kind is AttemptKind.DETERMINISTIC_GATE
    )
    if attempt not in gate_attempts:
        raise WorkflowError("terminal gate attempt is absent from its work item")
    children: dict[str, list[AttemptRecord]] = {}
    for entry in gate_attempts:
        if entry.predecessor_attempt_id is None:
            continue
        predecessor = next(
            (
                candidate
                for candidate in gate_attempts
                if candidate.attempt_id == entry.predecessor_attempt_id
            ),
            None,
        )
        if (
            predecessor is None
            or predecessor.role is not entry.role
            or predecessor.profile is not entry.profile
        ):
            raise WorkflowError("terminal gate retry lineage is incomplete or changes identity")
        children.setdefault(predecessor.attempt_id, []).append(entry)
    if any(len(entries) != 1 for entries in children.values()):
        raise WorkflowError("terminal gate retry lineage branches ambiguously")
    try:
        root = logical_attempt_root(item, attempt.attempt_id)
        gate_index = logical_attempt_ordinal(item, attempt.attempt_id) - 1
        validated_lineage = tuple(
            entry
            for entry in gate_attempts
            if entry.result_digest is not None
            and logical_attempt_root(item, entry.attempt_id).attempt_id == root.attempt_id
        )
    except AttemptEngineError as error:
        raise WorkflowError(f"terminal gate retry lineage is invalid: {error}") from error
    if len(validated_lineage) != 1 or validated_lineage[0].attempt_id != attempt.attempt_id:
        raise WorkflowError("terminal gate lineage has ambiguous validated evidence")
    if gate_index < 0 or gate_index >= len(gates):
        raise WorkflowError("terminal gate attempt exceeds the frozen suite")
    return gates[gate_index]


def _accepted_planning_evidence(
    workspace: Path,
    state: RunState,
    task: NormalizedTask,
) -> tuple[PlanningEvidence, ...]:
    """Revalidate and receipt the exact planning pair that admitted the run."""
    if not state.items or not state.planning_attempts:
        return ()
    from .common.artifacts import (
        INPUT_FILENAME,
        digest_file,
        load_input_manifest,
        worker_output_directory,
    )
    from .common.outcomes import PlanReviewDecision
    from .common.reporting import PLANNING_RECEIPT_SCHEMA_VERSION, PlanningEvidence
    from .common.runners import RoleProcessResult, load_role_spec
    from .controller.runtime import FrozenPlanError, recover_approved_plan

    try:
        plan, review = recover_approved_plan(workspace, task, state)
    except (FrozenPlanError, OSError, TypeError, ValueError) as error:
        raise WorkflowError(f"accepted planning evidence cannot be recovered: {error}") from error
    if review.outcome is not PlanReviewDecision.ACCEPT or review.plan_digest != plan.digest:
        raise WorkflowError("accepted planning evidence has no exact ACCEPT verdict")
    reviewer = state.planning_attempts[-1]
    drafter = next(
        (
            attempt
            for attempt in state.planning_attempts
            if attempt.attempt_id == reviewer.review_of_attempt_id
        ),
        None,
    )
    if drafter is None:
        raise WorkflowError("accepted planning evidence has no linked PlanDrafter")

    evidence: list[PlanningEvidence] = []
    for attempt, outcome, review_of in (
        (drafter, "DRAFTED", None),
        (reviewer, "ACCEPT", drafter.attempt_id),
    ):
        attempt_dir = workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id
        input_manifest, input_digest = load_input_manifest(attempt_dir / INPUT_FILENAME)
        role_input_path = attempt_dir / "role-input.json"
        role_input = load_role_spec(role_input_path)
        result_path = worker_output_directory(attempt_dir) / "role-result.json"
        result_value = _load_json_mapping(result_path, "planning role result")
        if set(result_value) != set(RoleProcessResult.__dataclass_fields__):
            raise WorkflowError("planning role result has an invalid strict schema")
        try:
            result = RoleProcessResult(**result_value)
        except TypeError as error:
            raise WorkflowError("planning role result fields are invalid") from error
        expected_identity = (
            state.run_id,
            state.task_digest,
            attempt.generation,
            PLANNING_ITEM_ID,
            attempt.attempt_id,
            attempt.role,
        )
        if (
            input_manifest.run_id,
            input_manifest.task_digest,
            input_manifest.generation,
            input_manifest.item_id,
            input_manifest.attempt_id,
            input_manifest.role,
        ) != expected_identity:
            raise WorkflowError("planning attempt input differs from authoritative state")
        if (
            role_input.run_id,
            role_input.task_digest,
            role_input.generation,
            role_input.item_id,
            role_input.attempt_id,
            role_input.role,
        ) != (
            state.run_id,
            state.task_digest,
            attempt.generation,
            "planning",
            attempt.attempt_id,
            attempt.role.value,
        ):
            raise WorkflowError("planning role input differs from authoritative state")
        if (
            result.run_id,
            result.task_digest,
            result.generation,
            result.item_id,
            result.attempt_id,
            result.prompt_id,
            result.role,
            result.profile,
        ) != (
            role_input.run_id,
            role_input.task_digest,
            role_input.generation,
            role_input.item_id,
            role_input.attempt_id,
            role_input.prompt_id,
            role_input.role,
            role_input.profile,
        ):
            raise WorkflowError("planning role result identity differs from immutable input")
        if hashlib.sha256(result.response.encode("utf-8")).hexdigest() != result.response_digest:
            raise WorkflowError("planning role response digest differs from response bytes")
        result_digest = digest_file(result_path)
        if result_digest != attempt.result_digest:
            raise WorkflowError("planning role result digest differs from authoritative state")
        role_input_digest = digest_file(role_input_path)
        receipt_path = (
            workspace / "receipts" / state.run_id / PLANNING_ITEM_ID / f"{attempt.attempt_id}.json"
        )
        payload: dict[str, object] = {
            "schema_version": PLANNING_RECEIPT_SCHEMA_VERSION,
            "run_id": state.run_id,
            "task_digest": state.task_digest,
            "item_id": PLANNING_ITEM_ID,
            "attempt_id": attempt.attempt_id,
            "generation": attempt.generation,
            "role": attempt.role.value,
            "outcome": outcome,
            "plan_digest": plan.digest,
            "attempt_input_digest": input_digest,
            "role_input_digest": role_input_digest,
            "result_digest": result_digest,
            "response_digest": result.response_digest,
            "review_of_attempt_id": review_of,
        }
        _write_once_json(receipt_path, payload)
        evidence.append(
            PlanningEvidence(
                attempt_id=attempt.attempt_id,
                role=attempt.role,
                outcome=outcome,
                plan_digest=plan.digest,
                attempt_input_digest=input_digest,
                role_input_digest=role_input_digest,
                result_digest=result_digest,
                response_digest=result.response_digest,
                receipt_path=receipt_path,
                receipt_digest=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                review_of_attempt_id=review_of,
            )
        )
    return tuple(evidence)


def _attempt_directories(
    workspace: Path,
    state: RunState,
) -> tuple[tuple[AttemptRecord, Path], ...]:
    work = tuple(
        (
            attempt,
            workspace / "items" / item.item_id / "attempts" / f"{attempt.sequence:04d}" / "output",
        )
        for item in state.items
        for attempt in item.attempts
    )
    return work


def _owned_scheduler_jobs(state: RunState) -> tuple[tuple[JobIdentity, str], ...]:
    references: list[tuple[JobReference, str]] = []
    if state.controller_job is not None:
        if state.controller_submission_token is None:
            raise WorkflowError("controller job has no immutable submission token")
        references.append((state.controller_job, state.controller_submission_token))
    for attempt in (
        *state.planning_attempts,
        *(attempt for item in state.items for attempt in item.attempts),
    ):
        if attempt.job is None:
            continue
        if attempt.submission_token is None:
            raise WorkflowError("attempt job has no immutable submission token")
        references.append((attempt.job, attempt.submission_token))
    identities = tuple(_scheduler_job(reference) for reference, _token in references)
    if len(set(identities)) != len(identities):
        raise WorkflowError("multiple owners claim the same exact scheduler job")
    return tuple((_scheduler_job(reference), token) for reference, token in references)


def _validate_poll_interval(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("poll_interval_seconds must be numeric")
    if not _MIN_POLL_INTERVAL_SECONDS <= value <= _MAX_POLL_INTERVAL_SECONDS:
        raise ValueError(
            "poll_interval_seconds must be between "
            f"{_MIN_POLL_INTERVAL_SECONDS} and {_MAX_POLL_INTERVAL_SECONDS}"
        )


@contextmanager
def _installed_signal_handlers(
    flags: _ControllerSignalFlags,
    *,
    enabled: bool,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    handlers = (
        (signal.SIGUSR1, flags.handle_advance),
        (signal.SIGTERM, flags.handle_termination),
        (signal.SIGINT, flags.handle_termination),
    )
    previous: list[tuple[int, object]] = []
    try:
        for signum, handler in handlers:
            old_handler = signal.getsignal(signum)
            signal.signal(signum, handler)
            previous.append((signum, old_handler))
    except ValueError as error:
        for signum, handler in reversed(previous):
            signal.signal(signum, handler)  # type: ignore[arg-type]
        raise WorkflowError("controller signal handlers require the main thread") from error
    try:
        yield
    finally:
        for signum, handler in reversed(previous):
            signal.signal(signum, handler)  # type: ignore[arg-type]


def _active_attempt_jobs(state: RunState) -> tuple[JobReference, ...]:
    attempts = _active_attempts(state)
    jobs = tuple(
        attempt.job
        for attempt in attempts
        if attempt.job is not None and attempt.status not in ATTEMPT_TERMINAL_STATUSES
    )
    if len(set(jobs)) != len(jobs):
        raise WorkflowError("multiple active attempts claim the same scheduler job")
    return tuple(
        sorted(
            jobs,
            key=lambda job: (job.cluster or "", job.job_id, job.array_task_id or ""),
        )
    )


def _active_attempts(state: RunState) -> tuple[AttemptRecord, ...]:
    attempts = [*state.planning_attempts]
    attempts.extend(attempt for item in state.items for attempt in item.attempts)
    active = tuple(attempt for attempt in attempts if not attempt.terminal)
    identities = [attempt.attempt_id for attempt in active]
    if len(set(identities)) != len(identities):
        raise WorkflowError("active attempt identities are not unique")
    return tuple(sorted(active, key=lambda attempt: attempt.attempt_id))


def _controller_lease_is_live(lease_path: Path) -> bool:
    """Prove whether another process currently holds the controller lock."""
    lock_path = lease_path.with_name(f".{lease_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        lock_file.close()


@contextmanager
def _held_controller_lease_lock(lease_path: Path) -> Iterator[bool]:
    """Hold the exact controller lock across an emergency scheduler mutation."""
    lock_path = lease_path.with_name(f".{lease_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()


def _canonical_workspace(task: NormalizedTask, workspace: Path) -> Path:
    root = task.repository.workspace_root.resolve(strict=True)
    candidate = workspace.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkflowError("workspace must resolve beneath repository.workspace_root") from exc
    return candidate


def _controller_resources(task: NormalizedTask, workspace: Path) -> ResourceRequest:
    controller = task.execution.slurm.controller
    mounts = tuple(
        Mount(mount.host_path, Path(mount.container_path), mount.read_only)
        for mount in controller.mounts
    )
    return ResourceRequest(
        label="controller",
        account=controller.account,
        partition=controller.partition,
        qos=controller.qos,
        reservation=controller.reservation,
        time_limit=_slurm_time(controller.time_limit_seconds),
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=controller.cpus_per_task,
        gpus_per_node=0,
        memory_mb=controller.memory_mib,
        output_path=workspace / "slurm/controller/%j.out",
        error_path=workspace / "slurm/controller/%j.err",
        signal_seconds=controller.advance_signal_lead_seconds,
        # ``controller.requeue`` authorizes a fenced successor generation.  A
        # same-job Slurm requeue cannot advance the generation or recover the
        # durable lease safely, so automatic scheduler requeue stays disabled.
        requeue=False,
        container_image=str(controller.image),
        mounts=mounts,
    )


def _slurm_time(seconds: int) -> str:
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _state_job(identity: JobIdentity) -> JobReference:
    return JobReference(
        identity.job_id,
        cluster=identity.cluster,
        array_task_id=identity.array_task_id,
    )


def _scheduler_job(reference: JobReference) -> JobIdentity:
    return JobIdentity(
        reference.job_id,
        array_task_id=reference.array_task_id,
        cluster=reference.cluster,
    )


def _job_payload(identity: JobIdentity) -> dict[str, str | None]:
    return {
        "job_id": identity.job_id,
        "array_task_id": identity.array_task_id,
        "cluster": identity.cluster,
    }


def _status_payload(
    workspace: Path,
    state: RunState,
    scheduler: Scheduler,
) -> dict[str, object]:
    payload = _json_safe(asdict(state))
    projection = _project_status(workspace, state, scheduler)
    payload["projection"] = projection
    controller = cast(dict[str, object], projection["controller"])
    if "scheduler_observation" in controller:
        payload["scheduler_observation"] = controller["scheduler_observation"]
    return cast(dict[str, object], payload)


def _project_status(
    workspace: Path,
    state: RunState,
    scheduler: Scheduler,
) -> dict[str, object]:
    read_errors: list[dict[str, object]] = []
    drift: list[dict[str, object]] = []
    controller = _project_controller_status(state, scheduler, read_errors, drift)
    attempts: list[dict[str, object]] = []
    for attempt in sorted(state.planning_attempts, key=lambda entry: entry.attempt_id):
        attempt_dir = workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id
        attempts.append(
            _project_attempt_status(
                attempt,
                attempt_dir,
                scheduler,
                read_errors,
                drift,
                planning=True,
            )
        )
    for item in sorted(state.items, key=lambda entry: entry.item_id):
        for attempt in sorted(item.attempts, key=lambda entry: entry.sequence):
            attempt_dir = (
                workspace / "items" / item.item_id / "attempts" / f"{attempt.sequence:04d}"
            )
            attempts.append(
                _project_attempt_status(
                    attempt,
                    attempt_dir,
                    scheduler,
                    read_errors,
                    drift,
                    planning=False,
                )
            )
    return {
        "controller": controller,
        "lease": _project_lease_status(workspace, state, read_errors),
        "markers": {
            "checkpoint_count": len(state.controller_checkpoints),
            "latest_checkpoint": (
                _json_safe(asdict(state.controller_checkpoints[-1]))
                if state.controller_checkpoints
                else None
            ),
            "terminal_report": _status_file_surface(
                workspace / "reports" / "terminal-report.json",
                "terminal_report",
                read_errors,
            ),
            "terminal_markdown": _status_file_surface(
                workspace / "reports" / "terminal-report.md",
                "terminal_markdown",
                read_errors,
            ),
        },
        "launchers": _project_launcher_status(workspace, read_errors),
        "attempts": attempts,
        "drift": drift,
        "read_errors": read_errors,
    }


def _project_controller_status(
    state: RunState,
    scheduler: Scheduler,
    read_errors: list[dict[str, object]],
    drift: list[dict[str, object]],
) -> dict[str, object]:
    if state.controller_job is None:
        return {"job": None, "scheduler_status": "ABSENT", "scheduler_source": "UNKNOWN"}
    identity = _scheduler_job(state.controller_job)
    result: dict[str, object] = {"job": _job_payload(identity)}
    try:
        if state.controller_submission_token is None:
            raise WorkflowError("controller has no immutable submission token")
        observation = scheduler.observe_owned(
            identity,
            state.controller_submission_token,
        )
    except (OSError, SchedulerError, ValueError) as error:
        _append_status_error(read_errors, "controller.scheduler", error)
        result.update(
            scheduler_status="ERROR",
            scheduler_source="UNKNOWN",
            scheduler_observation=None,
        )
        return result
    result.update(
        scheduler_status=observation.status.value,
        scheduler_source=observation.source.value,
        scheduler_observation=_json_safe(asdict(observation)),
    )
    if (
        state.terminal_status is RunTerminalStatus.ACTIVE
        and observation.status.terminal
        and observation.source is ObservationSource.ACCOUNTING
    ):
        drift.append(
            {
                "surface": "controller",
                "kind": "scheduler_terminal_state_active",
                "job": identity.scheduler_id,
                "scheduler_status": observation.status.value,
            }
        )
    return result


def _project_attempt_status(
    attempt: AttemptRecord,
    attempt_dir: Path,
    scheduler: Scheduler,
    read_errors: list[dict[str, object]],
    drift: list[dict[str, object]],
    *,
    planning: bool,
) -> dict[str, object]:
    output_dir = attempt_dir / "output"
    mailbox = {
        "input": _status_file_surface(attempt_dir / "input.json", "attempt.input", read_errors),
        "complete": _status_file_surface(output_dir / "COMPLETE", "attempt.complete", read_errors),
        "result": _status_file_surface(output_dir / "result.json", "attempt.result", read_errors),
    }
    if planning:
        mailbox.update(
            {
                "role_input": _status_file_surface(
                    attempt_dir / "role-input.json", "attempt.role_input", read_errors
                ),
                "role_result": _status_file_surface(
                    output_dir / "role-result.json", "attempt.role_result", read_errors
                ),
            }
        )
    projected: dict[str, object] = {
        "item_id": attempt.item_id,
        "attempt_id": attempt.attempt_id,
        "sequence": attempt.sequence,
        "planning": planning,
        "state_status": attempt.status.value,
        "scheduler_state": attempt.scheduler_state,
        "terminal_reason": attempt.terminal_reason,
        "job": _job_payload(_scheduler_job(attempt.job)) if attempt.job is not None else None,
        "mailbox": mailbox,
    }
    if attempt.job is not None:
        identity = _scheduler_job(attempt.job)
        try:
            if attempt.submission_token is None:
                raise WorkflowError("attempt has no immutable submission token")
            observation = scheduler.observe_owned(identity, attempt.submission_token)
        except (OSError, SchedulerError, ValueError) as error:
            _append_status_error(read_errors, f"attempt.{attempt.attempt_id}.scheduler", error)
            projected.update(scheduler_status="ERROR", scheduler_source="UNKNOWN")
        else:
            projected.update(
                scheduler_status=observation.status.value,
                scheduler_source=observation.source.value,
                scheduler_observation=_json_safe(asdict(observation)),
            )
            if not attempt.terminal and observation.status.terminal:
                drift.append(
                    {
                        "surface": "attempt",
                        "kind": "scheduler_terminal_state_active",
                        "attempt_id": attempt.attempt_id,
                        "scheduler_status": observation.status.value,
                    }
                )
    complete_ready = cast(dict[str, object], mailbox["complete"])["present"]
    result_ready = cast(dict[str, object], mailbox["result"])["present"]
    if complete_ready and result_ready and attempt.result_digest is None:
        drift.append(
            {
                "surface": "attempt",
                "kind": "result_ready_unadopted",
                "attempt_id": attempt.attempt_id,
            }
        )
    return projected


def _project_lease_status(
    workspace: Path,
    state: RunState,
    read_errors: list[dict[str, object]],
) -> dict[str, object]:
    lease_path = workspace / LEASE_FILENAME
    lock_path = lease_path.with_name(f".{lease_path.name}.lock")
    lock_status = "absent"
    if lock_path.exists():
        try:
            lock_file = lock_path.open("r", encoding="utf-8")
        except OSError as error:
            _append_status_error(read_errors, "lease.lock", error)
            lock_status = "error"
        else:
            try:
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    lock_status = "held"
                else:
                    lock_status = "available"
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                lock_file.close()
    record: object | None = None
    consistent = not lease_path.exists()
    if lease_path.exists():
        try:
            loaded = load_lease_record(lease_path)
        except (OSError, TypeError, ValueError) as error:
            _append_status_error(read_errors, "lease.record", error)
            consistent = False
        else:
            record = _json_safe(asdict(loaded))
            consistent = (
                loaded.run_id == state.run_id
                and loaded.generation == state.generation
                and loaded.controller_job == state.controller_job
            )
    return {
        "record_present": lease_path.exists(),
        "lock_status": lock_status,
        "consistent": consistent,
        "record": record,
    }


def _project_launcher_status(
    workspace: Path,
    read_errors: list[dict[str, object]],
) -> dict[str, object]:
    launcher_root = workspace / LAUNCHER_DIRECTORY
    names: set[str] = {"initial"}
    truncated = False
    try:
        entries = sorted(launcher_root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        _append_status_error(read_errors, "launcher.directory", error)
        entries = []
    for entry in entries:
        for suffix in (".intent.json", ".receipt.json"):
            if entry.name.endswith(suffix):
                names.add(entry.name[: -len(suffix)])
                break
        if len(names) >= _MAX_STATUS_LAUNCHERS:
            truncated = True
            break
    launches = []
    for name in sorted(names)[:_MAX_STATUS_LAUNCHERS]:
        launches.append(
            {
                "name": name,
                "intent": _status_file_surface(
                    launcher_root / f"{name}.intent.json", "launcher.intent", read_errors
                ),
                "receipt": _status_file_surface(
                    launcher_root / f"{name}.receipt.json", "launcher.receipt", read_errors
                ),
            }
        )
    human_launches: list[dict[str, object]] = []
    human_root = workspace / HUMAN_REQUESTS_DIRECTORY
    try:
        human_entries = sorted(human_root.iterdir(), key=lambda path: path.name)
    except FileNotFoundError:
        human_entries = []
    except OSError as error:
        _append_status_error(read_errors, "launcher.human_directory", error)
        human_entries = []
    for entry in human_entries:
        suffix = ".resume.intent.json"
        if not entry.name.endswith(suffix):
            continue
        request_id = entry.name[: -len(suffix)]
        human_launches.append(
            {
                "request_id": request_id,
                "intent": _status_file_surface(entry, "launcher.human_intent", read_errors),
                "receipt": _status_file_surface(
                    human_root / f"{request_id}.resume.receipt.json",
                    "launcher.human_receipt",
                    read_errors,
                ),
            }
        )
        if len(human_launches) >= _MAX_STATUS_LAUNCHERS:
            truncated = True
            break
    return {
        "entries": launches,
        "human_entries": human_launches,
        "truncated": truncated,
    }


def _status_file_surface(
    path: Path,
    surface: str,
    read_errors: list[dict[str, object]],
) -> dict[str, object]:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return {"present": False, "regular_file": False, "symlink": False}
    except OSError as error:
        _append_status_error(read_errors, surface, error)
        return {"present": None, "regular_file": False, "symlink": False, "error": True}
    return {
        "present": True,
        "regular_file": path.is_file() and not path.is_symlink(),
        "symlink": path.is_symlink(),
        "size_bytes": stat.st_size,
    }


def _append_status_error(
    errors: list[dict[str, object]],
    surface: str,
    error: BaseException,
) -> None:
    errors.append(
        {
            "surface": surface,
            "error_type": type(error).__name__,
            "message": str(error)[:_MAX_STATUS_ERROR_CHARS],
        }
    )


def _json_safe(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    return value


def _write_status(workspace: Path, state: RunState, observation: object | None) -> None:
    content = render_status_from_values(state, observation)
    _atomic_write_text(workspace / STATUS_FILENAME, content)


def render_status_from_values(state: RunState, observation: object | None) -> str:
    """Render a stable human-readable status without querying external state."""
    lines = [
        "# Staircase status",
        "",
        f"- Run: `{state.run_id}`",
        f"- Task digest: `{state.task_digest}`",
        f"- Controller generation/revision: `{state.generation}/{state.revision}`",
        f"- Run outcome: `{state.terminal_status.value}`",
        f"- Controller bootstrap: `{state.controller_bootstrap_status.value}`",
        f"- Stages / Goals / Items: `{len(state.stages)} / {len(state.goals)} / {len(state.items)}`",
    ]
    if observation is not None:
        lines.append(f"- Scheduler observation: `{observation}`")
    if state.terminal_reason:
        lines.append(f"- Terminal reason: {state.terminal_reason}")
    return "\n".join(lines) + "\n"


def _write_once_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise WorkflowError(f"immutable request already exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise WorkflowError(f"immutable request already exists with different content: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _load_json_mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise WorkflowError(f"{label} is not a regular file: {path}")
    if path.stat().st_size > _MAX_REQUEST_BYTES:
        raise WorkflowError(f"{label} exceeds {_MAX_REQUEST_BYTES} bytes")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"{label} is not valid readable JSON: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WorkflowError(f"{label} must be a JSON object with string keys")
    return cast(dict[str, object], value)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
