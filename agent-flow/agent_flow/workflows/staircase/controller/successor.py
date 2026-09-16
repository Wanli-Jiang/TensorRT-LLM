# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Crash-safe controller successor submission and activation protocol.

The scheduler and filesystem cannot participate in one transaction.  This
module therefore exposes each durable boundary as one bounded call.  A fresh
submission is authorized only by a durable one-shot permit written when its
intent is persisted.  After that boundary, restart recovery may consume an
unclaimed permit or adopt a unique exact job by token, but it never guesses
that an ``UNKNOWN`` lookup means no job was submitted.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from ..common.slurm import (
    DependencyType,
    InternalCommand,
    InternalEntrypoint,
    JobDependency,
    JobIdentity,
    JobStatus,
    ObservationSource,
    ResourceRequest,
    Scheduler,
    SchedulerError,
)
from ..common.submission_recovery import Clock as SubmissionClock
from ..common.submission_recovery import (
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
from ..state import (
    ControllerLease,
    ControllerLifecycle,
    InvalidTransitionError,
    JobReference,
    LeaseConflictError,
    RunState,
    RunTerminalStatus,
    StateConflictError,
    advance_generation,
    load_state,
    save_state,
)


class SuccessorAction(str, Enum):
    """One durable successor boundary completed, or one blocker observed."""

    INTENT_PERSISTED = "intent_persisted"
    SUBMITTED = "submitted"
    ADOPTED = "adopted"
    FENCED = "fenced"
    WAITING_PREDECESSOR = "waiting_predecessor"
    PREDECESSOR_RECONCILED = "predecessor_reconciled"
    READY = "ready"
    WAITING_ACCOUNTING = "waiting_accounting"
    MANUAL_RECOVERY = "manual_recovery"
    BLOCKED = "blocked"


class SuccessorDisposition(str, Enum):
    """Instruction for the process driving the bounded protocol."""

    CONTINUE = "continue"
    EXIT = "exit"
    WAIT = "wait"
    READY = "ready"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SuccessorSubmissionPermit:
    """Durable one-shot proof authorizing one exact scheduler submit call."""

    predecessor_generation: int
    submission_token: str
    intent_revision: int
    permit_path: Path


@dataclass(frozen=True, slots=True)
class SuccessorTickResult:
    """Result of one bounded submission or activation protocol step."""

    state: RunState
    action: SuccessorAction
    disposition: SuccessorDisposition
    reason: str
    job: JobIdentity | None = None
    permit: SuccessorSubmissionPermit | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("successor result reason must be non-empty")
        if self.action is SuccessorAction.INTENT_PERSISTED and self.permit is None:
            raise ValueError("persisted successor intent must return its one-shot permit")
        if self.action is not SuccessorAction.INTENT_PERSISTED and self.permit is not None:
            raise ValueError("only a newly persisted intent may return a submission permit")


class SuccessorProtocolError(RuntimeError):
    """Raised when a caller tries to bypass successor activation fencing."""


def tick_successor_submission(
    *,
    state_path: Path,
    scheduler: Scheduler,
    resources: ResourceRequest,
    command: InternalCommand,
    expected_generation: int,
    reason: str,
    environment: Mapping[str, str] | None = None,
    human_request_id: str | None = None,
    permit: SuccessorSubmissionPermit | None = None,
    submission_clock: SubmissionClock | None = None,
    submission_recovery_policy: SubmissionRecoveryPolicy | None = None,
) -> SuccessorTickResult:
    """Advance one crash-safe successor-submission boundary.

    A normal caller invokes this repeatedly.  The first call persists intent,
    the second uses the returned permit to submit with ``afterany`` on the
    exact current controller job, and the third atomically installs the next
    generation and tells the predecessor to exit.
    """
    state = load_state(state_path)
    stale_exit = _already_fenced_result(state, expected_generation)
    if stale_exit is not None:
        return stale_exit
    if state.generation != expected_generation:
        return _blocked(
            state,
            "controller generation does not match the active state fence",
        )
    validation_error = _validate_submission_request(
        state_path=state_path,
        state=state,
        resources=resources,
        command=command,
        reason=reason,
        human_request_id=human_request_id,
    )
    if validation_error is not None:
        return _blocked(state, validation_error)

    successor = state.controller_successors[-1] if state.controller_successors else None
    if successor is None or successor.predecessor_generation != state.generation:
        token = _submission_token(state, reason, human_request_id)
        try:
            desired = state.begin_successor_submission(
                submission_token=token,
                reason=reason,
                human_request_id=human_request_id,
            )
            save_state(
                state_path,
                desired,
                expected_revision=state.revision,
                expected_generation=state.generation,
            )
            permit = _persist_submission_permit(state_path, desired)
        except (InvalidTransitionError, StateConflictError, ValueError) as error:
            return _blocked(state, f"successor intent could not be persisted: {error}")
        return SuccessorTickResult(
            desired,
            SuccessorAction.INTENT_PERSISTED,
            SuccessorDisposition.CONTINUE,
            "successor submission intent is durable",
            permit=permit,
        )

    if successor.reason != reason or successor.human_request_id != human_request_id:
        return _blocked(state, "pending successor intent does not match this request")
    if successor.successor_job is None:
        return _tick_unbound_successor(
            state_path=state_path,
            state=state,
            scheduler=scheduler,
            resources=resources,
            command=command,
            environment=environment,
            permit=permit,
            clock=submission_clock or (lambda: datetime.now(UTC)),
            recovery_policy=submission_recovery_policy or SubmissionRecoveryPolicy(),
        )

    try:
        desired = state.fence_to_successor()
        advance_generation(
            state_path,
            desired,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )
    except (InvalidTransitionError, StateConflictError, ValueError) as error:
        return _blocked(state, f"successor generation fence failed: {error}")
    return SuccessorTickResult(
        desired,
        SuccessorAction.FENCED,
        SuccessorDisposition.EXIT,
        "successor generation and exact controller job are durable; predecessor must exit",
        job=_scheduler_identity(desired.controller_job),
    )


def activate_successor(
    *,
    state_path: Path,
    scheduler: Scheduler,
    generation: int,
    current_job: JobIdentity,
) -> SuccessorTickResult:
    """Gate a successor until exact predecessor terminal accounting is durable."""
    state = load_state(state_path)
    current_reference = _job_reference(current_job)
    if (
        state.generation != generation
        or state.controller_job != current_reference
        or current_job.array_task_id is not None
    ):
        return _blocked(state, "successor process does not own the exact current generation/job")
    latest = state.controller_successors[-1] if state.controller_successors else None
    if (
        latest is None
        or not latest.fenced
        or latest.successor_generation != generation
        or latest.successor_job != current_reference
    ):
        return _blocked(state, "current controller is not backed by a fenced successor record")
    if state.controller_lifecycle is ControllerLifecycle.RUNNING:
        receipt = next(
            (
                entry
                for entry in state.predecessor_reconciliations
                if entry.successor_generation == generation
            ),
            None,
        )
        if (
            receipt is None
            or receipt.predecessor_job != latest.predecessor_job
            or receipt.successor_job != current_reference
            or receipt.observation_source != ObservationSource.ACCOUNTING.value
        ):
            return _blocked(state, "running successor lacks exact durable accounting receipt")
        return SuccessorTickResult(
            state,
            SuccessorAction.READY,
            SuccessorDisposition.READY,
            "exact predecessor terminal accounting receipt is durable",
            job=current_job,
        )
    if state.controller_lifecycle is not ControllerLifecycle.AWAITING_PREDECESSOR:
        return _blocked(state, "successor lifecycle is not awaiting predecessor reconciliation")

    predecessor = _scheduler_identity(latest.predecessor_job)
    predecessor_history = next(
        (
            entry
            for entry in state.controller_generation_history
            if entry.generation == latest.predecessor_generation
            and entry.job == latest.predecessor_job
        ),
        None,
    )
    if predecessor_history is None:
        return _blocked(state, "predecessor has no immutable submission-token history")
    try:
        observation = scheduler.observe_owned(
            predecessor,
            predecessor_history.submission_token,
        )
    except SchedulerError as error:
        return _blocked(state, f"predecessor scheduler observation failed: {error}")
    if observation.identity != predecessor:
        return _blocked(state, "scheduler returned a different predecessor identity")
    if observation.status is JobStatus.UNKNOWN or not observation.status.terminal:
        return SuccessorTickResult(
            state,
            SuccessorAction.WAITING_PREDECESSOR,
            SuccessorDisposition.WAIT,
            "exact predecessor is not terminal in accounting",
            job=current_job,
        )
    if observation.source is not ObservationSource.ACCOUNTING:
        return SuccessorTickResult(
            state,
            SuccessorAction.WAITING_PREDECESSOR,
            SuccessorDisposition.WAIT,
            "terminal predecessor observation is not an accounting receipt",
            job=current_job,
        )
    try:
        desired = state.record_predecessor_terminal(
            predecessor_job=latest.predecessor_job,
            scheduler_state=observation.status.value,
            observation_source=observation.source.value,
        )
        save_state(
            state_path,
            desired,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )
    except (InvalidTransitionError, StateConflictError, ValueError) as error:
        return _blocked(state, f"predecessor accounting receipt could not be persisted: {error}")
    return SuccessorTickResult(
        desired,
        SuccessorAction.PREDECESSOR_RECONCILED,
        SuccessorDisposition.CONTINUE,
        "exact predecessor terminal accounting receipt is durable",
        job=current_job,
    )


def recover_successor_lease(
    *,
    lease_path: Path,
    state_path: Path,
    generation: int,
    current_job: JobIdentity,
    owner_nonce: str,
) -> ControllerLease:
    """Recover the controller lease only after the activation receipt exists."""
    state = load_state(state_path)
    current_reference = _job_reference(current_job)
    latest = state.controller_successors[-1] if state.controller_successors else None
    receipt = next(
        (
            entry
            for entry in state.predecessor_reconciliations
            if entry.successor_generation == generation
        ),
        None,
    )
    if (
        state.terminal_status is not RunTerminalStatus.ACTIVE
        or state.generation != generation
        or state.controller_job != current_reference
        or state.controller_lifecycle is not ControllerLifecycle.RUNNING
        or latest is None
        or receipt is None
        or receipt.predecessor_job != latest.predecessor_job
        or receipt.successor_job != current_reference
        or receipt.observation_source != ObservationSource.ACCOUNTING.value
    ):
        raise SuccessorProtocolError(
            "successor lease requires an exact durable predecessor accounting receipt"
        )
    try:
        return ControllerLease.recover(
            lease_path,
            run_id=state.run_id,
            generation=generation,
            controller_job=current_reference,
            scheduler_reconciled=True,
            owner_nonce=owner_nonce,
            expected_predecessor_job=latest.predecessor_job,
        )
    except LeaseConflictError as error:
        raise SuccessorProtocolError(f"successor lease recovery failed: {error}") from error


def bootstrap_successor_process(
    *,
    state_path: Path,
    scheduler: Scheduler,
    generation: int,
    current_job: JobIdentity,
    submission_token: str | None = None,
    reason: str | None = None,
    human_request_id: str | None = None,
    response_digest: str | None = None,
) -> RunState:
    """Let an accepted successor authoritatively install its own generation.

    This closes the accepted-before-job-ID-persistence window.  The process
    proves its exact scheduler comment/token ownership before it writes state;
    login-side launchers only publish immutable mailbox envelopes.
    """
    while True:
        state = load_state(state_path)
        if state.generation == generation:
            if state.controller_job != _job_reference(current_job):
                raise SuccessorProtocolError(
                    "current successor job differs from the installed generation fence"
                )
            return state
        if state.generation + 1 != generation:
            raise SuccessorProtocolError(
                "successor process generation does not immediately follow authoritative state"
            )
        pending = state.controller_successors[-1] if state.controller_successors else None
        if pending is not None and pending.predecessor_generation == state.generation:
            token = pending.submission_token
            successor_reason = pending.reason
            request_id = pending.human_request_id
        else:
            token = submission_token
            successor_reason = reason
            request_id = human_request_id
        if token is None or successor_reason is None:
            raise SuccessorProtocolError(
                "successor self-bootstrap requires a pending intent or frozen launcher envelope"
            )
        ownership = scheduler.verify_ownership(current_job, token)
        if not ownership.matched:
            raise SuccessorProtocolError(
                f"successor scheduler ownership could not be proven: {ownership.reason}"
            )
        lookup = scheduler.lookup_submission(token)
        if len(lookup.matches) > 1:
            receipt = _quarantine_duplicate_successors(
                state_path,
                token,
                lookup.matches,
                lookup.reason or "ambiguous successor submission token",
            )
            raise SuccessorProtocolError(f"duplicate successors are quarantined in {receipt}")
        if lookup.identity != current_job:
            raise SuccessorProtocolError(
                "successor token does not resolve to this exact scheduler process"
            )

        desired = state
        if request_id is not None:
            if response_digest is None:
                raise SuccessorProtocolError("human successor lacks a frozen response digest")
            latest_request = desired.human_requests[-1] if desired.human_requests else None
            if latest_request is None or latest_request.request_id != request_id:
                raise SuccessorProtocolError("human successor request is not authoritative")
            if latest_request.response_digest is None:
                desired = desired.record_human_response(
                    request_id=request_id,
                    response_digest=response_digest,
                )
                save_state(
                    state_path,
                    desired,
                    expected_revision=state.revision,
                    expected_generation=state.generation,
                )
                continue
            if latest_request.response_digest != response_digest:
                raise SuccessorProtocolError("human successor response digest conflicts")
        if pending is None or pending.predecessor_generation != desired.generation:
            desired = desired.begin_successor_submission(
                submission_token=token,
                reason=successor_reason,
                human_request_id=request_id,
            )
            save_state(
                state_path,
                desired,
                expected_revision=state.revision,
                expected_generation=state.generation,
            )
            continue
        if pending.successor_job is None:
            desired = desired.record_successor_submitted(
                submission_token=token,
                job=_job_reference(current_job),
            )
            save_state(
                state_path,
                desired,
                expected_revision=state.revision,
                expected_generation=state.generation,
            )
            continue
        if pending.successor_job != _job_reference(current_job):
            raise SuccessorProtocolError("pending successor is bound to another exact job")
        desired = desired.fence_to_successor()
        advance_generation(
            state_path,
            desired,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )


def _tick_unbound_successor(
    *,
    state_path: Path,
    state: RunState,
    scheduler: Scheduler,
    resources: ResourceRequest,
    command: InternalCommand,
    environment: Mapping[str, str] | None,
    permit: SuccessorSubmissionPermit | None,
    clock: SubmissionClock,
    recovery_policy: SubmissionRecoveryPolicy,
) -> SuccessorTickResult:
    successor = state.controller_successors[-1]
    effective_permit = permit or _load_submission_permit(state_path, state)
    if effective_permit is None:
        # A crash may occur after the authoritative intent save but before the
        # permit file is materialized. No compliant submit can happen before
        # the permit exists, so reconstructing this still-unclaimed proof is
        # the single safe next action.
        effective_permit = _persist_submission_permit(state_path, state)
    if effective_permit is None or not _permit_matches(state, effective_permit):
        return SuccessorTickResult(
            state,
            SuccessorAction.WAITING_ACCOUNTING,
            SuccessorDisposition.WAIT,
            "successor token has no visible exact job and no live one-shot submission permit; "
            "UNKNOWN is nonterminal and submission will not be repeated",
        )
    predecessor = _scheduler_identity(successor.predecessor_job)
    dependency = JobDependency(DependencyType.AFTERANY, predecessor)
    intent_revision = str(effective_permit.intent_revision)
    intent_digest = submission_intent_digest(
        intent_revision,
        resources,
        command,
        tuple(sorted((environment or {}).items())),
        dependency,
    )
    journal_dir = (
        state_path.parent / "requests" / "successor-submissions" / successor.submission_token
    )
    with submission_recovery_lock(journal_dir):
        legacy_claim = effective_permit.permit_path.with_suffix(".claimed.json")
        if legacy_claim.exists():
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=successor.submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason="legacy successor claim has no trusted timestamp",
                clock=clock,
            )
            return _manual_recovery(state, "legacy successor claim requires reconciliation")
        try:
            probe = scheduler.probe_submission(successor.submission_token)
        except SchedulerError as error:
            decision = record_probe_error(
                journal_dir=journal_dir,
                submission_token=successor.submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=str(error),
                clock=clock,
                policy=recovery_policy,
            )
            if decision.action is SubmissionRecoveryAction.MANUAL_RECOVERY:
                return _manual_recovery(state, decision.reason)
            return SuccessorTickResult(
                state,
                SuccessorAction.WAITING_ACCOUNTING,
                SuccessorDisposition.WAIT,
                decision.reason,
            )
        if len(probe.matches) > 1:
            receipt = _quarantine_duplicate_successors(
                state_path,
                successor.submission_token,
                probe.matches,
                probe.reason or "ambiguous successor submission token",
            )
            return _blocked(
                state,
                f"successor token is ambiguous across {len(probe.matches)} exact jobs; "
                f"quarantined in {receipt}",
            )
        if probe.matches and probe.identity is None:
            reason = probe.reason or "successor scheduler evidence is untrustworthy"
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=successor.submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=reason,
                clock=clock,
            )
            return _manual_recovery(state, reason)
        if probe.identity is not None:
            identity_error = _validate_successor_identity(successor.predecessor_job, probe.identity)
            if identity_error is not None:
                return _manual_recovery(state, identity_error)
            ownership = scheduler.verify_ownership(probe.identity, successor.submission_token)
            if not ownership.matched:
                record_manual_recovery(
                    journal_dir=journal_dir,
                    submission_token=successor.submission_token,
                    intent_revision=intent_revision,
                    intent_digest=intent_digest,
                    reason=f"successor ownership is ambiguous: {ownership.reason}",
                    clock=clock,
                )
                return _manual_recovery(state, ownership.reason)
            try:
                desired = state.record_successor_submitted(
                    submission_token=successor.submission_token,
                    job=_job_reference(probe.identity),
                )
                save_state(
                    state_path,
                    desired,
                    expected_revision=state.revision,
                    expected_generation=state.generation,
                )
            except (InvalidTransitionError, StateConflictError, ValueError) as error:
                return _blocked(
                    state, f"adopted successor identity could not be persisted: {error}"
                )
            return SuccessorTickResult(
                desired,
                SuccessorAction.ADOPTED,
                SuccessorDisposition.CONTINUE,
                "unique exact successor job was adopted by immutable token",
                job=probe.identity,
            )
        try:
            decision = decide_submission_recovery(
                journal_dir=journal_dir,
                submission_token=successor.submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                probe=probe,
                clock=clock,
                policy=recovery_policy,
            )
        except SubmissionRecoveryError as error:
            return _manual_recovery(state, f"invalid recovery journal: {error}")
        if decision.action is SubmissionRecoveryAction.MANUAL_RECOVERY:
            return _manual_recovery(state, decision.reason)
        if decision.action is SubmissionRecoveryAction.WAIT:
            return SuccessorTickResult(
                state,
                SuccessorAction.WAITING_ACCOUNTING,
                SuccessorDisposition.WAIT,
                decision.reason,
            )
        claim_sequence = decision.claim_sequence
        if claim_sequence is None:
            return _manual_recovery(state, "submission recovery omitted its claim sequence")
        try:
            identity = scheduler.submit(
                resources,
                command,
                successor.submission_token,
                environment,
                dependency,
            )
        except SchedulerError as error:
            record_submission_outcome(
                journal_dir=journal_dir,
                claim_sequence=claim_sequence,
                submission_token=successor.submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                outcome="uncertain",
                clock=clock,
            )
            return SuccessorTickResult(
                state,
                SuccessorAction.WAITING_ACCOUNTING,
                SuccessorDisposition.WAIT,
                "successor submit returned without an identity; bounded reconciliation "
                f"is required: {error}",
            )
        record_submission_outcome(
            journal_dir=journal_dir,
            claim_sequence=claim_sequence,
            submission_token=successor.submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
            outcome="accepted",
            clock=clock,
            scheduler_id=identity.scheduler_id,
        )
        identity_error = _validate_successor_identity(successor.predecessor_job, identity)
        if identity_error is not None:
            return _manual_recovery(
                state, f"scheduler returned an unsafe successor identity: {identity_error}"
            )
        ownership = scheduler.verify_ownership(identity, successor.submission_token)
        if not ownership.matched:
            record_manual_recovery(
                journal_dir=journal_dir,
                submission_token=successor.submission_token,
                intent_revision=intent_revision,
                intent_digest=intent_digest,
                reason=f"submitted successor ownership is ambiguous: {ownership.reason}",
                clock=clock,
            )
            return _manual_recovery(state, ownership.reason)
        try:
            desired = state.record_successor_submitted(
                submission_token=successor.submission_token,
                job=_job_reference(identity),
            )
            save_state(
                state_path,
                desired,
                expected_revision=state.revision,
                expected_generation=state.generation,
            )
        except (InvalidTransitionError, StateConflictError, ValueError) as error:
            return _blocked(
                state,
                "successor was accepted and must be adopted by token after persistence conflict: "
                f"{error}",
            )
        return SuccessorTickResult(
            desired,
            SuccessorAction.SUBMITTED,
            SuccessorDisposition.CONTINUE,
            "successor job and exact afterany predecessor dependency are durable",
            job=identity,
        )


def _validate_submission_request(
    *,
    state_path: Path,
    state: RunState,
    resources: ResourceRequest,
    command: InternalCommand,
    reason: str,
    human_request_id: str | None,
) -> str | None:
    if state.terminal_status is not RunTerminalStatus.ACTIVE:
        return "terminal runs cannot submit controller successors"
    if not reason.strip():
        return "successor reason must be non-empty"
    if resources.requeue:
        return "successor jobs must disable Slurm automatic requeue"
    if command.entrypoint is not InternalEntrypoint.CONTROLLER:
        return "successor command must use the fixed controller entrypoint"
    if command.generation != state.generation + 1:
        return "successor command generation must immediately follow the predecessor"
    if command.workspace != state_path.parent:
        return "successor command workspace must own the authoritative state path"
    if state.controller_job is None or state.controller_job.array_task_id is not None:
        return "successor submission requires an exact non-array current controller job"
    if human_request_id is not None and not human_request_id.strip():
        return "human request identity must be non-empty"
    return None


def _validate_successor_identity(
    predecessor: JobReference,
    successor: JobIdentity,
) -> str | None:
    if successor.array_task_id is not None:
        return "controller successor cannot be an array element"
    if successor.cluster != predecessor.cluster:
        return "successor cluster does not exactly match predecessor cluster"
    if _job_reference(successor) == predecessor:
        return "successor scheduler identity must differ from predecessor"
    return None


def _permit_matches(
    state: RunState,
    permit: SuccessorSubmissionPermit | None,
) -> bool:
    successor = state.controller_successors[-1]
    return (
        permit is not None
        and permit.predecessor_generation == state.generation
        and permit.submission_token == successor.submission_token
        and permit.intent_revision == state.revision
    )


def _submission_permit_path(
    state_path: Path,
    generation: int,
    submission_token: str,
) -> Path:
    return (
        state_path.parent
        / "requests"
        / "successor-permits"
        / f"g{generation}-{submission_token}.permit.json"
    )


def _persist_submission_permit(
    state_path: Path,
    state: RunState,
) -> SuccessorSubmissionPermit:
    successor = state.controller_successors[-1]
    permit_path = _submission_permit_path(
        state_path,
        successor.predecessor_generation,
        successor.submission_token,
    )
    permit = SuccessorSubmissionPermit(
        predecessor_generation=successor.predecessor_generation,
        submission_token=successor.submission_token,
        intent_revision=state.revision,
        permit_path=permit_path,
    )
    payload = _permit_payload(permit, status="armed")
    _write_once_json(permit_path, payload)
    return permit


def _load_submission_permit(
    state_path: Path,
    state: RunState,
) -> SuccessorSubmissionPermit | None:
    successor = state.controller_successors[-1]
    permit_path = _submission_permit_path(
        state_path,
        successor.predecessor_generation,
        successor.submission_token,
    )
    if not permit_path.is_file():
        return None
    try:
        raw = json.loads(permit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SuccessorProtocolError(f"successor permit is unreadable: {error}") from error
    expected = {
        "schema_version": 1,
        "predecessor_generation": successor.predecessor_generation,
        "submission_token": successor.submission_token,
        "intent_revision": state.revision,
        "status": "armed",
    }
    if raw != expected:
        raise SuccessorProtocolError("successor permit differs from the durable intent")
    return SuccessorSubmissionPermit(
        successor.predecessor_generation,
        successor.submission_token,
        state.revision,
        permit_path,
    )


def _permit_payload(
    permit: SuccessorSubmissionPermit,
    *,
    status: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "predecessor_generation": permit.predecessor_generation,
        "submission_token": permit.submission_token,
        "intent_revision": permit.intent_revision,
        "status": status,
    }


def _quarantine_duplicate_successors(
    state_path: Path,
    submission_token: str,
    matches: tuple[JobIdentity, ...],
    reason: str,
) -> Path:
    destination = (
        state_path.parent / "quarantine" / "submissions" / f"successor-{submission_token}.json"
    )
    payload = {
        "schema_version": 1,
        "kind": "controller_successor",
        "submission_token": submission_token,
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
    _write_once_json(destination, payload)
    return destination


def _write_once_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    fail_if_exists: bool = False,
) -> None:
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if fail_if_exists:
            raise
        if path.read_text(encoding="utf-8") != encoded:
            raise SuccessorProtocolError(f"immutable recovery receipt conflicts: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _already_fenced_result(
    state: RunState,
    expected_generation: int,
) -> SuccessorTickResult | None:
    if state.generation != expected_generation + 1 or not state.controller_successors:
        return None
    latest = state.controller_successors[-1]
    if not latest.fenced or latest.predecessor_generation != expected_generation:
        return None
    return SuccessorTickResult(
        state,
        SuccessorAction.FENCED,
        SuccessorDisposition.EXIT,
        "this predecessor generation is already fenced and must exit",
        job=_scheduler_identity(latest.successor_job),
    )


def _submission_token(
    state: RunState,
    reason: str,
    human_request_id: str | None,
) -> str:
    material = "\0".join(
        (
            state.run_id,
            str(state.generation),
            state.controller_job.scheduler_id if state.controller_job is not None else "",
            state.controller_job.cluster or "" if state.controller_job is not None else "",
            reason,
            human_request_id or "",
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]
    return f"staircase-successor-{digest}"


def _scheduler_identity(reference: JobReference | None) -> JobIdentity:
    if reference is None:
        raise SuccessorProtocolError("exact controller scheduler identity is missing")
    return JobIdentity(
        reference.job_id,
        array_task_id=reference.array_task_id,
        cluster=reference.cluster,
    )


def _job_reference(identity: JobIdentity) -> JobReference:
    return JobReference(
        identity.job_id,
        array_task_id=identity.array_task_id,
        cluster=identity.cluster,
    )


def _blocked(state: RunState, reason: str) -> SuccessorTickResult:
    return SuccessorTickResult(
        state,
        SuccessorAction.BLOCKED,
        SuccessorDisposition.BLOCKED,
        reason,
    )


def _manual_recovery(state: RunState, reason: str) -> SuccessorTickResult:
    return SuccessorTickResult(
        state,
        SuccessorAction.MANUAL_RECOVERY,
        SuccessorDisposition.BLOCKED,
        f"manual scheduler reconciliation required: {reason}",
    )


__all__ = [
    "SuccessorAction",
    "SuccessorDisposition",
    "SuccessorProtocolError",
    "SuccessorSubmissionPermit",
    "SuccessorTickResult",
    "activate_successor",
    "bootstrap_successor_process",
    "recover_successor_lease",
    "tick_successor_submission",
]
