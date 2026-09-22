# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure controller signal, generation, orphan, and human-input recovery policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialBroker,
    CredentialDescriptor,
    CredentialError,
    CredentialRevocationCause,
    CredentialRevocationReceipt,
    revoke_generation_fenced_provisions,
)
from agent_flow.workflows.staircase.common.slurm import JobIdentity, JobObservation, JobStatus
from agent_flow.workflows.staircase.state import (
    PLANNING_ITEM_ID,
    AttemptRecord,
    AttemptStatus,
    ControllerLifecycle,
    JobReference,
    RunState,
)


class ControllerSignal(str, Enum):
    """Controller signals that require a durable scheduling checkpoint."""

    ADVANCE = "advance"
    PREEMPTION = "preemption"


@dataclass(frozen=True, slots=True)
class SignalRecoveryIntent:
    """Side-effect-free intent produced for an advance or preemption signal."""

    signal: ControllerSignal
    stop_dispatch: bool
    checkpoint_required: bool
    leave_workers_running: bool
    controller_requeue: bool
    worker_requeue: bool


def signal_recovery_intent(
    signal: ControllerSignal,
    *,
    controller_requeue_enabled: bool,
) -> SignalRecoveryIntent:
    """Return the mandatory checkpoint behavior for a controller signal.

    Existing independent workers remain owned and running. A later controller
    generation reconciles them; this policy never requests automatic worker
    requeue.
    """
    if not isinstance(controller_requeue_enabled, bool):
        raise TypeError("controller_requeue_enabled must be a bool")
    return SignalRecoveryIntent(
        signal=signal,
        stop_dispatch=True,
        checkpoint_required=True,
        leave_workers_running=True,
        controller_requeue=controller_requeue_enabled,
        worker_requeue=False,
    )


@dataclass(frozen=True, slots=True)
class GenerationAdvanceDecision:
    """Whether exact predecessor reconciliation permits generation advance."""

    allowed: bool
    predecessor: JobReference
    observed_status: JobStatus | None
    reason: str

    def require_allowed(self) -> None:
        """Raise when the predecessor is not exactly and terminally reconciled."""
        if not self.allowed:
            raise RecoveryError(self.reason)


class RecoveryError(RuntimeError):
    """Raised when a caller attempts an unsafe controller recovery action."""


def revoke_reconciled_generation_credentials(
    state: RunState,
    *,
    broker: CredentialBroker,
    workspace: Path,
    descriptors: Sequence[CredentialDescriptor],
    adopted_bindings: Sequence[CredentialBinding],
) -> tuple[CredentialRevocationReceipt, ...]:
    """Fence every old-generation credential after exact reconciliation.

    The caller reconstructs ``descriptors`` only from immutable attempt inputs
    and supplies exact bindings for the attempts that remained live.  This
    boundary deliberately refuses to call the broker until every active
    old-generation attempt has a durable successor-adoption receipt.  It also
    verifies each returned typed receipt against its durable public bytes.
    """
    plan_attempt_adoption(state).require_new_work_allowed()
    attempts = {
        attempt.attempt_id: attempt
        for attempt in (
            *state.planning_attempts,
            *(attempt for item in state.items for attempt in item.attempts),
        )
    }
    adopted_attempt_ids = {
        record.attempt_id
        for record in state.attempt_adoptions
        if record.generation == state.generation
    }
    for descriptor in descriptors:
        if not isinstance(descriptor, CredentialDescriptor):
            raise RecoveryError("credential recovery requires typed descriptors")
        binding = descriptor.binding
        attempt = attempts.get(binding.attempt_id)
        if attempt is None:
            raise RecoveryError("credential descriptor references an unknown attempt")
        expected_item_id = "planning" if attempt.item_id == PLANNING_ITEM_ID else attempt.item_id
        if (
            binding.run_id != state.run_id
            or binding.task_digest != state.task_digest
            or binding.generation != attempt.generation
            or binding.generation >= state.generation
            or binding.item_id != expected_item_id
            or binding.attempt_id not in adopted_attempt_ids
        ):
            raise RecoveryError(
                "credential descriptor is not bound to an exactly reconciled old-generation attempt"
            )
    descriptor_bindings = {descriptor.binding for descriptor in descriptors}
    for binding in adopted_bindings:
        if binding not in descriptor_bindings:
            raise RecoveryError("adopted credential binding has no reconstructed descriptor")

    try:
        receipts = revoke_generation_fenced_provisions(
            broker=broker,
            workspace=workspace,
            current_generation=state.generation,
            descriptors=descriptors,
            adopted_bindings=adopted_bindings,
        )
    except CredentialError as error:
        raise RecoveryError(f"generation credential fence failed: {error}") from error

    expected = {descriptor.binding: descriptor for descriptor in descriptors}
    observed: set[CredentialBinding] = set()
    for receipt in receipts:
        if not isinstance(receipt, CredentialRevocationReceipt):
            raise RecoveryError("credential broker returned an untyped revocation receipt")
        binding = receipt.descriptor.binding
        if (
            binding in observed
            or expected.get(binding) != receipt.descriptor
            or receipt.cause is not CredentialRevocationCause.GENERATION_FENCE
        ):
            raise RecoveryError("credential broker returned a mismatched revocation receipt")
        observed.add(binding)
        _validate_durable_credential_receipt(receipt)
    if observed != set(expected):
        raise RecoveryError("credential broker omitted a generation-fence receipt")
    return receipts


@dataclass(frozen=True, slots=True)
class AttemptAdoptionTarget:
    """Exact persisted identity that one successor must adopt."""

    attempt_id: str
    attempt_generation: int
    status: AttemptStatus
    submission_token: str | None
    job: JobReference | None


@dataclass(frozen=True, slots=True)
class AttemptAdoptionPlan:
    """Pure successor plan that gates dispatch until adoption is complete."""

    generation: int
    pending: tuple[AttemptAdoptionTarget, ...]
    adopted_attempt_ids: tuple[str, ...]
    predecessor_reconciled: bool

    @property
    def new_work_allowed(self) -> bool:
        """Whether the successor may dispatch new attempts."""
        return self.predecessor_reconciled and not self.pending

    def require_new_work_allowed(self) -> None:
        """Fail closed when dispatch would race predecessor/adoption recovery."""
        if not self.predecessor_reconciled:
            raise RecoveryError("new work is gated until predecessor accounting is reconciled")
        if self.pending:
            raise RecoveryError("new work is gated until all old-generation attempts are adopted")


def plan_attempt_adoption(state: RunState) -> AttemptAdoptionPlan:
    """Plan exact adoption of every active old-generation attempt.

    PREPARED intents, SUBMITTING intents with a durable token, and active
    known-job attempts are all carried forward.  The function has no scheduler
    or state-writing side effects; callers append each exact receipt through
    :meth:`RunState.record_attempt_adoption` and re-plan.
    """
    predecessor_reconciled = state.controller_lifecycle is ControllerLifecycle.RUNNING
    attempts = (
        *state.planning_attempts,
        *(attempt for item in state.items for attempt in item.attempts),
    )
    active_old = tuple(
        sorted(
            (
                attempt
                for attempt in attempts
                if attempt.generation < state.generation and not attempt.terminal
            ),
            key=lambda attempt: (
                attempt.generation,
                attempt.item_id,
                attempt.sequence,
                attempt.attempt_id,
            ),
        )
    )
    for attempt in active_old:
        _validate_adoptable_attempt(attempt)
    adopted_ids = tuple(
        record.attempt_id
        for record in state.attempt_adoptions
        if record.generation == state.generation
    )
    adopted = set(adopted_ids)
    pending = (
        tuple(
            _attempt_adoption_target(attempt)
            for attempt in active_old
            if attempt.attempt_id not in adopted
        )
        if predecessor_reconciled
        else ()
    )
    return AttemptAdoptionPlan(
        generation=state.generation,
        pending=pending,
        adopted_attempt_ids=adopted_ids,
        predecessor_reconciled=predecessor_reconciled,
    )


def decide_generation_advance(
    predecessor: JobReference,
    observation: JobObservation | None,
) -> GenerationAdvanceDecision:
    """Require exact terminal scheduler evidence before advancing generation."""
    if observation is None:
        return GenerationAdvanceDecision(
            False,
            predecessor,
            None,
            "generation advance requires an exact predecessor scheduler observation",
        )
    expected = _scheduler_identity(predecessor)
    if observation.identity != expected:
        return GenerationAdvanceDecision(
            False,
            predecessor,
            observation.status,
            "generation advance observation does not match the exact predecessor identity",
        )
    if not observation.status.terminal:
        return GenerationAdvanceDecision(
            False,
            predecessor,
            observation.status,
            "generation advance requires a terminal predecessor observation; "
            f"observed {observation.status.value}",
        )
    return GenerationAdvanceDecision(
        True,
        predecessor,
        observation.status,
        f"exact predecessor reconciled as {observation.status.value}",
    )


class OrphanAction(str, Enum):
    """Controller action for exact owned children after predecessor death."""

    ADOPT = "adopt"
    WAIT_FOR_SUCCESSOR = "wait_for_successor"
    CLEAN_UP = "clean_up"


@dataclass(frozen=True, slots=True)
class OrphanRecoveryPlan:
    """Bounded orphan-grace decision for exact child scheduler identities."""

    action: OrphanAction
    adopt: tuple[JobIdentity, ...]
    cancel: tuple[JobIdentity, ...]
    terminal: tuple[JobIdentity, ...]
    grace_remaining_seconds: int


def plan_orphan_recovery(
    observations: tuple[JobObservation, ...],
    *,
    successor_present: bool,
    elapsed_seconds: int,
    grace_seconds: int,
) -> OrphanRecoveryPlan:
    """Plan adoption or exact cleanup after a bounded orphan grace period.

    Args:
        observations: Scheduler observations for exact, durably owned child IDs.
        successor_present: Whether a fenced successor controller is active.
        elapsed_seconds: Time since predecessor death was durably established.
        grace_seconds: Configured orphan-adoption grace period.

    Returns:
        A pure plan. Callers remain responsible for exact scheduler operations.
    """
    _require_nonnegative_int(elapsed_seconds, "elapsed_seconds")
    _require_nonnegative_int(grace_seconds, "grace_seconds")
    if not isinstance(successor_present, bool):
        raise TypeError("successor_present must be a bool")
    identities = tuple(observation.identity for observation in observations)
    if len(set(identities)) != len(identities):
        raise ValueError("orphan observations contain duplicate exact job identities")
    ordered = tuple(sorted(observations, key=lambda value: value.identity.scheduler_id))
    terminal = tuple(observation.identity for observation in ordered if observation.status.terminal)
    active = tuple(
        observation.identity for observation in ordered if not observation.status.terminal
    )
    if successor_present:
        return OrphanRecoveryPlan(OrphanAction.ADOPT, active, (), terminal, 0)
    remaining = max(0, grace_seconds - elapsed_seconds)
    if remaining:
        return OrphanRecoveryPlan(
            OrphanAction.WAIT_FOR_SUCCESSOR,
            (),
            (),
            terminal,
            remaining,
        )
    return OrphanRecoveryPlan(OrphanAction.CLEAN_UP, (), active, terminal, 0)


class HumanInputStatus(str, Enum):
    """Durable controller state required by a detached human request."""

    WAITING_FOR_INPUT = "WAITING_FOR_INPUT"


@dataclass(frozen=True, slots=True)
class HumanInputRecoveryIntent:
    """Checkpoint and requeue intent for detached human interaction."""

    request_id: str
    prompt_digest: str
    status: HumanInputStatus
    checkpoint_required: bool
    release_controller: bool
    controller_requeue: bool
    read_stdin: bool
    worker_requeue: bool


def human_input_recovery_intent(
    request_id: str,
    prompt_digest: str,
    *,
    controller_requeue_enabled: bool,
) -> HumanInputRecoveryIntent:
    """Return the durable, detached response policy for one human request."""
    if not request_id.strip():
        raise ValueError("request_id must not be empty")
    if len(prompt_digest) != 64 or any(
        character not in "0123456789abcdef" for character in prompt_digest
    ):
        raise ValueError("prompt_digest must be a lowercase SHA-256 digest")
    if not isinstance(controller_requeue_enabled, bool):
        raise TypeError("controller_requeue_enabled must be a bool")
    return HumanInputRecoveryIntent(
        request_id=request_id,
        prompt_digest=prompt_digest,
        status=HumanInputStatus.WAITING_FOR_INPUT,
        checkpoint_required=True,
        release_controller=True,
        controller_requeue=controller_requeue_enabled,
        read_stdin=False,
        worker_requeue=False,
    )


def _scheduler_identity(reference: JobReference) -> JobIdentity:
    return JobIdentity(
        job_id=reference.job_id,
        array_task_id=reference.array_task_id,
        cluster=reference.cluster,
    )


def _attempt_adoption_target(attempt: AttemptRecord) -> AttemptAdoptionTarget:
    return AttemptAdoptionTarget(
        attempt_id=attempt.attempt_id,
        attempt_generation=attempt.generation,
        status=attempt.status,
        submission_token=attempt.submission_token,
        job=attempt.job,
    )


def _validate_adoptable_attempt(attempt: AttemptRecord) -> None:
    if attempt.status is AttemptStatus.PREPARED:
        if attempt.submission_token is not None or attempt.job is not None:
            raise RecoveryError(
                "old-generation PREPARED attempt has unexpected submission identity"
            )
        return
    if attempt.status is AttemptStatus.SUBMITTING:
        if attempt.submission_token is None or attempt.job is not None:
            raise RecoveryError("old-generation SUBMITTING attempt lacks an exact durable token")
        return
    if attempt.submission_token is None or attempt.job is None:
        raise RecoveryError(
            "old-generation active submitted attempt lacks exact token/job identity"
        )


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_durable_credential_receipt(receipt: CredentialRevocationReceipt) -> None:
    path = receipt.receipt_path
    if path.is_symlink() or not path.is_file():
        raise RecoveryError("credential revocation receipt is not a durable regular file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise RecoveryError("credential revocation receipt is unreadable") from error
    if len(raw) > 1_048_576:
        raise RecoveryError("credential revocation receipt exceeds the size limit")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryError("credential revocation receipt is not strict JSON") from error
    if value != receipt.to_public_dict():
        raise RecoveryError("credential revocation receipt bytes do not match the typed receipt")
