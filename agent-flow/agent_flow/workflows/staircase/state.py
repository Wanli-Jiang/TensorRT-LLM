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

"""Durable, generation-fenced state for the Staircase controller."""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Iterator, TextIO

STATE_FILENAME = "state.json"
LEASE_FILENAME = "controller-lease.json"
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
LEASE_SCHEMA_VERSION = 1
_RESOURCE_CLASS_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


class StateError(RuntimeError):
    """Base class for invalid or conflicting authoritative state operations."""


class StateConflictError(StateError):
    """Raised when a revision or controller generation compare-and-swap fails."""


class StateMigrationError(StateError):
    """Raised when persisted state cannot be migrated without inventing evidence."""


class InvalidTransitionError(StateError):
    """Raised when a domain object is moved through an illegal transition."""


class LeaseConflictError(StateError):
    """Raised when another controller owns or left an unreconciled lease."""


class Role(str, Enum):
    """Isolated agent or deterministic gate role for an attempt."""

    PLAN_DRAFTER = "plan_drafter"
    PLAN_REVIEWER = "plan_reviewer"
    CODER = "coder"
    GATE = "gate"
    REVIEWER = "reviewer"
    QA = "qa"


class DomainProfile(str, Enum):
    """Staircase domain profile applied to a generic role."""

    SMITH = "smith"
    ASSEMBLER = "assembler"
    TUNER = "tuner"


class WorkItemKind(str, Enum):
    """Atomic work kinds understood by the deterministic controller."""

    FEASIBILITY = "feasibility"
    SEARCH = "search"
    CATALOG_VERIFY = "catalog_verify"
    CATALOG_ONBOARD = "catalog_onboard"
    ASSEMBLE_CORE = "assemble_core"
    ASSEMBLE_FEATURE = "assemble_feature"
    ROUTING = "routing"
    TUNE_HYPOTHESIS = "tune_hypothesis"
    GATE = "gate"


class WorkItemStatus(str, Enum):
    """Domain lifecycle, independent of scheduler state."""

    PLANNED = "planned"
    READY = "ready"
    CODING = "coding"
    CANDIDATE_READY = "candidate_ready"
    REVIEWING = "reviewing"
    APPROVED = "approved"
    INTEGRATING = "integrating"
    INTEGRATED = "integrated"
    BLOCKED = "blocked"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class AttemptKind(str, Enum):
    """Execution purpose of one immutable scheduler attempt."""

    ROLE = "role"
    DETERMINISTIC_GATE = "deterministic_gate"
    REVIEWER_ANALYSIS = "reviewer_analysis"
    REVIEWER_RERUN = "reviewer_rerun"
    QA = "qa"


class AttemptStatus(str, Enum):
    """Scheduler-attempt lifecycle, independent of domain approval."""

    PREPARED = "prepared"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    PENDING = "pending"
    RUNNING = "running"
    TERMINAL_OBSERVED = "terminal_observed"
    COLLECTING = "collecting"
    VALIDATED = "validated"
    RETRYABLE_FAILED = "retryable_failed"
    FAILED = "failed"
    PREEMPTED = "preempted"
    CANCELLED = "cancelled"
    LOST = "lost"


class RunTerminalStatus(str, Enum):
    """Terminal outcome of the complete run."""

    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    BLOCKED_INPUT = "blocked_input"
    EXHAUSTED = "exhausted"
    INTERRUPTED = "interrupted"
    CANCELLED = "cancelled"
    FAILED = "failed"


class WorkflowMode(str, Enum):
    """Top-level Staircase workflow selected when the run is created."""

    ONBOARD = "onboard"
    TUNE = "tune"


class HierarchyStatus(str, Enum):
    """Lifecycle of a typed Goal."""

    PLANNED = "planned"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    """Lifecycle of an independently QA-ed Stage."""

    PLANNED = "planned"
    ACTIVE = "active"
    READY_FOR_QA = "ready_for_qa"
    QA_PASSED = "qa_passed"
    QA_FAILED = "qa_failed"
    CLOSED = "closed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class LegacySucceededStagePolicy(str, Enum):
    """Explicit policy for unverifiable schema-v1 succeeded Stages."""

    REJECT = "reject"
    BLOCK = "block"


class ControllerBootstrapStatus(str, Enum):
    """Crash-recoverable login-side controller submission phase."""

    PREPARED = "prepared"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"


class ControllerLifecycle(str, Enum):
    """Nonterminal lifecycle of the currently fenced controller generation."""

    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    WAITING_FOR_INPUT = "waiting_for_input"
    AWAITING_PREDECESSOR = "awaiting_predecessor"


class ControllerCheckpointSignal(str, Enum):
    """Slurm signal that caused a durable controller checkpoint."""

    ADVANCE = "advance"
    PREEMPTION = "preemption"


class OrphanReceiptAction(str, Enum):
    """Exact controller action recorded for one owned orphan job."""

    ADOPTED = "adopted"
    CLEANED_UP = "cleaned_up"


class IntegrationEvidenceKind(str, Enum):
    """Typed evidence that authorizes one controller-owned integration."""

    CANDIDATE = "candidate"
    VERIFICATION = "verification"


WORK_ITEM_TERMINAL_STATUSES = frozenset(
    {
        WorkItemStatus.INTEGRATED,
        WorkItemStatus.BLOCKED,
        WorkItemStatus.REJECTED,
        WorkItemStatus.CANCELLED,
    }
)
ATTEMPT_TERMINAL_STATUSES = frozenset(
    {
        AttemptStatus.VALIDATED,
        AttemptStatus.RETRYABLE_FAILED,
        AttemptStatus.FAILED,
        AttemptStatus.PREEMPTED,
        AttemptStatus.CANCELLED,
        AttemptStatus.LOST,
    }
)
HIERARCHY_TERMINAL_STATUSES = frozenset(
    {HierarchyStatus.SUCCEEDED, HierarchyStatus.BLOCKED, HierarchyStatus.CANCELLED}
)
STAGE_TERMINAL_STATUSES = frozenset(
    {StageStatus.CLOSED, StageStatus.BLOCKED, StageStatus.CANCELLED}
)
PLANNING_ITEM_ID = "__planning__"

_WORK_ITEM_TRANSITIONS: dict[WorkItemStatus, frozenset[WorkItemStatus]] = {
    WorkItemStatus.PLANNED: frozenset(
        {WorkItemStatus.READY, WorkItemStatus.BLOCKED, WorkItemStatus.CANCELLED}
    ),
    WorkItemStatus.READY: frozenset(
        {WorkItemStatus.CODING, WorkItemStatus.BLOCKED, WorkItemStatus.CANCELLED}
    ),
    WorkItemStatus.CODING: frozenset(
        {
            WorkItemStatus.CANDIDATE_READY,
            WorkItemStatus.BLOCKED,
            WorkItemStatus.REJECTED,
            WorkItemStatus.CANCELLED,
        }
    ),
    WorkItemStatus.CANDIDATE_READY: frozenset(
        {WorkItemStatus.REVIEWING, WorkItemStatus.REJECTED, WorkItemStatus.CANCELLED}
    ),
    WorkItemStatus.REVIEWING: frozenset(
        {
            WorkItemStatus.APPROVED,
            WorkItemStatus.REJECTED,
            WorkItemStatus.BLOCKED,
            WorkItemStatus.CANCELLED,
        }
    ),
    WorkItemStatus.APPROVED: frozenset(
        {
            WorkItemStatus.INTEGRATING,
            WorkItemStatus.REJECTED,
            WorkItemStatus.BLOCKED,
            WorkItemStatus.CANCELLED,
        }
    ),
    WorkItemStatus.INTEGRATING: frozenset(
        {WorkItemStatus.INTEGRATED, WorkItemStatus.BLOCKED, WorkItemStatus.CANCELLED}
    ),
    **{status: frozenset() for status in WORK_ITEM_TERMINAL_STATUSES},
}

_ATTEMPT_TRANSITIONS: dict[AttemptStatus, frozenset[AttemptStatus]] = {
    AttemptStatus.PREPARED: frozenset({AttemptStatus.SUBMITTING, AttemptStatus.CANCELLED}),
    AttemptStatus.SUBMITTING: frozenset(
        {
            AttemptStatus.SUBMITTED,
            AttemptStatus.RETRYABLE_FAILED,
            AttemptStatus.FAILED,
            AttemptStatus.CANCELLED,
            AttemptStatus.LOST,
        }
    ),
    AttemptStatus.SUBMITTED: frozenset(
        {
            AttemptStatus.PENDING,
            AttemptStatus.RUNNING,
            AttemptStatus.TERMINAL_OBSERVED,
            AttemptStatus.CANCELLED,
            AttemptStatus.LOST,
        }
    ),
    AttemptStatus.PENDING: frozenset(
        {
            AttemptStatus.RUNNING,
            AttemptStatus.TERMINAL_OBSERVED,
            AttemptStatus.PREEMPTED,
            AttemptStatus.CANCELLED,
            AttemptStatus.LOST,
        }
    ),
    AttemptStatus.RUNNING: frozenset(
        {
            AttemptStatus.TERMINAL_OBSERVED,
            AttemptStatus.PREEMPTED,
            AttemptStatus.CANCELLED,
            AttemptStatus.LOST,
        }
    ),
    AttemptStatus.TERMINAL_OBSERVED: frozenset(
        {
            AttemptStatus.COLLECTING,
            AttemptStatus.RETRYABLE_FAILED,
            AttemptStatus.FAILED,
            AttemptStatus.PREEMPTED,
            AttemptStatus.CANCELLED,
            AttemptStatus.LOST,
        }
    ),
    AttemptStatus.COLLECTING: frozenset(
        {
            AttemptStatus.VALIDATED,
            AttemptStatus.RETRYABLE_FAILED,
            AttemptStatus.FAILED,
            AttemptStatus.CANCELLED,
            AttemptStatus.LOST,
        }
    ),
    **{status: frozenset() for status in ATTEMPT_TERMINAL_STATUSES},
}

_HIERARCHY_TRANSITIONS: dict[HierarchyStatus, frozenset[HierarchyStatus]] = {
    HierarchyStatus.PLANNED: frozenset(
        {HierarchyStatus.ACTIVE, HierarchyStatus.BLOCKED, HierarchyStatus.CANCELLED}
    ),
    HierarchyStatus.ACTIVE: frozenset(
        {HierarchyStatus.SUCCEEDED, HierarchyStatus.BLOCKED, HierarchyStatus.CANCELLED}
    ),
    **{status: frozenset() for status in HIERARCHY_TERMINAL_STATUSES},
}

_STAGE_TRANSITIONS: dict[StageStatus, frozenset[StageStatus]] = {
    StageStatus.PLANNED: frozenset(
        {StageStatus.ACTIVE, StageStatus.BLOCKED, StageStatus.CANCELLED}
    ),
    StageStatus.ACTIVE: frozenset(
        {StageStatus.READY_FOR_QA, StageStatus.BLOCKED, StageStatus.CANCELLED}
    ),
    StageStatus.READY_FOR_QA: frozenset(
        {
            StageStatus.QA_PASSED,
            StageStatus.QA_FAILED,
            StageStatus.BLOCKED,
            StageStatus.CANCELLED,
        }
    ),
    StageStatus.QA_PASSED: frozenset({StageStatus.CLOSED}),
    StageStatus.QA_FAILED: frozenset(
        {StageStatus.ACTIVE, StageStatus.BLOCKED, StageStatus.CANCELLED}
    ),
    **{status: frozenset() for status in STAGE_TERMINAL_STATUSES},
}

_CONTROLLER_LIFECYCLE_TRANSITIONS: dict[ControllerLifecycle, frozenset[ControllerLifecycle]] = {
    ControllerLifecycle.RUNNING: frozenset(
        {ControllerLifecycle.CHECKPOINTED, ControllerLifecycle.WAITING_FOR_INPUT}
    ),
    ControllerLifecycle.CHECKPOINTED: frozenset(),
    ControllerLifecycle.WAITING_FOR_INPUT: frozenset(),
    ControllerLifecycle.AWAITING_PREDECESSOR: frozenset({ControllerLifecycle.RUNNING}),
}

_TERMINAL_SCHEDULER_STATES = frozenset(
    {
        "COMPLETED",
        "CANCELLED",
        "FAILED",
        "PREEMPTED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "TIMEOUT",
    }
)


@dataclass(frozen=True)
class JobReference:
    """Persisted scheduler identity without scheduler behavior."""

    job_id: str
    array_task_id: str | None = None
    cluster: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("job_id", self.job_id)
        if not self.job_id.isdecimal() or int(self.job_id) < 1:
            raise ValueError("job_id must be a positive decimal Slurm ID")
        if self.array_task_id is not None and (
            not self.array_task_id.isdecimal() or int(self.array_task_id) < 0
        ):
            raise ValueError("array_task_id must be a non-negative decimal Slurm ID")
        if self.cluster is not None:
            _require_identifier("cluster", self.cluster)

    @property
    def scheduler_id(self) -> str:
        """Return the exact Slurm job or array-element identity."""
        if self.array_task_id is None:
            return self.job_id
        return f"{self.job_id}_{self.array_task_id}"


@dataclass(frozen=True)
class ControllerGenerationRecord:
    """Append-only exact scheduler identity for one controller generation."""

    generation: int
    submission_token: str
    job: JobReference

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("controller history generation must be positive")
        _require_optional_text("controller history submission_token", self.submission_token)
        if self.job.array_task_id is not None:
            raise ValueError("controller history cannot reference an array element")


@dataclass(frozen=True)
class ControllerSuccessorRecord:
    """Crash-recoverable successor intent and exact adopted job identity."""

    sequence: int
    predecessor_generation: int
    predecessor_job: JobReference
    successor_generation: int
    submission_token: str
    reason: str
    human_request_id: str | None = None
    successor_job: JobReference | None = None
    fenced: bool = False

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("successor sequence must be positive")
        if self.predecessor_generation < 1:
            raise ValueError("successor predecessor generation must be positive")
        if self.successor_generation != self.predecessor_generation + 1:
            raise ValueError("successor generation must immediately follow its predecessor")
        if self.predecessor_job.array_task_id is not None:
            raise ValueError("controller predecessor cannot be an array element")
        _require_optional_text("successor submission_token", self.submission_token)
        _require_optional_text("successor reason", self.reason)
        if self.human_request_id is not None:
            _require_identifier("successor human_request_id", self.human_request_id)
        if self.successor_job is not None:
            if self.successor_job.array_task_id is not None:
                raise ValueError("controller successor cannot be an array element")
            if self.successor_job.cluster != self.predecessor_job.cluster:
                raise ValueError("successor and predecessor clusters must match exactly")
            if self.successor_job == self.predecessor_job:
                raise ValueError("successor must have a distinct exact scheduler identity")
        if self.fenced and self.successor_job is None:
            raise ValueError("a fenced successor requires an exact scheduler identity")

    def record_job(self, job: JobReference) -> ControllerSuccessorRecord:
        """Bind the immutable successor token to one exact scheduler identity."""
        if self.successor_job is not None:
            if self.successor_job != job:
                raise InvalidTransitionError("successor job identity is immutable once recorded")
            return self
        return replace(self, successor_job=job)

    def fence(self) -> ControllerSuccessorRecord:
        """Mark the record as the atomically installed current generation."""
        if self.successor_job is None:
            raise InvalidTransitionError("cannot fence a successor without an exact job")
        if self.fenced:
            return self
        return replace(self, fenced=True)


@dataclass(frozen=True)
class ControllerCheckpointRecord:
    """Append-only checkpoint evidence for one controller generation."""

    sequence: int
    generation: int
    reason: str
    requeue_requested: bool
    signal: ControllerCheckpointSignal | None = None
    human_request_id: str | None = None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("controller checkpoint sequence must be positive")
        if self.generation < 1:
            raise ValueError("controller checkpoint generation must be positive")
        _require_optional_text("checkpoint reason", self.reason)
        _require_bool(self.requeue_requested, "requeue_requested")
        if self.human_request_id is not None:
            _require_identifier("human_request_id", self.human_request_id)
        if self.signal is not None and self.human_request_id is not None:
            raise ValueError("a signal checkpoint cannot also be a human-input checkpoint")


@dataclass(frozen=True)
class HumanRequestRecord:
    """Immutable detached human request with an optional frozen response."""

    request_id: str
    generation: int
    prompt_digest: str
    controller_requeue: bool
    response_digest: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("request_id", self.request_id)
        if self.generation < 1:
            raise ValueError("human request generation must be positive")
        _require_digest("prompt_digest", self.prompt_digest)
        _require_bool(self.controller_requeue, "controller_requeue")
        if self.response_digest is not None:
            _require_digest("response_digest", self.response_digest)

    def record_response(self, response_digest: str) -> HumanRequestRecord:
        """Freeze a response digest without changing the request identity."""
        if self.response_digest is not None:
            raise InvalidTransitionError("human request already has a frozen response")
        return replace(self, response_digest=response_digest)


@dataclass(frozen=True)
class PredecessorReconciliationReceipt:
    """Exact terminal reconciliation used to fence one predecessor generation."""

    predecessor_generation: int
    successor_generation: int
    predecessor_job: JobReference
    scheduler_state: str
    successor_job: JobReference | None = None
    observation_source: str = "ACCOUNTING"

    def __post_init__(self) -> None:
        if self.predecessor_generation < 1:
            raise ValueError("predecessor generation must be positive")
        if self.successor_generation != self.predecessor_generation + 1:
            raise ValueError("successor generation must immediately follow predecessor generation")
        _require_optional_text("predecessor scheduler state", self.scheduler_state)
        if self.scheduler_state not in _TERMINAL_SCHEDULER_STATES:
            raise ValueError("predecessor receipt requires a terminal scheduler state")
        if self.observation_source != "ACCOUNTING":
            raise ValueError("predecessor receipt requires terminal accounting evidence")
        if self.successor_job is not None:
            if self.successor_job.array_task_id is not None:
                raise ValueError("successor receipt cannot reference an array element")
            if self.successor_job.cluster != self.predecessor_job.cluster:
                raise ValueError("receipt predecessor and successor clusters must match")


@dataclass(frozen=True)
class OrphanGraceWindow:
    """Caller-timestamped bounded interval for adopting exact orphan jobs."""

    generation: int
    started_at: str
    deadline_at: str

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("orphan grace generation must be positive")
        started = _parse_utc_timestamp(self.started_at, "orphan grace started_at")
        deadline = _parse_utc_timestamp(self.deadline_at, "orphan grace deadline_at")
        if deadline <= started:
            raise ValueError("orphan grace deadline_at must be later than started_at")


@dataclass(frozen=True)
class OrphanActionReceipt:
    """Append-only receipt for exact adoption or cleanup of an owned child job."""

    generation: int
    action: OrphanReceiptAction
    job: JobReference
    scheduler_state: str

    def __post_init__(self) -> None:
        if self.generation < 1:
            raise ValueError("orphan receipt generation must be positive")
        _require_optional_text("orphan scheduler state", self.scheduler_state)


@dataclass(frozen=True)
class IntegrationRecord:
    """One append-only stable fan-in step in the integration commit chain."""

    sequence: int
    generation: int
    item_id: str
    previous_commit: str
    new_commit: str
    evidence_kind: IntegrationEvidenceKind
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("integration sequence must be positive")
        if self.generation < 1:
            raise ValueError("integration generation must be positive")
        _require_identifier("integration item_id", self.item_id)
        _require_commit("integration previous_commit", self.previous_commit)
        _require_commit("integration new_commit", self.new_commit)
        if not isinstance(self.evidence_kind, IntegrationEvidenceKind):
            raise TypeError("integration evidence_kind must use IntegrationEvidenceKind")
        _require_digest("integration evidence_digest", self.evidence_digest)


@dataclass(frozen=True)
class AttemptRecord:
    """One immutable-identity scheduler attempt for a work item."""

    attempt_id: str
    item_id: str
    sequence: int
    role: Role
    kind: AttemptKind
    generation: int
    status: AttemptStatus = AttemptStatus.PREPARED
    profile: DomainProfile | None = None
    submission_token: str | None = None
    job: JobReference | None = None
    scheduler_state: str | None = None
    result_digest: str | None = None
    candidate_digest: str | None = None
    review_of_attempt_id: str | None = None
    reviewed_candidate_digest: str | None = None
    terminal_reason: str | None = None
    resource_class: str | None = None
    predecessor_attempt_id: str | None = None
    resource_escalation_request_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("attempt_id", self.attempt_id)
        _require_identifier("item_id", self.item_id)
        if self.sequence < 1:
            raise ValueError("attempt sequence must be positive")
        if self.generation < 1:
            raise ValueError("attempt generation must be positive")
        if self.resource_class is not None:
            _require_resource_class(self.resource_class)
        if self.predecessor_attempt_id is not None:
            _require_identifier("predecessor_attempt_id", self.predecessor_attempt_id)
            if self.predecessor_attempt_id == self.attempt_id:
                raise ValueError("attempt cannot be its own predecessor")
        if self.resource_escalation_request_id is not None:
            _require_identifier(
                "resource_escalation_request_id",
                self.resource_escalation_request_id,
            )
            if self.predecessor_attempt_id is None:
                raise ValueError("attempt resource escalation requires a predecessor")
        if self.status not in {AttemptStatus.PREPARED, AttemptStatus.CANCELLED}:
            _require_optional_text("submission_token", self.submission_token)
        if (
            self.status
            in {
                AttemptStatus.SUBMITTED,
                AttemptStatus.PENDING,
                AttemptStatus.RUNNING,
                AttemptStatus.TERMINAL_OBSERVED,
                AttemptStatus.COLLECTING,
                AttemptStatus.VALIDATED,
                AttemptStatus.PREEMPTED,
            }
            and self.job is None
        ):
            raise ValueError(f"attempt in {self.status.value} must record a scheduler job")
        if self.status is AttemptStatus.VALIDATED:
            _require_digest("result_digest", self.result_digest)
        is_review = self.kind in {
            AttemptKind.REVIEWER_ANALYSIS,
            AttemptKind.REVIEWER_RERUN,
        }
        if is_review:
            _require_optional_text("review_of_attempt_id", self.review_of_attempt_id)
            _require_digest("reviewed_candidate_digest", self.reviewed_candidate_digest)
        elif self.review_of_attempt_id is not None or self.reviewed_candidate_digest is not None:
            raise ValueError("only reviewer attempts may carry frozen reviewer linkage")
        if self.candidate_digest is not None:
            _require_digest("candidate_digest", self.candidate_digest)
        if self.status in ATTEMPT_TERMINAL_STATUSES and self.status is not AttemptStatus.VALIDATED:
            _require_optional_text("terminal_reason", self.terminal_reason)

    @property
    def terminal(self) -> bool:
        """Whether this attempt has reached an immutable terminal scheduler outcome."""
        return self.status in ATTEMPT_TERMINAL_STATUSES

    def transition(self, status: AttemptStatus, **changes: object) -> AttemptRecord:
        """Return an attempt advanced through one legal scheduler transition.

        Args:
            status: Desired next scheduler-attempt status.
            **changes: Fields learned at the transition, such as ``job`` or
                ``result_digest``.

        Returns:
            A validated replacement record.
        """
        _check_transition("attempt", self.status, status, _ATTEMPT_TRANSITIONS)
        return replace(self, status=status, **changes)


@dataclass(frozen=True)
class AttemptAdoptionRecord:
    """Append-only proof that a successor resolved one old-generation attempt.

    The recorded status is the exact predecessor state.  A matching attempt may
    subsequently be terminally fenced in the same state revision when its
    orphan grace expires before scheduler submission.
    """

    sequence: int
    generation: int
    attempt_id: str
    attempt_generation: int
    status: AttemptStatus
    submission_token: str | None
    job: JobReference | None

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("attempt adoption sequence must be positive")
        if self.generation < 1:
            raise ValueError("attempt adoption generation must be positive")
        _require_identifier("attempt adoption attempt_id", self.attempt_id)
        if self.attempt_generation < 1:
            raise ValueError("attempt adoption attempt_generation must be positive")
        if self.attempt_generation >= self.generation:
            raise ValueError("attempt adoption must reference an older generation")
        if self.status in ATTEMPT_TERMINAL_STATUSES:
            raise ValueError("terminal attempts cannot be adopted")
        if self.status is AttemptStatus.PREPARED:
            if self.submission_token is not None or self.job is not None:
                raise ValueError("PREPARED adoption cannot contain submission identity")
            return
        _require_optional_text("attempt adoption submission_token", self.submission_token)
        if self.status is AttemptStatus.SUBMITTING:
            if self.job is not None:
                raise ValueError("SUBMITTING adoption cannot contain a scheduler job")
            return
        if self.job is None:
            raise ValueError("active submitted attempt adoption requires a scheduler job")


@dataclass(frozen=True)
class WorkItemRecord:
    """One atomic domain item and all of its historical attempts."""

    item_id: str
    stage_id: str
    goal_id: str
    kind: WorkItemKind
    profile: DomainProfile
    status: WorkItemStatus = WorkItemStatus.PLANNED
    dependencies: tuple[str, ...] = ()
    attempts: tuple[AttemptRecord, ...] = ()
    candidate_attempt_id: str | None = None
    candidate_digest: str | None = None
    reviewer_attempt_id: str | None = None
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("item_id", self.item_id)
        _require_identifier("stage_id", self.stage_id)
        _require_identifier("goal_id", self.goal_id)
        if self.item_id in self.dependencies:
            raise ValueError(f"work item {self.item_id!r} cannot depend on itself")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError(f"work item {self.item_id!r} has duplicate dependencies")
        attempt_ids = [attempt.attempt_id for attempt in self.attempts]
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError(f"work item {self.item_id!r} has duplicate attempt IDs")
        sequences = [attempt.sequence for attempt in self.attempts]
        if len(set(sequences)) != len(sequences):
            raise ValueError(f"work item {self.item_id!r} has duplicate attempt sequences")
        if sequences != list(range(1, len(sequences) + 1)):
            raise ValueError(f"work item {self.item_id!r} attempts must be in contiguous order")
        if any(attempt.item_id != self.item_id for attempt in self.attempts):
            raise ValueError(f"work item {self.item_id!r} contains an attempt for another item")
        _validate_attempt_lineage(self.item_id, self.attempts)
        if self.candidate_attempt_id is not None:
            _require_digest("candidate_digest", self.candidate_digest)
            candidate = self._attempt(self.candidate_attempt_id)
            if candidate.role is not Role.CODER or candidate.status is not AttemptStatus.VALIDATED:
                raise ValueError("candidate attempt must be a validated Coder attempt")
            if candidate.candidate_digest != self.candidate_digest:
                raise ValueError("candidate digest does not match frozen candidate digest")
        elif self.candidate_digest is not None:
            raise ValueError("candidate_digest requires candidate_attempt_id")
        if (
            self.status
            in {
                WorkItemStatus.CANDIDATE_READY,
                WorkItemStatus.REVIEWING,
                WorkItemStatus.APPROVED,
                WorkItemStatus.INTEGRATING,
                WorkItemStatus.INTEGRATED,
            }
            and self.candidate_attempt_id is None
        ):
            raise ValueError(f"work item in {self.status.value} must carry a frozen candidate")
        if self.reviewer_attempt_id is not None:
            reviewer = self._attempt(self.reviewer_attempt_id)
            if reviewer.kind not in {
                AttemptKind.REVIEWER_ANALYSIS,
                AttemptKind.REVIEWER_RERUN,
            }:
                raise ValueError("reviewer_attempt_id must identify a reviewer attempt")
            if reviewer.review_of_attempt_id != self.candidate_attempt_id:
                raise ValueError("reviewer is not linked to the frozen candidate attempt")
            if reviewer.reviewed_candidate_digest != self.candidate_digest:
                raise ValueError("reviewer did not inspect the frozen candidate digest")
        if (
            self.status
            in {
                WorkItemStatus.APPROVED,
                WorkItemStatus.INTEGRATING,
                WorkItemStatus.INTEGRATED,
            }
            and self.reviewer_attempt_id is None
        ):
            raise ValueError(f"work item in {self.status.value} must carry reviewer linkage")
        if (
            self.status
            in {
                WorkItemStatus.APPROVED,
                WorkItemStatus.INTEGRATING,
                WorkItemStatus.INTEGRATED,
            }
            and self._attempt(self.reviewer_attempt_id or "").status is not AttemptStatus.VALIDATED
        ):
            raise ValueError("approved work item must carry a validated reviewer attempt")
        if (
            self.status in WORK_ITEM_TERMINAL_STATUSES
            and self.status is not WorkItemStatus.INTEGRATED
        ):
            _require_optional_text("terminal_reason", self.terminal_reason)

    def _attempt(self, attempt_id: str) -> AttemptRecord:
        for attempt in self.attempts:
            if attempt.attempt_id == attempt_id:
                return attempt
        raise ValueError(f"unknown attempt {attempt_id!r} in work item {self.item_id!r}")

    @property
    def terminal(self) -> bool:
        """Whether this domain item has reached a terminal outcome."""
        return self.status in WORK_ITEM_TERMINAL_STATUSES

    def transition(self, status: WorkItemStatus, **changes: object) -> WorkItemRecord:
        """Return an item advanced through one legal domain transition.

        Args:
            status: Desired next domain status.
            **changes: Fields learned at the transition.

        Returns:
            A validated replacement record.
        """
        _check_transition("work item", self.status, status, _WORK_ITEM_TRANSITIONS)
        return replace(self, status=status, **changes)

    def add_attempt(self, attempt: AttemptRecord) -> WorkItemRecord:
        """Append a new attempt without replacing historical evidence.

        Args:
            attempt: Newly prepared attempt with a unique identity and sequence.

        Returns:
            A validated replacement item.
        """
        if attempt.item_id != self.item_id:
            raise ValueError("attempt item identity does not match its work item")
        expected_sequence = max((entry.sequence for entry in self.attempts), default=0) + 1
        if attempt.sequence != expected_sequence:
            raise ValueError(
                f"attempt sequence {attempt.sequence} does not follow {expected_sequence - 1}"
            )
        return replace(self, attempts=(*self.attempts, attempt))

    def replace_attempt(self, attempt: AttemptRecord) -> WorkItemRecord:
        """Replace the current record for an existing immutable attempt identity.

        Args:
            attempt: Updated lifecycle record.

        Returns:
            A validated replacement item.
        """
        existing = self._attempt(attempt.attempt_id)
        _validate_attempt_update(existing, attempt)
        updated = tuple(
            attempt if entry.attempt_id == attempt.attempt_id else entry for entry in self.attempts
        )
        return replace(self, attempts=updated)


@dataclass(frozen=True)
class GoalRecord:
    """Typed module/capability goal containing atomic work items."""

    goal_id: str
    stage_id: str
    required_item_ids: tuple[str, ...]
    status: HierarchyStatus = HierarchyStatus.PLANNED
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("goal_id", self.goal_id)
        _require_identifier("stage_id", self.stage_id)
        if not self.required_item_ids:
            raise ValueError("goal must name at least one required work item")
        if len(set(self.required_item_ids)) != len(self.required_item_ids):
            raise ValueError(f"goal {self.goal_id!r} has duplicate required work items")
        if self.status in {HierarchyStatus.BLOCKED, HierarchyStatus.CANCELLED}:
            _require_optional_text("terminal_reason", self.terminal_reason)
        elif self.terminal_reason is not None:
            raise ValueError("non-failed goal cannot have a terminal reason")

    @property
    def terminal(self) -> bool:
        """Whether this Goal is closed."""
        return self.status in HIERARCHY_TERMINAL_STATUSES

    def transition(self, status: HierarchyStatus, **changes: object) -> GoalRecord:
        """Return a Goal advanced through one legal lifecycle transition."""
        _check_transition("Goal", self.status, status, _HIERARCHY_TRANSITIONS)
        return replace(self, status=status, **changes)


@dataclass(frozen=True)
class StageRecord:
    """Typed independently QA-able bring-up milestone."""

    stage_id: str
    required_goal_ids: tuple[str, ...]
    status: StageStatus = StageStatus.PLANNED
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        _require_identifier("stage_id", self.stage_id)
        if not isinstance(self.status, StageStatus):
            raise TypeError("stage status must be a StageStatus")
        if not self.required_goal_ids:
            raise ValueError("stage must name at least one required Goal")
        if len(set(self.required_goal_ids)) != len(self.required_goal_ids):
            raise ValueError(f"stage {self.stage_id!r} has duplicate required Goals")
        if self.status in {StageStatus.BLOCKED, StageStatus.CANCELLED}:
            _require_optional_text("terminal_reason", self.terminal_reason)
        elif self.terminal_reason is not None:
            raise ValueError("non-failed stage cannot have a terminal reason")

    @property
    def terminal(self) -> bool:
        """Whether this Stage is closed."""
        return self.status in STAGE_TERMINAL_STATUSES

    def transition(self, status: StageStatus, **changes: object) -> StageRecord:
        """Return a Stage advanced through one legal lifecycle transition."""
        _check_transition("Stage", self.status, status, _STAGE_TRANSITIONS)
        return replace(self, status=status, **changes)


@dataclass(frozen=True)
class RunState:
    """Authoritative Staircase state written only by the active controller."""

    run_id: str
    task_digest: str
    base_commit: str
    generation: int
    workflow_mode: WorkflowMode = WorkflowMode.ONBOARD
    revision: int = 0
    stages: tuple[StageRecord, ...] = ()
    goals: tuple[GoalRecord, ...] = ()
    items: tuple[WorkItemRecord, ...] = ()
    planning_attempts: tuple[AttemptRecord, ...] = ()
    integration_history: tuple[IntegrationRecord, ...] = ()
    controller_bootstrap_status: ControllerBootstrapStatus = ControllerBootstrapStatus.PREPARED
    controller_submission_token: str | None = None
    controller_job: JobReference | None = None
    controller_lifecycle: ControllerLifecycle = ControllerLifecycle.RUNNING
    controller_checkpoints: tuple[ControllerCheckpointRecord, ...] = ()
    human_requests: tuple[HumanRequestRecord, ...] = ()
    controller_generation_history: tuple[ControllerGenerationRecord, ...] = ()
    controller_successors: tuple[ControllerSuccessorRecord, ...] = ()
    predecessor_reconciliations: tuple[PredecessorReconciliationReceipt, ...] = ()
    orphan_grace_windows: tuple[OrphanGraceWindow, ...] = ()
    orphan_action_receipts: tuple[OrphanActionReceipt, ...] = ()
    attempt_adoptions: tuple[AttemptAdoptionRecord, ...] = ()
    terminal_status: RunTerminalStatus = RunTerminalStatus.ACTIVE
    terminal_reason: str | None = None
    schema_version: int = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_identifier("run_id", self.run_id)
        _require_digest("task_digest", self.task_digest)
        _require_optional_text("base_commit", self.base_commit)
        if self.generation < 1:
            raise ValueError("controller generation must be positive")
        if self.revision < 0:
            raise ValueError("state revision must be non-negative")
        item_ids = [item.item_id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("run state has duplicate work-item IDs")
        attempt_ids = [attempt.attempt_id for item in self.items for attempt in item.attempts]
        if len(set(attempt_ids)) != len(attempt_ids):
            raise ValueError("run state has duplicate attempt IDs")
        known_ids = set(item_ids)
        items_by_id = {item.item_id: item for item in self.items}
        _validate_dependency_graph(self.items)
        goal_ids = [goal.goal_id for goal in self.goals]
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(set(goal_ids)) != len(goal_ids):
            raise ValueError("run state has duplicate Goal IDs")
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("run state has duplicate Stage IDs")
        goals_by_id = {goal.goal_id: goal for goal in self.goals}
        stages_by_id = {stage.stage_id: stage for stage in self.stages}
        if self.items and (not self.goals or not self.stages):
            raise ValueError("work items require typed Goal and Stage records")
        for goal in self.goals:
            if goal.stage_id not in stages_by_id:
                raise ValueError(f"goal {goal.goal_id!r} references an unknown Stage")
            for item_id in goal.required_item_ids:
                if item_id not in items_by_id:
                    raise ValueError(f"goal {goal.goal_id!r} references unknown item {item_id!r}")
                item = items_by_id[item_id]
                if item.goal_id != goal.goal_id or item.stage_id != goal.stage_id:
                    raise ValueError(f"item {item_id!r} does not belong to Goal {goal.goal_id!r}")
            required_items = tuple(items_by_id[item_id] for item_id in goal.required_item_ids)
            if goal.status is HierarchyStatus.SUCCEEDED and any(
                item.status is not WorkItemStatus.INTEGRATED for item in required_items
            ):
                raise ValueError(
                    f"Goal {goal.goal_id!r} succeeded before required items integrated"
                )
            if goal.status in {HierarchyStatus.BLOCKED, HierarchyStatus.CANCELLED} and any(
                not item.terminal for item in required_items
            ):
                raise ValueError(
                    f"Goal {goal.goal_id!r} closed before required items became terminal"
                )
        for stage in self.stages:
            for goal_id in stage.required_goal_ids:
                if goal_id not in goals_by_id:
                    raise ValueError(
                        f"Stage {stage.stage_id!r} references unknown Goal {goal_id!r}"
                    )
                if goals_by_id[goal_id].stage_id != stage.stage_id:
                    raise ValueError(
                        f"Goal {goal_id!r} does not belong to Stage {stage.stage_id!r}"
                    )
            required_goals = tuple(goals_by_id[goal_id] for goal_id in stage.required_goal_ids)
            if stage.status in {
                StageStatus.READY_FOR_QA,
                StageStatus.QA_PASSED,
                StageStatus.QA_FAILED,
                StageStatus.CLOSED,
            } and any(goal.status is not HierarchyStatus.SUCCEEDED for goal in required_goals):
                raise ValueError(
                    f"Stage {stage.stage_id!r} reached QA before required Goals succeeded"
                )
            if stage.status in {StageStatus.BLOCKED, StageStatus.CANCELLED} and any(
                not goal.terminal for goal in required_goals
            ):
                raise ValueError(
                    f"Stage {stage.stage_id!r} closed before required Goals became terminal"
                )
        for goal in self.goals:
            if goal.goal_id not in stages_by_id[goal.stage_id].required_goal_ids:
                raise ValueError(f"Goal {goal.goal_id!r} is not owned by its Stage")
        for item in self.items:
            if item.goal_id not in goals_by_id or item.stage_id not in stages_by_id:
                raise ValueError(f"work item {item.item_id!r} references an unknown Goal or Stage")
            if goals_by_id[item.goal_id].stage_id != item.stage_id:
                raise ValueError(
                    f"work item {item.item_id!r} has inconsistent Goal/Stage ownership"
                )
            if item.item_id not in goals_by_id[item.goal_id].required_item_ids:
                raise ValueError(f"work item {item.item_id!r} is not owned by its Goal")
            unknown_dependencies = set(item.dependencies) - known_ids
            if unknown_dependencies:
                raise ValueError(
                    f"work item {item.item_id!r} has unknown dependencies: "
                    f"{sorted(unknown_dependencies)!r}"
                )
            if any(attempt.generation > self.generation for attempt in item.attempts):
                raise ValueError("attempt generation cannot be newer than controller generation")
            if item.status in {
                WorkItemStatus.READY,
                WorkItemStatus.CODING,
                WorkItemStatus.CANDIDATE_READY,
                WorkItemStatus.REVIEWING,
                WorkItemStatus.APPROVED,
                WorkItemStatus.INTEGRATING,
                WorkItemStatus.INTEGRATED,
            } and any(
                items_by_id[dependency].status is not WorkItemStatus.INTEGRATED
                for dependency in item.dependencies
            ):
                raise ValueError(
                    f"work item {item.item_id!r} advanced before all dependencies were integrated"
                )
        planning_ids = [attempt.attempt_id for attempt in self.planning_attempts]
        if len(set(planning_ids)) != len(planning_ids):
            raise ValueError("run state has duplicate planning attempt IDs")
        if set(planning_ids) & set(attempt_ids):
            raise ValueError("planning and work-item attempt IDs must be globally unique")
        planning_sequences = [attempt.sequence for attempt in self.planning_attempts]
        if planning_sequences != list(range(1, len(planning_sequences) + 1)):
            raise ValueError("planning attempts must be in contiguous sequence order")
        for attempt in self.planning_attempts:
            if attempt.item_id != PLANNING_ITEM_ID:
                raise ValueError("planning attempts must use the typed planning item identity")
            if attempt.role not in {Role.PLAN_DRAFTER, Role.PLAN_REVIEWER}:
                raise ValueError("planning attempts must use PlanDrafter or PlanReviewer role")
            if attempt.generation > self.generation:
                raise ValueError("planning attempt generation cannot exceed controller generation")
        integration_sequences = [record.sequence for record in self.integration_history]
        if integration_sequences != list(range(1, len(integration_sequences) + 1)):
            raise ValueError("integration history must be in contiguous sequence order")
        integration_item_ids = [record.item_id for record in self.integration_history]
        if len(set(integration_item_ids)) != len(integration_item_ids):
            raise ValueError("each work item may be integrated only once")
        expected_integration_head = self.base_commit
        for record in self.integration_history:
            if record.generation > self.generation:
                raise ValueError("integration generation cannot exceed controller generation")
            if record.item_id not in items_by_id:
                raise ValueError(f"integration history references unknown item {record.item_id!r}")
            if record.previous_commit != expected_integration_head:
                raise ValueError("integration history is not a linear commit chain")
            item = items_by_id[record.item_id]
            if item.status not in {
                WorkItemStatus.APPROVED,
                WorkItemStatus.INTEGRATING,
                WorkItemStatus.INTEGRATED,
            }:
                raise ValueError("integration history requires an approved work item")
            if item.kind is WorkItemKind.CATALOG_VERIFY:
                if record.evidence_kind is not IntegrationEvidenceKind.VERIFICATION:
                    raise ValueError("CATALOG_VERIFY requires typed verification evidence")
                if record.new_commit != record.previous_commit:
                    raise ValueError("CATALOG_VERIFY must record a no-change integration")
            else:
                if record.evidence_kind is not IntegrationEvidenceKind.CANDIDATE:
                    raise ValueError("file-modifying integration requires candidate evidence")
                if record.new_commit == record.previous_commit:
                    raise ValueError("file-modifying integration must advance the commit")
                if record.evidence_digest != item.candidate_digest:
                    raise ValueError(
                        "integration candidate digest differs from the frozen candidate"
                    )
            expected_integration_head = record.new_commit
        if self.controller_bootstrap_status is ControllerBootstrapStatus.PREPARED:
            if self.controller_submission_token is not None or self.controller_job is not None:
                raise ValueError("prepared controller bootstrap cannot carry token or job identity")
        elif self.controller_bootstrap_status is ControllerBootstrapStatus.SUBMITTING:
            _require_optional_text("controller_submission_token", self.controller_submission_token)
            if self.controller_job is not None:
                raise ValueError("submitting controller cannot have a persisted job identity yet")
        else:
            _require_optional_text("controller_submission_token", self.controller_submission_token)
            if self.controller_job is None:
                raise ValueError("submitted controller must carry an exact job identity")
        checkpoint_sequences = [checkpoint.sequence for checkpoint in self.controller_checkpoints]
        if checkpoint_sequences != list(range(1, len(checkpoint_sequences) + 1)):
            raise ValueError("controller checkpoints must be in contiguous sequence order")
        if any(
            checkpoint.generation > self.generation for checkpoint in self.controller_checkpoints
        ):
            raise ValueError("controller checkpoint generation cannot exceed current generation")
        request_ids = [request.request_id for request in self.human_requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("run state has duplicate human request IDs")
        if any(request.generation > self.generation for request in self.human_requests):
            raise ValueError("human request generation cannot exceed current generation")
        request_checkpoints = {
            checkpoint.human_request_id: checkpoint
            for checkpoint in self.controller_checkpoints
            if checkpoint.human_request_id is not None
        }
        if len(request_checkpoints) != sum(
            checkpoint.human_request_id is not None for checkpoint in self.controller_checkpoints
        ):
            raise ValueError("each human request may have only one controller checkpoint")
        for request in self.human_requests:
            checkpoint = request_checkpoints.get(request.request_id)
            if checkpoint is None or checkpoint.generation != request.generation:
                raise ValueError(
                    f"human request {request.request_id!r} requires a same-generation checkpoint"
                )
            if checkpoint.requeue_requested != request.controller_requeue:
                raise ValueError("human request and checkpoint requeue intent must match")
        unknown_checkpoint_requests = set(request_checkpoints) - set(request_ids)
        if unknown_checkpoint_requests:
            raise ValueError(
                "controller checkpoints reference unknown human requests: "
                f"{sorted(unknown_checkpoint_requests)!r}"
            )
        unresolved_requests = [
            request for request in self.human_requests if request.response_digest is None
        ]
        if len(unresolved_requests) > 1:
            raise ValueError("only one human request may await a response")
        if self.controller_lifecycle is ControllerLifecycle.RUNNING and unresolved_requests:
            raise ValueError("a running controller cannot have an unresolved human request")
        if (
            unresolved_requests
            and self.controller_lifecycle is not ControllerLifecycle.WAITING_FOR_INPUT
        ):
            raise ValueError("an unresolved human request requires WAITING_FOR_INPUT")
        if self.controller_lifecycle is ControllerLifecycle.WAITING_FOR_INPUT:
            if self.terminal_status is not RunTerminalStatus.ACTIVE:
                raise ValueError("WAITING_FOR_INPUT is recoverable and requires an active run")
            if not self.human_requests or not self.controller_checkpoints:
                raise ValueError("WAITING_FOR_INPUT requires a durable human request checkpoint")
            latest_request = self.human_requests[-1]
            latest_checkpoint = self.controller_checkpoints[-1]
            if latest_checkpoint.human_request_id != latest_request.request_id:
                raise ValueError("WAITING_FOR_INPUT must reference the latest human request")
            if latest_request.generation != self.generation:
                raise ValueError("WAITING_FOR_INPUT request must belong to the current generation")
        elif self.controller_lifecycle is ControllerLifecycle.CHECKPOINTED:
            if (
                not self.controller_checkpoints
                or self.controller_checkpoints[-1].generation != self.generation
            ):
                raise ValueError("checkpointed controller requires a current-generation checkpoint")

        history_generations = [entry.generation for entry in self.controller_generation_history]
        if self.controller_generation_history:
            if history_generations != list(range(1, self.generation + 1)):
                raise ValueError(
                    "controller generation history must cover every installed generation"
                )
            current_history = self.controller_generation_history[-1]
            if (
                current_history.job != self.controller_job
                or current_history.submission_token != self.controller_submission_token
            ):
                raise ValueError("current controller identity must match generation history")
            history_jobs = [entry.job for entry in self.controller_generation_history]
            if len(set(history_jobs)) != len(history_jobs):
                raise ValueError("controller generation jobs must be globally unique")

        successor_sequences = [entry.sequence for entry in self.controller_successors]
        if successor_sequences != list(range(1, len(successor_sequences) + 1)):
            raise ValueError("controller successor records must be in contiguous sequence order")
        successor_generations = [entry.successor_generation for entry in self.controller_successors]
        if len(set(successor_generations)) != len(successor_generations):
            raise ValueError("a controller generation may have only one successor")
        for entry in self.controller_successors:
            if entry.fenced:
                if entry.successor_generation > self.generation:
                    raise ValueError("fenced successor cannot be newer than current generation")
            elif entry.predecessor_generation != self.generation:
                raise ValueError("an unfenced successor must belong to the current generation")
        unfenced_successors = [entry for entry in self.controller_successors if not entry.fenced]
        if len(unfenced_successors) > 1:
            raise ValueError("only one controller successor may be pending")
        if self.controller_generation_history and self.controller_successors:
            fenced = tuple(entry for entry in self.controller_successors if entry.fenced)
            if len(fenced) != self.generation - 1:
                raise ValueError("controller history and fenced successor count disagree")
            for entry in fenced:
                predecessor = self.controller_generation_history[entry.predecessor_generation - 1]
                successor = self.controller_generation_history[entry.successor_generation - 1]
                if (
                    predecessor.job != entry.predecessor_job
                    or successor.job != entry.successor_job
                    or successor.submission_token != entry.submission_token
                ):
                    raise ValueError("controller successor record disagrees with job history")

        reconciliation_generations = [
            receipt.successor_generation for receipt in self.predecessor_reconciliations
        ]
        if (
            not self.controller_successors
            and len(self.predecessor_reconciliations) != self.generation - 1
        ):
            raise ValueError(
                "predecessor reconciliation receipts must cover every advanced generation"
            )
        if reconciliation_generations != list(range(2, 2 + len(reconciliation_generations))):
            raise ValueError(
                "predecessor reconciliation receipts must form a contiguous generation prefix"
            )
        receipts_by_generation = {
            receipt.successor_generation: receipt for receipt in self.predecessor_reconciliations
        }
        if self.controller_successors:
            successors_by_generation = {
                entry.successor_generation: entry
                for entry in self.controller_successors
                if entry.fenced
            }
            for successor_generation, receipt in receipts_by_generation.items():
                successor = successors_by_generation.get(successor_generation)
                if successor is None or (
                    receipt.predecessor_job != successor.predecessor_job
                    or receipt.successor_job != successor.successor_job
                ):
                    raise ValueError("predecessor receipt disagrees with successor history")
        if self.controller_lifecycle is ControllerLifecycle.AWAITING_PREDECESSOR:
            if not self.controller_successors:
                raise ValueError("awaiting predecessor requires a fenced successor record")
            latest_successor = self.controller_successors[-1]
            if (
                not latest_successor.fenced
                or latest_successor.successor_generation != self.generation
                or latest_successor.successor_job != self.controller_job
                or self.generation in receipts_by_generation
            ):
                raise ValueError("awaiting predecessor state does not match current successor")
        elif (
            self.controller_successors
            and self.generation > 1
            and self.controller_lifecycle is ControllerLifecycle.RUNNING
            and self.generation not in receipts_by_generation
        ):
            raise ValueError("successor cannot run before predecessor accounting reconciliation")
        orphan_generations = [window.generation for window in self.orphan_grace_windows]
        if len(set(orphan_generations)) != len(orphan_generations):
            raise ValueError("run state has duplicate orphan grace generations")
        if any(generation > self.generation for generation in orphan_generations):
            raise ValueError("orphan grace generation cannot exceed current generation")
        owned_jobs = {
            attempt.job
            for attempt in (
                *self.planning_attempts,
                *(a for item in self.items for a in item.attempts),
            )
            if attempt.job is not None
        }
        orphan_receipt_keys = [
            (receipt.generation, receipt.job) for receipt in self.orphan_action_receipts
        ]
        if len(set(orphan_receipt_keys)) != len(orphan_receipt_keys):
            raise ValueError("an orphan job may have only one action receipt per generation")
        grace_generations = set(orphan_generations)
        for receipt in self.orphan_action_receipts:
            if receipt.generation not in grace_generations:
                raise ValueError("orphan action receipt requires a grace window")
            if receipt.job not in owned_jobs:
                raise ValueError("orphan action receipt must reference an exact owned child job")
        adoption_sequences = [record.sequence for record in self.attempt_adoptions]
        if adoption_sequences != list(range(1, len(adoption_sequences) + 1)):
            raise ValueError("attempt adoptions must be in contiguous sequence order")
        adoption_keys = [
            (record.generation, record.attempt_id) for record in self.attempt_adoptions
        ]
        if len(set(adoption_keys)) != len(adoption_keys):
            raise ValueError("an attempt may be adopted only once per controller generation")
        all_attempts = {
            attempt.attempt_id: attempt
            for attempt in (
                *self.planning_attempts,
                *(entry for item in self.items for entry in item.attempts),
            )
        }
        for record in self.attempt_adoptions:
            attempt = all_attempts.get(record.attempt_id)
            if attempt is None:
                raise ValueError("attempt adoption references an unknown attempt")
            if record.generation > self.generation:
                raise ValueError("attempt adoption generation cannot be in the future")
            if record.attempt_generation != attempt.generation:
                raise ValueError("attempt adoption generation identity does not match")
            if (
                record.submission_token is not None
                and record.submission_token != attempt.submission_token
            ):
                raise ValueError("attempt adoption submission token does not match")
            if record.job is not None and record.job != attempt.job:
                raise ValueError("attempt adoption job identity does not match")
        if self.terminal_status is RunTerminalStatus.ACTIVE:
            if self.terminal_reason is not None:
                raise ValueError("an active run cannot have a terminal reason")
        else:
            _require_optional_text("terminal_reason", self.terminal_reason)
            if any(not item.terminal for item in self.items):
                raise ValueError("a terminal run cannot contain non-terminal work items")
            if any(not goal.terminal for goal in self.goals) or any(
                not stage.terminal for stage in self.stages
            ):
                raise ValueError("a terminal run cannot contain open Goals or Stages")
            if self.terminal_status is RunTerminalStatus.SUCCEEDED and any(
                item.status is not WorkItemStatus.INTEGRATED for item in self.items
            ):
                raise ValueError("a successful run requires every work item to be integrated")
            if self.terminal_status is RunTerminalStatus.SUCCEEDED and (
                any(goal.status is not HierarchyStatus.SUCCEEDED for goal in self.goals)
                or any(stage.status is not StageStatus.CLOSED for stage in self.stages)
            ):
                raise ValueError("a successful run requires every Goal and Stage to succeed")

    def item(self, item_id: str) -> WorkItemRecord:
        """Return one work item by identity.

        Args:
            item_id: Exact work-item identity.

        Returns:
            The matching work-item record.
        """
        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(item_id)

    @property
    def integration_head(self) -> str:
        """Return the latest persisted integration commit without changing base_commit."""
        if not self.integration_history:
            return self.base_commit
        return self.integration_history[-1].new_commit

    def record_integration(
        self,
        *,
        item_id: str,
        previous_commit: str,
        new_commit: str,
        evidence_kind: IntegrationEvidenceKind,
        evidence_digest: str,
    ) -> RunState:
        """Append one controller-verified fan-in result without performing Git work."""
        if self.terminal_status is not RunTerminalStatus.ACTIVE:
            raise InvalidTransitionError("a terminal run cannot record integration")
        if previous_commit != self.integration_head:
            raise InvalidTransitionError("integration previous_commit is stale")
        if any(record.item_id == item_id for record in self.integration_history):
            raise InvalidTransitionError("work item already has an integration record")
        item = self.item(item_id)
        if item.status not in {WorkItemStatus.APPROVED, WorkItemStatus.INTEGRATING}:
            raise InvalidTransitionError("integration requires an approved work item")
        record = IntegrationRecord(
            sequence=len(self.integration_history) + 1,
            generation=self.generation,
            item_id=item_id,
            previous_commit=previous_commit,
            new_commit=new_commit,
            evidence_kind=evidence_kind,
            evidence_digest=evidence_digest,
        )
        return replace(
            self,
            integration_history=(*self.integration_history, record),
            revision=self.revision + 1,
        )

    def replace_item(self, item: WorkItemRecord) -> RunState:
        """Replace an existing item and increment the in-memory revision.

        Args:
            item: Updated work-item record with the same identity.

        Returns:
            The next state revision.
        """
        current = self.item(item.item_id)
        _validate_item_update(current, item, expected_generation=self.generation)
        updated = tuple(item if entry.item_id == item.item_id else entry for entry in self.items)
        return replace(self, items=updated, revision=self.revision + 1)

    def record_attempt_adoption(
        self,
        *,
        attempt_id: str,
        attempt_generation: int,
        status: AttemptStatus,
        submission_token: str | None,
        job: JobReference | None,
    ) -> RunState:
        """Append an exact, idempotent successor-adoption receipt.

        The caller supplies the complete durable identity it observed.  Any
        mismatch with authoritative state fails closed instead of adopting a
        different scheduler attempt.
        """
        attempts = {
            attempt.attempt_id: attempt
            for attempt in (
                *self.planning_attempts,
                *(entry for item in self.items for entry in item.attempts),
            )
        }
        attempt = attempts.get(attempt_id)
        if attempt is None:
            raise InvalidTransitionError("cannot adopt an unknown attempt")
        if attempt.generation >= self.generation:
            raise InvalidTransitionError("only an older-generation attempt may be adopted")
        expected = (
            attempt.generation,
            attempt.status,
            attempt.submission_token,
            attempt.job,
        )
        observed = (attempt_generation, status, submission_token, job)
        if observed != expected:
            raise StateConflictError("attempt adoption does not match the exact persisted identity")
        existing = next(
            (
                record
                for record in self.attempt_adoptions
                if record.generation == self.generation and record.attempt_id == attempt_id
            ),
            None,
        )
        if existing is not None:
            if (
                existing.attempt_generation,
                existing.status,
                existing.submission_token,
                existing.job,
            ) != observed:
                raise StateConflictError("attempt adoption receipt is immutable")
            return self
        record = AttemptAdoptionRecord(
            sequence=len(self.attempt_adoptions) + 1,
            generation=self.generation,
            attempt_id=attempt_id,
            attempt_generation=attempt_generation,
            status=status,
            submission_token=submission_token,
            job=job,
        )
        return replace(
            self,
            attempt_adoptions=(*self.attempt_adoptions, record),
            revision=self.revision + 1,
        )

    def record_expired_attempt_cleanup(
        self,
        *,
        attempt_id: str,
        attempt_generation: int,
        status: AttemptStatus,
        submission_token: str | None,
        job: JobReference | None,
    ) -> RunState:
        """Atomically receipt and terminally fence an expired no-job intent."""
        attempts = {
            attempt.attempt_id: attempt
            for attempt in (
                *self.planning_attempts,
                *(entry for item in self.items for entry in item.attempts),
            )
        }
        attempt = attempts.get(attempt_id)
        if attempt is None:
            raise InvalidTransitionError("cannot clean up an unknown attempt")
        if attempt.generation >= self.generation:
            raise InvalidTransitionError("only an older-generation attempt may be cleaned up")
        observed = (attempt_generation, status, submission_token, job)
        expected = (
            attempt.generation,
            attempt.status,
            attempt.submission_token,
            attempt.job,
        )
        if observed != expected:
            raise StateConflictError("attempt cleanup does not match the exact persisted identity")
        if status not in {AttemptStatus.PREPARED, AttemptStatus.SUBMITTING} or job is not None:
            raise InvalidTransitionError("only a pre-submission no-job intent may be cleaned up")
        if any(
            record.generation == self.generation and record.attempt_id == attempt_id
            for record in self.attempt_adoptions
        ):
            raise StateConflictError("attempt cleanup receipt already exists")

        cleaned = attempt.transition(
            AttemptStatus.CANCELLED,
            terminal_reason="old-generation attempt expired before scheduler submission",
        )
        planning_attempts = self.planning_attempts
        items = self.items
        if attempt.item_id == PLANNING_ITEM_ID:
            planning_attempts = tuple(
                cleaned if entry.attempt_id == attempt_id else entry
                for entry in self.planning_attempts
            )
        else:
            item = self.item(attempt.item_id)
            cleaned_item = item.replace_attempt(cleaned)
            items = tuple(
                cleaned_item if entry.item_id == item.item_id else entry for entry in self.items
            )
        record = AttemptAdoptionRecord(
            sequence=len(self.attempt_adoptions) + 1,
            generation=self.generation,
            attempt_id=attempt_id,
            attempt_generation=attempt_generation,
            status=status,
            submission_token=submission_token,
            job=job,
        )
        return replace(
            self,
            planning_attempts=planning_attempts,
            items=items,
            attempt_adoptions=(*self.attempt_adoptions, record),
            revision=self.revision + 1,
        )

    def record_controller_submitted(self, job: JobReference) -> RunState:
        """Bind the immutable submission token to the adopted controller job.

        Args:
            job: Exact scheduler job adopted after token lookup or submit.

        Returns:
            The next state revision.
        """
        if self.controller_bootstrap_status is not ControllerBootstrapStatus.SUBMITTING:
            raise InvalidTransitionError("controller job may only be recorded from SUBMITTING")
        return replace(
            self,
            controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
            controller_job=job,
            revision=self.revision + 1,
        )

    def checkpoint_controller(
        self,
        *,
        reason: str,
        requeue_requested: bool,
        signal: ControllerCheckpointSignal | None = None,
    ) -> RunState:
        """Checkpoint a running controller without performing scheduler actions."""
        self._require_active_running_controller()
        checkpoint = ControllerCheckpointRecord(
            sequence=len(self.controller_checkpoints) + 1,
            generation=self.generation,
            reason=reason,
            requeue_requested=requeue_requested,
            signal=signal,
        )
        return replace(
            self,
            controller_lifecycle=ControllerLifecycle.CHECKPOINTED,
            controller_checkpoints=(*self.controller_checkpoints, checkpoint),
            revision=self.revision + 1,
        )

    def wait_for_human_input(
        self,
        *,
        request_id: str,
        prompt_digest: str,
        controller_requeue: bool,
        reason: str,
    ) -> RunState:
        """Checkpoint one detached request without reading stdin or requeuing."""
        self._require_active_running_controller()
        if any(request.response_digest is None for request in self.human_requests):
            raise InvalidTransitionError("another human request is still awaiting a response")
        request = HumanRequestRecord(
            request_id=request_id,
            generation=self.generation,
            prompt_digest=prompt_digest,
            controller_requeue=controller_requeue,
        )
        checkpoint = ControllerCheckpointRecord(
            sequence=len(self.controller_checkpoints) + 1,
            generation=self.generation,
            reason=reason,
            requeue_requested=controller_requeue,
            human_request_id=request_id,
        )
        return replace(
            self,
            controller_lifecycle=ControllerLifecycle.WAITING_FOR_INPUT,
            controller_checkpoints=(*self.controller_checkpoints, checkpoint),
            human_requests=(*self.human_requests, request),
            revision=self.revision + 1,
        )

    def record_human_response(self, *, request_id: str, response_digest: str) -> RunState:
        """Freeze the response digest for the current detached human request."""
        if self.terminal_status is not RunTerminalStatus.ACTIVE:
            raise InvalidTransitionError("a terminal run cannot accept human input")
        if self.controller_lifecycle is not ControllerLifecycle.WAITING_FOR_INPUT:
            raise InvalidTransitionError("controller is not WAITING_FOR_INPUT")
        if not self.human_requests or self.human_requests[-1].request_id != request_id:
            raise InvalidTransitionError("response does not match the current human request")
        updated = self.human_requests[-1].record_response(response_digest)
        return replace(
            self,
            human_requests=(*self.human_requests[:-1], updated),
            revision=self.revision + 1,
        )

    def begin_successor_submission(
        self,
        *,
        submission_token: str,
        reason: str,
        human_request_id: str | None = None,
    ) -> RunState:
        """Persist exactly one successor intent before any scheduler submission."""
        if self.terminal_status is not RunTerminalStatus.ACTIVE:
            raise InvalidTransitionError("a terminal run cannot submit a successor")
        if self.controller_job is None or self.controller_submission_token is None:
            raise InvalidTransitionError("successor submission requires an exact current job")
        if self.controller_job.array_task_id is not None:
            raise InvalidTransitionError("controller successor cannot depend on an array element")
        if any(not successor.fenced for successor in self.controller_successors):
            raise InvalidTransitionError("a successor submission is already pending")
        if any(
            successor.predecessor_generation == self.generation
            for successor in self.controller_successors
        ):
            raise InvalidTransitionError("current controller generation already has a successor")
        if human_request_id is None:
            if self.controller_lifecycle not in {
                ControllerLifecycle.RUNNING,
                ControllerLifecycle.CHECKPOINTED,
            }:
                raise InvalidTransitionError(
                    "controller successor requires a running or checkpointed generation"
                )
        else:
            if self.controller_lifecycle is not ControllerLifecycle.WAITING_FOR_INPUT:
                raise InvalidTransitionError(
                    "login-side human successor requires WAITING_FOR_INPUT"
                )
            if not self.human_requests or self.human_requests[-1].request_id != human_request_id:
                raise InvalidTransitionError("successor does not match the latest human request")
            request = self.human_requests[-1]
            if request.response_digest is None:
                raise InvalidTransitionError(
                    "human response must be immutably recorded before successor submission"
                )
            if not request.controller_requeue:
                raise InvalidTransitionError("human request did not authorize controller requeue")
        record = ControllerSuccessorRecord(
            sequence=len(self.controller_successors) + 1,
            predecessor_generation=self.generation,
            predecessor_job=self.controller_job,
            successor_generation=self.generation + 1,
            submission_token=submission_token,
            reason=reason,
            human_request_id=human_request_id,
        )
        return replace(
            self,
            controller_successors=(*self.controller_successors, record),
            revision=self.revision + 1,
        )

    def record_successor_submitted(
        self,
        *,
        submission_token: str,
        job: JobReference,
    ) -> RunState:
        """Persist the exact submitted or adopted successor scheduler identity."""
        if not self.controller_successors:
            raise InvalidTransitionError("no successor submission intent exists")
        successor = self.controller_successors[-1]
        if successor.fenced or successor.predecessor_generation != self.generation:
            raise InvalidTransitionError("latest successor intent is not pending")
        if successor.submission_token != submission_token:
            raise InvalidTransitionError("successor token does not match pending intent")
        updated = successor.record_job(job)
        return replace(
            self,
            controller_successors=(*self.controller_successors[:-1], updated),
            revision=self.revision + 1,
        )

    def fence_to_successor(self) -> RunState:
        """Atomically install the submitted successor and fence the current writer."""
        if not self.controller_successors:
            raise InvalidTransitionError("no successor submission exists to fence")
        successor = self.controller_successors[-1]
        if successor.fenced or successor.predecessor_generation != self.generation:
            raise InvalidTransitionError("latest successor is not fenceable")
        fenced = successor.fence()
        if fenced.successor_job is None:
            raise InvalidTransitionError("successor has no exact scheduler identity")
        if self.controller_generation_history:
            history = self.controller_generation_history
        else:
            if self.controller_job is None or self.controller_submission_token is None:
                raise InvalidTransitionError("current controller identity is incomplete")
            history = (
                ControllerGenerationRecord(
                    generation=self.generation,
                    submission_token=self.controller_submission_token,
                    job=self.controller_job,
                ),
            )
        history = (
            *history,
            ControllerGenerationRecord(
                generation=fenced.successor_generation,
                submission_token=fenced.submission_token,
                job=fenced.successor_job,
            ),
        )
        return replace(
            self,
            generation=fenced.successor_generation,
            controller_submission_token=fenced.submission_token,
            controller_job=fenced.successor_job,
            controller_lifecycle=ControllerLifecycle.AWAITING_PREDECESSOR,
            controller_generation_history=history,
            controller_successors=(*self.controller_successors[:-1], fenced),
            revision=self.revision + 1,
        )

    def record_predecessor_terminal(
        self,
        *,
        predecessor_job: JobReference,
        scheduler_state: str,
        observation_source: str,
    ) -> RunState:
        """Record exact terminal accounting evidence before successor work starts."""
        if self.controller_lifecycle is not ControllerLifecycle.AWAITING_PREDECESSOR:
            raise InvalidTransitionError("controller is not awaiting predecessor reconciliation")
        if not self.controller_successors:
            raise InvalidTransitionError("successor history is missing")
        successor = self.controller_successors[-1]
        if (
            not successor.fenced
            or successor.successor_generation != self.generation
            or successor.predecessor_job != predecessor_job
            or successor.successor_job != self.controller_job
        ):
            raise InvalidTransitionError("predecessor does not match fenced successor history")
        receipt = PredecessorReconciliationReceipt(
            predecessor_generation=successor.predecessor_generation,
            successor_generation=successor.successor_generation,
            predecessor_job=predecessor_job,
            successor_job=successor.successor_job,
            scheduler_state=scheduler_state,
            observation_source=observation_source,
        )
        return replace(
            self,
            controller_lifecycle=ControllerLifecycle.RUNNING,
            predecessor_reconciliations=(*self.predecessor_reconciliations, receipt),
            revision=self.revision + 1,
        )

    def resume_after_reconciliation(
        self,
        *,
        predecessor_job: JobReference,
        scheduler_state: str,
    ) -> RunState:
        """Advance one generation after caller-proven exact terminal reconciliation."""
        if self.terminal_status is not RunTerminalStatus.ACTIVE:
            raise InvalidTransitionError("a terminal run cannot advance controller generation")
        if self.controller_job is None or predecessor_job != self.controller_job:
            raise InvalidTransitionError("predecessor job does not match the exact controller job")
        if any(request.response_digest is None for request in self.human_requests):
            raise InvalidTransitionError("controller cannot resume before human input is recorded")
        receipt = PredecessorReconciliationReceipt(
            predecessor_generation=self.generation,
            successor_generation=self.generation + 1,
            predecessor_job=predecessor_job,
            scheduler_state=scheduler_state,
            successor_job=None,
        )
        return replace(
            self,
            generation=self.generation + 1,
            controller_lifecycle=ControllerLifecycle.RUNNING,
            predecessor_reconciliations=(*self.predecessor_reconciliations, receipt),
            revision=self.revision + 1,
        )

    def start_orphan_grace(self, *, started_at: str, deadline_at: str) -> RunState:
        """Record a caller-timestamped orphan grace window for this generation."""
        if self.terminal_status is not RunTerminalStatus.ACTIVE:
            raise InvalidTransitionError("a terminal run cannot start orphan recovery")
        if any(window.generation == self.generation for window in self.orphan_grace_windows):
            raise InvalidTransitionError("orphan grace already started for this generation")
        window = OrphanGraceWindow(
            generation=self.generation,
            started_at=started_at,
            deadline_at=deadline_at,
        )
        return replace(
            self,
            orphan_grace_windows=(*self.orphan_grace_windows, window),
            revision=self.revision + 1,
        )

    def record_orphan_action(
        self,
        *,
        action: OrphanReceiptAction,
        job: JobReference,
        scheduler_state: str,
    ) -> RunState:
        """Record exact adoption or cleanup after its scheduler side effect."""
        if self.terminal_status is not RunTerminalStatus.ACTIVE:
            raise InvalidTransitionError("a terminal run cannot record orphan recovery")
        receipt = OrphanActionReceipt(
            generation=self.generation,
            action=action,
            job=job,
            scheduler_state=scheduler_state,
        )
        return replace(
            self,
            orphan_action_receipts=(*self.orphan_action_receipts, receipt),
            revision=self.revision + 1,
        )

    def _require_active_running_controller(self) -> None:
        if self.terminal_status is not RunTerminalStatus.ACTIVE:
            raise InvalidTransitionError("a terminal run cannot checkpoint its controller")
        if self.controller_lifecycle is not ControllerLifecycle.RUNNING:
            raise InvalidTransitionError("only a running controller may create a checkpoint")

    def finish(self, status: RunTerminalStatus, reason: str) -> RunState:
        """Produce a terminal run state.

        Args:
            status: Non-active terminal result.
            reason: Evidence-backed terminal explanation.

        Returns:
            The next state revision.
        """
        if status is RunTerminalStatus.ACTIVE:
            raise ValueError("finish requires a terminal status")
        return replace(
            self,
            terminal_status=status,
            terminal_reason=reason,
            revision=self.revision + 1,
        )


@dataclass(frozen=True)
class ControllerLeaseRecord:
    """Durable identity of the controller holding the filesystem lease."""

    run_id: str
    generation: int
    owner_nonce: str
    controller_job: JobReference | None
    heartbeat_at: str
    schema_version: int = field(default=LEASE_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_identifier("run_id", self.run_id)
        if self.generation < 1:
            raise ValueError("lease generation must be positive")
        _require_optional_text("owner_nonce", self.owner_nonce)
        _parse_utc_timestamp(self.heartbeat_at)


class ControllerLease:
    """Process-held exclusive lock paired with a durable lease record.

    A leftover record is never considered dead from its heartbeat age. Recovery
    requires the caller to explicitly attest that exact scheduler reconciliation
    completed.
    """

    def __init__(self, path: Path, lock_file: TextIO, record: ControllerLeaseRecord) -> None:
        self.path = path
        self._lock_file = lock_file
        self.record = record
        self._closed = False

    @classmethod
    def acquire(
        cls,
        path: Path,
        *,
        run_id: str,
        generation: int,
        controller_job: JobReference | None,
        owner_nonce: str | None = None,
    ) -> ControllerLease:
        """Acquire a new lease only when no durable predecessor exists.

        Args:
            path: Durable lease record path.
            run_id: Owning run identity.
            generation: Active controller generation.
            controller_job: Exact controller scheduler identity, when known.
            owner_nonce: Optional test-supplied cryptographic owner nonce.

        Returns:
            A process-held lease.

        Raises:
            LeaseConflictError: If another process holds the lock or an
                unreconciled durable record remains.
        """
        lock_file = _acquire_lock(path, blocking=False)
        try:
            if path.exists():
                existing = load_lease_record(path)
                raise LeaseConflictError(
                    "controller lease record already exists for "
                    f"generation {existing.generation}; reconcile its exact scheduler "
                    "identity before recovery"
                )
            record = ControllerLeaseRecord(
                run_id=run_id,
                generation=generation,
                owner_nonce=owner_nonce or secrets.token_hex(16),
                controller_job=controller_job,
                heartbeat_at=_utc_now(),
            )
            _atomic_write_json(path, _lease_to_dict(record))
            return cls(path, lock_file, record)
        except BaseException:
            _unlock_and_close(lock_file)
            raise

    @classmethod
    def recover(
        cls,
        path: Path,
        *,
        run_id: str,
        generation: int,
        controller_job: JobReference | None,
        scheduler_reconciled: bool,
        owner_nonce: str | None = None,
        expected_predecessor_job: JobReference | None = None,
    ) -> ControllerLease:
        """Replace a predecessor lease after exact external reconciliation.

        Args:
            path: Durable lease record path.
            run_id: Owning run identity.
            generation: Strictly newer controller generation.
            controller_job: New exact controller scheduler identity.
            scheduler_reconciled: Explicit proof signal from scheduler code;
                heartbeat time alone is never sufficient.
            owner_nonce: Optional test-supplied owner nonce.
            expected_predecessor_job: Exact prior lease identity required by a
                controller-successor handoff.

        Returns:
            A process-held recovered lease.
        """
        if not scheduler_reconciled:
            raise LeaseConflictError("lease recovery requires exact scheduler reconciliation")
        lock_file = _acquire_lock(path, blocking=False)
        try:
            if not path.exists():
                raise LeaseConflictError("cannot recover a missing lease; acquire a new lease")
            previous = load_lease_record(path)
            if previous.run_id != run_id:
                raise LeaseConflictError("cannot recover a lease owned by another run")
            if (
                expected_predecessor_job is not None
                and previous.controller_job != expected_predecessor_job
            ):
                raise LeaseConflictError(
                    "durable lease does not match the reconciled predecessor job"
                )
            if generation <= previous.generation:
                raise LeaseConflictError("recovered lease generation must strictly increase")
            record = ControllerLeaseRecord(
                run_id=run_id,
                generation=generation,
                owner_nonce=owner_nonce or secrets.token_hex(16),
                controller_job=controller_job,
                heartbeat_at=_utc_now(),
            )
            _atomic_write_json(path, _lease_to_dict(record))
            return cls(path, lock_file, record)
        except BaseException:
            _unlock_and_close(lock_file)
            raise

    def heartbeat(self) -> ControllerLeaseRecord:
        """Durably refresh the heartbeat after validating owner fencing."""
        self._require_open()
        current = load_lease_record(self.path)
        if (
            current.run_id != self.record.run_id
            or current.generation != self.record.generation
            or current.owner_nonce != self.record.owner_nonce
        ):
            raise LeaseConflictError("controller lease fencing identity changed")
        self.record = replace(self.record, heartbeat_at=_utc_now())
        _atomic_write_json(self.path, _lease_to_dict(self.record))
        return self.record

    def release(self, *, remove_record: bool = True) -> None:
        """Release the process lock and optionally remove its owned record.

        Args:
            remove_record: Keep the durable record for requeue/recovery when
                ``False``. It still cannot be stolen based on age.
        """
        if self._closed:
            return
        try:
            if remove_record and self.path.exists():
                current = load_lease_record(self.path)
                if (
                    current.generation != self.record.generation
                    or current.owner_nonce != self.record.owner_nonce
                ):
                    raise LeaseConflictError(
                        "refusing to remove a lease owned by another controller"
                    )
                self.path.unlink()
                _fsync_directory(self.path.parent)
        finally:
            _unlock_and_close(self._lock_file)
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise LeaseConflictError("controller lease is closed")

    def __enter__(self) -> ControllerLease:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


def migrate_state_v1(
    payload: dict[str, object],
    *,
    succeeded_stage_policy: LegacySucceededStagePolicy = LegacySucceededStagePolicy.REJECT,
) -> dict[str, object]:
    """Return an explicit schema-v2 migration of one schema-v1 state payload.

    Schema v1 recorded only ``succeeded`` for a Stage and therefore contains no
    immutable evidence that independent Stage QA passed. Such a Stage can never
    be silently promoted to ``closed``. Callers must either reject the state or
    explicitly downgrade it to ``blocked`` for operator reconciliation.
    """
    if payload.get("schema_version") != LEGACY_SCHEMA_VERSION:
        raise StateMigrationError(f"migrate_state_v1 requires schema {LEGACY_SCHEMA_VERSION} input")
    if not isinstance(succeeded_stage_policy, LegacySucceededStagePolicy):
        raise TypeError("succeeded_stage_policy must be a LegacySucceededStagePolicy")
    migrated = deepcopy(payload)
    stages = migrated.get("stages")
    if not isinstance(stages, list):
        raise StateMigrationError("schema-v1 state stages must be a list")
    downgraded_stage_ids: list[str] = []
    for index, value in enumerate(stages):
        if not isinstance(value, dict):
            raise StateMigrationError(f"schema-v1 Stage {index} must be an object")
        status = value.get("status")
        if status == HierarchyStatus.SUCCEEDED.value:
            if succeeded_stage_policy is LegacySucceededStagePolicy.REJECT:
                raise StateMigrationError(
                    "schema-v1 succeeded Stage lacks immutable QA evidence; "
                    "choose the explicit BLOCK policy or restart from trusted evidence"
                )
            stage_id = value.get("stage_id")
            if not isinstance(stage_id, str) or not stage_id:
                raise StateMigrationError("schema-v1 succeeded Stage lacks a valid stage_id")
            value["status"] = StageStatus.BLOCKED.value
            value["terminal_reason"] = (
                "schema-v1 succeeded Stage downgraded: independent QA closure is unverifiable"
            )
            downgraded_stage_ids.append(stage_id)
        elif status not in {
            StageStatus.PLANNED.value,
            StageStatus.ACTIVE.value,
            StageStatus.BLOCKED.value,
            StageStatus.CANCELLED.value,
        }:
            raise StateMigrationError(f"unsupported schema-v1 Stage status {status!r}")
    if (
        downgraded_stage_ids
        and migrated.get("terminal_status") == RunTerminalStatus.SUCCEEDED.value
    ):
        migrated["terminal_status"] = RunTerminalStatus.BLOCKED.value
        migrated["terminal_reason"] = (
            "schema-v1 run downgraded because Stage QA closure is unverifiable: "
            + ", ".join(downgraded_stage_ids)
        )
    migrated["schema_version"] = SCHEMA_VERSION
    return migrated


def load_state(path: Path) -> RunState:
    """Load and validate authoritative state.

    Args:
        path: JSON state path.

    Returns:
        Validated state.
    """
    raw = _load_json_object(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        if raw.get("schema_version") == LEGACY_SCHEMA_VERSION:
            raise StateMigrationError(
                "Staircase state schema 1 requires explicit migrate_state_v1(); "
                "legacy succeeded Stages cannot be inferred as QA-closed"
            )
        raise ValueError(
            f"unsupported Staircase state schema {raw.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    expected = {
        "schema_version",
        "run_id",
        "task_digest",
        "base_commit",
        "generation",
        "workflow_mode",
        "revision",
        "stages",
        "goals",
        "items",
        "planning_attempts",
        "integration_history",
        "controller_bootstrap_status",
        "controller_submission_token",
        "controller_job",
        "controller_lifecycle",
        "controller_checkpoints",
        "human_requests",
        "controller_generation_history",
        "controller_successors",
        "predecessor_reconciliations",
        "orphan_grace_windows",
        "orphan_action_receipts",
        "attempt_adoptions",
        "terminal_status",
        "terminal_reason",
    }
    _reject_unknown_keys(raw, expected, "run state")
    items_raw = _require_list(raw.get("items"), "items")
    return RunState(
        run_id=_require_str(raw.get("run_id"), "run_id"),
        task_digest=_require_str(raw.get("task_digest"), "task_digest"),
        base_commit=_require_str(raw.get("base_commit"), "base_commit"),
        generation=_require_int(raw.get("generation"), "generation"),
        workflow_mode=WorkflowMode(_require_str(raw.get("workflow_mode"), "workflow_mode")),
        revision=_require_int(raw.get("revision"), "revision"),
        stages=tuple(
            _stage_from_dict(_require_dict(stage, "Stage"))
            for stage in _require_list(raw.get("stages"), "stages")
        ),
        goals=tuple(
            _goal_from_dict(_require_dict(goal, "Goal"))
            for goal in _require_list(raw.get("goals"), "goals")
        ),
        items=tuple(_work_item_from_dict(_require_dict(item, "work item")) for item in items_raw),
        planning_attempts=tuple(
            _attempt_from_dict(_require_dict(attempt, "planning attempt"))
            for attempt in _require_list(raw.get("planning_attempts"), "planning_attempts")
        ),
        integration_history=tuple(
            _integration_record_from_dict(_require_dict(entry, "integration record"))
            for entry in _require_list(raw.get("integration_history"), "integration_history")
        ),
        controller_bootstrap_status=ControllerBootstrapStatus(
            _require_str(raw.get("controller_bootstrap_status"), "controller_bootstrap_status")
        ),
        controller_submission_token=_optional_str(
            raw.get("controller_submission_token"), "controller_submission_token"
        ),
        controller_job=_job_from_value(raw.get("controller_job")),
        controller_lifecycle=ControllerLifecycle(
            _require_str(raw.get("controller_lifecycle"), "controller_lifecycle")
        ),
        controller_checkpoints=tuple(
            _controller_checkpoint_from_dict(_require_dict(entry, "controller checkpoint"))
            for entry in _require_list(raw.get("controller_checkpoints"), "controller_checkpoints")
        ),
        human_requests=tuple(
            _human_request_from_dict(_require_dict(entry, "human request"))
            for entry in _require_list(raw.get("human_requests"), "human_requests")
        ),
        controller_generation_history=tuple(
            _controller_generation_from_dict(_require_dict(entry, "controller generation"))
            for entry in _require_list(
                raw.get("controller_generation_history"), "controller_generation_history"
            )
        ),
        controller_successors=tuple(
            _controller_successor_from_dict(_require_dict(entry, "controller successor"))
            for entry in _require_list(raw.get("controller_successors"), "controller_successors")
        ),
        predecessor_reconciliations=tuple(
            _predecessor_reconciliation_from_dict(
                _require_dict(entry, "predecessor reconciliation")
            )
            for entry in _require_list(
                raw.get("predecessor_reconciliations"), "predecessor_reconciliations"
            )
        ),
        orphan_grace_windows=tuple(
            _orphan_grace_from_dict(_require_dict(entry, "orphan grace window"))
            for entry in _require_list(raw.get("orphan_grace_windows"), "orphan_grace_windows")
        ),
        orphan_action_receipts=tuple(
            _orphan_action_from_dict(_require_dict(entry, "orphan action receipt"))
            for entry in _require_list(raw.get("orphan_action_receipts"), "orphan_action_receipts")
        ),
        attempt_adoptions=tuple(
            _attempt_adoption_from_dict(_require_dict(entry, "attempt adoption"))
            for entry in _require_list(raw.get("attempt_adoptions", []), "attempt_adoptions")
        ),
        terminal_status=RunTerminalStatus(
            _require_str(raw.get("terminal_status"), "terminal_status")
        ),
        terminal_reason=_optional_str(raw.get("terminal_reason"), "terminal_reason"),
    )


def initialize_state(path: Path, state: RunState) -> None:
    """Create revision-zero state without replacing an existing run.

    Args:
        path: JSON state path.
        state: Initial generation and revision-zero state.
    """
    if state.revision != 0:
        raise StateConflictError("initial state revision must be zero")
    with _locked(path):
        if path.exists():
            raise StateConflictError(f"state already exists at {path}")
        _atomic_write_json(path, _state_to_dict(state))


def save_state(
    path: Path,
    state: RunState,
    *,
    expected_revision: int,
    expected_generation: int,
) -> None:
    """Atomically persist state using revision and generation fencing.

    Args:
        path: Existing authoritative state path.
        state: Next state; its revision must be exactly one greater than the
            on-disk revision and its generation must match the active writer.
        expected_revision: Revision observed by the caller.
        expected_generation: Active controller generation observed by the
            caller.
    """
    with _locked(path):
        current = load_state(path)
        if current.revision != expected_revision:
            raise StateConflictError(
                f"state revision changed from {expected_revision} to {current.revision}"
            )
        if current.generation != expected_generation or state.generation != expected_generation:
            raise StateConflictError(
                "controller generation fence failed: "
                f"disk={current.generation}, writer={expected_generation}, next={state.generation}"
            )
        if state.revision != expected_revision + 1:
            raise StateConflictError(
                f"next revision must be {expected_revision + 1}, got {state.revision}"
            )
        if state.run_id != current.run_id or state.task_digest != current.task_digest:
            raise StateConflictError("run and task identities are immutable")
        if state.base_commit != current.base_commit:
            raise StateConflictError("base commit identity is immutable")
        if state.workflow_mode is not current.workflow_mode:
            raise StateConflictError("workflow mode is immutable")
        _validate_state_update(current, state)
        _atomic_write_json(path, _state_to_dict(state))


def advance_generation(
    path: Path,
    state: RunState,
    *,
    expected_revision: int,
    expected_generation: int,
) -> None:
    """Atomically fence out a reconciled predecessor controller generation.

    Args:
        path: Existing authoritative state path.
        state: Next state with generation and revision each incremented once.
        expected_revision: Revision observed before recovery.
        expected_generation: Predecessor generation reconciled by the caller.
    """
    with _locked(path):
        current = load_state(path)
        if current.revision != expected_revision or current.generation != expected_generation:
            raise StateConflictError("state changed while advancing controller generation")
        if state.revision != expected_revision + 1:
            raise StateConflictError("generation advance must increment revision exactly once")
        if state.generation != expected_generation + 1:
            raise StateConflictError("generation advance must increment generation exactly once")
        if state.run_id != current.run_id or state.task_digest != current.task_digest:
            raise StateConflictError("run and task identities are immutable")
        if state.base_commit != current.base_commit:
            raise StateConflictError("base commit identity is immutable")
        if state.workflow_mode is not current.workflow_mode:
            raise StateConflictError("workflow mode is immutable")
        _validate_state_update(current, state, generation_advancing=True)
        _atomic_write_json(path, _state_to_dict(state))


def load_lease_record(path: Path) -> ControllerLeaseRecord:
    """Load a durable lease record without interpreting heartbeat age.

    Args:
        path: Lease JSON path.

    Returns:
        Validated lease record.
    """
    raw = _load_json_object(path)
    expected = {
        "schema_version",
        "run_id",
        "generation",
        "owner_nonce",
        "controller_job",
        "heartbeat_at",
    }
    _reject_unknown_keys(raw, expected, "controller lease")
    if raw.get("schema_version") != LEASE_SCHEMA_VERSION:
        raise ValueError("unsupported controller lease schema")
    return ControllerLeaseRecord(
        run_id=_require_str(raw.get("run_id"), "run_id"),
        generation=_require_int(raw.get("generation"), "generation"),
        owner_nonce=_require_str(raw.get("owner_nonce"), "owner_nonce"),
        controller_job=_job_from_value(raw.get("controller_job")),
        heartbeat_at=_require_str(raw.get("heartbeat_at"), "heartbeat_at"),
    )


def _state_to_dict(state: RunState) -> dict[str, object]:
    payload = asdict(state)
    payload["workflow_mode"] = state.workflow_mode.value
    payload["terminal_status"] = state.terminal_status.value
    payload["controller_bootstrap_status"] = state.controller_bootstrap_status.value
    payload["controller_lifecycle"] = state.controller_lifecycle.value
    payload["stages"] = [_stage_to_dict(stage) for stage in state.stages]
    payload["goals"] = [_goal_to_dict(goal) for goal in state.goals]
    payload["items"] = [_work_item_to_dict(item) for item in state.items]
    payload["planning_attempts"] = [
        _attempt_to_dict(attempt) for attempt in state.planning_attempts
    ]
    payload["integration_history"] = [
        _integration_record_to_dict(record) for record in state.integration_history
    ]
    payload["controller_job"] = _job_to_dict(state.controller_job)
    payload["controller_checkpoints"] = [
        _controller_checkpoint_to_dict(checkpoint) for checkpoint in state.controller_checkpoints
    ]
    payload["human_requests"] = [asdict(request) for request in state.human_requests]
    payload["controller_generation_history"] = [
        _controller_generation_to_dict(record) for record in state.controller_generation_history
    ]
    payload["controller_successors"] = [
        _controller_successor_to_dict(record) for record in state.controller_successors
    ]
    payload["predecessor_reconciliations"] = [
        _predecessor_reconciliation_to_dict(receipt)
        for receipt in state.predecessor_reconciliations
    ]
    payload["orphan_grace_windows"] = [asdict(window) for window in state.orphan_grace_windows]
    payload["orphan_action_receipts"] = [
        _orphan_action_to_dict(receipt) for receipt in state.orphan_action_receipts
    ]
    payload["attempt_adoptions"] = [
        _attempt_adoption_to_dict(record) for record in state.attempt_adoptions
    ]
    return payload


def _controller_checkpoint_to_dict(
    checkpoint: ControllerCheckpointRecord,
) -> dict[str, object]:
    payload = asdict(checkpoint)
    payload["signal"] = checkpoint.signal.value if checkpoint.signal is not None else None
    return payload


def _predecessor_reconciliation_to_dict(
    receipt: PredecessorReconciliationReceipt,
) -> dict[str, object]:
    payload = asdict(receipt)
    payload["predecessor_job"] = _job_to_dict(receipt.predecessor_job)
    payload["successor_job"] = _job_to_dict(receipt.successor_job)
    return payload


def _controller_generation_to_dict(
    record: ControllerGenerationRecord,
) -> dict[str, object]:
    payload = asdict(record)
    payload["job"] = _job_to_dict(record.job)
    return payload


def _controller_successor_to_dict(record: ControllerSuccessorRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["predecessor_job"] = _job_to_dict(record.predecessor_job)
    payload["successor_job"] = _job_to_dict(record.successor_job)
    return payload


def _orphan_action_to_dict(receipt: OrphanActionReceipt) -> dict[str, object]:
    payload = asdict(receipt)
    payload["action"] = receipt.action.value
    payload["job"] = _job_to_dict(receipt.job)
    return payload


def _integration_record_to_dict(record: IntegrationRecord) -> dict[str, object]:
    payload = asdict(record)
    payload["evidence_kind"] = record.evidence_kind.value
    return payload


def _stage_to_dict(stage: StageRecord) -> dict[str, object]:
    payload = asdict(stage)
    payload["required_goal_ids"] = list(stage.required_goal_ids)
    payload["status"] = stage.status.value
    return payload


def _goal_to_dict(goal: GoalRecord) -> dict[str, object]:
    payload = asdict(goal)
    payload["required_item_ids"] = list(goal.required_item_ids)
    payload["status"] = goal.status.value
    return payload


def _work_item_to_dict(item: WorkItemRecord) -> dict[str, object]:
    payload = asdict(item)
    payload["kind"] = item.kind.value
    payload["profile"] = item.profile.value
    payload["status"] = item.status.value
    payload["dependencies"] = list(item.dependencies)
    payload["attempts"] = [_attempt_to_dict(attempt) for attempt in item.attempts]
    return payload


def _attempt_to_dict(attempt: AttemptRecord) -> dict[str, object]:
    payload = asdict(attempt)
    payload["role"] = attempt.role.value
    payload["kind"] = attempt.kind.value
    payload["profile"] = attempt.profile.value if attempt.profile is not None else None
    payload["status"] = attempt.status.value
    payload["job"] = _job_to_dict(attempt.job)
    return payload


def _attempt_adoption_to_dict(
    record: AttemptAdoptionRecord,
) -> dict[str, object]:
    payload = asdict(record)
    payload["status"] = record.status.value
    payload["job"] = _job_to_dict(record.job)
    return payload


def _lease_to_dict(lease: ControllerLeaseRecord) -> dict[str, object]:
    payload = asdict(lease)
    payload["controller_job"] = _job_to_dict(lease.controller_job)
    return payload


def _controller_checkpoint_from_dict(raw: dict[str, object]) -> ControllerCheckpointRecord:
    expected = {
        "sequence",
        "generation",
        "reason",
        "requeue_requested",
        "signal",
        "human_request_id",
    }
    _reject_unknown_keys(raw, expected, "controller checkpoint")
    signal = _optional_str(raw.get("signal"), "checkpoint signal")
    return ControllerCheckpointRecord(
        sequence=_require_int(raw.get("sequence"), "checkpoint sequence"),
        generation=_require_int(raw.get("generation"), "checkpoint generation"),
        reason=_require_str(raw.get("reason"), "checkpoint reason"),
        requeue_requested=_require_bool(raw.get("requeue_requested"), "requeue_requested"),
        signal=ControllerCheckpointSignal(signal) if signal is not None else None,
        human_request_id=_optional_str(raw.get("human_request_id"), "human_request_id"),
    )


def _human_request_from_dict(raw: dict[str, object]) -> HumanRequestRecord:
    expected = {
        "request_id",
        "generation",
        "prompt_digest",
        "controller_requeue",
        "response_digest",
    }
    _reject_unknown_keys(raw, expected, "human request")
    return HumanRequestRecord(
        request_id=_require_str(raw.get("request_id"), "request_id"),
        generation=_require_int(raw.get("generation"), "human request generation"),
        prompt_digest=_require_str(raw.get("prompt_digest"), "prompt_digest"),
        controller_requeue=_require_bool(raw.get("controller_requeue"), "controller_requeue"),
        response_digest=_optional_str(raw.get("response_digest"), "response_digest"),
    )


def _controller_generation_from_dict(
    raw: dict[str, object],
) -> ControllerGenerationRecord:
    expected = {"generation", "submission_token", "job"}
    _reject_unknown_keys(raw, expected, "controller generation")
    job = _job_from_value(raw.get("job"))
    if job is None:
        raise ValueError("controller generation job must be an exact scheduler job")
    return ControllerGenerationRecord(
        generation=_require_int(raw.get("generation"), "controller generation"),
        submission_token=_require_str(
            raw.get("submission_token"), "controller generation submission_token"
        ),
        job=job,
    )


def _controller_successor_from_dict(raw: dict[str, object]) -> ControllerSuccessorRecord:
    expected = {
        "sequence",
        "predecessor_generation",
        "predecessor_job",
        "successor_generation",
        "submission_token",
        "reason",
        "human_request_id",
        "successor_job",
        "fenced",
    }
    _reject_unknown_keys(raw, expected, "controller successor")
    predecessor_job = _job_from_value(raw.get("predecessor_job"))
    if predecessor_job is None:
        raise ValueError("controller successor requires an exact predecessor job")
    return ControllerSuccessorRecord(
        sequence=_require_int(raw.get("sequence"), "successor sequence"),
        predecessor_generation=_require_int(
            raw.get("predecessor_generation"), "successor predecessor_generation"
        ),
        predecessor_job=predecessor_job,
        successor_generation=_require_int(raw.get("successor_generation"), "successor generation"),
        submission_token=_require_str(raw.get("submission_token"), "successor submission_token"),
        reason=_require_str(raw.get("reason"), "successor reason"),
        human_request_id=_optional_str(raw.get("human_request_id"), "successor human_request_id"),
        successor_job=_job_from_value(raw.get("successor_job")),
        fenced=_require_bool(raw.get("fenced"), "successor fenced"),
    )


def _predecessor_reconciliation_from_dict(
    raw: dict[str, object],
) -> PredecessorReconciliationReceipt:
    expected = {
        "predecessor_generation",
        "successor_generation",
        "predecessor_job",
        "successor_job",
        "scheduler_state",
        "observation_source",
    }
    _reject_unknown_keys(raw, expected, "predecessor reconciliation")
    predecessor_job = _job_from_value(raw.get("predecessor_job"))
    if predecessor_job is None:
        raise ValueError("predecessor_job must be an exact scheduler job")
    return PredecessorReconciliationReceipt(
        predecessor_generation=_require_int(
            raw.get("predecessor_generation"), "predecessor_generation"
        ),
        successor_generation=_require_int(raw.get("successor_generation"), "successor_generation"),
        predecessor_job=predecessor_job,
        scheduler_state=_require_str(raw.get("scheduler_state"), "scheduler_state"),
        successor_job=_job_from_value(raw.get("successor_job")),
        observation_source=_require_str(raw.get("observation_source"), "observation_source"),
    )


def _orphan_grace_from_dict(raw: dict[str, object]) -> OrphanGraceWindow:
    expected = {"generation", "started_at", "deadline_at"}
    _reject_unknown_keys(raw, expected, "orphan grace window")
    return OrphanGraceWindow(
        generation=_require_int(raw.get("generation"), "orphan grace generation"),
        started_at=_require_str(raw.get("started_at"), "orphan grace started_at"),
        deadline_at=_require_str(raw.get("deadline_at"), "orphan grace deadline_at"),
    )


def _orphan_action_from_dict(raw: dict[str, object]) -> OrphanActionReceipt:
    expected = {"generation", "action", "job", "scheduler_state"}
    _reject_unknown_keys(raw, expected, "orphan action receipt")
    job = _job_from_value(raw.get("job"))
    if job is None:
        raise ValueError("orphan action job must be an exact scheduler job")
    return OrphanActionReceipt(
        generation=_require_int(raw.get("generation"), "orphan action generation"),
        action=OrphanReceiptAction(_require_str(raw.get("action"), "orphan action")),
        job=job,
        scheduler_state=_require_str(raw.get("scheduler_state"), "orphan scheduler state"),
    )


def _integration_record_from_dict(raw: dict[str, object]) -> IntegrationRecord:
    expected = {
        "sequence",
        "generation",
        "item_id",
        "previous_commit",
        "new_commit",
        "evidence_kind",
        "evidence_digest",
    }
    _reject_unknown_keys(raw, expected, "integration record")
    return IntegrationRecord(
        sequence=_require_int(raw.get("sequence"), "integration sequence"),
        generation=_require_int(raw.get("generation"), "integration generation"),
        item_id=_require_str(raw.get("item_id"), "integration item_id"),
        previous_commit=_require_str(raw.get("previous_commit"), "integration previous_commit"),
        new_commit=_require_str(raw.get("new_commit"), "integration new_commit"),
        evidence_kind=IntegrationEvidenceKind(
            _require_str(raw.get("evidence_kind"), "integration evidence_kind")
        ),
        evidence_digest=_require_str(raw.get("evidence_digest"), "integration evidence_digest"),
    )


def _work_item_from_dict(raw: dict[str, object]) -> WorkItemRecord:
    expected = {
        "item_id",
        "stage_id",
        "goal_id",
        "kind",
        "profile",
        "status",
        "dependencies",
        "attempts",
        "candidate_attempt_id",
        "candidate_digest",
        "reviewer_attempt_id",
        "terminal_reason",
    }
    _reject_unknown_keys(raw, expected, "work item")
    attempt_values = tuple(
        _require_dict(entry, "attempt") for entry in _require_list(raw.get("attempts"), "attempts")
    )
    _reject_ambiguous_legacy_escalation(attempt_values)
    return WorkItemRecord(
        item_id=_require_str(raw.get("item_id"), "item_id"),
        stage_id=_require_str(raw.get("stage_id"), "stage_id"),
        goal_id=_require_str(raw.get("goal_id"), "goal_id"),
        kind=WorkItemKind(_require_str(raw.get("kind"), "kind")),
        profile=DomainProfile(_require_str(raw.get("profile"), "profile")),
        status=WorkItemStatus(_require_str(raw.get("status"), "status")),
        dependencies=tuple(
            _require_str(entry, "dependency")
            for entry in _require_list(raw.get("dependencies"), "dependencies")
        ),
        attempts=tuple(_attempt_from_dict(entry) for entry in attempt_values),
        candidate_attempt_id=_optional_str(raw.get("candidate_attempt_id"), "candidate_attempt_id"),
        candidate_digest=_optional_str(raw.get("candidate_digest"), "candidate_digest"),
        reviewer_attempt_id=_optional_str(raw.get("reviewer_attempt_id"), "reviewer_attempt_id"),
        terminal_reason=_optional_str(raw.get("terminal_reason"), "terminal_reason"),
    )


def _reject_ambiguous_legacy_escalation(attempts: tuple[dict[str, object], ...]) -> None:
    lineage_keys = {
        "resource_class",
        "predecessor_attempt_id",
        "resource_escalation_request_id",
    }
    ambiguous_counts: dict[tuple[object, ...], int] = {}
    for raw in attempts:
        if lineage_keys & raw.keys():
            continue
        role = Role(_require_str(raw.get("role"), "role"))
        kind = AttemptKind(_require_str(raw.get("kind"), "kind"))
        if _derive_legacy_resource_class(role, kind) is not None:
            continue
        key = (
            role,
            kind,
            raw.get("profile"),
            raw.get("review_of_attempt_id"),
        )
        ambiguous_counts[key] = ambiguous_counts.get(key, 0) + 1
        if ambiguous_counts[key] > 1:
            raise ValueError(
                "ambiguous legacy attempt history cannot safely derive resource escalation lineage"
            )


def _stage_from_dict(raw: dict[str, object]) -> StageRecord:
    expected = {"stage_id", "required_goal_ids", "status", "terminal_reason"}
    _reject_unknown_keys(raw, expected, "Stage")
    return StageRecord(
        stage_id=_require_str(raw.get("stage_id"), "stage_id"),
        required_goal_ids=tuple(
            _require_str(goal_id, "required_goal_id")
            for goal_id in _require_list(raw.get("required_goal_ids"), "required_goal_ids")
        ),
        status=StageStatus(_require_str(raw.get("status"), "Stage status")),
        terminal_reason=_optional_str(raw.get("terminal_reason"), "terminal_reason"),
    )


def _goal_from_dict(raw: dict[str, object]) -> GoalRecord:
    expected = {"goal_id", "stage_id", "required_item_ids", "status", "terminal_reason"}
    _reject_unknown_keys(raw, expected, "Goal")
    return GoalRecord(
        goal_id=_require_str(raw.get("goal_id"), "goal_id"),
        stage_id=_require_str(raw.get("stage_id"), "stage_id"),
        required_item_ids=tuple(
            _require_str(item_id, "required_item_id")
            for item_id in _require_list(raw.get("required_item_ids"), "required_item_ids")
        ),
        status=HierarchyStatus(_require_str(raw.get("status"), "Goal status")),
        terminal_reason=_optional_str(raw.get("terminal_reason"), "terminal_reason"),
    )


def _attempt_from_dict(raw: dict[str, object]) -> AttemptRecord:
    expected = {
        "attempt_id",
        "item_id",
        "sequence",
        "role",
        "kind",
        "generation",
        "status",
        "profile",
        "resource_class",
        "predecessor_attempt_id",
        "resource_escalation_request_id",
        "submission_token",
        "job",
        "scheduler_state",
        "result_digest",
        "candidate_digest",
        "review_of_attempt_id",
        "reviewed_candidate_digest",
        "terminal_reason",
    }
    _reject_unknown_keys(raw, expected, "attempt")
    lineage_keys = {
        "resource_class",
        "predecessor_attempt_id",
        "resource_escalation_request_id",
    }
    present_lineage_keys = lineage_keys & raw.keys()
    if present_lineage_keys and "resource_class" not in raw:
        raise ValueError("attempt lineage cannot be loaded without an explicit resource_class")
    profile = _optional_str(raw.get("profile"), "profile")
    role = Role(_require_str(raw.get("role"), "role"))
    kind = AttemptKind(_require_str(raw.get("kind"), "kind"))
    resource_class = _optional_str(raw.get("resource_class"), "resource_class")
    if not present_lineage_keys:
        resource_class = _derive_legacy_resource_class(role, kind)
    return AttemptRecord(
        attempt_id=_require_str(raw.get("attempt_id"), "attempt_id"),
        item_id=_require_str(raw.get("item_id"), "item_id"),
        sequence=_require_int(raw.get("sequence"), "sequence"),
        role=role,
        kind=kind,
        generation=_require_int(raw.get("generation"), "generation"),
        status=AttemptStatus(_require_str(raw.get("status"), "status")),
        profile=DomainProfile(profile) if profile is not None else None,
        resource_class=resource_class,
        predecessor_attempt_id=_optional_str(
            raw.get("predecessor_attempt_id"), "predecessor_attempt_id"
        ),
        resource_escalation_request_id=_optional_str(
            raw.get("resource_escalation_request_id"),
            "resource_escalation_request_id",
        ),
        submission_token=_optional_str(raw.get("submission_token"), "submission_token"),
        job=_job_from_value(raw.get("job")),
        scheduler_state=_optional_str(raw.get("scheduler_state"), "scheduler_state"),
        result_digest=_optional_str(raw.get("result_digest"), "result_digest"),
        candidate_digest=_optional_str(raw.get("candidate_digest"), "candidate_digest"),
        review_of_attempt_id=_optional_str(raw.get("review_of_attempt_id"), "review_of_attempt_id"),
        reviewed_candidate_digest=_optional_str(
            raw.get("reviewed_candidate_digest"), "reviewed_candidate_digest"
        ),
        terminal_reason=_optional_str(raw.get("terminal_reason"), "terminal_reason"),
    )


def _attempt_adoption_from_dict(raw: dict[str, object]) -> AttemptAdoptionRecord:
    expected = {
        "sequence",
        "generation",
        "attempt_id",
        "attempt_generation",
        "status",
        "submission_token",
        "job",
    }
    _reject_unknown_keys(raw, expected, "attempt adoption")
    return AttemptAdoptionRecord(
        sequence=_require_int(raw.get("sequence"), "attempt adoption sequence"),
        generation=_require_int(raw.get("generation"), "attempt adoption generation"),
        attempt_id=_require_str(raw.get("attempt_id"), "attempt adoption attempt_id"),
        attempt_generation=_require_int(
            raw.get("attempt_generation"), "attempt adoption attempt_generation"
        ),
        status=AttemptStatus(_require_str(raw.get("status"), "attempt adoption status")),
        submission_token=_optional_str(
            raw.get("submission_token"), "attempt adoption submission_token"
        ),
        job=_job_from_value(raw.get("job")),
    )


def _job_to_dict(job: JobReference | None) -> dict[str, object] | None:
    return asdict(job) if job is not None else None


def _job_from_value(value: object) -> JobReference | None:
    if value is None:
        return None
    raw = _require_dict(value, "job")
    _reject_unknown_keys(raw, {"job_id", "cluster", "array_task_id"}, "job")
    array_task_id = raw.get("array_task_id")
    return JobReference(
        job_id=_require_str(raw.get("job_id"), "job_id"),
        cluster=_optional_str(raw.get("cluster"), "cluster"),
        array_task_id=(
            _require_str(array_task_id, "array_task_id") if array_task_id is not None else None
        ),
    )


def _check_transition(
    label: str,
    current: Enum,
    desired: Enum,
    transitions: dict[object, frozenset[object]],
) -> None:
    if desired not in transitions[current]:
        raise InvalidTransitionError(
            f"illegal {label} transition {current.value!r} -> {desired.value!r}"
        )


def _validate_bootstrap_update(
    current: RunState,
    desired: RunState,
    *,
    generation_advancing: bool,
) -> None:
    successor_install = False
    if generation_advancing and current.controller_successors:
        successor = current.controller_successors[-1]
        successor_install = (
            not successor.fenced
            and successor.predecessor_generation == current.generation
            and successor.successor_generation == desired.generation
            and successor.successor_job is not None
            and desired.controller_submission_token == successor.submission_token
            and desired.controller_job == successor.successor_job
        )
    if not successor_install:
        if current.controller_submission_token != desired.controller_submission_token:
            raise StateConflictError("controller submission token is immutable")
        if current.controller_job is not None and desired.controller_job != current.controller_job:
            raise StateConflictError("controller scheduler identity is immutable once recorded")
    elif (
        current.controller_bootstrap_status is not ControllerBootstrapStatus.SUBMITTED
        or desired.controller_bootstrap_status is not ControllerBootstrapStatus.SUBMITTED
    ):
        raise StateConflictError("a successor fence requires submitted controller identities")
    legal = desired.controller_bootstrap_status is current.controller_bootstrap_status or (
        current.controller_bootstrap_status is ControllerBootstrapStatus.SUBMITTING
        and desired.controller_bootstrap_status is ControllerBootstrapStatus.SUBMITTED
        and current.controller_job is None
        and desired.controller_job is not None
    )
    if not legal:
        raise StateConflictError("illegal controller bootstrap status update")


def _validate_state_update(
    current: RunState,
    desired: RunState,
    *,
    generation_advancing: bool = False,
) -> None:
    _validate_bootstrap_update(
        current,
        desired,
        generation_advancing=generation_advancing,
    )
    _validate_controller_recovery_update(
        current,
        desired,
        generation_advancing=generation_advancing,
    )
    _validate_integration_update(
        current,
        desired,
        generation_advancing=generation_advancing,
    )
    _validate_hierarchy_collection_update(current.stages, desired.stages, "Stage")
    _validate_hierarchy_collection_update(current.goals, desired.goals, "Goal")
    current_items = {item.item_id: item for item in current.items}
    desired_items = {item.item_id: item for item in desired.items}
    removed_items = set(current_items) - set(desired_items)
    if removed_items:
        raise StateConflictError(f"work-item history cannot be removed: {sorted(removed_items)!r}")
    for item_id, item in current_items.items():
        _validate_item_update(
            item,
            desired_items[item_id],
            expected_generation=current.generation,
        )
    _validate_attempt_collection_update(
        current.planning_attempts,
        desired.planning_attempts,
        "planning attempt",
        expected_generation=current.generation,
    )
    if current.terminal_status is not RunTerminalStatus.ACTIVE:
        if (
            desired.terminal_status is not current.terminal_status
            or desired.terminal_reason != current.terminal_reason
        ):
            raise StateConflictError("terminal run outcome is immutable")
    elif desired.terminal_status is RunTerminalStatus.ACTIVE:
        if desired.terminal_reason is not None:
            raise StateConflictError("active run cannot gain a terminal reason")


def _validate_integration_update(
    current: RunState,
    desired: RunState,
    *,
    generation_advancing: bool,
) -> None:
    _validate_immutable_prefix(
        current.integration_history,
        desired.integration_history,
        "integration",
    )
    appended = desired.integration_history[len(current.integration_history) :]
    if generation_advancing and appended:
        raise StateConflictError("integration history cannot change during generation advance")
    if len(appended) > 1:
        raise StateConflictError("one state revision may append only one integration record")
    if appended:
        record = appended[0]
        if record.generation != current.generation:
            raise StateConflictError(
                "integration record must belong to the active controller generation"
            )
        if record.previous_commit != current.integration_head:
            raise StateConflictError("integration record does not extend the persisted head")
        if any(entry.item_id == record.item_id for entry in current.integration_history):
            raise StateConflictError("work item already has a persisted integration record")


def _validate_controller_recovery_update(
    current: RunState,
    desired: RunState,
    *,
    generation_advancing: bool,
) -> None:
    _validate_immutable_prefix(
        current.controller_checkpoints,
        desired.controller_checkpoints,
        "controller checkpoint",
    )
    _validate_human_request_update(current.human_requests, desired.human_requests)
    _validate_immutable_prefix(
        current.controller_generation_history,
        desired.controller_generation_history,
        "controller generation",
    )
    _validate_successor_update(current.controller_successors, desired.controller_successors)
    _validate_immutable_prefix(
        current.orphan_grace_windows,
        desired.orphan_grace_windows,
        "orphan grace window",
    )
    _validate_immutable_prefix(
        current.orphan_action_receipts,
        desired.orphan_action_receipts,
        "orphan action receipt",
    )
    _validate_immutable_prefix(
        current.attempt_adoptions,
        desired.attempt_adoptions,
        "attempt adoption",
    )
    new_adoptions = desired.attempt_adoptions[len(current.attempt_adoptions) :]
    if generation_advancing and new_adoptions:
        raise StateConflictError("attempt adoption cannot be recorded during generation advance")
    if len(new_adoptions) > 1:
        raise StateConflictError("one state revision may append only one attempt adoption")
    if new_adoptions:
        record = new_adoptions[0]
        if record.generation != current.generation:
            raise StateConflictError("attempt adoption must belong to the active generation")
        attempts = {
            attempt.attempt_id: attempt
            for attempt in (
                *current.planning_attempts,
                *(entry for item in current.items for entry in item.attempts),
            )
        }
        attempt = attempts.get(record.attempt_id)
        if attempt is None:
            raise StateConflictError("attempt adoption references an unknown attempt")
        if (
            record.attempt_generation != attempt.generation
            or record.status is not attempt.status
            or record.submission_token != attempt.submission_token
            or record.job != attempt.job
        ):
            raise StateConflictError(
                "attempt adoption must match the exact persisted attempt identity"
            )
    _validate_immutable_prefix(
        current.predecessor_reconciliations,
        desired.predecessor_reconciliations,
        "predecessor reconciliation receipt",
    )
    new_reconciliations = desired.predecessor_reconciliations[
        len(current.predecessor_reconciliations) :
    ]
    if generation_advancing:
        pending_successor = (
            current.controller_successors[-1] if current.controller_successors else None
        )
        successor_fence = (
            pending_successor is not None
            and not pending_successor.fenced
            and pending_successor.successor_job is not None
        )
        if successor_fence:
            if new_reconciliations:
                raise StateConflictError(
                    "successor fence cannot claim predecessor terminal reconciliation"
                )
            installed = desired.controller_successors[-1]
            if installed != pending_successor.fence():
                raise StateConflictError(
                    "generation advance must fence the exact pending successor"
                )
            if desired.controller_lifecycle is not ControllerLifecycle.AWAITING_PREDECESSOR:
                raise StateConflictError(
                    "a successor must await predecessor accounting reconciliation"
                )
            if current.controller_job is None or current.controller_submission_token is None:
                raise StateConflictError("predecessor controller identity is incomplete")
            history = current.controller_generation_history
            if not history:
                history = (
                    ControllerGenerationRecord(
                        generation=current.generation,
                        submission_token=current.controller_submission_token,
                        job=current.controller_job,
                    ),
                )
            expected_history = (
                *history,
                ControllerGenerationRecord(
                    generation=installed.successor_generation,
                    submission_token=installed.submission_token,
                    job=installed.successor_job,
                ),
            )
            if desired.controller_generation_history != expected_history:
                raise StateConflictError(
                    "successor fence must append exact controller generation history"
                )
        else:
            if len(new_reconciliations) != 1:
                raise StateConflictError(
                    "legacy generation advance requires one predecessor reconciliation receipt"
                )
            receipt = new_reconciliations[0]
            if (
                receipt.predecessor_generation != current.generation
                or receipt.successor_generation != desired.generation
                or receipt.predecessor_job != current.controller_job
                or receipt.successor_job is not None
            ):
                raise StateConflictError(
                    "predecessor reconciliation receipt does not match the generation fence"
                )
            if desired.controller_lifecycle is not ControllerLifecycle.RUNNING:
                raise StateConflictError("a recovered controller generation must resume as running")
            if desired.controller_generation_history != current.controller_generation_history:
                raise StateConflictError("legacy recovery cannot alter controller job history")
        if any(request.response_digest is None for request in current.human_requests):
            raise StateConflictError("generation cannot advance while human input is unresolved")
    else:
        if desired.controller_generation_history != current.controller_generation_history:
            raise StateConflictError("controller generation history changes only at a fence")
        if new_reconciliations:
            if len(new_reconciliations) != 1:
                raise StateConflictError("only one predecessor receipt may be recorded per update")
            receipt = new_reconciliations[0]
            latest = current.controller_successors[-1] if current.controller_successors else None
            if (
                current.controller_lifecycle is not ControllerLifecycle.AWAITING_PREDECESSOR
                or desired.controller_lifecycle is not ControllerLifecycle.RUNNING
                or latest is None
                or not latest.fenced
                or receipt.predecessor_generation != latest.predecessor_generation
                or receipt.successor_generation != current.generation
                or receipt.predecessor_job != latest.predecessor_job
                or receipt.successor_job != current.controller_job
            ):
                raise StateConflictError(
                    "predecessor receipt must activate the exact fenced successor"
                )
        elif current.controller_lifecycle is not desired.controller_lifecycle:
            _check_transition(
                "controller lifecycle",
                current.controller_lifecycle,
                desired.controller_lifecycle,
                _CONTROLLER_LIFECYCLE_TRANSITIONS,
            )


def _validate_successor_update(
    current: tuple[ControllerSuccessorRecord, ...],
    desired: tuple[ControllerSuccessorRecord, ...],
) -> None:
    if len(desired) < len(current) or len(desired) > len(current) + 1:
        raise StateConflictError("controller successor history is append-only")
    for index, previous in enumerate(current):
        updated = desired[index]
        if previous == updated:
            continue
        if index != len(current) - 1:
            raise StateConflictError("completed controller successor history is immutable")
        immutable = (
            previous.sequence == updated.sequence,
            previous.predecessor_generation == updated.predecessor_generation,
            previous.predecessor_job == updated.predecessor_job,
            previous.successor_generation == updated.successor_generation,
            previous.submission_token == updated.submission_token,
            previous.reason == updated.reason,
            previous.human_request_id == updated.human_request_id,
        )
        if not all(immutable):
            raise StateConflictError("controller successor intent is immutable")
        recorded_job = (
            previous.successor_job is None
            and updated.successor_job is not None
            and not previous.fenced
            and not updated.fenced
        )
        fenced = (
            previous.successor_job is not None
            and updated.successor_job == previous.successor_job
            and not previous.fenced
            and updated.fenced
        )
        if not (recorded_job or fenced):
            raise StateConflictError("successor may only record one exact job and then be fenced")
    if len(desired) == len(current) + 1:
        appended = desired[-1]
        if appended.successor_job is not None or appended.fenced:
            raise StateConflictError("new successor intent cannot already carry a job or fence")


def _validate_immutable_prefix(
    current: tuple[object, ...],
    desired: tuple[object, ...],
    label: str,
) -> None:
    if len(desired) < len(current) or desired[: len(current)] != current:
        raise StateConflictError(f"{label} history is immutable and append-only")


def _validate_human_request_update(
    current: tuple[HumanRequestRecord, ...],
    desired: tuple[HumanRequestRecord, ...],
) -> None:
    if len(desired) < len(current):
        raise StateConflictError("human request history cannot be removed")
    for previous, next_request in zip(current, desired):
        immutable_fields = (
            previous.request_id == next_request.request_id,
            previous.generation == next_request.generation,
            previous.prompt_digest == next_request.prompt_digest,
            previous.controller_requeue == next_request.controller_requeue,
        )
        if not all(immutable_fields):
            raise StateConflictError(
                "human request identity, prompt, and requeue intent are immutable"
            )
        if previous.response_digest is not None and (
            previous.response_digest != next_request.response_digest
        ):
            raise StateConflictError("human response digest is immutable once recorded")
    if tuple(request.request_id for request in desired[: len(current)]) != tuple(
        request.request_id for request in current
    ):
        raise StateConflictError("human request history cannot be reordered")


def _validate_item_update(
    current: WorkItemRecord,
    desired: WorkItemRecord,
    *,
    expected_generation: int,
) -> None:
    immutable_fields = (
        ("stage_id", current.stage_id, desired.stage_id),
        ("goal_id", current.goal_id, desired.goal_id),
        ("kind", current.kind, desired.kind),
        ("profile", current.profile, desired.profile),
        ("dependencies", current.dependencies, desired.dependencies),
    )
    for name, previous, next_value in immutable_fields:
        if previous != next_value:
            raise StateConflictError(f"work-item {name} is immutable")
    if current.status is not desired.status:
        _check_transition("work item", current.status, desired.status, _WORK_ITEM_TRANSITIONS)
    if current.terminal and desired.terminal_reason != current.terminal_reason:
        raise StateConflictError("terminal work-item reason is immutable")
    if current.candidate_attempt_id is not None and (
        desired.candidate_attempt_id != current.candidate_attempt_id
        or desired.candidate_digest != current.candidate_digest
    ):
        raise StateConflictError("frozen work-item candidate linkage is immutable")
    if current.reviewer_attempt_id is not None and (
        desired.reviewer_attempt_id != current.reviewer_attempt_id
    ):
        raise StateConflictError("frozen work-item reviewer linkage is immutable")
    _validate_attempt_collection_update(
        current.attempts,
        desired.attempts,
        "work-item attempt",
        expected_generation=expected_generation,
    )


def _validate_attempt_collection_update(
    current: tuple[AttemptRecord, ...],
    desired: tuple[AttemptRecord, ...],
    label: str,
    *,
    expected_generation: int,
) -> None:
    current_by_id = {attempt.attempt_id: attempt for attempt in current}
    desired_by_id = {attempt.attempt_id: attempt for attempt in desired}
    removed = set(current_by_id) - set(desired_by_id)
    if removed:
        raise StateConflictError(f"{label} history cannot be removed: {sorted(removed)!r}")
    for attempt_id, attempt in current_by_id.items():
        _validate_attempt_update(attempt, desired_by_id[attempt_id])
    current_ids = tuple(attempt.attempt_id for attempt in current)
    desired_prefix = tuple(attempt.attempt_id for attempt in desired[: len(current)])
    if current_ids != desired_prefix:
        raise StateConflictError(f"{label} history cannot be reordered")
    for attempt in desired[len(current) :]:
        if attempt.generation != expected_generation:
            raise StateConflictError(f"new {label} must belong to the active controller generation")
        if attempt.resource_class is None:
            raise StateConflictError(f"new {label} must persist its selected resource_class")


def _validate_hierarchy_collection_update(
    current: tuple[StageRecord, ...] | tuple[GoalRecord, ...],
    desired: tuple[StageRecord, ...] | tuple[GoalRecord, ...],
    label: str,
) -> None:
    def identity(record: StageRecord | GoalRecord) -> str:
        return record.stage_id if isinstance(record, StageRecord) else record.goal_id

    current_by_id = {identity(record): record for record in current}
    desired_by_id = {identity(record): record for record in desired}
    if not current_by_id:
        return
    if set(current_by_id) != set(desired_by_id):
        raise StateConflictError(
            f"{label} identities cannot be added or removed after initialization"
        )
    for record_id, previous in current_by_id.items():
        next_record = desired_by_id[record_id]
        if isinstance(previous, StageRecord) and isinstance(next_record, StageRecord):
            if previous.required_goal_ids != next_record.required_goal_ids:
                raise StateConflictError("Stage required Goals are immutable")
        elif isinstance(previous, GoalRecord) and isinstance(next_record, GoalRecord):
            if (
                previous.stage_id != next_record.stage_id
                or previous.required_item_ids != next_record.required_item_ids
            ):
                raise StateConflictError("Goal ownership and required items are immutable")
        else:
            raise StateConflictError(f"{label} record type changed")
        if previous.status is not next_record.status:
            transitions = (
                _STAGE_TRANSITIONS if isinstance(previous, StageRecord) else _HIERARCHY_TRANSITIONS
            )
            _check_transition(label, previous.status, next_record.status, transitions)


def _validate_attempt_update(current: AttemptRecord, desired: AttemptRecord) -> None:
    immutable_fields = (
        ("attempt_id", current.attempt_id, desired.attempt_id),
        ("item_id", current.item_id, desired.item_id),
        ("sequence", current.sequence, desired.sequence),
        ("role", current.role, desired.role),
        ("kind", current.kind, desired.kind),
        ("profile", current.profile, desired.profile),
        ("resource_class", current.resource_class, desired.resource_class),
        (
            "predecessor_attempt_id",
            current.predecessor_attempt_id,
            desired.predecessor_attempt_id,
        ),
        (
            "resource_escalation_request_id",
            current.resource_escalation_request_id,
            desired.resource_escalation_request_id,
        ),
        ("generation", current.generation, desired.generation),
        ("review_of_attempt_id", current.review_of_attempt_id, desired.review_of_attempt_id),
        (
            "reviewed_candidate_digest",
            current.reviewed_candidate_digest,
            desired.reviewed_candidate_digest,
        ),
    )
    for name, previous, next_value in immutable_fields:
        if previous != next_value:
            raise StateConflictError(f"attempt {name} is immutable")
    if current.status is not desired.status:
        _check_transition("attempt", current.status, desired.status, _ATTEMPT_TRANSITIONS)
    frozen_when_set = (
        ("submission_token", current.submission_token, desired.submission_token),
        ("job", current.job, desired.job),
        ("result_digest", current.result_digest, desired.result_digest),
        ("candidate_digest", current.candidate_digest, desired.candidate_digest),
    )
    for name, previous, next_value in frozen_when_set:
        if previous is not None and previous != next_value:
            raise StateConflictError(f"attempt {name} is immutable once recorded")


def _require_identifier(name: str, value: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or value.strip() != value
        or any(char in value for char in ("/", "\\", "\x00"))
    ):
        raise ValueError(f"{name} must be a non-empty path-safe identifier")


def _require_resource_class(value: str) -> None:
    if _RESOURCE_CLASS_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "resource_class must start with an ASCII letter and contain only "
            "ASCII letters, digits, '.', '_' or '-' (maximum 128 characters)"
        )


def _derive_legacy_resource_class(role: Role, kind: AttemptKind) -> str | None:
    by_kind = {
        AttemptKind.DETERMINISTIC_GATE: "deterministic_gate",
        AttemptKind.REVIEWER_ANALYSIS: "reviewer_analysis",
        AttemptKind.REVIEWER_RERUN: "reviewer_rerun",
        AttemptKind.QA: "reviewer_analysis",
    }
    if kind in by_kind:
        return by_kind[kind]
    if role is Role.PLAN_DRAFTER:
        return "plan_drafter"
    if role is Role.PLAN_REVIEWER:
        return "plan_reviewer"
    return None


def _validate_attempt_lineage(item_id: str, attempts: tuple[AttemptRecord, ...]) -> None:
    by_id = {attempt.attempt_id: attempt for attempt in attempts}
    predecessor_children: dict[str, str] = {}
    introduced_escalations: dict[str, tuple[str, str]] = {}
    logical_fields = (
        "role",
        "kind",
        "profile",
        "review_of_attempt_id",
        "reviewed_candidate_digest",
        "candidate_digest",
    )
    for attempt in attempts:
        predecessor_id = attempt.predecessor_attempt_id
        if predecessor_id is None:
            continue
        predecessor = by_id.get(predecessor_id)
        if predecessor is None:
            raise ValueError(f"attempt {attempt.attempt_id!r} has an unknown predecessor")
        if predecessor.item_id != item_id:
            raise ValueError("attempt predecessor must belong to the same work item")
        if predecessor.sequence >= attempt.sequence:
            raise ValueError("attempt predecessor must be earlier in sequence order")
        if not predecessor.terminal:
            raise ValueError("attempt predecessor must be terminal")
        previous_child = predecessor_children.setdefault(predecessor_id, attempt.attempt_id)
        if previous_child != attempt.attempt_id:
            raise ValueError("an attempt predecessor may have only one child")
        if any(getattr(attempt, name) != getattr(predecessor, name) for name in logical_fields):
            raise ValueError("attempt retry/escalation must preserve immutable logical fields")

        if attempt.resource_escalation_request_id == predecessor.resource_escalation_request_id:
            if attempt.resource_class != predecessor.resource_class:
                raise ValueError("infrastructure retry must preserve selected resource_class")
            continue
        request_id = attempt.resource_escalation_request_id
        if request_id is None:
            raise ValueError("resource class change requires an explicit escalation request")
        if attempt.resource_class == predecessor.resource_class:
            raise ValueError("resource escalation request must select a different resource_class")
        transition = (predecessor.attempt_id, attempt.attempt_id)
        previous_transition = introduced_escalations.setdefault(request_id, transition)
        if previous_transition != transition:
            raise ValueError("resource escalation request cannot introduce multiple transitions")


def _validate_dependency_graph(items: tuple[WorkItemRecord, ...]) -> None:
    dependencies = {item.item_id: set(item.dependencies) for item in items}
    remaining = set(dependencies)
    while remaining:
        ready = {item_id for item_id in remaining if not (dependencies[item_id] & remaining)}
        if not ready:
            raise ValueError(f"work-item dependencies contain a cycle: {sorted(remaining)!r}")
        remaining -= ready


def _require_digest(name: str, value: str | None) -> None:
    if value is None or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_commit(name: str, value: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase 40-character Git commit")


def _require_optional_text(name: str, value: str | None) -> None:
    if value is None or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_str(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, name)


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _reject_unknown_keys(raw: dict[str, object], expected: set[str], label: str) -> None:
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown:
        raise ValueError(f"unknown {label} fields: {sorted(unknown)!r}")
    if missing:
        raise ValueError(f"missing {label} fields: {sorted(missing)!r}")


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    return _require_dict(value, str(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_utc_timestamp(value: str, name: str = "heartbeat_at") -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a timezone")
    return parsed


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    lock_file = _acquire_lock(path, blocking=True)
    try:
        yield
    finally:
        _unlock_and_close(lock_file)


def _acquire_lock(path: Path, *, blocking: bool) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_file = lock_path.open("a+", encoding="utf-8")
    operation = fcntl.LOCK_EX
    if not blocking:
        operation |= fcntl.LOCK_NB
    try:
        fcntl.flock(lock_file.fileno(), operation)
    except BlockingIOError as error:
        lock_file.close()
        raise LeaseConflictError(f"exclusive controller lock is held for {path}") from error
    return lock_file


def _unlock_and_close(lock_file: TextIO) -> None:
    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    lock_file.close()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
