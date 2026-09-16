# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic ready-wave selection for Staircase controller snapshots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .policy import (
    ASSEMBLER_ITEM_KINDS,
    CertifiedClaimCell,
    PolicyViolation,
    WorkItemProposal,
    paths_overlap,
    validate_work_item_proposals,
)

_ACTIVE_ATTEMPT_STATUSES = frozenset(
    {
        "prepared",
        "submitting",
        "submitted",
        "pending",
        "running",
        "terminal_observed",
        "collecting",
    }
)
_KNOWN_ATTEMPT_STATUSES = frozenset(
    {
        "prepared",
        "submitting",
        "submitted",
        "pending",
        "running",
        "terminal_observed",
        "collecting",
        "validated",
        "retryable_failed",
        "failed",
        "preempted",
        "cancelled",
        "lost",
    }
)
_DEPENDENCY_COMPLETE_STATUSES = frozenset({"approved", "integrated"})
_KNOWN_ITEM_STATUSES = frozenset(
    {
        "planned",
        "ready",
        "coding",
        "candidate_ready",
        "reviewing",
        "approved",
        "integrating",
        "integrated",
        "blocked",
        "rejected",
        "cancelled",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True, kw_only=True)
class ResourceCaps:
    """Aggregate controller limits for active and newly selected work."""

    max_jobs: int
    max_nodes: int
    max_gpus: int

    def __post_init__(self) -> None:
        _require_non_negative_int("max_jobs", self.max_jobs)
        _require_non_negative_int("max_nodes", self.max_nodes)
        _require_non_negative_int("max_gpus", self.max_gpus)


@dataclass(frozen=True, kw_only=True)
class AttemptReservation:
    """Immutable resource and lock projection of one scheduler attempt."""

    attempt_id: str
    item_id: str
    status: str | Enum
    jobs: int
    nodes: int
    gpus: int
    path_locks: tuple[str, ...] = ()
    claim_locks: tuple[CertifiedClaimCell, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.attempt_id, str)
            or not _SAFE_ID.fullmatch(self.attempt_id)
            or not isinstance(self.item_id, str)
            or not _SAFE_ID.fullmatch(self.item_id)
        ):
            raise PolicyViolation("attempt and item IDs must be safe non-empty identifiers")
        if _enum_value(self.status) not in _KNOWN_ATTEMPT_STATUSES:
            raise PolicyViolation(f"unknown attempt status {_enum_value(self.status)!r}")
        _require_non_negative_int("jobs", self.jobs)
        _require_non_negative_int("nodes", self.nodes)
        _require_non_negative_int("gpus", self.gpus)
        if self.active and (self.jobs < 1 or self.nodes < 1):
            raise PolicyViolation("an active attempt must reserve at least one job and node")
        for path in self.path_locks:
            paths_overlap(path, path)
        if len(self.path_locks) != len(set(self.path_locks)):
            raise PolicyViolation("an attempt cannot hold duplicate path locks")
        claim_keys = [claim.lock_key for claim in self.claim_locks]
        if len(claim_keys) != len(set(claim_keys)):
            raise PolicyViolation("an attempt cannot hold duplicate claim locks")

    @property
    def active(self) -> bool:
        """Whether this attempt currently consumes dispatch capacity."""
        return _enum_value(self.status) in _ACTIVE_ATTEMPT_STATUSES


def select_ready_wave(
    items: Sequence[WorkItemProposal],
    *,
    item_statuses: Mapping[str, str | Enum],
    attempts: Sequence[AttemptReservation],
    caps: ResourceCaps,
) -> tuple[WorkItemProposal, ...]:
    """Select a stable maximal wave within current locks and resource caps.

    Selection is a pure function of the supplied snapshot. Items are examined
    in lexical ``item_id`` order. A failure or rejection elsewhere does not
    trigger fail-fast behavior; only an explicit dependency on that item can
    keep a candidate out of the wave.

    Args:
        items: Validated plan proposals.
        item_statuses: Authoritative domain status projected by item ID.
        attempts: Scheduler-attempt resource and lock projections.
        caps: Aggregate job, node, and GPU bounds.

    Returns:
        The deterministic tuple of items the controller may reserve next.

    Raises:
        PolicyViolation: If the snapshot is internally inconsistent or an
            Assembler item was marked ready before all inputs were integrated.
    """
    validate_work_item_proposals(items)
    item_ids = {item.item_id for item in items}
    missing_statuses = item_ids - set(item_statuses)
    if missing_statuses:
        raise PolicyViolation(f"missing item statuses for {sorted(missing_statuses)!r}")
    for item_id in item_ids:
        if _enum_value(item_statuses[item_id]) not in _KNOWN_ITEM_STATUSES:
            raise PolicyViolation(
                f"unknown work-item status {_enum_value(item_statuses[item_id])!r} for {item_id!r}"
            )

    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise PolicyViolation("attempt reservations contain duplicate attempt IDs")

    active_attempts = tuple(attempt for attempt in attempts if attempt.active)
    active_item_ids = {attempt.item_id for attempt in active_attempts}
    if len(active_item_ids) != len(active_attempts):
        raise PolicyViolation("a work item cannot have multiple active attempts")
    _validate_active_locks(active_attempts)
    used_jobs = sum(attempt.jobs for attempt in active_attempts)
    used_nodes = sum(attempt.nodes for attempt in active_attempts)
    used_gpus = sum(attempt.gpus for attempt in active_attempts)
    if used_jobs > caps.max_jobs or used_nodes > caps.max_nodes or used_gpus > caps.max_gpus:
        raise PolicyViolation("active attempts already exceed the aggregate resource caps")

    path_locks = [path for attempt in active_attempts for path in attempt.path_locks]
    claim_locks = {claim.lock_key for attempt in active_attempts for claim in attempt.claim_locks}
    selected: list[WorkItemProposal] = []

    for item in sorted(items, key=lambda proposal: proposal.item_id):
        status = _enum_value(item_statuses[item.item_id])
        if status != "ready":
            continue
        dependency_statuses = tuple(
            _enum_value(item_statuses[dependency]) for dependency in item.dependencies
        )
        if item.kind in ASSEMBLER_ITEM_KINDS and any(
            state != "integrated" for state in dependency_statuses
        ):
            raise PolicyViolation(
                f"Assembler item {item.item_id!r} became READY before every dependency "
                "was INTEGRATED"
            )
        if any(state not in _DEPENDENCY_COMPLETE_STATUSES for state in dependency_statuses):
            continue
        if item.item_id in active_item_ids:
            continue
        if _has_path_conflict(item.allowed_paths, path_locks):
            continue
        item_claims = {claim.lock_key for claim in item.certified_claim_cells}
        if item_claims & claim_locks:
            continue

        next_jobs = used_jobs + 1
        next_nodes = used_nodes + item.execution.nodes
        next_gpus = used_gpus + item.execution.total_gpus
        if next_jobs > caps.max_jobs or next_nodes > caps.max_nodes or next_gpus > caps.max_gpus:
            continue

        selected.append(item)
        used_jobs = next_jobs
        used_nodes = next_nodes
        used_gpus = next_gpus
        path_locks.extend(item.allowed_paths)
        claim_locks.update(item_claims)

    return tuple(selected)


def validate_candidate_capacity(
    attempts: Sequence[AttemptReservation],
    candidate: AttemptReservation,
    *,
    caps: ResourceCaps,
) -> None:
    """Validate active reservations plus one exact candidate against caps.

    This is used by direct planners such as resource escalation that do not
    participate in ready-wave selection.  Terminal predecessors release their
    capacity; every controller-owned nonterminal phase remains active.
    """
    attempt_ids = [attempt.attempt_id for attempt in attempts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise PolicyViolation("attempt reservations contain duplicate attempt IDs")
    if candidate.attempt_id in set(attempt_ids):
        raise PolicyViolation("candidate attempt ID is already reserved")
    if not candidate.active or candidate.jobs < 1 or candidate.nodes < 1:
        raise PolicyViolation("candidate must be an active positive reservation")

    active = tuple(attempt for attempt in attempts if attempt.active)
    active_item_ids = [attempt.item_id for attempt in active]
    if len(active_item_ids) != len(set(active_item_ids)):
        raise PolicyViolation("a work item cannot have multiple active attempts")
    if candidate.item_id in set(active_item_ids):
        raise PolicyViolation("candidate item already has an active attempt")

    used = (
        sum(attempt.jobs for attempt in active),
        sum(attempt.nodes for attempt in active),
        sum(attempt.gpus for attempt in active),
    )
    limits = (caps.max_jobs, caps.max_nodes, caps.max_gpus)
    if any(value > limit for value, limit in zip(used, limits, strict=True)):
        raise PolicyViolation("active attempts already exceed the aggregate resource caps")
    combined = (
        used[0] + candidate.jobs,
        used[1] + candidate.nodes,
        used[2] + candidate.gpus,
    )
    exceeded = tuple(
        name
        for name, value, limit in zip(("jobs", "nodes", "gpus"), combined, limits, strict=True)
        if value > limit
    )
    if exceeded:
        raise PolicyViolation(
            f"active attempts plus candidate exceed aggregate resource caps: {exceeded!r}"
        )


def _has_path_conflict(candidate_paths: Sequence[str], locked_paths: Sequence[str]) -> bool:
    return any(
        paths_overlap(candidate, locked) for candidate in candidate_paths for locked in locked_paths
    )


def _validate_active_locks(attempts: Sequence[AttemptReservation]) -> None:
    path_owners: list[tuple[str, str]] = []
    claim_owners: dict[tuple[str, str], str] = {}
    for attempt in attempts:
        for path in attempt.path_locks:
            for locked_path, owner in path_owners:
                if paths_overlap(path, locked_path):
                    raise PolicyViolation(
                        f"active attempts {owner!r} and {attempt.attempt_id!r} "
                        "hold overlapping path locks"
                    )
            path_owners.append((path, attempt.attempt_id))
        for claim in attempt.claim_locks:
            owner = claim_owners.setdefault(claim.lock_key, attempt.attempt_id)
            if owner != attempt.attempt_id:
                raise PolicyViolation(
                    f"active attempts {owner!r} and {attempt.attempt_id!r} hold the same claim lock"
                )


def _enum_value(value: str | Enum) -> str:
    if isinstance(value, Enum):
        enum_value = value.value
        if not isinstance(enum_value, str):
            raise PolicyViolation("status enums must carry string values")
        return enum_value
    if not isinstance(value, str):
        raise PolicyViolation("status values must be strings or string enums")
    return value


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyViolation(f"{name} must be a non-negative integer")
