# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, action-based orchestration for parallel Smith work.

This module contains no scheduler loop.  A controller tick selects a bounded
wave from an immutable snapshot and returns explicit actions plus the next
authoritative state.  Side effects are performed separately, which keeps the
decision path testable and makes crash recovery inspectable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..common.artifacts import (
    INPUT_FILENAME,
    JsonValue,
    WorkerInputManifest,
    load_input_manifest,
    write_input_manifest,
)
from ..common.dispatch import (
    AttemptReservation,
    ResourceCaps,
    select_ready_wave,
    validate_candidate_capacity,
)
from ..common.gitops import CandidateVerification, ControllerGitOps, GitOpsError
from ..common.outcomes import PlanDraftOutcome, PlanReviewDecision, PlanReviewOutcome
from ..common.placement import (
    PLACEMENT_RECEIPT_FILENAME,
    HorizontalPlacementContract,
    PlacementError,
    PlacementExpectation,
    WorkerPlacementReceipt,
    load_worker_placement,
    validate_horizontal_smith_wave,
)
from ..common.policy import (
    CATALOG_ITEM_KINDS,
    INITIAL_CODER_RESOURCE_CLASS,
    PolicyViolation,
    ResourceEscalationRequest,
    WorkItemProposal,
    paths_overlap,
    validate_resource_escalation,
    validate_work_item_proposals,
)
from ..common.slurm import JobIdentity, JobPlacementEvidence, Scheduler, SchedulerError
from ..state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    Role,
    RunState,
    WorkItemRecord,
    WorkItemStatus,
)
from ..task_schema import ResourceClass, SmithConfig


@dataclass(frozen=True)
class SmithAttemptAction:
    """One controller-approved attempt that may be materialized and dispatched."""

    proposal: WorkItemProposal
    attempt: AttemptRecord
    resource: ResourceClass
    worktree: Path
    branch: str
    base_commit: str
    attempt_dir: Path
    manifest: WorkerInputManifest

    def __post_init__(self) -> None:
        if self.proposal.item_id != self.attempt.item_id:
            raise ValueError("attempt action proposal and attempt identities differ")
        if self.manifest.item_id != self.attempt.item_id:
            raise ValueError("attempt action manifest belongs to another item")
        if self.manifest.attempt_id != self.attempt.attempt_id:
            raise ValueError("attempt action manifest belongs to another attempt")
        if self.attempt.resource_class != self.resource.name:
            raise ValueError("attempt action resource differs from durable attempt selection")
        if self.manifest.worktree != str(self.worktree):
            raise ValueError("attempt action manifest does not name its canonical worktree")


@dataclass(frozen=True)
class SmithTick:
    """Pure result of one Smith controller decision tick."""

    state: RunState
    actions: tuple[SmithAttemptAction, ...]


@dataclass(frozen=True)
class MaterializedAttempt:
    """One immutable mailbox and controller-created worktree ready for dispatch."""

    action: SmithAttemptAction
    input_digest: str


@dataclass(frozen=True)
class CandidateSnapshot:
    """Controller-frozen Coder commit used by gates, reviewers, and fan-in."""

    item_id: str
    coder_attempt_id: str
    repository: Path
    base_commit: str
    candidate_commit: str
    candidate_digest: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HorizontalSmithWave:
    """Deterministic cohort derived from the immutable approved plan."""

    wave_id: str
    item_ids: tuple[str, ...]


def derive_horizontal_smith_waves(
    plan: PlanDraftOutcome,
    *,
    smith: SmithConfig,
) -> tuple[HorizontalSmithWave, ...]:
    """Derive recoverable Smith cohorts from dependencies, locks, and caps."""
    catalog = tuple(item for item in plan.items if item.kind in CATALOG_ITEM_KINDS)
    groups: dict[tuple[str, ...], list[WorkItemProposal]] = {}
    for proposal in catalog:
        groups.setdefault(proposal.dependencies, []).append(proposal)
    waves: list[HorizontalSmithWave] = []
    for dependencies in sorted(groups):
        current: list[WorkItemProposal] = []
        jobs = nodes = gpus = 0
        paths: list[str] = []
        claims: set[tuple[str, str]] = set()

        def flush() -> None:
            nonlocal current, jobs, nodes, gpus, paths, claims
            if not current:
                return
            item_ids = tuple(item.item_id for item in current)
            identity = json.dumps(
                {"dependencies": dependencies, "item_ids": item_ids},
                sort_keys=True,
                separators=(",", ":"),
            )
            waves.append(
                HorizontalSmithWave(
                    f"smith-wave-{hashlib.sha256(identity.encode()).hexdigest()[:24]}",
                    item_ids,
                )
            )
            current = []
            jobs = nodes = gpus = 0
            paths = []
            claims = set()

        for proposal in sorted(groups[dependencies], key=lambda item: item.item_id):
            proposal_claims = {claim.lock_key for claim in proposal.certified_claim_cells}
            conflicts = any(
                paths_overlap(candidate, held)
                for candidate in proposal.allowed_paths
                for held in paths
            ) or bool(proposal_claims & claims)
            exceeds = (
                jobs + 1 > smith.max_parallel_items
                or nodes + proposal.execution.nodes > smith.max_nodes_total
                or gpus + proposal.execution.total_gpus > smith.max_gpus_total
            )
            if current and (conflicts or exceeds):
                flush()
            if (
                proposal.execution.nodes > smith.max_nodes_total
                or proposal.execution.total_gpus > smith.max_gpus_total
            ):
                raise PolicyViolation(
                    f"Smith item {proposal.item_id!r} exceeds horizontal wave caps"
                )
            current.append(proposal)
            jobs += 1
            nodes += proposal.execution.nodes
            gpus += proposal.execution.total_gpus
            paths.extend(proposal.allowed_paths)
            claims.update(proposal_claims)
        flush()
    return tuple(waves)


def horizontal_smith_wave_for_item(
    plan: PlanDraftOutcome,
    item_id: str,
    *,
    smith: SmithConfig,
) -> HorizontalSmithWave:
    """Return the one deterministic wave containing a catalog item."""
    matches = tuple(
        wave
        for wave in derive_horizontal_smith_waves(plan, smith=smith)
        if item_id in wave.item_ids
    )
    if len(matches) != 1:
        raise PlacementError(
            f"catalog item {item_id!r} must resolve exactly one horizontal Smith wave"
        )
    return matches[0]


def validate_coder_wave_placement(
    state: RunState,
    attempt_ids: Sequence[str],
    *,
    smith: SmithConfig,
    workspace: Path,
    scheduler: Scheduler,
) -> tuple[WorkerPlacementReceipt, ...]:
    """Validate one controller-selected horizontal Smith cohort fail closed."""
    return validate_smith_role_wave_placement(
        state,
        attempt_ids,
        attempt_kind=AttemptKind.ROLE,
        smith=smith,
        workspace=workspace,
        scheduler=scheduler,
    )


def validate_smith_role_wave_placement(
    state: RunState,
    attempt_ids: Sequence[str],
    *,
    attempt_kind: AttemptKind,
    smith: SmithConfig,
    workspace: Path,
    scheduler: Scheduler,
) -> tuple[WorkerPlacementReceipt, ...]:
    """Validate one Smith role-stage cohort without cross-stage node coupling."""
    contract = HorizontalPlacementContract(
        smith.distinct_nodes_required,
        smith.exclusive,
    )
    if not contract.distinct_nodes_required:
        return ()
    if not attempt_ids or len(set(attempt_ids)) != len(attempt_ids):
        raise PlacementError("Smith placement cohort requires unique attempt identities")
    attempts = {
        attempt.attempt_id: attempt
        for item in state.items
        for attempt in item.attempts
        if attempt.attempt_id in set(attempt_ids)
    }
    if set(attempts) != set(attempt_ids):
        raise PlacementError("Smith placement cohort contains an unknown attempt")
    expected_role = {
        AttemptKind.ROLE: Role.CODER,
        AttemptKind.REVIEWER_ANALYSIS: Role.REVIEWER,
        AttemptKind.REVIEWER_RERUN: Role.REVIEWER,
    }.get(attempt_kind)
    if expected_role is None:
        raise PlacementError("unsupported horizontal Smith placement role stage")
    expectations: list[PlacementExpectation] = []
    receipts: list[WorkerPlacementReceipt] = []
    scheduler_evidence: list[JobPlacementEvidence] = []
    for attempt_id in sorted(attempts):
        attempt = attempts[attempt_id]
        if (
            attempt.role is not expected_role
            or attempt.kind is not attempt_kind
            or attempt.profile is not DomainProfile.SMITH
            or attempt.job is None
            or attempt.submission_token is None
        ):
            raise PlacementError(
                "Smith placement cohort requires dispatched attempts from one exact role stage"
            )
        expectations.append(
            PlacementExpectation(
                state.run_id,
                attempt.item_id,
                attempt.attempt_id,
                state.task_digest,
                attempt.generation,
                attempt.job,
                attempt.submission_token,
            )
        )
        identity = JobIdentity(
            attempt.job.job_id,
            array_task_id=attempt.job.array_task_id,
            cluster=attempt.job.cluster,
        )
        try:
            scheduler_evidence.append(
                scheduler.verify_single_node_placement(identity, attempt.submission_token)
            )
        except SchedulerError as error:
            raise PlacementError(
                f"scheduler placement lookup failed for {attempt.attempt_id!r}: {error}"
            ) from error
        receipt_path = (
            workspace
            / "items"
            / attempt.item_id
            / "attempts"
            / f"{attempt.sequence:04d}"
            / PLACEMENT_RECEIPT_FILENAME
        )
        receipts.append(load_worker_placement(receipt_path))
    return validate_horizontal_smith_wave(
        expectations,
        receipts,
        scheduler_evidence,
        contract,
    )


def plan_coder_wave(
    state: RunState,
    plan: PlanDraftOutcome,
    review: PlanReviewOutcome,
    *,
    smith: SmithConfig,
    workspace: Path,
    allowed_path_roots: Sequence[str],
) -> SmithTick:
    """Select a bounded catalog wave and prepare one isolated Coder per item.

    The caller must persist the returned state before dispatching any returned
    action.  Worktree and mailbox creation is deliberately deferred to
    :func:`materialize_attempts`.
    """
    _require_approved_plan(plan, review)
    resource_names = tuple(resource.name for resource in smith.resource_classes)
    validate_work_item_proposals(
        plan.items,
        allowed_resource_classes=resource_names,
        allowed_path_roots=allowed_path_roots,
    )
    proposals = tuple(item for item in plan.items if item.kind in CATALOG_ITEM_KINDS)
    by_id = {item.item_id: item for item in proposals}
    statuses: dict[str, WorkItemStatus] = {}
    for proposal in plan.items:
        record = state.item(proposal.item_id)
        if proposal.item_id in by_id:
            _validate_smith_record(record, proposal)
            statuses[proposal.item_id] = record.status
        else:
            # Keep non-catalog dependencies visible to the shared dispatcher,
            # but do not let this Smith-only tick select their READY work.
            statuses[proposal.item_id] = (
                WorkItemStatus.PLANNED if record.status is WorkItemStatus.READY else record.status
            )

    reservations = _attempt_reservations(state, by_id, smith)
    selected = tuple(
        proposal
        for proposal in select_ready_wave(
            plan.items,
            item_statuses=statuses,
            attempts=reservations,
            caps=ResourceCaps(
                max_jobs=smith.max_parallel_items,
                max_nodes=smith.max_nodes_total,
                max_gpus=smith.max_gpus_total,
            ),
        )
        if proposal.kind in CATALOG_ITEM_KINDS
    )

    next_state = state
    actions: list[SmithAttemptAction] = []
    for proposal in selected:
        resource = smith.resource_class(proposal.resource_class)
        _validate_coder_shape(proposal, resource)
        item = next_state.item(proposal.item_id)
        attempt = _new_attempt(
            item,
            state=next_state,
            role=Role.CODER,
            kind=AttemptKind.ROLE,
            profile=DomainProfile.SMITH,
            resource_class=resource.name,
        )
        item = item.add_attempt(attempt).transition(WorkItemStatus.CODING)
        next_state = next_state.replace_item(item)
        actions.append(
            _attempt_action(
                next_state,
                proposal,
                attempt,
                resource,
                workspace=workspace,
                base_commit=state.base_commit,
            )
        )
    return SmithTick(next_state, tuple(actions))


def materialize_attempts(
    actions: Sequence[SmithAttemptAction],
    *,
    git: ControllerGitOps,
) -> tuple[MaterializedAttempt, ...]:
    """Create metadata-free candidate overlays and publish immutable worker inputs.

    Existing overlay directories and matching manifests are adopted after a
    controller restart. A mismatched artifact fails closed. Candidate identity
    is established later when the controller binds the overlay under its Git
    lock; workers never receive shared Git metadata.
    """
    materialized: list[MaterializedAttempt] = []
    with git.transaction() as transaction:
        for action in actions:
            if action.worktree.exists():
                if not action.worktree.is_dir() or action.worktree.is_symlink():
                    raise GitOpsError(
                        f"existing candidate overlay is invalid for action "
                        f"{action.attempt.attempt_id!r}"
                    )
            else:
                transaction.create_candidate_overlay(
                    action.worktree,
                    base_commit=action.base_commit,
                )
            input_path = action.attempt_dir / INPUT_FILENAME
            if input_path.exists():
                existing, input_digest = load_input_manifest(input_path)
                if existing != action.manifest:
                    raise ValueError(
                        f"immutable input differs for attempt {action.attempt.attempt_id!r}"
                    )
            else:
                input_digest = write_input_manifest(input_path, action.manifest)
            materialized.append(MaterializedAttempt(action, input_digest))
    return tuple(materialized)


def freeze_coder_candidate(
    action: SmithAttemptAction,
    *,
    git: ControllerGitOps,
    changed_paths: Sequence[str],
    scanned_overlay_digest: str,
    expected_patch_sha256: str,
    commit_message: str,
) -> CandidateSnapshot:
    """Bind and verify one metadata-free Coder overlay under controller authority."""
    if action.attempt.role is not Role.CODER:
        raise ValueError("only a Coder action can produce a Smith candidate")
    changed = tuple(sorted(changed_paths))
    if not changed or not set(changed).issubset(action.proposal.allowed_paths):
        raise PolicyViolation("Coder changed paths must be a non-empty subset of its path claim")
    workspace = action.worktree.parents[2]
    repository = (
        workspace / "controller-candidates" / action.proposal.item_id / action.attempt.attempt_id
    )
    with git.transaction() as transaction:
        binding = transaction.bind_candidate_overlay(
            action.worktree,
            repository=repository,
            branch=action.branch,
            base_commit=action.base_commit,
            expected_paths=changed,
            expected_overlay_sha256=scanned_overlay_digest,
            message=commit_message,
        )
    verified = git.verify_candidate(
        repository,
        base_commit=action.base_commit,
        candidate_commit=binding.candidate_commit,
        allowed_paths=action.proposal.allowed_paths,
        expected_patch_sha256=expected_patch_sha256,
    )
    return _candidate_snapshot(action, verified, repository=repository)


def plan_gate_attempt(
    state: RunState,
    proposal: WorkItemProposal,
    candidate: CandidateSnapshot,
    *,
    smith: SmithConfig,
    workspace: Path,
) -> SmithTick:
    """Freeze a validated Coder candidate and prepare a separate gate attempt."""
    item = state.item(proposal.item_id)
    _validate_smith_record(item, proposal)
    if item.status is not WorkItemStatus.CODING:
        raise ValueError("a deterministic gate may only follow CODING")
    coder = _attempt(item, candidate.coder_attempt_id)
    if coder.role is not Role.CODER or coder.status is not AttemptStatus.VALIDATED:
        raise ValueError("candidate must come from a validated Coder attempt")
    if coder.candidate_digest != candidate.candidate_digest:
        raise ValueError("Coder state does not match the frozen candidate digest")
    resource = smith.resource_class("deterministic_gate")
    gate = _new_attempt(
        item,
        state=state,
        role=Role.GATE,
        kind=AttemptKind.DETERMINISTIC_GATE,
        profile=DomainProfile.SMITH,
        resource_class=resource.name,
    )
    item = item.transition(
        WorkItemStatus.CANDIDATE_READY,
        candidate_attempt_id=coder.attempt_id,
        candidate_digest=candidate.candidate_digest,
    ).add_attempt(gate)
    next_state = state.replace_item(item)
    action = _attempt_action(
        next_state,
        proposal,
        gate,
        resource,
        workspace=workspace,
        base_commit=candidate.candidate_commit,
        candidate=candidate,
    )
    return SmithTick(next_state, (action,))


def plan_reviewer_analysis_attempt(
    state: RunState,
    proposal: WorkItemProposal,
    candidate: CandidateSnapshot,
    *,
    gate_attempt_id: str,
    smith: SmithConfig,
    workspace: Path,
) -> SmithTick:
    """Prepare a fresh Reviewer analysis after the deterministic gate passes."""
    item = state.item(proposal.item_id)
    _validate_candidate_item(item, proposal, candidate, WorkItemStatus.CANDIDATE_READY)
    gate = _attempt(item, gate_attempt_id)
    if (
        gate.kind is not AttemptKind.DETERMINISTIC_GATE
        or gate.status is not AttemptStatus.VALIDATED
    ):
        raise ValueError("Reviewer analysis requires a separately validated deterministic gate")
    resource = smith.resource_class("reviewer_analysis")
    reviewer = _new_attempt(
        item,
        state=state,
        role=Role.REVIEWER,
        kind=AttemptKind.REVIEWER_ANALYSIS,
        profile=DomainProfile.SMITH,
        resource_class=resource.name,
        review_of_attempt_id=candidate.coder_attempt_id,
        reviewed_candidate_digest=candidate.candidate_digest,
    )
    item = item.transition(WorkItemStatus.REVIEWING).add_attempt(reviewer)
    next_state = state.replace_item(item)
    action = _attempt_action(
        next_state,
        proposal,
        reviewer,
        resource,
        workspace=workspace,
        base_commit=candidate.candidate_commit,
        candidate=candidate,
    )
    return SmithTick(next_state, (action,))


def plan_reviewer_rerun_attempt(
    state: RunState,
    proposal: WorkItemProposal,
    candidate: CandidateSnapshot,
    *,
    analysis_attempt_id: str,
    smith: SmithConfig,
    workspace: Path,
) -> SmithTick:
    """Prepare a second fresh attempt that reruns the frozen candidate evidence."""
    item = state.item(proposal.item_id)
    _validate_candidate_item(item, proposal, candidate, WorkItemStatus.REVIEWING)
    analysis = _attempt(item, analysis_attempt_id)
    _validate_reviewer_attempt(analysis, candidate, AttemptKind.REVIEWER_ANALYSIS)
    if analysis.status is not AttemptStatus.VALIDATED:
        raise ValueError("Reviewer rerun requires validated Reviewer analysis")
    resource = smith.resource_class("reviewer_rerun")
    rerun = _new_attempt(
        item,
        state=state,
        role=Role.REVIEWER,
        kind=AttemptKind.REVIEWER_RERUN,
        profile=DomainProfile.SMITH,
        resource_class=resource.name,
        review_of_attempt_id=candidate.coder_attempt_id,
        reviewed_candidate_digest=candidate.candidate_digest,
    )
    item = item.add_attempt(rerun)
    next_state = state.replace_item(item)
    action = _attempt_action(
        next_state,
        proposal,
        rerun,
        resource,
        workspace=workspace,
        base_commit=candidate.candidate_commit,
        candidate=candidate,
    )
    return SmithTick(next_state, (action,))


def approve_reviewer_rerun(
    state: RunState,
    *,
    item_id: str,
    reviewer_attempt_id: str,
) -> RunState:
    """Approve an item only from a distinct, validated, digest-pinned rerun."""
    item = state.item(item_id)
    if item.status is not WorkItemStatus.REVIEWING or item.candidate_attempt_id is None:
        raise ValueError("only a reviewing frozen candidate can be approved")
    reviewer = _attempt(item, reviewer_attempt_id)
    if reviewer.kind is not AttemptKind.REVIEWER_RERUN:
        raise ValueError("final approval requires a Reviewer rerun attempt")
    if reviewer.status is not AttemptStatus.VALIDATED:
        raise ValueError("final Reviewer rerun has not validated")
    if (
        reviewer.review_of_attempt_id != item.candidate_attempt_id
        or reviewer.reviewed_candidate_digest != item.candidate_digest
    ):
        raise ValueError("final Reviewer rerun is not pinned to the frozen candidate")
    _validate_reviewer_chain(item, reviewer)
    return state.replace_item(
        item.transition(
            WorkItemStatus.APPROVED,
            reviewer_attempt_id=reviewer_attempt_id,
        )
    )


def plan_resource_escalation_attempt(
    state: RunState,
    proposal: WorkItemProposal,
    request: ResourceEscalationRequest,
    *,
    smith: SmithConfig,
    workspace: Path,
) -> SmithTick:
    """Validate a bounded escalation and prepare a new individual Coder attempt."""
    item = state.item(proposal.item_id)
    _validate_smith_record(item, proposal)
    if item.status is not WorkItemStatus.CODING:
        raise ValueError("resource escalation is only valid while CODING")
    source = _attempt(item, request.attempt_id)
    if (
        source.role is not Role.CODER
        or source.kind is not AttemptKind.ROLE
        or source.profile is not DomainProfile.SMITH
        or source.status is not AttemptStatus.VALIDATED
    ):
        raise ValueError("resource escalation must follow a validated Coder result")
    coder_attempts = tuple(attempt for attempt in item.attempts if attempt.role is Role.CODER)
    consumed_request_ids = tuple(
        attempt.resource_escalation_request_id
        for attempt in item.attempts
        if attempt.resource_escalation_request_id is not None
    )
    if request.request_id in consumed_request_ids:
        raise PolicyViolation("resource escalation request was already consumed")
    if not coder_attempts or source.sequence != max(attempt.sequence for attempt in coder_attempts):
        raise PolicyViolation("resource escalation source is not the latest Coder attempt")
    if source.resource_class is None:
        raise PolicyViolation("resource escalation source has no durable selected resource")
    resource_history = _resource_class_lineage(item, source)
    validate_resource_escalation(
        request,
        proposal,
        current_attempt_id=source.attempt_id,
        current_resource_class=source.resource_class,
        current_role=source.role,
        resource_class_history=resource_history,
        consumed_request_ids=consumed_request_ids,
        allowed_resource_classes=tuple(resource.name for resource in smith.resource_classes),
    )
    resource = smith.resource_class(request.requested_resource_class)
    _validate_per_item_override_bounds(resource, smith)
    sequence = len(item.attempts) + 1
    attempt_id = _attempt_id(item.item_id, sequence, AttemptKind.ROLE)
    candidate_reservation = AttemptReservation(
        attempt_id=attempt_id,
        item_id=item.item_id,
        status=AttemptStatus.PREPARED,
        jobs=1,
        nodes=resource.nodes,
        gpus=resource.total_gpus,
        path_locks=proposal.allowed_paths,
        claim_locks=proposal.certified_claim_cells,
    )
    validate_candidate_capacity(
        _all_smith_resource_reservations(state, smith),
        candidate_reservation,
        caps=ResourceCaps(
            max_jobs=smith.max_parallel_items,
            max_nodes=smith.max_nodes_total,
            max_gpus=smith.max_gpus_total,
        ),
    )
    attempt = _new_attempt(
        item,
        state=state,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        profile=DomainProfile.SMITH,
        resource_class=resource.name,
        predecessor_attempt_id=source.attempt_id,
        resource_escalation_request_id=request.request_id,
    )
    item = item.add_attempt(attempt)
    next_state = state.replace_item(item)
    action = _attempt_action(
        next_state,
        proposal,
        attempt,
        resource,
        workspace=workspace,
        base_commit=state.base_commit,
        extra_payload={
            "resource_escalation_request_id": request.request_id,
            "resource_escalation_reason": request.reason,
        },
    )
    return SmithTick(next_state, (action,))


def mark_ready_after_integrated_dependencies(state: RunState, item_id: str) -> RunState:
    """Move a planned item to READY only after every dependency is INTEGRATED."""
    item = state.item(item_id)
    if item.status is not WorkItemStatus.PLANNED:
        raise ValueError("only a planned item can become ready")
    if any(
        state.item(dependency).status is not WorkItemStatus.INTEGRATED
        for dependency in item.dependencies
    ):
        raise ValueError("all dependencies must be INTEGRATED before an item becomes READY")
    return state.replace_item(item.transition(WorkItemStatus.READY))


def _require_approved_plan(plan: PlanDraftOutcome, review: PlanReviewOutcome) -> None:
    if review.outcome is not PlanReviewDecision.ACCEPT:
        raise PolicyViolation("Smith dispatch requires an approved PlanReviewer verdict")
    if review.plan_digest != plan.digest:
        raise PolicyViolation("PlanReviewer verdict does not match the frozen plan digest")


def _validate_smith_record(item: WorkItemRecord, proposal: WorkItemProposal) -> None:
    if item.item_id != proposal.item_id or item.goal_id != proposal.goal_id:
        raise PolicyViolation("work-item state does not match the reviewed proposal")
    if item.profile is not DomainProfile.SMITH:
        raise PolicyViolation("catalog work must use the Smith domain profile")
    if item.kind.value != proposal.kind.value or proposal.kind not in CATALOG_ITEM_KINDS:
        raise PolicyViolation("Smith orchestration accepts only matching catalog work items")
    if item.dependencies != proposal.dependencies:
        raise PolicyViolation("work-item dependencies changed after plan review")


def _validate_coder_shape(proposal: WorkItemProposal, resource: ResourceClass) -> None:
    if resource.name != INITIAL_CODER_RESOURCE_CLASS:
        raise PolicyViolation(f"initial Smith Coder must use {INITIAL_CODER_RESOURCE_CLASS!r}")
    if (
        proposal.execution.nodes != resource.nodes
        or proposal.execution.ranks_per_node != resource.tasks_per_node
        or proposal.execution.gpus_per_node != resource.gpus_per_node
    ):
        raise PolicyViolation("Coder execution shape must exactly match its resource class")
    if proposal.modifies_files and proposal.execution.array_element:
        raise PolicyViolation("file-modifying Smith Coder work must use an individual job")


def _attempt_reservations(
    state: RunState,
    proposals: Mapping[str, WorkItemProposal],
    smith: SmithConfig,
) -> tuple[AttemptReservation, ...]:
    reservations: list[AttemptReservation] = []
    for item in state.items:
        proposal = proposals.get(item.item_id)
        if proposal is None:
            continue
        for attempt in item.attempts:
            if attempt.resource_class is None:
                raise PolicyViolation(
                    f"attempt {attempt.attempt_id!r} has no durable selected resource"
                )
            resource = _resource_class(smith, attempt.resource_class)
            reservations.append(
                AttemptReservation(
                    attempt_id=attempt.attempt_id,
                    item_id=item.item_id,
                    status=attempt.status,
                    jobs=1,
                    nodes=resource.nodes,
                    gpus=resource.total_gpus,
                    path_locks=proposal.allowed_paths,
                    claim_locks=proposal.certified_claim_cells,
                )
            )
    return tuple(reservations)


def _new_attempt(
    item: WorkItemRecord,
    *,
    state: RunState,
    role: Role,
    kind: AttemptKind,
    profile: DomainProfile,
    resource_class: str,
    predecessor_attempt_id: str | None = None,
    resource_escalation_request_id: str | None = None,
    review_of_attempt_id: str | None = None,
    reviewed_candidate_digest: str | None = None,
) -> AttemptRecord:
    sequence = len(item.attempts) + 1
    attempt_id = _attempt_id(item.item_id, sequence, kind)
    return AttemptRecord(
        attempt_id=attempt_id,
        item_id=item.item_id,
        sequence=sequence,
        role=role,
        kind=kind,
        generation=state.generation,
        profile=profile,
        resource_class=resource_class,
        predecessor_attempt_id=predecessor_attempt_id,
        resource_escalation_request_id=resource_escalation_request_id,
        review_of_attempt_id=review_of_attempt_id,
        reviewed_candidate_digest=reviewed_candidate_digest,
    )


def _attempt_id(item_id: str, sequence: int, kind: AttemptKind) -> str:
    suffix = f".{sequence:04d}.{kind.value}"
    candidate = f"{item_id}{suffix}"
    if len(candidate) <= 128:
        return candidate
    digest = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:16]
    prefix_length = 128 - len(suffix) - len(digest) - 1
    return f"{item_id[:prefix_length]}.{digest}{suffix}"


def _attempt_action(
    state: RunState,
    proposal: WorkItemProposal,
    attempt: AttemptRecord,
    resource: ResourceClass,
    *,
    workspace: Path,
    base_commit: str,
    candidate: CandidateSnapshot | None = None,
    extra_payload: Mapping[str, JsonValue] | None = None,
) -> SmithAttemptAction:
    root = workspace.expanduser().resolve()
    worktree = root / "candidates" / proposal.item_id / attempt.attempt_id
    attempt_dir = root / "items" / proposal.item_id / "attempts" / f"{attempt.sequence:04d}"
    payload: dict[str, JsonValue] = {
        "attempt_kind": attempt.kind.value,
        "entry_id": proposal.entry_id,
        "work_item_kind": proposal.kind.value,
        "resource_class": resource.name,
        "predecessor_attempt_id": attempt.predecessor_attempt_id,
        "resource_escalation_request_id": attempt.resource_escalation_request_id,
        "execution": {
            "nodes": resource.nodes,
            "tasks_per_node": resource.tasks_per_node,
            "gpus_per_node": resource.gpus_per_node,
        },
        "dependencies": list(proposal.dependencies),
        "certified_claim_cells": [
            {"entry_id": claim.entry_id, "cell_id": claim.cell_id}
            for claim in proposal.certified_claim_cells
        ],
    }
    if candidate is not None:
        payload.update(
            {
                "candidate_attempt_id": candidate.coder_attempt_id,
                "candidate_commit": candidate.candidate_commit,
                "candidate_digest": candidate.candidate_digest,
            }
        )
    if extra_payload:
        payload.update(extra_payload)
    manifest = WorkerInputManifest(
        run_id=state.run_id,
        item_id=proposal.item_id,
        attempt_id=attempt.attempt_id,
        task_digest=state.task_digest,
        generation=state.generation,
        role=attempt.role,
        profile=DomainProfile.SMITH,
        worktree=str(worktree),
        allowed_paths=proposal.allowed_paths if attempt.role is Role.CODER else (),
        payload=payload,
    )
    return SmithAttemptAction(
        proposal=proposal,
        attempt=attempt,
        resource=resource,
        worktree=worktree,
        branch=f"staircase/{state.run_id}/{attempt.attempt_id}",
        base_commit=base_commit,
        attempt_dir=attempt_dir,
        manifest=manifest,
    )


def _candidate_snapshot(
    action: SmithAttemptAction,
    verification: CandidateVerification,
    *,
    repository: Path,
) -> CandidateSnapshot:
    return CandidateSnapshot(
        item_id=action.proposal.item_id,
        coder_attempt_id=action.attempt.attempt_id,
        repository=repository,
        base_commit=verification.base_commit,
        candidate_commit=verification.candidate_commit,
        candidate_digest=verification.patch_sha256,
        changed_paths=verification.changed_paths,
    )


def _validate_candidate_item(
    item: WorkItemRecord,
    proposal: WorkItemProposal,
    candidate: CandidateSnapshot,
    required_status: WorkItemStatus,
) -> None:
    _validate_smith_record(item, proposal)
    if item.status is not required_status:
        raise ValueError(f"item must be {required_status.value} for this Reviewer transition")
    if (
        candidate.item_id != item.item_id
        or candidate.coder_attempt_id != item.candidate_attempt_id
        or candidate.candidate_digest != item.candidate_digest
    ):
        raise ValueError("Reviewer candidate snapshot does not match authoritative state")


def _validate_reviewer_attempt(
    attempt: AttemptRecord,
    candidate: CandidateSnapshot,
    kind: AttemptKind,
) -> None:
    if (
        attempt.kind is not kind
        or attempt.role is not Role.REVIEWER
        or attempt.review_of_attempt_id != candidate.coder_attempt_id
        or attempt.reviewed_candidate_digest != candidate.candidate_digest
    ):
        raise ValueError("Reviewer attempt is not pinned to the frozen candidate")


def _validate_reviewer_chain(item: WorkItemRecord, rerun: AttemptRecord) -> None:
    """Require distinct validated Coder, gate, analysis, and rerun attempts."""
    if item.candidate_attempt_id is None:
        raise ValueError("Reviewer chain requires a frozen Coder attempt")
    coder = _attempt(item, item.candidate_attempt_id)
    gates = tuple(
        attempt
        for attempt in item.attempts
        if coder.sequence < attempt.sequence < rerun.sequence
        and attempt.kind is AttemptKind.DETERMINISTIC_GATE
        and attempt.status is AttemptStatus.VALIDATED
    )
    analyses = tuple(
        attempt
        for attempt in item.attempts
        if coder.sequence < attempt.sequence < rerun.sequence
        and attempt.kind is AttemptKind.REVIEWER_ANALYSIS
        and attempt.status is AttemptStatus.VALIDATED
        and attempt.review_of_attempt_id == coder.attempt_id
        and attempt.reviewed_candidate_digest == item.candidate_digest
    )
    if not gates or not analyses:
        raise ValueError("approval requires validated gate and Reviewer analysis attempts")
    chain = (coder, gates[-1], analyses[-1], rerun)
    scheduler_ids = tuple(attempt.job.scheduler_id for attempt in chain if attempt.job is not None)
    if len(scheduler_ids) != len(chain) or len(scheduler_ids) != len(set(scheduler_ids)):
        raise ValueError(
            "Coder, deterministic gate, Reviewer analysis, and Reviewer rerun "
            "must use distinct scheduler jobs"
        )


def _validate_per_item_override_bounds(resource: ResourceClass, smith: SmithConfig) -> None:
    bounds = smith.per_item_override_bounds
    checks = (
        ("nodes", resource.nodes, bounds.max_nodes),
        ("tasks_per_node", resource.tasks_per_node, bounds.max_tasks_per_node),
        ("gpus_per_node", resource.gpus_per_node, bounds.max_gpus_per_node),
        ("cpus_per_task", resource.cpus_per_task, bounds.max_cpus_per_task),
        ("memory_mib", resource.memory_mib, bounds.max_memory_mib),
        ("time_limit_seconds", resource.time_limit_seconds, bounds.max_time_limit_seconds),
    )
    exceeded = [name for name, value, limit in checks if value > limit]
    if exceeded:
        raise PolicyViolation(f"resource escalation exceeds per-item bounds: {exceeded}")


def _all_smith_resource_reservations(
    state: RunState, smith: SmithConfig
) -> tuple[AttemptReservation, ...]:
    reservations: list[AttemptReservation] = []
    for item in state.items:
        if item.profile is not DomainProfile.SMITH:
            continue
        for attempt in item.attempts:
            if attempt.resource_class is None:
                raise PolicyViolation(
                    f"attempt {attempt.attempt_id!r} has no durable selected resource"
                )
            resource = _resource_class(smith, attempt.resource_class)
            reservations.append(
                AttemptReservation(
                    attempt_id=attempt.attempt_id,
                    item_id=item.item_id,
                    status=attempt.status,
                    jobs=1,
                    nodes=resource.nodes,
                    gpus=resource.total_gpus,
                )
            )
    return tuple(reservations)


def _resource_class(smith: SmithConfig, name: str) -> ResourceClass:
    try:
        return smith.resource_class(name)
    except KeyError as error:
        raise PolicyViolation(f"attempt selects unknown resource class {name!r}") from error


def _resource_class_lineage(item: WorkItemRecord, source: AttemptRecord) -> tuple[str, ...]:
    by_id = {attempt.attempt_id: attempt for attempt in item.attempts}
    lineage: list[str] = []
    seen: set[str] = set()
    current = source
    while True:
        if current.attempt_id in seen:
            raise PolicyViolation("attempt predecessor lineage contains a cycle")
        seen.add(current.attempt_id)
        if current.resource_class is None:
            raise PolicyViolation("attempt predecessor lineage lacks a selected resource")
        if not lineage or lineage[-1] != current.resource_class:
            lineage.append(current.resource_class)
        predecessor_id = current.predecessor_attempt_id
        if predecessor_id is None:
            break
        try:
            predecessor = by_id[predecessor_id]
        except KeyError as error:
            raise PolicyViolation(
                "attempt predecessor lineage references an unknown attempt"
            ) from error
        if predecessor.sequence >= current.sequence:
            raise PolicyViolation("attempt predecessor lineage is not strictly ordered")
        current = predecessor
    return tuple(reversed(lineage))


def _attempt(item: WorkItemRecord, attempt_id: str) -> AttemptRecord:
    for attempt in item.attempts:
        if attempt.attempt_id == attempt_id:
            return attempt
    raise KeyError(attempt_id)
