# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serial, deterministic fan-in of independently reviewed Smith candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..common.gitops import ControllerGitOps, GitConflictError, GitOpsError
from ..common.index import IndexDelta, IndexUpdate, update_catalog_index
from ..common.outcomes import PlanDraftOutcome, PlanReviewOutcome
from ..common.policy import CATALOG_ITEM_KINDS, WorkItemKind, WorkItemProposal
from ..state import AttemptKind, AttemptStatus, RunState, WorkItemStatus
from .smith import CandidateSnapshot, _require_approved_plan, _validate_reviewer_chain


class FanInError(RuntimeError):
    """Raised when fan-in cannot safely finish after preflight."""

    def __init__(self, message: str, *, integrated_item_ids: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.integrated_item_ids = integrated_item_ids


@dataclass(frozen=True)
class ApprovedSmithCandidate:
    """Reviewed catalog item plus its frozen commit and semantic index delta."""

    proposal: WorkItemProposal
    snapshot: CandidateSnapshot | None
    index_delta: IndexDelta | None

    def __post_init__(self) -> None:
        if self.proposal.kind not in CATALOG_ITEM_KINDS:
            raise ValueError("fan-in accepts catalog work only")
        if self.proposal.modifies_files:
            if self.snapshot is None:
                raise ValueError("file-modifying fan-in requires a frozen candidate commit")
            if self.proposal.kind is WorkItemKind.CATALOG_ONBOARD and self.index_delta is None:
                raise ValueError("catalog onboarding requires a typed IndexDelta")
        elif self.snapshot is not None or self.index_delta is not None:
            raise ValueError("read-only catalog verification cannot mutate Git or the index")
        if self.snapshot is not None and self.snapshot.item_id != self.proposal.item_id:
            raise ValueError("candidate snapshot belongs to another item")
        if self.index_delta is not None:
            if self.index_delta.item_id != self.proposal.item_id:
                raise ValueError("IndexDelta belongs to another item")
            if self.index_delta.row.entry_id != self.proposal.entry_id:
                raise ValueError("IndexDelta entry does not match the atomic catalog item")


@dataclass(frozen=True)
class AggregateValidator:
    """Named deterministic validation executed under the controller Git lock."""

    name: str
    validate: Callable[[Path], None]

    def __post_init__(self) -> None:
        if not self.name.strip() or "\n" in self.name:
            raise ValueError("aggregate validator name must be a non-empty single line")


@dataclass(frozen=True)
class FanInBatch:
    """Stable controller action containing only state-approved Smith items."""

    candidates: tuple[ApprovedSmithCandidate, ...]


@dataclass(frozen=True)
class FanInItemResult:
    """Independent integration result for one approved sibling."""

    item_id: str
    integrated: bool
    integrated_commit: str | None
    reason: str | None = None


@dataclass(frozen=True)
class FanInResult:
    """Auditable result of one serial fan-in transaction."""

    items: tuple[FanInItemResult, ...]
    final_head: str
    index_update: IndexUpdate | None
    aggregate_validations: tuple[str, ...]

    @property
    def integrated_item_ids(self) -> tuple[str, ...]:
        """Return successful items in their deterministic integration order."""
        return tuple(result.item_id for result in self.items if result.integrated)


def plan_fan_in(
    state: RunState,
    plan: PlanDraftOutcome,
    review: PlanReviewOutcome,
    candidates: Sequence[ApprovedSmithCandidate],
) -> FanInBatch:
    """Validate Reviewer linkage and return stable item-ID fan-in actions."""
    _require_approved_plan(plan, review)
    proposals = {proposal.item_id: proposal for proposal in plan.items}
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.proposal.item_id))
    item_ids = [candidate.proposal.item_id for candidate in ordered]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("fan-in batch contains duplicate item IDs")

    for candidate in ordered:
        item_id = candidate.proposal.item_id
        if proposals.get(item_id) != candidate.proposal:
            raise ValueError(f"candidate {item_id!r} differs from the reviewed plan")
        item = state.item(item_id)
        if item.status is not WorkItemStatus.APPROVED:
            raise ValueError(f"candidate {item_id!r} has not reached APPROVED")
        if item.reviewer_attempt_id is None or item.candidate_attempt_id is None:
            raise ValueError(f"candidate {item_id!r} lacks frozen Reviewer linkage")
        reviewer = next(
            attempt for attempt in item.attempts if attempt.attempt_id == item.reviewer_attempt_id
        )
        if (
            reviewer.kind is not AttemptKind.REVIEWER_RERUN
            or reviewer.status is not AttemptStatus.VALIDATED
            or reviewer.review_of_attempt_id != item.candidate_attempt_id
            or reviewer.reviewed_candidate_digest != item.candidate_digest
        ):
            raise ValueError(f"candidate {item_id!r} lacks a valid fresh Reviewer rerun")
        _validate_reviewer_chain(item, reviewer)
        if candidate.snapshot is not None and (
            candidate.snapshot.coder_attempt_id != item.candidate_attempt_id
            or candidate.snapshot.candidate_digest != item.candidate_digest
        ):
            raise ValueError(f"candidate {item_id!r} snapshot differs from reviewed state")
    return FanInBatch(ordered)


def execute_fan_in(
    batch: FanInBatch,
    *,
    git: ControllerGitOps,
    index_path: Path,
    catalog_root: Path,
    validators: Sequence[AggregateValidator] = (),
    index_commit_message: str = "agent-flow: integrate reviewed Smith catalog index deltas",
) -> FanInResult:
    """Integrate approved siblings and their index rows under one Git lock.

    Candidate conflicts are isolated to the candidate that conflicts; later
    siblings still run.  Structural Git or index failures fail closed because
    they make safe continuation ambiguous.  This operation never resolves a
    conflict, merges, pushes, removes a worktree, or rewrites history.
    """
    names = [validator.name for validator in validators]
    if len(names) != len(set(names)):
        raise ValueError("aggregate validator names must be unique")
    repository = git.repository
    resolved_index = index_path.expanduser().resolve()
    resolved_catalog = catalog_root.expanduser().resolve()
    try:
        index_relative = resolved_index.relative_to(repository).as_posix()
        resolved_catalog.relative_to(repository)
    except ValueError as error:
        raise ValueError(
            "catalog index and root must be inside the integration repository"
        ) from error
    if resolved_index.parent != resolved_catalog:
        raise ValueError("catalog index must be directly inside catalog_root")

    # Verification is read-only and happens before any integration mutation.
    for candidate in batch.candidates:
        if candidate.snapshot is None:
            continue
        verification = git.verify_candidate(
            candidate.snapshot.repository,
            base_commit=candidate.snapshot.base_commit,
            candidate_commit=candidate.snapshot.candidate_commit,
            allowed_paths=candidate.proposal.allowed_paths,
            expected_patch_sha256=candidate.snapshot.candidate_digest,
        )
        if index_relative in verification.changed_paths:
            raise GitOpsError("Smith candidates must return semantic deltas, not index patches")
        if candidate.index_delta is not None:
            wrapper_relative = (
                resolved_catalog.relative_to(repository) / candidate.index_delta.row.path
            ).as_posix()
            if wrapper_relative not in candidate.proposal.allowed_paths:
                raise GitOpsError("IndexDelta wrapper is outside the item's allowed paths")

    item_results: list[FanInItemResult] = []
    integrated_candidates: list[ApprovedSmithCandidate] = []
    integrated_ids: list[str] = []
    index_update: IndexUpdate | None = None
    try:
        with git.transaction() as transaction:
            expected_head = git.inspect().head
            if not git.inspect().clean:
                raise GitOpsError("fan-in requires a clean integration checkout")
            for candidate in batch.candidates:
                item_id = candidate.proposal.item_id
                if candidate.snapshot is None:
                    item_results.append(FanInItemResult(item_id, True, expected_head))
                    integrated_candidates.append(candidate)
                    integrated_ids.append(item_id)
                    continue
                try:
                    integrated_commit = transaction.integrate_commit(
                        candidate_commit=candidate.snapshot.candidate_commit,
                        expected_head=expected_head,
                    )
                except GitConflictError as error:
                    item_results.append(FanInItemResult(item_id, False, None, str(error)))
                    continue
                expected_head = integrated_commit
                item_results.append(FanInItemResult(item_id, True, integrated_commit))
                integrated_candidates.append(candidate)
                integrated_ids.append(item_id)

            deltas = tuple(
                candidate.index_delta
                for candidate in integrated_candidates
                if candidate.index_delta is not None
            )
            if deltas:
                index_update = update_catalog_index(
                    resolved_index,
                    resolved_catalog,
                    deltas,
                    controller_lock=transaction.lock,
                )
                if index_update.changed:
                    transaction.stage_paths(repository, (index_relative,))
                    expected_head = transaction.commit_signed_off(
                        repository,
                        message=index_commit_message,
                        expected_paths=(index_relative,),
                    )

            completed_validations: list[str] = []
            for validator in validators:
                validator.validate(repository)
                completed_validations.append(validator.name)
            inspection = git.inspect()
            if not inspection.clean or inspection.head != expected_head:
                raise GitOpsError("aggregate validation changed the integration checkout")
    except (GitOpsError, ValueError) as error:
        raise FanInError(
            f"Smith fan-in failed closed: {error}",
            integrated_item_ids=tuple(integrated_ids),
        ) from error

    return FanInResult(
        items=tuple(item_results),
        final_head=git.inspect().head,
        index_update=index_update,
        aggregate_validations=tuple(completed_validations),
    )


def apply_fan_in_result(state: RunState, result: FanInResult) -> RunState:
    """Advance only successfully integrated siblings to domain INTEGRATED."""
    next_state = state
    for item_id in result.integrated_item_ids:
        item = next_state.item(item_id)
        if item.status is not WorkItemStatus.APPROVED:
            raise ValueError(f"fan-in result item {item_id!r} is no longer APPROVED")
        integrating = item.transition(WorkItemStatus.INTEGRATING)
        next_state = next_state.replace_item(integrating)
        next_state = next_state.replace_item(integrating.transition(WorkItemStatus.INTEGRATED))
    return next_state
