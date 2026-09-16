# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for stable, locked Smith candidate fan-in."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent_flow.workflows.staircase.common.gitops import ControllerGitOps
from agent_flow.workflows.staircase.common.index import IndexDelta
from agent_flow.workflows.staircase.common.outcomes import (
    GoalProposal,
    PlanDraftOutcome,
    PlanReviewDecision,
    PlanReviewOutcome,
    StageProposal,
)
from agent_flow.workflows.staircase.common.policy import ExecutionShape, WorkItemProposal
from agent_flow.workflows.staircase.common.policy import WorkItemKind as PolicyWorkItemKind
from agent_flow.workflows.staircase.controller.fan_in import (
    AggregateValidator,
    ApprovedSmithCandidate,
    apply_fan_in_result,
    execute_fan_in,
    plan_fan_in,
)
from agent_flow.workflows.staircase.controller.smith import CandidateSnapshot
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    GoalRecord,
    JobReference,
    Role,
    RunState,
    StageRecord,
    WorkItemKind,
    WorkItemRecord,
    WorkItemStatus,
)

_DIGEST = "a" * 64
_INDEX = """# Catalog Index
# preserve this comment

entries:
  # --- norm ---
  - path: norm/base.py
    impl: torch.ops.trtllm.base
    summary: "Base"
"""


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def _repository(tmp_path: Path) -> tuple[Path, Path, Path, str, ControllerGitOps]:
    repository = tmp_path / "repository"
    catalog_root = repository / "catalog"
    (catalog_root / "norm").mkdir(parents=True)
    (catalog_root / "norm" / "base.py").write_text("VALUE = 0\n", encoding="utf-8")
    index_path = catalog_root / "index.yaml"
    index_path.write_text(_INDEX, encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Staircase Tests")
    _git(repository, "config", "user.email", "staircase@example.com")
    _git(repository, "add", "--", "catalog")
    _git(repository, "commit", "-q", "-s", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")
    controller = ControllerGitOps(repository, tmp_path / "git.lock", "fan-in-controller")
    return repository, catalog_root, index_path, base, controller


def _proposal(item_id: str) -> WorkItemProposal:
    return WorkItemProposal(
        item_id=item_id,
        goal_id="catalog-goal",
        kind=PolicyWorkItemKind.CATALOG_ONBOARD,
        resource_class="coder_analysis",
        execution=ExecutionShape(nodes=1, ranks_per_node=1, gpus_per_node=1),
        modifies_files=True,
        entry_ids=(item_id,),
        allowed_paths=(f"catalog/norm/{item_id}.py",),
    )


def _plan(proposals: tuple[WorkItemProposal, ...]) -> tuple[PlanDraftOutcome, PlanReviewOutcome]:
    item_ids = tuple(proposal.item_id for proposal in proposals)
    plan = PlanDraftOutcome(
        stages=(StageProposal("catalog-stage", ("catalog-goal",), ("catalog-valid",)),),
        goals=(GoalProposal("catalog-goal", "catalog-stage", "catalog", item_ids),),
        items=proposals,
        digest="b" * 64,
    )
    return plan, PlanReviewOutcome(PlanReviewDecision.ACCEPT, plan.digest, ())


def _candidate(
    controller: ControllerGitOps,
    proposal: WorkItemProposal,
    *,
    base: str,
    root: Path,
) -> CandidateSnapshot:
    worktree = root / f"candidate-{proposal.item_id}"
    with controller.transaction() as transaction:
        transaction.create_candidate_worktree(
            worktree,
            branch=f"staircase/candidate/{proposal.item_id}",
            base_commit=base,
        )
        wrapper = worktree / proposal.allowed_paths[0]
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(f"VALUE = {proposal.item_id!r}\n", encoding="utf-8")
        transaction.stage_paths(worktree, proposal.allowed_paths)
        commit = transaction.commit_signed_off(
            worktree,
            message=f"smith: add {proposal.item_id}",
            expected_paths=proposal.allowed_paths,
        )
    verification = controller.verify_candidate(
        worktree,
        base_commit=base,
        candidate_commit=commit,
        allowed_paths=proposal.allowed_paths,
    )
    return CandidateSnapshot(
        item_id=proposal.item_id,
        coder_attempt_id=f"{proposal.item_id}.coder",
        repository=worktree,
        base_commit=base,
        candidate_commit=commit,
        candidate_digest=verification.patch_sha256,
        changed_paths=verification.changed_paths,
    )


def _delta(item_id: str) -> IndexDelta:
    return IndexDelta.from_mapping(
        {
            "item_id": item_id,
            "row": {
                "entry_id": item_id,
                "path": f"norm/{item_id}.py",
                "implementation": f"torch.ops.trtllm.{item_id}",
                "summary": f"Smith {item_id}",
            },
            "certification_cells": [],
            "expected_row_sha256": None,
        }
    )


def _approved_state(
    proposals: tuple[WorkItemProposal, ...],
    snapshots: dict[str, CandidateSnapshot],
    *,
    base: str,
) -> RunState:
    items: list[WorkItemRecord] = []
    for position, proposal in enumerate(proposals, start=1):
        snapshot = snapshots[proposal.item_id]
        coder = AttemptRecord(
            attempt_id=snapshot.coder_attempt_id,
            item_id=proposal.item_id,
            sequence=1,
            role=Role.CODER,
            kind=AttemptKind.ROLE,
            generation=1,
            status=AttemptStatus.VALIDATED,
            profile=DomainProfile.SMITH,
            submission_token=f"coder-{position}",
            job=JobReference(str(100 + position)),
            result_digest=f"{position:064x}",
            candidate_digest=snapshot.candidate_digest,
        )
        gate = AttemptRecord(
            attempt_id=f"{proposal.item_id}.gate",
            item_id=proposal.item_id,
            sequence=2,
            role=Role.GATE,
            kind=AttemptKind.DETERMINISTIC_GATE,
            generation=1,
            status=AttemptStatus.VALIDATED,
            profile=DomainProfile.SMITH,
            submission_token=f"gate-{position}",
            job=JobReference(str(200 + position)),
            result_digest=f"{position + 10:064x}",
        )
        analysis = AttemptRecord(
            attempt_id=f"{proposal.item_id}.analysis",
            item_id=proposal.item_id,
            sequence=3,
            role=Role.REVIEWER,
            kind=AttemptKind.REVIEWER_ANALYSIS,
            generation=1,
            status=AttemptStatus.VALIDATED,
            profile=DomainProfile.SMITH,
            submission_token=f"analysis-{position}",
            job=JobReference(str(300 + position)),
            result_digest=f"{position + 20:064x}",
            review_of_attempt_id=coder.attempt_id,
            reviewed_candidate_digest=snapshot.candidate_digest,
        )
        reviewer = AttemptRecord(
            attempt_id=f"{proposal.item_id}.reviewer",
            item_id=proposal.item_id,
            sequence=4,
            role=Role.REVIEWER,
            kind=AttemptKind.REVIEWER_RERUN,
            generation=1,
            status=AttemptStatus.VALIDATED,
            profile=DomainProfile.SMITH,
            submission_token=f"reviewer-{position}",
            job=JobReference(str(400 + position)),
            result_digest=f"{position + 30:064x}",
            review_of_attempt_id=coder.attempt_id,
            reviewed_candidate_digest=snapshot.candidate_digest,
        )
        items.append(
            WorkItemRecord(
                item_id=proposal.item_id,
                stage_id="catalog-stage",
                goal_id="catalog-goal",
                kind=WorkItemKind.CATALOG_ONBOARD,
                profile=DomainProfile.SMITH,
                status=WorkItemStatus.APPROVED,
                attempts=(coder, gate, analysis, reviewer),
                candidate_attempt_id=coder.attempt_id,
                candidate_digest=snapshot.candidate_digest,
                reviewer_attempt_id=reviewer.attempt_id,
            )
        )
    item_ids = tuple(proposal.item_id for proposal in proposals)
    return RunState(
        run_id="fan-in-run",
        task_digest=_DIGEST,
        base_commit=base,
        generation=1,
        stages=(StageRecord("catalog-stage", ("catalog-goal",)),),
        goals=(GoalRecord("catalog-goal", "catalog-stage", item_ids),),
        items=tuple(items),
    )


def test_fan_in_is_stable_locked_and_comment_preserving(tmp_path: Path) -> None:
    repository, catalog_root, index_path, base, controller = _repository(tmp_path)
    proposals = (_proposal("alpha"), _proposal("bravo"))
    snapshots = {
        proposal.item_id: _candidate(controller, proposal, base=base, root=tmp_path)
        for proposal in proposals
    }
    state = _approved_state(proposals, snapshots, base=base)
    plan, review = _plan(proposals)
    candidates = tuple(
        ApprovedSmithCandidate(proposal, snapshots[proposal.item_id], _delta(proposal.item_id))
        for proposal in reversed(proposals)
    )
    observed: list[str] = []

    def validate(repository_path: Path) -> None:
        assert (repository_path / "catalog" / "norm" / "alpha.py").is_file()
        assert (repository_path / "catalog" / "norm" / "bravo.py").is_file()
        observed.append("aggregate")

    batch = plan_fan_in(state, plan, review, candidates)
    assert [entry.proposal.item_id for entry in batch.candidates] == ["alpha", "bravo"]
    result = execute_fan_in(
        batch,
        git=controller,
        index_path=index_path,
        catalog_root=catalog_root,
        validators=(AggregateValidator("catalog-contract", validate),),
    )
    integrated = apply_fan_in_result(state, result)

    assert result.integrated_item_ids == ("alpha", "bravo")
    assert observed == ["aggregate"]
    assert all(
        integrated.item(item_id).status is WorkItemStatus.INTEGRATED
        for item_id in ("alpha", "bravo")
    )
    index_text = index_path.read_text(encoding="utf-8")
    assert "# preserve this comment" in index_text
    assert index_text.index("norm/alpha.py") < index_text.index("norm/bravo.py")
    subjects = _git(repository, "log", "--reverse", "--format=%s", f"{base}..HEAD").splitlines()
    assert subjects == [
        "smith: add alpha",
        "smith: add bravo",
        "agent-flow: integrate reviewed Smith catalog index deltas",
    ]


def test_one_conflict_does_not_cancel_independent_sibling(tmp_path: Path) -> None:
    repository, catalog_root, index_path, base, controller = _repository(tmp_path)
    proposals = (_proposal("alpha"), _proposal("bravo"))
    snapshots = {
        proposal.item_id: _candidate(controller, proposal, base=base, root=tmp_path)
        for proposal in proposals
    }
    state = _approved_state(proposals, snapshots, base=base)
    plan, review = _plan(proposals)

    conflicting = repository / "catalog" / "norm" / "alpha.py"
    conflicting.write_text("VALUE = 'integration divergence'\n", encoding="utf-8")
    _git(repository, "add", "--", "catalog/norm/alpha.py")
    _git(repository, "commit", "-q", "-s", "-m", "independent divergence")

    batch = plan_fan_in(
        state,
        plan,
        review,
        tuple(
            ApprovedSmithCandidate(proposal, snapshots[proposal.item_id], _delta(proposal.item_id))
            for proposal in proposals
        ),
    )
    result = execute_fan_in(
        batch,
        git=controller,
        index_path=index_path,
        catalog_root=catalog_root,
    )
    updated = apply_fan_in_result(state, result)

    assert result.items[0].item_id == "alpha"
    assert result.items[0].integrated is False
    assert result.items[1].item_id == "bravo"
    assert result.items[1].integrated is True
    assert updated.item("alpha").status is WorkItemStatus.APPROVED
    assert updated.item("bravo").status is WorkItemStatus.INTEGRATED
    assert "norm/bravo.py" in index_path.read_text(encoding="utf-8")
    assert "norm/alpha.py" not in index_path.read_text(encoding="utf-8")
