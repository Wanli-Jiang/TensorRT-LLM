# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic parallel Smith controller actions."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.gitops import ControllerGitOps
from agent_flow.workflows.staircase.common.isolation import digest_metadata_free_tree
from agent_flow.workflows.staircase.common.outcomes import (
    GoalProposal,
    PlanDraftOutcome,
    PlanReviewDecision,
    PlanReviewOutcome,
    StageProposal,
)
from agent_flow.workflows.staircase.common.policy import (
    ExecutionShape,
    PolicyViolation,
    ResourceEscalationRequest,
    WorkItemProposal,
)
from agent_flow.workflows.staircase.common.policy import WorkItemKind as PolicyWorkItemKind
from agent_flow.workflows.staircase.controller.smith import (
    approve_reviewer_rerun,
    derive_horizontal_smith_waves,
    freeze_coder_candidate,
    mark_ready_after_integrated_dependencies,
    materialize_attempts,
    plan_coder_wave,
    plan_gate_attempt,
    plan_resource_escalation_attempt,
    plan_reviewer_analysis_attempt,
    plan_reviewer_rerun_attempt,
)
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
from agent_flow.workflows.staircase.task_schema import (
    OverrideBounds,
    ResourceClass,
    RetryPolicy,
    SmithConfig,
)

_DIGEST = "a" * 64


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Staircase Tests")
    _git(repository, "config", "user.email", "staircase@example.com")
    (repository / "catalog" / "norm").mkdir(parents=True)
    (repository / "catalog" / "norm" / "alpha.py").write_text("VALUE = 0\n", encoding="utf-8")
    _git(repository, "add", "--", "catalog/norm/alpha.py")
    _git(repository, "commit", "-q", "-s", "-m", "base")
    return repository, _git(repository, "rev-parse", "HEAD")


def _smith_config(
    *,
    escalation_nodes: int = 2,
    max_parallel_items: int = 2,
    max_nodes_total: int = 4,
    max_gpus_total: int = 4,
) -> SmithConfig:
    resources = (
        ResourceClass("coder_analysis", 1, 1, 1, 4, 4096, 600),
        ResourceClass("exploratory_probe", escalation_nodes, 1, 1, 4, 4096, 600),
        ResourceClass("deterministic_gate", 1, 1, 1, 4, 4096, 600),
        ResourceClass("reviewer_analysis", 1, 1, 1, 4, 4096, 600),
        ResourceClass("reviewer_rerun", 1, 1, 1, 4, 4096, 600),
    )
    return SmithConfig(
        max_parallel_items=max_parallel_items,
        max_nodes_total=max_nodes_total,
        max_gpus_total=max_gpus_total,
        resource_classes=resources,
        per_item_override_bounds=OverrideBounds(2, 2, 2, 8, 8192, 1200),
        retry_policy=RetryPolicy(1, 1),
    )


def _proposal(item_id: str, *, path: str | None = None) -> WorkItemProposal:
    entry_id = item_id.replace("-", "_")
    return WorkItemProposal(
        item_id=item_id,
        goal_id="catalog-goal",
        kind=PolicyWorkItemKind.CATALOG_ONBOARD,
        resource_class="coder_analysis",
        execution=ExecutionShape(nodes=1, ranks_per_node=1, gpus_per_node=1),
        modifies_files=True,
        entry_ids=(entry_id,),
        allowed_paths=(path or f"catalog/norm/{entry_id}.py",),
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


def _state(
    proposals: tuple[WorkItemProposal, ...],
    *,
    base_commit: str,
    statuses: dict[str, WorkItemStatus] | None = None,
) -> RunState:
    item_ids = tuple(proposal.item_id for proposal in proposals)
    items = tuple(
        WorkItemRecord(
            item_id=proposal.item_id,
            stage_id="catalog-stage",
            goal_id="catalog-goal",
            kind=WorkItemKind(proposal.kind.value),
            profile=DomainProfile.SMITH,
            status=(statuses or {}).get(proposal.item_id, WorkItemStatus.READY),
            dependencies=proposal.dependencies,
        )
        for proposal in proposals
    )
    return RunState(
        run_id="smith-run",
        task_digest=_DIGEST,
        base_commit=base_commit,
        generation=1,
        stages=(StageRecord("catalog-stage", ("catalog-goal",)),),
        goals=(GoalRecord("catalog-goal", "catalog-stage", item_ids),),
        items=items,
    )


def _validated_attempt(
    attempt: AttemptRecord,
    *,
    job_id: str,
    candidate_digest: str | None = None,
) -> AttemptRecord:
    return replace(
        attempt,
        status=AttemptStatus.VALIDATED,
        submission_token=f"submission-{attempt.sequence}",
        job=JobReference(job_id),
        result_digest=hashlib.sha256(attempt.attempt_id.encode("utf-8")).hexdigest(),
        candidate_digest=candidate_digest,
    )


def _replace_attempt(state: RunState, item_id: str, attempt: AttemptRecord) -> RunState:
    item = state.item(item_id)
    attempts = tuple(
        attempt if existing.attempt_id == attempt.attempt_id else existing
        for existing in item.attempts
    )
    replacement = replace(item, attempts=attempts)
    items = tuple(
        replacement if existing.item_id == item_id else existing for existing in state.items
    )
    return replace(state, items=items)


def test_horizontal_wave_identity_is_plan_derived_and_cap_bounded() -> None:
    proposals = (
        _proposal("catalog-a"),
        _proposal("catalog-b"),
        _proposal("catalog-c"),
    )
    plan, _review = _plan(proposals)

    first = derive_horizontal_smith_waves(
        plan,
        smith=_smith_config(max_parallel_items=2),
    )
    second = derive_horizontal_smith_waves(
        plan,
        smith=_smith_config(max_parallel_items=2),
    )

    assert first == second
    assert tuple(wave.item_ids for wave in first) == (
        ("catalog-a", "catalog-b"),
        ("catalog-c",),
    )
    assert len({wave.wave_id for wave in first}) == 2


def test_coder_tick_selects_bounded_stable_parallel_wave(tmp_path: Path) -> None:
    proposals = (_proposal("charlie"), _proposal("alpha"), _proposal("bravo"))
    plan, review = _plan(proposals)
    state = _state(proposals, base_commit="f" * 40)
    tick = plan_coder_wave(
        state,
        plan,
        review,
        smith=_smith_config(),
        workspace=tmp_path / "workspace",
        allowed_path_roots=("catalog",),
    )

    assert [action.proposal.item_id for action in tick.actions] == ["alpha", "bravo"]
    assert tick.state.item("alpha").status is WorkItemStatus.CODING
    assert tick.state.item("bravo").status is WorkItemStatus.CODING
    assert tick.state.item("charlie").status is WorkItemStatus.READY
    assert all(action.attempt.role is Role.CODER for action in tick.actions)
    assert all(action.attempt.resource_class == "coder_analysis" for action in tick.actions)
    assert len({action.worktree for action in tick.actions}) == 2


def test_initial_smith_coder_rejects_non_coder_resource_class(tmp_path: Path) -> None:
    proposal = replace(_proposal("alpha"), resource_class="deterministic_gate")
    plan, review = _plan((proposal,))
    with pytest.raises(PolicyViolation, match="initial Smith Coder"):
        plan_coder_wave(
            _state((proposal,), base_commit="f" * 40),
            plan,
            review,
            smith=_smith_config(),
            workspace=tmp_path,
            allowed_path_roots=("catalog",),
        )


def test_materialization_is_serial_and_restart_idempotent(tmp_path: Path) -> None:
    repository, base = _repository(tmp_path)
    proposal = _proposal("alpha", path="catalog/norm/alpha.py")
    plan, review = _plan((proposal,))
    tick = plan_coder_wave(
        _state((proposal,), base_commit=base),
        plan,
        review,
        smith=_smith_config(),
        workspace=tmp_path / "workspace",
        allowed_path_roots=("catalog",),
    )
    controller = ControllerGitOps(repository, tmp_path / "git.lock", "smith-controller")

    first = materialize_attempts(tick.actions, git=controller)
    second = materialize_attempts(tick.actions, git=controller)

    assert first[0].input_digest == second[0].input_digest
    assert first[0].action.worktree.is_dir()
    assert not (first[0].action.worktree / ".git").exists()
    assert (first[0].action.attempt_dir / "input.json").is_file()


def test_gate_and_review_are_distinct_frozen_attempts(tmp_path: Path) -> None:
    repository, base = _repository(tmp_path)
    proposal = _proposal("alpha", path="catalog/norm/alpha.py")
    plan, review = _plan((proposal,))
    smith = _smith_config()
    coder_tick = plan_coder_wave(
        _state((proposal,), base_commit=base),
        plan,
        review,
        smith=smith,
        workspace=tmp_path / "workspace",
        allowed_path_roots=("catalog",),
    )
    controller = ControllerGitOps(repository, tmp_path / "git.lock", "smith-controller")
    materialize_attempts(coder_tick.actions, git=controller)
    coder_action = coder_tick.actions[0]
    (coder_action.worktree / "catalog" / "norm" / "alpha.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    candidate_repository = (
        tmp_path
        / "workspace"
        / "controller-candidates"
        / coder_action.proposal.item_id
        / coder_action.attempt.attempt_id
    )
    with controller.transaction() as transaction:
        binding = transaction.bind_candidate_overlay(
            coder_action.worktree,
            repository=candidate_repository,
            branch=coder_action.branch,
            base_commit=base,
            expected_paths=("catalog/norm/alpha.py",),
            expected_overlay_sha256=digest_metadata_free_tree(coder_action.worktree),
            message="smith: alpha candidate",
        )
    candidate_digest = controller.verify_candidate(
        candidate_repository,
        base_commit=base,
        candidate_commit=binding.candidate_commit,
        allowed_paths=("catalog/norm/alpha.py",),
    ).patch_sha256
    candidate = freeze_coder_candidate(
        coder_action,
        git=controller,
        changed_paths=("catalog/norm/alpha.py",),
        scanned_overlay_digest=digest_metadata_free_tree(coder_action.worktree),
        expected_patch_sha256=candidate_digest,
        commit_message="smith: alpha candidate",
    )
    coder = _validated_attempt(
        coder_action.attempt,
        job_id="101",
        candidate_digest=candidate.candidate_digest,
    )
    state = _replace_attempt(coder_tick.state, "alpha", coder)

    gate_tick = plan_gate_attempt(
        state, proposal, candidate, smith=smith, workspace=tmp_path / "workspace"
    )
    gate = _validated_attempt(gate_tick.actions[0].attempt, job_id="102")
    state = _replace_attempt(gate_tick.state, "alpha", gate)
    analysis_tick = plan_reviewer_analysis_attempt(
        state,
        proposal,
        candidate,
        gate_attempt_id=gate.attempt_id,
        smith=smith,
        workspace=tmp_path / "workspace",
    )
    analysis = _validated_attempt(analysis_tick.actions[0].attempt, job_id="103")
    state = _replace_attempt(analysis_tick.state, "alpha", analysis)
    rerun_tick = plan_reviewer_rerun_attempt(
        state,
        proposal,
        candidate,
        analysis_attempt_id=analysis.attempt_id,
        smith=smith,
        workspace=tmp_path / "workspace",
    )
    reused_job_rerun = _validated_attempt(rerun_tick.actions[0].attempt, job_id="103")
    reused_job_state = _replace_attempt(rerun_tick.state, "alpha", reused_job_rerun)
    with pytest.raises(ValueError, match="distinct scheduler jobs"):
        approve_reviewer_rerun(
            reused_job_state,
            item_id="alpha",
            reviewer_attempt_id=reused_job_rerun.attempt_id,
        )

    rerun = _validated_attempt(rerun_tick.actions[0].attempt, job_id="104")
    state = _replace_attempt(rerun_tick.state, "alpha", rerun)
    approved = approve_reviewer_rerun(state, item_id="alpha", reviewer_attempt_id=rerun.attempt_id)

    attempts = approved.item("alpha").attempts
    assert [attempt.kind for attempt in attempts] == [
        AttemptKind.ROLE,
        AttemptKind.DETERMINISTIC_GATE,
        AttemptKind.REVIEWER_ANALYSIS,
        AttemptKind.REVIEWER_RERUN,
    ]
    assert len({attempt.attempt_id for attempt in attempts}) == 4
    assert analysis.reviewed_candidate_digest == candidate.candidate_digest
    assert rerun.reviewed_candidate_digest == candidate.candidate_digest
    assert analysis_tick.actions[0].worktree != rerun_tick.actions[0].worktree
    assert candidate.repository == candidate_repository
    assert not (coder_action.worktree / ".git").exists()
    assert approved.item("alpha").status is WorkItemStatus.APPROVED


def test_resource_escalation_is_typed_and_bounded(tmp_path: Path) -> None:
    proposal = _proposal("alpha")
    plan, review = _plan((proposal,))
    smith = _smith_config()
    tick = plan_coder_wave(
        _state((proposal,), base_commit="f" * 40),
        plan,
        review,
        smith=smith,
        workspace=tmp_path,
        allowed_path_roots=("catalog",),
    )
    source = _validated_attempt(tick.actions[0].attempt, job_id="201")
    state = _replace_attempt(tick.state, "alpha", source)
    request = ResourceEscalationRequest(
        request_id="escalate-alpha",
        item_id="alpha",
        attempt_id=source.attempt_id,
        current_resource_class="coder_analysis",
        requested_resource_class="exploratory_probe",
        reason="single-node probe exhausted memory",
    )
    escalated = plan_resource_escalation_attempt(
        state, proposal, request, smith=smith, workspace=tmp_path
    )
    action = escalated.actions[0]
    assert action.resource.name == "exploratory_probe"
    assert action.attempt.sequence == 2
    assert action.attempt.resource_class == "exploratory_probe"
    assert action.attempt.predecessor_attempt_id == source.attempt_id
    assert action.attempt.resource_escalation_request_id == request.request_id
    assert action.manifest.payload["resource_class"] == "exploratory_probe"
    assert action.manifest.payload["predecessor_attempt_id"] == source.attempt_id
    assert action.manifest.payload["resource_escalation_request_id"] == request.request_id

    replay = plan_resource_escalation_attempt(
        state, proposal, request, smith=smith, workspace=tmp_path
    )
    assert replay == escalated
    with pytest.raises(PolicyViolation, match="already consumed"):
        plan_resource_escalation_attempt(
            escalated.state, proposal, request, smith=smith, workspace=tmp_path
        )

    with pytest.raises(PolicyViolation, match="per-item bounds"):
        plan_resource_escalation_attempt(
            state,
            proposal,
            request,
            smith=_smith_config(escalation_nodes=3),
            workspace=tmp_path,
        )


def test_resource_escalation_uses_actual_attempt_resource_class(tmp_path: Path) -> None:
    proposal = _proposal("alpha")
    plan, review = _plan((proposal,))
    smith = _smith_config()
    tick = plan_coder_wave(
        _state((proposal,), base_commit="f" * 40),
        plan,
        review,
        smith=smith,
        workspace=tmp_path,
        allowed_path_roots=("catalog",),
    )
    source = _validated_attempt(tick.actions[0].attempt, job_id="202")
    state = _replace_attempt(tick.state, "alpha", source)
    stale = ResourceEscalationRequest(
        request_id="stale-class",
        item_id="alpha",
        attempt_id=source.attempt_id,
        current_resource_class="exploratory_probe",
        requested_resource_class="coder_analysis",
        reason="stale request",
    )
    with pytest.raises(PolicyViolation, match="actual attempt-selected"):
        plan_resource_escalation_attempt(state, proposal, stale, smith=smith, workspace=tmp_path)


def test_resource_escalation_counts_active_plus_candidate_caps(tmp_path: Path) -> None:
    alpha = _proposal("alpha")
    bravo = _proposal("bravo")
    proposals = (alpha, bravo)
    plan, review = _plan(proposals)
    smith = _smith_config(max_nodes_total=2, max_gpus_total=3)
    tick = plan_coder_wave(
        _state(proposals, base_commit="f" * 40),
        plan,
        review,
        smith=smith,
        workspace=tmp_path,
        allowed_path_roots=("catalog",),
    )
    alpha_action = next(action for action in tick.actions if action.proposal.item_id == "alpha")
    source = _validated_attempt(alpha_action.attempt, job_id="203")
    state = _replace_attempt(tick.state, "alpha", source)
    request = ResourceEscalationRequest(
        request_id="aggregate-cap",
        item_id="alpha",
        attempt_id=source.attempt_id,
        current_resource_class="coder_analysis",
        requested_resource_class="exploratory_probe",
        reason="requires a bounded two-node probe",
    )

    with pytest.raises(PolicyViolation, match="active attempts plus candidate.*nodes"):
        plan_resource_escalation_attempt(state, alpha, request, smith=smith, workspace=tmp_path)


def test_dependency_cannot_become_ready_before_integration() -> None:
    smith_proposal = _proposal("smith")
    assembler = WorkItemRecord(
        item_id="assembler",
        stage_id="catalog-stage",
        goal_id="catalog-goal",
        kind=WorkItemKind.ASSEMBLE_CORE,
        profile=DomainProfile.ASSEMBLER,
        dependencies=("smith",),
    )
    smith_item = WorkItemRecord(
        item_id="smith",
        stage_id="catalog-stage",
        goal_id="catalog-goal",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
    )
    state = RunState(
        run_id="dependency-run",
        task_digest=_DIGEST,
        base_commit="f" * 40,
        generation=1,
        stages=(StageRecord("catalog-stage", ("catalog-goal",)),),
        goals=(GoalRecord("catalog-goal", "catalog-stage", ("smith", "assembler")),),
        items=(smith_item, assembler),
    )
    del smith_proposal
    with pytest.raises(ValueError, match="INTEGRATED"):
        mark_ready_after_integrated_dependencies(state, "assembler")
