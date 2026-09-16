# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure non-Smith controller action planning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from agent_flow.workflows.staircase.common.gates import (
    EvidenceScope,
    GateCommand,
    GatePhase,
    GatePurpose,
    GateSpec,
)
from agent_flow.workflows.staircase.common.policy import ExecutionShape, WorkItemProposal
from agent_flow.workflows.staircase.common.policy import WorkItemKind as PolicyWorkItemKind
from agent_flow.workflows.staircase.controller.domain import (
    AgentSettings,
    DomainActionKind,
    DomainBlockCode,
    FrozenCandidate,
    GateExecutionContract,
    plan_assembler_coder_action,
    plan_deterministic_gate_action,
    plan_qa_action,
    plan_reviewer_analysis_action,
    plan_reviewer_rerun_action,
    plan_tuner_coder_action,
)
from agent_flow.workflows.staircase.onboarding.workflow import (
    AssemblyPreparation,
    AssemblyScope,
    CatalogGapProposal,
    CoreAssemblerInput,
    derive_route_identity,
)
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    GoalRecord,
    IntegrationEvidenceKind,
    IntegrationRecord,
    JobReference,
    Role,
    RunState,
    StageRecord,
    WorkItemKind,
    WorkItemRecord,
    WorkItemStatus,
)
from agent_flow.workflows.staircase.task_schema import (
    ParallelMapping,
    ReferenceConfig,
    ResourceClass,
    TargetConfig,
)
from agent_flow.workflows.staircase.tuning.workflow import (
    KnobChange,
    KnobKind,
    MetricDirection,
    TuningHypothesis,
    UncertaintyRule,
)

_TASK_DIGEST = "a" * 64
_CANDIDATE_DIGEST = "b" * 64
_RESULT_DIGEST = "c" * 64
_CANDIDATE_COMMIT = "d" * 40
_SETTINGS = AgentSettings("codex", "test-model")


def _resource(name: str, *, gpus: int = 0) -> ResourceClass:
    return ResourceClass(name, 1, 1, gpus, 4, 4096, 600)


def _route():
    return derive_route_identity(
        ReferenceConfig(
            checkpoint=Path("/checkpoint"),
            provenance="test",
            architecture="ExampleForCausalLM",
            additional_sources=(),
        ),
        TargetConfig(
            family="example",
            checkpoint_id="example_7b",
            sm=100,
            world_size=1,
            mapping=ParallelMapping(1, 1, 1, 1, 1),
            features=(),
            expected_route="tensorrt_llm/_torch/modeling_v2/models/example",
            synthetic_target=False,
        ),
    )


def _proposal(
    *,
    kind: PolicyWorkItemKind = PolicyWorkItemKind.ASSEMBLE_CORE,
    status: WorkItemStatus = WorkItemStatus.READY,
    dependencies: tuple[str, ...] = (),
    allowed_paths: tuple[str, ...] | None = None,
    modifies_files: bool = True,
) -> tuple[RunState, WorkItemProposal]:
    route = _route()
    paths = allowed_paths
    if paths is None:
        paths = (str(route.target_directory / "modeling.py"),)
    proposal = WorkItemProposal(
        item_id="domain-item",
        goal_id="domain-goal",
        kind=kind,
        resource_class="coder_analysis",
        execution=ExecutionShape(),
        modifies_files=modifies_files,
        domain_input=(
            _hypothesis_for_item("domain-item")
            if kind is PolicyWorkItemKind.TUNE_HYPOTHESIS
            else None
        ),
        dependencies=dependencies,
        allowed_paths=paths,
    )
    item_profile = (
        DomainProfile.TUNER
        if kind is PolicyWorkItemKind.TUNE_HYPOTHESIS
        else DomainProfile.ASSEMBLER
    )
    items = [
        WorkItemRecord(
            item_id=proposal.item_id,
            stage_id="domain-stage",
            goal_id=proposal.goal_id,
            kind=WorkItemKind(kind.value),
            profile=item_profile,
            status=status,
            dependencies=dependencies,
        )
    ]
    if dependencies:
        items.insert(
            0,
            WorkItemRecord(
                item_id=dependencies[0],
                stage_id="domain-stage",
                goal_id=proposal.goal_id,
                kind=WorkItemKind.CATALOG_VERIFY,
                profile=DomainProfile.SMITH,
            ),
        )
    item_ids = tuple(item.item_id for item in items)
    state = RunState(
        run_id="domain-run",
        task_digest=_TASK_DIGEST,
        base_commit="e" * 40,
        generation=1,
        stages=(StageRecord("domain-stage", (proposal.goal_id,)),),
        goals=(GoalRecord(proposal.goal_id, "domain-stage", item_ids),),
        items=tuple(items),
    )
    return state, proposal


def _preparation(state: RunState) -> AssemblyPreparation:
    item = state.item("domain-item")
    scope = AssemblyScope(item.item_id, _route(), item.dependencies, ())
    return AssemblyPreparation(CoreAssemblerInput(scope), ())


def _gates() -> tuple[GateSpec, ...]:
    return (
        GateSpec(
            "native-contract",
            GatePhase.NATIVE_CONTRACT,
            GatePurpose.CORRECTNESS,
            GateCommand(("python3", "-m", "pytest", "tests/native.py")),
        ),
        GateSpec(
            "boot",
            GatePhase.BOOT,
            GatePurpose.CORRECTNESS,
            GateCommand(("python3", "-m", "pytest", "tests/boot.py")),
        ),
    )


def _cpu_execution() -> GateExecutionContract:
    return GateExecutionContract(EvidenceScope.CPU_STATIC, 0, None, None, False)


def _single_gpu_execution() -> GateExecutionContract:
    return GateExecutionContract(
        EvidenceScope.SINGLE_GPU_PRODUCT,
        1,
        0,
        0,
        True,
    )


@pytest.mark.parametrize(
    "contract",
    [
        (EvidenceScope.CPU_STATIC, 1, None, None, False),
        (EvidenceScope.CPU_STATIC, 0, None, None, True),
        (EvidenceScope.SINGLE_GPU_PRODUCT, 1, None, 0, True),
        (EvidenceScope.SINGLE_GPU_PRODUCT, 1, 0, 0, False),
    ],
)
def test_gate_execution_contract_rejects_noncanonical_generic_shapes(
    contract: tuple[EvidenceScope, int, int | None, int | None, bool],
) -> None:
    with pytest.raises(ValueError, match="requires execution shape"):
        GateExecutionContract(*contract)


def _validated_attempt(
    state: RunState,
    attempt_id: str,
    *,
    candidate_digest: str,
    job_id: str,
) -> RunState:
    item = state.item("domain-item")
    attempts = tuple(
        replace(
            attempt,
            status=AttemptStatus.VALIDATED,
            submission_token=f"submit-{attempt.sequence}",
            job=JobReference(job_id),
            result_digest=_RESULT_DIGEST,
            candidate_digest=candidate_digest,
        )
        if attempt.attempt_id == attempt_id
        else attempt
        for attempt in item.attempts
    )
    desired_item = replace(item, attempts=attempts)
    return replace(
        state,
        items=tuple(
            desired_item if entry.item_id == desired_item.item_id else entry
            for entry in state.items
        ),
    )


def _candidate(state: RunState) -> FrozenCandidate:
    coder = state.item("domain-item").attempts[0]
    return FrozenCandidate(
        item_id="domain-item",
        coder_attempt_id=coder.attempt_id,
        commit=_CANDIDATE_COMMIT,
        digest=_CANDIDATE_DIGEST,
        changed_paths=(state.item("domain-item").kind is WorkItemKind.TUNE_HYPOTHESIS)
        and ()
        or (str(_route().target_directory / "modeling.py"),),
    )


def _with_integrated_dependency(state: RunState, dependency_id: str) -> RunState:
    candidate_digest = "7" * 64
    coder = AttemptRecord(
        attempt_id=f"{dependency_id}-coder",
        item_id=dependency_id,
        sequence=1,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        generation=1,
        profile=DomainProfile.SMITH,
        status=AttemptStatus.VALIDATED,
        submission_token="dependency-coder",
        job=JobReference("801"),
        result_digest="8" * 64,
        candidate_digest=candidate_digest,
    )
    reviewer = AttemptRecord(
        attempt_id=f"{dependency_id}-reviewer",
        item_id=dependency_id,
        sequence=2,
        role=Role.REVIEWER,
        kind=AttemptKind.REVIEWER_RERUN,
        generation=1,
        profile=DomainProfile.SMITH,
        status=AttemptStatus.VALIDATED,
        submission_token="dependency-reviewer",
        job=JobReference("802"),
        result_digest="9" * 64,
        review_of_attempt_id=coder.attempt_id,
        reviewed_candidate_digest=candidate_digest,
    )
    dependency = WorkItemRecord(
        item_id=dependency_id,
        stage_id="domain-stage",
        goal_id="domain-goal",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
        status=WorkItemStatus.INTEGRATED,
        attempts=(coder, reviewer),
        candidate_attempt_id=coder.attempt_id,
        candidate_digest=candidate_digest,
        reviewer_attempt_id=reviewer.attempt_id,
    )
    dependent = replace(state.item("domain-item"), status=WorkItemStatus.READY)
    items = tuple(
        dependency
        if item.item_id == dependency_id
        else dependent
        if item.item_id == dependent.item_id
        else item
        for item in state.items
    )
    return replace(
        state,
        items=items,
        integration_history=(
            IntegrationRecord(
                sequence=1,
                generation=1,
                item_id=dependency_id,
                previous_commit=state.base_commit,
                new_commit="f" * 40,
                evidence_kind=IntegrationEvidenceKind.CANDIDATE,
                evidence_digest=candidate_digest,
            ),
        ),
    )


def _hypothesis_for_item(item_id: str) -> TuningHypothesis:
    return TuningHypothesis(
        hypothesis_id="one-knob",
        item_id=item_id,
        statement="The alternative backend improves throughput.",
        metric="tokens_per_second",
        direction=MetricDirection.HIGHER_IS_BETTER,
        changes=(
            KnobChange(
                name="attention.backend",
                kind=KnobKind.CONFIGURATION,
                baseline_value="baseline",
                candidate_value="candidate",
            ),
        ),
        uncertainty=UncertaintyRule(
            minimum_effect=5.0,
            noise_threshold=1.0,
            maximum_combined_uncertainty=3.0,
        ),
    )


def _hypothesis(proposal: WorkItemProposal) -> TuningHypothesis:
    return _hypothesis_for_item(proposal.item_id)


def test_assembler_coder_emits_exact_generic_envelope_without_side_effects(
    tmp_path: Path,
) -> None:
    state, proposal = _proposal()
    workspace = tmp_path / "not-created"

    plan = plan_assembler_coder_action(
        state,
        proposal,
        _preparation(state),
        resource=_resource("coder_analysis"),
        workspace=workspace,
        settings=_SETTINGS,
        evidence_paths=("reports/assembler.json",),
    )

    assert plan.ready
    assert plan.action is not None
    assert plan.action.kind is DomainActionKind.ASSEMBLER_CODER
    assert plan.action.attempt.role is Role.CODER
    assert plan.action.attempt.profile is DomainProfile.ASSEMBLER
    assert not plan.action.read_only
    assert plan.state.item("domain-item").status is WorkItemStatus.CODING
    assert set(plan.action.manifest.payload) == {"runtime", "context"}
    runtime = plan.action.manifest.payload["runtime"]
    assert runtime == {
        "schema_version": 1,
        "kind": "agent",
        "backend_kind": "codex",
        "model": "test-model",
        "prompt_id": plan.action.attempt.attempt_id,
        "prompt": (
            "Implement exactly the controller-authorized Assembler unit using only the "
            "integrated dependencies in context. Stay within allowed_paths and publish the "
            "authorized evidence."
        ),
        "evidence_paths": ["reports/assembler.json"],
        "candidate_digest": None,
    }
    assert plan.action.manifest.allowed_paths == proposal.allowed_paths
    assert not workspace.exists()


def test_assembler_blocks_nonintegrated_dependencies_catalog_gaps_and_extra_paths(
    tmp_path: Path,
) -> None:
    state, proposal = _proposal(
        status=WorkItemStatus.PLANNED,
        dependencies=("catalog-entry",),
    )
    dependency = plan_assembler_coder_action(
        state,
        proposal,
        _preparation(state),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert dependency.blocked is not None
    assert dependency.blocked.code is DomainBlockCode.DEPENDENCY_NOT_INTEGRATED
    assert dependency.state is state

    ready_state, ready_proposal = _proposal()
    gap = CatalogGapProposal(
        proposal_id="gap-entry",
        consumer_item_id="domain-item",
        goal_id="domain-goal",
        entry_id="missing-entry",
        catalog_path=PurePosixPath("tensorrt_llm/_torch/modeling_v2/catalog/attention/missing.py"),
        reason="required by flat forward",
    )
    gaps = plan_assembler_coder_action(
        ready_state,
        ready_proposal,
        AssemblyPreparation(None, (gap,)),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert gaps.blocked is not None
    assert gaps.blocked.code is DomainBlockCode.CATALOG_GAP

    route = _route()
    extra_state, extra_proposal = _proposal(
        allowed_paths=(
            str(route.target_directory / "modeling.py"),
            str(route.target_directory / "weights.py"),
        )
    )
    extra = plan_assembler_coder_action(
        extra_state,
        extra_proposal,
        _preparation(extra_state),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert extra.blocked is not None
    assert extra.blocked.code is DomainBlockCode.ALLOWED_PATHS


def test_tuner_coder_carries_exactly_one_changed_variable(tmp_path: Path) -> None:
    state, proposal = _proposal(
        kind=PolicyWorkItemKind.TUNE_HYPOTHESIS,
        allowed_paths=(),
        modifies_files=False,
    )
    hypothesis = _hypothesis(proposal)

    plan = plan_tuner_coder_action(
        state,
        proposal,
        hypothesis,
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=("reports/paired-ab.json",),
    )

    assert plan.action is not None
    context = plan.action.manifest.payload["context"]
    assert context["action"] == "tuner_coder"
    assert context["hypothesis"]["change"] == {
        "name": "attention.backend",
        "kind": "configuration",
        "baseline_value": "baseline",
        "candidate_value": "candidate",
    }
    assert plan.action.manifest.allowed_paths == ()


def test_assembler_and_tuner_use_persisted_integration_head_for_dependencies(
    tmp_path: Path,
) -> None:
    assembler_state, assembler_proposal = _proposal(
        status=WorkItemStatus.PLANNED,
        dependencies=("catalog-entry",),
    )
    assembler_state = _with_integrated_dependency(assembler_state, "catalog-entry")
    assembler = plan_assembler_coder_action(
        assembler_state,
        assembler_proposal,
        _preparation(assembler_state),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert assembler.action is not None
    assert assembler.action.base_commit == "f" * 40
    assert assembler_state.base_commit == "e" * 40

    tuner_state, tuner_proposal = _proposal(
        kind=PolicyWorkItemKind.TUNE_HYPOTHESIS,
        status=WorkItemStatus.PLANNED,
        dependencies=("catalog-entry",),
        allowed_paths=(),
        modifies_files=False,
    )
    integrated_tuner_state = _with_integrated_dependency(tuner_state, "catalog-entry")
    tuner = plan_tuner_coder_action(
        integrated_tuner_state,
        tuner_proposal,
        _hypothesis(tuner_proposal),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert tuner.action is not None
    assert tuner.action.base_commit == "f" * 40

    missing_receipt_state = replace(integrated_tuner_state, integration_history=())
    missing_receipt = plan_tuner_coder_action(
        missing_receipt_state,
        tuner_proposal,
        _hypothesis(tuner_proposal),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert missing_receipt.blocked is not None
    assert missing_receipt.blocked.code is DomainBlockCode.DEPENDENCY_NOT_INTEGRATED


def test_gate_reviewer_and_qa_are_distinct_candidate_pinned_attempts(
    tmp_path: Path,
) -> None:
    state, proposal = _proposal()
    coder = plan_assembler_coder_action(
        state,
        proposal,
        _preparation(state),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert coder.action is not None
    state = _validated_attempt(
        coder.state,
        coder.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="101",
    )
    candidate = _candidate(state)

    first_gate = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[0],
        _cpu_execution(),
        resource=_resource("deterministic_gate"),
        workspace=tmp_path,
    )
    assert first_gate.action is not None
    first_runtime = first_gate.action.manifest.payload["runtime"]
    assert first_runtime == {
        "schema_version": 1,
        "kind": "gate",
        "command": {
            "command_id": "native-contract",
            "argv": ["python3", "-m", "pytest", "tests/native.py"],
            "environment": {"TRTLLM_MODELING_V2": "require"},
            "cwd": ".",
        },
        "candidate_attempt_id": candidate.coder_attempt_id,
        "candidate_digest": _CANDIDATE_DIGEST,
        "receipt": {
            "gate_id": "native-contract",
            "purpose": "correctness",
            "scope": "cpu_static",
            "certification_mode": "LOCAL",
            "expected_world_size": 0,
            "expected_rank": None,
            "expected_local_rank": None,
            "product_rank_body": False,
            "accuracy": None,
        },
    }
    assert first_gate.action.attempt.sequence == 2
    assert first_gate.state.item("domain-item").status is WorkItemStatus.CODING
    state = _validated_attempt(
        first_gate.state,
        first_gate.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="102",
    )
    second_gate = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[1],
        _single_gpu_execution(),
        resource=_resource("deterministic_gate", gpus=1),
        workspace=tmp_path,
    )
    assert second_gate.action is not None
    second_runtime = second_gate.action.manifest.payload["runtime"]
    assert second_runtime["command"]["command_id"] == "boot"
    assert second_runtime["receipt"]["scope"] == "single_gpu_product"
    assert second_runtime["receipt"]["expected_world_size"] == 1
    assert second_runtime["receipt"]["expected_rank"] == 0
    assert second_runtime["receipt"]["expected_local_rank"] == 0
    assert "placements" not in second_runtime["receipt"]
    assert second_gate.action.attempt.sequence == 3
    state = _validated_attempt(
        second_gate.state,
        second_gate.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="103",
    )
    item = state.item("domain-item").transition(WorkItemStatus.CANDIDATE_READY)
    state = replace(state, items=(item,))
    gate_attempt_ids = (
        first_gate.action.attempt.attempt_id,
        second_gate.action.attempt.attempt_id,
    )

    missing_receipt = plan_reviewer_analysis_action(
        state,
        proposal,
        candidate,
        _gates(),
        gate_attempt_ids=(gate_attempt_ids[0],),
        resource=_resource("reviewer_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=("reports/review.json",),
    )
    assert missing_receipt.blocked is not None
    assert missing_receipt.blocked.code is DomainBlockCode.GATE_CONTRACT

    analysis = plan_reviewer_analysis_action(
        state,
        proposal,
        candidate,
        _gates(),
        gate_attempt_ids=gate_attempt_ids,
        resource=_resource("reviewer_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=("reports/review.json",),
    )
    assert analysis.action is not None
    state = _validated_attempt(
        analysis.state,
        analysis.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="104",
    )
    rerun = plan_reviewer_rerun_action(
        state,
        proposal,
        candidate,
        _gates(),
        analysis_attempt_id=analysis.action.attempt.attempt_id,
        gate_attempt_ids=gate_attempt_ids,
        resource=_resource("reviewer_rerun"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=("reports/rerun.json",),
    )
    assert rerun.action is not None
    state = _validated_attempt(
        rerun.state,
        rerun.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="105",
    )
    item = state.item("domain-item")
    item = replace(
        item,
        status=WorkItemStatus.APPROVED,
        reviewer_attempt_id=rerun.action.attempt.attempt_id,
    )
    state = replace(state, items=(item,))

    qa = plan_qa_action(
        state,
        proposal,
        candidate,
        _gates(),
        gate_attempt_ids=gate_attempt_ids,
        gate_result_digests={attempt_id: _RESULT_DIGEST for attempt_id in gate_attempt_ids},
        resource=_resource("reviewer_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=("reports/qa.json",),
    )

    assert qa.action is not None
    attempts = qa.state.item("domain-item").attempts
    assert [attempt.role for attempt in attempts] == [
        Role.CODER,
        Role.GATE,
        Role.GATE,
        Role.REVIEWER,
        Role.REVIEWER,
        Role.QA,
    ]
    assert len({attempt.attempt_id for attempt in attempts}) == len(attempts)
    assert len({attempt.job.scheduler_id for attempt in attempts[:-1]}) == 5
    for action in (analysis.action, rerun.action, qa.action):
        assert action.read_only
        runtime = action.manifest.payload["runtime"]
        assert runtime["candidate_digest"] == _CANDIDATE_DIGEST
        assert set(action.manifest.payload) == {"runtime", "context"}
    assert qa.action.manifest.payload["context"]["gates"][0]["command"]["argv"] == [
        "python3",
        "-m",
        "pytest",
        "tests/native.py",
    ]


def test_gate_rejects_suite_in_one_attempt_and_wrong_topology_without_creating_paths(
    tmp_path: Path,
) -> None:
    state, proposal = _proposal()
    coder = plan_assembler_coder_action(
        state,
        proposal,
        _preparation(state),
        resource=_resource("coder_analysis"),
        workspace=tmp_path / "workspace",
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert coder.action is not None
    state = _validated_attempt(
        coder.state,
        coder.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="201",
    )
    candidate = _candidate(state)

    configured_resource = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[0],
        _cpu_execution(),
        resource=_resource("reviewer_analysis"),
        workspace=tmp_path / "workspace",
    )
    assert configured_resource.action is not None

    cpu_with_gpu = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[0],
        _cpu_execution(),
        resource=_resource("deterministic_gate", gpus=1),
        workspace=tmp_path / "workspace",
    )
    assert cpu_with_gpu.blocked is not None
    assert cpu_with_gpu.blocked.code is DomainBlockCode.RESOURCE_CONTRACT

    cpu_multi_rank = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[0],
        _cpu_execution(),
        resource=ResourceClass("deterministic_gate", 1, 2, 0, 4, 4096, 600),
        workspace=tmp_path / "workspace",
    )
    assert cpu_multi_rank.blocked is not None
    assert cpu_multi_rank.blocked.code is DomainBlockCode.RESOURCE_CONTRACT

    gpu_without_gpu = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[1],
        _single_gpu_execution(),
        resource=_resource("deterministic_gate"),
        workspace=tmp_path / "workspace",
    )
    assert gpu_without_gpu.blocked is not None
    assert gpu_without_gpu.blocked.code is DomainBlockCode.RESOURCE_CONTRACT

    suite_in_one_attempt = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates(),
        _cpu_execution(),
        resource=_resource("deterministic_gate"),
        workspace=tmp_path / "workspace",
    )
    assert suite_in_one_attempt.blocked is not None
    assert suite_in_one_attempt.blocked.code is DomainBlockCode.GATE_CONTRACT
    assert not (tmp_path / "workspace").exists()


def test_gate_rejects_collective_and_multi_rank_contracts_with_typed_block(
    tmp_path: Path,
) -> None:
    state, proposal = _proposal()
    coder = plan_assembler_coder_action(
        state,
        proposal,
        _preparation(state),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert coder.action is not None
    state = _validated_attempt(
        coder.state,
        coder.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="211",
    )
    candidate = _candidate(state)
    collective = GateSpec(
        "collective",
        GatePhase.COLLECTIVE,
        GatePurpose.CORRECTNESS,
        GateCommand(("python3", "-m", "pytest", "tests/collective.py")),
    )
    collective_plan = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        collective,
        _single_gpu_execution(),
        resource=_resource("deterministic_gate", gpus=1),
        workspace=tmp_path,
    )
    assert collective_plan.blocked is not None
    assert collective_plan.blocked.code is DomainBlockCode.RANK_LAUNCHER_REQUIRED

    four_gpu = GateExecutionContract(
        EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
        4,
        None,
        None,
        True,
    )
    multi_rank_plan = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[1],
        four_gpu,
        resource=ResourceClass("deterministic_gate", 1, 4, 4, 4, 4096, 600),
        workspace=tmp_path,
    )
    assert multi_rank_plan.blocked is not None
    assert multi_rank_plan.blocked.code is DomainBlockCode.RANK_LAUNCHER_REQUIRED


def test_reviewer_rejects_reused_scheduler_job_identity(tmp_path: Path) -> None:
    state, proposal = _proposal()
    coder = plan_assembler_coder_action(
        state,
        proposal,
        _preparation(state),
        resource=_resource("coder_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )
    assert coder.action is not None
    state = _validated_attempt(
        coder.state,
        coder.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="301",
    )
    candidate = _candidate(state)
    gate = plan_deterministic_gate_action(
        state,
        proposal,
        candidate,
        _gates()[0],
        _cpu_execution(),
        resource=_resource("deterministic_gate"),
        workspace=tmp_path,
    )
    assert gate.action is not None
    state = _validated_attempt(
        gate.state,
        gate.action.attempt.attempt_id,
        candidate_digest=_CANDIDATE_DIGEST,
        job_id="301",
    )
    item = state.item("domain-item").transition(WorkItemStatus.CANDIDATE_READY)
    state = replace(state, items=(item,))

    review = plan_reviewer_analysis_action(
        state,
        proposal,
        candidate,
        (_gates()[0],),
        gate_attempt_ids=(gate.action.attempt.attempt_id,),
        resource=_resource("reviewer_analysis"),
        workspace=tmp_path,
        settings=_SETTINGS,
        evidence_paths=(),
    )

    assert review.blocked is not None
    assert review.blocked.code is DomainBlockCode.INVALID_STATE
    assert "distinct scheduler jobs" in review.blocked.reason
