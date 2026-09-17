# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure ModelingV2 route and Assembler orchestration."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from agent_flow.workflows.staircase.onboarding.workflow import (
    ROUTER_INDEX_PATH,
    AssemblerUnit,
    CatalogRequirement,
    CoreAssemblerInput,
    DependencyNotIntegratedError,
    FeatureAssemblerInput,
    IntegratedCatalogSurface,
    OnboardingContractError,
    ReconciliationDisposition,
    RoutingAssemblerInput,
    WeightsAssemblerInput,
    derive_route_identity,
    prepare_assembler_input,
    reconcile_modeling_v2_target,
    validate_assembler_output,
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
    ParallelMapping,
    ReferenceConfig,
    TargetConfig,
)

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64
_CATALOG_PATH = PurePosixPath(
    "tensorrt_llm/_torch/modeling_v2/catalog/attention/example_attention.py"
)


def _route(
    *,
    architecture: str = "GptOssForCausalLM",
    family: str = "gpt_oss",
    checkpoint_id: str = "gpt_oss_120b",
    sm: int = 103,
    mapping: ParallelMapping | None = None,
    features: tuple[str, ...] = (),
    expected_route: str | None = None,
):
    mapping = mapping or ParallelMapping(1, 1, 1, 1, 1)
    target = TargetConfig(
        family=family,
        checkpoint_id=checkpoint_id,
        sm=sm,
        world_size=mapping.tensor_parallel_size * mapping.pipeline_parallel_size,
        mapping=mapping,
        features=features,
        expected_route=(expected_route or f"tensorrt_llm/_torch/modeling_v2/models/{family}"),
        synthetic_target=False,
    )
    reference = ReferenceConfig(
        checkpoint=Path("/checkpoint"),
        provenance="test",
        architecture=architecture,
        additional_sources=(),
    )
    return derive_route_identity(reference, target)


def _validated_attempts(item_id: str) -> tuple[AttemptRecord, AttemptRecord]:
    coder = AttemptRecord(
        attempt_id=f"coder-{item_id}",
        item_id=item_id,
        sequence=1,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        submission_token=f"submit-{item_id}",
        job=JobReference(job_id="101"),
        result_digest=_DIGEST_A,
        candidate_digest=_DIGEST_B,
    )
    reviewer = AttemptRecord(
        attempt_id=f"reviewer-{item_id}",
        item_id=item_id,
        sequence=2,
        role=Role.REVIEWER,
        kind=AttemptKind.REVIEWER_RERUN,
        generation=1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        submission_token=f"review-{item_id}",
        job=JobReference(job_id="102"),
        result_digest=_DIGEST_C,
        review_of_attempt_id=coder.attempt_id,
        reviewed_candidate_digest=_DIGEST_B,
    )
    return coder, reviewer


def _catalog_item(*, status: WorkItemStatus = WorkItemStatus.INTEGRATED) -> WorkItemRecord:
    if status is not WorkItemStatus.INTEGRATED:
        return WorkItemRecord(
            item_id="smith-attention",
            stage_id="onboard",
            goal_id="model",
            kind=WorkItemKind.CATALOG_ONBOARD,
            profile=DomainProfile.SMITH,
            status=status,
        )
    coder, reviewer = _validated_attempts("smith-attention")
    return WorkItemRecord(
        item_id="smith-attention",
        stage_id="onboard",
        goal_id="model",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
        status=status,
        attempts=(coder, reviewer),
        candidate_attempt_id=coder.attempt_id,
        candidate_digest=_DIGEST_B,
        reviewer_attempt_id=reviewer.attempt_id,
    )


def _state(
    *,
    dependency_status: WorkItemStatus = WorkItemStatus.INTEGRATED,
    assembler_kind: WorkItemKind = WorkItemKind.ASSEMBLE_CORE,
) -> RunState:
    catalog = _catalog_item(status=dependency_status)
    assembler = WorkItemRecord(
        item_id="assemble-model",
        stage_id="onboard",
        goal_id="model",
        kind=assembler_kind,
        profile=DomainProfile.ASSEMBLER,
        dependencies=(catalog.item_id,),
    )
    return RunState(
        run_id="run-onboard",
        task_digest=_DIGEST_A,
        base_commit="base",
        generation=1,
        stages=(StageRecord("onboard", ("model",)),),
        goals=(GoalRecord("model", "onboard", (catalog.item_id, assembler.item_id)),),
        items=(catalog, assembler),
    )


def _write_router_index(repository: Path, routers: str) -> None:
    path = repository / ROUTER_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        f"MODELING_V2_ROUTERS = {routers}\n",
        encoding="utf-8",
    )


def _write_family_routing(repository: Path, route, *, target: bool) -> None:
    path = repository / route.family_root / "routing.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    if target:
        target_name = route.synthetic_class
        targets = repr({route.route_table_key: target_name})
        modules = repr({target_name: route.target_module})
    else:
        targets = "{}"
        modules = "{}"
    path.write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        f"_SM = {divmod(route.sm, 10)!r}\n"
        f"_CHECKPOINTS = {{(1,): {route.checkpoint_id!r}}}\n"
        f"_TARGETS = {targets}\n"
        f"TARGET_MODULES = {modules}\n",
        encoding="utf-8",
    )


def _write_target(repository: Path, route, *, extra_file: str | None = None) -> None:
    directory = repository / route.target_directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "modeling.py").write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        f'@register_auto_model("{route.synthetic_class}")\n'
        "class Target:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (directory / "weights.py").write_text(
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
        "All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )
    if extra_file is not None:
        (directory / extra_file).write_text("unexpected\n", encoding="utf-8")


def test_route_identity_preserves_full_mapping_and_structural_features() -> None:
    mapping = ParallelMapping(
        tensor_parallel_size=8,
        pipeline_parallel_size=2,
        moe_expert_parallel_size=4,
        moe_tensor_parallel_size=2,
        attention_data_parallel_size=2,
    )
    route = _route(
        architecture="Qwen3MoeForCausalLM",
        family="qwen3_moe",
        checkpoint_id="qwen3_30b_a3b",
        sm=100,
        mapping=mapping,
        features=("mtp-3", "eplb"),
    )
    assert route.mapping.world_size == 16
    assert route.mapping.moe_expert_parallel_size == 4
    assert route.mapping.moe_tensor_parallel_size == 2
    assert route.mapping.attention_data_parallel_size == 2
    assert route.structural_features == ("eplb", "mtp_3")
    assert route.target_segment == "tp8_pp2_ep4_mtp2_adp2_eplb_mtp_3"
    assert route.route_table_key == ("qwen3_30b_a3b", route.target_segment)
    assert str(route.target_directory).endswith(f"/sm_100/{route.target_segment}")


def test_route_identity_recognizes_attention_dp_and_rejects_conflicting_path() -> None:
    route = _route(mapping=ParallelMapping(4, 1, 4, 1, 4))
    assert route.parallel_segment == "dep4"
    with pytest.raises(OnboardingContractError, match="remain unique"):
        _route(features=("mtp-3", "mtp_3"))
    with pytest.raises(OnboardingContractError, match="conflicts with derived target"):
        _route(
            expected_route=(
                "tensorrt_llm/_torch/modeling_v2/models/gpt_oss/targets/"
                "another_checkpoint/sm_103/tp1"
            )
        )


def test_existing_gpt_oss_target_reconciles_as_shadow_noop() -> None:
    reconciliation = reconcile_modeling_v2_target(_REPOSITORY_ROOT, _route())
    assert reconciliation.disposition is ReconciliationDisposition.SHADOW_NOOP
    assert reconciliation.synthetic_class == "ModelingV2GptOss120bSm103Tp1"
    assert reconciliation.proposed_paths == ()
    assert reconciliation.new_architecture_family is False


def test_synthetic_missing_target_distinguishes_new_and_existing_family(tmp_path: Path) -> None:
    new_route = _route(
        architecture="SyntheticForCausalLM",
        family="synthetic",
        checkpoint_id="synthetic_7b",
    )
    _write_router_index(tmp_path, "{}")
    missing_family = reconcile_modeling_v2_target(tmp_path, new_route)
    assert missing_family.disposition is ReconciliationDisposition.MISSING_TARGET
    assert missing_family.new_architecture_family is True
    assert ROUTER_INDEX_PATH in missing_family.proposed_paths
    assert set(new_route.target_products).issubset(missing_family.proposed_paths)

    existing_route = _route(
        architecture="SyntheticForCausalLM",
        family="synthetic",
        checkpoint_id="synthetic_7b",
    )
    _write_router_index(
        tmp_path,
        "{'SyntheticForCausalLM': 'models.synthetic.routing'}",
    )
    _write_family_routing(tmp_path, existing_route, target=False)
    missing_target = reconcile_modeling_v2_target(tmp_path, existing_route)
    assert missing_target.new_architecture_family is False
    assert ROUTER_INDEX_PATH not in missing_target.proposed_paths


def test_existing_family_cannot_be_smuggled_into_router_index(tmp_path: Path) -> None:
    route = _route(
        architecture="AliasForCausalLM",
        family="synthetic",
        checkpoint_id="synthetic_7b",
    )
    _write_router_index(tmp_path, "{}")
    (tmp_path / route.family_root).mkdir(parents=True)
    with pytest.raises(OnboardingContractError, match="genuinely new architecture family"):
        reconcile_modeling_v2_target(tmp_path, route)


def test_synthetic_class_path_table_and_two_product_contract_are_enforced(
    tmp_path: Path,
) -> None:
    route = _route(
        architecture="SyntheticForCausalLM",
        family="synthetic",
        checkpoint_id="synthetic_7b",
    )
    _write_router_index(
        tmp_path,
        "{'SyntheticForCausalLM': 'models.synthetic.routing'}",
    )
    _write_family_routing(tmp_path, route, target=True)
    _write_target(tmp_path, route, extra_file="smoke.py")
    with pytest.raises(OnboardingContractError, match="non-product files"):
        reconcile_modeling_v2_target(tmp_path, route)

    (tmp_path / route.target_directory / "smoke.py").unlink()
    reconciliation = reconcile_modeling_v2_target(tmp_path, route)
    assert reconciliation.disposition is ReconciliationDisposition.SHADOW_NOOP

    routing_path = tmp_path / route.family_root / "routing.py"
    source = routing_path.read_text(encoding="utf-8")
    routing_path.write_text(
        source.replace(route.target_module, "models.synthetic.targets.wrong.modeling"),
        encoding="utf-8",
    )
    with pytest.raises(OnboardingContractError, match="do not agree"):
        reconcile_modeling_v2_target(tmp_path, route)


def test_assembler_consumes_only_integrated_catalog_dependencies() -> None:
    requirement = CatalogRequirement("example-attention", _CATALOG_PATH, "forward needs attention")
    surface = IntegratedCatalogSurface("smith-attention", "example-attention", _CATALOG_PATH)
    preparation = prepare_assembler_input(
        _state(),
        "assemble-model",
        _route(),
        AssemblerUnit.CORE,
        (requirement,),
        (surface,),
    )
    assert isinstance(preparation.assembler_input, CoreAssemblerInput)
    assert preparation.catalog_gaps == ()
    assert preparation.assembler_input.scope.catalog_surfaces == (surface,)

    wrong_surface = IntegratedCatalogSurface(
        "smith-attention",
        "example-attention",
        PurePosixPath("tensorrt_llm/_torch/modeling_v2/catalog/attention/different_attention.py"),
    )
    with pytest.raises(OnboardingContractError, match="unexpected surface"):
        prepare_assembler_input(
            _state(),
            "assemble-model",
            _route(),
            AssemblerUnit.CORE,
            (requirement,),
            (wrong_surface,),
        )

    with pytest.raises(DependencyNotIntegratedError, match="non-integrated dependencies"):
        prepare_assembler_input(
            _state(dependency_status=WorkItemStatus.PLANNED),
            "assemble-model",
            _route(),
            AssemblerUnit.CORE,
            (requirement,),
            (surface,),
        )


def test_missing_catalog_surface_becomes_typed_gap_not_assembler_scope() -> None:
    requirement = CatalogRequirement("missing-attention", _CATALOG_PATH, "no certified surface")
    preparation = prepare_assembler_input(
        _state(),
        "assemble-model",
        _route(),
        AssemblerUnit.CORE,
        (requirement,),
    )
    assert preparation.assembler_input is None
    assert len(preparation.catalog_gaps) == 1
    gap = preparation.catalog_gaps[0]
    assert gap.consumer_item_id == "assemble-model"
    assert gap.kind is WorkItemKind.CATALOG_ONBOARD
    assert gap.entry_id == "missing-attention"


def test_typed_core_feature_weights_and_routing_outputs_are_path_limited() -> None:
    core_preparation = prepare_assembler_input(
        _state(), "assemble-model", _route(), AssemblerUnit.CORE
    )
    core = core_preparation.assembler_input
    assert isinstance(core, CoreAssemblerInput)
    modeling_path, weights_path = core.scope.route.target_products
    core_output = validate_assembler_output(core, (modeling_path,))
    assert core_output.unit is AssemblerUnit.CORE
    with pytest.raises(OnboardingContractError, match="output paths"):
        validate_assembler_output(core, (_CATALOG_PATH,))

    weights_preparation = prepare_assembler_input(
        _state(), "assemble-model", _route(), AssemblerUnit.WEIGHTS
    )
    weights = weights_preparation.assembler_input
    assert isinstance(weights, WeightsAssemblerInput)
    assert validate_assembler_output(weights, (weights_path,)).unit is AssemblerUnit.WEIGHTS

    feature_route = _route(features=("mtp3",))
    feature_preparation = prepare_assembler_input(
        _state(assembler_kind=WorkItemKind.ASSEMBLE_FEATURE),
        "assemble-model",
        feature_route,
        AssemblerUnit.FEATURE,
        feature="mtp3",
    )
    feature = feature_preparation.assembler_input
    assert isinstance(feature, FeatureAssemblerInput)
    feature_products = feature.scope.route.target_products
    assert validate_assembler_output(feature, feature_products).unit is AssemblerUnit.FEATURE

    routing_preparation = prepare_assembler_input(
        _state(assembler_kind=WorkItemKind.ROUTING),
        "assemble-model",
        _route(),
        AssemblerUnit.ROUTING,
        new_architecture_family=False,
    )
    routing = routing_preparation.assembler_input
    assert isinstance(routing, RoutingAssemblerInput)
    routing_path = routing.scope.route.family_root / "routing.py"
    assert validate_assembler_output(routing, (routing_path,)).unit is AssemblerUnit.ROUTING
    with pytest.raises(OnboardingContractError, match="exactly when onboarding"):
        validate_assembler_output(
            routing,
            (routing_path, ROUTER_INDEX_PATH),
            updates_router_index=True,
        )

    new_family_routing = RoutingAssemblerInput(routing.scope, new_architecture_family=True)
    new_family_output = validate_assembler_output(
        new_family_routing,
        (routing_path, ROUTER_INDEX_PATH),
        updates_router_index=True,
    )
    assert new_family_output.updates_router_index is True
