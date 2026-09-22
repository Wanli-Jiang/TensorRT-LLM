# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic ModelingV2 gate plans and evidence scope."""

from __future__ import annotations

import pytest

from agent_flow.workflows.staircase.common.gates import (
    AccuracyCriteria,
    ClaimScope,
    EvidenceScope,
    GateCommand,
    GatePhase,
    GatePolicyError,
    GatePurpose,
    GateReceipt,
    GateSpec,
    RankPlacement,
    build_gate_suite,
    validate_receipt_claim,
)


def _placements(nodes: tuple[str, ...], ranks_per_node: int) -> tuple[RankPlacement, ...]:
    return tuple(
        RankPlacement(rank=rank, node=node, local_rank=local_rank)
        for rank, (node, local_rank) in enumerate(
            (node, local_rank) for node in nodes for local_rank in range(ranks_per_node)
        )
    )


def test_gate_suite_is_shell_free_ordered_and_require_mode() -> None:
    accuracy = AccuracyCriteria(
        selector=(
            "tests/integration/defs/accuracy/test_modeling_v2_qwen3.py::TestQwen::test_gsm8k"
        ),
        reference="gsm8k:stock-qwen3",
        protocol="exact-match-v1",
        tolerance=0.01,
    )
    suite = build_gate_suite(
        entry_gpu_tests=("tests/unittest/_torch/modeling_v2/norm/test_modeling_v2_rms.py",),
        collective_tests=(
            "tests/unittest/_torch/modeling_v2/comm/test_modeling_v2_allgather_op_matrix.py",
        ),
        boot_tests=("tests/integration/defs/models/test_boot.py::test_qwen",),
        accuracy=accuracy,
        feature_signal_tests=(
            "tests/integration/defs/accuracy/test_qwen3.py::test_mtp_acceptance",
        ),
        base_environment={"LLM_MODELS_ROOT": "/models"},
    )

    assert [gate.phase for gate in suite] == [
        GatePhase.CLAIMS_ROUTING_NO_STALE,
        GatePhase.NATIVE_CONTRACT,
        GatePhase.ENTRY_GPU,
        GatePhase.COLLECTIVE,
        GatePhase.BOOT,
        GatePhase.ACCURACY,
        GatePhase.FEATURE_SIGNAL,
    ]
    assert all(gate.command.argv[:3] == ("python3", "-m", "pytest") for gate in suite)
    assert all(dict(gate.command.environment)["TRTLLM_MODELING_V2"] == "require" for gate in suite)
    assert all(dict(gate.command.environment)["LLM_MODELS_ROOT"] == "/models" for gate in suite)
    assert next(gate for gate in suite if gate.phase is GatePhase.ACCURACY).accuracy == accuracy


def test_gate_suite_rejects_missing_release_gates_and_mode_override() -> None:
    with pytest.raises(GatePolicyError, match="boot_tests"):
        build_gate_suite(
            entry_gpu_tests=(),
            collective_tests=(),
            boot_tests=(),
            accuracy=AccuracyCriteria(
                selector="tests/accuracy.py::test_model",
                reference="reference",
                protocol="protocol",
                tolerance=0.0,
            ),
            feature_signal_tests=(),
        )
    with pytest.raises(GatePolicyError, match="cannot be overridden"):
        build_gate_suite(
            entry_gpu_tests=(),
            collective_tests=(),
            boot_tests=("tests/boot.py::test_model",),
            accuracy=AccuracyCriteria(
                selector="tests/accuracy.py::test_model",
                reference="reference",
                protocol="protocol",
                tolerance=0.0,
            ),
            feature_signal_tests=(),
            base_environment={"TRTLLM_MODELING_V2": "auto"},
        )


def test_performance_is_never_a_universal_hard_correctness_gate() -> None:
    with pytest.raises(GatePolicyError, match="cannot be a universal"):
        GateSpec(
            gate_id="throughput",
            phase=GatePhase.PERFORMANCE,
            purpose=GatePurpose.PERFORMANCE,
            command=GateCommand(argv=("benchmark", "--config", "measured.yaml")),
            hard_gate=True,
        )

    performance = GateReceipt(
        gate_id="throughput",
        purpose=GatePurpose.PERFORMANCE,
        scope=EvidenceScope.CPU_STATIC,
        placements=(),
        product_rank_body=False,
        passed=True,
    )
    validate_receipt_claim(performance, ClaimScope.PERFORMANCE)
    with pytest.raises(GatePolicyError, match="cannot support a correctness claim"):
        validate_receipt_claim(performance, ClaimScope.PRODUCT_CORRECTNESS)


@pytest.mark.parametrize(
    "selector",
    (
        "pytest -q tests/accuracy.py::test_model",
        "tests/accuracy.py::test_model --maxfail=1",
        "run the MTP acceptance check",
        "tests/accuracy",
        "../tests/accuracy.py::test_model",
    ),
)
def test_pytest_gates_reject_commands_directories_and_prose(selector: str) -> None:
    with pytest.raises(GatePolicyError, match="selectors|node|commands"):
        AccuracyCriteria(
            selector=selector,
            reference="reference",
            protocol="protocol",
            tolerance=0.0,
        )


def test_feature_signal_prose_is_not_treated_as_an_executable_gate() -> None:
    with pytest.raises(GatePolicyError, match="selectors|commands"):
        build_gate_suite(
            entry_gpu_tests=(),
            collective_tests=(),
            boot_tests=("tests/boot.py::test_model",),
            accuracy=AccuracyCriteria(
                selector="tests/accuracy.py::test_model",
                reference="reference",
                protocol="protocol",
                tolerance=0.0,
            ),
            feature_signal_tests=("graph and eager tokens match",),
        )


def test_accuracy_claim_requires_exact_frozen_acceptance_criteria() -> None:
    criteria = AccuracyCriteria(
        selector="tests/integration/defs/accuracy/test_model.py::test_accuracy",
        reference="gsm8k:golden-v1",
        protocol="exact-match-v1",
        tolerance=0.01,
    )
    suite = build_gate_suite(
        entry_gpu_tests=(),
        collective_tests=(),
        boot_tests=("tests/integration/defs/models/test_boot.py::test_model",),
        accuracy=criteria,
        feature_signal_tests=(),
    )
    spec = next(gate for gate in suite if gate.phase is GatePhase.ACCURACY)
    receipt = GateReceipt(
        gate_id="accuracy",
        purpose=GatePurpose.ACCURACY,
        scope=EvidenceScope.SINGLE_GPU_PRODUCT,
        placements=_placements(("node-a",), 1),
        product_rank_body=True,
        passed=True,
        accuracy=criteria,
    )
    validate_receipt_claim(receipt, ClaimScope.ACCURACY, gate_spec=spec)

    with pytest.raises(GatePolicyError, match="frozen gate specification"):
        validate_receipt_claim(receipt, ClaimScope.ACCURACY)

    mismatched = GateReceipt(
        gate_id="accuracy",
        purpose=GatePurpose.ACCURACY,
        scope=EvidenceScope.SINGLE_GPU_PRODUCT,
        placements=_placements(("node-a",), 1),
        product_rank_body=True,
        passed=True,
        accuracy=AccuracyCriteria(
            selector=criteria.selector,
            reference=criteria.reference,
            protocol=criteria.protocol,
            tolerance=0.02,
        ),
    )
    with pytest.raises(GatePolicyError, match="do not match the frozen task"):
        validate_receipt_claim(mismatched, ClaimScope.ACCURACY, gate_spec=spec)


@pytest.mark.parametrize("tolerance", (-0.01, float("nan"), float("inf"), True))
def test_accuracy_tolerance_is_finite_non_negative(tolerance: float) -> None:
    with pytest.raises(GatePolicyError, match="finite and non-negative"):
        AccuracyCriteria(
            selector="tests/accuracy.py::test_model",
            reference="reference",
            protocol="protocol",
            tolerance=tolerance,
        )


def test_local_four_gpu_evidence_has_one_node_and_exactly_four_product_ranks() -> None:
    receipt = GateReceipt(
        gate_id="collective",
        purpose=GatePurpose.CORRECTNESS,
        scope=EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
        placements=_placements(("node-a",), 4),
        product_rank_body=True,
        passed=True,
    )
    validate_receipt_claim(receipt, ClaimScope.LOCAL_FOUR_GPU_PRODUCT)
    validate_receipt_claim(receipt, ClaimScope.PRODUCT_CORRECTNESS)

    with pytest.raises(GatePolicyError, match="four product ranks on one node"):
        GateReceipt(
            gate_id="collective",
            purpose=GatePurpose.CORRECTNESS,
            scope=EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
            placements=_placements(("node-a", "node-b"), 2),
            product_rank_body=True,
            passed=True,
        )


def test_synthetic_multi_node_canary_cannot_certify_product() -> None:
    receipt = GateReceipt(
        gate_id="runner-canary",
        purpose=GatePurpose.CORRECTNESS,
        scope=EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
        placements=_placements(("node-a", "node-b"), 1),
        product_rank_body=False,
        passed=True,
    )
    validate_receipt_claim(receipt, ClaimScope.SYNTHETIC_RUNNER)
    with pytest.raises(GatePolicyError, match="real topology-bearing product receipt"):
        validate_receipt_claim(receipt, ClaimScope.MULTI_NODE_PRODUCT)
    with pytest.raises(GatePolicyError, match="cannot certify product correctness"):
        validate_receipt_claim(receipt, ClaimScope.PRODUCT_CORRECTNESS)


def test_multi_node_product_claim_requires_topology_bearing_product_receipt() -> None:
    receipt = GateReceipt(
        gate_id="collective",
        purpose=GatePurpose.CORRECTNESS,
        scope=EvidenceScope.REAL_MULTI_NODE_PRODUCT,
        placements=_placements(("node-a", "node-b"), 2),
        product_rank_body=True,
        passed=True,
    )
    validate_receipt_claim(receipt, ClaimScope.MULTI_NODE_PRODUCT)

    with pytest.raises(GatePolicyError, match="topology-bearing product rank placements"):
        GateReceipt(
            gate_id="collective",
            purpose=GatePurpose.CORRECTNESS,
            scope=EvidenceScope.REAL_MULTI_NODE_PRODUCT,
            placements=_placements(("node-a",), 4),
            product_rank_body=True,
            passed=True,
        )


def test_failed_receipt_never_supports_a_claim() -> None:
    receipt = GateReceipt(
        gate_id="claims",
        purpose=GatePurpose.STRUCTURE,
        scope=EvidenceScope.CPU_STATIC,
        placements=(),
        product_rank_body=False,
        passed=False,
    )
    with pytest.raises(GatePolicyError, match="failed gate"):
        validate_receipt_claim(receipt, ClaimScope.STRUCTURAL)
