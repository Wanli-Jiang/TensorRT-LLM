# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact shell-free rank launcher contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.gates import (
    ClaimScope,
    EvidenceScope,
    GateCommand,
    GatePolicyError,
    GatePurpose,
    validate_receipt_claim,
)
from agent_flow.workflows.staircase.common.launchers import (
    AllocationNode,
    ExpectedProductIdentity,
    LaunchKind,
    LaunchPolicyError,
    ObservedProductIdentity,
    ObservedRank,
    RankAllocation,
    RankBinding,
    RankPlacementReceipt,
    build_single_node_multi_rank_plan,
    build_single_process_plan,
    build_slurm_multi_node_product_plan,
    build_synthetic_multi_node_plan,
    make_rank_placement_receipt,
)
from agent_flow.workflows.staircase.task_schema import ParallelMapping

_RANK_INPUT = Path("/shared/staircase/attempt/rank-input.json")


def _mapping(world_size: int) -> ParallelMapping:
    return ParallelMapping(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=1,
        moe_tensor_parallel_size=world_size,
        attention_data_parallel_size=1,
    )


def _command(
    *, environment: tuple[tuple[str, str], ...] = (("TRTLLM_MODELING_V2", "require"),)
) -> GateCommand:
    return GateCommand(
        argv=("python3", "-m", "pytest", "tests/product.py::test_collective"),
        environment=environment,
    )


def _observations(plan: object) -> tuple[ObservedRank, ...]:
    return tuple(
        ObservedRank(
            rank=placement.rank,
            hostname=placement.hostname,
            local_rank=placement.local_rank,
            gpu_id=placement.gpu_id,
            product_identity=(
                ObservedProductIdentity(
                    **plan.expected_product_identity.to_dict(),
                    cuda_device_uuid=f"GPU-{placement.rank}",
                    cuda_device_model="Test GPU",
                    body_evidence_digest="a" * 64,
                )
                if plan.expected_product_identity is not None
                else None
            ),
        )
        for placement in plan.placements
    )


def test_single_process_clears_environment_and_pins_one_gpu() -> None:
    allocation = RankAllocation(nodes=(AllocationNode("node-a", (3,)),))
    plan = build_single_process_plan(
        mapping=_mapping(1),
        world_size=1,
        allocation=allocation,
        command=_command(
            environment=(("LLM_MODELS_ROOT", "/models"), ("TRTLLM_MODELING_V2", "require"))
        ),
        rank_input_path=_RANK_INPUT,
    )

    assert plan.kind is LaunchKind.SINGLE_PROCESS
    assert plan.argv[:2] == ("/usr/bin/env", "-i")
    assert "CUDA_VISIBLE_DEVICES=3" in plan.argv
    assert "RANK=0" in plan.argv
    assert "TRTLLM_MODELING_V2=require" in plan.argv
    assert plan.argv[-3:] == ("rank-worker", "--input", str(_RANK_INPUT))
    assert plan.rank_command.argv == (
        "python3",
        "-m",
        "pytest",
        "tests/product.py::test_collective",
    )
    assert plan.placements[0].gpu_id == 3


def test_single_node_multi_rank_uses_exact_slurm_step_and_explicit_export() -> None:
    allocation = RankAllocation(
        nodes=(AllocationNode("node-a", (2, 4, 5, 7)),), slurm_job_id="1234"
    )
    plan = build_single_node_multi_rank_plan(
        mapping=_mapping(4),
        world_size=4,
        allocation=allocation,
        command=_command(),
        rank_input_path=_RANK_INPUT,
        cpus_per_rank=6,
    )

    assert plan.argv[:3] == ("srun", "--jobid=1234", "--exact")
    assert "--nodes=1" in plan.argv
    assert "--ntasks=4" in plan.argv
    assert "--ntasks-per-node=4" in plan.argv
    assert "--cpus-per-task=6" in plan.argv
    assert "--nodelist=node-a" in plan.argv
    assert "--gpu-bind=map_gpu:2,4,5,7" in plan.argv
    assert "--export=TRTLLM_MODELING_V2=require" in plan.argv
    assert not any("ALL" in argument for argument in plan.argv if argument.startswith("--export="))
    assert plan.argv[-3:] == ("rank-worker", "--input", str(_RANK_INPUT))
    receipt = make_rank_placement_receipt(plan, _observations(plan))
    validate_receipt_claim(
        receipt.to_gate_receipt(gate_id="collective", purpose=GatePurpose.CORRECTNESS, passed=True),
        ClaimScope.LOCAL_FOUR_GPU_PRODUCT,
    )


def test_non_four_gpu_single_node_receipt_does_not_invent_gate_scope() -> None:
    allocation = RankAllocation(nodes=(AllocationNode("node-a", (0, 1)),), slurm_job_id="1234")
    plan = build_single_node_multi_rank_plan(
        mapping=_mapping(2),
        world_size=2,
        allocation=allocation,
        command=_command(),
        rank_input_path=_RANK_INPUT,
    )
    receipt = make_rank_placement_receipt(plan, _observations(plan))

    with pytest.raises(GatePolicyError, match="no gate evidence scope"):
        receipt.to_gate_receipt(gate_id="collective", purpose=GatePurpose.CORRECTNESS, passed=True)


def test_synthetic_multi_node_receipt_can_only_certify_runner() -> None:
    allocation = RankAllocation(
        nodes=(AllocationNode("node-a", (0,)), AllocationNode("node-b", (0,))),
        slurm_job_id="5678",
    )
    plan = build_synthetic_multi_node_plan(
        mapping=_mapping(2),
        world_size=2,
        allocation=allocation,
        command=_command(),
        rank_input_path=_RANK_INPUT,
    )
    receipt = make_rank_placement_receipt(plan, _observations(plan)).to_gate_receipt(
        gate_id="runner-canary", purpose=GatePurpose.CORRECTNESS, passed=True
    )

    assert plan.kind is LaunchKind.SYNTHETIC_MULTI_NODE_RUNNER
    assert not plan.product_rank_body
    validate_receipt_claim(receipt, ClaimScope.SYNTHETIC_RUNNER)
    with pytest.raises(GatePolicyError, match="real topology-bearing"):
        validate_receipt_claim(receipt, ClaimScope.MULTI_NODE_PRODUCT)

    with pytest.raises(LaunchPolicyError, match="evidence-scope boundary"):
        RankPlacementReceipt(
            schema_version=1,
            plan_digest=plan.plan_digest,
            kind=LaunchKind.SYNTHETIC_MULTI_NODE_RUNNER,
            world_size=2,
            placements=(
                RankBinding(0, "node-a", 0, 0),
                RankBinding(1, "node-b", 0, 0),
            ),
            product_rank_body=True,
            evidence_scope=EvidenceScope.REAL_MULTI_NODE_PRODUCT,
        )


def test_real_multi_node_product_plan_supports_only_real_product_receipt() -> None:
    allocation = RankAllocation(
        nodes=(AllocationNode("node-a", (1, 3)), AllocationNode("node-b", (1, 3))),
        slurm_job_id="9012",
    )
    plan = build_slurm_multi_node_product_plan(
        mapping=_mapping(4),
        world_size=4,
        allocation=allocation,
        command=_command(),
        rank_input_path=_RANK_INPUT,
        expected_product_identity=ExpectedProductIdentity(
            repository_commit="1" * 40,
            python_tensorrt_llm_path="/workspace/tensorrt_llm/__init__.py",
            native_build_identity="build-1",
            image_identity="image-1",
            compute_capability="10.0",
            collective_backend="nccl",
            transport="ib",
        ),
    )
    receipt = make_rank_placement_receipt(plan, reversed(_observations(plan))).to_gate_receipt(
        gate_id="collective", purpose=GatePurpose.CORRECTNESS, passed=True
    )

    assert plan.kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT
    assert plan.product_rank_body
    assert [(item.rank, item.node, item.local_rank) for item in receipt.placements] == [
        (0, "node-a", 0),
        (1, "node-a", 1),
        (2, "node-b", 0),
        (3, "node-b", 1),
    ]
    validate_receipt_claim(receipt, ClaimScope.MULTI_NODE_PRODUCT)


def test_real_product_receipt_requires_per_rank_identity_and_distinct_cuda_uuid() -> None:
    identity = ExpectedProductIdentity(
        repository_commit="1" * 40,
        python_tensorrt_llm_path="/workspace/tensorrt_llm/__init__.py",
        native_build_identity="build-1",
        image_identity="image-1",
        compute_capability="10.0",
        collective_backend="nccl",
        transport="ib",
    )
    plan = build_slurm_multi_node_product_plan(
        mapping=_mapping(2),
        world_size=2,
        allocation=RankAllocation(
            (AllocationNode("node-a", (0,)), AllocationNode("node-b", (0,))),
            slurm_job_id="9012",
        ),
        command=_command(),
        rank_input_path=_RANK_INPUT,
        expected_product_identity=identity,
    )

    with pytest.raises(LaunchPolicyError, match="identity proof from every rank"):
        make_rank_placement_receipt(
            plan,
            tuple(
                ObservedRank(
                    binding.rank,
                    binding.hostname,
                    binding.local_rank,
                    binding.gpu_id,
                )
                for binding in plan.placements
            ),
        )

    repeated_uuid = ObservedProductIdentity(
        **identity.to_dict(),
        cuda_device_uuid="GPU-repeated",
        cuda_device_model="Test GPU",
        body_evidence_digest="a" * 64,
    )
    with pytest.raises(LaunchPolicyError, match="distinct CUDA UUID"):
        make_rank_placement_receipt(
            plan,
            tuple(
                ObservedRank(
                    binding.rank,
                    binding.hostname,
                    binding.local_rank,
                    binding.gpu_id,
                    repeated_uuid,
                )
                for binding in plan.placements
            ),
        )


@pytest.mark.parametrize(
    ("world_size", "allocation", "match"),
    [
        (4, RankAllocation((AllocationNode("node-a", (0, 1)),), "123"), "GPU count"),
        (2, RankAllocation((AllocationNode("node-a", (0, 1)),), "123"), "Mapping"),
    ],
)
def test_world_size_must_match_mapping_and_allocation(
    world_size: int, allocation: RankAllocation, match: str
) -> None:
    mapping = _mapping(4)
    with pytest.raises(LaunchPolicyError, match=match):
        build_single_node_multi_rank_plan(
            mapping=mapping,
            world_size=world_size,
            allocation=allocation,
            command=_command(),
            rank_input_path=_RANK_INPUT,
        )


def test_multi_rank_requires_exact_slurm_job_and_symmetric_gpu_map() -> None:
    without_job = RankAllocation(
        nodes=(AllocationNode("node-a", (0,)), AllocationNode("node-b", (0,)))
    )
    with pytest.raises(LaunchPolicyError, match="exact Slurm allocation"):
        build_synthetic_multi_node_plan(
            mapping=_mapping(2),
            world_size=2,
            allocation=without_job,
            command=_command(),
            rank_input_path=_RANK_INPUT,
        )

    asymmetric = RankAllocation(
        nodes=(AllocationNode("node-a", (0,)), AllocationNode("node-b", (1,))),
        slurm_job_id="123",
    )
    with pytest.raises(LaunchPolicyError, match="same GPU ID map"):
        build_synthetic_multi_node_plan(
            mapping=_mapping(2),
            world_size=2,
            allocation=asymmetric,
            command=_command(),
            rank_input_path=_RANK_INPUT,
        )


@pytest.mark.parametrize("hostname", ("node*", "node[1-2]", "node,a", "-node"))
def test_allocation_rejects_wildcard_or_non_exact_hosts(hostname: str) -> None:
    with pytest.raises(LaunchPolicyError, match="hostname"):
        AllocationNode(hostname, (0,))


@pytest.mark.parametrize("program", ("bash", "/bin/sh", "ssh", "mpirun", "mpiexec", "srun"))
def test_rank_body_rejects_shell_ssh_and_nested_launchers(program: str) -> None:
    command = GateCommand(argv=(program, "payload"))
    with pytest.raises(LaunchPolicyError, match="not a shell, SSH, or nested rank launcher"):
        build_single_process_plan(
            mapping=_mapping(1),
            world_size=1,
            allocation=RankAllocation((AllocationNode("node-a", (0,)),)),
            command=command,
            rank_input_path=_RANK_INPUT,
        )


@pytest.mark.parametrize(
    "environment",
    (
        (("TRTLLM_MODELING_V2", "require"), ("API_TOKEN", "secret")),
        (("TRTLLM_MODELING_V2", "require"), ("RANK", "0")),
        (("TRTLLM_MODELING_V2", "require"), ("LLM_MODELS_ROOT", "/a,/b")),
    ),
)
def test_environment_rejects_secrets_rank_spoofing_and_export_delimiters(
    environment: tuple[tuple[str, str], ...],
) -> None:
    command = GateCommand(argv=("python3", "worker.py"), environment=environment)
    with pytest.raises(LaunchPolicyError):
        build_single_process_plan(
            mapping=_mapping(1),
            world_size=1,
            allocation=RankAllocation((AllocationNode("node-a", (0,)),)),
            command=command,
            rank_input_path=_RANK_INPUT,
        )


def test_receipt_rejects_missing_duplicate_or_misplaced_rank() -> None:
    allocation = RankAllocation(
        nodes=(AllocationNode("node-a", (0,)), AllocationNode("node-b", (0,))),
        slurm_job_id="123",
    )
    plan = build_synthetic_multi_node_plan(
        mapping=_mapping(2),
        world_size=2,
        allocation=allocation,
        command=_command(),
        rank_input_path=_RANK_INPUT,
    )

    with pytest.raises(LaunchPolicyError, match="differs from plan"):
        make_rank_placement_receipt(plan, _observations(plan)[:1])
    with pytest.raises(LaunchPolicyError, match="duplicate rank"):
        make_rank_placement_receipt(plan, (_observations(plan)[0], _observations(plan)[0]))
    with pytest.raises(LaunchPolicyError, match="differs from plan"):
        make_rank_placement_receipt(
            plan,
            (
                _observations(plan)[0],
                ObservedRank(rank=1, hostname="node-b", local_rank=0, gpu_id=7),
            ),
        )


def test_plan_digest_changes_with_rank_body() -> None:
    allocation = RankAllocation((AllocationNode("node-a", (0,)),))
    first = build_single_process_plan(
        mapping=_mapping(1),
        world_size=1,
        allocation=allocation,
        command=_command(),
        rank_input_path=_RANK_INPUT,
    )
    second = build_single_process_plan(
        mapping=_mapping(1),
        world_size=1,
        allocation=allocation,
        command=GateCommand(argv=("python3", "other.py")),
        rank_input_path=_RANK_INPUT,
    )

    assert first.plan_digest != second.plan_digest
