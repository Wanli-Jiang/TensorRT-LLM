# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the private in-allocation rank supervisor worker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.artifacts import (
    WorkerInputManifest,
    WorkerResultStatus,
    load_result_manifest,
    write_input_manifest,
)
from agent_flow.workflows.staircase.common.gates import EvidenceScope
from agent_flow.workflows.staircase.common.launchers import (
    AllocationNode,
    ExpectedProductIdentity,
    ObservedRank,
    RankAllocation,
    make_rank_placement_receipt,
)
from agent_flow.workflows.staircase.common.rank_supervisor_worker import (
    RankSupervisorWorkerError,
    _slurm_allocation,
    execute_rank_supervisor,
    load_rank_supervisor_input,
)
from agent_flow.workflows.staircase.common.slurm import InternalCommand, InternalEntrypoint
from agent_flow.workflows.staircase.common.supervisor import (
    RankSupervisionResult,
    RankSupervisorError,
)
from agent_flow.workflows.staircase.state import DomainProfile, Role


def _expected_identity(worktree: Path) -> ExpectedProductIdentity:
    return ExpectedProductIdentity(
        repository_commit="2" * 40,
        python_tensorrt_llm_path=str(worktree / "tensorrt_llm" / "__init__.py"),
        native_build_identity="build-1",
        image_identity="image-1",
        compute_capability="10.0",
        collective_backend="nccl",
        transport="ib",
    )


def _input(tmp_path: Path, scope: EvidenceScope) -> Path:
    worktree = tmp_path / "worktree"
    attempt_dir = tmp_path / "attempt"
    output_dir = attempt_dir / "output"
    worktree.mkdir()
    output_dir.mkdir(parents=True)
    product = scope in {
        EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
        EvidenceScope.REAL_MULTI_NODE_PRODUCT,
    }
    local_product = scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT
    world_size = 4 if local_product else 2
    nodes = 1 if local_product else 2
    ranks_per_node = 4 if local_product else 1
    identity = (
        _expected_identity(worktree) if scope is EvidenceScope.REAL_MULTI_NODE_PRODUCT else None
    )
    runtime = {
        "schema_version": 2,
        "kind": "rank_supervisor",
        "certification_mode": {
            EvidenceScope.LOCAL_FOUR_GPU_PRODUCT: "LOCAL",
            EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER: "SYNTHETIC",
            EvidenceScope.REAL_MULTI_NODE_PRODUCT: "REAL",
        }[scope],
        "candidate": {
            "attempt_id": "item.0001.role",
            "digest": "d" * 64,
            "commit": "2" * 40,
            "receipt_sha256": "e" * 64,
        },
        "gate": {
            "gate_id": "collective",
            "phase": 40,
            "purpose": "correctness",
            "command": {
                "argv": ["python3", "collective.py"],
                "environment": {"TRTLLM_MODELING_V2": "require"},
            },
            "hard_gate": True,
            "accuracy": None,
        },
        "execution": {
            "scope": scope.value,
            "expected_world_size": world_size,
            "expected_rank": None,
            "expected_local_rank": None,
            "product_rank_body": product,
        },
        "mapping": {
            "tensor_parallel_size": world_size,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 1,
            "moe_tensor_parallel_size": world_size,
            "attention_data_parallel_size": 1,
        },
        "resource": {
            "name": "deterministic_gate",
            "nodes": nodes,
            "tasks_per_node": ranks_per_node,
            "gpus_per_node": ranks_per_node,
            "cpus_per_task": 4,
            "memory_mib": 4096,
            "time_limit_seconds": 600,
        },
        "rank_input_path": str(output_dir / "rank-input.json"),
        "rank_report_directory": str(output_dir / "rank-reports"),
        "expected_product_identity": identity.to_dict() if identity is not None else None,
    }
    manifest = WorkerInputManifest(
        run_id="run-1",
        item_id="item",
        attempt_id="item.0002.deterministic_gate",
        task_digest="a" * 64,
        generation=1,
        role=Role.GATE,
        profile=DomainProfile.ASSEMBLER,
        worktree=str(worktree),
        payload={"runtime": runtime, "context": {}},
    )
    input_path = attempt_dir / "input.json"
    write_input_manifest(input_path, manifest)
    return input_path


def _allocation(_resource: object, _environment: object) -> RankAllocation:
    return RankAllocation(
        (AllocationNode("node-a", (0,)), AllocationNode("node-b", (0,))),
        slurm_job_id="1234",
    )


def _local_allocation(_resource: object, _environment: object) -> RankAllocation:
    return RankAllocation(
        (AllocationNode("node-a", (0, 1, 2, 3)),),
        slurm_job_id="1234",
    )


def test_synthetic_supervisor_publishes_normal_atomic_worker_result(tmp_path: Path) -> None:
    input_path = _input(tmp_path, EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER)
    loaded = load_rank_supervisor_input(input_path)

    def supervisor(plan: object, *, cwd: Path) -> RankSupervisionResult:
        assert cwd == Path(loaded.manifest.worktree)
        observations = tuple(
            ObservedRank(
                binding.rank,
                binding.hostname,
                binding.local_rank,
                binding.gpu_id,
            )
            for binding in plan.placements
        )
        return RankSupervisionResult(
            make_rank_placement_receipt(plan, observations), b"runner output", b""
        )

    execute_rank_supervisor(
        input_path,
        environment={"SLURM_JOB_ID": "1234"},
        allocation_provider=_allocation,
        supervisor=supervisor,
    )

    result, _digest = load_result_manifest(input_path.parent / "output")
    receipt = result.payload["receipt"]
    assert result.status is WorkerResultStatus.SUCCEEDED
    assert result.payload["certification_mode"] == "SYNTHETIC"
    assert isinstance(receipt, dict)
    assert receipt["scope"] == "synthetic_multi_node_runner"
    assert receipt["passed"] is True
    diagnostic = json.loads((input_path.parent / "output" / "rank-supervision.json").read_text())
    assert diagnostic["identity_provenance"]["product_reported"] == [
        "collective_backend",
        "transport",
    ]
    assert (input_path.parent / "output" / "COMPLETE").is_file()


def test_product_identity_failure_rejects_without_real_scope(tmp_path: Path) -> None:
    input_path = _input(tmp_path, EvidenceScope.REAL_MULTI_NODE_PRODUCT)

    def rejecting_supervisor(_plan: object, *, cwd: Path) -> RankSupervisionResult:
        _ = cwd
        raise RankSupervisorError("rank identity evidence is missing")

    execute_rank_supervisor(
        input_path,
        environment={"SLURM_JOB_ID": "1234"},
        allocation_provider=_allocation,
        supervisor=rejecting_supervisor,
    )

    result, _digest = load_result_manifest(input_path.parent / "output")
    receipt = result.payload["receipt"]
    assert result.status is WorkerResultStatus.REJECTED
    assert result.payload["certification_mode"] == "REAL"
    assert isinstance(receipt, dict)
    assert receipt["scope"] == "cpu_static"
    assert receipt["placements"] == []
    assert receipt["product_rank_body"] is False
    assert receipt["passed"] is False


def test_local_four_gpu_product_never_upgrades_to_multi_node_scope(tmp_path: Path) -> None:
    input_path = _input(tmp_path, EvidenceScope.LOCAL_FOUR_GPU_PRODUCT)

    def supervisor(plan: object, *, cwd: Path) -> RankSupervisionResult:
        _ = cwd
        observations = tuple(
            ObservedRank(
                binding.rank,
                binding.hostname,
                binding.local_rank,
                binding.gpu_id,
            )
            for binding in plan.placements
        )
        return RankSupervisionResult(
            make_rank_placement_receipt(plan, observations), b"local output", b""
        )

    execute_rank_supervisor(
        input_path,
        environment={"SLURM_JOB_ID": "1234"},
        allocation_provider=_local_allocation,
        supervisor=supervisor,
    )

    result, _digest = load_result_manifest(input_path.parent / "output")
    receipt = result.payload["receipt"]
    assert result.status is WorkerResultStatus.SUCCEEDED
    assert result.payload["certification_mode"] == "LOCAL"
    assert isinstance(receipt, dict)
    assert receipt["scope"] == "local_four_gpu_product"
    assert {placement["node"] for placement in receipt["placements"]} == {"node-a"}


def test_internal_rank_supervisor_command_is_fixed_and_shell_free(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    command = InternalCommand(
        InternalEntrypoint.RANK_SUPERVISOR,
        input_bundle=input_path,
    )

    assert command.argv() == (
        "python3",
        "-m",
        "agent_flow.workflows.staircase.internal",
        "rank-supervisor",
        "--input",
        str(input_path),
    )


def test_slurm_allocation_resolves_site_scontrol_before_clearing_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_scontrol = Path("/cm/local/apps/slurm/current/bin/scontrol")
    if not site_scontrol.is_file():
        pytest.skip("site-style Slurm client is unavailable")
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    def run(argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs["env"]))  # type: ignore[arg-type]
        assert kwargs["shell"] is False
        return subprocess.CompletedProcess(argv, 0, "node-a\nnode-b\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    resource = load_rank_supervisor_input(
        _input(tmp_path, EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER)
    ).resource

    allocation = _slurm_allocation(
        resource,
        {
            "PATH": str(site_scontrol.parent),
            "SLURM_JOB_ID": "763209",
            "SLURM_JOB_NODELIST": "node-[a-b]",
        },
    )

    assert allocation.slurm_job_id == "763209"
    assert tuple(node.hostname for node in allocation.nodes) == ("node-a", "node-b")
    assert calls == [
        (
            (
                str(site_scontrol.resolve()),
                "show",
                "hostnames",
                "node-[a-b]",
            ),
            {},
        )
    ]


def test_slurm_allocation_rejects_worker_owned_path_hijack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hijack_dir = tmp_path / "bin"
    hijack_dir.mkdir()
    hijack = hijack_dir / "scontrol"
    hijack.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hijack.chmod(0o755)
    invoked = False

    def run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal invoked
        invoked = True
        raise AssertionError("PATH-hijacked scheduler client was executed")

    monkeypatch.setattr(subprocess, "run", run)
    input_root = tmp_path / "input-root"
    input_root.mkdir()
    resource = load_rank_supervisor_input(
        _input(input_root, EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER)
    ).resource

    with pytest.raises(RankSupervisorWorkerError, match="worker-owned"):
        _slurm_allocation(
            resource,
            {
                "PATH": str(hijack_dir),
                "SLURM_JOB_ID": "763209",
                "SLURM_JOB_NODELIST": "node-[a-b]",
            },
        )

    assert not invoked
