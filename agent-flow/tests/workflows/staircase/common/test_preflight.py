# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure Staircase production preflight policy."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from agent_flow.workflows.staircase.common.dispatch_probe import (
    DispatchContract,
    DispatchProbeResult,
    probe_dispatch_capability,
)
from agent_flow.workflows.staircase.common.preflight import (
    ObservedMount,
    PreflightError,
    PreflightPhase,
    PreflightSnapshot,
    required_commands,
    validate_production_preflight,
)
from agent_flow.workflows.staircase.task_schema import (
    AccuracyGate,
    AgentExecutionConfig,
    CertificationConfig,
    CertificationMode,
    ControllerConfig,
    DeliveryConfig,
    ExecutionConfig,
    GatesConfig,
    MountConfig,
    NormalizedTask,
    OverrideBounds,
    ParallelMapping,
    ReferenceConfig,
    RepositoryConfig,
    ResourceClass,
    RetryPolicy,
    SlurmConfig,
    SmithConfig,
    TargetConfig,
)

_BASE_COMMIT = "a" * 40
_BUILD_IDENTITY = "trtllm-local-dev@aabbccdd"
_RESOURCE_NAMES = (
    "coder_analysis",
    "exploratory_probe",
    "deterministic_gate",
    "reviewer_analysis",
    "reviewer_rerun",
)


def _task(
    tmp_path: Path,
    *,
    dispatch_mode: Literal[
        "nested_submission", "login_dispatcher", "preallocated_pool"
    ] = "nested_submission",
) -> NormalizedTask:
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    checkpoint = tmp_path / "checkpoint"
    image = tmp_path / "image.sqsh"
    for directory in (repository, workspace, checkpoint):
        directory.mkdir(parents=True)
    image.touch()
    resources = tuple(
        ResourceClass(
            name=name,
            nodes=1,
            tasks_per_node=2 if name in {"deterministic_gate", "reviewer_rerun"} else 1,
            gpus_per_node=2 if name in {"deterministic_gate", "reviewer_rerun"} else 1,
            cpus_per_task=2,
            memory_mib=4_096,
            time_limit_seconds=1_200,
        )
        for name in _RESOURCE_NAMES
    )
    controller = ControllerConfig(
        account="coreai",
        partition="batch",
        qos=None,
        reservation=None,
        time_limit_seconds=3_600,
        cpus_per_task=2,
        memory_mib=4_096,
        image=image.resolve(),
        mounts=(MountConfig(tmp_path.resolve(), "/job", False),),
        environment=(),
        build_identity=_BUILD_IDENTITY,
        dispatch_mode=dispatch_mode,
        requeue=True,
        advance_signal_lead_seconds=60,
        heartbeat_timeout_seconds=30,
        lease_timeout_seconds=90,
        orphan_grace_seconds=120,
    )
    smith = SmithConfig(
        max_parallel_items=2,
        max_nodes_total=4,
        max_gpus_total=8,
        resource_classes=resources,
        per_item_override_bounds=OverrideBounds(2, 2, 4, 8, 16_384, 3_600),
        retry_policy=RetryPolicy(1, 1),
    )
    return NormalizedTask(
        schema_version=1,
        repository=RepositoryConfig(
            repository.resolve(), _BASE_COMMIT, "reject", workspace.resolve()
        ),
        reference=ReferenceConfig(checkpoint.resolve(), "fixture", "ExampleModel", ()),
        target=TargetConfig(
            "example",
            "example_tp2",
            100,
            2,
            ParallelMapping(2, 1, 1, 1, 1),
            (),
            "tensorrt_llm/_torch/modeling_v2/models/example",
            True,
        ),
        gates=GatesConfig(
            AccuracyGate("tests/accuracy.py::test_smoke", "fixture", "greedy", 0.0),
            (),
            (),
            (),
            (),
        ),
        certification=CertificationConfig(CertificationMode.LOCAL, None),
        execution=ExecutionConfig(
            "slurm", SlurmConfig(controller, smith), AgentExecutionConfig("codex", "test-model")
        ),
        delivery=DeliveryConfig("diff_only", None, False, False),
        digest="b" * 64,
    )


def _snapshot(task: NormalizedTask, phase: PreflightPhase) -> PreflightSnapshot:
    mount = task.execution.slurm.controller.mounts[0]
    return PreflightSnapshot(
        phase=phase,
        repository_root=task.repository.root,
        workspace_root=task.repository.workspace_root,
        container_image=task.execution.slurm.controller.image,
        repository_head=task.repository.base_commit,
        repository_dirty=False,
        build_identity=task.execution.slurm.controller.build_identity,
        mounts=(ObservedMount(mount.host_path, Path(mount.container_path), mount.read_only),),
        available_commands=required_commands(task, phase),
        dispatch=_dispatch_contract(task, phase),
    )


def _dispatch_contract(
    task: NormalizedTask,
    phase: PreflightPhase,
    *,
    available: bool = True,
) -> DispatchContract:
    def run_probe(
        argv: tuple[str, ...],
        *,
        input_text: str,
        timeout_seconds: float,
        output_limit_bytes: int,
    ) -> DispatchProbeResult:
        del argv, input_text, timeout_seconds, output_limit_bytes
        if available:
            return DispatchProbeResult(
                0,
                "sbatch: Job 12345 to start at 2026-09-17T12:00:00\n",
            )
        return DispatchProbeResult(1, stderr="nested sbatch denied")

    environment = (
        {"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "alpha"}
        if phase is PreflightPhase.CONTROLLER
        else None
    )
    return probe_dispatch_capability(
        task,
        phase,
        runner=run_probe,
        environment=environment,
    )


@pytest.mark.parametrize("phase", [PreflightPhase.LOGIN, PreflightPhase.CONTROLLER])
def test_preflight_accepts_matching_snapshot(tmp_path: Path, phase: PreflightPhase) -> None:
    task = _task(tmp_path)

    report = validate_production_preflight(task, _snapshot(task, phase))

    assert report.ok
    report.require_ok()


def test_preflight_accepts_explicit_single_rank_gpu_allocation_padding(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    resources = tuple(
        replace(
            resource,
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=4,
            gpu_allocation_padding=True,
            partition="batch",
            qos="normal",
        )
        if resource.name == "deterministic_gate"
        else (
            replace(resource, nodes=1, tasks_per_node=1, gpus_per_node=1)
            if resource.name == "reviewer_rerun"
            else resource
        )
        for resource in task.execution.slurm.smith.resource_classes
    )
    task = replace(
        task,
        target=replace(
            task.target,
            world_size=1,
            mapping=ParallelMapping(1, 1, 1, 1, 1),
        ),
        execution=replace(
            task.execution,
            slurm=replace(
                task.execution.slurm,
                smith=replace(task.execution.slurm.smith, resource_classes=resources),
            ),
        ),
    )

    report = validate_production_preflight(task, _snapshot(task, PreflightPhase.LOGIN))

    assert report.ok


def test_preflight_batches_independent_failures(tmp_path: Path) -> None:
    task = _task(tmp_path)
    other_repository = tmp_path / "other-repo"
    other_repository.mkdir()
    snapshot = replace(
        _snapshot(task, PreflightPhase.LOGIN),
        repository_root=other_repository,
        repository_head="c" * 40,
        repository_dirty=True,
        build_identity="wrong-build",
        mounts=(),
        available_commands=frozenset({"python3"}),
        dispatch=_dispatch_contract(task, PreflightPhase.LOGIN, available=False),
    )

    report = validate_production_preflight(task, snapshot)

    codes = [issue.code for issue in report.issues]
    assert "repository_identity" in codes
    assert "base_commit" in codes
    assert "dirty_repository" in codes
    assert "build_identity" in codes
    assert "mount_identity" in codes
    assert "mount_visibility" in codes
    assert "dispatch_unavailable" in codes
    assert codes.count("missing_command") == 5
    with pytest.raises(PreflightError, match="repository_identity"):
        report.require_ok()


def test_controller_requires_sbatch_only_for_nested_submission(tmp_path: Path) -> None:
    nested = _task(tmp_path / "nested")
    assert "sbatch" in required_commands(nested, PreflightPhase.CONTROLLER)

    dispatcher = _task(tmp_path / "dispatcher", dispatch_mode="login_dispatcher")
    assert "sbatch" not in required_commands(dispatcher, PreflightPhase.CONTROLLER)


def test_login_dispatch_receipt_cannot_certify_controller_phase(tmp_path: Path) -> None:
    task = _task(tmp_path)
    snapshot = replace(
        _snapshot(task, PreflightPhase.CONTROLLER),
        dispatch=_dispatch_contract(task, PreflightPhase.LOGIN),
    )

    report = validate_production_preflight(task, snapshot)

    assert "dispatch_phase" in {issue.code for issue in report.issues}


def test_preflight_rejects_receipt_not_bound_to_fixed_probe(tmp_path: Path) -> None:
    task = _task(tmp_path)
    valid = _dispatch_contract(task, PreflightPhase.LOGIN)
    forged_receipt = replace(valid.receipt, command_digest="0" * 64)
    snapshot = replace(
        _snapshot(task, PreflightPhase.LOGIN),
        dispatch=DispatchContract(forged_receipt),
    )

    report = validate_production_preflight(task, snapshot)

    assert "dispatch_receipt" in {issue.code for issue in report.issues}


def test_preflight_rechecks_resource_caps_and_topology(tmp_path: Path) -> None:
    task = _task(tmp_path)
    classes = list(task.execution.slurm.smith.resource_classes)
    classes[0] = replace(classes[0], nodes=5)
    classes[2] = replace(classes[2], tasks_per_node=1, gpus_per_node=1)
    invalid_smith = replace(task.execution.slurm.smith, resource_classes=tuple(classes))
    invalid_task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(task.execution.slurm, smith=invalid_smith),
        ),
    )

    report = validate_production_preflight(
        invalid_task,
        _snapshot(invalid_task, PreflightPhase.LOGIN),
    )

    codes = {issue.code for issue in report.issues}
    assert "resource_class" in codes
    assert "resource_topology" in codes
