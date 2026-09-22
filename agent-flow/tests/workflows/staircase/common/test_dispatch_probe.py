# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the non-mutating dispatch capability probe."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.dispatch_probe import (
    DispatchPhase,
    DispatchProbeResult,
    probe_dispatch_capability,
    run_dispatch_probe_command,
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


class RecordingRunner:
    """Return one typed result while retaining the exact safe invocation."""

    def __init__(self, result: DispatchProbeResult | BaseException) -> None:
        self.result = result
        self.calls: list[tuple[tuple[str, ...], str, float, int]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        input_text: str,
        timeout_seconds: float,
        output_limit_bytes: int,
    ) -> DispatchProbeResult:
        self.calls.append((argv, input_text, timeout_seconds, output_limit_bytes))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _task(tmp_path: Path, *, mode: str = "nested_submission") -> NormalizedTask:
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    checkpoint = tmp_path / "checkpoint"
    image = tmp_path / "image.sqsh"
    for directory in (repository, workspace, checkpoint):
        directory.mkdir(parents=True)
    image.touch()
    controller = ControllerConfig(
        account="coreai",
        partition="batch",
        qos="normal",
        reservation="bringup",
        time_limit_seconds=3_600,
        cpus_per_task=2,
        memory_mib=4_096,
        image=image.resolve(),
        mounts=(MountConfig(tmp_path.resolve(), "/job", False),),
        environment=(),
        build_identity="test-build",
        dispatch_mode=mode,
        requeue=True,
        advance_signal_lead_seconds=60,
        heartbeat_timeout_seconds=30,
        lease_timeout_seconds=90,
        orphan_grace_seconds=120,
    )
    names = (
        "coder_analysis",
        "exploratory_probe",
        "deterministic_gate",
        "reviewer_analysis",
        "reviewer_rerun",
    )
    resources = tuple(ResourceClass(name, 1, 1, 0, 1, 1_024, 600) for name in names)
    smith = SmithConfig(
        1,
        1,
        0,
        resources,
        OverrideBounds(1, 1, 0, 2, 2_048, 1_200),
        RetryPolicy(1, 1),
    )
    return NormalizedTask(
        schema_version=1,
        repository=RepositoryConfig(repository.resolve(), "a" * 40, "reject", workspace.resolve()),
        reference=ReferenceConfig(checkpoint.resolve(), "fixture", "ExampleModel", ()),
        target=TargetConfig(
            "example",
            "example_tp1",
            100,
            1,
            ParallelMapping(1, 1, 1, 1, 1),
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


def _accepted() -> DispatchProbeResult:
    return DispatchProbeResult(0, "sbatch: Job 12345 to start at 2026-09-17T12:00:00\n")


def test_login_probe_is_fixed_shell_free_test_only_and_digest_bound(tmp_path: Path) -> None:
    task = _task(tmp_path)
    runner = RecordingRunner(_accepted())

    contract = probe_dispatch_capability(task, DispatchPhase.LOGIN, runner=runner)

    assert contract.available
    assert contract.phase is DispatchPhase.LOGIN
    assert contract.receipt.controller_job_id is None
    assert contract.receipt.cluster is None
    assert len(contract.receipt.command_digest) == 64
    assert len(contract.receipt_digest) == 64
    assert len(contract.evidence) <= 512
    argv, script, timeout, output_limit = runner.calls[0]
    assert argv[:2] == ("sbatch", "--test-only")
    assert "--account=coreai" in argv
    assert "--partition=batch" in argv
    assert "--qos=normal" in argv
    assert "--reservation=bringup" in argv
    assert "--nodes=1" in argv
    assert "--ntasks=1" in argv
    assert "--cpus-per-task=1" in argv
    assert "--no-requeue" in argv
    assert not any(value == "--wrap" or value.startswith("--wrap=") for value in argv)
    assert script == "#!/bin/true\n"
    assert timeout == 60.0
    assert output_limit == 4_096
    assert all("scancel" not in value for value in argv)


def test_controller_probe_requires_and_binds_exact_allocation(tmp_path: Path) -> None:
    task = _task(tmp_path)
    runner = RecordingRunner(_accepted())

    missing = probe_dispatch_capability(
        task,
        DispatchPhase.CONTROLLER,
        runner=runner,
        environment={},
    )
    assert not missing.available
    assert not runner.calls

    malformed = probe_dispatch_capability(
        task,
        DispatchPhase.CONTROLLER,
        runner=runner,
        environment={"SLURM_JOB_ID": "123*", "SLURM_CLUSTER_NAME": "alpha"},
    )
    assert not malformed.available
    assert not runner.calls

    contract = probe_dispatch_capability(
        task,
        DispatchPhase.CONTROLLER,
        runner=runner,
        environment={"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "alpha"},
    )
    assert contract.available
    assert contract.receipt.controller_job_id == "123"
    assert contract.receipt.cluster == "alpha"
    argv = runner.calls[0][0]
    assert "--clusters=alpha" in argv
    assert "--comment=staircase-dispatch-probe-parent-123" in argv


@pytest.mark.parametrize(
    "result",
    [
        DispatchProbeResult(1, stderr="submission denied"),
        DispatchProbeResult(-1, timed_out=True),
        DispatchProbeResult(-1, output_limited=True),
        DispatchProbeResult(0, "unexpected success text\n"),
        DispatchProbeResult(0, "sbatch: Job 1 to start at soon\nextra\n"),
    ],
)
def test_failure_timeout_and_malformed_output_are_typed_unavailable(
    tmp_path: Path,
    result: DispatchProbeResult,
) -> None:
    contract = probe_dispatch_capability(
        _task(tmp_path),
        DispatchPhase.LOGIN,
        runner=RecordingRunner(result),
    )

    assert not contract.available
    assert contract.receipt.receipt_digest


def test_runner_exception_is_unavailable_with_sanitized_bounded_evidence(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner(RuntimeError("denied\n" + "x" * 2_000))

    contract = probe_dispatch_capability(
        _task(tmp_path),
        DispatchPhase.LOGIN,
        runner=runner,
    )

    assert not contract.available
    assert "\n" not in contract.evidence
    assert len(contract.evidence) <= 512


@pytest.mark.parametrize("mode", ["login_dispatcher", "preallocated_pool"])
def test_unimplemented_dispatch_adapters_are_typed_unavailable(
    tmp_path: Path,
    mode: str,
) -> None:
    runner = RecordingRunner(_accepted())

    contract = probe_dispatch_capability(
        _task(tmp_path, mode=mode),
        DispatchPhase.LOGIN,
        runner=runner,
    )

    assert not contract.available
    assert "no concrete" in contract.evidence
    assert not runner.calls


def test_receipt_is_immutable_and_phase_changes_its_digest(tmp_path: Path) -> None:
    task = _task(tmp_path)
    login = probe_dispatch_capability(
        task,
        DispatchPhase.LOGIN,
        runner=RecordingRunner(_accepted()),
    )
    controller = probe_dispatch_capability(
        task,
        DispatchPhase.CONTROLLER,
        runner=RecordingRunner(_accepted()),
        environment={"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "alpha"},
    )

    assert login.receipt_digest != controller.receipt_digest
    with pytest.raises(FrozenInstanceError):
        login.receipt.available = False  # type: ignore[misc]
    altered = replace(login.receipt, partition="debug")
    assert altered.receipt_digest != login.receipt_digest


def test_production_runner_rejects_wrap_and_nonfixed_script() -> None:
    argv = (
        "sbatch",
        "--test-only",
        "--job-name=staircase-dispatch-probe",
        "--account=coreai",
        "--partition=batch",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=16M",
        "--time=00:01:00",
        "--no-requeue",
        "--export=NONE",
        "--comment=staircase-dispatch-probe-login",
        "--wrap=touch /tmp/nope",
    )
    with pytest.raises(ValueError, match="unsupported option"):
        run_dispatch_probe_command(
            argv,
            input_text="#!/bin/true\n",
            timeout_seconds=15.0,
            output_limit_bytes=4_096,
        )
    with pytest.raises(ValueError, match="fixed minimal script"):
        run_dispatch_probe_command(
            argv[:-1],
            input_text="#!/bin/sh\ntouch /tmp/nope\n",
            timeout_seconds=15.0,
            output_limit_bytes=4_096,
        )
