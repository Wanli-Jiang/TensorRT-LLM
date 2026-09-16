# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for read-only Staircase preflight fact discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.discovery import (
    CommandResult,
    DiscoveryError,
    ObservedExecutionFacts,
    collect_preflight_snapshot,
    run_read_only_command,
)
from agent_flow.workflows.staircase.common.dispatch_probe import DispatchProbeResult
from agent_flow.workflows.staircase.common.preflight import (
    ObservedMount,
    PreflightPhase,
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

_HEAD = "a" * 40
_BUILD_IDENTITY = "trtllm-local-dev@aabbccdd"
_RESOURCE_NAMES = (
    "coder_analysis",
    "exploratory_probe",
    "deterministic_gate",
    "reviewer_analysis",
    "reviewer_rerun",
)


class FakeRunner:
    """Return fixed Git observations while recording exact argv calls."""

    def __init__(self, *, head: str = _HEAD, status: str = "") -> None:
        self._head = head
        self._status = status
        self.calls: list[tuple[tuple[str, ...], int]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        output_limit_bytes: int,
    ) -> CommandResult:
        self.calls.append((argv, output_limit_bytes))
        if argv[-3:] == ("rev-parse", "--verify", "HEAD"):
            return CommandResult(0, self._head + "\n", "")
        if argv[-3:] == ("status", "--porcelain=v1", "--untracked-files=normal"):
            return CommandResult(0, self._status, "")
        raise AssertionError(f"unexpected discovery argv: {argv!r}")


class FakeDispatchRunner:
    """Return a deterministic ``sbatch --test-only`` observation."""

    def __init__(self, *, available: bool = True) -> None:
        self.available = available
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        input_text: str,
        timeout_seconds: float,
        output_limit_bytes: int,
    ) -> DispatchProbeResult:
        del input_text, timeout_seconds, output_limit_bytes
        self.calls.append(argv)
        if self.available:
            return DispatchProbeResult(
                0,
                "sbatch: Job 12345 to start at 2026-09-17T12:00:00\n",
            )
        return DispatchProbeResult(1, stderr="nested sbatch denied")


def _task(tmp_path: Path) -> NormalizedTask:
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
        dispatch_mode="nested_submission",
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
        repository=RepositoryConfig(repository.resolve(), _HEAD, "reject", workspace.resolve()),
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


def _facts(
    task: NormalizedTask,
    *,
    build_identity: str = _BUILD_IDENTITY,
) -> ObservedExecutionFacts:
    mount = task.execution.slurm.controller.mounts[0]
    return ObservedExecutionFacts(
        repository_root=task.repository.root,
        workspace_root=task.repository.workspace_root,
        container_image=task.execution.slurm.controller.image,
        build_identity=build_identity,
        mounts=(ObservedMount(mount.host_path, Path(mount.container_path), mount.read_only),),
    )


def test_collects_canonical_matching_snapshot_with_fixed_shell_free_git(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    repository_alias = tmp_path / "repository-alias"
    repository_alias.symlink_to(task.repository.root, target_is_directory=True)
    facts = _facts(task)
    facts = ObservedExecutionFacts(
        repository_alias,
        facts.workspace_root,
        facts.container_image,
        facts.build_identity,
        facts.mounts,
    )
    runner = FakeRunner()
    located: list[str] = []

    def locate(command: str) -> str:
        located.append(command)
        return f"/commands/{command}"

    snapshot = collect_preflight_snapshot(
        task,
        PreflightPhase.LOGIN,
        facts,
        command_runner=runner,
        dispatch_probe_runner=FakeDispatchRunner(),
        command_locator=locate,
    )

    assert snapshot.repository_root == task.repository.root
    assert snapshot.repository_head == _HEAD
    assert snapshot.repository_dirty is False
    assert snapshot.build_identity == _BUILD_IDENTITY
    assert snapshot.available_commands == required_commands(task, PreflightPhase.LOGIN)
    assert located == sorted(required_commands(task, PreflightPhase.LOGIN))
    assert len(runner.calls) == 2
    assert all(call[0][0] == "git" for call in runner.calls)
    assert all("--no-optional-locks" in call[0] for call in runner.calls)
    assert all(call[1] == 16_384 for call in runner.calls)
    assert validate_production_preflight(task, snapshot).ok


def test_dirty_state_and_explicit_build_identity_are_preserved(tmp_path: Path) -> None:
    task = _task(tmp_path)
    runner = FakeRunner(status="?? local-file\n")

    snapshot = collect_preflight_snapshot(
        task,
        PreflightPhase.CONTROLLER,
        _facts(task, build_identity="explicit-local-dev-build"),
        command_runner=runner,
        dispatch_probe_runner=FakeDispatchRunner(),
        dispatch_environment={"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "alpha"},
        command_locator=lambda command: f"/commands/{command}",
    )

    assert snapshot.repository_dirty is True
    assert snapshot.build_identity == "explicit-local-dev-build"


def test_command_presence_never_implies_nested_dispatch_capability(tmp_path: Path) -> None:
    task = _task(tmp_path)

    snapshot = collect_preflight_snapshot(
        task,
        PreflightPhase.CONTROLLER,
        _facts(task),
        command_runner=FakeRunner(),
        dispatch_probe_runner=FakeDispatchRunner(available=False),
        dispatch_environment={"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "alpha"},
        command_locator=lambda command: f"/commands/{command}",
    )

    assert "sbatch" in snapshot.available_commands
    assert snapshot.dispatch.available is False
    report = validate_production_preflight(task, snapshot)
    assert "dispatch_unavailable" in {issue.code for issue in report.issues}


def test_only_phase_required_commands_are_looked_up(tmp_path: Path) -> None:
    task = _task(tmp_path)
    looked_up: list[str] = []

    def locate(command: str) -> str | None:
        looked_up.append(command)
        return None if command == "sacct" else f"/commands/{command}"

    snapshot = collect_preflight_snapshot(
        task,
        PreflightPhase.CONTROLLER,
        _facts(task),
        command_runner=FakeRunner(),
        dispatch_probe_runner=FakeDispatchRunner(),
        dispatch_environment={"SLURM_JOB_ID": "123", "SLURM_CLUSTER_NAME": "alpha"},
        command_locator=locate,
    )

    assert looked_up == sorted(required_commands(task, PreflightPhase.CONTROLLER))
    assert "sacct" not in snapshot.available_commands


@pytest.mark.parametrize("head", ["abc123", "g" * 40, _HEAD + "\n" + _HEAD])
def test_rejects_non_exact_repository_head(tmp_path: Path, head: str) -> None:
    task = _task(tmp_path)

    with pytest.raises(DiscoveryError, match="one exact commit ID"):
        collect_preflight_snapshot(
            task,
            PreflightPhase.LOGIN,
            _facts(task),
            command_runner=FakeRunner(head=head),
            dispatch_probe_runner=FakeDispatchRunner(),
            command_locator=lambda command: f"/commands/{command}",
        )


def test_injected_runner_output_is_bounded_by_contract(tmp_path: Path) -> None:
    task = _task(tmp_path)

    def oversized_runner(
        argv: tuple[str, ...],
        *,
        output_limit_bytes: int,
    ) -> CommandResult:
        del argv, output_limit_bytes
        return CommandResult(0, "x" * 20_000, "")

    with pytest.raises(DiscoveryError, match="output contract"):
        collect_preflight_snapshot(
            task,
            PreflightPhase.LOGIN,
            _facts(task),
            command_runner=oversized_runner,
            dispatch_probe_runner=FakeDispatchRunner(),
            command_locator=lambda command: f"/commands/{command}",
        )


def test_command_and_path_errors_have_bounded_diagnostics(tmp_path: Path) -> None:
    task = _task(tmp_path)

    def failed_runner(
        argv: tuple[str, ...],
        *,
        output_limit_bytes: int,
    ) -> CommandResult:
        del argv, output_limit_bytes
        return CommandResult(2, "", "s" * 10_000)

    with pytest.raises(DiscoveryError) as command_error:
        collect_preflight_snapshot(
            task,
            PreflightPhase.LOGIN,
            _facts(task),
            command_runner=failed_runner,
            dispatch_probe_runner=FakeDispatchRunner(),
            command_locator=lambda command: f"/commands/{command}",
        )
    assert len(str(command_error.value)) < 2_200
    assert str(command_error.value).endswith("...[truncated]")

    def failed_resolver(path: Path) -> Path:
        del path
        raise OSError("p" * 10_000)

    with pytest.raises(DiscoveryError) as path_error:
        collect_preflight_snapshot(
            task,
            PreflightPhase.LOGIN,
            _facts(task),
            command_runner=FakeRunner(),
            dispatch_probe_runner=FakeDispatchRunner(),
            command_locator=lambda command: f"/commands/{command}",
            path_resolver=failed_resolver,
        )
    assert len(str(path_error.value)) < 2_200
    assert str(path_error.value).endswith("...[truncated]")


def test_production_runner_rejects_non_discovery_argv() -> None:
    with pytest.raises(DiscoveryError, match="only fixed read-only Git"):
        run_read_only_command(("git", "status"), output_limit_bytes=100)
