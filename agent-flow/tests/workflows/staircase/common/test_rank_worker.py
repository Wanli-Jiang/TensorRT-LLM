# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fixed private rank body wrapper."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import internal
from agent_flow.workflows.staircase.common import rank_worker
from agent_flow.workflows.staircase.common.gates import GateCommand
from agent_flow.workflows.staircase.common.launchers import (
    AllocationNode,
    RankAllocation,
    RankLaunchPlan,
    build_single_process_plan,
    build_synthetic_multi_node_plan,
)
from agent_flow.workflows.staircase.common.rank_worker import (
    PRODUCT_EVIDENCE_PATH_ENVIRONMENT,
    RankInput,
    RankWorkerError,
    execute_rank,
    load_rank_input,
    publish_product_identity_evidence,
    publish_rank_input,
)
from agent_flow.workflows.staircase.task_schema import ParallelMapping


def _mapping(world_size: int) -> ParallelMapping:
    return ParallelMapping(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=1,
        moe_tensor_parallel_size=world_size,
        attention_data_parallel_size=1,
    )


def _single_plan(tmp_path: Path, command: GateCommand | None = None) -> RankLaunchPlan:
    return build_single_process_plan(
        mapping=_mapping(1),
        world_size=1,
        allocation=RankAllocation((AllocationNode("node-a", (3,)),)),
        command=command or GateCommand(("python3", "product.py")),
        rank_input_path=tmp_path / "rank-input.json",
    )


def _multi_plan(tmp_path: Path) -> RankLaunchPlan:
    return build_synthetic_multi_node_plan(
        mapping=_mapping(2),
        world_size=2,
        allocation=RankAllocation(
            (AllocationNode("node-a", (0,)), AllocationNode("node-b", (0,))),
            slurm_job_id="1234",
        ),
        command=GateCommand(("python3", "runner.py")),
        rank_input_path=tmp_path / "rank-input.json",
    )


def _publish(plan: RankLaunchPlan, tmp_path: Path) -> tuple[RankInput, str]:
    return publish_rank_input(
        plan,
        cwd=tmp_path,
        report_directory=tmp_path / "rank-reports",
    )


class _BodyExecutor:
    def __init__(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self.calls: list[tuple[tuple[str, ...], Path, tuple[tuple[str, str], ...], bool]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        environment: tuple[tuple[str, str], ...],
        shell: bool,
    ) -> int:
        self.calls.append((argv, cwd, environment, shell))
        return self.returncode


def test_publish_creates_read_only_digest_bound_input_and_empty_report_directory(
    tmp_path: Path,
) -> None:
    plan = _single_plan(tmp_path)

    rank_input, input_digest = _publish(plan, tmp_path)
    loaded, loaded_digest = load_rank_input(plan.rank_input_path)

    assert loaded == rank_input
    assert loaded_digest == input_digest
    assert rank_input.plan_digest == plan.plan_digest
    assert rank_input.rank_command_digest == plan.rank_command_digest
    assert not rank_input.report_directory.is_symlink()
    assert list(rank_input.report_directory.iterdir()) == []
    assert not plan.rank_input_path.stat().st_mode & stat.S_IWUSR


def test_product_body_publishes_only_actual_collective_evidence_once(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "product-evidence.json"

    published = publish_product_identity_evidence(
        collective_backend="nccl",
        transport="ib",
        environment={PRODUCT_EVIDENCE_PATH_ENVIRONMENT: str(destination)},
    )

    assert published == destination
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "collective_backend": "nccl",
        "transport": "ib",
    }
    assert not destination.stat().st_mode & stat.S_IWUSR
    assert not destination.with_name(f".{destination.name}.pending").exists()
    with pytest.raises(RankWorkerError, match="must be a new absolute path"):
        publish_product_identity_evidence(
            collective_backend="nccl",
            transport="ib",
            environment={PRODUCT_EVIDENCE_PATH_ENVIRONMENT: str(destination)},
        )


@pytest.mark.parametrize(
    ("backend", "transport"),
    (("", "ib"), ("nccl\nforged", "ib"), ("nccl", "")),
)
def test_product_evidence_rejects_unsafe_or_missing_observations(
    tmp_path: Path, backend: str, transport: str
) -> None:
    with pytest.raises(RankWorkerError, match="bounded non-empty single-line"):
        publish_product_identity_evidence(
            collective_backend=backend,
            transport=transport,
            environment={PRODUCT_EVIDENCE_PATH_ENVIRONMENT: str(tmp_path / "evidence.json")},
        )

    with pytest.raises(RankWorkerError, match="lacks wrapper-owned"):
        publish_product_identity_evidence(collective_backend="nccl", transport="ib", environment={})


def test_product_evidence_rejects_symlink_destination(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    destination = tmp_path / "product-evidence.json"
    destination.symlink_to(target)

    with pytest.raises(RankWorkerError, match="must be a new absolute path"):
        publish_product_identity_evidence(
            collective_backend="nccl",
            transport="ib",
            environment={PRODUCT_EVIDENCE_PATH_ENVIRONMENT: str(destination)},
        )


def test_single_rank_derives_binding_runs_exact_body_and_writes_report(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    rank_input, input_digest = _publish(plan, tmp_path)
    executor = _BodyExecutor()
    runtime = {
        "CUDA_VISIBLE_DEVICES": "3",
        "LOCAL_RANK": "0",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "UNAUTHORIZED_AMBIENT": "drop-me",
    }

    assert (
        execute_rank(
            plan.rank_input_path,
            environment=runtime,
            hostname_provider=lambda: "node-a",
            executor=executor,
        )
        == 0
    )

    expected_environment = tuple(
        sorted(
            {
                "CUDA_VISIBLE_DEVICES": "3",
                "LOCAL_RANK": "0",
                "RANK": "0",
                "TRTLLM_MODELING_V2": "require",
                "WORLD_SIZE": "1",
            }.items()
        )
    )
    assert executor.calls == [(plan.rank_command.argv, tmp_path, expected_environment, False)]
    report = json.loads((rank_input.report_directory / "rank-0.json").read_text())
    assert report["input_digest"] == input_digest
    assert report["plan_digest"] == plan.plan_digest
    assert report["rank_command_digest"] == plan.rank_command_digest
    assert (report["rank"], report["hostname"], report["local_rank"], report["gpu_id"]) == (
        0,
        "node-a",
        0,
        3,
    )


def test_multi_rank_uses_only_slurm_runtime_identity(tmp_path: Path) -> None:
    plan = _multi_plan(tmp_path)
    rank_input, _input_digest = _publish(plan, tmp_path)
    executor = _BodyExecutor()
    runtime = {
        "CUDA_VISIBLE_DEVICES": "0",
        "RANK": "0",
        "LOCAL_RANK": "99",
        "WORLD_SIZE": "99",
        "SLURM_PROCID": "1",
        "SLURM_LOCALID": "0",
        "SLURM_NTASKS": "2",
    }

    execute_rank(
        plan.rank_input_path,
        environment=runtime,
        hostname_provider=lambda: "node-b",
        executor=executor,
    )

    report = json.loads((rank_input.report_directory / "rank-1.json").read_text())
    assert (report["rank"], report["hostname"], report["local_rank"], report["gpu_id"]) == (
        1,
        "node-b",
        0,
        0,
    )
    assert dict(executor.calls[0][2])["RANK"] == "1"
    assert dict(executor.calls[0][2])["WORLD_SIZE"] == "2"


@pytest.mark.parametrize(
    ("environment", "hostname", "match"),
    [
        (
            {
                "CUDA_VISIBLE_DEVICES": "7",
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "1",
            },
            "node-a",
            "differs from planned binding",
        ),
        (
            {
                "CUDA_VISIBLE_DEVICES": "3",
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "1",
            },
            "node-b",
            "differs from planned binding",
        ),
        (
            {
                "CUDA_VISIBLE_DEVICES": "0,1",
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "1",
            },
            "node-a",
            "exactly one numeric CUDA device",
        ),
    ],
)
def test_runtime_placement_mismatch_fails_before_body(
    tmp_path: Path, environment: dict[str, str], hostname: str, match: str
) -> None:
    plan = _single_plan(tmp_path)
    _publish(plan, tmp_path)
    executor = _BodyExecutor()

    with pytest.raises(RankWorkerError, match=match):
        execute_rank(
            plan.rank_input_path,
            environment=environment,
            hostname_provider=lambda: hostname,
            executor=executor,
        )
    assert executor.calls == []


def test_body_failure_is_propagated_without_success_report(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    rank_input, _input_digest = _publish(plan, tmp_path)

    returncode = execute_rank(
        plan.rank_input_path,
        environment={
            "CUDA_VISIBLE_DEVICES": "3",
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
        },
        hostname_provider=lambda: "node-a",
        executor=_BodyExecutor(17),
    )

    assert returncode == 17
    assert not (rank_input.report_directory / "rank-0.json").exists()
    assert (rank_input.report_directory / "rank-0.pending").exists()


def test_duplicate_rank_cannot_execute_body_twice(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    _publish(plan, tmp_path)
    runtime = {
        "CUDA_VISIBLE_DEVICES": "3",
        "LOCAL_RANK": "0",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }
    execute_rank(
        plan.rank_input_path,
        environment=runtime,
        hostname_provider=lambda: "node-a",
        executor=_BodyExecutor(),
    )
    duplicate = _BodyExecutor()

    with pytest.raises(RankWorkerError, match="reserve exclusive"):
        execute_rank(
            plan.rank_input_path,
            environment=runtime,
            hostname_provider=lambda: "node-a",
            executor=duplicate,
        )
    assert duplicate.calls == []


def test_rank_input_rejects_unknown_fields_and_writable_mode(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    _publish(plan, tmp_path)
    plan.rank_input_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with pytest.raises(RankWorkerError, match="read-only"):
        load_rank_input(plan.rank_input_path)

    payload = json.loads(plan.rank_input_path.read_text())
    payload["unknown"] = True
    plan.rank_input_path.write_text(json.dumps(payload), encoding="utf-8")
    plan.rank_input_path.chmod(stat.S_IRUSR)
    with pytest.raises(RankWorkerError, match="keys differ from schema"):
        load_rank_input(plan.rank_input_path)


def test_rank_input_cannot_invoke_nested_scheduler(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    _publish(plan, tmp_path)
    plan.rank_input_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    payload = json.loads(plan.rank_input_path.read_text())
    payload["argv"] = ["sbatch", "payload"]
    plan.rank_input_path.write_text(json.dumps(payload), encoding="utf-8")
    plan.rank_input_path.chmod(stat.S_IRUSR)

    with pytest.raises(RankWorkerError, match="cannot invoke"):
        load_rank_input(plan.rank_input_path)


def test_internal_rank_worker_routing_does_not_touch_generic_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = (tmp_path / "rank-input.json").resolve()
    observed: list[Path] = []
    monkeypatch.setattr(rank_worker, "execute_rank", lambda path: observed.append(path) or 0)

    internal.main(["rank-worker", "--input", str(input_path)])

    assert observed == [input_path]


def test_internal_rank_worker_propagates_body_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = (tmp_path / "rank-input.json").resolve()
    monkeypatch.setattr(rank_worker, "execute_rank", lambda _path: 29)

    with pytest.raises(SystemExit) as error:
        internal.main(["rank-worker", "--input", str(input_path)])

    assert error.value.code == 29


def test_default_body_executor_propagates_failure_without_shell(tmp_path: Path) -> None:
    body = tmp_path / "body.py"
    body.write_text("raise SystemExit(23)\n", encoding="utf-8")
    plan = _single_plan(tmp_path, GateCommand((sys.executable, str(body))))
    _publish(plan, tmp_path)

    assert (
        execute_rank(
            plan.rank_input_path,
            environment={
                "CUDA_VISIBLE_DEVICES": "3",
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "1",
            },
            hostname_provider=lambda: "node-a",
        )
        == 23
    )
