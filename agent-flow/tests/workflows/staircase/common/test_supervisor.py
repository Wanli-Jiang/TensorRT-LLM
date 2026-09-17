# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the trusted, bounded rank process supervisor."""

from __future__ import annotations

import json
import socket
import stat
import sys
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common import supervisor as supervisor_module
from agent_flow.workflows.staircase.common.gates import EvidenceScope, GateCommand
from agent_flow.workflows.staircase.common.launchers import (
    AllocationNode,
    LaunchKind,
    RankAllocation,
    RankLaunchPlan,
    build_single_process_plan,
    build_synthetic_multi_node_plan,
)
from agent_flow.workflows.staircase.common.rank_worker import publish_rank_input
from agent_flow.workflows.staircase.common.supervisor import (
    ProcessCapture,
    RankSupervisorError,
    supervise_rank_launch,
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


def _single_plan(
    tmp_path: Path,
    command: GateCommand | None = None,
    *,
    hostname: str = "node-a",
) -> RankLaunchPlan:
    return build_single_process_plan(
        mapping=_mapping(1),
        world_size=1,
        allocation=RankAllocation((AllocationNode(hostname, (3,)),)),
        command=command or GateCommand(("python3", "product.py")),
        rank_input_path=tmp_path / "rank-input.json",
    )


def _synthetic_plan(tmp_path: Path) -> RankLaunchPlan:
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


def _publish(plan: RankLaunchPlan, tmp_path: Path) -> tuple[Path, str]:
    rank_input, input_digest = publish_rank_input(
        plan,
        cwd=tmp_path,
        report_directory=tmp_path / "rank-reports",
    )
    return rank_input.report_directory, input_digest


def _report(
    plan: RankLaunchPlan,
    input_digest: str,
    rank: int,
    *,
    body_kind: str | None = None,
    hostname: str | None = None,
    gpu_id: int | None = None,
    extra: dict[str, object] | None = None,
) -> bytes:
    placement = plan.placements[rank]
    payload: dict[str, object] = {
        "schema_version": 1,
        "input_digest": input_digest,
        "plan_digest": plan.plan_digest,
        "rank_command_digest": plan.rank_command_digest,
        "launch_kind": plan.kind.value,
        "body_kind": body_kind or ("product" if plan.product_rank_body else "synthetic"),
        "rank": placement.rank,
        "hostname": hostname or placement.hostname,
        "local_rank": placement.local_rank,
        "gpu_id": placement.gpu_id if gpu_id is None else gpu_id,
        "product_identity": None,
    }
    if extra:
        payload.update(extra)
    return (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


class _RecordingExecutor:
    def __init__(self, capture: ProcessCapture, reports: dict[Path, bytes] | None = None) -> None:
        self.capture = capture
        self.reports = reports or {}
        self.calls: list[
            tuple[
                tuple[str, ...],
                Path,
                tuple[tuple[str, str], ...],
                bool,
                int,
            ]
        ] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        environment: tuple[tuple[str, str], ...],
        shell: bool,
        capture_limit_bytes: int,
    ) -> ProcessCapture:
        self.calls.append((argv, cwd, environment, shell, capture_limit_bytes))
        for path, payload in self.reports.items():
            path.write_bytes(payload)
        return self.capture


def test_supervisor_executes_exact_wrapper_and_collects_report_files(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    report_directory, input_digest = _publish(plan, tmp_path)
    executor = _RecordingExecutor(
        ProcessCapture(0, b"product log\n", b""),
        {report_directory / "rank-0.json": _report(plan, input_digest, 0)},
    )

    result = supervise_rank_launch(plan, cwd=tmp_path, executor=executor)

    assert executor.calls == [(plan.argv, tmp_path, plan.environment, False, 1024 * 1024)]
    assert result.receipt.plan_digest == plan.plan_digest
    assert result.receipt.placements == plan.placements
    assert result.stdout == b"product log\n"


def test_synthetic_multi_node_receipt_stays_runner_only(tmp_path: Path) -> None:
    plan = _synthetic_plan(tmp_path)
    report_directory, input_digest = _publish(plan, tmp_path)
    reports = {
        report_directory / f"rank-{rank}.json": _report(plan, input_digest, rank)
        for rank in range(plan.world_size)
    }
    executor = _RecordingExecutor(ProcessCapture(0, b"", b"runner diagnostic"), reports)

    result = supervise_rank_launch(plan, cwd=tmp_path, executor=executor)

    assert result.receipt.kind is LaunchKind.SYNTHETIC_MULTI_NODE_RUNNER
    assert not result.receipt.product_rank_body
    assert result.receipt.evidence_scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER
    assert executor.calls[0][0] == plan.argv


def test_default_multi_rank_executor_resolves_absolute_site_srun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site_srun = Path("/cm/local/apps/slurm/current/bin/srun")
    if not site_srun.is_file():
        pytest.skip("site-style Slurm client is unavailable")
    plan = _synthetic_plan(tmp_path)
    report_directory, input_digest = _publish(plan, tmp_path)
    calls: list[tuple[str, ...]] = []

    def execute(
        argv: tuple[str, ...],
        _cwd: Path,
        _environment: tuple[tuple[str, str], ...],
        _shell: bool,
        _capture_limit_bytes: int,
    ) -> ProcessCapture:
        calls.append(argv)
        for rank in range(plan.world_size):
            (report_directory / f"rank-{rank}.json").write_bytes(_report(plan, input_digest, rank))
        return ProcessCapture(0, b"", b"")

    monkeypatch.setattr(supervisor_module, "_execute_bounded", execute)

    supervise_rank_launch(
        plan,
        cwd=tmp_path,
        supervisor_environment={"PATH": str(site_srun.parent)},
    )

    assert calls[0][0] == str(site_srun.resolve())
    assert calls[0][1:] == plan.argv[1:]


def test_default_multi_rank_executor_rejects_writable_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hijack_dir = tmp_path / "bin"
    hijack_dir.mkdir()
    hijack = hijack_dir / "srun"
    hijack.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    hijack.chmod(0o755)
    plan = _synthetic_plan(tmp_path)
    _publish(plan, tmp_path)
    invoked = False

    def execute(*_args: object, **_kwargs: object) -> ProcessCapture:
        nonlocal invoked
        invoked = True
        raise AssertionError("PATH-substituted srun was executed")

    monkeypatch.setattr(supervisor_module, "_execute_bounded", execute)

    with pytest.raises(RankSupervisorError, match="worker-owned|group/world writable"):
        supervise_rank_launch(
            plan,
            cwd=tmp_path,
            supervisor_environment={"PATH": str(hijack_dir)},
        )

    assert not invoked


def test_report_cannot_upgrade_synthetic_body_to_product(tmp_path: Path) -> None:
    plan = _synthetic_plan(tmp_path)
    report_directory, input_digest = _publish(plan, tmp_path)
    reports = {
        report_directory / f"rank-{rank}.json": _report(
            plan, input_digest, rank, body_kind="product"
        )
        for rank in range(plan.world_size)
    }

    with pytest.raises(RankSupervisorError, match="product/synthetic body identity"):
        supervise_rank_launch(
            plan,
            cwd=tmp_path,
            executor=_RecordingExecutor(ProcessCapture(0, b"", b""), reports),
        )


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("missing", "report files are incomplete"),
        ("malformed", "not valid UTF-8 JSON"),
        ("extra", "report files are incomplete"),
        ("payload-rank", "filename differs"),
    ],
)
def test_missing_malformed_or_misnamed_report_fails_closed(
    tmp_path: Path, mode: str, match: str
) -> None:
    plan = _single_plan(tmp_path)
    report_directory, input_digest = _publish(plan, tmp_path)
    reports: dict[Path, bytes] = {}
    if mode == "malformed":
        reports[report_directory / "rank-0.json"] = b"{"
    elif mode == "extra":
        reports[report_directory / "rank-0.json"] = _report(plan, input_digest, 0)
        reports[report_directory / "unexpected"] = b"x"
    elif mode == "payload-rank":
        payload = json.loads(_report(plan, input_digest, 0))
        payload["rank"] = 1
        reports[report_directory / "rank-0.json"] = json.dumps(payload).encode()

    with pytest.raises(RankSupervisorError, match=match):
        supervise_rank_launch(
            plan,
            cwd=tmp_path,
            executor=_RecordingExecutor(ProcessCapture(0, b"", b""), reports),
        )


@pytest.mark.parametrize(
    ("stdout", "stderr", "stdout_truncated", "returncode", "match"),
    [
        (b"", b"failed", False, 7, "exited with status 7"),
        (b"x" * 9, b"", False, 0, "violated the requested capture bound"),
        (b"short", b"", True, 0, "exceeded the capture bound"),
    ],
)
def test_process_failure_or_unbounded_capture_fails_before_receipt(
    tmp_path: Path,
    stdout: bytes,
    stderr: bytes,
    stdout_truncated: bool,
    returncode: int,
    match: str,
) -> None:
    plan = _single_plan(tmp_path)
    _publish(plan, tmp_path)
    capture = ProcessCapture(returncode, stdout, stderr, stdout_truncated=stdout_truncated)
    with pytest.raises(RankSupervisorError, match=match):
        supervise_rank_launch(
            plan,
            cwd=tmp_path,
            executor=_RecordingExecutor(capture),
            capture_limit_bytes=8,
        )


@pytest.mark.parametrize(
    ("hostname", "gpu_id"),
    [("node-b", None), (None, 7)],
)
def test_actual_node_and_gpu_must_match_plan(
    tmp_path: Path, hostname: str | None, gpu_id: int | None
) -> None:
    plan = _single_plan(tmp_path)
    report_directory, input_digest = _publish(plan, tmp_path)
    reports = {
        report_directory / "rank-0.json": _report(
            plan, input_digest, 0, hostname=hostname, gpu_id=gpu_id
        )
    }
    with pytest.raises(RankSupervisorError, match="placement verification"):
        supervise_rank_launch(
            plan,
            cwd=tmp_path,
            executor=_RecordingExecutor(ProcessCapture(0, b"", b""), reports),
        )


def test_input_or_report_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    report_directory, input_digest = _publish(plan, tmp_path)
    reports = {
        report_directory / "rank-0.json": _report(
            plan,
            input_digest,
            0,
            extra={"input_digest": "0" * 64},
        )
    }
    with pytest.raises(RankSupervisorError, match="input digest"):
        supervise_rank_launch(
            plan,
            cwd=tmp_path,
            executor=_RecordingExecutor(ProcessCapture(0, b"", b""), reports),
        )


def test_preexisting_report_directory_content_is_rejected(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    report_directory, _input_digest = _publish(plan, tmp_path)
    (report_directory / "rank-0.json").write_text("stale", encoding="utf-8")

    with pytest.raises(RankSupervisorError, match="must be empty before execution"):
        supervise_rank_launch(
            plan, cwd=tmp_path, executor=_RecordingExecutor(ProcessCapture(0, b"", b""))
        )


def test_rank_input_must_remain_read_only(tmp_path: Path) -> None:
    plan = _single_plan(tmp_path)
    _publish(plan, tmp_path)
    plan.rank_input_path.chmod(stat.S_IRUSR | stat.S_IWUSR)

    with pytest.raises(RankSupervisorError, match="read-only"):
        supervise_rank_launch(plan, cwd=tmp_path)


def test_default_executor_runs_fixed_wrapper_without_ambient_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path(__file__).parents[4]
    script = tmp_path / "rank_body.py"
    script.write_text(
        "import os\n"
        "expected = {'CUDA_VISIBLE_DEVICES': '3', 'LOCAL_RANK': '0', "
        "'RANK': '0', 'TRTLLM_MODELING_V2': 'require', 'WORLD_SIZE': '1'}\n"
        "if any(os.environ.get(key) != value for key, value in expected.items()):\n"
        "    raise SystemExit(12)\n"
        "if 'STAIRCASE_UNAUTHORIZED_AMBIENT' in os.environ:\n"
        "    raise SystemExit(13)\n",
        encoding="utf-8",
    )
    plan = _single_plan(
        tmp_path,
        GateCommand(
            (sys.executable, str(script)),
            environment=(
                ("PYTHONPATH", str(source_root)),
                ("TRTLLM_MODELING_V2", "require"),
            ),
        ),
        hostname=socket.gethostname(),
    )
    _publish(plan, tmp_path)
    monkeypatch.setenv("STAIRCASE_UNAUTHORIZED_AMBIENT", "must-not-leak")

    result = supervise_rank_launch(plan, cwd=tmp_path)

    assert result.receipt.placements == plan.placements


def test_default_executor_drains_but_rejects_output_above_bound(tmp_path: Path) -> None:
    source_root = Path(__file__).parents[4]
    script = tmp_path / "noisy_rank_body.py"
    script.write_text(
        "import sys\nsys.stdout.buffer.write(b'x' * 4096)\nsys.stderr.buffer.write(b'y' * 4096)\n",
        encoding="utf-8",
    )
    plan = _single_plan(
        tmp_path,
        GateCommand(
            (sys.executable, str(script)),
            environment=(
                ("PYTHONPATH", str(source_root)),
                ("TRTLLM_MODELING_V2", "require"),
            ),
        ),
        hostname=socket.gethostname(),
    )
    _publish(plan, tmp_path)

    with pytest.raises(RankSupervisorError, match="exceeded the capture bound"):
        supervise_rank_launch(plan, cwd=tmp_path, capture_limit_bytes=64)
