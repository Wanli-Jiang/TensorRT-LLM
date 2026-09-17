# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted, shell-free execution of validated rank launch plans.

The supervisor is intentionally narrower than a scheduler adapter.  It runs
one already-authorized :class:`RankLaunchPlan`, captures bounded output, and
turns complete runtime placement reports into a receipt.  It cannot submit,
cancel, or recursively launch another allocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Callable, Literal, Mapping, cast

from .launchers import (
    LaunchKind,
    ObservedProductIdentity,
    ObservedRank,
    RankLaunchPlan,
    RankPlacementReceipt,
    make_rank_placement_receipt,
)
from .rank_worker import RankInput, RankWorkerError, load_rank_input

DEFAULT_CAPTURE_LIMIT_BYTES = 1024 * 1024
MAX_CAPTURE_LIMIT_BYTES = 16 * 1024 * 1024

_MAX_REPORT_BYTES = 64 * 1024
_MAX_ERROR_DETAIL_BYTES = 4 * 1024
_REPORT_KEYS = frozenset(
    {
        "schema_version",
        "input_digest",
        "plan_digest",
        "rank_command_digest",
        "launch_kind",
        "body_kind",
        "rank",
        "hostname",
        "local_rank",
        "gpu_id",
        "product_identity",
    }
)
_PROHIBITED_RANK_PROGRAMS = frozenset(
    {
        "bash",
        "dash",
        "mpiexec",
        "mpirun",
        "sacct",
        "sbatch",
        "scancel",
        "sh",
        "squeue",
        "srun",
        "ssh",
        "zsh",
    }
)


class RankSupervisorError(RuntimeError):
    """Raised when execution cannot produce trustworthy rank evidence."""


@dataclass(frozen=True, slots=True)
class ProcessCapture:
    """Bounded output returned by a shell-free process executor."""

    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise RankSupervisorError("process returncode must be an integer")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise RankSupervisorError("captured stdout and stderr must be bytes")
        if not isinstance(self.stdout_truncated, bool) or not isinstance(
            self.stderr_truncated, bool
        ):
            raise RankSupervisorError("capture truncation flags must be booleans")


RankExecutor = Callable[
    [
        tuple[str, ...],
        Path,
        tuple[tuple[str, str], ...],
        Literal[False],
        int,
    ],
    ProcessCapture,
]


@dataclass(frozen=True, slots=True)
class RankSupervisionResult:
    """Successful bounded process output and its verified placement receipt."""

    receipt: RankPlacementReceipt
    stdout: bytes
    stderr: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, RankPlacementReceipt):
            raise RankSupervisorError("receipt must use RankPlacementReceipt")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise RankSupervisorError("captured stdout and stderr must be bytes")


def supervise_rank_launch(
    plan: RankLaunchPlan,
    *,
    cwd: Path,
    executor: RankExecutor | None = None,
    capture_limit_bytes: int = DEFAULT_CAPTURE_LIMIT_BYTES,
    supervisor_environment: Mapping[str, str] | None = None,
) -> RankSupervisionResult:
    """Execute an exact launch plan and verify one report from every rank.

    Args:
        plan: Controller-authorized launch plan.
        cwd: Canonical existing working directory for the product command.
        executor: Injectable shell-free executor. The default drains both
            process streams while retaining at most ``capture_limit_bytes``
            from each stream.
        capture_limit_bytes: Positive per-stream capture bound, capped by the
            supervisor's fixed maximum.

    Returns:
        A placement receipt and bounded process output after every check passes.

    Raises:
        RankSupervisorError: If execution or any rank report fails closed.
    """
    _validate_executable_plan(plan)
    canonical_cwd = _canonical_directory(cwd)
    _validate_capture_limit(capture_limit_bytes)
    try:
        rank_input, input_digest = load_rank_input(plan.rank_input_path)
    except RankWorkerError as error:
        raise RankSupervisorError(f"rank input validation failed: {error}") from error
    _validate_rank_input_for_plan(rank_input, plan, canonical_cwd)
    if any(rank_input.report_directory.iterdir()):
        raise RankSupervisorError("rank report directory must be empty before execution")
    process_executor = executor or _execute_bounded
    execution_argv = plan.argv
    if supervisor_environment is None:
        execution_environment = (
            tuple(sorted(os.environ.items())) if executor is None else plan.environment
        )
    else:
        if not all(
            isinstance(name, str) and isinstance(value, str)
            for name, value in supervisor_environment.items()
        ):
            raise RankSupervisorError("supervisor environment must contain only strings")
        execution_environment = tuple(sorted(supervisor_environment.items()))
    if executor is None and plan.kind is not LaunchKind.SINGLE_PROCESS:
        trusted_srun = resolve_trusted_scheduler_client(
            "srun",
            dict(execution_environment),
        )
        execution_argv = (str(trusted_srun), *plan.argv[1:])
    try:
        capture = process_executor(
            execution_argv,
            canonical_cwd,
            execution_environment,
            False,
            capture_limit_bytes,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RankSupervisorError(f"rank process execution failed: {error}") from error
    if not isinstance(capture, ProcessCapture):
        raise RankSupervisorError("rank executor returned an invalid process capture")
    if len(capture.stdout) > capture_limit_bytes or len(capture.stderr) > capture_limit_bytes:
        raise RankSupervisorError("rank executor violated the requested capture bound")
    if capture.stdout_truncated or capture.stderr_truncated:
        raise RankSupervisorError("rank output exceeded the capture bound")
    if capture.returncode != 0:
        stderr = capture.stderr[:_MAX_ERROR_DETAIL_BYTES].decode("utf-8", errors="replace").strip()
        detail = f": {stderr}" if stderr else ""
        raise RankSupervisorError(f"rank process exited with status {capture.returncode}{detail}")

    observations = _collect_rank_reports(plan, rank_input, input_digest)
    try:
        receipt = make_rank_placement_receipt(plan, observations)
    except ValueError as error:
        raise RankSupervisorError(f"rank placement verification failed: {error}") from error
    return RankSupervisionResult(receipt=receipt, stdout=capture.stdout, stderr=capture.stderr)


def _execute_bounded(
    argv: tuple[str, ...],
    cwd: Path,
    environment: tuple[tuple[str, str], ...],
    shell: Literal[False],
    capture_limit_bytes: int,
) -> ProcessCapture:
    """Run one process with no ambient environment and bounded stream retention."""
    if shell is not False:
        raise RankSupervisorError("rank execution must set shell=False")
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(environment),
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise RankSupervisorError("rank process pipes were not created")

    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    stdout_truncated = [False]
    stderr_truncated = [False]
    reader_errors: list[OSError] = []
    stdout_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stdout, stdout_chunks, stdout_truncated, capture_limit_bytes, reader_errors),
        name="staircase-rank-stdout",
    )
    stderr_thread = threading.Thread(
        target=_drain_stream,
        args=(process.stderr, stderr_chunks, stderr_truncated, capture_limit_bytes, reader_errors),
        name="staircase-rank-stderr",
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    process.stdout.close()
    process.stderr.close()
    if reader_errors:
        raise RankSupervisorError(f"could not capture rank process output: {reader_errors[0]}")
    return ProcessCapture(
        returncode=returncode,
        stdout=b"".join(stdout_chunks),
        stderr=b"".join(stderr_chunks),
        stdout_truncated=stdout_truncated[0],
        stderr_truncated=stderr_truncated[0],
    )


def resolve_trusted_scheduler_client(
    executable: str,
    environment: Mapping[str, str],
) -> Path:
    """Resolve a scheduler client through an immutable absolute trust path."""
    search_path = environment.get("PATH")
    if not search_path:
        raise RankSupervisorError("trusted scheduler client PATH is unavailable")
    entries = search_path.split(os.pathsep)
    if any(not entry or not Path(entry).is_absolute() for entry in entries):
        raise RankSupervisorError("scheduler client PATH must contain absolute entries")
    candidate = shutil.which(executable, path=search_path)
    if candidate is None:
        raise RankSupervisorError(f"trusted scheduler client {executable!r} is unavailable")
    unresolved = Path(candidate)
    if not unresolved.is_absolute():
        raise RankSupervisorError("scheduler client resolution returned a relative path")
    try:
        resolved = unresolved.resolve(strict=True)
        executable_info = resolved.stat()
    except OSError as error:
        raise RankSupervisorError(f"scheduler client cannot be inspected: {error}") from error
    if not stat.S_ISREG(executable_info.st_mode) or not os.access(resolved, os.X_OK):
        raise RankSupervisorError("scheduler client must be an executable regular file")

    effective_uid = os.geteuid()
    component = resolved
    while True:
        try:
            info = component.stat()
        except OSError as error:
            raise RankSupervisorError(
                f"scheduler client trust path cannot be inspected: {error}"
            ) from error
        if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RankSupervisorError("scheduler client trust path is group/world writable")
        if effective_uid != 0 and info.st_uid == effective_uid:
            raise RankSupervisorError("scheduler client trust path is worker-owned")
        if component.parent == component:
            break
        component = component.parent
    return resolved


def _drain_stream(
    stream: object,
    chunks: list[bytes],
    truncated: list[bool],
    capture_limit_bytes: int,
    errors: list[OSError],
) -> None:
    retained = 0
    try:
        while True:
            block = stream.read(64 * 1024)
            if not block:
                return
            remaining = capture_limit_bytes - retained
            if remaining > 0:
                kept = block[:remaining]
                chunks.append(kept)
                retained += len(kept)
            if len(block) > remaining:
                truncated[0] = True
    except OSError as error:
        errors.append(error)


def _collect_rank_reports(
    plan: RankLaunchPlan, rank_input: RankInput, input_digest: str
) -> tuple[ObservedRank, ...]:
    observations: list[ObservedRank] = []
    seen_ranks: set[int] = set()
    expected_body_kind = "product" if plan.product_rank_body else "synthetic"
    expected_names = {f"rank-{rank}.json" for rank in range(plan.world_size)}
    if plan.kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT:
        expected_names.update(
            f"rank-{rank}-product-evidence.json" for rank in range(plan.world_size)
        )
    actual_paths = tuple(rank_input.report_directory.iterdir())
    actual_names = {path.name for path in actual_paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise RankSupervisorError(
            f"rank report files are incomplete: missing={missing!r}, unexpected={unexpected!r}"
        )
    report_paths = tuple(
        path
        for path in actual_paths
        if path.name.endswith(".json") and "-product-evidence" not in path.name
    )
    for path in report_paths:
        if path.is_symlink() or not path.is_file():
            raise RankSupervisorError("rank report must be a regular non-symlink file")
        if path.stat().st_size > _MAX_REPORT_BYTES:
            raise RankSupervisorError("rank report exceeds the fixed size bound")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise RankSupervisorError(f"could not read rank report: {error}") from error
        report = _decode_report(payload)
        if report["input_digest"] != input_digest:
            raise RankSupervisorError("rank report input digest differs from rank input")
        if report["plan_digest"] != plan.plan_digest:
            raise RankSupervisorError("rank report plan digest differs from the launch plan")
        if report["rank_command_digest"] != plan.rank_command_digest:
            raise RankSupervisorError("rank report command identity differs from the launch plan")
        if report["launch_kind"] != plan.kind.value:
            raise RankSupervisorError("rank report launch kind differs from the launch plan")
        if report["body_kind"] != expected_body_kind:
            raise RankSupervisorError(
                "rank report product/synthetic body identity differs from plan"
            )
        rank = _report_integer(report, "rank")
        if path.name != f"rank-{rank}.json":
            raise RankSupervisorError("rank report filename differs from its payload rank")
        if rank in seen_ranks:
            raise RankSupervisorError(f"duplicate report for rank {rank}")
        seen_ranks.add(rank)
        product_identity = _report_product_identity(report["product_identity"])
        if plan.kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT:
            if product_identity is None:
                raise RankSupervisorError(
                    "real multi-node rank report lacks product identity evidence"
                )
            evidence_path = rank_input.report_directory / f"rank-{rank}-product-evidence.json"
            try:
                body_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            except OSError as error:
                raise RankSupervisorError(
                    f"could not re-read product identity evidence: {error}"
                ) from error
            if body_digest != product_identity.body_evidence_digest:
                raise RankSupervisorError(
                    "product identity evidence digest differs from rank report"
                )
        elif product_identity is not None:
            raise RankSupervisorError(
                "non-product-multi-node report cannot carry product identity evidence"
            )
        observations.append(
            ObservedRank(
                rank=rank,
                hostname=_report_string(report, "hostname"),
                local_rank=_report_integer(report, "local_rank"),
                gpu_id=_report_integer(report, "gpu_id"),
                product_identity=product_identity,
            )
        )
    expected_ranks = set(range(plan.world_size))
    if seen_ranks != expected_ranks:
        raise RankSupervisorError("rank report filenames and payload ranks do not agree")
    return tuple(observations)


def _report_product_identity(value: object) -> ObservedProductIdentity | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise RankSupervisorError("rank report product_identity must be an object or null")
    identity = cast(dict[str, object], value)
    expected_keys = {
        "repository_commit",
        "python_tensorrt_llm_path",
        "native_build_identity",
        "image_identity",
        "compute_capability",
        "collective_backend",
        "transport",
        "cuda_device_uuid",
        "cuda_device_model",
        "body_evidence_digest",
    }
    if set(identity) != expected_keys:
        missing = sorted(expected_keys - set(identity))
        unknown = sorted(set(identity) - expected_keys)
        raise RankSupervisorError(
            "rank report product_identity keys differ from schema: "
            f"missing={missing!r}, unknown={unknown!r}"
        )
    try:
        return ObservedProductIdentity(
            repository_commit=_report_string(identity, "repository_commit"),
            python_tensorrt_llm_path=_report_string(identity, "python_tensorrt_llm_path"),
            native_build_identity=_report_string(identity, "native_build_identity"),
            image_identity=_report_string(identity, "image_identity"),
            compute_capability=_report_string(identity, "compute_capability"),
            collective_backend=_report_string(identity, "collective_backend"),
            transport=_report_string(identity, "transport"),
            cuda_device_uuid=_report_string(identity, "cuda_device_uuid"),
            cuda_device_model=_report_string(identity, "cuda_device_model"),
            body_evidence_digest=_report_string(identity, "body_evidence_digest"),
        )
    except ValueError as error:
        raise RankSupervisorError(f"invalid rank product identity: {error}") from error


def _decode_report(payload: bytes) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RankSupervisorError("rank report is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RankSupervisorError("rank report must be a JSON object")
    report = cast(dict[str, object], value)
    if set(report) != _REPORT_KEYS:
        missing = sorted(_REPORT_KEYS - set(report))
        unknown = sorted(set(report) - _REPORT_KEYS)
        raise RankSupervisorError(
            f"rank report keys differ from schema: missing={missing!r}, unknown={unknown!r}"
        )
    if report["schema_version"] != 1 or isinstance(report["schema_version"], bool):
        raise RankSupervisorError("rank report schema_version must equal integer 1")
    for name in (
        "input_digest",
        "plan_digest",
        "rank_command_digest",
        "launch_kind",
        "body_kind",
    ):
        _report_string(report, name)
    return report


def _report_integer(report: Mapping[str, object], name: str) -> int:
    value = report[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RankSupervisorError(f"rank report {name} must be a non-negative integer")
    return value


def _report_string(report: Mapping[str, object], name: str) -> str:
    value = report[name]
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise RankSupervisorError(f"rank report {name} must be a non-empty single-line string")
    return value


def _validate_executable_plan(plan: RankLaunchPlan) -> None:
    if not isinstance(plan, RankLaunchPlan):
        raise RankSupervisorError("plan must use RankLaunchPlan")
    body_program = PurePath(plan.rank_command.argv[0]).name.lower()
    if body_program in _PROHIBITED_RANK_PROGRAMS:
        raise RankSupervisorError(
            "rank body cannot invoke a shell, scheduler, SSH, or rank launcher"
        )
    wrapper = (
        str(Path(sys.executable).absolute()),
        "-m",
        "agent_flow.workflows.staircase.internal",
        "rank-worker",
        "--input",
        str(plan.rank_input_path),
    )
    if tuple(plan.argv[-len(wrapper) :]) != wrapper:
        raise RankSupervisorError("launch argv must end with the fixed private rank worker")
    assignments = tuple(f"{name}={value}" for name, value in plan.environment)
    if plan.kind is LaunchKind.SINGLE_PROCESS:
        expected = ("/usr/bin/env", "-i", *assignments, *wrapper)
        if plan.argv != expected:
            raise RankSupervisorError("single-process argv differs from the isolated launch plan")
        return
    if plan.argv[0] != "srun":
        raise RankSupervisorError("multi-rank execution requires the plan's exact srun step")
    export_arguments = tuple(argument for argument in plan.argv if argument.startswith("--export="))
    if export_arguments != (f"--export={','.join(assignments)}",):
        raise RankSupervisorError("srun export differs from the plan environment")
    job_arguments = tuple(argument for argument in plan.argv if argument.startswith("--jobid="))
    expected_job = f"--jobid={plan.allocation.slurm_job_id}"
    if job_arguments != (expected_job,):
        raise RankSupervisorError("srun job identity differs from the exact allocation")


def _validate_rank_input_for_plan(rank_input: RankInput, plan: RankLaunchPlan, cwd: Path) -> None:
    expected_body_kind = "product" if plan.product_rank_body else "synthetic"
    expected_environment = tuple(sorted(plan.rank_command.environment))
    if rank_input.plan_digest != plan.plan_digest:
        raise RankSupervisorError("rank input plan digest differs from launch plan")
    if rank_input.rank_command_digest != plan.rank_command_digest:
        raise RankSupervisorError("rank input command digest differs from launch plan")
    if rank_input.launch_kind is not plan.kind or rank_input.body_kind != expected_body_kind:
        raise RankSupervisorError("rank input launch/body identity differs from launch plan")
    if rank_input.argv != plan.rank_command.argv or rank_input.environment != expected_environment:
        raise RankSupervisorError("rank input body command differs from launch plan")
    if rank_input.cwd != cwd:
        raise RankSupervisorError("rank input body cwd differs from supervisor cwd")
    if rank_input.world_size != plan.world_size or rank_input.placements != plan.placements:
        raise RankSupervisorError("rank input topology differs from launch plan")
    if rank_input.expected_product_identity != plan.expected_product_identity:
        raise RankSupervisorError("rank input product identity differs from launch plan")


def _canonical_directory(path: Path) -> Path:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or path.is_symlink()
        or not path.is_dir()
    ):
        raise RankSupervisorError("rank working directory must be an absolute regular directory")
    canonical = path.resolve(strict=True)
    if canonical != path:
        raise RankSupervisorError("rank working directory must be canonical")
    return canonical


def _validate_capture_limit(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 < value <= MAX_CAPTURE_LIMIT_BYTES
    ):
        raise RankSupervisorError(
            f"capture_limit_bytes must be between 1 and {MAX_CAPTURE_LIMIT_BYTES}"
        )


__all__ = [
    "DEFAULT_CAPTURE_LIMIT_BYTES",
    "MAX_CAPTURE_LIMIT_BYTES",
    "ProcessCapture",
    "RankExecutor",
    "RankSupervisionResult",
    "RankSupervisorError",
    "supervise_rank_launch",
]
