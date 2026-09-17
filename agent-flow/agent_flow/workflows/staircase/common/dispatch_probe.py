# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, non-mutating proof of Staircase child-dispatch capability."""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from agent_flow.workflows.staircase.task_schema import NormalizedTask

_DISPATCH_MODES = frozenset({"nested_submission", "login_dispatcher", "preallocated_pool"})
_SLURM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_TEST_ONLY_OUTPUT = re.compile(r"sbatch:\s+Job\s+[1-9][0-9]*\s+to\s+start\s+at\s+[ -~]{1,256}\Z")
_SCRIPT = "#!/bin/true\n"
_OUTPUT_LIMIT_BYTES = 4_096
_EVIDENCE_LIMIT_CHARACTERS = 512
# Slurm's test-only RPC can legitimately queue behind controller/accounting
# traffic on a busy production cluster.  Keep the probe bounded, but allow a
# full scheduler round trip before failing closed.
_TIMEOUT_SECONDS = 60.0
_READ_SIZE_BYTES = 1_024
_EMPTY_COMMAND_DIGEST = hashlib.sha256(b"").hexdigest()


class DispatchPhase(str, Enum):
    """Execution boundary whose dispatch capability was actually probed."""

    LOGIN = "login"
    CONTROLLER = "controller"


@dataclass(frozen=True, slots=True)
class DispatchProbeResult:
    """Bounded result from the fixed shell-free probe subprocess."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    output_limited: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise TypeError("probe returncode must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("probe output must be text")
        if not isinstance(self.timed_out, bool) or not isinstance(self.output_limited, bool):
            raise TypeError("probe boundary flags must be booleans")


class DispatchProbeRunner(Protocol):
    """Execution seam for the one fixed ``sbatch --test-only`` command."""

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        input_text: str,
        timeout_seconds: float,
        output_limit_bytes: int,
    ) -> DispatchProbeResult:
        """Run the fixed probe without a shell or scheduler mutation."""


@dataclass(frozen=True, slots=True)
class DispatchProbeReceipt:
    """Immutable phase- and resource-bound proof of one probe outcome."""

    phase: DispatchPhase
    mode: str
    available: bool
    account: str
    partition: str
    qos: str | None
    reservation: str | None
    controller_job_id: str | None
    cluster: str | None
    command_digest: str
    evidence: str
    receipt_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.phase, DispatchPhase):
            raise TypeError("dispatch receipt phase must be a DispatchPhase")
        if self.mode not in _DISPATCH_MODES:
            raise ValueError(f"unsupported dispatch receipt mode: {self.mode!r}")
        if not isinstance(self.available, bool):
            raise TypeError("dispatch receipt availability must be a bool")
        for name, value in (("account", self.account), ("partition", self.partition)):
            _require_slurm_name(value, name)
        for name, value in (("qos", self.qos), ("reservation", self.reservation)):
            if value is not None:
                _require_slurm_name(value, name)
        if self.controller_job_id is not None and _JOB_ID.fullmatch(self.controller_job_id) is None:
            raise ValueError("controller_job_id must be one exact positive numeric job ID")
        if self.cluster is not None:
            _require_slurm_name(self.cluster, "cluster")
        if self.phase is DispatchPhase.LOGIN and (
            self.controller_job_id is not None or self.cluster is not None
        ):
            raise ValueError("login dispatch receipt cannot claim a controller allocation")
        if (
            self.phase is DispatchPhase.CONTROLLER
            and self.available
            and (self.controller_job_id is None or self.cluster is None)
        ):
            raise ValueError("available controller dispatch requires exact job and cluster binding")
        _require_digest(self.command_digest, "command_digest")
        if (
            not self.evidence
            or self.evidence != _sanitize_evidence(self.evidence)
            or len(self.evidence) > _EVIDENCE_LIMIT_CHARACTERS
        ):
            raise ValueError("dispatch evidence must be non-empty, sanitized, and bounded")
        object.__setattr__(self, "receipt_digest", _receipt_digest(self))


@dataclass(frozen=True, slots=True)
class DispatchContract:
    """Preflight contract backed exclusively by an immutable probe receipt."""

    receipt: DispatchProbeReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, DispatchProbeReceipt):
            raise TypeError("dispatch contract requires a typed probe receipt")
        if self.receipt.receipt_digest != _receipt_digest(self.receipt):
            raise ValueError("dispatch receipt digest does not match its immutable fields")

    @property
    def phase(self) -> DispatchPhase:
        """Return the execution boundary actually probed."""
        return self.receipt.phase

    @property
    def mode(self) -> str:
        """Return the configured dispatch mode actually probed."""
        return self.receipt.mode

    @property
    def available(self) -> bool:
        """Whether the fixed phase-appropriate probe succeeded."""
        return self.receipt.available

    @property
    def evidence(self) -> str:
        """Return bounded sanitized diagnostic evidence."""
        return self.receipt.evidence

    @property
    def receipt_digest(self) -> str:
        """Return the canonical immutable receipt digest."""
        return self.receipt.receipt_digest


def probe_dispatch_capability(
    task: NormalizedTask,
    phase: DispatchPhase,
    *,
    runner: DispatchProbeRunner | None = None,
    environment: Mapping[str, str] | None = None,
) -> DispatchContract:
    """Probe dispatch without creating or cancelling a scheduler job.

    ``nested_submission`` is tested only through ``sbatch --test-only`` with a
    fixed stdin script and minimal CPU resources.  The other configured modes
    remain unavailable until concrete non-mutating adapters are implemented.
    """
    if not isinstance(task, NormalizedTask):
        raise TypeError("task must be a NormalizedTask")
    if not isinstance(phase, DispatchPhase):
        raise TypeError("phase must be a DispatchPhase")
    controller = task.execution.slurm.controller
    mode = controller.dispatch_mode
    if mode != "nested_submission":
        return _contract(
            task,
            phase,
            available=False,
            controller_job_id=None,
            cluster=None,
            command_digest=_EMPTY_COMMAND_DIGEST,
            evidence=f"{mode} has no concrete non-mutating dispatch adapter",
        )

    controller_job_id: str | None = None
    cluster: str | None = None
    if phase is DispatchPhase.CONTROLLER:
        observed_environment = os.environ if environment is None else environment
        try:
            controller_job_id = observed_environment.get("SLURM_JOB_ID")
            cluster = observed_environment.get("SLURM_CLUSTER_NAME")
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            controller_job_id = None
            cluster = None
        if not isinstance(controller_job_id, str) or _JOB_ID.fullmatch(controller_job_id) is None:
            return _contract(
                task,
                phase,
                available=False,
                controller_job_id=None,
                cluster=None,
                command_digest=_EMPTY_COMMAND_DIGEST,
                evidence="controller probe requires exact numeric SLURM_JOB_ID",
            )
        if not isinstance(cluster, str) or _SLURM_NAME.fullmatch(cluster) is None:
            return _contract(
                task,
                phase,
                available=False,
                controller_job_id=controller_job_id,
                cluster=None,
                command_digest=_EMPTY_COMMAND_DIGEST,
                evidence="controller probe requires a safe exact SLURM_CLUSTER_NAME",
            )

    argv = _probe_argv(task, phase, controller_job_id, cluster)
    command_digest = _command_digest(argv)
    command_runner = runner or run_dispatch_probe_command
    try:
        result = command_runner(
            argv,
            input_text=_SCRIPT,
            timeout_seconds=_TIMEOUT_SECONDS,
            output_limit_bytes=_OUTPUT_LIMIT_BYTES,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return _contract(
            task,
            phase,
            available=False,
            controller_job_id=controller_job_id,
            cluster=cluster,
            command_digest=command_digest,
            evidence=f"dispatch probe runner failed: {type(error).__name__}: {error}",
        )
    if not isinstance(result, DispatchProbeResult):
        return _contract(
            task,
            phase,
            available=False,
            controller_job_id=controller_job_id,
            cluster=cluster,
            command_digest=command_digest,
            evidence="dispatch probe runner returned an invalid typed result",
        )
    output_size = len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8"))
    if result.timed_out:
        evidence = "sbatch --test-only exceeded its bounded timeout"
        available = False
    elif result.output_limited or output_size > _OUTPUT_LIMIT_BYTES:
        evidence = "sbatch --test-only exceeded its bounded output limit"
        available = False
    elif result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        evidence = f"sbatch --test-only unavailable (exit {result.returncode}): {diagnostic}"
        available = False
    else:
        output = _single_probe_output(result.stdout, result.stderr)
        if output is None or _TEST_ONLY_OUTPUT.fullmatch(output) is None:
            evidence = "sbatch --test-only returned malformed success output"
            available = False
        else:
            evidence = f"fixed sbatch --test-only accepted: {output}"
            available = True
    return _contract(
        task,
        phase,
        available=available,
        controller_job_id=controller_job_id,
        cluster=cluster,
        command_digest=command_digest,
        evidence=evidence,
    )


def dispatch_contract_matches(
    task: NormalizedTask,
    phase: DispatchPhase,
    contract: DispatchContract,
) -> bool:
    """Verify that a receipt is bound to the exact normalized fixed probe."""
    if (
        not isinstance(task, NormalizedTask)
        or not isinstance(phase, DispatchPhase)
        or not isinstance(contract, DispatchContract)
    ):
        return False
    controller = task.execution.slurm.controller
    receipt = contract.receipt
    if (
        receipt.phase is not phase
        or receipt.mode != controller.dispatch_mode
        or receipt.account != controller.account
        or receipt.partition != controller.partition
        or receipt.qos != controller.qos
        or receipt.reservation != controller.reservation
        or receipt.receipt_digest != _receipt_digest(receipt)
    ):
        return False
    if not receipt.available:
        return True
    if receipt.mode != "nested_submission":
        return False
    if phase is DispatchPhase.LOGIN:
        if receipt.controller_job_id is not None or receipt.cluster is not None:
            return False
    elif receipt.controller_job_id is None or receipt.cluster is None:
        return False
    expected_argv = _probe_argv(
        task,
        phase,
        receipt.controller_job_id,
        receipt.cluster,
    )
    return receipt.command_digest == _command_digest(expected_argv)


def run_dispatch_probe_command(
    argv: tuple[str, ...],
    *,
    input_text: str,
    timeout_seconds: float,
    output_limit_bytes: int,
) -> DispatchProbeResult:
    """Execute only the internally generated bounded ``sbatch --test-only``."""
    _validate_probe_invocation(argv, input_text, timeout_seconds, output_limit_bytes)
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
        )
    except OSError as error:
        return DispatchProbeResult(-1, stderr=f"could not start sbatch: {error}")
    if process.stdin is None:
        process.kill()
        process.wait()
        return DispatchProbeResult(-1, stderr="sbatch stdin was unavailable")
    try:
        process.stdin.write(input_text.encode("utf-8"))
        process.stdin.close()
    except (BrokenPipeError, OSError) as error:
        _terminate(process)
        return DispatchProbeResult(-1, stderr=f"could not write fixed probe script: {error}")
    return _read_bounded(process, timeout_seconds, output_limit_bytes)


def _read_bounded(
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
    output_limit_bytes: int,
) -> DispatchProbeResult:
    if process.stdout is None or process.stderr is None:
        _terminate(process)
        return DispatchProbeResult(-1, stderr="sbatch output pipes were unavailable")
    selector = selectors.DefaultSelector()
    streams = (("stdout", process.stdout), ("stderr", process.stderr))
    for label, stream in streams:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, label)
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate(process)
                return _decoded_result(-1, captured, timed_out=True)
            events = selector.select(remaining)
            if not events:
                continue
            for key, _ in events:
                stream = key.fileobj
                try:
                    chunk = os.read(stream.fileno(), _READ_SIZE_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                total += len(chunk)
                if total > output_limit_bytes:
                    _terminate(process)
                    return _decoded_result(-1, captured, output_limited=True)
                captured[key.data].extend(chunk)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate(process)
            return _decoded_result(-1, captured, timed_out=True)
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate(process)
            return _decoded_result(-1, captured, timed_out=True)
        return _decoded_result(process.returncode, captured)
    finally:
        selector.close()
        for _, stream in streams:
            if not stream.closed:
                stream.close()


def _decoded_result(
    returncode: int,
    captured: dict[str, bytearray],
    *,
    timed_out: bool = False,
    output_limited: bool = False,
) -> DispatchProbeResult:
    return DispatchProbeResult(
        returncode,
        captured["stdout"].decode("utf-8", errors="replace"),
        captured["stderr"].decode("utf-8", errors="replace"),
        timed_out,
        output_limited,
    )


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait()


def _probe_argv(
    task: NormalizedTask,
    phase: DispatchPhase,
    controller_job_id: str | None,
    cluster: str | None,
) -> tuple[str, ...]:
    controller = task.execution.slurm.controller
    values = [
        "sbatch",
        "--test-only",
        "--job-name=staircase-dispatch-probe",
        f"--account={controller.account}",
        f"--partition={controller.partition}",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=16M",
        "--time=00:01:00",
        "--no-requeue",
        "--export=NONE",
    ]
    if controller.qos is not None:
        values.append(f"--qos={controller.qos}")
    if controller.reservation is not None:
        values.append(f"--reservation={controller.reservation}")
    if phase is DispatchPhase.LOGIN:
        values.append("--comment=staircase-dispatch-probe-login")
    else:
        values.extend(
            (
                f"--clusters={cluster}",
                f"--comment=staircase-dispatch-probe-parent-{controller_job_id}",
            )
        )
    return tuple(values)


def _validate_probe_invocation(
    argv: tuple[str, ...],
    input_text: str,
    timeout_seconds: float,
    output_limit_bytes: int,
) -> None:
    if not isinstance(argv, tuple) or any(
        not isinstance(value, str) or "\x00" in value for value in argv
    ):
        raise ValueError("dispatch probe accepts only fixed sbatch --test-only argv")
    required = {
        "sbatch",
        "--test-only",
        "--job-name=staircase-dispatch-probe",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=16M",
        "--time=00:01:00",
        "--no-requeue",
        "--export=NONE",
    }
    if (
        len(argv) != len(set(argv))
        or not required.issubset(argv)
        or argv[:2]
        != (
            "sbatch",
            "--test-only",
        )
    ):
        raise ValueError("dispatch probe accepts only fixed sbatch --test-only argv")
    dynamic_prefixes = (
        "--account=",
        "--partition=",
        "--qos=",
        "--reservation=",
        "--clusters=",
        "--comment=",
    )
    dynamic_counts = {prefix: 0 for prefix in dynamic_prefixes}
    for value in argv:
        if value in required:
            continue
        prefix = next((item for item in dynamic_prefixes if value.startswith(item)), None)
        if prefix is None:
            raise ValueError("dispatch probe argv contains an unsupported option")
        dynamic_counts[prefix] += 1
        argument = value.removeprefix(prefix)
        if prefix == "--comment=":
            if (
                argument != "staircase-dispatch-probe-login"
                and re.fullmatch(r"staircase-dispatch-probe-parent-[1-9][0-9]*", argument) is None
            ):
                raise ValueError("dispatch probe comment is not a fixed phase binding")
        elif _SLURM_NAME.fullmatch(argument) is None:
            raise ValueError("dispatch probe scheduler option contains unsafe characters")
    if (
        dynamic_counts["--account="] != 1
        or dynamic_counts["--partition="] != 1
        or dynamic_counts["--comment="] != 1
        or dynamic_counts["--qos="] > 1
        or dynamic_counts["--reservation="] > 1
        or dynamic_counts["--clusters="] > 1
    ):
        raise ValueError("dispatch probe scheduler options are missing or duplicated")
    comment = next(
        value.removeprefix("--comment=") for value in argv if value.startswith("--comment=")
    )
    if (comment == "staircase-dispatch-probe-login" and dynamic_counts["--clusters="] != 0) or (
        comment.startswith("staircase-dispatch-probe-parent-")
        and dynamic_counts["--clusters="] != 1
    ):
        raise ValueError("dispatch probe phase binding and cluster binding disagree")
    if input_text != _SCRIPT:
        raise ValueError("dispatch probe accepts only its fixed minimal script")
    if timeout_seconds != _TIMEOUT_SECONDS:
        raise ValueError("dispatch probe timeout is fixed")
    if output_limit_bytes != _OUTPUT_LIMIT_BYTES:
        raise ValueError("dispatch probe output limit is fixed")


def _single_probe_output(stdout: str, stderr: str) -> str | None:
    lines = [
        line.strip() for value in (stdout, stderr) for line in value.splitlines() if line.strip()
    ]
    if len(lines) != 1:
        return None
    return lines[0]


def _command_digest(argv: tuple[str, ...]) -> str:
    payload = "\0".join((*argv, _SCRIPT)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract(
    task: NormalizedTask,
    phase: DispatchPhase,
    *,
    available: bool,
    controller_job_id: str | None,
    cluster: str | None,
    command_digest: str,
    evidence: str,
) -> DispatchContract:
    controller = task.execution.slurm.controller
    return DispatchContract(
        DispatchProbeReceipt(
            phase=phase,
            mode=controller.dispatch_mode,
            available=available,
            account=controller.account,
            partition=controller.partition,
            qos=controller.qos,
            reservation=controller.reservation,
            controller_job_id=controller_job_id,
            cluster=cluster,
            command_digest=command_digest,
            evidence=_sanitize_evidence(evidence),
        )
    )


def _sanitize_evidence(value: str) -> str:
    printable = "".join(character if " " <= character <= "~" else " " for character in value)
    normalized = " ".join(printable.split()) or "no diagnostic"
    if len(normalized) <= _EVIDENCE_LIMIT_CHARACTERS:
        return normalized
    return normalized[: _EVIDENCE_LIMIT_CHARACTERS - 14] + "...[truncated]"


def _receipt_digest(receipt: DispatchProbeReceipt) -> str:
    payload = {
        "phase": receipt.phase.value,
        "mode": receipt.mode,
        "available": receipt.available,
        "account": receipt.account,
        "partition": receipt.partition,
        "qos": receipt.qos,
        "reservation": receipt.reservation,
        "controller_job_id": receipt.controller_job_id,
        "cluster": receipt.cluster,
        "command_digest": receipt.command_digest,
        "evidence": receipt.evidence,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_slurm_name(value: str, name: str) -> None:
    if not isinstance(value, str) or _SLURM_NAME.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe Slurm name")


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "DispatchContract",
    "DispatchPhase",
    "DispatchProbeReceipt",
    "DispatchProbeResult",
    "DispatchProbeRunner",
    "dispatch_contract_matches",
    "probe_dispatch_capability",
    "run_dispatch_probe_command",
]
