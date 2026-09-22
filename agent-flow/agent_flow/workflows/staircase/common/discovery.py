# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only collection of production preflight facts for Staircase.

The collector deliberately separates observations supplied by the launcher
from facts that can be discovered locally. In particular, build identity,
mounts, and dispatch capability are explicit inputs: command presence is not
evidence that a site permits nested Slurm submission.
"""

from __future__ import annotations

import os
import re
import selectors
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from agent_flow.workflows.staircase.task_schema import NormalizedTask

from .dispatch_probe import DispatchProbeRunner, probe_dispatch_capability
from .preflight import ObservedMount, PreflightPhase, PreflightSnapshot, required_commands

_EXACT_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}\Z")
_MAX_OUTPUT_BYTES = 16_384
_MAX_DIAGNOSTIC_CHARACTERS = 2_000
_COMMAND_TIMEOUT_SECONDS = 30.0
_READ_SIZE_BYTES = 4_096


class DiscoveryError(RuntimeError):
    """Raised when a required read-only fact cannot be collected safely."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Bounded result returned by a shell-free command runner."""

    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        if isinstance(self.returncode, bool) or not isinstance(self.returncode, int):
            raise TypeError("command returncode must be an integer")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise TypeError("command output must be text")


class CommandRunner(Protocol):
    """Shell-free, output-bounded command execution seam."""

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        output_limit_bytes: int,
    ) -> CommandResult:
        """Run one argv without a shell and return bounded output."""


@dataclass(frozen=True, slots=True)
class ObservedExecutionFacts:
    """Launcher-supplied facts that cannot be inferred from local commands.

    ``build_identity`` must identify the exact runtime build selected by the
    caller, including an explicitly selected local-development build when one
    is used. Dispatch is intentionally absent: the collector runs the fixed
    non-mutating typed probe instead of accepting an operator assertion.
    """

    repository_root: Path
    workspace_root: Path
    container_image: Path
    build_identity: str
    mounts: tuple[ObservedMount, ...]

    def __post_init__(self) -> None:
        for name in ("repository_root", "workspace_root", "container_image"):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be a Path")
        if not isinstance(self.build_identity, str) or not self.build_identity.strip():
            raise ValueError("build_identity must be an explicit non-empty fact")
        if not isinstance(self.mounts, tuple) or not all(
            isinstance(mount, ObservedMount) for mount in self.mounts
        ):
            raise TypeError("mounts must be a tuple of ObservedMount values")


PathResolver = Callable[[Path], Path]
CommandLocator = Callable[[str], str | None]


def collect_preflight_snapshot(
    task: NormalizedTask,
    phase: PreflightPhase,
    facts: ObservedExecutionFacts,
    *,
    command_runner: CommandRunner | None = None,
    dispatch_probe_runner: DispatchProbeRunner | None = None,
    dispatch_environment: Mapping[str, str] | None = None,
    command_locator: CommandLocator = shutil.which,
    path_resolver: PathResolver | None = None,
) -> PreflightSnapshot:
    """Collect a deterministic preflight snapshot without external mutation.

    Args:
        task: Frozen normalized task whose command requirements apply.
        phase: Login or controller execution boundary being observed.
        facts: Explicit launcher, mount, and build observations. Dispatch is
            always collected by the typed probe.
        command_runner: Injectable shell-free runner. The production default
            only accepts the collector's fixed read-only Git commands.
        dispatch_probe_runner: Injectable runner for the fixed bounded
            ``sbatch --test-only`` capability probe.
        dispatch_environment: Controller allocation identity environment. The
            production default reads the current process environment.
        command_locator: Injectable command availability lookup.
        path_resolver: Injectable strict canonical-path resolver.

    Returns:
        A snapshot suitable for :func:`validate_production_preflight`.

    Raises:
        DiscoveryError: If a path or required repository fact cannot be
            collected exactly within the bounded command contract.
    """
    if not isinstance(task, NormalizedTask):
        raise TypeError("task must be a NormalizedTask")
    if not isinstance(phase, PreflightPhase):
        raise TypeError("phase must be a PreflightPhase")
    if not isinstance(facts, ObservedExecutionFacts):
        raise TypeError("facts must be ObservedExecutionFacts")

    resolver = path_resolver or _strict_resolve
    runner = command_runner or run_read_only_command
    repository_root = _canonical_path(facts.repository_root, "repository root", resolver)
    workspace_root = _canonical_path(facts.workspace_root, "workspace root", resolver)
    container_image = _canonical_path(facts.container_image, "container image", resolver)
    mounts = tuple(
        ObservedMount(
            _canonical_path(mount.host_path, "mount source", resolver),
            mount.container_path,
            mount.read_only,
        )
        for mount in facts.mounts
    )

    available_commands = _find_available_commands(task, phase, command_locator)
    repository_head = _repository_head(repository_root, runner)
    repository_dirty = _repository_dirty(repository_root, runner)
    dispatch = probe_dispatch_capability(
        task,
        phase,
        runner=dispatch_probe_runner,
        environment=dispatch_environment,
    )
    return PreflightSnapshot(
        phase=phase,
        repository_root=repository_root,
        workspace_root=workspace_root,
        container_image=container_image,
        repository_head=repository_head,
        repository_dirty=repository_dirty,
        build_identity=facts.build_identity,
        mounts=mounts,
        available_commands=available_commands,
        dispatch=dispatch,
    )


def run_read_only_command(
    argv: tuple[str, ...],
    *,
    output_limit_bytes: int,
) -> CommandResult:
    """Run an allowlisted read-only Git argv with bounded output and time.

    Args:
        argv: Exact argument vector constructed by this module.
        output_limit_bytes: Combined stdout/stderr byte ceiling.

    Returns:
        Decoded command result.

    Raises:
        DiscoveryError: If the argv is not allowlisted, cannot start, times
            out, or exceeds the output ceiling.
    """
    _validate_read_only_git_argv(argv)
    if isinstance(output_limit_bytes, bool) or not isinstance(output_limit_bytes, int):
        raise TypeError("output_limit_bytes must be an integer")
    if output_limit_bytes <= 0 or output_limit_bytes > _MAX_OUTPUT_BYTES:
        raise ValueError(f"output_limit_bytes must be in [1, {_MAX_OUTPUT_BYTES}]")
    try:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
        )
    except OSError as exc:
        raise DiscoveryError(f"could not start read-only command: {_bounded_error(exc)}") from None

    stdout, stderr = _read_bounded_process(process, output_limit_bytes)
    return CommandResult(
        process.returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )


def _read_bounded_process(
    process: subprocess.Popen[bytes],
    output_limit_bytes: int,
) -> tuple[bytes, bytes]:
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise DiscoveryError("read-only command did not expose captured output")

    selector = selectors.DefaultSelector()
    streams = (("stdout", process.stdout), ("stderr", process.stderr))
    for label, stream in streams:
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, label)
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    deadline = time.monotonic() + _COMMAND_TIMEOUT_SECONDS
    try:
        while selector.get_map():
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                _terminate_process(process)
                raise DiscoveryError("read-only command exceeded the 30 second timeout")
            events = selector.select(remaining_seconds)
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
                    _terminate_process(process)
                    raise DiscoveryError(
                        f"read-only command output exceeded {output_limit_bytes} bytes"
                    )
                captured[key.data].extend(chunk)
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            _terminate_process(process)
            raise DiscoveryError("read-only command exceeded the 30 second timeout")
        try:
            process.wait(timeout=remaining_seconds)
        except subprocess.TimeoutExpired:
            _terminate_process(process)
            raise DiscoveryError("read-only command exceeded the 30 second timeout") from None
    finally:
        selector.close()
        for _, stream in streams:
            if not stream.closed:
                stream.close()
    return bytes(captured["stdout"]), bytes(captured["stderr"])


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait()


def _strict_resolve(path: Path) -> Path:
    return path.resolve(strict=True)


def _canonical_path(path: Path, label: str, resolver: PathResolver) -> Path:
    try:
        canonical = resolver(path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise DiscoveryError(f"could not resolve {label}: {_bounded_error(exc)}") from None
    if not isinstance(canonical, Path):
        raise DiscoveryError(f"resolver returned a non-Path value for {label}")
    if not canonical.is_absolute() or ".." in canonical.parts:
        raise DiscoveryError(f"resolver returned a non-canonical path for {label}")
    return canonical


def _find_available_commands(
    task: NormalizedTask,
    phase: PreflightPhase,
    locator: CommandLocator,
) -> frozenset[str]:
    available: set[str] = set()
    for command in sorted(required_commands(task, phase)):
        try:
            location = locator(command)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DiscoveryError(
                f"could not inspect command {command!r}: {_bounded_error(exc)}"
            ) from None
        if location:
            available.add(command)
    return frozenset(available)


def _repository_head(repository_root: Path, runner: CommandRunner) -> str:
    argv = _git_prefix(repository_root) + ("rev-parse", "--verify", "HEAD")
    result = _checked_command("repository HEAD", argv, runner)
    head = result.stdout.strip()
    if not _EXACT_COMMIT.fullmatch(head):
        raise DiscoveryError("repository HEAD command did not return one exact commit ID")
    return head


def _repository_dirty(repository_root: Path, runner: CommandRunner) -> bool:
    argv = _git_prefix(repository_root) + (
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    result = _checked_command("repository dirty state", argv, runner)
    return bool(result.stdout)


def _git_prefix(repository_root: Path) -> tuple[str, ...]:
    return (
        "git",
        "--no-optional-locks",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(repository_root),
    )


def _checked_command(
    label: str,
    argv: tuple[str, ...],
    runner: CommandRunner,
) -> CommandResult:
    try:
        result = runner(argv, output_limit_bytes=_MAX_OUTPUT_BYTES)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DiscoveryError(f"could not collect {label}: {_bounded_error(exc)}") from None
    if not isinstance(result, CommandResult):
        raise DiscoveryError(f"command runner returned an invalid result for {label}")
    encoded_size = len(result.stdout.encode("utf-8")) + len(result.stderr.encode("utf-8"))
    if encoded_size > _MAX_OUTPUT_BYTES:
        raise DiscoveryError(f"command runner exceeded the output contract for {label}")
    if result.returncode != 0:
        diagnostic = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise DiscoveryError(
            f"could not collect {label} (exit {result.returncode}): {_bounded_text(diagnostic)}"
        )
    return result


def _validate_read_only_git_argv(argv: tuple[str, ...]) -> None:
    if not isinstance(argv, tuple) or not all(isinstance(value, str) for value in argv):
        raise TypeError("argv must be a tuple of strings")
    if len(argv) < 10 or argv[:2] != ("git", "--no-optional-locks"):
        raise DiscoveryError("only fixed read-only Git discovery commands are allowed")
    expected_options = (
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
    )
    if argv[2:7] != expected_options:
        raise DiscoveryError("only fixed read-only Git discovery commands are allowed")
    repository_root = Path(argv[7])
    if not repository_root.is_absolute() or ".." in repository_root.parts:
        raise DiscoveryError("Git discovery repository path must be canonical")
    suffix = argv[8:]
    if suffix not in {
        ("rev-parse", "--verify", "HEAD"),
        ("status", "--porcelain=v1", "--untracked-files=normal"),
    }:
        raise DiscoveryError("only fixed read-only Git discovery commands are allowed")


def _bounded_error(error: BaseException) -> str:
    return _bounded_text(str(error))


def _bounded_text(value: str) -> str:
    normalized = value.replace("\x00", "\\0")
    if len(normalized) <= _MAX_DIAGNOSTIC_CHARACTERS:
        return normalized
    return normalized[:_MAX_DIAGNOSTIC_CHARACTERS] + "...[truncated]"


__all__ = [
    "CommandResult",
    "CommandRunner",
    "DiscoveryError",
    "ObservedExecutionFacts",
    "collect_preflight_snapshot",
    "run_read_only_command",
]
