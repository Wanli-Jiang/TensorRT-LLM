# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Typed, shell-free Slurm boundary for the Staircase controller.

Only trusted deterministic controller code may instantiate a real
:class:`SlurmScheduler`. Agent roles receive typed input bundles and never raw
scheduler arguments. The adapter deliberately supports a small structured
surface: extending it requires adding and validating a field rather than
passing through arbitrary ``sbatch`` options.
"""

from __future__ import annotations

import getpass
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePath
from typing import Mapping, Protocol, Sequence

_SCHEDULER_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SUBMISSION_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{7,63}\Z")
_JOB_ID_RE = re.compile(r"(?P<job_id>[1-9][0-9]*)(?:_(?P<array_task_id>[0-9]+))?\Z")
_TIME_LIMIT_RE = re.compile(
    r"(?:(?P<days>[0-9]+)-)?(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9]{2})\Z"
)
_CONTAINER_IMAGE_RE = re.compile(
    r"(?:/[A-Za-z0-9._/@:+#-]{1,511}|[A-Za-z0-9][A-Za-z0-9._/@:+#-]{0,511})\Z"
)
_SENSITIVE_ENV_RE = re.compile(r"(?:AUTH|COOKIE|CREDENTIAL|KEY|PASS|SECRET|TOKEN)", re.IGNORECASE)
_COMMENT_PREFIX = "staircase:"
_INTERNAL_MODULE = "agent_flow.workflows.staircase.internal"
_MAX_RESOURCE_COUNT = 100_000
_MAX_CAPTURED_ERROR_CHARS = 4_000
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_SINGLE_NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_CONTAINER_LAUNCH_MODE = "in_allocation_srun"
AGENT_POLICY_DIGEST_ENVIRONMENT = "STAIRCASE_AGENT_POLICY_DIGEST"

# Pyxis base images can define CUDA and distributed-placement variables even
# for CPU allocations.  CPU role workers fail closed when those variables are
# present, so remove only the fixed placement surface before entering the
# trusted internal entrypoint.  Canonical Slurm task identity variables remain
# available for scheduler ownership checks.
_CPU_CONTAINER_ENVIRONMENT_UNSET = (
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "NVIDIA_VISIBLE_DEVICES",
    "RANK",
    "SLURM_GPUS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_GPUS_PER_TASK",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "WORLD_SIZE",
)

DEFAULT_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "CUDA_HOME",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "LLM_MODELS_ROOT",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED",
        "QWEN3_5_BUILTIN_REFERENCE_DIR",
        "QWEN3_5_HF_REFERENCE_DIR",
        "TLLM_FMHA_LIBS",
        "TLLM_LOG_LEVEL",
        "TLLM_LOG_LEVEL_BY_MODULE",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
    }
)


class SchedulerError(RuntimeError):
    """Base error raised by a scheduler implementation."""


class SlurmCommandError(SchedulerError):
    """Raised when a Slurm command cannot be executed successfully."""


class JobStatus(str, Enum):
    """Normalized scheduler status, independent from domain approval state."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    PREEMPTED = "PREEMPTED"
    NODE_FAIL = "NODE_FAIL"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        """Whether Slurm has reported a terminal state."""
        return self in {
            JobStatus.COMPLETED,
            JobStatus.CANCELLED,
            JobStatus.FAILED,
            JobStatus.PREEMPTED,
            JobStatus.NODE_FAIL,
            JobStatus.OUT_OF_MEMORY,
            JobStatus.TIMEOUT,
        }


class ObservationSource(str, Enum):
    """Scheduler surface from which an observation was obtained."""

    QUEUE = "QUEUE"
    ACCOUNTING = "ACCOUNTING"
    UNKNOWN = "UNKNOWN"


class DependencyType(str, Enum):
    """Closed Slurm dependency kinds accepted by the trusted adapter."""

    AFTERANY = "afterany"


class InternalEntrypoint(str, Enum):
    """Fixed non-public Staircase process entrypoints."""

    CONTROLLER = "controller"
    WORKER = "role-worker"
    RANK_SUPERVISOR = "rank-supervisor"


@dataclass(frozen=True)
class Mount:
    """A validated Pyxis/Enroot bind mount."""

    source: Path
    target: Path
    read_only: bool = True

    def __post_init__(self) -> None:
        _validate_mount_path(self.source, "mount source")
        _validate_mount_path(self.target, "mount target")
        if not isinstance(self.read_only, bool):
            raise TypeError("mount read_only must be a bool")

    def render(self) -> str:
        """Render the mount for the structured ``--container-mounts`` flag."""
        suffix = ":ro" if self.read_only else ""
        return f"{self.source}:{self.target}{suffix}"


@dataclass(frozen=True)
class ResourceRequest:
    """Validated resources accepted by the Staircase Slurm adapter.

    This is an internal scheduler value object, not the user-facing task
    configuration. Task normalization constructs it only after applying
    configured resource-class and aggregate-cap checks.
    """

    label: str
    account: str
    partition: str
    time_limit: str
    output_path: Path
    error_path: Path
    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 1
    gpus_per_node: int = 0
    memory_mb: int | None = None
    qos: str | None = None
    reservation: str | None = None
    signal_seconds: int | None = None
    requeue: bool = False
    exclusive: bool = False
    container_image: str | None = None
    mounts: tuple[Mount, ...] = ()
    container_launch_mode: str = _CONTAINER_LAUNCH_MODE
    agent_policy_digest: str | None = None
    gpu_allocation_padding: bool = False

    def __post_init__(self) -> None:
        _validate_scheduler_name(self.label, "resource label")
        _validate_scheduler_name(self.account, "account")
        _validate_scheduler_name(self.partition, "partition")
        _validate_time_limit(self.time_limit)
        _validate_log_path(self.output_path, "output path")
        _validate_log_path(self.error_path, "error path")
        _validate_positive_int(self.nodes, "nodes")
        _validate_positive_int(self.tasks_per_node, "tasks_per_node")
        _validate_positive_int(self.cpus_per_task, "cpus_per_task")
        _validate_nonnegative_int(self.gpus_per_node, "gpus_per_node")
        if self.memory_mb is not None:
            _validate_positive_int(self.memory_mb, "memory_mb")
        if self.qos is not None:
            _validate_scheduler_name(self.qos, "qos")
        if self.reservation is not None:
            _validate_scheduler_name(self.reservation, "reservation")
        if self.signal_seconds is not None:
            _validate_positive_int(self.signal_seconds, "signal_seconds")
        if not isinstance(self.requeue, bool):
            raise TypeError("requeue must be a bool")
        if not isinstance(self.exclusive, bool):
            raise TypeError("exclusive must be a bool")
        if not isinstance(self.gpu_allocation_padding, bool):
            raise TypeError("gpu_allocation_padding must be a bool")
        if self.gpu_allocation_padding:
            if self.nodes != 1 or self.tasks_per_node != 1:
                raise ValueError("gpu_allocation_padding requires one node and one task")
            if self.gpus_per_node <= 1:
                raise ValueError("gpu_allocation_padding requires more than one allocated GPU")
            if self.container_image is None:
                raise ValueError(
                    "gpu_allocation_padding requires the fixed in-allocation container launch"
                )
        elif self.gpus_per_node > self.tasks_per_node:
            raise ValueError(
                "more allocated GPUs than tasks requires explicit gpu_allocation_padding"
            )
        if self.container_image is not None:
            _validate_container_image(self.container_image)
        if not isinstance(self.mounts, tuple) or not all(
            isinstance(mount, Mount) for mount in self.mounts
        ):
            raise TypeError("mounts must be a tuple of Mount values")
        if self.mounts and self.container_image is None:
            raise ValueError("container mounts require container_image")
        if (
            self.container_image is not None
            and self.container_launch_mode != _CONTAINER_LAUNCH_MODE
        ):
            raise ValueError(
                "containerized jobs require the site-certified in_allocation_srun launch mode"
            )
        if self.agent_policy_digest is not None:
            if _DIGEST_RE.fullmatch(self.agent_policy_digest) is None:
                raise ValueError("agent_policy_digest must be a lowercase SHA-256")
            if self.container_image is None:
                raise ValueError("agent policy binding requires a container image")


@dataclass(frozen=True)
class JobIdentity:
    """Exact Slurm allocation identity, optionally for one array element."""

    job_id: str
    array_task_id: str | None = None
    cluster: str | None = None

    def __post_init__(self) -> None:
        _validate_numeric_identifier(self.job_id, "job_id", positive=True)
        if self.array_task_id is not None:
            _validate_numeric_identifier(self.array_task_id, "array_task_id", positive=False)
        if self.cluster is not None:
            _validate_scheduler_name(self.cluster, "cluster")

    @property
    def scheduler_id(self) -> str:
        """Return the exact ID accepted by ``squeue``, ``sacct`` and ``scancel``."""
        if self.array_task_id is None:
            return self.job_id
        return f"{self.job_id}_{self.array_task_id}"

    @classmethod
    def parse(cls, raw_identity: str, *, cluster: str | None = None) -> JobIdentity:
        """Parse an exact job or array-element identity.

        Ranges, wildcard expressions, step IDs and heterogeneous-job syntax are
        rejected because they are not valid cancellation ownership units.

        Args:
            raw_identity: Slurm job ID such as ``123`` or ``123_4``.
            cluster: Optional validated Slurm cluster name.

        Returns:
            The parsed exact identity.
        """
        match = _JOB_ID_RE.fullmatch(raw_identity)
        if match is None:
            raise ValueError(f"invalid exact Slurm job identity: {raw_identity!r}")
        return cls(
            job_id=match.group("job_id"),
            array_task_id=match.group("array_task_id"),
            cluster=cluster,
        )


@dataclass(frozen=True)
class JobDependency:
    """Typed dependency on one exact Slurm allocation identity."""

    kind: DependencyType
    job: JobIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DependencyType):
            raise TypeError("dependency kind must use DependencyType")
        if not isinstance(self.job, JobIdentity):
            raise TypeError("dependency job must use JobIdentity")
        if self.job.array_task_id is not None:
            raise ValueError("controller dependency requires an exact non-array job")

    def render(self) -> str:
        """Render the validated value accepted by ``sbatch --dependency``."""
        return f"{self.kind.value}:{self.job.scheduler_id}"


@dataclass(frozen=True)
class JobObservation:
    """One reconciled observation of an exact Slurm job."""

    identity: JobIdentity
    status: JobStatus
    reason: str | None
    source: ObservationSource
    raw_state: str | None = None


@dataclass(frozen=True)
class SubmissionLookup:
    """Result of adopting a job by its immutable submission token."""

    submission_token: str
    matches: tuple[JobIdentity, ...]
    status: JobStatus
    reason: str | None
    source: ObservationSource

    @property
    def identity(self) -> JobIdentity | None:
        """Return the unique adoptable identity, or ``None`` when uncertain."""
        if len(self.matches) != 1:
            return None
        return self.matches[0]


@dataclass(frozen=True)
class SubmissionProbe:
    """Strict queue-and-accounting evidence for one immutable submit token."""

    submission_token: str
    matches: tuple[JobIdentity, ...]
    status: JobStatus
    reason: str | None
    source: ObservationSource
    queue_complete: bool
    accounting_complete: bool
    trustworthy: bool

    @property
    def identity(self) -> JobIdentity | None:
        """Return the sole exact identity when the evidence is trustworthy."""
        if not self.trustworthy or len(self.matches) != 1:
            return None
        return self.matches[0]

    @property
    def absence_proven(self) -> bool:
        """Whether both scheduler surfaces cleanly prove zero exact matches."""
        return (
            self.trustworthy
            and self.queue_complete
            and self.accounting_complete
            and not self.matches
        )


@dataclass(frozen=True)
class JobOwnership:
    """Scheduler evidence binding one exact job to a Staircase submission."""

    identity: JobIdentity
    submission_token: str
    username: str | None
    matched: bool
    reason: str
    source: ObservationSource

    def __post_init__(self) -> None:
        _validate_submission_token(self.submission_token)
        if self.username is not None:
            _validate_scheduler_name(self.username, "ownership username")
        if not self.reason.strip():
            raise ValueError("ownership reason must be non-empty")


@dataclass(frozen=True)
class JobPlacementEvidence:
    """Controller-trusted scheduler evidence for one exact single-node job."""

    identity: JobIdentity
    submission_token: str
    username: str | None
    node_count: int | None
    node_list: str | None
    matched: bool
    reason: str
    source: ObservationSource

    def __post_init__(self) -> None:
        if not isinstance(self.identity, JobIdentity):
            raise TypeError("placement evidence requires a typed job identity")
        _validate_submission_token(self.submission_token)
        if self.username is not None:
            _validate_scheduler_name(self.username, "placement username")
        if self.node_count is not None:
            _validate_nonnegative_int(self.node_count, "placement node count")
        if self.node_list is not None and _SINGLE_NODE_RE.fullmatch(self.node_list) is None:
            raise ValueError("placement node_list must be one expanded safe hostname")
        if self.matched and (
            self.username is None or self.node_count != 1 or self.node_list is None
        ):
            raise ValueError("matched placement evidence requires one exact allocated node")
        if not self.reason.strip():
            raise ValueError("placement evidence reason must be non-empty")


@dataclass(frozen=True)
class InternalCommand:
    """Arguments for a fixed controller or worker process invocation."""

    entrypoint: InternalEntrypoint
    workspace: Path | None = None
    generation: int | None = None
    owner_nonce: str | None = None
    input_bundle: Path | None = None

    def __post_init__(self) -> None:
        if self.entrypoint is InternalEntrypoint.CONTROLLER:
            if self.workspace is None:
                raise ValueError("controller command requires a workspace")
            _validate_absolute_path(self.workspace, "workspace")
            if self.generation is None:
                raise ValueError("controller command requires a generation")
            _validate_positive_int(self.generation, "generation")
            if self.owner_nonce is None:
                raise ValueError("controller command requires an owner nonce")
            _validate_safe_token(self.owner_nonce, "owner_nonce")
            if self.input_bundle is not None:
                raise ValueError("controller command cannot have an input bundle")
        elif self.entrypoint in {
            InternalEntrypoint.WORKER,
            InternalEntrypoint.RANK_SUPERVISOR,
        }:
            if self.input_bundle is None:
                raise ValueError("worker command requires an input bundle")
            _validate_absolute_path(self.input_bundle, "input bundle")
            if (
                self.workspace is not None
                or self.generation is not None
                or self.owner_nonce is not None
            ):
                raise ValueError("worker command accepts only an input bundle")
        else:
            raise ValueError(f"unsupported internal entrypoint: {self.entrypoint!r}")

    def argv(self) -> tuple[str, ...]:
        """Return the fixed internal Python invocation."""
        if self.entrypoint is InternalEntrypoint.CONTROLLER:
            return (
                "python3",
                "-m",
                _INTERNAL_MODULE,
                self.entrypoint.value,
                "--workspace",
                str(self.workspace),
                "--generation",
                str(self.generation),
                "--owner-nonce",
                str(self.owner_nonce),
            )
        return (
            "python3",
            "-m",
            _INTERNAL_MODULE,
            self.entrypoint.value,
            "--input",
            str(self.input_bundle),
        )


@dataclass(frozen=True)
class CommandResult:
    """Captured result from a shell-free scheduler subprocess."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandExecutor(Protocol):
    """Minimal subprocess seam used by CPU-only scheduler tests."""

    def run(self, argv: Sequence[str], *, input_text: str | None = None) -> CommandResult:
        """Run one scheduler command without a shell."""


class Scheduler(Protocol):
    """Scheduler operations available to deterministic controller code."""

    def submit(
        self,
        resources: ResourceRequest,
        command: InternalCommand,
        submission_token: str,
        environment: Mapping[str, str] | None = None,
        dependency: JobDependency | None = None,
    ) -> JobIdentity:
        """Submit exactly one attempt and return its scheduler identity."""

    def observe(self, identity: JobIdentity) -> JobObservation:
        """Reconcile queue then accounting state for one exact job."""

    def observe_owned(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobObservation:
        """Observe state only when exact scheduler ownership remains bound."""

    def cancel(self, identity: JobIdentity) -> None:
        """Cancel one exact owned job or array element."""

    def lookup_submission(self, submission_token: str) -> SubmissionLookup:
        """Find a previously submitted attempt by immutable token."""

    def probe_submission(self, submission_token: str) -> SubmissionProbe:
        """Strictly query both queue and accounting for recovery evidence."""

    def verify_ownership(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobOwnership:
        """Verify exact cluster, user, comment, and token ownership."""

    def verify_single_node_placement(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobPlacementEvidence:
        """Verify exact ownership and one expanded allocated scheduler node."""


def render_internal_script(
    command: InternalCommand,
    resources: ResourceRequest | None = None,
    *,
    container_launcher: str = "srun",
) -> str:
    """Render a fixed Bash script for an internal Staircase entrypoint.

    ``shlex.join`` quotes paths as data. The command kind, Python executable,
    module and flags are fixed by :class:`InternalCommand`; callers cannot add
    raw script fragments or scheduler arguments.

    Args:
        command: Validated internal controller or worker command.
        resources: Optional validated resource and container launch contract.
        container_launcher: Resolved host executable used to enter a container.

    Returns:
        A complete script suitable for submission through ``sbatch`` stdin.
    """
    argv = command.argv()
    if resources is not None and resources.container_image is not None:
        if resources.container_launch_mode != _CONTAINER_LAUNCH_MODE:
            raise ValueError("unsupported container launch mode")
        launcher = [
            container_launcher,
            "--nodes=1",
            "--ntasks=1",
            "--overlap",
            "--no-container-mount-home",
        ]
        if resources.gpu_allocation_padding:
            launcher.append("--gpus-per-task=1")
        launcher.append(f"--container-image={resources.container_image}")
        if resources.mounts:
            launcher.append(
                "--container-mounts=" + ",".join(mount.render() for mount in resources.mounts)
            )
        if resources.gpus_per_node == 0:
            environment_cleanup = ["/usr/bin/env"]
            for name in _CPU_CONTAINER_ENVIRONMENT_UNSET:
                environment_cleanup.extend(("-u", name))
            argv = (*launcher, *environment_cleanup, *argv)
        else:
            argv = (*launcher, *argv)
    return "\n".join(
        ("#!/bin/bash", "set -euo pipefail", "umask 077", f"exec {shlex.join(argv)}", "")
    )


def normalize_environment(
    environment: Mapping[str, str] | None,
    *,
    allowlist: frozenset[str] = DEFAULT_ENVIRONMENT_ALLOWLIST,
) -> tuple[tuple[str, str], ...]:
    """Validate and sort the non-secret environment exported to a Slurm job.

    Args:
        environment: Requested environment values, or ``None``.
        allowlist: Explicit names accepted by the trusted adapter.

    Returns:
        A deterministic tuple of name/value pairs.

    Raises:
        ValueError: If a variable is not allowlisted, looks sensitive, or
            cannot be represented by Slurm's comma-delimited export field.
        TypeError: If a key or value is not a string.
    """
    if environment is None:
        return ()
    normalized: list[tuple[str, str]] = []
    for name, value in environment.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("environment names and values must be strings")
        if _SENSITIVE_ENV_RE.search(name):
            raise ValueError(f"credential-like environment variable is prohibited: {name!r}")
        if name not in allowlist:
            raise ValueError(f"environment variable is not allowlisted: {name!r}")
        if any(character in value for character in ("\x00", "\n", "\r", ",")):
            raise ValueError(f"environment variable cannot be safely serialized: {name!r}")
        normalized.append((name, value))
    return tuple(sorted(normalized))


def redact_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return environment data with credential-like values redacted."""
    return {
        name: "<redacted>" if _SENSITIVE_ENV_RE.search(name) else value
        for name, value in environment.items()
    }


class _SubprocessExecutor:
    def __init__(self, timeout_seconds: int) -> None:
        _validate_positive_int(timeout_seconds, "command timeout")
        self._timeout_seconds = timeout_seconds

    def run(self, argv: Sequence[str], *, input_text: str | None = None) -> CommandResult:
        try:
            completed = subprocess.run(
                list(argv),
                input=input_text,
                capture_output=True,
                check=False,
                shell=False,
                text=True,
                timeout=self._timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise SlurmCommandError(f"Slurm command timed out: {_display_argv(argv)}") from error
        except OSError as error:
            raise SlurmCommandError(
                f"could not execute Slurm command {_display_argv(argv)}: {error}"
            ) from error
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


class SlurmScheduler:
    """Safe subprocess-backed implementation of :class:`Scheduler`."""

    def __init__(
        self,
        *,
        executor: CommandExecutor | None = None,
        cluster: str | None = None,
        username: str | None = None,
        command_timeout_seconds: int = 30,
        submission_absence_certified: bool = False,
        container_launcher: str | None = None,
    ) -> None:
        if cluster is not None:
            _validate_scheduler_name(cluster, "cluster")
        resolved_username = username if username is not None else getpass.getuser()
        _validate_scheduler_name(resolved_username, "username")
        if not isinstance(submission_absence_certified, bool):
            raise TypeError("submission_absence_certified must be a bool")
        self._executor = executor or _SubprocessExecutor(command_timeout_seconds)
        self._cluster = cluster
        self._username = resolved_username
        self._submission_absence_certified = submission_absence_certified
        if container_launcher is None and executor is None:
            discovered_launcher = shutil.which("srun")
            if discovered_launcher is None:
                raise ValueError("production Slurm scheduler cannot resolve srun")
            launcher_path = Path(discovered_launcher).resolve(strict=True)
            if not launcher_path.is_file() or not os.access(launcher_path, os.X_OK):
                raise ValueError("production Slurm scheduler resolved a non-executable srun")
            self._container_launcher = launcher_path.as_posix()
        else:
            self._container_launcher = container_launcher or "srun"
            if (
                not isinstance(self._container_launcher, str)
                or not self._container_launcher
                or any(character in self._container_launcher for character in ("\x00", "\n", "\r"))
            ):
                raise ValueError("container launcher must be a non-empty single-line string")

    def submit(
        self,
        resources: ResourceRequest,
        command: InternalCommand,
        submission_token: str,
        environment: Mapping[str, str] | None = None,
        dependency: JobDependency | None = None,
    ) -> JobIdentity:
        """Submit one fixed script using only validated structured options."""
        token = _validate_submission_token(submission_token)
        exported_environment = normalize_environment(environment)
        self._validate_dependency_cluster(dependency)
        argv = self._build_submit_argv(resources, token, exported_environment, dependency)
        result = self._run(
            argv,
            input_text=render_internal_script(
                command,
                resources,
                container_launcher=self._container_launcher,
            ),
        )
        output = result.stdout.strip()
        if not output or "\n" in output:
            raise SlurmCommandError(f"unexpected sbatch --parsable output: {output!r}")
        raw_identity, separator, returned_cluster = output.partition(";")
        cluster = returned_cluster if separator else self._cluster
        if returned_cluster and self._cluster is not None and returned_cluster != self._cluster:
            raise SlurmCommandError(
                f"sbatch returned cluster {returned_cluster!r}, expected {self._cluster!r}"
            )
        try:
            return JobIdentity.parse(raw_identity, cluster=cluster)
        except ValueError as error:
            raise SlurmCommandError(f"unexpected sbatch --parsable output: {output!r}") from error

    def observe(self, identity: JobIdentity) -> JobObservation:
        """Query ``squeue`` first and fall back to ``sacct`` after queue exit."""
        queue_argv = [
            "squeue",
            "--noheader",
            f"--jobs={identity.scheduler_id}",
            "--format=%i|%T|%R|%k",
        ]
        self._append_cluster(queue_argv, identity.cluster)
        queue_records = _parse_observation_records(
            self._run(queue_argv).stdout,
            identity,
            ObservationSource.QUEUE,
        )
        if queue_records:
            return _select_observation(identity, queue_records, ObservationSource.QUEUE)

        accounting_argv = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--jobs={identity.scheduler_id}",
            "--format=JobIDRaw,State,Reason,Comment",
        ]
        self._append_cluster(accounting_argv, identity.cluster)
        accounting_records = _parse_observation_records(
            self._run(accounting_argv).stdout,
            identity,
            ObservationSource.ACCOUNTING,
        )
        if accounting_records:
            return _select_observation(identity, accounting_records, ObservationSource.ACCOUNTING)
        return JobObservation(
            identity=identity,
            status=JobStatus.UNKNOWN,
            reason="job is absent from squeue and may not yet be visible in sacct",
            source=ObservationSource.UNKNOWN,
        )

    def observe_owned(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobObservation:
        """Observe one known job through the strict token-bound evidence path."""
        token = _validate_submission_token(submission_token)
        if identity.cluster != self._cluster:
            return JobObservation(
                identity,
                JobStatus.UNKNOWN,
                "exact job cluster does not match the scheduler submission cluster",
                ObservationSource.UNKNOWN,
            )
        probe = self.probe_submission(token)
        if not probe.trustworthy:
            return JobObservation(
                identity,
                JobStatus.UNKNOWN,
                probe.reason or "owned scheduler evidence is untrustworthy",
                ObservationSource.UNKNOWN,
            )
        if probe.identity != identity:
            reason = (
                "owned scheduler evidence is absent"
                if not probe.matches
                else "owned scheduler evidence is ambiguous or names another job"
            )
            return JobObservation(
                identity,
                JobStatus.UNKNOWN,
                reason,
                ObservationSource.UNKNOWN,
            )
        return JobObservation(
            identity,
            probe.status,
            probe.reason,
            probe.source,
            raw_state=probe.status.value,
        )

    def cancel(self, identity: JobIdentity) -> None:
        """Cancel only the supplied exact job or array-element identity."""
        argv = ["scancel"]
        self._append_cluster(argv, identity.cluster)
        argv.append(identity.scheduler_id)
        self._run(argv)

    def lookup_submission(self, submission_token: str) -> SubmissionLookup:
        """Look up a crash-interrupted submission by immutable token.

        A missing accounting record or multiple matches is deliberately
        ``UNKNOWN``. The controller must wait/reconcile rather than submitting
        a replacement that could duplicate live work.
        """
        token = _validate_submission_token(submission_token)
        job_name = _submission_job_name(token)
        queue_argv = [
            "squeue",
            "--noheader",
            f"--user={self._username}",
            f"--name={job_name}",
            "--format=%i|%T|%R|%k",
        ]
        self._append_cluster(queue_argv, self._cluster)
        queue_matches = _parse_lookup_records(
            self._run(queue_argv).stdout,
            token,
            self._cluster,
        )
        if queue_matches:
            return _select_lookup(token, queue_matches, ObservationSource.QUEUE)

        accounting_argv = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--name={job_name}",
            f"--user={self._username}",
            "--format=JobIDRaw,State,Reason,Comment",
        ]
        self._append_cluster(accounting_argv, self._cluster)
        accounting_matches = _parse_lookup_records(
            self._run(accounting_argv).stdout,
            token,
            self._cluster,
        )
        if accounting_matches:
            return _select_lookup(token, accounting_matches, ObservationSource.ACCOUNTING)
        return SubmissionLookup(
            submission_token=token,
            matches=(),
            status=JobStatus.UNKNOWN,
            reason="submission is absent from squeue and may not yet be visible in sacct",
            source=ObservationSource.UNKNOWN,
        )

    def probe_submission(self, submission_token: str) -> SubmissionProbe:
        """Query both scheduler surfaces without discarding malformed evidence."""
        token = _validate_submission_token(submission_token)
        job_name = _submission_job_name(token)
        queue_argv = [
            "squeue",
            "--noheader",
            f"--user={self._username}",
            f"--name={job_name}",
            "--format=%i|%T|%R|%k|%u|%j",
        ]
        self._append_cluster(queue_argv, self._cluster)
        queue_records, queue_error = _parse_probe_records(
            self._run(queue_argv).stdout,
            token,
            self._cluster,
            "queue",
            expected_username=self._username,
            expected_job_name=job_name,
        )
        accounting_argv = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--name={job_name}",
            f"--user={self._username}",
            "--format=JobIDRaw,State,Reason,Comment,User,JobName",
        ]
        self._append_cluster(accounting_argv, self._cluster)
        accounting_records, accounting_error = _parse_probe_records(
            self._run(accounting_argv).stdout,
            token,
            self._cluster,
            "accounting",
            expected_username=self._username,
            expected_job_name=job_name,
        )
        probe = _merge_submission_probe(
            token,
            queue_records,
            accounting_records,
            queue_error=queue_error,
            accounting_error=accounting_error,
        )
        if probe.absence_proven and not self._submission_absence_certified:
            return SubmissionProbe(
                submission_token=token,
                matches=(),
                status=JobStatus.UNKNOWN,
                reason=(
                    "Slurm accounting retention is not certified for same-token absence proof; "
                    "manual recovery is required"
                ),
                source=ObservationSource.UNKNOWN,
                queue_complete=True,
                accounting_complete=True,
                trustworthy=False,
            )
        return probe

    def verify_ownership(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobOwnership:
        """Verify exact job ownership from queue, then accounting metadata."""
        token = _validate_submission_token(submission_token)
        if identity.cluster != self._cluster:
            return JobOwnership(
                identity,
                token,
                None,
                False,
                "exact job cluster does not match the scheduler submission cluster",
                ObservationSource.UNKNOWN,
            )
        queue_argv = [
            "squeue",
            "--noheader",
            f"--jobs={identity.scheduler_id}",
            "--format=%i|%u|%k",
        ]
        self._append_cluster(queue_argv, identity.cluster)
        queue_records = _parse_ownership_records(
            self._run(queue_argv).stdout,
            identity,
            token,
            self._username,
        )
        if queue_records:
            return _select_ownership(identity, token, queue_records, ObservationSource.QUEUE)

        accounting_argv = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--jobs={identity.scheduler_id}",
            "--format=JobIDRaw,User,Comment",
        ]
        self._append_cluster(accounting_argv, identity.cluster)
        accounting_records = _parse_ownership_records(
            self._run(accounting_argv).stdout,
            identity,
            token,
            self._username,
        )
        if accounting_records:
            return _select_ownership(
                identity,
                token,
                accounting_records,
                ObservationSource.ACCOUNTING,
            )
        return JobOwnership(
            identity,
            token,
            None,
            False,
            "exact job ownership is absent from queue and accounting",
            ObservationSource.UNKNOWN,
        )

    def verify_single_node_placement(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobPlacementEvidence:
        """Verify exact job ownership and allocated NodeList from Slurm."""
        token = _validate_submission_token(submission_token)
        if identity.cluster != self._cluster:
            return JobPlacementEvidence(
                identity,
                token,
                None,
                None,
                None,
                False,
                "exact job cluster does not match the scheduler submission cluster",
                ObservationSource.UNKNOWN,
            )
        queue_argv = [
            "squeue",
            "--noheader",
            f"--jobs={identity.scheduler_id}",
            "--format=%i|%u|%k|%D|%N",
        ]
        self._append_cluster(queue_argv, identity.cluster)
        queue = _parse_placement_evidence(
            self._run(queue_argv).stdout,
            identity,
            token,
            self._username,
            ObservationSource.QUEUE,
        )
        if queue is not None:
            return queue

        accounting_argv = [
            "sacct",
            "--noheader",
            "--parsable2",
            "--allocations",
            f"--jobs={identity.scheduler_id}",
            "--format=JobIDRaw,User,Comment,NNodes,NodeList",
        ]
        self._append_cluster(accounting_argv, identity.cluster)
        accounting = _parse_placement_evidence(
            self._run(accounting_argv).stdout,
            identity,
            token,
            self._username,
            ObservationSource.ACCOUNTING,
        )
        if accounting is not None:
            return accounting
        return JobPlacementEvidence(
            identity,
            token,
            None,
            None,
            None,
            False,
            "exact placement is absent from queue and accounting",
            ObservationSource.UNKNOWN,
        )

    def _build_submit_argv(
        self,
        resources: ResourceRequest,
        submission_token: str,
        environment: tuple[tuple[str, str], ...],
        dependency: JobDependency | None,
    ) -> list[str]:
        comment = f"{_COMMENT_PREFIX}{submission_token};label={resources.label}"
        argv = [
            "sbatch",
            "--parsable",
            f"--job-name={_submission_job_name(submission_token)}",
            f"--comment={comment}",
            f"--account={resources.account}",
            f"--partition={resources.partition}",
            f"--time={resources.time_limit}",
            f"--nodes={resources.nodes}",
            f"--ntasks-per-node={resources.tasks_per_node}",
            f"--cpus-per-task={resources.cpus_per_task}",
            f"--gpus-per-node={resources.gpus_per_node}",
            f"--output={resources.output_path}",
            f"--error={resources.error_path}",
            "--requeue" if resources.requeue else "--no-requeue",
        ]
        if resources.memory_mb is not None:
            argv.append(f"--mem={resources.memory_mb}M")
        if resources.qos is not None:
            argv.append(f"--qos={resources.qos}")
        if resources.reservation is not None:
            argv.append(f"--reservation={resources.reservation}")
        if resources.signal_seconds is not None:
            argv.append(f"--signal=B:USR1@{resources.signal_seconds}")
        if resources.exclusive:
            argv.append("--exclusive")
        if dependency is not None:
            argv.append(f"--dependency={dependency.render()}")
        effective_environment = list(environment)
        if resources.agent_policy_digest is not None:
            if any(name == AGENT_POLICY_DIGEST_ENVIRONMENT for name, _value in environment):
                raise SchedulerError("agent policy binding environment is controller-owned")
            effective_environment.append(
                (AGENT_POLICY_DIGEST_ENVIRONMENT, resources.agent_policy_digest)
            )
        if effective_environment:
            exports = ",".join(f"{name}={value}" for name, value in sorted(effective_environment))
            argv.append(f"--export={exports}")
        else:
            argv.append("--export=NIL")
        self._append_cluster(argv, self._cluster)
        return argv

    def _validate_dependency_cluster(self, dependency: JobDependency | None) -> None:
        if dependency is not None and dependency.job.cluster != self._cluster:
            raise SchedulerError("dependency job cluster must exactly match the submission cluster")

    def _append_cluster(self, argv: list[str], cluster: str | None) -> None:
        if cluster is not None:
            argv.append(f"--clusters={cluster}")

    def _run(self, argv: Sequence[str], *, input_text: str | None = None) -> CommandResult:
        result = self._executor.run(tuple(argv), input_text=input_text)
        if result.returncode != 0:
            detail = (result.stderr.strip() or result.stdout.strip())[:_MAX_CAPTURED_ERROR_CHARS]
            raise SlurmCommandError(
                f"Slurm command failed ({result.returncode}): {_display_argv(argv)}: {detail}"
            )
        return result


@dataclass(frozen=True)
class FakeSubmission:
    """Immutable submission captured by :class:`FakeScheduler`."""

    identity: JobIdentity
    resources: ResourceRequest
    command: InternalCommand
    submission_token: str
    environment: tuple[tuple[str, str], ...]
    dependency: JobDependency | None


@dataclass
class _FakeRecord:
    submission: FakeSubmission
    status: JobStatus
    reason: str | None
    source: ObservationSource
    visible: bool
    observed_identity: JobIdentity
    username: str
    comment: str
    job_name: str


class FakeScheduler:
    """Deterministic in-memory scheduler for controller and CPU tests."""

    def __init__(
        self,
        *,
        first_job_id: int = 10_000,
        cluster: str | None = None,
        username: str = "fake-user",
    ) -> None:
        _validate_positive_int(first_job_id, "first_job_id")
        if cluster is not None:
            _validate_scheduler_name(cluster, "cluster")
        _validate_scheduler_name(username, "username")
        self._next_job_id = first_job_id
        self._cluster = cluster
        self._username = username
        self._records: dict[str, _FakeRecord] = {}
        self._tokens: dict[str, list[str]] = {}
        self._cancelled: list[JobIdentity] = []
        self._placements: dict[str, JobPlacementEvidence] = {}

    @property
    def submissions(self) -> tuple[FakeSubmission, ...]:
        """Return submissions in deterministic job-ID order."""
        return tuple(record.submission for record in self._records.values())

    @property
    def cancelled(self) -> tuple[JobIdentity, ...]:
        """Return exact identities passed to :meth:`cancel`."""
        return tuple(self._cancelled)

    def submit(
        self,
        resources: ResourceRequest,
        command: InternalCommand,
        submission_token: str,
        environment: Mapping[str, str] | None = None,
        dependency: JobDependency | None = None,
    ) -> JobIdentity:
        """Capture a submission and initially expose it as queued/pending."""
        token = _validate_submission_token(submission_token)
        normalized_environment = normalize_environment(environment)
        if dependency is not None and dependency.job.cluster != self._cluster:
            raise SchedulerError("dependency job cluster must exactly match the submission cluster")
        identity = JobIdentity(str(self._next_job_id), cluster=self._cluster)
        self._next_job_id += 1
        submission = FakeSubmission(
            identity=identity,
            resources=resources,
            command=command,
            submission_token=token,
            environment=normalized_environment,
            dependency=dependency,
        )
        self._records[identity.scheduler_id] = _FakeRecord(
            submission=submission,
            status=JobStatus.PENDING,
            reason=None,
            source=ObservationSource.QUEUE,
            visible=True,
            observed_identity=identity,
            username=self._username,
            comment=f"{_COMMENT_PREFIX}{token};label={resources.label}",
            job_name=_submission_job_name(token),
        )
        self._tokens.setdefault(token, []).append(identity.scheduler_id)
        return identity

    def observe(self, identity: JobIdentity) -> JobObservation:
        """Return the configured observation or an accounting-lag ``UNKNOWN``."""
        record = self._records.get(identity.scheduler_id)
        if record is None or not record.visible:
            return JobObservation(
                identity=identity,
                status=JobStatus.UNKNOWN,
                reason="job is not visible in the fake scheduler",
                source=ObservationSource.UNKNOWN,
            )
        return JobObservation(
            identity=identity,
            status=record.status,
            reason=record.reason,
            source=record.source,
            raw_state=record.status.value,
        )

    def observe_owned(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobObservation:
        """Observe a fake job only after exact identity and token validation."""
        token = _validate_submission_token(submission_token)
        ownership = self.verify_ownership(identity, token)
        if not ownership.matched:
            return JobObservation(
                identity,
                JobStatus.UNKNOWN,
                ownership.reason,
                ObservationSource.UNKNOWN,
            )
        return self.observe(identity)

    def cancel(self, identity: JobIdentity) -> None:
        """Cancel one exact fake job identity."""
        record = self._records.get(identity.scheduler_id)
        if record is None:
            raise SchedulerError(f"unknown fake job: {identity.scheduler_id}")
        record.status = JobStatus.CANCELLED
        record.reason = "cancelled by controller"
        record.source = ObservationSource.ACCOUNTING
        record.visible = True
        self._cancelled.append(identity)

    def lookup_submission(self, submission_token: str) -> SubmissionLookup:
        """Find visible fake jobs by their exact immutable token."""
        token = _validate_submission_token(submission_token)
        record_ids = self._tokens.get(token, [])
        queue_records = [
            self._records[record_id]
            for record_id in record_ids
            if self._records[record_id].visible
            and self._records[record_id].source is ObservationSource.QUEUE
        ]
        if queue_records:
            return _select_fake_lookup(token, queue_records, ObservationSource.QUEUE)
        accounting_records = [
            self._records[record_id]
            for record_id in record_ids
            if self._records[record_id].visible
            and self._records[record_id].source is ObservationSource.ACCOUNTING
        ]
        if accounting_records:
            return _select_fake_lookup(token, accounting_records, ObservationSource.ACCOUNTING)
        return SubmissionLookup(
            submission_token=token,
            matches=(),
            status=JobStatus.UNKNOWN,
            reason="submission is not visible in the fake scheduler",
            source=ObservationSource.UNKNOWN,
        )

    def probe_submission(self, submission_token: str) -> SubmissionProbe:
        """Return complete deterministic queue-and-accounting fake evidence."""
        lookup = self.lookup_submission(submission_token)
        return SubmissionProbe(
            submission_token=lookup.submission_token,
            matches=lookup.matches,
            status=lookup.status,
            reason=lookup.reason,
            source=lookup.source,
            queue_complete=True,
            accounting_complete=True,
            trustworthy=True,
        )

    def verify_ownership(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobOwnership:
        """Verify an exact fake job and immutable token."""
        token = _validate_submission_token(submission_token)
        record = self._records.get(identity.scheduler_id)
        if record is None or not record.visible or record.observed_identity != identity:
            return JobOwnership(
                identity,
                token,
                None,
                False,
                "exact fake job is not visible",
                ObservationSource.UNKNOWN,
            )
        observed_token = _submission_token_from_comment(record.comment)
        expected_job_name = _submission_job_name(token)
        matched = (
            record.username == self._username
            and observed_token == token
            and record.job_name == expected_job_name
        )
        if record.username != self._username:
            reason = "exact fake job scheduler user does not match"
        elif observed_token != token:
            reason = "exact fake job submission token does not match"
        elif record.job_name != expected_job_name:
            reason = "exact fake job name does not match"
        else:
            reason = "exact fake job user/name/comment/token ownership matches"
        return JobOwnership(
            identity,
            token,
            record.username,
            matched,
            reason,
            record.source,
        )

    def set_ownership_evidence(
        self,
        identity: JobIdentity,
        *,
        observed_identity: JobIdentity,
        username: str,
        comment: str,
        job_name: str,
    ) -> None:
        """Replace exact fake scheduler metadata for adversarial ownership tests."""
        record = self._records.get(identity.scheduler_id)
        if record is None:
            raise SchedulerError(f"unknown fake job: {identity.scheduler_id}")
        record.observed_identity = observed_identity
        record.username = username
        record.comment = comment
        record.job_name = job_name

    def set_placement(
        self,
        identity: JobIdentity,
        submission_token: str,
        node_list: str,
        *,
        source: ObservationSource = ObservationSource.QUEUE,
    ) -> None:
        """Install exact trusted placement evidence for a controller test."""
        token = _validate_submission_token(submission_token)
        if identity.cluster != self._cluster:
            raise SchedulerError("fake placement cluster does not match the scheduler cluster")
        self._placements[identity.scheduler_id] = JobPlacementEvidence(
            identity,
            token,
            self._username,
            1,
            node_list,
            True,
            "exact fake scheduler placement matches",
            source,
        )

    def verify_single_node_placement(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobPlacementEvidence:
        """Return independently seeded exact fake scheduler placement."""
        token = _validate_submission_token(submission_token)
        if identity.cluster != self._cluster:
            return JobPlacementEvidence(
                identity,
                token,
                None,
                None,
                None,
                False,
                "fake placement cluster does not match the scheduler cluster",
                ObservationSource.UNKNOWN,
            )
        evidence = self._placements.get(identity.scheduler_id)
        if evidence is None or evidence.identity != identity:
            return JobPlacementEvidence(
                identity,
                token,
                None,
                None,
                None,
                False,
                "exact fake scheduler placement is unavailable",
                ObservationSource.UNKNOWN,
            )
        if evidence.submission_token != token:
            return JobPlacementEvidence(
                identity,
                token,
                evidence.username,
                evidence.node_count,
                evidence.node_list,
                False,
                "exact fake scheduler placement submission token does not match",
                evidence.source,
            )
        return evidence

    def transition(
        self,
        identity: JobIdentity,
        status: JobStatus,
        *,
        reason: str | None = None,
        source: ObservationSource = ObservationSource.QUEUE,
        visible: bool = True,
    ) -> None:
        """Set a deterministic observation for one fake job."""
        record = self._records.get(identity.scheduler_id)
        if record is None:
            raise SchedulerError(f"unknown fake job: {identity.scheduler_id}")
        record.status = status
        record.reason = reason
        record.source = source
        record.visible = visible


def _select_fake_lookup(
    token: str,
    records: list[_FakeRecord],
    source: ObservationSource,
) -> SubmissionLookup:
    matches = tuple(record.submission.identity for record in records)
    if len(records) != 1:
        return SubmissionLookup(
            submission_token=token,
            matches=matches,
            status=JobStatus.UNKNOWN,
            reason=f"ambiguous submission token: {len(records)} scheduler jobs match",
            source=source,
        )
    record = records[0]
    return SubmissionLookup(token, matches, record.status, record.reason, source)


def _parse_observation_records(
    output: str,
    expected_identity: JobIdentity,
    source: ObservationSource,
) -> list[JobObservation]:
    records: list[JobObservation] = []
    for line in output.splitlines():
        fields = line.strip().split("|", maxsplit=3)
        if len(fields) != 4:
            continue
        raw_identity, raw_state, raw_reason, _comment = fields
        if raw_identity != expected_identity.scheduler_id:
            continue
        records.append(
            JobObservation(
                identity=expected_identity,
                status=_normalize_job_status(raw_state),
                reason=_normalize_optional_field(raw_reason),
                source=source,
                raw_state=raw_state,
            )
        )
    return records


def _select_observation(
    identity: JobIdentity,
    records: list[JobObservation],
    source: ObservationSource,
) -> JobObservation:
    if len(records) == 1:
        return records[0]
    return JobObservation(
        identity=identity,
        status=JobStatus.UNKNOWN,
        reason=f"ambiguous scheduler response: {len(records)} records for exact job ID",
        source=source,
    )


def _parse_lookup_records(
    output: str,
    submission_token: str,
    cluster: str | None,
) -> list[tuple[JobIdentity, JobStatus, str | None]]:
    records: list[tuple[JobIdentity, JobStatus, str | None]] = []
    for line in output.splitlines():
        fields = line.strip().split("|", maxsplit=3)
        if len(fields) != 4:
            continue
        raw_identity, raw_state, raw_reason, comment = fields
        if _submission_token_from_comment(comment) != submission_token:
            continue
        try:
            identity = JobIdentity.parse(raw_identity, cluster=cluster)
        except ValueError:
            continue
        records.append(
            (identity, _normalize_job_status(raw_state), _normalize_optional_field(raw_reason))
        )
    return records


def _parse_probe_records(
    output: str,
    submission_token: str,
    cluster: str | None,
    surface: str,
    *,
    expected_username: str,
    expected_job_name: str,
) -> tuple[list[tuple[JobIdentity, JobStatus, str | None]], str | None]:
    records: list[tuple[JobIdentity, JobStatus, str | None]] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split("|", maxsplit=5)
        if len(fields) != 6:
            return records, f"{surface} row {line_number} is malformed"
        raw_identity, raw_state, raw_reason, comment, username, job_name = fields
        if username != expected_username:
            return records, f"{surface} row {line_number} has a mismatched scheduler owner"
        if job_name != expected_job_name:
            return records, f"{surface} row {line_number} has a mismatched scheduler job name"
        if _submission_token_from_comment(comment) != submission_token:
            return records, f"{surface} row {line_number} has a mismatched submission token"
        try:
            identity = JobIdentity.parse(raw_identity, cluster=cluster)
        except ValueError:
            return records, f"{surface} row {line_number} has an invalid exact job identity"
        records.append(
            (identity, _normalize_job_status(raw_state), _normalize_optional_field(raw_reason))
        )
    return records, None


def _merge_submission_probe(
    submission_token: str,
    queue_records: list[tuple[JobIdentity, JobStatus, str | None]],
    accounting_records: list[tuple[JobIdentity, JobStatus, str | None]],
    *,
    queue_error: str | None,
    accounting_error: str | None,
) -> SubmissionProbe:
    by_identity: dict[JobIdentity, tuple[JobStatus, str | None, ObservationSource]] = {}
    for identity, status, reason in accounting_records:
        by_identity[identity] = (status, reason, ObservationSource.ACCOUNTING)
    for identity, status, reason in queue_records:
        by_identity[identity] = (status, reason, ObservationSource.QUEUE)
    matches = tuple(sorted(by_identity, key=lambda identity: identity.scheduler_id))
    evidence_error = queue_error or accounting_error
    if evidence_error is not None:
        return SubmissionProbe(
            submission_token,
            matches,
            JobStatus.UNKNOWN,
            evidence_error,
            ObservationSource.UNKNOWN,
            queue_error is None,
            accounting_error is None,
            False,
        )
    if len(matches) != 1:
        reason = (
            "submission is absent from successful queue and accounting queries"
            if not matches
            else f"ambiguous submission token: {len(matches)} scheduler jobs match"
        )
        return SubmissionProbe(
            submission_token,
            matches,
            JobStatus.UNKNOWN,
            reason,
            ObservationSource.UNKNOWN,
            True,
            True,
            True,
        )
    status, reason, source = by_identity[matches[0]]
    return SubmissionProbe(
        submission_token,
        matches,
        status,
        reason,
        source,
        True,
        True,
        True,
    )


def _select_lookup(
    submission_token: str,
    records: list[tuple[JobIdentity, JobStatus, str | None]],
    source: ObservationSource,
) -> SubmissionLookup:
    matches = tuple(record[0] for record in records)
    if len(records) != 1:
        return SubmissionLookup(
            submission_token=submission_token,
            matches=matches,
            status=JobStatus.UNKNOWN,
            reason=f"ambiguous submission token: {len(records)} scheduler jobs match",
            source=source,
        )
    identity, status, reason = records[0]
    return SubmissionLookup(submission_token, (identity,), status, reason, source)


def _parse_ownership_records(
    output: str,
    expected_identity: JobIdentity,
    submission_token: str,
    expected_username: str,
) -> list[tuple[str, bool, str]]:
    records: list[tuple[str, bool, str]] = []
    for line in output.splitlines():
        fields = line.strip().split("|", maxsplit=2)
        if len(fields) != 3:
            continue
        raw_identity, username, comment = fields
        if raw_identity != expected_identity.scheduler_id:
            continue
        observed_token = _submission_token_from_comment(comment)
        matched = username == expected_username and observed_token == submission_token
        if username != expected_username:
            reason = (
                f"scheduler owner {username!r} does not match expected user {expected_username!r}"
            )
        elif observed_token != submission_token:
            reason = "scheduler comment does not contain the expected submission token"
        else:
            reason = "exact cluster, user, scheduler comment, and submission token match"
        records.append((username, matched, reason))
    return records


def _select_ownership(
    identity: JobIdentity,
    submission_token: str,
    records: list[tuple[str, bool, str]],
    source: ObservationSource,
) -> JobOwnership:
    if len(records) != 1:
        return JobOwnership(
            identity,
            submission_token,
            None,
            False,
            f"ownership response is ambiguous across {len(records)} exact records",
            source,
        )
    username, matched, reason = records[0]
    return JobOwnership(
        identity,
        submission_token,
        username,
        matched,
        reason,
        source,
    )


def _parse_placement_evidence(
    output: str,
    expected_identity: JobIdentity,
    submission_token: str,
    expected_username: str,
    source: ObservationSource,
) -> JobPlacementEvidence | None:
    lines = tuple(line.strip() for line in output.splitlines() if line.strip())
    if not lines:
        return None

    def unmatched(reason: str) -> JobPlacementEvidence:
        return JobPlacementEvidence(
            expected_identity,
            submission_token,
            None,
            None,
            None,
            False,
            reason,
            source,
        )

    if len(lines) != 1:
        return unmatched(f"placement response is ambiguous across {len(lines)} records")
    fields = lines[0].split("|", maxsplit=4)
    if len(fields) != 5:
        return unmatched("placement response is malformed")
    raw_identity, username, comment, raw_node_count, node_list = fields
    if raw_identity != expected_identity.scheduler_id:
        return unmatched("placement response does not contain the exact job identity")
    if _SCHEDULER_NAME_RE.fullmatch(username) is None:
        return unmatched("placement response contains an invalid scheduler owner")
    if username != expected_username:
        return JobPlacementEvidence(
            expected_identity,
            submission_token,
            username,
            None,
            None,
            False,
            "scheduler placement owner does not match the expected user",
            source,
        )
    if _submission_token_from_comment(comment) != submission_token:
        return JobPlacementEvidence(
            expected_identity,
            submission_token,
            username,
            None,
            None,
            False,
            "scheduler placement comment does not contain the expected submission token",
            source,
        )
    if raw_node_count != "1":
        node_count = (
            int(raw_node_count) if raw_node_count.isascii() and raw_node_count.isdigit() else None
        )
        if node_count is not None and node_count > _MAX_RESOURCE_COUNT:
            node_count = None
        return JobPlacementEvidence(
            expected_identity,
            submission_token,
            username,
            node_count,
            None,
            False,
            "scheduler placement is not one canonical allocated node",
            source,
        )
    if _SINGLE_NODE_RE.fullmatch(node_list) is None:
        return JobPlacementEvidence(
            expected_identity,
            submission_token,
            username,
            1,
            None,
            False,
            "scheduler placement NodeList is compressed, ambiguous, or unsafe",
            source,
        )
    return JobPlacementEvidence(
        expected_identity,
        submission_token,
        username,
        1,
        node_list,
        True,
        "exact cluster, job, user, comment, token, and single-node allocation match",
        source,
    )


def _normalize_job_status(raw_state: str) -> JobStatus:
    state = raw_state.strip().upper().split(maxsplit=1)[0].rstrip("+")
    if state in {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD"}:
        return JobStatus.PENDING
    if state in {"RUNNING", "COMPLETING", "RESIZING", "SIGNALING", "STAGE_OUT", "SUSPENDED"}:
        return JobStatus.RUNNING
    if state == "COMPLETED":
        return JobStatus.COMPLETED
    if state == "CANCELLED":
        return JobStatus.CANCELLED
    if state in {"FAILED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}:
        return JobStatus.FAILED
    if state == "PREEMPTED":
        return JobStatus.PREEMPTED
    if state == "NODE_FAIL":
        return JobStatus.NODE_FAIL
    if state == "OUT_OF_MEMORY":
        return JobStatus.OUT_OF_MEMORY
    if state == "TIMEOUT":
        return JobStatus.TIMEOUT
    return JobStatus.UNKNOWN


def _normalize_optional_field(value: str) -> str | None:
    normalized = value.strip()
    if normalized in {"", "(null)", "None", "Unknown"}:
        return None
    return normalized


def _submission_job_name(submission_token: str) -> str:
    return f"staircase-{submission_token}"


def _submission_token_from_comment(comment: str) -> str | None:
    normalized = comment.strip()
    if not normalized.startswith(_COMMENT_PREFIX):
        return None
    token_and_metadata = normalized.removeprefix(_COMMENT_PREFIX)
    token, _separator, _metadata = token_and_metadata.partition(";")
    try:
        return _validate_submission_token(token)
    except ValueError:
        return None


def _validate_submission_token(submission_token: str) -> str:
    _validate_safe_token(submission_token, "submission_token")
    return submission_token


def _validate_safe_token(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not _SUBMISSION_TOKEN_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be 8-64 safe ASCII characters starting with alphanumeric"
        )


def _validate_scheduler_name(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not _SCHEDULER_NAME_RE.fullmatch(value):
        raise ValueError(f"{field_name} contains unsupported characters")


def _validate_numeric_identifier(value: str, field_name: str, *, positive: bool) -> None:
    if not isinstance(value, str) or not value.isascii() or not value.isdigit():
        raise ValueError(f"{field_name} must contain only ASCII decimal digits")
    if positive and int(value) <= 0:
        raise ValueError(f"{field_name} must be positive")


def _validate_time_limit(time_limit: str) -> None:
    if not isinstance(time_limit, str):
        raise TypeError("time_limit must be a string")
    match = _TIME_LIMIT_RE.fullmatch(time_limit)
    if match is None:
        raise ValueError("time_limit must be normalized as [days-]HH:MM:SS")
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError("time_limit minutes and seconds must be less than 60")
    if all(int(match.group(field) or 0) == 0 for field in ("days", "hours", "minutes", "seconds")):
        raise ValueError("time_limit must be greater than zero")


def _validate_container_image(container_image: str) -> None:
    if not isinstance(container_image, str):
        raise TypeError("container_image must be a string")
    if not _CONTAINER_IMAGE_RE.fullmatch(container_image):
        raise ValueError("container_image contains unsupported characters")
    if container_image.startswith("/") and ".." in PurePath(container_image).parts:
        raise ValueError("container_image cannot contain '..'")


def _validate_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0 or value > _MAX_RESOURCE_COUNT:
        raise ValueError(f"{field_name} must be in [1, {_MAX_RESOURCE_COUNT}]")


def _validate_nonnegative_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 0 or value > _MAX_RESOURCE_COUNT:
        raise ValueError(f"{field_name} must be in [0, {_MAX_RESOURCE_COUNT}]")


def _validate_log_path(path: Path, field_name: str) -> None:
    _validate_absolute_path(path, field_name)
    value = str(path)
    if any(character in value for character in (",", "\r", "\n", "\x00")):
        raise ValueError(f"{field_name} cannot be serialized safely")
    for match in re.finditer("%", value):
        if match.start() + 1 >= len(value) or value[match.start() + 1] not in "AaJjNntux":
            raise ValueError(f"{field_name} contains an unsupported Slurm replacement")


def _validate_mount_path(path: Path, field_name: str) -> None:
    _validate_absolute_path(path, field_name)
    if any(character in str(path) for character in (":", ",")):
        raise ValueError(f"{field_name} cannot contain ':' or ','")


def _validate_absolute_path(path: Path, field_name: str) -> None:
    if not isinstance(path, Path):
        raise TypeError(f"{field_name} must be a pathlib.Path")
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    if any(part == ".." for part in PurePath(path).parts):
        raise ValueError(f"{field_name} cannot contain '..'")
    if any(character in str(path) for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{field_name} contains a control character")


def _display_argv(argv: Sequence[str]) -> str:
    redacted = ["--export=<redacted>" if item.startswith("--export=") else item for item in argv]
    return shlex.join(redacted)


__all__ = [
    "AGENT_POLICY_DIGEST_ENVIRONMENT",
    "CommandExecutor",
    "CommandResult",
    "DEFAULT_ENVIRONMENT_ALLOWLIST",
    "DependencyType",
    "FakeScheduler",
    "FakeSubmission",
    "InternalCommand",
    "InternalEntrypoint",
    "JobDependency",
    "JobIdentity",
    "JobObservation",
    "JobPlacementEvidence",
    "JobStatus",
    "Mount",
    "ObservationSource",
    "ResourceRequest",
    "Scheduler",
    "SchedulerError",
    "SlurmCommandError",
    "SlurmScheduler",
    "SubmissionLookup",
    "SubmissionProbe",
    "normalize_environment",
    "redact_environment",
    "render_internal_script",
]
