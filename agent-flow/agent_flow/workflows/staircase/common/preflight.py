# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure production preflight policy for Staircase.

Environment discovery is deliberately outside this module. Callers collect a
non-mutating snapshot at the login or controller boundary and pass it here.
This keeps validation deterministic and makes all failures visible in one
report; the typed discovery probe may contact Slurm but never creates a job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent_flow.workflows.staircase.task_schema import NormalizedTask

from .dispatch_probe import DispatchContract, DispatchPhase, dispatch_contract_matches

PreflightPhase = DispatchPhase


@dataclass(frozen=True, slots=True)
class ObservedMount:
    """One mount observed in the configured controller container launch."""

    host_path: Path
    container_path: Path
    read_only: bool

    def __post_init__(self) -> None:
        if not self.host_path.is_absolute():
            raise ValueError("observed mount host_path must be absolute")
        if not self.container_path.is_absolute() or ".." in self.container_path.parts:
            raise ValueError("observed mount container_path must be absolute without traversal")
        if not isinstance(self.read_only, bool):
            raise TypeError("observed mount read_only must be a bool")


@dataclass(frozen=True, slots=True)
class PreflightSnapshot:
    """Read-only facts collected at one production execution boundary."""

    phase: PreflightPhase
    repository_root: Path
    workspace_root: Path
    container_image: Path
    repository_head: str
    repository_dirty: bool
    build_identity: str
    mounts: tuple[ObservedMount, ...]
    available_commands: frozenset[str]
    dispatch: DispatchContract

    def __post_init__(self) -> None:
        for name in ("repository_root", "workspace_root", "container_image"):
            if not getattr(self, name).is_absolute():
                raise ValueError(f"{name} must be absolute")
        if not self.repository_head.strip():
            raise ValueError("repository_head must not be empty")
        if not isinstance(self.repository_dirty, bool):
            raise TypeError("repository_dirty must be a bool")
        if not self.build_identity.strip():
            raise ValueError("build_identity must not be empty")
        if not all(isinstance(command, str) and command for command in self.available_commands):
            raise ValueError("available_commands must contain non-empty command names")
        if not isinstance(self.dispatch, DispatchContract):
            raise TypeError("dispatch must be backed by a typed probe receipt")


@dataclass(frozen=True, slots=True)
class PreflightIssue:
    """One independently actionable preflight failure."""

    code: str
    phase: PreflightPhase
    message: str


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Typed batched result of one production preflight validation."""

    phase: PreflightPhase
    task_digest: str
    issues: tuple[PreflightIssue, ...]

    @property
    def ok(self) -> bool:
        """Whether every production preflight check passed."""
        return not self.issues

    def require_ok(self) -> None:
        """Raise one error containing every independently discovered failure."""
        if self.ok:
            return
        bullets = "\n".join(f"  - [{issue.code}] {issue.message}" for issue in self.issues)
        raise PreflightError(f"{self.phase.value} Staircase preflight failed:\n{bullets}")


class PreflightError(RuntimeError):
    """Raised when a caller requires a failed preflight report to pass."""


def required_commands(task: NormalizedTask, phase: PreflightPhase) -> frozenset[str]:
    """Return commands required at an execution boundary.

    Args:
        task: Frozen normalized task.
        phase: Login or controller execution boundary.

    Returns:
        Exact command names whose availability must have been observed.
    """
    if phase is PreflightPhase.LOGIN:
        return frozenset({"git", "python3", "sbatch", "squeue", "sacct", "scancel"})
    commands = {"git", "python3", "squeue", "sacct", "scancel"}
    if task.execution.slurm.controller.dispatch_mode == "nested_submission":
        commands.add("sbatch")
    return frozenset(commands)


def validate_production_preflight(
    task: NormalizedTask,
    snapshot: PreflightSnapshot,
) -> PreflightReport:
    """Validate a production environment snapshot without external mutation.

    Args:
        task: Frozen normalized task that will own the run.
        snapshot: Read-only facts collected by the caller at the named phase.

    Returns:
        A deterministic report containing all independently discoverable
        failures. The function does not submit, cancel, or query scheduler jobs.
    """
    issues: list[PreflightIssue] = []

    def add(code: str, message: str) -> None:
        issues.append(PreflightIssue(code, snapshot.phase, message))

    _validate_path_identity(
        "repository root",
        task.repository.root,
        snapshot.repository_root,
        "repository_identity",
        add,
    )
    _validate_path_identity(
        "shared workspace root",
        task.repository.workspace_root,
        snapshot.workspace_root,
        "workspace_identity",
        add,
    )
    _validate_path_identity(
        "controller container image",
        task.execution.slurm.controller.image,
        snapshot.container_image,
        "container_identity",
        add,
    )

    if snapshot.repository_head != task.repository.base_commit:
        add(
            "base_commit",
            "repository HEAD does not match the normalized expected base commit: "
            f"expected {task.repository.base_commit}, observed {snapshot.repository_head}",
        )
    if task.repository.dirty_policy == "reject" and snapshot.repository_dirty:
        add("dirty_repository", "repository has local changes but dirty_policy is 'reject'")
    if snapshot.build_identity != task.execution.slurm.controller.build_identity:
        add(
            "build_identity",
            "runtime build identity does not match the normalized task: "
            f"expected {task.execution.slurm.controller.build_identity!r}, "
            f"observed {snapshot.build_identity!r}",
        )

    _validate_mounts(task, snapshot, add)
    _validate_dispatch(task, snapshot, add)
    _validate_resources(task, add)

    missing_commands = sorted(required_commands(task, snapshot.phase) - snapshot.available_commands)
    for command in missing_commands:
        add(
            "missing_command",
            f"required {snapshot.phase.value}-phase command is unavailable: {command}",
        )

    return PreflightReport(snapshot.phase, task.digest, tuple(issues))


def _validate_path_identity(
    label: str,
    expected: Path,
    observed: Path,
    code: str,
    add: Callable[[str, str], None],
) -> None:
    try:
        canonical_observed = observed.resolve(strict=True)
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        add(code, f"observed {label} does not resolve: {observed} ({exc})")
        return
    if canonical_observed != expected:
        add(
            code,
            f"observed {label} differs from normalized identity: "
            f"expected {expected}, observed {canonical_observed}",
        )


def _validate_mounts(
    task: NormalizedTask,
    snapshot: PreflightSnapshot,
    add: Callable[[str, str], None],
) -> None:
    expected = {
        (mount.host_path, Path(mount.container_path), mount.read_only)
        for mount in task.execution.slurm.controller.mounts
    }
    observed: set[tuple[Path, Path, bool]] = set()
    targets: set[Path] = set()
    for mount in snapshot.mounts:
        try:
            host_path = mount.host_path.resolve(strict=True)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            add(
                "mount_identity",
                f"observed mount source does not resolve: {mount.host_path} ({exc})",
            )
            continue
        identity = (host_path, mount.container_path, mount.read_only)
        if identity in observed:
            add("mount_identity", f"duplicate observed mount: {identity!r}")
        if mount.container_path in targets:
            add(
                "mount_identity",
                f"multiple observed mounts target {mount.container_path}",
            )
        observed.add(identity)
        targets.add(mount.container_path)

    for missing in sorted(expected - observed, key=_mount_sort_key):
        add("mount_identity", f"configured controller mount was not observed: {missing!r}")
    for unexpected in sorted(observed - expected, key=_mount_sort_key):
        add("mount_identity", f"unexpected controller mount was observed: {unexpected!r}")

    writable_required = (task.repository.root, task.repository.workspace_root)
    visible_required = writable_required + (
        task.reference.checkpoint,
        *task.reference.additional_sources,
    )
    for required_path in visible_required:
        covering = [
            identity
            for identity in observed
            if required_path == identity[0] or required_path.is_relative_to(identity[0])
        ]
        if not covering:
            add(
                "mount_visibility",
                f"required host path is not visible in the observed mounts: {required_path}",
            )
        elif required_path in writable_required and all(identity[2] for identity in covering):
            add(
                "mount_writability",
                f"required writable host path is only mounted read-only: {required_path}",
            )


def _validate_dispatch(
    task: NormalizedTask,
    snapshot: PreflightSnapshot,
    add: Callable[[str, str], None],
) -> None:
    expected_mode = task.execution.slurm.controller.dispatch_mode
    if snapshot.dispatch.mode != expected_mode:
        add(
            "dispatch_mode",
            f"dispatch contract mismatch: expected {expected_mode!r}, "
            f"observed {snapshot.dispatch.mode!r}",
        )
    if snapshot.dispatch.phase is not snapshot.phase:
        add(
            "dispatch_phase",
            "dispatch receipt was collected at the wrong execution boundary: "
            f"required {snapshot.phase.value!r}, observed {snapshot.dispatch.phase.value!r}",
        )
    controller = task.execution.slurm.controller
    receipt = snapshot.dispatch.receipt
    if (
        receipt.account != controller.account
        or receipt.partition != controller.partition
        or receipt.qos != controller.qos
        or receipt.reservation != controller.reservation
    ):
        add(
            "dispatch_binding",
            "dispatch receipt is not bound to the normalized Slurm account, partition, "
            "QoS, and reservation",
        )
    if not dispatch_contract_matches(task, snapshot.phase, snapshot.dispatch):
        add(
            "dispatch_receipt",
            "dispatch receipt does not match the exact fixed phase-appropriate probe",
        )
    if not snapshot.dispatch.available:
        add(
            "dispatch_unavailable",
            f"dispatch contract {expected_mode!r} is unavailable: {snapshot.dispatch.evidence}",
        )


def _validate_resources(
    task: NormalizedTask,
    add: Callable[[str, str], None],
) -> None:
    smith = task.execution.slurm.smith
    bounds = smith.per_item_override_bounds
    if smith.max_parallel_items > smith.max_nodes_total:
        add(
            "resource_caps",
            "Smith max_parallel_items exceeds max_nodes_total",
        )
    names: set[str] = set()
    for resource in smith.resource_classes:
        if resource.name in names:
            add("resource_class", f"duplicate Smith resource class: {resource.name}")
        names.add(resource.name)
        violations = (
            (resource.nodes > smith.max_nodes_total, "nodes exceed max_nodes_total"),
            (resource.total_gpus > smith.max_gpus_total, "GPUs exceed max_gpus_total"),
            (resource.nodes > bounds.max_nodes, "nodes exceed per-item bound"),
            (
                resource.tasks_per_node > bounds.max_tasks_per_node,
                "tasks_per_node exceeds per-item bound",
            ),
            (
                resource.gpus_per_node > bounds.max_gpus_per_node,
                "gpus_per_node exceeds per-item bound",
            ),
            (
                resource.cpus_per_task > bounds.max_cpus_per_task,
                "cpus_per_task exceeds per-item bound",
            ),
            (resource.memory_mib > bounds.max_memory_mib, "memory exceeds per-item bound"),
            (
                resource.time_limit_seconds > bounds.max_time_limit_seconds,
                "time limit exceeds per-item bound",
            ),
        )
        for failed, message in violations:
            if failed:
                add(
                    "resource_class",
                    f"resource class {resource.name!r} {message}",
                )
        if resource.name in {"deterministic_gate", "reviewer_rerun"} and (
            resource.total_tasks != task.target.world_size
            or resource.total_gpus != task.target.world_size
        ):
            add(
                "resource_topology",
                f"resource class {resource.name!r} does not request exactly "
                f"target.world_size={task.target.world_size} ranks and GPUs",
            )


def _mount_sort_key(identity: tuple[Path, Path, bool]) -> tuple[str, str, bool]:
    return str(identity[0]), str(identity[1]), identity[2]
