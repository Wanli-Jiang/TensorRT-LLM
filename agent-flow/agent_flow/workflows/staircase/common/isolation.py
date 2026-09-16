# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed container, filesystem, and environment isolation for workers.

Controller mounts describe where host paths are visible in the controller
container.  They are deliberately not reused for role workers.  This module
uses them only as a path-translation table, then emits exact worker mounts for
one metadata-free code surface, one attempt mailbox, and the configured
reference inputs.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence

from .credentials import (
    BACKEND_CREDENTIAL_ALLOWLIST,
    CREDENTIAL_MOUNT_PATH,
    CREDENTIAL_ROOT_NAME,
    CredentialDescriptor,
    CredentialError,
    CredentialProvision,
    scan_stream_for_secret_material,
)
from .runners import BackendKind, RoleProcessSpec, validate_role_spec
from .slurm import DEFAULT_ENVIRONMENT_ALLOWLIST, InternalCommand, InternalEntrypoint, Mount

_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SENSITIVE_ENVIRONMENT = re.compile(
    r"(?:AUTH|COOKIE|CREDENTIAL|KEY|PASS|SECRET|TOKEN)", re.IGNORECASE
)
_SCHEDULER_ENVIRONMENT_PREFIXES = (
    "MUNGE_",
    "PMI_",
    "PMIX_",
    "SALLOC_",
    "SBATCH_",
    "SLURM_",
    "SRUN_",
)
_SCHEDULER_ENVIRONMENT_NAMES = frozenset({"KRB5CCNAME", "SSH_AUTH_SOCK"})

WORKER_TASK_ENVIRONMENT_ALLOWLIST = DEFAULT_ENVIRONMENT_ALLOWLIST


class IsolationPolicyError(ValueError):
    """Raised when a worker launch cannot be isolated unambiguously."""


def digest_metadata_free_tree(
    root: Path,
    *,
    secret_scan_paths: Sequence[str] = (),
    credential_values: Mapping[str, str] | None = None,
    ignore_root_git_file: bool = False,
) -> str:
    """Digest one stable overlay and optionally ignore a linked-worktree Git file."""
    canonical = _canonical_existing_directory(root, "candidate overlay")
    if not ignore_root_git_file:
        _validate_metadata_free_candidate(canonical)
    elif not (canonical / ".git").is_file() or (canonical / ".git").is_symlink():
        raise IsolationPolicyError("controller checkout must have regular root Git metadata")
    scan_paths = frozenset(_candidate_relative_path(value) for value in secret_scan_paths)
    paths = _candidate_tree_paths(canonical, ignore_root_git_file=ignore_root_git_file)
    root_identity = _stat_identity(canonical.lstat())
    identities: dict[str, tuple[int, ...]] = {}
    scanned_paths: set[str] = set()
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(canonical).as_posix()
        try:
            metadata = path.lstat()
        except OSError as error:
            raise IsolationPolicyError(
                f"candidate overlay changed while inspecting {relative}"
            ) from error
        identities[relative] = _stat_identity(metadata)
        if stat.S_ISLNK(metadata.st_mode):
            raise IsolationPolicyError(f"candidate overlay contains a symlink: {relative}")
        mode = metadata.st_mode & 0o777
        if stat.S_ISDIR(metadata.st_mode):
            digest.update(f"d\0{relative}\0{mode:o}\0".encode("utf-8"))
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise IsolationPolicyError(
                f"candidate overlay contains a non-regular entry: {relative}"
            )
        if metadata.st_nlink != 1:
            raise IsolationPolicyError(
                f"candidate overlay file must have exactly one hard link: {relative}"
            )
        digest.update(f"f\0{relative}\0{mode:o}\0{metadata.st_size}\0".encode("utf-8"))
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise IsolationPolicyError(
                f"candidate overlay changed while opening {relative}"
            ) from error
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if _stat_identity(opened) != identities[relative] or not stat.S_ISREG(opened.st_mode):
                raise IsolationPolicyError(f"candidate overlay changed while opening {relative}")
            if relative in scan_paths:
                try:
                    size = scan_stream_for_secret_material(
                        stream,
                        credential_values or {},
                        label=f"candidate file {relative!r}",
                        consume=digest.update,
                    )
                except CredentialError as error:
                    raise IsolationPolicyError(str(error)) from error
                scanned_paths.add(relative)
            else:
                size = 0
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    digest.update(block)
            if (
                size != opened.st_size
                or _stat_identity(os.fstat(stream.fileno())) != identities[relative]
            ):
                raise IsolationPolicyError(f"candidate overlay changed while reading {relative}")
        digest.update(b"\0")
    existing_unscanned = scan_paths.difference(scanned_paths)
    if any((canonical / relative).exists() for relative in existing_unscanned):
        raise IsolationPolicyError("candidate secret scan path is not a regular file")
    if (
        _candidate_tree_paths(canonical, ignore_root_git_file=ignore_root_git_file) != paths
        or _stat_identity(canonical.lstat()) != root_identity
    ):
        raise IsolationPolicyError("candidate overlay changed while computing its digest")
    for relative, identity in identities.items():
        path = canonical / relative
        try:
            observed = _stat_identity(path.lstat())
        except OSError as error:
            raise IsolationPolicyError(
                "candidate overlay changed while computing its digest"
            ) from error
        if observed != identity:
            raise IsolationPolicyError("candidate overlay changed while computing its digest")
    return digest.hexdigest()


def _candidate_relative_path(value: str) -> str:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise IsolationPolicyError("candidate secret scan path must be repository-relative")
    return value


def _candidate_tree_paths(root: Path, *, ignore_root_git_file: bool) -> tuple[Path, ...]:
    try:
        paths = tuple(
            path
            for path in root.rglob("*")
            if not (ignore_root_git_file and path.relative_to(root).parts[0] == ".git")
        )
    except OSError as error:
        raise IsolationPolicyError(
            "candidate overlay changed while enumerating its entries"
        ) from error
    for path in paths:
        if ".git" in path.relative_to(root).parts:
            raise IsolationPolicyError("candidate overlay cannot expose Git metadata")
    return tuple(sorted(paths, key=lambda entry: entry.relative_to(root).as_posix()))


def _stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


@dataclass(frozen=True, slots=True)
class WorkerEnvironment:
    """Explicit non-secret environment safe for scheduler export."""

    values: tuple[tuple[str, str], ...]
    credential_names: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        names = [name for name, _value in self.values]
        if names != sorted(names) or len(names) != len(set(names)):
            raise IsolationPolicyError("worker environment must have sorted unique names")
        if not self.credential_names.issubset(names):
            raise IsolationPolicyError("credential names must identify exported variables")
        if self.credential_names:
            raise IsolationPolicyError("credentials cannot be exported through the scheduler")
        for name, value in self.values:
            _validate_environment_entry(name, value)
            if _SENSITIVE_ENVIRONMENT.search(name):
                raise IsolationPolicyError(
                    f"credential-like worker environment is prohibited: {name!r}"
                )

    @property
    def mapping(self) -> Mapping[str, str]:
        """Return a scheduler-ready copy of the explicit environment."""
        return dict(self.values)

    @property
    def redacted(self) -> Mapping[str, str]:
        """Return a diagnostic copy that never exposes credential values."""
        return {
            name: "<redacted>" if name in self.credential_names else value
            for name, value in self.values
        }


@dataclass(frozen=True, slots=True)
class ContainerPathTranslator:
    """Canonical bidirectional translation through controller-visible mounts."""

    controller_mounts: tuple[Mount, ...]

    @classmethod
    def from_mounts(cls, mounts: Sequence[Mount]) -> ContainerPathTranslator:
        """Canonicalize a configured controller mount table.

        Duplicate host sources or container targets are ambiguous and rejected.
        Nested mounts are supported; the most-specific covering mount wins in
        the same way that the nested mount shadows its parent at runtime.
        """
        canonical: list[Mount] = []
        sources: set[Path] = set()
        targets: set[Path] = set()
        for mount in mounts:
            source = _canonical_existing_path(mount.source, "controller mount source")
            target = _absolute_normalized_path(mount.target, "controller mount target")
            if source in sources:
                raise IsolationPolicyError(f"ambiguous duplicate controller source {source}")
            if target in targets:
                raise IsolationPolicyError(f"ambiguous duplicate controller target {target}")
            sources.add(source)
            targets.add(target)
            canonical.append(Mount(source, target, mount.read_only))
        if not canonical:
            raise IsolationPolicyError("worker path translation requires controller mounts")
        return cls(tuple(canonical))

    def to_container(self, host_path: Path, *, require_exists: bool = True) -> Path:
        """Translate one host path to its unambiguous container-visible identity."""
        host = _canonical_path(host_path, "host path", require_exists=require_exists)
        source, target = self._most_specific(host, host_side=True)
        return target / host.relative_to(source)

    def to_host(self, container_path: Path, *, require_exists: bool = True) -> Path:
        """Translate one container-visible path back to its canonical host identity."""
        container = _absolute_normalized_path(container_path, "container path")
        target, source = self._most_specific(container, host_side=False)
        host = source / container.relative_to(target)
        return _canonical_path(host, "translated host path", require_exists=require_exists)

    def _most_specific(self, path: Path, *, host_side: bool) -> tuple[Path, Path]:
        candidates: list[tuple[Path, Path]] = []
        for mount in self.controller_mounts:
            source = mount.source if host_side else mount.target
            target = mount.target if host_side else mount.source
            if _contains(source, path):
                candidates.append((source, target))
        if not candidates:
            side = "host" if host_side else "container"
            raise IsolationPolicyError(f"{side} path is not covered by a configured mount: {path}")
        maximum = max(len(source.parts) for source, _target in candidates)
        most_specific = [pair for pair in candidates if len(pair[0].parts) == maximum]
        if len(most_specific) != 1:
            raise IsolationPolicyError(f"ambiguous configured mount translation for {path}")
        return most_specific[0]


@dataclass(frozen=True, slots=True)
class WorkerIsolation:
    """Exact mounts, paths, and environment for one worker allocation."""

    translator: ContainerPathTranslator
    mounts: tuple[Mount, ...]
    environment: WorkerEnvironment
    workspace: Path
    repository: Path
    mailbox: Path
    input_bundle: Path
    checkpoint: Path
    reference_sources: tuple[Path, ...]
    # Retain the public field name while existing persisted action schemas are
    # migrated. The value is a metadata-free overlay, never a Git worktree.
    worktree: Path | None = None
    credential_descriptor: CredentialDescriptor | None = None

    def container_path(self, host_path: Path, *, require_exists: bool = True) -> Path:
        """Translate a path and require it to be inside an exact worker mount."""
        host = _canonical_path(host_path, "worker path", require_exists=require_exists)
        covering = [mount for mount in self.mounts if _contains(mount.source, host)]
        if not covering:
            raise IsolationPolicyError(f"path is outside the exact worker mounts: {host}")
        maximum = max(len(mount.source.parts) for mount in covering)
        exact = [mount for mount in covering if len(mount.source.parts) == maximum]
        if len(exact) != 1:
            raise IsolationPolicyError(f"ambiguous exact worker mount for {host}")
        mount = exact[0]
        return mount.target / host.relative_to(mount.source)

    def worker_command(self, input_bundle: Path) -> InternalCommand:
        """Build a worker command using the exact read-only input mount."""
        host_input = _canonical_existing_path(input_bundle, "worker input bundle")
        if host_input != self.input_bundle:
            raise IsolationPolicyError("worker input differs from its immutable input mount")
        return InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=self.container_path(host_input),
        )

    def to_worker_role_spec(self, host_spec: RoleProcessSpec) -> RoleProcessSpec:
        """Translate a host-validated role spec for execution in the container."""
        checked = validate_role_spec(host_spec)
        source_root = Path(checked.source_root)
        cwd = Path(checked.cwd)
        result_path = _canonical_path(
            Path(checked.result_path), "role result path", require_exists=False
        )
        allowed_roots = {self.worktree} if self.worktree is not None else {self.repository}
        if source_root not in allowed_roots or cwd not in allowed_roots:
            raise IsolationPolicyError("role source_root and cwd must name an exact code mount")
        if not _contains(self.mailbox, result_path):
            raise IsolationPolicyError("role result path must be inside its mailbox")
        return replace(
            checked,
            source_root=str(self.container_path(source_root)),
            cwd=str(self.container_path(cwd)),
            result_path=str(self.container_path(result_path, require_exists=False)),
        )

    def to_host_role_spec(self, worker_spec: RoleProcessSpec) -> RoleProcessSpec:
        """Translate a persisted container role spec for controller validation."""
        translated = replace(
            worker_spec,
            source_root=str(self.translator.to_host(Path(worker_spec.source_root))),
            cwd=str(self.translator.to_host(Path(worker_spec.cwd))),
            result_path=str(
                self.translator.to_host(Path(worker_spec.result_path), require_exists=False)
            ),
        )
        checked = validate_role_spec(translated)
        if self.to_worker_role_spec(checked) != worker_spec:
            raise IsolationPolicyError("worker role spec does not round-trip through exact mounts")
        return checked


def preserve_controller_mounts(mounts: Sequence[Mount]) -> tuple[Mount, ...]:
    """Return canonical controller mounts without narrowing or changing access."""
    return ContainerPathTranslator.from_mounts(mounts).controller_mounts


def build_agent_worker_environment(
    *,
    backend_kind: BackendKind,
    task_environment: Mapping[str, str] | None,
    ambient_environment: Mapping[str, str] | None,
) -> WorkerEnvironment:
    """Build a scheduler-safe agent environment from task values only.

    Ambient values, including allowlisted credentials, are never copied into
    scheduler export.  The argument remains explicit so a caller cannot assume
    ambient inheritance; credentials use :mod:`.credentials` file transport.
    Task values are rejected, rather than ignored, when they are secret-like,
    scheduler-owned, or outside the non-secret allowlist.
    """
    if backend_kind not in BACKEND_CREDENTIAL_ALLOWLIST:
        raise IsolationPolicyError(f"unsupported agent backend {backend_kind!r}")
    values: dict[str, str] = {}
    for name, value in (task_environment or {}).items():
        _validate_environment_entry(name, value)
        if _is_scheduler_environment(name):
            raise IsolationPolicyError(f"scheduler environment is controller-only: {name!r}")
        if _SENSITIVE_ENVIRONMENT.search(name):
            raise IsolationPolicyError(f"credential-like task environment is prohibited: {name!r}")
        if name not in WORKER_TASK_ENVIRONMENT_ALLOWLIST:
            raise IsolationPolicyError(f"task environment is not worker-allowlisted: {name!r}")
        values[name] = value

    # Deliberately do not inspect or copy ambient values here.  The controller
    # may pass them only to prepare_credential_provision().
    _ = ambient_environment
    return WorkerEnvironment(tuple(sorted(values.items())))


def build_gate_worker_environment(environment: Mapping[str, str]) -> WorkerEnvironment:
    """Validate the complete and exact environment of a deterministic gate."""
    values: list[tuple[str, str]] = []
    for name, value in environment.items():
        _validate_environment_entry(name, value)
        if _is_scheduler_environment(name):
            raise IsolationPolicyError(f"scheduler environment is controller-only: {name!r}")
        if _SENSITIVE_ENVIRONMENT.search(name):
            raise IsolationPolicyError(f"deterministic gate cannot receive credentials: {name!r}")
        values.append((name, value))
    return WorkerEnvironment(tuple(sorted(values)))


def build_worker_isolation(
    *,
    controller_mounts: Sequence[Mount],
    workspace: Path,
    repository: Path,
    mailbox: Path,
    input_bundle: Path,
    checkpoint: Path,
    reference_sources: Sequence[Path] = (),
    worktree: Path | None = None,
    worktree_writable: bool = False,
    environment: WorkerEnvironment | None = None,
    input_bundle_prepublication: bool = False,
) -> WorkerIsolation:
    """Build exact worker mounts without exposing controller-owned directories."""
    translator = ContainerPathTranslator.from_mounts(controller_mounts)
    canonical_workspace = _canonical_existing_directory(workspace, "workspace")
    canonical_repository = _canonical_existing_directory(repository, "repository")
    canonical_mailbox = _canonical_existing_directory(mailbox, "mailbox")
    canonical_input = _canonical_path(
        input_bundle,
        "worker input bundle",
        require_exists=not input_bundle_prepublication,
    )
    canonical_checkpoint = _canonical_existing_path(checkpoint, "checkpoint")
    canonical_references = tuple(
        _canonical_existing_path(source, "reference source") for source in reference_sources
    )

    if _overlaps(canonical_workspace, canonical_repository):
        raise IsolationPolicyError("workspace and repository must not overlap")
    _validate_worker_io_topology(
        canonical_workspace,
        canonical_mailbox,
        canonical_input,
        input_bundle_prepublication=input_bundle_prepublication,
    )
    for source in (canonical_checkpoint, *canonical_references):
        if _overlaps(source, canonical_workspace):
            raise IsolationPolicyError("a reference source cannot overlap controller state")

    canonical_worktree: Path | None = None
    if worktree is not None:
        canonical_worktree = _canonical_existing_directory(worktree, "candidate overlay")
        _validate_worktree_topology(canonical_workspace, canonical_worktree)
        _validate_metadata_free_candidate(canonical_worktree)
        if _overlaps(canonical_worktree, canonical_mailbox):
            raise IsolationPolicyError("candidate worktree and mailbox must not overlap")
    elif worktree_writable:
        raise IsolationPolicyError("worktree_writable requires a candidate worktree")

    requested: list[tuple[Path, bool]] = []
    if canonical_worktree is not None:
        requested.append((canonical_worktree, not worktree_writable))
    else:
        requested.append((canonical_repository, True))
    requested.extend(
        [
            (canonical_mailbox, False),
            (canonical_input, True),
            (canonical_checkpoint, True),
            *((source, True) for source in canonical_references),
        ]
    )
    exact_mounts = _exact_mounts(
        translator,
        requested,
        allow_missing_sources=(
            frozenset({canonical_input}) if input_bundle_prepublication else frozenset()
        ),
    )
    if canonical_worktree is None:
        metadata_mask = _prepare_git_metadata_mask(canonical_workspace)
        metadata_target = translator.to_container(canonical_repository) / ".git"
        if any(mount.target == metadata_target for mount in exact_mounts):
            raise IsolationPolicyError("Git metadata mask target collides with a worker mount")
        exact_mounts = (*exact_mounts, Mount(metadata_mask, metadata_target, True))
    return WorkerIsolation(
        translator=translator,
        mounts=exact_mounts,
        environment=environment or WorkerEnvironment(()),
        workspace=canonical_workspace,
        repository=canonical_repository,
        worktree=canonical_worktree,
        mailbox=canonical_mailbox,
        input_bundle=canonical_input,
        checkpoint=canonical_checkpoint,
        reference_sources=canonical_references,
    )


def attach_agent_credential_provision(
    isolation: WorkerIsolation,
    provision: CredentialProvision,
) -> WorkerIsolation:
    """Attach zero or one fixed-path secret mount to an agent isolation plan.

    Deterministic gates must use :func:`build_worker_isolation` directly and
    therefore never receive this mount.
    """
    if isolation.credential_descriptor is not None:
        raise IsolationPolicyError("agent isolation already has a credential descriptor")
    if provision.descriptor.state.value == "none":
        if provision.mounts:
            raise IsolationPolicyError("no-credential provision unexpectedly has mounts")
        return replace(isolation, credential_descriptor=provision.descriptor)
    if len(provision.mounts) != 1:
        raise IsolationPolicyError("credential provision must contain exactly one mount")
    mount = provision.mounts[0]
    if not mount.read_only or mount.target != CREDENTIAL_MOUNT_PATH:
        raise IsolationPolicyError("credential bundle must use the fixed read-only mount")
    source = _canonical_existing_path(mount.source, "credential bundle")
    secret_root = isolation.workspace / CREDENTIAL_ROOT_NAME
    if not _contains(secret_root, source):
        raise IsolationPolicyError("credential bundle must be under controller-owned secrets")
    protected = (
        isolation.mailbox,
        isolation.input_bundle,
        isolation.repository,
        isolation.checkpoint,
        *isolation.reference_sources,
    )
    if isolation.worktree is not None:
        protected = (*protected, isolation.worktree)
    if any(_overlaps(source, path) for path in protected):
        raise IsolationPolicyError("credential bundle must not overlap a worker data mount")
    if any(existing.target == CREDENTIAL_MOUNT_PATH for existing in isolation.mounts):
        raise IsolationPolicyError("fixed credential mount target is already occupied")
    return replace(
        isolation,
        mounts=(*isolation.mounts, Mount(source, CREDENTIAL_MOUNT_PATH, True)),
        credential_descriptor=provision.descriptor,
    )


def _exact_mounts(
    translator: ContainerPathTranslator,
    requested: Sequence[tuple[Path, bool]],
    *,
    allow_missing_sources: frozenset[Path] = frozenset(),
) -> tuple[Mount, ...]:
    mounts: list[Mount] = []
    by_source: dict[Path, Mount] = {}
    targets: set[Path] = set()
    for source, read_only in requested:
        target = translator.to_container(
            source,
            require_exists=source not in allow_missing_sources,
        )
        mount = Mount(source, target, read_only)
        previous = by_source.get(source)
        if previous is not None:
            if previous != mount:
                raise IsolationPolicyError(
                    f"one worker source requested with conflicting access: {source}"
                )
            continue
        if target in targets:
            raise IsolationPolicyError(f"worker mount targets collide at {target}")
        by_source[source] = mount
        targets.add(target)
        mounts.append(mount)
    return tuple(mounts)


def _validate_worker_io_topology(
    workspace: Path,
    mailbox: Path,
    input_bundle: Path,
    *,
    input_bundle_prepublication: bool,
) -> None:
    try:
        relative = mailbox.relative_to(workspace)
    except ValueError as error:
        raise IsolationPolicyError("mailbox must be inside its workspace") from error
    if (
        len(relative.parts) != 5
        or relative.parts[0] != "items"
        or relative.parts[2] != "attempts"
        or relative.parts[4] != "output"
    ):
        raise IsolationPolicyError(
            "mailbox must be one exact workspace/items/<item>/attempts/<attempt>/output directory"
        )
    valid_location = input_bundle.parent == mailbox.parent and input_bundle.name in {
        "input.json",
        "role-input.json",
    }
    valid_published_file = (
        not input_bundle_prepublication
        and input_bundle.is_file()
        and input_bundle.stat(follow_symlinks=False).st_nlink == 1
    )
    valid_prepublication = input_bundle_prepublication and not input_bundle.exists()
    if not valid_location or not (valid_published_file or valid_prepublication):
        raise IsolationPolicyError(
            "worker input must be one exact regular sibling of its output mailbox"
        )


def _validate_worktree_topology(workspace: Path, worktree: Path) -> None:
    try:
        relative = worktree.relative_to(workspace / "candidates")
    except ValueError as error:
        raise IsolationPolicyError(
            "candidate overlay must be inside its workspace candidates directory"
        ) from error
    if len(relative.parts) != 2:
        raise IsolationPolicyError(
            "candidate overlay must be one exact candidates/<item>/<attempt> directory"
        )


def _validate_metadata_free_candidate(candidate: Path) -> None:
    for path in candidate.rglob("*"):
        relative = path.relative_to(candidate)
        if ".git" in relative.parts:
            raise IsolationPolicyError("candidate overlay cannot expose Git metadata")
        if path.is_symlink() and not path.resolve(strict=False).is_relative_to(candidate):
            raise IsolationPolicyError("candidate overlay symlink escapes its root")
        if path.is_symlink() and ".git" in path.resolve(strict=False).relative_to(candidate).parts:
            raise IsolationPolicyError(
                "candidate overlay symlink would expose Git metadata after import"
            )


def _prepare_git_metadata_mask(workspace: Path) -> Path:
    """Create the empty read-only surface that shadows a source checkout's .git."""
    surfaces = workspace / "worker-surfaces"
    surfaces.mkdir(mode=0o700, exist_ok=True)
    canonical_surfaces = _canonical_existing_directory(surfaces, "worker surface directory")
    mask = canonical_surfaces / "empty-git"
    mask.mkdir(mode=0o700, exist_ok=True)
    canonical_mask = _canonical_existing_directory(mask, "Git metadata mask")
    if any(canonical_mask.iterdir()):
        raise IsolationPolicyError("Git metadata mask must remain empty")
    return canonical_mask


def _validate_environment_entry(name: str, value: str) -> None:
    if not isinstance(name, str) or not isinstance(value, str):
        raise IsolationPolicyError("environment names and values must be strings")
    if _ENVIRONMENT_NAME.fullmatch(name) is None:
        raise IsolationPolicyError(f"unsafe environment name {name!r}")
    if not value or any(character in value for character in ("\x00", "\n", "\r", ",")):
        raise IsolationPolicyError(f"environment value for {name!r} cannot be exported safely")


def _is_scheduler_environment(name: str) -> bool:
    return name in _SCHEDULER_ENVIRONMENT_NAMES or name.startswith(_SCHEDULER_ENVIRONMENT_PREFIXES)


def _canonical_existing_directory(path: Path, name: str) -> Path:
    canonical = _canonical_existing_path(path, name)
    if not canonical.is_dir():
        raise IsolationPolicyError(f"{name} must be a directory: {canonical}")
    return canonical


def _canonical_existing_path(path: Path, name: str) -> Path:
    return _canonical_path(path, name, require_exists=True)


def _canonical_path(path: Path, name: str, *, require_exists: bool) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise IsolationPolicyError(f"{name} must be an absolute pathlib.Path")
    if path.is_symlink():
        raise IsolationPolicyError(f"{name} cannot be a symlink: {path}")
    try:
        canonical = path.resolve(strict=require_exists)
    except (OSError, RuntimeError) as error:
        raise IsolationPolicyError(f"{name} does not resolve: {path} ({error})") from error
    if require_exists and not canonical.exists():
        raise IsolationPolicyError(f"{name} does not exist: {canonical}")
    if canonical != path:
        raise IsolationPolicyError(f"{name} must already be canonical: {path}")
    return canonical


def _absolute_normalized_path(path: Path, name: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise IsolationPolicyError(f"{name} must be an absolute normalized pathlib.Path")
    return path


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or child.is_relative_to(parent)


def _overlaps(first: Path, second: Path) -> bool:
    return _contains(first, second) or _contains(second, first)


__all__ = [
    "BACKEND_CREDENTIAL_ALLOWLIST",
    "WORKER_TASK_ENVIRONMENT_ALLOWLIST",
    "ContainerPathTranslator",
    "IsolationPolicyError",
    "WorkerEnvironment",
    "WorkerIsolation",
    "attach_agent_credential_provision",
    "build_agent_worker_environment",
    "build_gate_worker_environment",
    "build_worker_isolation",
    "digest_metadata_free_tree",
    "preserve_controller_mounts",
]
