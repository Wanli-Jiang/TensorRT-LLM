# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller-owned Git operations for Staircase candidate integration.

The public :class:`ControllerGitOps` object exposes read-only inspection at
all times. Mutating operations are available only through a transaction that
holds an exclusive, controller-identity-bearing filesystem lock. This keeps
parallel role workers away from shared Git metadata and makes controller
fan-in serial by construction.

Every Git invocation uses an argument vector and an explicit ``-C`` path.
This module deliberately has no reset, clean, bulk-stage, hook-bypass,
three-way merge, push, or branch-deletion primitive.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import TracebackType

from .isolation import IsolationPolicyError, digest_metadata_free_tree


class GitOpsError(RuntimeError):
    """Raised when a Git operation or invariant fails."""


class GitConflictError(GitOpsError):
    """Raised when an explicit candidate integration cannot be applied."""


class ControllerLockError(GitOpsError):
    """Raised when a mutation is attempted without the controller lock."""


@dataclass(frozen=True)
class RepositoryInspection:
    """Immutable view of a repository's current checkout state."""

    root: Path
    head: str
    branch: str | None
    changed_paths: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether tracked, staged, and untracked path sets are empty."""
        return not self.changed_paths


@dataclass(frozen=True)
class CandidateVerification:
    """Content identity established for a one-commit candidate."""

    base_commit: str
    candidate_commit: str
    tree_hash: str
    patch_sha256: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True)
class IntegratedPatchExpectation:
    """Exact content expected for one commit in a recovered fan-in sequence."""

    patch_sha256: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True)
class IntegratedSequenceVerification:
    """Content identity for an exact linear controller fan-in sequence."""

    previous_head: str
    final_head: str
    final_tree_hash: str
    commits: tuple[str, ...]
    evidence_sha256: str


@dataclass(frozen=True)
class OverlayBinding:
    """Controller-owned Git binding for one metadata-free worker overlay."""

    overlay: Path
    repository: Path
    base_commit: str
    candidate_commit: str
    changed_paths: tuple[str, ...]


@dataclass(frozen=True)
class DeliveryCheckout:
    """Isolated controller integration checkout for signed-commit delivery."""

    repository: Path
    branch: str | None
    head: str
    git: ControllerGitOps


@dataclass(frozen=True)
class DeliveryPatch:
    """Content-addressed review artifact for diff-only delivery."""

    path: Path
    base_commit: str
    head_commit: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _OverlayReceipt:
    """Write-once identity for an overlay staged before its candidate commit."""

    base_commit: str
    branch: str
    changed_paths: tuple[str, ...]
    overlay_sha256: str
    tree_hash: str
    receipt_sha256: str


class ControllerFileLock:
    """Exclusive filesystem lock carrying the active controller identity."""

    def __init__(self, path: str | Path, owner: str) -> None:
        if not owner or "\n" in owner:
            raise ValueError("controller lock owner must be a non-empty single line")
        self.path = Path(path).expanduser().resolve()
        self.owner = owner
        self._file_descriptor: int | None = None

    @property
    def held(self) -> bool:
        """Whether this exact lock instance is held by the current process."""
        return self._file_descriptor is not None

    def acquire(self) -> None:
        """Acquire the lock without waiting behind another controller."""
        if self.held:
            raise ControllerLockError("controller lock is already held by this instance")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            metadata = json.dumps({"owner": self.owner, "pid": os.getpid()}, sort_keys=True).encode(
                "utf-8"
            )
            os.ftruncate(file_descriptor, 0)
            os.write(file_descriptor, metadata + b"\n")
            os.fsync(file_descriptor)
        except (BlockingIOError, OSError) as error:
            os.close(file_descriptor)
            raise ControllerLockError(
                f"controller Git lock is already held: {self.path}"
            ) from error
        self._file_descriptor = file_descriptor

    def assert_held(self) -> None:
        """Fail unless this exact lock instance currently owns the lock."""
        if not self.held:
            raise ControllerLockError("controller mutation requires the held Git lock")

    def release(self) -> None:
        """Release the lock while retaining its audit metadata file."""
        self.assert_held()
        assert self._file_descriptor is not None
        file_descriptor = self._file_descriptor
        self._file_descriptor = None
        try:
            fcntl.flock(file_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(file_descriptor)

    def __enter__(self) -> ControllerFileLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.release()


def _command_text(arguments: Sequence[str]) -> str:
    """Render an argv for diagnostics without passing it through a shell."""
    return " ".join(arguments)


def _run_git_bytes(repo: Path, arguments: Sequence[str]) -> bytes:
    """Run Git in ``repo`` and return stdout bytes."""
    command = ["git", "-C", str(repo), *arguments]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as error:
        raise GitOpsError(f"could not execute `{_command_text(command)}`: {error}") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise GitOpsError(
            f"`{_command_text(command)}` failed with exit code {result.returncode}: {detail}"
        )
    return result.stdout


def _run_git(repo: Path, *arguments: str) -> str:
    """Run Git in ``repo`` and return decoded, stripped stdout."""
    return _run_git_bytes(repo, arguments).decode("utf-8").strip()


def _resolve_commit(repo: Path, revision: str) -> str:
    """Resolve one revision to a commit hash without option injection."""
    if not revision or "\x00" in revision or "\n" in revision or "\r" in revision:
        raise GitOpsError("Git revision must be a non-empty single line without NUL")
    return _run_git(
        repo,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{revision}^{{commit}}",
    )


def _run_git_unchecked(repo: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    """Run Git for a caller that must inspect and recover from failure."""
    command = ["git", "-C", str(repo), *arguments]
    try:
        return subprocess.run(command, capture_output=True, check=False)
    except OSError as error:
        raise GitOpsError(f"could not execute `{_command_text(command)}`: {error}") from error


def _decode_nul_paths(output: bytes) -> tuple[str, ...]:
    """Decode NUL-separated Git paths, rejecting non-UTF-8 repository names."""
    try:
        paths = (part.decode("utf-8") for part in output.split(b"\0") if part)
        return tuple(paths)
    except UnicodeDecodeError as error:
        raise GitOpsError("Git returned a path that is not valid UTF-8") from error


def _normalize_relative_path(path: str | Path) -> str:
    """Normalize one safe, repository-relative POSIX path."""
    value = Path(path).as_posix()
    pure_path = PurePosixPath(value)
    if (
        not value
        or pure_path.is_absolute()
        or value != pure_path.as_posix()
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or pure_path.parts[0] == ".git"
    ):
        raise GitOpsError(f"unsafe repository-relative path: {value!r}")
    return value


def _normalize_relative_paths(paths: Sequence[str | Path]) -> tuple[str, ...]:
    """Normalize an explicit non-empty path set and reject duplicates."""
    normalized = tuple(_normalize_relative_path(path) for path in paths)
    if not normalized:
        raise GitOpsError("an explicit non-empty path set is required")
    if len(set(normalized)) != len(normalized):
        raise GitOpsError("explicit path set contains duplicates")
    return normalized


def _changed_paths(repo: Path) -> tuple[str, ...]:
    """Return the union of staged, unstaged, and untracked paths."""
    outputs = (
        _run_git_bytes(repo, ("diff", "--name-only", "-z", "--ignore-submodules=all")),
        _run_git_bytes(repo, ("diff", "--cached", "--name-only", "-z", "--ignore-submodules=all")),
        # Do not honor ignore rules here. A worker-created ignored file is still
        # an undeclared candidate mutation and must fail the exact path claim.
        _run_git_bytes(repo, ("ls-files", "--others", "-z")),
    )
    paths = {
        _normalize_relative_path(path) for output in outputs for path in _decode_nul_paths(output)
    }
    return tuple(sorted(paths))


def _overlay_digest(path: Path, *, controller_checkout: bool = False) -> str:
    """Hash one stable candidate tree with the worker's content-manifest algorithm."""
    try:
        return digest_metadata_free_tree(path, ignore_root_git_file=controller_checkout)
    except (IsolationPolicyError, OSError) as error:
        raise GitOpsError(f"candidate overlay content manifest failed: {error}") from error


def _overlay_receipt_path(repository: Path) -> Path:
    """Return the controller-only receipt adjacent to a candidate worktree."""
    return repository.parent / f".{repository.name}.overlay-receipt.json"


def _overlay_receipt_payload(
    receipt: _OverlayReceipt, *, include_digest: bool
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "base_commit": receipt.base_commit,
        "branch": receipt.branch,
        "changed_paths": list(receipt.changed_paths),
        "overlay_sha256": receipt.overlay_sha256,
        "tree_hash": receipt.tree_hash,
    }
    if include_digest:
        payload["receipt_sha256"] = receipt.receipt_sha256
    return payload


def _overlay_receipt_digest(receipt: _OverlayReceipt) -> str:
    encoded = json.dumps(
        _overlay_receipt_payload(receipt, include_digest=False),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _new_overlay_receipt(
    *,
    base_commit: str,
    branch: str,
    changed_paths: tuple[str, ...],
    overlay_sha256: str,
    tree_hash: str,
) -> _OverlayReceipt:
    provisional = _OverlayReceipt(
        base_commit,
        branch,
        changed_paths,
        overlay_sha256,
        tree_hash,
        "",
    )
    return _OverlayReceipt(
        base_commit,
        branch,
        changed_paths,
        overlay_sha256,
        tree_hash,
        _overlay_receipt_digest(provisional),
    )


def _load_overlay_receipt(path: Path) -> _OverlayReceipt:
    """Load and content-verify one pre-commit overlay receipt."""
    if path.is_symlink() or not path.is_file():
        raise GitOpsError("overlay receipt is missing or is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GitOpsError(f"cannot load overlay receipt: {error}") from error
    keys = {
        "schema_version",
        "base_commit",
        "branch",
        "changed_paths",
        "overlay_sha256",
        "tree_hash",
        "receipt_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys or value["schema_version"] != 1:
        raise GitOpsError("overlay receipt has an invalid schema")
    changed_paths = value["changed_paths"]
    if not isinstance(changed_paths, list) or not all(
        isinstance(path_value, str) for path_value in changed_paths
    ):
        raise GitOpsError("overlay receipt changed_paths must be a string list")
    fields = (
        value["base_commit"],
        value["branch"],
        value["overlay_sha256"],
        value["tree_hash"],
        value["receipt_sha256"],
    )
    if not all(isinstance(field, str) for field in fields):
        raise GitOpsError("overlay receipt identity fields must be strings")
    receipt = _OverlayReceipt(
        value["base_commit"],
        value["branch"],
        tuple(changed_paths),
        value["overlay_sha256"],
        value["tree_hash"],
        value["receipt_sha256"],
    )
    if (
        re.fullmatch(r"[0-9a-f]{40}", receipt.base_commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", receipt.tree_hash) is None
        or re.fullmatch(r"[0-9a-f]{64}", receipt.overlay_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", receipt.receipt_sha256) is None
        or receipt.receipt_sha256 != _overlay_receipt_digest(receipt)
    ):
        raise GitOpsError("overlay receipt content digest or Git identity is invalid")
    return receipt


def _write_overlay_receipt_once(path: Path, receipt: _OverlayReceipt) -> None:
    """Publish an immutable pre-commit receipt without replacing an existing one."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(
                _overlay_receipt_payload(receipt, include_digest=True),
                output,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _load_overlay_receipt(path) != receipt:
                raise GitOpsError("immutable overlay receipt already differs")
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class ControllerGitOps:
    """Safe Git facade whose mutating surface requires an exclusive transaction."""

    def __init__(self, repository: str | Path, lock_path: str | Path, owner: str) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.lock = ControllerFileLock(lock_path, owner)
        root = Path(_run_git(self.repository, "rev-parse", "--show-toplevel")).resolve()
        if root != self.repository:
            raise GitOpsError(
                f"repository path must be the explicit Git top level: {self.repository} != {root}"
            )

    def inspect(self, repository: str | Path | None = None) -> RepositoryInspection:
        """Inspect the controller repository or an explicit linked worktree."""
        repo = self._resolve_managed_repository(repository)
        branch_result = _run_git_unchecked(repo, ("symbolic-ref", "--quiet", "--short", "HEAD"))
        if branch_result.returncode not in {0, 1}:
            detail = (
                (branch_result.stderr or branch_result.stdout)
                .decode("utf-8", errors="replace")
                .strip()
            )
            raise GitOpsError(f"could not inspect current branch: {detail}")
        branch_output = branch_result.stdout.decode("utf-8").strip()
        branch = branch_output or None
        return RepositoryInspection(
            root=repo,
            head=_run_git(repo, "rev-parse", "--verify", "HEAD^{commit}"),
            branch=branch,
            changed_paths=_changed_paths(repo),
        )

    def verify_candidate(
        self,
        candidate_repository: str | Path,
        *,
        base_commit: str,
        candidate_commit: str,
        allowed_paths: Sequence[str | Path],
        expected_patch_sha256: str | None = None,
    ) -> CandidateVerification:
        """Verify a frozen, one-commit candidate and its allowed path boundary."""
        repo = self._resolve_managed_repository(candidate_repository)
        allowed = set(_normalize_relative_paths(allowed_paths))
        base = _resolve_commit(repo, base_commit)
        candidate = _resolve_commit(repo, candidate_commit)
        parent_lines = _run_git(repo, "rev-list", "--parents", "-n", "1", candidate).split()
        if len(parent_lines) != 2 or parent_lines[1] != base:
            raise GitOpsError(
                "candidate must be exactly one commit whose sole parent is the frozen base"
            )

        changed = tuple(
            sorted(
                _normalize_relative_path(path)
                for path in _decode_nul_paths(
                    _run_git_bytes(repo, ("diff", "--name-only", "-z", base, candidate))
                )
            )
        )
        unexpected = sorted(set(changed) - allowed)
        if unexpected:
            raise GitOpsError(f"candidate changed paths outside its claim: {unexpected}")
        if not changed:
            raise GitOpsError("candidate commit has no changed paths")
        _run_git(repo, "diff", "--check", base, candidate)
        patch = _run_git_bytes(repo, ("diff", "--binary", "--full-index", base, candidate))
        patch_sha256 = hashlib.sha256(patch).hexdigest()
        if expected_patch_sha256 is not None and patch_sha256 != expected_patch_sha256:
            raise GitOpsError(
                "candidate patch hash mismatch: "
                f"expected {expected_patch_sha256}, observed {patch_sha256}"
            )
        return CandidateVerification(
            base_commit=base,
            candidate_commit=candidate,
            tree_hash=_run_git(repo, "rev-parse", f"{candidate}^{{tree}}"),
            patch_sha256=patch_sha256,
            changed_paths=changed,
        )

    def verify_linear_patch_sequence(
        self,
        *,
        previous_head: str,
        observed_head: str,
        expectations: Sequence[IntegratedPatchExpectation],
    ) -> IntegratedSequenceVerification:
        """Prove an observed fan-in is exactly the expected linear patch sequence.

        This read-only primitive deliberately knows nothing about catalog index
        semantics. The fan-in layer must derive each expected patch digest and
        exact path set from its frozen candidate receipts and semantic deltas.
        """
        previous = _resolve_commit(self.repository, previous_head)
        observed = _resolve_commit(self.repository, observed_head)
        normalized: list[IntegratedPatchExpectation] = []
        for expectation in expectations:
            if re.fullmatch(r"[0-9a-f]{64}", expectation.patch_sha256) is None:
                raise GitOpsError("integrated patch expectation has an invalid SHA-256 digest")
            paths = tuple(
                sorted(_normalize_relative_path(path) for path in expectation.changed_paths)
            )
            if not paths or len(paths) != len(set(paths)):
                raise GitOpsError("integrated patch expectation requires unique changed paths")
            normalized.append(IntegratedPatchExpectation(expectation.patch_sha256, paths))

        reverse_commits: list[str] = []
        cursor = observed
        for _expectation in reversed(normalized):
            parents = _run_git(self.repository, "rev-list", "--parents", "-n", "1", cursor).split()
            if len(parents) != 2:
                raise GitOpsError("integrated fan-in commits must form a linear one-parent history")
            reverse_commits.append(cursor)
            cursor = parents[1]
        if cursor != previous:
            raise GitOpsError(
                "integrated fan-in history has missing, extra, reordered, or substituted commits"
            )

        commits = tuple(reversed(reverse_commits))
        parent = previous
        evidence_entries: list[dict[str, object]] = []
        for commit, expectation in zip(commits, normalized, strict=True):
            changed_paths = tuple(
                sorted(
                    _normalize_relative_path(path)
                    for path in _decode_nul_paths(
                        _run_git_bytes(
                            self.repository,
                            ("diff", "--name-only", "-z", parent, commit),
                        )
                    )
                )
            )
            if changed_paths != expectation.changed_paths:
                raise GitOpsError(
                    "integrated fan-in commit changed paths differ from its frozen expectation"
                )
            _run_git(self.repository, "diff", "--check", parent, commit)
            patch = _run_git_bytes(
                self.repository,
                ("diff", "--binary", "--full-index", parent, commit),
            )
            patch_sha256 = hashlib.sha256(patch).hexdigest()
            if patch_sha256 != expectation.patch_sha256:
                raise GitOpsError(
                    "integrated fan-in commit patch differs from its frozen expectation"
                )
            tree_hash = _run_git(self.repository, "rev-parse", f"{commit}^{{tree}}")
            evidence_entries.append(
                {
                    "commit": commit,
                    "parent": parent,
                    "tree_hash": tree_hash,
                    "patch_sha256": patch_sha256,
                    "changed_paths": list(changed_paths),
                }
            )
            parent = commit

        final_tree = _run_git(self.repository, "rev-parse", f"{observed}^{{tree}}")
        evidence = json.dumps(
            {
                "previous_head": previous,
                "final_head": observed,
                "final_tree_hash": final_tree,
                "commits": evidence_entries,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return IntegratedSequenceVerification(
            previous,
            observed,
            final_tree,
            commits,
            hashlib.sha256(evidence).hexdigest(),
        )

    def render_patch(self, *, base_commit: str, head_commit: str) -> bytes:
        """Render one deterministic binary patch without mutating any checkout."""
        base = _resolve_commit(self.repository, base_commit)
        head = _resolve_commit(self.repository, head_commit)
        ancestry = _run_git_unchecked(
            self.repository,
            ("merge-base", "--is-ancestor", base, head),
        )
        if ancestry.returncode != 0:
            raise GitOpsError("delivery head is not descended from its frozen base")
        return _run_git_bytes(
            self.repository,
            ("diff", "--binary", "--full-index", base, head),
        )

    @contextmanager
    def transaction(self) -> Iterator[LockedGitOps]:
        """Yield the only mutation-capable facade while holding the controller lock."""
        with self.lock:
            yield LockedGitOps(self)

    def _resolve_managed_repository(self, repository: str | Path | None) -> Path:
        """Resolve and verify a repository or linked worktree path."""
        repo = self.repository if repository is None else Path(repository).expanduser().resolve()
        if not repo.is_dir():
            raise GitOpsError(f"Git repository path is not a directory: {repo}")
        common_dir = Path(_run_git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir"))
        root_common_dir = Path(
            _run_git(self.repository, "rev-parse", "--path-format=absolute", "--git-common-dir")
        )
        if common_dir.resolve() != root_common_dir.resolve():
            raise GitOpsError(
                f"path is not a worktree managed by the controller repository: {repo}"
            )
        return repo


class LockedGitOps:
    """Mutation-capable Git facade valid only for one held controller transaction."""

    def __init__(self, controller: ControllerGitOps) -> None:
        self._controller = controller

    @property
    def lock(self) -> ControllerFileLock:
        """Return the held lock for an atomic index update in this transaction."""
        self._controller.lock.assert_held()
        return self._controller.lock

    def create_candidate_worktree(
        self,
        path: str | Path,
        *,
        branch: str,
        base_commit: str,
    ) -> Path:
        """Create one candidate branch and linked worktree serially."""
        self._assert_locked()
        worktree = Path(path).expanduser().resolve()
        if worktree.exists():
            raise GitOpsError(f"candidate worktree path already exists: {worktree}")
        _run_git(self._controller.repository, "check-ref-format", "--branch", branch)
        base = _resolve_commit(self._controller.repository, base_commit)
        worktree.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            self._controller.repository,
            "worktree",
            "add",
            "-b",
            branch,
            str(worktree),
            base,
        )
        return worktree

    def create_candidate_overlay(self, path: str | Path, *, base_commit: str) -> Path:
        """Atomically export a commit into a worker-visible tree without Git metadata."""
        self._assert_locked()
        overlay = Path(path).expanduser().resolve()
        if overlay.exists():
            raise GitOpsError(f"candidate overlay path already exists: {overlay}")
        base = _resolve_commit(self._controller.repository, base_commit)
        overlay.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{overlay.name}.materializing-", dir=overlay.parent)
        )
        try:
            archive_bytes = _run_git_bytes(
                self._controller.repository,
                ("archive", "--format=tar", base),
            )
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:") as archive:
                archive.extractall(temporary, filter="data")
            _validate_metadata_free_overlay(temporary)
            os.replace(temporary, overlay)
            _fsync_directory(overlay.parent)
        except (OSError, tarfile.TarError) as error:
            raise GitOpsError(f"could not materialize candidate overlay: {error}") from error
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return overlay

    def create_detached_worktree(self, path: str | Path, *, base_commit: str) -> Path:
        """Create one controller-only detached integration worktree."""
        self._assert_locked()
        worktree = Path(path).expanduser().resolve()
        if worktree.exists():
            raise GitOpsError(f"detached worktree path already exists: {worktree}")
        base = _resolve_commit(self._controller.repository, base_commit)
        worktree.parent.mkdir(parents=True, exist_ok=True)
        _run_git(
            self._controller.repository,
            "worktree",
            "add",
            "--detach",
            str(worktree),
            base,
        )
        return worktree

    def bind_candidate_overlay(
        self,
        overlay: str | Path,
        *,
        repository: str | Path,
        branch: str,
        base_commit: str,
        expected_paths: Sequence[str | Path],
        expected_overlay_sha256: str,
        message: str,
    ) -> OverlayBinding:
        """Import an untrusted overlay into a controller-only linked worktree.

        The worker never sees ``repository``. The full overlay is synchronized
        before Git computes the changed path set, so undeclared additions,
        deletions, mode changes, symlinks, and ignored files fail closed.
        """
        self._assert_locked()
        source = Path(overlay).expanduser().resolve(strict=True)
        _validate_metadata_free_overlay(source)
        destination = Path(repository).expanduser().resolve()
        base = _resolve_commit(self._controller.repository, base_commit)
        expected = tuple(sorted(_normalize_relative_path(path) for path in expected_paths))
        if len(expected) != len(set(expected)):
            raise GitOpsError("candidate path claim contains duplicates")
        if re.fullmatch(r"[0-9a-f]{64}", expected_overlay_sha256) is None:
            raise GitOpsError("candidate overlay requires a valid scanned content manifest")

        overlay_sha256 = _overlay_digest(source)
        if overlay_sha256 != expected_overlay_sha256:
            raise GitOpsError("candidate overlay differs from its scanned content manifest")
        receipt_path = _overlay_receipt_path(destination)
        receipt = _load_overlay_receipt(receipt_path) if receipt_path.exists() else None
        if receipt is not None and (
            receipt.base_commit != base
            or receipt.branch != branch
            or receipt.changed_paths != expected
            or receipt.overlay_sha256 != overlay_sha256
        ):
            raise GitOpsError("immutable overlay receipt differs from the candidate claim")

        if destination.exists():
            inspection = self._controller.inspect(destination)
            if inspection.branch != branch:
                raise GitOpsError("existing controller candidate repository differs from binding")
            if inspection.head != base:
                if not inspection.clean:
                    raise GitOpsError("recovered controller candidate repository is not clean")
                if receipt is None:
                    raise GitOpsError("recovered candidate commit lacks its pre-commit receipt")
                verification = self._controller.verify_candidate(
                    destination,
                    base_commit=base,
                    candidate_commit=inspection.head,
                    allowed_paths=expected,
                )
                if verification.changed_paths != expected:
                    raise GitOpsError("recovered controller candidate paths differ from claim")
                if verification.tree_hash != receipt.tree_hash:
                    raise GitOpsError("recovered candidate tree differs from its overlay receipt")
                return OverlayBinding(
                    source,
                    destination,
                    base,
                    verification.candidate_commit,
                    verification.changed_paths,
                )
        else:
            self.create_candidate_worktree(
                destination,
                branch=branch,
                base_commit=base,
            )

        inspection = self._controller.inspect(destination)
        if inspection.head != base or inspection.branch != branch:
            raise GitOpsError("controller candidate repository is not at the frozen base")
        _replace_checkout_from_overlay(source, destination)
        if _overlay_digest(destination, controller_checkout=True) != expected_overlay_sha256:
            raise GitOpsError("imported candidate differs from its scanned content manifest")
        observed = _changed_paths(destination)
        if observed != expected:
            raise GitOpsError(
                "overlay changed paths differ from the strict claim: "
                f"expected {list(expected)!r}, observed {list(observed)!r}"
            )
        if not expected:
            if receipt is not None:
                raise GitOpsError("unchanged candidate unexpectedly has a pre-commit receipt")
            return OverlayBinding(source, destination, base, base, ())
        self.stage_paths(destination, expected)
        tree_hash = _run_git(destination, "write-tree")
        if _overlay_digest(source) != expected_overlay_sha256:
            raise GitOpsError("candidate overlay changed while it was being bound")
        expected_receipt = _new_overlay_receipt(
            base_commit=base,
            branch=branch,
            changed_paths=expected,
            overlay_sha256=overlay_sha256,
            tree_hash=tree_hash,
        )
        if receipt is not None and receipt != expected_receipt:
            raise GitOpsError("staged candidate tree differs from its immutable overlay receipt")
        _write_overlay_receipt_once(receipt_path, expected_receipt)
        candidate = self.commit_signed_off(
            destination,
            message=message,
            expected_paths=expected,
        )
        if _run_git(destination, "rev-parse", f"{candidate}^{{tree}}") != tree_hash:
            raise GitOpsError("candidate commit tree differs from its pre-commit overlay receipt")
        return OverlayBinding(source, destination, base, candidate, expected)

    def stage_paths(self, repository: str | Path, paths: Sequence[str | Path]) -> tuple[str, ...]:
        """Stage only an explicit path set after rejecting unrelated changes."""
        self._assert_locked()
        repo = self._controller._resolve_managed_repository(repository)
        normalized = _normalize_relative_paths(paths)
        unexpected = sorted(set(_changed_paths(repo)) - set(normalized))
        if unexpected:
            raise GitOpsError(f"worktree has unexpected changed paths: {unexpected}")
        _run_git(repo, "add", "--", *normalized)
        staged = tuple(
            sorted(
                _normalize_relative_path(path)
                for path in _decode_nul_paths(
                    _run_git_bytes(repo, ("diff", "--cached", "--name-only", "-z"))
                )
            )
        )
        if set(staged) != set(normalized):
            raise GitOpsError(
                f"staged path set does not match explicit claim: expected {normalized}, got {staged}"
            )
        return staged

    def commit_signed_off(
        self,
        repository: str | Path,
        *,
        message: str,
        expected_paths: Sequence[str | Path],
        gpg_signing_key: str | None = None,
    ) -> str:
        """Commit exactly the staged paths with DCO sign-off and optional GPG signing."""
        self._assert_locked()
        if not message.strip() or "\x00" in message:
            raise GitOpsError("commit message must be non-empty and contain no NUL")
        repo = self._controller._resolve_managed_repository(repository)
        expected = set(_normalize_relative_paths(expected_paths))
        staged = set(
            _normalize_relative_path(path)
            for path in _decode_nul_paths(
                _run_git_bytes(repo, ("diff", "--cached", "--name-only", "-z"))
            )
        )
        if staged != expected:
            raise GitOpsError(
                f"staged paths differ from the commit claim: expected {sorted(expected)}, "
                f"got {sorted(staged)}"
            )
        arguments = ["commit", "-s"]
        if gpg_signing_key is not None:
            if not re.fullmatch(r"[A-Za-z0-9]+", gpg_signing_key):
                raise GitOpsError("GPG signing key must be an alphanumeric key ID")
            arguments.append(f"-S{gpg_signing_key}")
        arguments.extend(("-m", message))
        _run_git(repo, *arguments)
        if _changed_paths(repo):
            raise GitOpsError(
                "commit hooks left uncommitted changes; review and restage explicitly"
            )
        return _run_git(repo, "rev-parse", "--verify", "HEAD^{commit}")

    def integrate_fast_forward(self, *, candidate_ref: str, expected_head: str) -> str:
        """Fast-forward the clean integration checkout to an explicit candidate ref."""
        self._assert_integration_preconditions(expected_head)
        candidate = _resolve_commit(self._controller.repository, candidate_ref)
        _run_git(self._controller.repository, "merge", "--ff-only", candidate)
        return _run_git(self._controller.repository, "rev-parse", "--verify", "HEAD^{commit}")

    def integrate_commit(self, *, candidate_commit: str, expected_head: str) -> str:
        """Apply one candidate commit with sign-off, failing closed on conflicts.

        This is the non-fast-forward, cherry-pick-like fan-in path. Git hooks
        remain enabled. On any failure the in-progress cherry-pick is aborted;
        no conflict resolution or automatic merge is attempted.
        """
        self._assert_integration_preconditions(expected_head)
        candidate = _resolve_commit(self._controller.repository, candidate_commit)
        result = _run_git_unchecked(self._controller.repository, ("cherry-pick", "-s", candidate))
        if result.returncode != 0:
            abort_result = _run_git_unchecked(
                self._controller.repository, ("cherry-pick", "--abort")
            )
            detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
            recovered_head = _run_git(
                self._controller.repository, "rev-parse", "--verify", "HEAD^{commit}"
            )
            residual_paths = _changed_paths(self._controller.repository)
            if abort_result.returncode != 0 or recovered_head != expected_head or residual_paths:
                abort_detail = (
                    (abort_result.stderr or abort_result.stdout)
                    .decode("utf-8", errors="replace")
                    .strip()
                )
                raise GitOpsError(
                    "candidate integration failed and its explicit abort did not restore the "
                    f"frozen checkout; cherry-pick={detail!r}, abort={abort_detail!r}, "
                    f"head={recovered_head}, changed_paths={list(residual_paths)}"
                )
            raise GitConflictError(f"candidate integration failed without resolution: {detail}")
        if _changed_paths(self._controller.repository):
            raise GitOpsError("integration hooks left uncommitted changes")
        return _run_git(self._controller.repository, "rev-parse", "--verify", "HEAD^{commit}")

    def _assert_integration_preconditions(self, expected_head: str) -> None:
        """Check lock, clean checkout, and compare-and-swap HEAD guard."""
        self._assert_locked()
        current_head = _run_git(
            self._controller.repository, "rev-parse", "--verify", "HEAD^{commit}"
        )
        expected = _resolve_commit(self._controller.repository, expected_head)
        if current_head != expected:
            raise GitOpsError(
                f"integration HEAD changed: expected {expected}, observed {current_head}"
            )
        changed = _changed_paths(self._controller.repository)
        if changed:
            raise GitOpsError(f"integration checkout is not clean: {list(changed)}")

    def _assert_locked(self) -> None:
        """Fail if this transaction outlived its controller lock context."""
        self._controller.lock.assert_held()


def _validate_metadata_free_overlay(path: Path) -> None:
    """Reject Git metadata and links that can escape a candidate overlay."""
    if not path.is_dir() or path.is_symlink():
        raise GitOpsError("candidate overlay must be a non-symlink directory")
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in (*directories, *files):
            entry = root_path / name
            relative = entry.relative_to(path)
            if ".git" in relative.parts:
                raise GitOpsError("candidate overlay contains prohibited Git metadata")
            if entry.is_symlink():
                resolved = entry.resolve(strict=False)
                if not resolved.is_relative_to(path):
                    raise GitOpsError("candidate overlay symlink escapes its root")
                if ".git" in resolved.relative_to(path).parts:
                    raise GitOpsError(
                        "candidate overlay symlink would expose Git metadata after import"
                    )


def _replace_checkout_from_overlay(overlay: Path, checkout: Path) -> None:
    """Synchronize a validated overlay into a controller-owned worktree."""
    _validate_metadata_free_overlay(overlay)
    for child in checkout.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in overlay.iterdir():
        destination = checkout / child.name
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, destination, symlinks=True)
        elif child.is_symlink():
            destination.symlink_to(os.readlink(child))
        else:
            shutil.copy2(child, destination, follow_symlinks=False)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_isolated_delivery_checkout(
    source_repository: str | Path,
    *,
    workspace: Path,
    branch: str,
    base_commit: str,
    expected_head: str | None,
    lock_path: Path,
    owner: str,
) -> DeliveryCheckout:
    """Create or adopt the only checkout allowed to receive delivery commits.

    The user's supplied checkout remains on its original branch and HEAD. The
    delivery branch is checked out only below the run workspace, so controller
    fan-in cannot mutate the shared/user working tree.
    """
    source = Path(source_repository).expanduser().resolve(strict=True)
    root = workspace.expanduser().resolve(strict=True)
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise GitOpsError("delivery workspace and source repository must not overlap")
    checkout = root / "delivery" / "repository"
    source_git = ControllerGitOps(source, lock_path, owner)
    source_before = source_git.inspect()
    frozen_base = _resolve_commit(source, base_commit)
    expected = frozen_base if expected_head is None else _resolve_commit(source, expected_head)

    if not checkout.exists():
        if expected != frozen_base:
            raise GitOpsError("missing delivery checkout cannot resume a non-base integration head")
        with source_git.transaction() as transaction:
            transaction.create_candidate_worktree(
                checkout,
                branch=branch,
                base_commit=frozen_base,
            )

    source_after = source_git.inspect()
    if source_after != source_before:
        raise GitOpsError("creating the isolated delivery checkout changed the source checkout")
    delivery_git = ControllerGitOps(checkout, lock_path, owner)
    inspection = delivery_git.inspect()
    if inspection.branch != branch:
        raise GitOpsError("isolated delivery checkout is on the wrong branch")
    if inspection.head != expected:
        raise GitOpsError(
            f"isolated delivery HEAD differs from state: expected {expected}, "
            f"observed {inspection.head}"
        )
    if not inspection.clean:
        raise GitOpsError("isolated delivery checkout must be clean")
    return DeliveryCheckout(checkout, branch, inspection.head, delivery_git)


def prepare_diff_delivery_checkout(
    source_repository: str | Path,
    *,
    workspace: Path,
    base_commit: str,
    expected_head: str | None,
    lock_path: Path,
    owner: str,
) -> DeliveryCheckout:
    """Create or adopt a detached controller checkout for diff-only delivery.

    Internal controller commits provide restart-safe content identities, but no
    delivery branch or user checkout is moved. The terminal product is a patch
    from ``base_commit`` to this detached head.
    """
    source = Path(source_repository).expanduser().resolve(strict=True)
    root = workspace.expanduser().resolve(strict=True)
    if root == source or root.is_relative_to(source) or source.is_relative_to(root):
        raise GitOpsError("delivery workspace and source repository must not overlap")
    checkout = root / "delivery" / "diff-repository"
    source_git = ControllerGitOps(source, lock_path, owner)
    source_before = source_git.inspect()
    frozen_base = _resolve_commit(source, base_commit)
    expected = frozen_base if expected_head is None else _resolve_commit(source, expected_head)
    if not checkout.exists():
        if expected != frozen_base:
            raise GitOpsError("missing diff checkout cannot resume a non-base integration head")
        with source_git.transaction() as transaction:
            transaction.create_detached_worktree(checkout, base_commit=frozen_base)
    source_after = source_git.inspect()
    if source_after != source_before:
        raise GitOpsError("creating the diff checkout changed the source checkout")
    delivery_git = ControllerGitOps(checkout, lock_path, owner)
    inspection = delivery_git.inspect()
    if inspection.branch is not None:
        raise GitOpsError("diff-only delivery checkout must remain detached")
    if inspection.head != expected:
        raise GitOpsError(
            f"diff-only delivery HEAD differs from state: expected {expected}, "
            f"observed {inspection.head}"
        )
    if not inspection.clean:
        raise GitOpsError("diff-only delivery checkout must be clean")
    return DeliveryCheckout(checkout, None, inspection.head, delivery_git)


def publish_delivery_patch(
    delivery: DeliveryCheckout,
    *,
    base_commit: str,
    output_path: Path,
) -> DeliveryPatch:
    """Atomically publish the exact diff-only artifact for an integration head."""
    if delivery.branch is not None:
        raise GitOpsError("a named-branch delivery cannot be published as diff-only")
    patch = delivery.git.render_patch(
        base_commit=base_commit,
        head_commit=delivery.head,
    )
    digest = hashlib.sha256(patch).hexdigest()
    requested = output_path.expanduser()
    if not requested.is_absolute():
        requested = Path.cwd() / requested
    if requested.is_symlink():
        raise GitOpsError("delivery patch path cannot be a symlink")
    delivery_root = delivery.repository.parent.resolve(strict=True)
    parent = requested.parent.resolve(strict=False)
    if not parent.is_relative_to(delivery_root):
        raise GitOpsError("delivery patch parent escapes the isolated delivery directory")
    parent.mkdir(parents=True, exist_ok=True)
    parent = parent.resolve(strict=True)
    destination = parent / requested.name
    if destination.is_symlink():
        raise GitOpsError("delivery patch path cannot be a symlink")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(patch)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file():
                raise GitOpsError("delivery patch path is not a regular file")
            if destination.read_bytes() != patch:
                raise GitOpsError("existing delivery patch differs from integration head")
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)
    return DeliveryPatch(
        destination,
        _resolve_commit(delivery.repository, base_commit),
        delivery.head,
        digest,
        len(patch),
    )
