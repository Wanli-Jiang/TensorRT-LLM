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

"""Immutable worker mailboxes and evidence validation for Staircase."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypeAlias, cast

from agent_flow.workflows.staircase.state import DomainProfile, Role

MANIFEST_SCHEMA_VERSION = 1
INPUT_FILENAME = "input.json"
OUTPUT_DIRECTORY = "output"
RESULT_FILENAME = "result.json"
COMPLETE_FILENAME = "COMPLETE"
QUARANTINE_FILENAME = "QUARANTINE.json"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ArtifactError(RuntimeError):
    """Base class for invalid or conflicting mailbox artifacts."""


class ImmutableArtifactError(ArtifactError):
    """Raised when a writer tries to replace an immutable artifact."""


class IncompleteResultError(ArtifactError):
    """Raised when a result lacks the marker written last by its worker."""


class ArtifactDigestError(ArtifactError):
    """Raised when manifest or evidence bytes do not match recorded digests."""


class DuplicateResultError(ArtifactError):
    """Raised when an attempt result was already ingested."""


class StaleResultError(ArtifactError):
    """Raised after a late or identity-mismatched result is quarantined."""


class WorkerResultStatus(str, Enum):
    """Typed worker outcome; scheduler completion is not domain approval."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYABLE_FAILED = "retryable_failed"
    BLOCKED = "blocked"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    RESOURCE_ESCALATION = "resource_escalation"


@dataclass(frozen=True)
class EvidenceFile:
    """Content-addressed file published beside a worker result."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_relative_path(self.path)
        _validate_digest("evidence sha256", self.sha256)
        if self.size_bytes < 0:
            raise ValueError("evidence size_bytes must be non-negative")


@dataclass(frozen=True)
class WorkerInputManifest:
    """Immutable input contract for exactly one isolated worker process."""

    run_id: str
    item_id: str
    attempt_id: str
    task_digest: str
    generation: int
    role: Role
    profile: DomainProfile | None
    worktree: str
    allowed_paths: tuple[str, ...] = ()
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identity(self.run_id, self.item_id, self.attempt_id)
        _validate_digest("task_digest", self.task_digest)
        if self.generation < 1:
            raise ValueError("input generation must be positive")
        if not self.worktree:
            raise ValueError("worker input requires a canonical worktree")
        for path in self.allowed_paths:
            _validate_relative_path(path)
        if len(set(self.allowed_paths)) != len(self.allowed_paths):
            raise ValueError("worker input has duplicate allowed paths")
        _validate_json_value(self.payload, "payload")


@dataclass(frozen=True)
class WorkerResultManifest:
    """Immutable typed result for exactly one worker attempt."""

    run_id: str
    item_id: str
    attempt_id: str
    task_digest: str
    generation: int
    input_digest: str
    status: WorkerResultStatus
    summary: str
    evidence: tuple[EvidenceFile, ...] = ()
    candidate_digest: str | None = None
    reviewed_candidate_digest: str | None = None
    payload: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identity(self.run_id, self.item_id, self.attempt_id)
        _validate_digest("task_digest", self.task_digest)
        _validate_digest("input_digest", self.input_digest)
        if self.generation < 1:
            raise ValueError("result generation must be positive")
        if not self.summary.strip():
            raise ValueError("result summary must be non-empty")
        paths = [entry.path for entry in self.evidence]
        if len(set(paths)) != len(paths):
            raise ValueError("result contains duplicate evidence paths")
        if self.candidate_digest is not None:
            _validate_digest("candidate_digest", self.candidate_digest)
        if self.reviewed_candidate_digest is not None:
            _validate_digest("reviewed_candidate_digest", self.reviewed_candidate_digest)
        _validate_json_value(self.payload, "payload")


@dataclass(frozen=True)
class ResultExpectation:
    """Controller-owned identity against which a mailbox is ingested."""

    run_id: str
    item_id: str
    attempt_id: str
    task_digest: str
    generation: int
    input_digest: str

    def __post_init__(self) -> None:
        _validate_identity(self.run_id, self.item_id, self.attempt_id)
        _validate_digest("task_digest", self.task_digest)
        _validate_digest("input_digest", self.input_digest)
        if self.generation < 1:
            raise ValueError("expected generation must be positive")


@dataclass(frozen=True)
class IngestedResult:
    """A validated result plus its content digest and durable receipt."""

    manifest: WorkerResultManifest
    result_digest: str
    receipt_path: Path


def digest_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular, non-symlink file.

    Args:
        path: Evidence file to hash.

    Returns:
        Lowercase SHA-256 digest.
    """
    if path.is_symlink() or not path.is_file():
        raise ArtifactDigestError(f"evidence is not a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_evidence(attempt_dir: Path, relative_path: str) -> EvidenceFile:
    """Build a content-addressed evidence record under an attempt directory.

    Args:
        attempt_dir: Worker attempt mailbox root.
        relative_path: Safe POSIX-relative evidence path.

    Returns:
        Validated evidence description.
    """
    evidence_path = _resolve_member(attempt_dir, relative_path)
    return EvidenceFile(
        path=relative_path,
        sha256=digest_file(evidence_path),
        size_bytes=evidence_path.stat().st_size,
    )


def write_input_manifest(path: Path, manifest: WorkerInputManifest) -> str:
    """Publish an immutable worker input manifest atomically.

    Args:
        path: Usually ``<attempt>/input.json``.
        manifest: Frozen worker contract.

    Returns:
        Digest workers must echo in their result.
    """
    payload = _input_to_dict(manifest)
    envelope = _manifest_envelope("worker_input", payload)
    _write_once_json(path, envelope)
    return _require_envelope_digest(envelope)


def load_input_manifest(path: Path) -> tuple[WorkerInputManifest, str]:
    """Load an input manifest and validate its content digest.

    Args:
        path: Immutable input manifest path.

    Returns:
        Manifest and its canonical digest.
    """
    envelope = _load_envelope(path, "worker_input")
    payload = _require_object(envelope.get("payload"), "worker input payload")
    _expect_keys(
        payload,
        {
            "run_id",
            "item_id",
            "attempt_id",
            "task_digest",
            "generation",
            "role",
            "profile",
            "worktree",
            "allowed_paths",
            "payload",
        },
        "worker input",
    )
    allowed_paths = _require_list(payload["allowed_paths"], "allowed_paths")
    manifest = WorkerInputManifest(
        run_id=_require_string(payload["run_id"], "run_id"),
        item_id=_require_string(payload["item_id"], "item_id"),
        attempt_id=_require_string(payload["attempt_id"], "attempt_id"),
        task_digest=_require_string(payload["task_digest"], "task_digest"),
        generation=_require_integer(payload["generation"], "generation"),
        role=Role(_require_string(payload["role"], "role")),
        profile=(
            DomainProfile(profile)
            if (profile := _optional_string(payload["profile"], "profile")) is not None
            else None
        ),
        worktree=_require_string(payload["worktree"], "worktree"),
        allowed_paths=tuple(_require_string(entry, "allowed path") for entry in allowed_paths),
        payload=_json_object(payload["payload"], "payload"),
    )
    return manifest, _require_envelope_digest(envelope)


def worker_output_directory(attempt_dir: Path) -> Path:
    """Return the only worker-writable mailbox beneath an attempt root."""
    return attempt_dir / OUTPUT_DIRECTORY


def publish_result(
    attempt_dir: Path,
    manifest: WorkerResultManifest,
    *,
    input_path: Path | None = None,
) -> str:
    """Publish a result and write ``COMPLETE`` only after all bytes are durable.

    Args:
        attempt_dir: Attempt mailbox containing input and evidence.
        manifest: Typed worker outcome.

    Returns:
        Canonical result-manifest digest.
    """
    _require_attempt_directory(attempt_dir)
    worker_input, input_digest = load_input_manifest(
        attempt_dir / INPUT_FILENAME if input_path is None else input_path
    )
    if manifest.input_digest != input_digest:
        raise ArtifactDigestError("result input_digest does not match immutable input.json")
    input_identity = (
        worker_input.run_id,
        worker_input.item_id,
        worker_input.attempt_id,
        worker_input.task_digest,
        worker_input.generation,
    )
    result_identity = (
        manifest.run_id,
        manifest.item_id,
        manifest.attempt_id,
        manifest.task_digest,
        manifest.generation,
    )
    if result_identity != input_identity:
        raise ArtifactError("result identity does not match immutable input.json")
    _validate_evidence(attempt_dir, manifest.evidence)
    _make_evidence_durable(attempt_dir, manifest.evidence)
    payload = _result_to_dict(manifest)
    envelope = _manifest_envelope("worker_result", payload)
    result_digest = _require_envelope_digest(envelope)
    _write_once_json(attempt_dir / RESULT_FILENAME, envelope)
    _fsync_directory(attempt_dir)
    complete = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "worker_complete",
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "generation": manifest.generation,
        "result_digest": result_digest,
    }
    _write_once_json(attempt_dir / COMPLETE_FILENAME, complete)
    return result_digest


def load_result_manifest(attempt_dir: Path) -> tuple[WorkerResultManifest, str]:
    """Load a complete result and verify marker, manifest, and evidence bytes.

    Args:
        attempt_dir: Attempt mailbox root.

    Returns:
        Result manifest and its canonical digest.
    """
    _require_attempt_directory(attempt_dir)
    complete_path = attempt_dir / COMPLETE_FILENAME
    if not complete_path.is_file():
        raise IncompleteResultError(f"attempt has no durable {COMPLETE_FILENAME} marker")
    complete = _load_json_object(complete_path)
    _expect_keys(
        complete,
        {
            "schema_version",
            "kind",
            "run_id",
            "item_id",
            "attempt_id",
            "generation",
            "result_digest",
        },
        "completion marker",
    )
    if (
        complete["schema_version"] != MANIFEST_SCHEMA_VERSION
        or complete["kind"] != "worker_complete"
    ):
        raise ArtifactError("unsupported completion marker")
    envelope = _load_envelope(attempt_dir / RESULT_FILENAME, "worker_result")
    result_digest = _require_envelope_digest(envelope)
    if complete["result_digest"] != result_digest:
        raise ArtifactDigestError("COMPLETE does not name the durable result manifest")
    payload = _require_object(envelope.get("payload"), "worker result payload")
    _expect_keys(
        payload,
        {
            "run_id",
            "item_id",
            "attempt_id",
            "task_digest",
            "generation",
            "input_digest",
            "status",
            "summary",
            "evidence",
            "candidate_digest",
            "reviewed_candidate_digest",
            "payload",
        },
        "worker result",
    )
    evidence_raw = _require_list(payload["evidence"], "evidence")
    evidence = tuple(_evidence_from_object(entry) for entry in evidence_raw)
    manifest = WorkerResultManifest(
        run_id=_require_string(payload["run_id"], "run_id"),
        item_id=_require_string(payload["item_id"], "item_id"),
        attempt_id=_require_string(payload["attempt_id"], "attempt_id"),
        task_digest=_require_string(payload["task_digest"], "task_digest"),
        generation=_require_integer(payload["generation"], "generation"),
        input_digest=_require_string(payload["input_digest"], "input_digest"),
        status=WorkerResultStatus(_require_string(payload["status"], "status")),
        summary=_require_string(payload["summary"], "summary"),
        evidence=evidence,
        candidate_digest=_optional_string(payload["candidate_digest"], "candidate_digest"),
        reviewed_candidate_digest=_optional_string(
            payload["reviewed_candidate_digest"], "reviewed_candidate_digest"
        ),
        payload=_json_object(payload["payload"], "payload"),
    )
    if (
        complete["run_id"] != manifest.run_id
        or complete["item_id"] != manifest.item_id
        or complete["attempt_id"] != manifest.attempt_id
        or complete["generation"] != manifest.generation
    ):
        raise ArtifactError("COMPLETE identity does not match result.json")
    _validate_evidence(attempt_dir, manifest.evidence)
    return manifest, result_digest


def ingest_result(
    attempt_dir: Path,
    expectation: ResultExpectation,
    *,
    receipt_root: Path,
    quarantine_root: Path,
) -> IngestedResult:
    """Validate and record exactly one accepted result for an attempt.

    Identity, task, input, or generation mismatches are moved into quarantine
    before :class:`StaleResultError` is raised. A durable receipt created with
    exclusive filesystem semantics prevents duplicate ingestion.

    Args:
        attempt_dir: Completed worker mailbox.
        expectation: Controller-owned current attempt identity.
        receipt_root: Controller-only directory for ingestion receipts.
        quarantine_root: Controller-only directory for stale mailboxes.

    Returns:
        The validated result and durable receipt identity.
    """
    manifest, result_digest = load_result_manifest(attempt_dir)
    mismatch = _identity_mismatch(manifest, expectation)
    if mismatch is not None:
        quarantine_path = quarantine_result(attempt_dir, quarantine_root, mismatch)
        raise StaleResultError(f"{mismatch}; quarantined at {quarantine_path}")
    receipt_path = receipt_root / manifest.run_id / manifest.item_id / f"{manifest.attempt_id}.json"
    receipt = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "generation": manifest.generation,
        "task_digest": manifest.task_digest,
        "input_digest": manifest.input_digest,
        "result_digest": result_digest,
    }
    try:
        _write_once_json(receipt_path, receipt)
    except ImmutableArtifactError as error:
        raise DuplicateResultError(
            f"result for attempt {manifest.attempt_id!r} was already ingested"
        ) from error
    return IngestedResult(
        manifest=manifest,
        result_digest=result_digest,
        receipt_path=receipt_path,
    )


def quarantine_result(attempt_dir: Path, quarantine_root: Path, reason: str) -> Path:
    """Atomically move a stale worker mailbox into controller quarantine.

    Args:
        attempt_dir: Late or mismatched attempt mailbox.
        quarantine_root: Controller-only quarantine directory on the same filesystem.
        reason: Human-readable quarantine explanation.

    Returns:
        New mailbox path.
    """
    if not attempt_dir.is_dir() or attempt_dir.is_symlink():
        raise ArtifactError(f"attempt mailbox is not a directory: {attempt_dir}")
    quarantine_root.mkdir(parents=True, exist_ok=True)
    suffix = hashlib.sha256(reason.encode("utf-8")).hexdigest()[:12]
    destination = quarantine_root / f"{attempt_dir.name}-{suffix}"
    if destination.exists():
        raise ImmutableArtifactError(f"quarantine destination already exists: {destination}")
    receipt = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "quarantine",
        "reason": reason,
    }
    _write_once_json(attempt_dir / QUARANTINE_FILENAME, receipt)
    attempt_dir.rename(destination)
    _fsync_directory(destination.parent)
    return destination


def _input_to_dict(manifest: WorkerInputManifest) -> dict[str, JsonValue]:
    return {
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "task_digest": manifest.task_digest,
        "generation": manifest.generation,
        "role": manifest.role.value,
        "profile": manifest.profile.value if manifest.profile is not None else None,
        "worktree": manifest.worktree,
        "allowed_paths": list(manifest.allowed_paths),
        "payload": manifest.payload,
    }


def _result_to_dict(manifest: WorkerResultManifest) -> dict[str, JsonValue]:
    return {
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "task_digest": manifest.task_digest,
        "generation": manifest.generation,
        "input_digest": manifest.input_digest,
        "status": manifest.status.value,
        "summary": manifest.summary,
        "evidence": [asdict(entry) for entry in manifest.evidence],
        "candidate_digest": manifest.candidate_digest,
        "reviewed_candidate_digest": manifest.reviewed_candidate_digest,
        "payload": manifest.payload,
    }


def _evidence_from_object(value: object) -> EvidenceFile:
    raw = _require_object(value, "evidence entry")
    _expect_keys(raw, {"path", "sha256", "size_bytes"}, "evidence entry")
    return EvidenceFile(
        path=_require_string(raw["path"], "evidence path"),
        sha256=_require_string(raw["sha256"], "evidence sha256"),
        size_bytes=_require_integer(raw["size_bytes"], "evidence size_bytes"),
    )


def _manifest_envelope(kind: str, payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    digest = _canonical_digest(payload)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "digest": digest,
        "payload": payload,
    }


def _load_envelope(path: Path, expected_kind: str) -> dict[str, object]:
    raw = _load_json_object(path)
    _expect_keys(raw, {"schema_version", "kind", "digest", "payload"}, "manifest envelope")
    if raw["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ArtifactError(f"unsupported manifest schema in {path}")
    if raw["kind"] != expected_kind:
        raise ArtifactError(f"expected {expected_kind!r} in {path}, got {raw['kind']!r}")
    payload = _require_object(raw["payload"], "manifest payload")
    recorded_digest = _require_string(raw["digest"], "manifest digest")
    _validate_digest("manifest digest", recorded_digest)
    actual_digest = _canonical_digest(payload)
    if recorded_digest != actual_digest:
        raise ArtifactDigestError(f"manifest digest mismatch in {path}")
    return raw


def _require_envelope_digest(envelope: dict[str, JsonValue] | dict[str, object]) -> str:
    return _require_string(envelope["digest"], "manifest digest")


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_evidence(attempt_dir: Path, evidence: tuple[EvidenceFile, ...]) -> None:
    for entry in evidence:
        path = _resolve_member(attempt_dir, entry.path)
        if path.stat().st_size != entry.size_bytes:
            raise ArtifactDigestError(f"evidence size mismatch: {entry.path}")
        if digest_file(path) != entry.sha256:
            raise ArtifactDigestError(f"evidence digest mismatch: {entry.path}")


def _make_evidence_durable(attempt_dir: Path, evidence: tuple[EvidenceFile, ...]) -> None:
    directories: set[Path] = set()
    for entry in evidence:
        path = _resolve_member(attempt_dir, entry.path)
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directories.add(path.parent)
    for directory in directories:
        _fsync_directory(directory)


def _identity_mismatch(
    manifest: WorkerResultManifest, expectation: ResultExpectation
) -> str | None:
    fields: tuple[tuple[str, object, object], ...] = (
        ("run_id", manifest.run_id, expectation.run_id),
        ("item_id", manifest.item_id, expectation.item_id),
        ("attempt_id", manifest.attempt_id, expectation.attempt_id),
        ("task_digest", manifest.task_digest, expectation.task_digest),
        ("generation", manifest.generation, expectation.generation),
        ("input_digest", manifest.input_digest, expectation.input_digest),
    )
    for name, actual, expected in fields:
        if actual != expected:
            return f"stale result {name}: expected {expected!r}, got {actual!r}"
    return None


def _resolve_member(root: Path, relative_path: str) -> Path:
    _validate_relative_path(relative_path)
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ArtifactError(f"artifact escapes attempt directory: {relative_path}") from error
    if candidate.is_symlink():
        raise ArtifactError(f"artifact must not be a symlink: {relative_path}")
    resolved_root = root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=True)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ArtifactError(
            f"artifact resolves outside attempt directory: {relative_path}"
        ) from error
    return resolved_candidate


def _require_attempt_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ArtifactError(f"attempt mailbox is not a regular directory: {path}")


def _validate_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value in {".", ".."}
        or "\\" in value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"artifact path must be canonical and relative: {value!r}")


def _validate_identity(run_id: str, item_id: str, attempt_id: str) -> None:
    for name, value in (
        ("run_id", run_id),
        ("item_id", item_id),
        ("attempt_id", attempt_id),
    ):
        if (
            not value
            or value in {".", ".."}
            or value.strip() != value
            or any(char in value for char in ("/", "\\", "\x00"))
        ):
            raise ValueError(f"{name} must be a non-empty path-safe identity")


def _validate_digest(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_json_value(value: object, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or infinity")
        return
    if isinstance(value, list):
        for index, entry in enumerate(value):
            _validate_json_value(entry, f"{path}[{index}]")
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, entry in value.items():
            _validate_json_value(entry, f"{path}.{key}")
        return
    raise ValueError(f"{path} is not a JSON value")


def _json_object(value: object, name: str) -> dict[str, JsonValue]:
    raw = _require_object(value, name)
    _validate_json_value(raw, name)
    return cast(dict[str, JsonValue], raw)


def _expect_keys(raw: dict[str, object], expected: set[str], name: str) -> None:
    missing = expected - set(raw)
    unknown = set(raw) - expected
    if missing:
        raise ArtifactError(f"missing {name} fields: {sorted(missing)!r}")
    if unknown:
        raise ArtifactError(f"unknown {name} fields: {sorted(unknown)!r}")


def _require_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ArtifactError(f"{name} must be an object")
    return value


def _require_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ArtifactError(f"{name} must be a list")
    return value


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ArtifactError(f"{name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, name)


def _require_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactError(f"{name} must be an integer")
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"manifest is not a regular non-symlink file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactError(f"invalid JSON in {path}: {error}") from error
    return _require_object(value, str(path))


def _write_once_json(path: Path, payload: dict[str, object] | dict[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, allow_nan=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise ImmutableArtifactError(f"immutable artifact already exists: {path}") from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
