# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hash-frozen TensorRT-LLM native artifacts for deterministic gates."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sysconfig
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, cast

NATIVE_ARTIFACT_MANIFEST_PATH = Path("/run/staircase/native-artifacts/manifest.json")


class NativeArtifactError(ValueError):
    """Raised when a native artifact bundle is not exact and immutable."""


@dataclass(frozen=True, slots=True)
class NativeArtifactEntry:
    """One immutable native file relative to the TensorRT-LLM package root."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class NativeArtifactManifest:
    """Validated native build identity and its complete artifact inventory."""

    native_build_identity: str
    audited_base_commit: str
    python_ext_suffix: str
    entries: tuple[NativeArtifactEntry, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedNativeArtifactBundle:
    """Canonical host bundle safe to mount into one gate candidate."""

    package_root: Path
    manifest_path: Path
    manifest: NativeArtifactManifest

    @property
    def top_level_extensions(self) -> tuple[NativeArtifactEntry, ...]:
        """Return top-level Python extensions in manifest order."""
        return tuple(entry for entry in self.manifest.entries if "/" not in entry.path)


def validate_native_artifact_bundle(
    *,
    package_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
    expected_native_build_identity: str,
    expected_base_commit: str,
) -> ValidatedNativeArtifactBundle:
    """Validate an exact, read-only native artifact bundle.

    The inventory contains every regular file under ``libs/`` and every
    top-level Python extension matching the active interpreter's EXT_SUFFIX.
    Other package source files are outside this bundle contract.
    """
    package = _canonical_directory(package_root, "native package root")
    if package.name != "tensorrt_llm":
        raise NativeArtifactError("native package root must be named 'tensorrt_llm'")
    manifest_file = _canonical_regular_file(manifest_path, "native artifact manifest")
    _require_read_only(manifest_file, "native artifact manifest")
    raw, observed_manifest_sha256 = _read_frozen_file(manifest_file, "native artifact manifest")
    if observed_manifest_sha256 != manifest_sha256:
        raise NativeArtifactError("native artifact manifest SHA-256 differs from task")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise NativeArtifactError("native artifact manifest is not valid UTF-8 JSON") from error
    data = _exact_object(
        value,
        {
            "native_build_identity",
            "audited_base_commit",
            "python_ext_suffix",
            "entries",
        },
        "native artifact manifest",
    )
    native_build_identity = _nonempty_string(data["native_build_identity"], "native_build_identity")
    if native_build_identity != expected_native_build_identity:
        raise NativeArtifactError("native build identity differs from certified product identity")
    audited_base_commit = _lower_hex(data["audited_base_commit"], 40, "audited_base_commit")
    if audited_base_commit != expected_base_commit:
        raise NativeArtifactError("native artifact audited base commit differs from task")
    python_ext_suffix = _nonempty_string(data["python_ext_suffix"], "python_ext_suffix")
    active_ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(active_ext_suffix, str) or not active_ext_suffix:
        raise NativeArtifactError("active Python interpreter has no EXT_SUFFIX")
    if python_ext_suffix != active_ext_suffix:
        raise NativeArtifactError("native artifact Python EXT_SUFFIX differs from controller")

    raw_entries = data["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise NativeArtifactError("native artifact entries must be a non-empty list")
    entries: list[NativeArtifactEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _exact_object(
            raw_entry,
            {"path", "size_bytes", "sha256"},
            f"native artifact entries[{index}]",
        )
        relative = _artifact_path(entry["path"], python_ext_suffix)
        size_bytes = entry["size_bytes"]
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise NativeArtifactError(
                f"native artifact entries[{index}].size_bytes must be non-negative"
            )
        digest = _lower_hex(entry["sha256"], 64, f"entries[{index}].sha256")
        entries.append(NativeArtifactEntry(relative, size_bytes, digest))
    paths = [entry.path for entry in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise NativeArtifactError("native artifact entries must have sorted unique paths")

    actual = _enumerate_artifacts(package, python_ext_suffix)
    if set(paths) != set(actual):
        missing = sorted(set(actual).difference(paths))
        unlisted = sorted(set(paths).difference(actual))
        raise NativeArtifactError(
            "native artifact inventory differs from package root: "
            f"missing_entries={missing!r}, absent_files={unlisted!r}"
        )
    for entry in entries:
        artifact = _canonical_regular_file(package / entry.path, f"native artifact {entry.path!r}")
        if artifact.parent != package and package not in artifact.parents:
            raise NativeArtifactError("native artifact resolves outside package root")
        _require_read_only(artifact, f"native artifact {entry.path!r}")
        content, observed_digest = _read_frozen_file(artifact, f"native artifact {entry.path!r}")
        if len(content) != entry.size_bytes or observed_digest != entry.sha256:
            raise NativeArtifactError(
                f"native artifact content differs from manifest: {entry.path}"
            )
    return ValidatedNativeArtifactBundle(
        package,
        manifest_file,
        NativeArtifactManifest(
            native_build_identity,
            audited_base_commit,
            python_ext_suffix,
            tuple(entries),
            observed_manifest_sha256,
        ),
    )


def native_artifact_runtime(bundle: ValidatedNativeArtifactBundle) -> dict[str, str]:
    """Return the public gate-worker binding for a validated bundle."""
    return {
        "manifest_path": NATIVE_ARTIFACT_MANIFEST_PATH.as_posix(),
        "manifest_sha256": bundle.manifest.sha256,
        "native_build_identity": bundle.manifest.native_build_identity,
        "audited_base_commit": bundle.manifest.audited_base_commit,
        "python_ext_suffix": bundle.manifest.python_ext_suffix,
    }


def _enumerate_artifacts(package_root: Path, python_ext_suffix: str) -> Mapping[str, Path]:
    artifacts: dict[str, Path] = {}
    try:
        top_level = tuple(package_root.iterdir())
    except OSError as error:
        raise NativeArtifactError("native package root changed while enumerating") from error
    for path in top_level:
        if path.name.endswith(python_ext_suffix):
            _record_artifact(artifacts, package_root, path)
    libs = package_root / "libs"
    if libs.is_symlink() or not libs.is_dir():
        raise NativeArtifactError("native package root must contain a regular libs directory")
    try:
        library_entries = tuple(libs.rglob("*"))
    except OSError as error:
        raise NativeArtifactError("native libs directory changed while enumerating") from error
    for path in library_entries:
        metadata = path.lstat()
        relative = path.relative_to(package_root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise NativeArtifactError(f"native artifact bundle contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise NativeArtifactError(
                f"native artifact bundle contains a non-regular file: {relative}"
            )
        _record_artifact(artifacts, package_root, path)
    if not artifacts:
        raise NativeArtifactError("native artifact bundle has no artifacts")
    return artifacts


def _record_artifact(artifacts: dict[str, Path], package_root: Path, path: Path) -> None:
    relative = path.relative_to(package_root).as_posix()
    _canonical_regular_file(path, f"native artifact {relative!r}")
    if relative in artifacts:
        raise NativeArtifactError(f"duplicate native artifact path: {relative}")
    artifacts[relative] = path


def _artifact_path(value: object, python_ext_suffix: str) -> str:
    path = _nonempty_string(value, "native artifact path")
    relative = PurePosixPath(path)
    if (
        relative.is_absolute()
        or relative.as_posix() != path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise NativeArtifactError("native artifact path must be normalized and relative")
    is_top_level_extension = len(relative.parts) == 1 and path.endswith(python_ext_suffix)
    is_library = len(relative.parts) >= 2 and relative.parts[0] == "libs"
    if not (is_top_level_extension or is_library):
        raise NativeArtifactError(
            "native artifacts are limited to top-level Python extensions and libs/ files"
        )
    return path


def _canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise NativeArtifactError(f"{label} must be an absolute regular directory")
    canonical = path.resolve(strict=True)
    if canonical != path:
        raise NativeArtifactError(f"{label} must be canonical")
    return canonical


def _canonical_regular_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise NativeArtifactError(f"{label} must be an absolute regular file")
    canonical = path.resolve(strict=True)
    if canonical != path:
        raise NativeArtifactError(f"{label} must be canonical")
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise NativeArtifactError(f"{label} must be regular with exactly one hard link")
    return canonical


def _require_read_only(path: Path, label: str) -> None:
    if path.stat(follow_symlinks=False).st_mode & 0o222:
        raise NativeArtifactError(f"{label} must not have writable mode bits")


def _read_frozen_file(path: Path, label: str) -> tuple[bytes, str]:
    before = path.stat(follow_symlinks=False)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise NativeArtifactError(f"{label} changed while opening") from error
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if _stat_identity(opened) != _stat_identity(before):
            raise NativeArtifactError(f"{label} changed while opening")
        content = stream.read()
        if _stat_identity(os.fstat(stream.fileno())) != _stat_identity(before):
            raise NativeArtifactError(f"{label} changed while reading")
    if _stat_identity(path.stat(follow_symlinks=False)) != _stat_identity(before):
        raise NativeArtifactError(f"{label} changed while validating")
    return content, hashlib.sha256(content).hexdigest()


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


def _exact_object(value: object, keys: set[str], label: str) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or not all(isinstance(key, str) for key in value)
    ):
        raise NativeArtifactError(f"{label} must contain exactly {sorted(keys)!r}")
    return cast(dict[str, object], value)


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise NativeArtifactError(f"{label} must be a non-empty single-line string")
    return value


def _lower_hex(value: object, length: int, label: str) -> str:
    text = _nonempty_string(value, label)
    if len(text) != length or any(character not in "0123456789abcdef" for character in text):
        raise NativeArtifactError(f"{label} must be {length} lowercase hexadecimal characters")
    return text


__all__ = [
    "NATIVE_ARTIFACT_MANIFEST_PATH",
    "NativeArtifactEntry",
    "NativeArtifactError",
    "NativeArtifactManifest",
    "ValidatedNativeArtifactBundle",
    "native_artifact_runtime",
    "validate_native_artifact_bundle",
]
