# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for exact Staircase native artifact bundle validation."""

from __future__ import annotations

import hashlib
import json
import os
import sysconfig
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.native_artifacts import (
    NativeArtifactError,
    validate_native_artifact_bundle,
)

_BUILD_IDENTITY = "local-dev-test"
_BASE_COMMIT = "a" * 40


def _bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    package = tmp_path / "tensorrt_llm"
    libraries = package / "libs"
    libraries.mkdir(parents=True)
    (package / "__init__.py").write_text("# candidate package\n", encoding="utf-8")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert isinstance(ext_suffix, str)
    extension = package / f"bindings{ext_suffix}"
    library = libraries / "libth_common.so"
    extension.write_bytes(b"extension")
    library.write_bytes(b"library")
    entries = []
    for path in sorted(
        (extension, library), key=lambda value: value.relative_to(package).as_posix()
    ):
        content = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(package).as_posix(),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        path.chmod(0o444)
    manifest = tmp_path / "native-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "native_build_identity": _BUILD_IDENTITY,
                "audited_base_commit": _BASE_COMMIT,
                "python_ext_suffix": ext_suffix,
                "entries": entries,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    manifest.chmod(0o444)
    return package, manifest, manifest_sha256


def test_bundle_validates_complete_read_only_inventory(tmp_path: Path) -> None:
    package, manifest, digest = _bundle(tmp_path)

    bundle = validate_native_artifact_bundle(
        package_root=package,
        manifest_path=manifest,
        manifest_sha256=digest,
        expected_native_build_identity=_BUILD_IDENTITY,
        expected_base_commit=_BASE_COMMIT,
    )

    assert bundle.manifest.sha256 == digest
    assert [entry.path for entry in bundle.top_level_extensions] == [
        f"bindings{sysconfig.get_config_var('EXT_SUFFIX')}"
    ]
    assert [entry.path for entry in bundle.manifest.entries][-1] == "libs/libth_common.so"


def test_bundle_rejects_unlisted_writable_and_hardlinked_files(tmp_path: Path) -> None:
    package, manifest, digest = _bundle(tmp_path)
    unlisted = package / "libs" / "unlisted.so"
    unlisted.write_bytes(b"unlisted")
    unlisted.chmod(0o444)
    with pytest.raises(NativeArtifactError, match="inventory differs"):
        validate_native_artifact_bundle(
            package_root=package,
            manifest_path=manifest,
            manifest_sha256=digest,
            expected_native_build_identity=_BUILD_IDENTITY,
            expected_base_commit=_BASE_COMMIT,
        )

    unlisted.unlink()
    library = package / "libs" / "libth_common.so"
    library.chmod(0o644)
    with pytest.raises(NativeArtifactError, match="writable mode"):
        validate_native_artifact_bundle(
            package_root=package,
            manifest_path=manifest,
            manifest_sha256=digest,
            expected_native_build_identity=_BUILD_IDENTITY,
            expected_base_commit=_BASE_COMMIT,
        )

    library.chmod(0o444)
    hardlink = tmp_path / "hardlink.so"
    os.link(library, hardlink)
    with pytest.raises(NativeArtifactError, match="exactly one hard link"):
        validate_native_artifact_bundle(
            package_root=package,
            manifest_path=manifest,
            manifest_sha256=digest,
            expected_native_build_identity=_BUILD_IDENTITY,
            expected_base_commit=_BASE_COMMIT,
        )


def test_bundle_rejects_symlink_under_libraries(tmp_path: Path) -> None:
    package, manifest, digest = _bundle(tmp_path)
    (package / "libs" / "alias.so").symlink_to("libth_common.so")

    with pytest.raises(NativeArtifactError, match="symlink"):
        validate_native_artifact_bundle(
            package_root=package,
            manifest_path=manifest,
            manifest_sha256=digest,
            expected_native_build_identity=_BUILD_IDENTITY,
            expected_base_commit=_BASE_COMMIT,
        )
