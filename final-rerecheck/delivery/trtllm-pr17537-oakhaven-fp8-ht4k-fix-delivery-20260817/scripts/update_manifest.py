#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refresh the delivery file inventory and checksum list deterministically."""

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delivery-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--updated-at", required=True)
    args = parser.parse_args()

    root = args.delivery_root.resolve()
    manifest_path = root / "MANIFEST.json"
    checksums_path = root / "SHA256SUMS"
    manifest = json.loads(manifest_path.read_text())
    manifest["documentation_updated_at"] = args.updated_at

    payloads = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path not in {manifest_path, checksums_path}
        and "__pycache__" not in path.parts
    )
    manifest["files"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in payloads
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    checksum_paths = sorted([manifest_path, *payloads])
    checksums = [
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}"
        for path in checksum_paths
    ]
    checksums_path.write_text("\n".join(checksums) + "\n")


if __name__ == "__main__":
    main()
