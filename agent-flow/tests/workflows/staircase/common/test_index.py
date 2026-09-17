# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic ModelingV2 catalog index fan-in."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.gitops import ControllerFileLock, ControllerLockError
from agent_flow.workflows.staircase.common.index import (
    CatalogIndexError,
    IndexDelta,
    load_catalog_index,
    render_index_deltas,
    update_catalog_index,
)

_INDEX = """# Catalog Index
# keep this hand-written header byte-for-byte

entries:
  # --- torch ---
  - path: torch/add.py
    impl: torch.add
    summary: "Elementwise add"

  # --- norm ---
  # preserve the category note
  - path: norm/rms.py
    impl: torch.ops.trtllm.rms
    summary: "RMS norm"
"""


@pytest.fixture
def catalog(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal catalog with a hand-commented index."""
    root = tmp_path / "catalog"
    (root / "torch").mkdir(parents=True)
    (root / "norm").mkdir()
    (root / "activation").mkdir()
    (root / "torch" / "add.py").write_text("def add(): ...\n", encoding="utf-8")
    (root / "norm" / "rms.py").write_text("def rms(): ...\n", encoding="utf-8")
    (root / "norm" / "new_norm.py").write_text("def new_norm(): ...\n", encoding="utf-8")
    (root / "activation" / "swiglu.py").write_text("def swiglu(): ...\n", encoding="utf-8")
    index_path = root / "index.yaml"
    index_path.write_text(_INDEX, encoding="utf-8")
    return root, index_path


def _delta(
    *,
    item_id: str,
    path: str,
    implementation: str,
    summary: str,
    expected_hash: str | None = None,
    dimensions: dict[str, str] | None = None,
) -> IndexDelta:
    """Build a delta through the untrusted mapping validation path."""
    return IndexDelta.from_mapping(
        {
            "item_id": item_id,
            "row": {
                "entry_id": Path(path).stem,
                "path": path,
                "implementation": implementation,
                "summary": summary,
            },
            "certification_cells": (
                [] if dimensions is None else [{"entry_path": path, "dimensions": dimensions}]
            ),
            "expected_row_sha256": expected_hash,
        }
    )


def test_additions_use_stable_item_order_and_preserve_comments(catalog: tuple[Path, Path]) -> None:
    root, index_path = catalog
    index = load_catalog_index(index_path, root)
    updates = render_index_deltas(
        index,
        [
            _delta(
                item_id="item-20",
                path="activation/swiglu.py",
                implementation="torch.ops.trtllm.swiglu",
                summary="SwiGLU",
            ),
            _delta(
                item_id="item-10",
                path="norm/new_norm.py",
                implementation="torch.ops.trtllm.new_norm",
                summary="New norm",
            ),
        ],
        root,
    )
    assert updates.applied_item_ids == ("item-10", "item-20")
    assert "# keep this hand-written header byte-for-byte" in updates.text
    assert "# preserve the category note" in updates.text
    assert updates.text.index("path: norm/new_norm.py") < updates.text.index("# --- activation ---")
    assert updates.text.endswith(
        '  - path: activation/swiglu.py\n    impl: torch.ops.trtllm.swiglu\n    summary: "SwiGLU"\n'
    )


def test_repeated_application_is_byte_idempotent(catalog: tuple[Path, Path]) -> None:
    root, index_path = catalog
    delta = _delta(
        item_id="item-01",
        path="norm/new_norm.py",
        implementation="torch.ops.trtllm.new_norm",
        summary="New norm",
    )
    lock = ControllerFileLock(index_path.parent / ".controller.lock", "controller")
    with lock:
        first = update_catalog_index(index_path, root, [delta], controller_lock=lock)
    first_bytes = index_path.read_bytes()
    with lock:
        second = update_catalog_index(index_path, root, [delta], controller_lock=lock)
    assert first.changed is True
    assert second.changed is False
    assert index_path.read_bytes() == first_bytes


def test_existing_summary_update_is_compare_and_swap_and_minimal(
    catalog: tuple[Path, Path],
) -> None:
    root, index_path = catalog
    before = load_catalog_index(index_path, root)
    existing = next(row for row in before.rows if row.path == "norm/rms.py")
    delta = _delta(
        item_id="item-rms",
        path=existing.path,
        implementation=existing.implementation,
        summary="RMS norm with explicit epsilon",
        expected_hash=existing.digest,
    )
    update = render_index_deltas(before, [delta], root)
    expected = _INDEX.replace('summary: "RMS norm"', 'summary: "RMS norm with explicit epsilon"')
    assert update.text == expected

    stale = _delta(
        item_id="item-rms",
        path=existing.path,
        implementation=existing.implementation,
        summary="Another summary",
        expected_hash="0" * 64,
    )
    with pytest.raises(CatalogIndexError, match="changed since"):
        render_index_deltas(before, [stale], root)


@pytest.mark.parametrize("attribute", ["path", "entry", "implementation"])
def test_fan_in_rejects_duplicate_row_claims(catalog: tuple[Path, Path], attribute: str) -> None:
    root, index_path = catalog
    first_path = "norm/new_norm.py"
    second_path = "activation/swiglu.py"
    first_impl = "torch.ops.trtllm.new_norm"
    second_impl = "torch.ops.trtllm.swiglu"
    if attribute == "path":
        second_path = first_path
    elif attribute == "entry":
        (root / "activation" / "new_norm.py").write_text("pass\n", encoding="utf-8")
        second_path = "activation/new_norm.py"
    else:
        second_impl = first_impl
    deltas = [
        _delta(
            item_id="item-1",
            path=first_path,
            implementation=first_impl,
            summary="first",
        ),
        _delta(
            item_id="item-2",
            path=second_path,
            implementation=second_impl,
            summary="second",
        ),
    ]
    with pytest.raises(CatalogIndexError, match="duplicate"):
        render_index_deltas(load_catalog_index(index_path, root), deltas, root)


def test_fan_in_rejects_duplicate_certification_cells(catalog: tuple[Path, Path]) -> None:
    root, index_path = catalog
    dimensions = {"architecture": "sm_103", "precision": "bf16", "world_size": "1"}
    # The row conflicts too, but certification conflicts are checked explicitly
    # after row-identity preflight. Use two cells in one delta to isolate it.
    mapping = {
        "item_id": "item-1",
        "row": {
            "entry_id": "new_norm",
            "path": "norm/new_norm.py",
            "implementation": "torch.ops.trtllm.new_norm",
            "summary": "new norm",
        },
        "certification_cells": [
            {"entry_path": "norm/new_norm.py", "dimensions": dimensions},
            {"entry_path": "norm/new_norm.py", "dimensions": dimensions},
        ],
        "expected_row_sha256": None,
    }
    with pytest.raises(CatalogIndexError, match="duplicate certification"):
        IndexDelta.from_mapping(mapping)

    valid = _delta(
        item_id="item-1",
        path="norm/new_norm.py",
        implementation="torch.ops.trtllm.new_norm",
        summary="new norm",
        dimensions=dimensions,
    )
    assert render_index_deltas(load_catalog_index(index_path, root), [valid], root).changed

    duplicate_across_items = _delta(
        item_id="item-2",
        path="norm/new_norm.py",
        implementation="torch.ops.trtllm.new_norm",
        summary="new norm",
        dimensions=dimensions,
    )
    with pytest.raises(CatalogIndexError, match="duplicate certification"):
        render_index_deltas(
            load_catalog_index(index_path, root), [valid, duplicate_across_items], root
        )


def test_orphan_and_unsafe_paths_are_rejected(catalog: tuple[Path, Path]) -> None:
    root, index_path = catalog
    orphan = _delta(
        item_id="item-orphan",
        path="norm/orphan.py",
        implementation="torch.ops.trtllm.orphan",
        summary="orphan",
    )
    with pytest.raises(CatalogIndexError, match="orphan wrapper"):
        render_index_deltas(load_catalog_index(index_path, root), [orphan], root)

    with pytest.raises(CatalogIndexError, match="unsafe"):
        _delta(
            item_id="item-unsafe",
            path="../escape.py",
            implementation="torch.ops.trtllm.escape",
            summary="escape",
        )


def test_existing_orphan_and_duplicate_implementation_are_rejected(
    catalog: tuple[Path, Path],
) -> None:
    root, index_path = catalog
    index_path.write_text(
        _INDEX
        + "\n  - path: activation/swiglu.py\n"
        + "    impl: torch.add\n"
        + '    summary: "duplicate impl"\n',
        encoding="utf-8",
    )
    with pytest.raises(CatalogIndexError, match="duplicate catalog implementation"):
        load_catalog_index(index_path, root)

    index_path.write_text(
        _INDEX
        + "\n  - path: norm/orphan.py\n"
        + "    impl: torch.ops.trtllm.orphan\n"
        + '    summary: "orphan"\n',
        encoding="utf-8",
    )
    with pytest.raises(CatalogIndexError, match="orphan paths"):
        load_catalog_index(index_path, root)


def test_file_update_requires_held_controller_lock(catalog: tuple[Path, Path]) -> None:
    root, index_path = catalog
    delta = _delta(
        item_id="item-1",
        path="norm/new_norm.py",
        implementation="torch.ops.trtllm.new_norm",
        summary="new norm",
    )
    lock = ControllerFileLock(index_path.parent / ".controller.lock", "controller")
    with pytest.raises(ControllerLockError, match="requires the held Git lock"):
        update_catalog_index(index_path, root, [delta], controller_lock=lock)
