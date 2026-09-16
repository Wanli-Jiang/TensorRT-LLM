# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, comment-preserving ModelingV2 catalog index integration.

The ModelingV2 index is a hand-maintained document. This module uses YAML
only for semantic validation; it never serializes the document with a generic
YAML dumper. Existing bytes are returned unchanged for an idempotent update,
existing row edits replace only their three scalar lines, and new rows are
inserted at the end of their category section.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from .gitops import ControllerFileLock


class CatalogIndexError(ValueError):
    """Raised when an index or proposed delta violates its contract."""


_SAFE_COMPONENT = re.compile(r"[a-z][a-z0-9_]*")
_SAFE_IMPLEMENTATION = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_SAFE_ITEM_ID = re.compile(r"[a-z0-9][a-z0-9_.-]*")
_HASH = re.compile(r"[0-9a-f]{64}")
_ROW_START = re.compile(r"^  - path:\s*(.+?)\s*$")
_ROW_FIELD = {
    "path": re.compile(r"^(  - path:\s*).+?(\r?\n)?$"),
    "impl": re.compile(r"^(    impl:\s*).+?(\r?\n)?$"),
    "summary": re.compile(r"^(    summary:\s*).+?(\r?\n)?$"),
}


def _expect_exact_keys(mapping: Mapping[str, object], expected: set[str], *, context: str) -> None:
    """Reject missing and unknown keys in a typed manifest object."""
    actual = set(mapping)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise CatalogIndexError(f"{context} keys mismatch; missing={missing}, unknown={unknown}")


def _expect_string(value: object, *, context: str) -> str:
    """Return a non-empty, single-line string or fail."""
    if not isinstance(value, str) or not value or "\n" in value or "\r" in value:
        raise CatalogIndexError(f"{context} must be a non-empty single-line string")
    return value


def _safe_catalog_path(value: object) -> str:
    """Validate one canonical path relative to ``catalog/``."""
    path = _expect_string(value, context="catalog row path")
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or path != pure_path.as_posix()
        or len(pure_path.parts) != 2
        or pure_path.suffix != ".py"
        or pure_path.name == "__init__.py"
        or not all(_SAFE_COMPONENT.fullmatch(part) for part in (pure_path.parts[0], pure_path.stem))
    ):
        raise CatalogIndexError(f"unsafe or non-canonical catalog path: {path!r}")
    return path


@dataclass(frozen=True)
class CatalogIndexRow:
    """One semantic row in ``catalog/index.yaml``."""

    entry_id: str
    path: str
    implementation: str
    summary: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> CatalogIndexRow:
        """Validate and construct a row from an untrusted result manifest."""
        _expect_exact_keys(
            mapping,
            {"entry_id", "path", "implementation", "summary"},
            context="catalog row",
        )
        path = _safe_catalog_path(mapping["path"])
        entry_id = _expect_string(mapping["entry_id"], context="catalog row entry_id")
        if entry_id != PurePosixPath(path).stem or not _SAFE_COMPONENT.fullmatch(entry_id):
            raise CatalogIndexError("entry_id must exactly match the catalog path stem")
        implementation = _expect_string(
            mapping["implementation"], context="catalog row implementation"
        )
        if not _SAFE_IMPLEMENTATION.fullmatch(implementation):
            raise CatalogIndexError(f"unsafe implementation name: {implementation!r}")
        summary = _expect_string(mapping["summary"], context="catalog row summary")
        return cls(entry_id, path, implementation, summary)

    @property
    def digest(self) -> str:
        """Return a stable semantic hash used for compare-and-swap updates."""
        payload = json.dumps(
            {
                "entry_id": self.entry_id,
                "implementation": self.implementation,
                "path": self.path,
                "summary": self.summary,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CertificationCell:
    """An opaque but canonicalized certification claim for conflict checks."""

    entry_path: str
    dimensions: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> CertificationCell:
        """Validate one claim cell from an untrusted result manifest."""
        _expect_exact_keys(mapping, {"entry_path", "dimensions"}, context="certification cell")
        entry_path = _safe_catalog_path(mapping["entry_path"])
        raw_dimensions = mapping["dimensions"]
        if not isinstance(raw_dimensions, Mapping) or not raw_dimensions:
            raise CatalogIndexError("certification cell dimensions must be a non-empty mapping")
        dimensions: list[tuple[str, str]] = []
        for raw_key, raw_value in raw_dimensions.items():
            key = _expect_string(raw_key, context="certification dimension name")
            value = _expect_string(raw_value, context=f"certification dimension {key}")
            if not _SAFE_COMPONENT.fullmatch(key):
                raise CatalogIndexError(f"unsafe certification dimension name: {key!r}")
            dimensions.append((key, value))
        return cls(entry_path, tuple(sorted(dimensions)))


@dataclass(frozen=True)
class IndexDelta:
    """Worker proposal consumed only by serial controller fan-in."""

    item_id: str
    row: CatalogIndexRow
    certification_cells: tuple[CertificationCell, ...]
    expected_row_sha256: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> IndexDelta:
        """Validate a strict typed delta mapping from a worker result."""
        _expect_exact_keys(
            mapping,
            {"item_id", "row", "certification_cells", "expected_row_sha256"},
            context="index delta",
        )
        item_id = _expect_string(mapping["item_id"], context="index delta item_id")
        if not _SAFE_ITEM_ID.fullmatch(item_id):
            raise CatalogIndexError(f"unsafe item_id: {item_id!r}")
        raw_row = mapping["row"]
        if not isinstance(raw_row, Mapping):
            raise CatalogIndexError("index delta row must be a mapping")
        raw_cells = mapping["certification_cells"]
        if not isinstance(raw_cells, Sequence) or isinstance(raw_cells, (str, bytes)):
            raise CatalogIndexError("certification_cells must be a sequence")
        cells: list[CertificationCell] = []
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, Mapping):
                raise CatalogIndexError("each certification cell must be a mapping")
            cells.append(CertificationCell.from_mapping(raw_cell))
        expected_hash = mapping["expected_row_sha256"]
        if expected_hash is not None and (
            not isinstance(expected_hash, str) or not _HASH.fullmatch(expected_hash)
        ):
            raise CatalogIndexError("expected_row_sha256 must be null or 64 lowercase hex digits")
        row = CatalogIndexRow.from_mapping(raw_row)
        if any(cell.entry_path != row.path for cell in cells):
            raise CatalogIndexError("every certification cell must belong to the delta row path")
        if len(set(cells)) != len(cells):
            raise CatalogIndexError("index delta contains duplicate certification cells")
        return cls(item_id, row, tuple(cells), expected_hash)


@dataclass(frozen=True)
class CatalogIndex:
    """Validated semantic view paired with the original text."""

    text: str
    rows: tuple[CatalogIndexRow, ...]


@dataclass(frozen=True)
class IndexUpdate:
    """Result of a deterministic catalog index integration."""

    changed: bool
    text: str
    applied_item_ids: tuple[str, ...]


def _rows_from_yaml(text: str) -> tuple[CatalogIndexRow, ...]:
    """Parse index semantics without using YAML for output."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise CatalogIndexError(f"catalog index is not valid YAML: {error}") from error
    if not isinstance(document, Mapping) or set(document) != {"entries"}:
        raise CatalogIndexError("catalog index root must contain only 'entries'")
    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise CatalogIndexError("catalog index entries must be a list")
    rows: list[CatalogIndexRow] = []
    for position, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"path", "impl", "summary"}:
            raise CatalogIndexError(f"catalog index entry {position} has an invalid shape")
        path = _safe_catalog_path(raw_entry["path"])
        row_mapping: dict[str, object] = {
            "entry_id": PurePosixPath(path).stem,
            "path": path,
            "implementation": raw_entry["impl"],
            "summary": raw_entry["summary"],
        }
        rows.append(CatalogIndexRow.from_mapping(row_mapping))
    return tuple(rows)


def _validate_unique_rows(rows: Sequence[CatalogIndexRow]) -> None:
    """Reject duplicate entry, path, and implementation identities."""
    for attribute in ("entry_id", "path", "implementation"):
        values = [getattr(row, attribute) for row in rows]
        duplicates = sorted(value for value in set(values) if values.count(value) > 1)
        if duplicates:
            raise CatalogIndexError(f"duplicate catalog {attribute} values: {duplicates}")


def load_catalog_index(index_path: str | Path, catalog_root: str | Path) -> CatalogIndex:
    """Load and validate index semantics, uniqueness, and wrapper existence."""
    path = Path(index_path).expanduser().resolve()
    root = Path(catalog_root).expanduser().resolve()
    if path.parent != root:
        raise CatalogIndexError("index_path must be directly inside the explicit catalog_root")
    text = path.read_bytes().decode("utf-8")
    rows = _rows_from_yaml(text)
    _validate_unique_rows(rows)
    orphaned: list[str] = []
    for row in rows:
        wrapper = (root / row.path).resolve()
        try:
            wrapper.relative_to(root)
        except ValueError:
            orphaned.append(row.path)
            continue
        if not wrapper.is_file():
            orphaned.append(row.path)
    if orphaned:
        raise CatalogIndexError(f"catalog index contains orphan paths: {sorted(orphaned)}")
    return CatalogIndex(text, rows)


def _quoted_scalar(value: str) -> str:
    """Render a deterministic double-quoted YAML scalar."""
    return json.dumps(value, ensure_ascii=False)


def _render_row(row: CatalogIndexRow, newline: str) -> str:
    """Render one canonical row without serializing the surrounding YAML."""
    return (
        f"  - path: {row.path}{newline}"
        f"    impl: {row.implementation}{newline}"
        f"    summary: {_quoted_scalar(row.summary)}{newline}"
    )


def _replace_existing_row(text: str, old: CatalogIndexRow, new: CatalogIndexRow) -> str:
    """Replace only scalar lines belonging to one existing row."""
    lines = text.splitlines(keepends=True)
    row_start: int | None = None
    row_end = len(lines)
    for index, line in enumerate(lines):
        match = _ROW_START.match(line.rstrip("\r\n"))
        if match:
            try:
                parsed_path = yaml.safe_load(f"value: {match.group(1)}")["value"]
            except (TypeError, yaml.YAMLError) as error:
                raise CatalogIndexError(f"cannot locate canonical row for {old.path}") from error
            if parsed_path == old.path:
                row_start = index
                continue
            if row_start is not None:
                row_end = index
                break
    if row_start is None:
        raise CatalogIndexError(f"semantic row has no textual row block: {old.path}")
    expected = {"path": old.path, "impl": old.implementation, "summary": old.summary}
    replacement = {"path": new.path, "impl": new.implementation, "summary": new.summary}
    positions: dict[str, int] = {}
    for index in range(row_start, row_end):
        for field, pattern in _ROW_FIELD.items():
            match = pattern.match(lines[index])
            if not match:
                continue
            scalar = lines[index][len(match.group(1)) :].rstrip("\r\n")
            try:
                parsed = yaml.safe_load(f"value: {scalar}")["value"]
            except (TypeError, yaml.YAMLError) as error:
                raise CatalogIndexError(f"cannot parse {field} line for {old.path}") from error
            if parsed != expected[field] or field in positions:
                raise CatalogIndexError(f"non-canonical {field} line for {old.path}")
            positions[field] = index
    if set(positions) != set(_ROW_FIELD):
        raise CatalogIndexError(f"row lacks canonical path/impl/summary lines: {old.path}")
    for field, index in positions.items():
        prefix = _ROW_FIELD[field].match(lines[index])
        assert prefix is not None
        newline = (
            "\r\n" if lines[index].endswith("\r\n") else "\n" if lines[index].endswith("\n") else ""
        )
        value = replacement[field]
        rendered = _quoted_scalar(value) if field == "summary" else value
        lines[index] = f"{prefix.group(1)}{rendered}{newline}"
    return "".join(lines)


def _append_row(text: str, row: CatalogIndexRow) -> str:
    """Append a new row to its existing category section, or create one."""
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    category = PurePosixPath(row.path).parts[0]
    header = re.compile(rf"^  #.*\b{re.escape(category)}\b")
    header_index = next(
        (index for index, line in enumerate(lines) if header.match(line.rstrip("\r\n"))), None
    )
    block = _render_row(row, newline)
    if header_index is None:
        separator = "" if text.endswith(("\n", "\r")) else newline
        return text + separator + newline + f"  # --- {category} ---{newline}" + block

    insertion = len(lines)
    for index in range(header_index + 1, len(lines)):
        if lines[index].startswith("  #") and re.search(r"[-─]{3,}", lines[index]):
            insertion = index
            break
    while insertion > header_index + 1 and not lines[insertion - 1].strip():
        insertion -= 1
    prefix = "" if insertion == header_index + 1 else newline
    lines.insert(insertion, prefix + block)
    return "".join(lines)


def render_index_deltas(
    index: CatalogIndex,
    deltas: Sequence[IndexDelta],
    catalog_root: str | Path,
) -> IndexUpdate:
    """Validate and deterministically render a serial batch of typed deltas."""
    root = Path(catalog_root).expanduser().resolve()
    ordered = tuple(sorted(deltas, key=lambda delta: delta.item_id))
    if len({delta.item_id for delta in ordered}) != len(ordered):
        raise CatalogIndexError("fan-in contains duplicate item_id values")
    cells = [cell for delta in ordered for cell in delta.certification_cells]
    duplicates = sorted(
        {cell for cell in cells if cells.count(cell) > 1},
        key=lambda cell: (cell.entry_path, cell.dimensions),
    )
    if duplicates:
        raise CatalogIndexError(f"fan-in contains duplicate certification cells: {duplicates}")
    for attribute in ("entry_id", "path", "implementation"):
        values = [getattr(delta.row, attribute) for delta in ordered]
        duplicates = sorted(value for value in set(values) if values.count(value) > 1)
        if duplicates:
            raise CatalogIndexError(f"fan-in contains duplicate {attribute} claims: {duplicates}")
    text = index.text
    rows = list(index.rows)
    for delta in ordered:
        wrapper = (root / delta.row.path).resolve()
        try:
            wrapper.relative_to(root)
        except ValueError as error:
            raise CatalogIndexError(
                f"catalog path escapes catalog root: {delta.row.path}"
            ) from error
        if not wrapper.is_file():
            raise CatalogIndexError(f"delta points at an orphan wrapper: {delta.row.path}")
        by_path = {row.path: row for row in rows}
        by_entry = {row.entry_id: row for row in rows}
        by_impl = {row.implementation: row for row in rows}
        existing = by_path.get(delta.row.path)
        entry_conflict = by_entry.get(delta.row.entry_id)
        impl_conflict = by_impl.get(delta.row.implementation)
        if entry_conflict is not None and entry_conflict.path != delta.row.path:
            raise CatalogIndexError(
                f"entry_id {delta.row.entry_id!r} already belongs to {entry_conflict.path}"
            )
        if impl_conflict is not None and impl_conflict.path != delta.row.path:
            raise CatalogIndexError(
                f"implementation {delta.row.implementation!r} already belongs to {impl_conflict.path}"
            )
        if existing == delta.row:
            continue
        if existing is None:
            if delta.expected_row_sha256 is not None:
                raise CatalogIndexError("a new row cannot declare expected_row_sha256")
            text = _append_row(text, delta.row)
            category = PurePosixPath(delta.row.path).parts[0]
            category_positions = [
                position
                for position, row in enumerate(rows)
                if PurePosixPath(row.path).parts[0] == category
            ]
            insertion = category_positions[-1] + 1 if category_positions else len(rows)
            rows.insert(insertion, delta.row)
            continue
        if delta.expected_row_sha256 != existing.digest:
            raise CatalogIndexError(
                f"row {existing.path} changed since the worker snapshot or lacks its expected hash"
            )
        if (
            existing.entry_id != delta.row.entry_id
            or existing.implementation != delta.row.implementation
        ):
            raise CatalogIndexError("existing row updates may change only the summary")
        text = _replace_existing_row(text, existing, delta.row)
        rows[rows.index(existing)] = delta.row

    rendered_rows = _rows_from_yaml(text)
    _validate_unique_rows(rendered_rows)
    if tuple(rows) != rendered_rows:
        raise CatalogIndexError(
            "rendered catalog index changed row ordering or semantics unexpectedly"
        )
    return IndexUpdate(text != index.text, text, tuple(delta.item_id for delta in ordered))


def update_catalog_index(
    index_path: str | Path,
    catalog_root: str | Path,
    deltas: Sequence[IndexDelta],
    *,
    controller_lock: ControllerFileLock,
) -> IndexUpdate:
    """Atomically update the catalog index under the held controller Git lock."""
    controller_lock.assert_held()
    path = Path(index_path).expanduser().resolve()
    index = load_catalog_index(path, catalog_root)
    update = render_index_deltas(index, deltas, catalog_root)
    if not update.changed:
        return update
    encoded = update.text.encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, path.stat().st_mode & 0o777)
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return update
