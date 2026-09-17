# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only prompt/response ledger for isolated Staircase roles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


class PromptLogError(RuntimeError):
    """Raised when an immutable prompt record conflicts with durable state."""


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class PromptRecord:
    """One immutable role invocation and response record."""

    run_id: str
    role: str
    prompt_id: str
    task_digest: str
    input_digest: str
    prompt_digest: str
    response_digest: str
    created_at: str
    prompt_path: str
    response_path: str


def sha256_text(text: str) -> str:
    """Return the lowercase SHA-256 digest of UTF-8 ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def record_prompt_exchange(
    ledger_root: Path,
    *,
    run_id: str,
    role: str,
    prompt_id: str,
    task_digest: str,
    input_digest: str,
    prompt: str,
    response: str,
) -> PromptRecord:
    """Persist one immutable prompt exchange and return its typed record.

    Replaying the exact same record is idempotent. Reusing ``prompt_id`` for
    different bytes fails closed so a resumed controller cannot rewrite agent
    history.
    """
    if not all(value and value.strip() for value in (run_id, role, prompt_id, task_digest)):
        raise ValueError("run_id, role, prompt_id, and task_digest must be non-empty")
    if not _SAFE_ID.fullmatch(prompt_id):
        raise ValueError("prompt_id must be a safe path component")
    record_dir = ledger_root / prompt_id
    prompt_path = record_dir / "prompt.md"
    response_path = record_dir / "response.md"
    record_path = record_dir / "record.json"
    record = PromptRecord(
        run_id=run_id,
        role=role,
        prompt_id=prompt_id,
        task_digest=task_digest,
        input_digest=input_digest,
        prompt_digest=sha256_text(prompt),
        response_digest=sha256_text(response),
        created_at=datetime.now(timezone.utc).isoformat(),
        prompt_path=str(prompt_path),
        response_path=str(response_path),
    )
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        immutable_fields = (
            "run_id",
            "role",
            "prompt_id",
            "task_digest",
            "input_digest",
            "prompt_digest",
            "response_digest",
            "prompt_path",
            "response_path",
        )
        if any(existing.get(field) != getattr(record, field) for field in immutable_fields):
            raise PromptLogError(
                f"prompt record {prompt_id!r} already exists with different content"
            )
        return PromptRecord(**existing)

    _atomic_write(prompt_path, prompt)
    _atomic_write(response_path, response)
    _atomic_write(record_path, json.dumps(asdict(record), indent=2, sort_keys=True) + "\n")
    return record


def load_prompt_record(ledger_root: Path, prompt_id: str) -> PromptRecord:
    """Load and verify one durable prompt record."""
    if not _SAFE_ID.fullmatch(prompt_id):
        raise ValueError("prompt_id must be a safe path component")
    record_path = ledger_root / prompt_id / "record.json"
    data = json.loads(record_path.read_text(encoding="utf-8"))
    record = PromptRecord(**data)
    prompt = Path(record.prompt_path).read_text(encoding="utf-8")
    response = Path(record.response_path).read_text(encoding="utf-8")
    if sha256_text(prompt) != record.prompt_digest:
        raise PromptLogError(f"prompt bytes for {prompt_id!r} do not match the recorded digest")
    if sha256_text(response) != record.response_digest:
        raise PromptLogError(f"response bytes for {prompt_id!r} do not match the recorded digest")
    return record
