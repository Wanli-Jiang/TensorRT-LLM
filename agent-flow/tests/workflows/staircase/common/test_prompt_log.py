# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the immutable Staircase prompt ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.prompt_log import (
    PromptLogError,
    load_prompt_record,
    record_prompt_exchange,
)


def _record(root: Path, *, response: str = "done"):
    return record_prompt_exchange(
        root,
        run_id="run-1",
        role="smith-coder",
        prompt_id="prompt-1",
        task_digest="a" * 64,
        input_digest="b" * 64,
        prompt="implement one entry",
        response=response,
    )


def test_record_round_trip_and_idempotent_replay(tmp_path: Path) -> None:
    first = _record(tmp_path)
    second = _record(tmp_path)
    assert first == second
    assert load_prompt_record(tmp_path, "prompt-1") == first


def test_prompt_id_cannot_be_reused_for_different_content(tmp_path: Path) -> None:
    _record(tmp_path)
    with pytest.raises(PromptLogError, match="different content"):
        _record(tmp_path, response="different")


def test_digest_verification_detects_tampering(tmp_path: Path) -> None:
    record = _record(tmp_path)
    Path(record.response_path).write_text("tampered", encoding="utf-8")
    with pytest.raises(PromptLogError, match="recorded digest"):
        load_prompt_record(tmp_path, "prompt-1")


def test_prompt_id_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="safe path component"):
        record_prompt_exchange(
            tmp_path,
            run_id="run-1",
            role="coder",
            prompt_id="../escape",
            task_digest="a" * 64,
            input_digest="b" * 64,
            prompt="prompt",
            response="response",
        )
