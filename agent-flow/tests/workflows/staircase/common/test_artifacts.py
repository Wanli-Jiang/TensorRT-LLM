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

"""Tests for Staircase immutable worker mailbox artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import state
from agent_flow.workflows.staircase.common import artifacts

TASK_DIGEST = "a" * 64
CANDIDATE_DIGEST = "b" * 64


def _publish_input(attempt_dir: Path, *, generation: int = 1) -> str:
    attempt_dir.mkdir(parents=True)
    manifest = artifacts.WorkerInputManifest(
        run_id="run-1",
        item_id="item-1",
        attempt_id="attempt-1",
        task_digest=TASK_DIGEST,
        generation=generation,
        role=state.Role.CODER,
        profile=state.DomainProfile.SMITH,
        worktree="/shared/worktrees/item-1",
        allowed_paths=("tensorrt_llm/_torch/modeling_v2/catalog/entry.py",),
        payload={"command": ["pytest", "-q"], "limit": 2},
    )
    return artifacts.write_input_manifest(attempt_dir / artifacts.INPUT_FILENAME, manifest)


def _result(
    attempt_dir: Path, input_digest: str, *, generation: int = 1
) -> artifacts.WorkerResultManifest:
    evidence_path = attempt_dir / "evidence" / "gate.txt"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text("passed\n", encoding="utf-8")
    evidence = artifacts.describe_evidence(attempt_dir, "evidence/gate.txt")
    return artifacts.WorkerResultManifest(
        run_id="run-1",
        item_id="item-1",
        attempt_id="attempt-1",
        task_digest=TASK_DIGEST,
        generation=generation,
        input_digest=input_digest,
        status=artifacts.WorkerResultStatus.SUCCEEDED,
        summary="catalog entry and focused gate passed",
        evidence=(evidence,),
        candidate_digest=CANDIDATE_DIGEST,
        payload={"tests": 1},
    )


def _expectation(input_digest: str, *, generation: int = 1) -> artifacts.ResultExpectation:
    return artifacts.ResultExpectation(
        run_id="run-1",
        item_id="item-1",
        attempt_id="attempt-1",
        task_digest=TASK_DIGEST,
        generation=generation,
        input_digest=input_digest,
    )


def test_input_is_immutable_and_content_addressed(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    input_digest = _publish_input(attempt_dir)
    loaded, loaded_digest = artifacts.load_input_manifest(attempt_dir / artifacts.INPUT_FILENAME)
    assert loaded_digest == input_digest
    assert loaded.role is state.Role.CODER

    with pytest.raises(artifacts.ImmutableArtifactError, match="already exists"):
        artifacts.write_input_manifest(attempt_dir / artifacts.INPUT_FILENAME, loaded)


def test_tampered_input_digest_is_rejected(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    _publish_input(attempt_dir)
    input_path = attempt_dir / artifacts.INPUT_FILENAME
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    payload["payload"]["worktree"] = "/tampered"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(artifacts.ArtifactDigestError, match="digest mismatch"):
        artifacts.load_input_manifest(input_path)


def test_result_requires_complete_written_last(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    input_digest = _publish_input(attempt_dir)
    result = _result(attempt_dir, input_digest)

    with pytest.raises(artifacts.IncompleteResultError, match="no durable COMPLETE"):
        artifacts.load_result_manifest(attempt_dir)

    result_digest = artifacts.publish_result(attempt_dir, result)
    loaded, loaded_digest = artifacts.load_result_manifest(attempt_dir)
    assert loaded == result
    assert loaded_digest == result_digest
    assert (attempt_dir / artifacts.RESULT_FILENAME).stat().st_mtime_ns <= (
        attempt_dir / artifacts.COMPLETE_FILENAME
    ).stat().st_mtime_ns


def test_result_identity_must_match_immutable_input(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    input_digest = _publish_input(attempt_dir)
    result = _result(attempt_dir, input_digest)
    mismatched = artifacts.WorkerResultManifest(
        run_id="another-run",
        item_id=result.item_id,
        attempt_id=result.attempt_id,
        task_digest=result.task_digest,
        generation=result.generation,
        input_digest=result.input_digest,
        status=result.status,
        summary=result.summary,
        evidence=result.evidence,
        candidate_digest=result.candidate_digest,
        payload=result.payload,
    )
    with pytest.raises(artifacts.ArtifactError, match="identity does not match"):
        artifacts.publish_result(attempt_dir, mismatched)
    assert not (attempt_dir / artifacts.RESULT_FILENAME).exists()
    assert not (attempt_dir / artifacts.COMPLETE_FILENAME).exists()


def test_evidence_tampering_is_rejected_after_publication(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    input_digest = _publish_input(attempt_dir)
    result = _result(attempt_dir, input_digest)
    artifacts.publish_result(attempt_dir, result)
    (attempt_dir / "evidence" / "gate.txt").write_text("failed\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactDigestError, match="digest mismatch"):
        artifacts.load_result_manifest(attempt_dir)


def test_result_ingestion_has_exactly_once_receipt(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    input_digest = _publish_input(attempt_dir)
    result = _result(attempt_dir, input_digest)
    result_digest = artifacts.publish_result(attempt_dir, result)
    receipts = tmp_path / "receipts"
    quarantine = tmp_path / "quarantine"

    ingested = artifacts.ingest_result(
        attempt_dir,
        _expectation(input_digest),
        receipt_root=receipts,
        quarantine_root=quarantine,
    )
    assert ingested.result_digest == result_digest
    assert ingested.receipt_path.is_file()

    with pytest.raises(artifacts.DuplicateResultError, match="already ingested"):
        artifacts.ingest_result(
            attempt_dir,
            _expectation(input_digest),
            receipt_root=receipts,
            quarantine_root=quarantine,
        )


def test_stale_generation_is_quarantined_before_ingestion(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    input_digest = _publish_input(attempt_dir, generation=1)
    artifacts.publish_result(attempt_dir, _result(attempt_dir, input_digest, generation=1))
    quarantine = tmp_path / "quarantine"

    with pytest.raises(artifacts.StaleResultError, match="stale result generation"):
        artifacts.ingest_result(
            attempt_dir,
            _expectation(input_digest, generation=2),
            receipt_root=tmp_path / "receipts",
            quarantine_root=quarantine,
        )
    assert not attempt_dir.exists()
    quarantined = list(quarantine.iterdir())
    assert len(quarantined) == 1
    assert (quarantined[0] / artifacts.QUARANTINE_FILENAME).is_file()
    assert not (tmp_path / "receipts").exists()


def test_evidence_path_traversal_and_symlink_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical and relative"):
        artifacts.EvidenceFile(path="../outside", sha256=TASK_DIGEST, size_bytes=1)

    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (attempt_dir / "link.txt").symlink_to(outside)
    with pytest.raises(artifacts.ArtifactError, match="symlink"):
        artifacts.describe_evidence(attempt_dir, "link.txt")


def test_non_finite_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        artifacts.WorkerInputManifest(
            run_id="run-1",
            item_id="item-1",
            attempt_id="attempt-1",
            task_digest=TASK_DIGEST,
            generation=1,
            role=state.Role.CODER,
            profile=state.DomainProfile.SMITH,
            worktree="/worktree",
            payload={"bad": float("nan")},
        )
