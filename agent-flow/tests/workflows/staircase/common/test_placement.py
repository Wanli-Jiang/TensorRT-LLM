# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Horizontal Smith placement receipt contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.artifacts import WorkerInputManifest
from agent_flow.workflows.staircase.common.placement import (
    PLACEMENT_RECEIPT_FILENAME,
    HorizontalPlacementContract,
    PlacementError,
    PlacementExpectation,
    load_worker_placement,
    publish_worker_placement,
    validate_horizontal_smith_wave,
)
from agent_flow.workflows.staircase.common.slurm import (
    JobIdentity,
    JobPlacementEvidence,
    ObservationSource,
)
from agent_flow.workflows.staircase.state import DomainProfile, JobReference, Role


def _manifest(item: str, attempt: str) -> WorkerInputManifest:
    return WorkerInputManifest(
        run_id="run-1",
        item_id=item,
        attempt_id=attempt,
        task_digest="a" * 64,
        generation=3,
        role=Role.CODER,
        profile=DomainProfile.SMITH,
        worktree="/workspace/candidate",
    )


def _publish(
    root: Path,
    *,
    item: str,
    attempt: str,
    job: str,
    node: str,
):
    root.mkdir()
    return publish_worker_placement(
        root,
        _manifest(item, attempt),
        HorizontalPlacementContract(True, True),
        environment={
            "SLURM_JOB_ID": job,
            "SLURM_CLUSTER_NAME": "alpha",
            "SLURMD_NODENAME": node,
            "SLURM_JOB_NODELIST": node,
            "SLURM_NNODES": "1",
        },
        hostname=node,
    )


def _expectation(item: str, attempt: str, job: str) -> PlacementExpectation:
    return PlacementExpectation(
        run_id="run-1",
        item_id=item,
        attempt_id=attempt,
        task_digest="a" * 64,
        generation=3,
        job=JobReference(job, cluster="alpha"),
        submission_token=f"token-{attempt}",
    )


def _evidence(attempt: str, job: str, node: str) -> JobPlacementEvidence:
    return JobPlacementEvidence(
        identity=JobIdentity(job, cluster="alpha"),
        submission_token=f"token-{attempt}",
        username="tester",
        node_count=1,
        node_list=node,
        matched=True,
        reason="exact test scheduler placement",
        source=ObservationSource.QUEUE,
    )


def test_reverse_completion_validates_in_stable_attempt_order(tmp_path: Path) -> None:
    first = _publish(
        tmp_path / "first",
        item="item-a",
        attempt="attempt-a",
        job="101",
        node="node-a",
    )
    second = _publish(
        tmp_path / "second",
        item="item-b",
        attempt="attempt-b",
        job="102",
        node="node-b",
    )
    assert first is not None and second is not None

    validated = validate_horizontal_smith_wave(
        (
            _expectation("item-a", "attempt-a", "101"),
            _expectation("item-b", "attempt-b", "102"),
        ),
        (second, first),
        (
            _evidence("attempt-b", "102", "node-b"),
            _evidence("attempt-a", "101", "node-a"),
        ),
        HorizontalPlacementContract(True, True),
    )

    assert tuple(receipt.attempt_id for receipt in validated) == ("attempt-a", "attempt-b")


def test_duplicate_missing_and_reused_node_receipts_fail_closed(tmp_path: Path) -> None:
    first = _publish(
        tmp_path / "first",
        item="item-a",
        attempt="attempt-a",
        job="101",
        node="node-a",
    )
    second = _publish(
        tmp_path / "second",
        item="item-b",
        attempt="attempt-b",
        job="102",
        node="node-a",
    )
    assert first is not None and second is not None
    expectations = (
        _expectation("item-a", "attempt-a", "101"),
        _expectation("item-b", "attempt-b", "102"),
    )
    contract = HorizontalPlacementContract(True, True)
    evidence = (
        _evidence("attempt-a", "101", "node-a"),
        _evidence("attempt-b", "102", "node-a"),
    )

    with pytest.raises(PlacementError, match="duplicate attempt"):
        validate_horizontal_smith_wave(expectations, (first, first), evidence, contract)
    with pytest.raises(PlacementError, match="missing or unexpected"):
        validate_horizontal_smith_wave(expectations, (first,), evidence, contract)
    with pytest.raises(PlacementError, match="reused a trusted scheduler node"):
        validate_horizontal_smith_wave(expectations, (first, second), evidence, contract)


def test_spoofed_attempt_job_and_receipt_bytes_fail_closed(tmp_path: Path) -> None:
    receipt = _publish(
        tmp_path / "receipt",
        item="item-a",
        attempt="attempt-a",
        job="999",
        node="node-a",
    )
    assert receipt is not None
    with pytest.raises(PlacementError, match="identity mismatch"):
        validate_horizontal_smith_wave(
            (_expectation("item-a", "attempt-a", "101"),),
            (receipt,),
            (_evidence("attempt-a", "101", "node-a"),),
            HorizontalPlacementContract(True, True),
        )

    path = tmp_path / "receipt" / PLACEMENT_RECEIPT_FILENAME
    value = json.loads(path.read_text(encoding="utf-8"))
    value["hostname"] = "spoofed-node"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(PlacementError, match="differs|digest mismatch"):
        load_worker_placement(path)


def test_rehashed_mailbox_forgery_cannot_replace_scheduler_placement(tmp_path: Path) -> None:
    first = _publish(
        tmp_path / "first",
        item="item-a",
        attempt="attempt-a",
        job="101",
        node="node-a",
    )
    second = _publish(
        tmp_path / "second",
        item="item-b",
        attempt="attempt-b",
        job="102",
        node="node-a",
    )
    assert first is not None and second is not None
    path = tmp_path / "second" / PLACEMENT_RECEIPT_FILENAME
    value = json.loads(path.read_text(encoding="utf-8"))
    value["hostname"] = "forged-node-b"
    value["node_list"] = "forged-node-b"
    digest_payload = dict(value)
    digest_payload.pop("receipt_sha256")
    value["receipt_sha256"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(value), encoding="utf-8")
    forged = load_worker_placement(path)

    with pytest.raises(PlacementError, match="differs from trusted scheduler placement"):
        validate_horizontal_smith_wave(
            (
                _expectation("item-a", "attempt-a", "101"),
                _expectation("item-b", "attempt-b", "102"),
            ),
            (first, forged),
            (
                _evidence("attempt-a", "101", "node-a"),
                _evidence("attempt-b", "102", "node-a"),
            ),
            HorizontalPlacementContract(True, True),
        )


def test_receipt_requires_exact_single_node_runtime(tmp_path: Path) -> None:
    with pytest.raises(PlacementError, match="exactly one"):
        publish_worker_placement(
            tmp_path,
            _manifest("item-a", "attempt-a"),
            HorizontalPlacementContract(True, True),
            environment={
                "SLURM_JOB_ID": "101",
                "SLURMD_NODENAME": "node-a",
                "SLURM_JOB_NODELIST": "node-[a-b]",
                "SLURM_NNODES": "2",
            },
            hostname="node-a",
        )
