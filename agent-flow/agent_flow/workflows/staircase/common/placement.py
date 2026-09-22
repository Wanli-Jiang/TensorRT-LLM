# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact horizontal-Smith placement receipts and wave validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..state import JobReference
from .artifacts import WorkerInputManifest
from .slurm import JobIdentity, JobPlacementEvidence

PLACEMENT_RECEIPT_FILENAME = "placement-receipt.json"
PLACEMENT_RECEIPT_SCHEMA_VERSION = 1

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SAFE_NODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_ARRAY_ID = re.compile(r"[0-9]+\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class PlacementError(RuntimeError):
    """Raised when placement evidence is absent, ambiguous, or spoofed."""


@dataclass(frozen=True, slots=True)
class HorizontalPlacementContract:
    """Task-level opt-in for independent Smith jobs on distinct nodes."""

    distinct_nodes_required: bool
    exclusive: bool

    def __post_init__(self) -> None:
        if not isinstance(self.distinct_nodes_required, bool) or not isinstance(
            self.exclusive, bool
        ):
            raise TypeError("horizontal placement contract fields must be booleans")
        if self.distinct_nodes_required and not self.exclusive:
            raise PlacementError("distinct-node Smith placement requires exclusive allocations")

    def to_public_dict(self) -> dict[str, bool]:
        """Return the strict manifest representation."""
        return {
            "distinct_nodes_required": self.distinct_nodes_required,
            "exclusive": self.exclusive,
        }

    @classmethod
    def from_public_dict(cls, value: object) -> HorizontalPlacementContract:
        """Strictly decode a manifest placement contract."""
        if not isinstance(value, dict) or set(value) != {
            "distinct_nodes_required",
            "exclusive",
        }:
            raise PlacementError("horizontal placement contract has invalid keys")
        return cls(value["distinct_nodes_required"], value["exclusive"])


@dataclass(frozen=True, slots=True)
class PlacementExpectation:
    """Controller-owned binding for one dispatched Smith attempt."""

    run_id: str
    item_id: str
    attempt_id: str
    task_digest: str
    generation: int
    job: JobReference
    submission_token: str


@dataclass(frozen=True, slots=True)
class WorkerPlacementReceipt:
    """Worker-observed node placement bound to exact attempt and Slurm job."""

    run_id: str
    item_id: str
    attempt_id: str
    task_digest: str
    generation: int
    job_id: str
    array_task_id: str | None
    cluster: str | None
    hostname: str
    node_list: str
    exclusive_requested: bool
    receipt_sha256: str
    schema_version: int = PLACEMENT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for value in (self.run_id, self.item_id, self.attempt_id):
            if _SAFE_ID.fullmatch(value) is None:
                raise PlacementError("placement receipt contains an unsafe identity")
        if _DIGEST.fullmatch(self.task_digest) is None:
            raise PlacementError("placement receipt task_digest must be a SHA-256")
        if isinstance(self.generation, bool) or self.generation < 1:
            raise PlacementError("placement receipt generation must be positive")
        if _JOB_ID.fullmatch(self.job_id) is None:
            raise PlacementError("placement receipt job_id must be exact and numeric")
        if self.array_task_id is not None and _ARRAY_ID.fullmatch(self.array_task_id) is None:
            raise PlacementError("placement receipt array_task_id must be exact and numeric")
        if self.cluster is not None and _SAFE_ID.fullmatch(self.cluster) is None:
            raise PlacementError("placement receipt cluster is unsafe")
        if (
            _SAFE_NODE.fullmatch(self.hostname) is None
            or _SAFE_NODE.fullmatch(self.node_list) is None
        ):
            raise PlacementError("single-node placement must contain exact safe hostnames")
        if self.hostname != self.node_list:
            raise PlacementError("worker hostname differs from its exact single-node NodeList")
        if not isinstance(self.exclusive_requested, bool):
            raise PlacementError("exclusive_requested must be a boolean")
        if _DIGEST.fullmatch(self.receipt_sha256) is None:
            raise PlacementError("placement receipt digest must be a SHA-256")
        if self.receipt_sha256 != _receipt_digest(self):
            raise PlacementError("placement receipt content digest mismatch")


def publish_worker_placement(
    attempt_dir: Path,
    manifest: WorkerInputManifest,
    contract: HorizontalPlacementContract,
    *,
    environment: Mapping[str, str] | None = None,
    hostname: str | None = None,
) -> WorkerPlacementReceipt | None:
    """Publish one immutable receipt from trusted Slurm placement variables."""
    if not contract.distinct_nodes_required:
        return None
    observed = os.environ if environment is None else environment
    host = socket.gethostname() if hostname is None else hostname
    job_id = observed.get("SLURM_JOB_ID", "")
    node = observed.get("SLURMD_NODENAME", "")
    node_list = observed.get("SLURM_JOB_NODELIST", "")
    node_count = observed.get("SLURM_NNODES", "")
    if node_count != "1":
        raise PlacementError("horizontal Smith worker must observe exactly one allocated node")
    if node != host:
        raise PlacementError("SLURMD_NODENAME differs from the worker hostname")
    array_task_id = observed.get("SLURM_ARRAY_TASK_ID")
    if array_task_id in {None, "", "4294967294"}:
        array_task_id = None
    cluster = observed.get("SLURM_CLUSTER_NAME") or None
    values = {
        "schema_version": PLACEMENT_RECEIPT_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "task_digest": manifest.task_digest,
        "generation": manifest.generation,
        "job_id": job_id,
        "array_task_id": array_task_id,
        "cluster": cluster,
        "hostname": node,
        "node_list": node_list,
        "exclusive_requested": contract.exclusive,
    }
    receipt = WorkerPlacementReceipt(
        **values,
        receipt_sha256=_receipt_digest_payload(values),
    )
    _write_once(attempt_dir / PLACEMENT_RECEIPT_FILENAME, receipt)
    return receipt


def load_worker_placement(path: Path) -> WorkerPlacementReceipt:
    """Load one strict immutable placement receipt."""
    if path.is_symlink() or not path.is_file():
        raise PlacementError(f"placement receipt is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlacementError(f"invalid placement receipt: {error}") from error
    expected = set(WorkerPlacementReceipt.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) != expected:
        raise PlacementError("placement receipt has invalid keys")
    try:
        return WorkerPlacementReceipt(**value)
    except (TypeError, ValueError) as error:
        raise PlacementError(f"invalid placement receipt: {error}") from error


def validate_horizontal_smith_wave(
    expectations: Sequence[PlacementExpectation],
    receipts: Sequence[WorkerPlacementReceipt],
    scheduler_evidence: Sequence[JobPlacementEvidence],
    contract: HorizontalPlacementContract,
) -> tuple[WorkerPlacementReceipt, ...]:
    """Validate worker observations against trusted exact scheduler placement."""
    if not contract.distinct_nodes_required:
        if receipts or scheduler_evidence:
            raise PlacementError("placement evidence is not admitted when proof is disabled")
        return ()
    expected_by_attempt = {entry.attempt_id: entry for entry in expectations}
    if len(expected_by_attempt) != len(expectations):
        raise PlacementError("placement expectations contain duplicate attempts")
    receipt_by_attempt = {entry.attempt_id: entry for entry in receipts}
    if len(receipt_by_attempt) != len(receipts):
        raise PlacementError("placement wave contains duplicate attempt receipts")
    if set(receipt_by_attempt) != set(expected_by_attempt):
        raise PlacementError("placement wave has missing or unexpected attempt receipts")
    evidence_by_identity = {entry.identity: entry for entry in scheduler_evidence}
    if len(evidence_by_identity) != len(scheduler_evidence):
        raise PlacementError("placement wave contains duplicate scheduler evidence")
    expected_identities = {
        JobIdentity(
            entry.job.job_id,
            array_task_id=entry.job.array_task_id,
            cluster=entry.job.cluster,
        )
        for entry in expectations
    }
    if set(evidence_by_identity) != expected_identities:
        raise PlacementError("placement wave has missing or unexpected scheduler evidence")
    ordered: list[WorkerPlacementReceipt] = []
    trusted_nodes: list[str] = []
    for attempt_id in sorted(expected_by_attempt):
        expected = expected_by_attempt[attempt_id]
        receipt = receipt_by_attempt[attempt_id]
        expected_identity = (
            expected.run_id,
            expected.item_id,
            expected.attempt_id,
            expected.task_digest,
            expected.generation,
            expected.job.job_id,
            expected.job.array_task_id,
            expected.job.cluster,
        )
        observed_identity = (
            receipt.run_id,
            receipt.item_id,
            receipt.attempt_id,
            receipt.task_digest,
            receipt.generation,
            receipt.job_id,
            receipt.array_task_id,
            receipt.cluster,
        )
        if observed_identity != expected_identity:
            raise PlacementError(f"placement receipt identity mismatch for {attempt_id!r}")
        scheduler_identity = JobIdentity(
            expected.job.job_id,
            array_task_id=expected.job.array_task_id,
            cluster=expected.job.cluster,
        )
        evidence = evidence_by_identity[scheduler_identity]
        if evidence.submission_token != expected.submission_token:
            raise PlacementError(f"scheduler placement token mismatch for {attempt_id!r}")
        if not evidence.matched or evidence.node_list is None:
            raise PlacementError(
                f"scheduler placement is not trusted for {attempt_id!r}: {evidence.reason}"
            )
        if receipt.hostname != evidence.node_list or receipt.node_list != evidence.node_list:
            raise PlacementError(
                f"worker placement differs from trusted scheduler placement for {attempt_id!r}"
            )
        if receipt.exclusive_requested != contract.exclusive:
            raise PlacementError(f"placement receipt exclusivity mismatch for {attempt_id!r}")
        ordered.append(receipt)
        trusted_nodes.append(evidence.node_list)
    if len(set(trusted_nodes)) != len(trusted_nodes):
        raise PlacementError("distinct-node Smith wave reused a trusted scheduler node")
    return tuple(ordered)


def _receipt_digest(receipt: WorkerPlacementReceipt) -> str:
    payload = asdict(receipt)
    payload.pop("receipt_sha256", None)
    return _receipt_digest_payload(payload)


def _receipt_digest_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_once(path: Path, receipt: WorkerPlacementReceipt) -> None:
    payload = json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        existing = load_worker_placement(path)
        if existing != receipt:
            raise PlacementError("immutable placement receipt differs from observed placement")


__all__ = [
    "HorizontalPlacementContract",
    "PLACEMENT_RECEIPT_FILENAME",
    "PlacementError",
    "PlacementExpectation",
    "WorkerPlacementReceipt",
    "load_worker_placement",
    "publish_worker_placement",
    "validate_horizontal_smith_wave",
]
