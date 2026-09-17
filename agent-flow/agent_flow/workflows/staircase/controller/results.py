# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict worker-result decoding and controller-owned candidate binding.

Workers publish typed proposals and evidence; they never choose lifecycle
transitions or create Git commits.  This module rejects unknown payload fields,
binds successful Coder changes to a DCO-signed controller commit, and persists
an immutable receipt that can be verified after controller restart.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypeAlias

from ..common.artifacts import EvidenceFile, WorkerResultManifest, WorkerResultStatus
from ..common.gates import (
    AccuracyCriteria,
    EvidenceScope,
    GatePurpose,
    GateReceipt,
    GateSpec,
    RankPlacement,
)
from ..common.gitops import ControllerGitOps, GitOpsError
from ..common.index import IndexDelta
from ..common.policy import (
    CATALOG_ITEM_KINDS,
    PolicyViolation,
    ResourceEscalationRequest,
    WorkItemKind,
    WorkItemProposal,
    validate_resource_escalation,
)
from ..state import AttemptKind, AttemptRecord, AttemptStatus, DomainProfile, Role
from ..task_schema import CertificationMode
from ..tuning.artifacts import (
    BASELINE_FILENAME,
    CAMPAIGN_FILENAME,
    CANDIDATE_FILENAME,
    EVALUATION_FILENAME,
    PROMOTION_DECISION_FILENAME,
    TuningArtifactError,
    load_baseline_artifact,
    load_campaign_artifact,
    load_candidate_artifact,
    load_evaluation_artifact,
    parse_baseline_measurement_payload,
    parse_candidate_measurement_payload,
)
from ..tuning.contracts import TuningHypothesis
from ..tuning.workflow import (
    BaselineMeasurement,
    CandidateMeasurement,
    TuningContractError,
    evaluate_tuning_campaign,
    new_tuning_campaign,
    record_tuning_evidence,
)
from .smith import CandidateSnapshot

CANDIDATE_RECEIPT_FILENAME = "candidate-receipt.json"
CANDIDATE_RECEIPT_SCHEMA_VERSION = 1
WORKER_RESULT_SCHEMA_VERSION = 1

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ResultContractError(ValueError):
    """Raised when untrusted worker output violates its typed contract."""


class CandidateReceiptError(RuntimeError):
    """Raised when a candidate receipt or its Git binding cannot be verified."""


class ReviewerVerdict(str, Enum):
    """Closed Reviewer decision vocabulary."""

    APPROVE = "approve"
    REJECT = "reject"
    BLOCKED = "blocked"


class QaVerdict(str, Enum):
    """Closed QA decision vocabulary."""

    APPROVE = "approve"
    REJECT = "reject"
    BLOCKED = "blocked"


class CandidateDisposition(str, Enum):
    """Controller-owned distinction between patch and no-change candidates."""

    PATCH = "patch"
    VERIFICATION = "verification"


@dataclass(frozen=True, slots=True)
class CoderResult:
    """Strict Coder proposal before the controller grants Git identity."""

    changed_paths: tuple[str, ...]
    index_delta: IndexDelta | None


@dataclass(frozen=True, slots=True)
class ReviewerResult:
    """Digest-pinned Reviewer analysis or fresh rerun verdict."""

    attempt_id: str
    candidate_attempt_id: str
    candidate_digest: str
    kind: AttemptKind
    verdict: ReviewerVerdict
    findings: tuple[str, ...]
    evidence: tuple[EvidenceFile, ...]

    @property
    def final_approval(self) -> bool:
        """Whether this is the fresh rerun that may approve a WorkItem."""
        return self.kind is AttemptKind.REVIEWER_RERUN and self.verdict is ReviewerVerdict.APPROVE


@dataclass(frozen=True, slots=True)
class GateResult:
    """Candidate-pinned deterministic gate receipt."""

    candidate_attempt_id: str
    candidate_digest: str
    certification_mode: CertificationMode
    receipt: GateReceipt
    evidence: tuple[EvidenceFile, ...]


@dataclass(frozen=True, slots=True)
class TunerResult:
    """Strict measured A/B evidence emitted by one Tuner-profile worker."""

    baseline: BaselineMeasurement
    candidate: CandidateMeasurement
    evidence: tuple[EvidenceFile, ...]


@dataclass(frozen=True, slots=True)
class QaGateReference:
    """Exact controller-ingested gate result audited by QA."""

    gate_attempt_id: str
    result_digest: str


@dataclass(frozen=True, slots=True)
class QaResult:
    """Candidate-pinned QA verdict referencing controller-owned gate results."""

    candidate_attempt_id: str
    candidate_digest: str
    verdict: QaVerdict
    findings: tuple[str, ...]
    gate_results: tuple[QaGateReference, ...]
    evidence: tuple[EvidenceFile, ...]


@dataclass(frozen=True, slots=True)
class CandidateReceipt:
    """Immutable controller binding from one Coder attempt to one Git commit."""

    run_id: str
    item_id: str
    attempt_id: str
    disposition: CandidateDisposition
    base_commit: str
    candidate_commit: str
    candidate_digest: str
    changed_paths: tuple[str, ...]
    index_delta: IndexDelta | None
    receipt_sha256: str
    schema_version: int = CANDIDATE_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("item_id", self.item_id),
            ("attempt_id", self.attempt_id),
        ):
            _safe_id(value, name)
        _digest(self.base_commit, "base_commit", _SHA1)
        _digest(self.candidate_commit, "candidate_commit", _SHA1)
        _digest(self.candidate_digest, "candidate_digest", _SHA256)
        _digest(self.receipt_sha256, "receipt_sha256", _SHA256)
        if not isinstance(self.disposition, CandidateDisposition):
            raise CandidateReceiptError("candidate disposition must be typed")
        _changed_paths(self.changed_paths)
        if self.disposition is CandidateDisposition.PATCH and not self.changed_paths:
            raise CandidateReceiptError("patch candidate receipt requires changed paths")
        if self.disposition is CandidateDisposition.VERIFICATION and (
            self.candidate_commit != self.base_commit
            or self.changed_paths
            or self.index_delta is not None
        ):
            raise CandidateReceiptError(
                "verification candidate must be an unchanged commit without IndexDelta"
            )
        if self.index_delta is not None and self.index_delta.item_id != self.item_id:
            raise CandidateReceiptError("candidate receipt IndexDelta belongs to another item")
        if self.receipt_sha256 != _receipt_digest(self):
            raise CandidateReceiptError("candidate receipt content digest mismatch")


@dataclass(frozen=True, slots=True)
class CandidateBinding:
    """Verified receipt projected into the Smith fan-in candidate type."""

    receipt: CandidateReceipt
    snapshot: CandidateSnapshot

    @property
    def index_delta(self) -> IndexDelta | None:
        """Return the optional semantic catalog-index proposal."""
        return self.receipt.index_delta


TypedWorkerResult: TypeAlias = (
    CoderResult | TunerResult | ReviewerResult | GateResult | QaResult | ResourceEscalationRequest
)


def decode_coder_result(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    proposal: WorkItemProposal,
) -> CoderResult:
    """Strictly decode one successful Coder proposal without granting Git authority."""
    _manifest_attempt(manifest, attempt)
    if attempt.role is not Role.CODER or attempt.kind is not AttemptKind.ROLE:
        raise ResultContractError("Coder result requires a Coder role attempt")
    if manifest.status is not WorkerResultStatus.SUCCEEDED:
        raise ResultContractError("Coder payload requires worker status 'succeeded'")
    if manifest.candidate_digest is None:
        raise ResultContractError("worker Coder requires a scanned candidate overlay digest")
    if manifest.reviewed_candidate_digest is not None:
        raise ResultContractError("Coder result cannot carry reviewed_candidate_digest")
    if manifest.item_id != proposal.item_id:
        raise ResultContractError("Coder result belongs to another frozen WorkItem")
    payload = _object(manifest.payload, "Coder payload")
    _exact_keys(
        payload,
        {"schema_version", "result_kind", "changed_paths", "index_delta"},
        "Coder payload",
    )
    _schema(payload)
    _literal(payload["result_kind"], "coder", "Coder result_kind")
    paths = _string_sequence(payload["changed_paths"], "Coder changed_paths")
    _changed_paths(paths)
    if tuple(sorted(paths)) != paths:
        raise ResultContractError("Coder changed_paths must be sorted")
    unexpected = sorted(set(paths) - set(proposal.allowed_paths))
    if unexpected:
        raise ResultContractError(f"Coder changed paths exceed frozen claim: {unexpected}")
    if proposal.modifies_files != bool(paths):
        raise ResultContractError(
            "Coder changed_paths must be non-empty exactly for file-modifying work"
        )
    raw_delta = payload["index_delta"]
    index_delta: IndexDelta | None
    if raw_delta is None:
        index_delta = None
    else:
        if not isinstance(raw_delta, Mapping):
            raise ResultContractError("Coder index_delta must be an object or null")
        try:
            index_delta = IndexDelta.from_mapping(raw_delta)
            _validate_index_delta(index_delta, proposal)
        except ValueError as error:
            raise ResultContractError(f"invalid Coder IndexDelta: {error}") from error
    if proposal.kind is WorkItemKind.CATALOG_ONBOARD and index_delta is None:
        raise ResultContractError("catalog onboarding Coder must return a typed IndexDelta")
    if proposal.kind is not WorkItemKind.CATALOG_ONBOARD and index_delta is not None:
        raise ResultContractError("only catalog onboarding may return an IndexDelta")
    return CoderResult(paths, index_delta)


def decode_tuner_result(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    proposal: WorkItemProposal,
) -> TunerResult:
    """Decode measured tuning arms without accepting a worker promotion decision."""
    _manifest_attempt(manifest, attempt)
    if (
        attempt.role is not Role.CODER
        or attempt.kind is not AttemptKind.ROLE
        or attempt.profile is not DomainProfile.TUNER
    ):
        raise ResultContractError("Tuner result requires a Tuner-profile Coder attempt")
    if proposal.kind is not WorkItemKind.TUNE_HYPOTHESIS or not isinstance(
        proposal.domain_input, TuningHypothesis
    ):
        raise ResultContractError("Tuner result requires a typed tuning hypothesis")
    if manifest.status is not WorkerResultStatus.SUCCEEDED:
        raise ResultContractError("Tuner measurement requires worker status 'succeeded'")
    if manifest.candidate_digest is None or manifest.reviewed_candidate_digest is not None:
        raise ResultContractError(
            "Tuner worker requires a scanned overlay digest and cannot review a candidate digest"
        )
    payload = _object(manifest.payload, "Tuner payload")
    _exact_keys(
        payload,
        {"schema_version", "result_kind", "baseline", "candidate", "evidence"},
        "Tuner payload",
    )
    _schema(payload)
    _literal(payload["result_kind"], "tuner_measurement", "Tuner result_kind")
    try:
        baseline = parse_baseline_measurement_payload(payload["baseline"])
        candidate = parse_candidate_measurement_payload(payload["candidate"])
        evaluate_tuning_campaign(proposal, proposal.domain_input, baseline, candidate)
    except (TuningArtifactError, TuningContractError) as error:
        raise ResultContractError(f"invalid Tuner measurement: {error}") from error
    evidence = _evidence(payload["evidence"], manifest.evidence, "Tuner evidence")
    if not evidence:
        raise ResultContractError("Tuner measurement requires immutable evidence")
    return TunerResult(baseline, candidate, evidence)


def decode_reviewer_result(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    candidate: CandidateReceipt,
) -> ReviewerResult:
    """Decode a Reviewer analysis/rerun pinned to one immutable candidate."""
    _manifest_attempt(manifest, attempt)
    if attempt.role is not Role.REVIEWER or attempt.kind not in {
        AttemptKind.REVIEWER_ANALYSIS,
        AttemptKind.REVIEWER_RERUN,
    }:
        raise ResultContractError("Reviewer payload requires a Reviewer analysis or rerun")
    expected_kind = (
        "reviewer_analysis" if attempt.kind is AttemptKind.REVIEWER_ANALYSIS else "reviewer_rerun"
    )
    payload = _object(manifest.payload, "Reviewer payload")
    _exact_keys(
        payload,
        {
            "schema_version",
            "result_kind",
            "candidate_attempt_id",
            "candidate_digest",
            "verdict",
            "findings",
            "evidence",
        },
        "Reviewer payload",
    )
    _schema(payload)
    _literal(payload["result_kind"], expected_kind, "Reviewer result_kind")
    candidate_attempt_id = _string(payload["candidate_attempt_id"], "candidate_attempt_id")
    candidate_digest = _hex_digest(payload["candidate_digest"], "candidate_digest", _SHA256)
    _candidate_pin(manifest, attempt, candidate, candidate_attempt_id, candidate_digest)
    verdict = _enum(payload["verdict"], ReviewerVerdict, "Reviewer verdict")
    _status_for_decision(manifest.status, verdict.value, "Reviewer")
    findings = _nonempty_strings(payload["findings"], "Reviewer findings", allow_empty=True)
    evidence = _evidence(payload["evidence"], manifest.evidence, "Reviewer evidence")
    return ReviewerResult(
        attempt.attempt_id,
        candidate_attempt_id,
        candidate_digest,
        attempt.kind,
        verdict,
        findings,
        evidence,
    )


def decode_gate_result(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    candidate: CandidateReceipt,
    *,
    expected_spec: GateSpec | None = None,
) -> GateResult:
    """Decode one deterministic gate result with candidate and evidence pins."""
    _manifest_attempt(manifest, attempt)
    if attempt.role is not Role.GATE or attempt.kind is not AttemptKind.DETERMINISTIC_GATE:
        raise ResultContractError("gate payload requires a deterministic Gate attempt")
    payload = _object(manifest.payload, "gate payload")
    _exact_keys(
        payload,
        {
            "schema_version",
            "result_kind",
            "certification_mode",
            "candidate_attempt_id",
            "candidate_digest",
            "receipt",
            "evidence",
        },
        "gate payload",
    )
    _schema(payload)
    _literal(payload["result_kind"], "deterministic_gate", "gate result_kind")
    certification_mode = _enum(
        payload["certification_mode"], CertificationMode, "gate certification_mode"
    )
    candidate_attempt_id = _string(payload["candidate_attempt_id"], "candidate_attempt_id")
    candidate_digest = _hex_digest(payload["candidate_digest"], "candidate_digest", _SHA256)
    _candidate_pin(manifest, attempt, candidate, candidate_attempt_id, candidate_digest)
    receipt = _gate_receipt(payload["receipt"])
    expected_certification = {
        EvidenceScope.CPU_STATIC: CertificationMode.LOCAL,
        EvidenceScope.SINGLE_GPU_PRODUCT: CertificationMode.LOCAL,
        EvidenceScope.LOCAL_FOUR_GPU_PRODUCT: CertificationMode.LOCAL,
        EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER: CertificationMode.SYNTHETIC,
        EvidenceScope.REAL_MULTI_NODE_PRODUCT: CertificationMode.REAL,
    }[receipt.scope]
    if certification_mode is not expected_certification:
        raise ResultContractError(
            "gate certification_mode disagrees with the immutable evidence scope"
        )
    expected_status = (
        WorkerResultStatus.SUCCEEDED if receipt.passed else WorkerResultStatus.REJECTED
    )
    if manifest.status is not expected_status:
        raise ResultContractError("gate worker status disagrees with typed receipt verdict")
    if expected_spec is not None:
        _validate_gate_spec(receipt, expected_spec)
    evidence = _evidence(payload["evidence"], manifest.evidence, "gate evidence")
    if not evidence:
        raise ResultContractError("gate result requires immutable evidence")
    return GateResult(
        candidate_attempt_id,
        candidate_digest,
        certification_mode,
        receipt,
        evidence,
    )


def decode_qa_result(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    candidate: CandidateReceipt,
    *,
    expected_gate_results: Mapping[str, str] | None = None,
) -> QaResult:
    """Decode QA references without accepting QA-authored gate receipts."""
    _manifest_attempt(manifest, attempt)
    if attempt.role is not Role.QA or attempt.kind is not AttemptKind.QA:
        raise ResultContractError("QA payload requires a QA attempt")
    payload = _object(manifest.payload, "QA payload")
    _exact_keys(
        payload,
        {
            "schema_version",
            "result_kind",
            "candidate_attempt_id",
            "candidate_digest",
            "verdict",
            "findings",
            "gate_results",
            "evidence",
        },
        "QA payload",
    )
    _schema(payload)
    _literal(payload["result_kind"], "qa", "QA result_kind")
    candidate_attempt_id = _string(payload["candidate_attempt_id"], "candidate_attempt_id")
    candidate_digest = _hex_digest(payload["candidate_digest"], "candidate_digest", _SHA256)
    _candidate_pin(manifest, attempt, candidate, candidate_attempt_id, candidate_digest)
    verdict = _enum(payload["verdict"], QaVerdict, "QA verdict")
    _status_for_decision(manifest.status, verdict.value, "QA")
    findings = _nonempty_strings(payload["findings"], "QA findings", allow_empty=True)
    if verdict is not QaVerdict.APPROVE and not findings:
        raise ResultContractError("non-approving QA verdict requires at least one finding")
    raw_references = _sequence(payload["gate_results"], "QA gate_results")
    references: list[QaGateReference] = []
    for raw_reference in raw_references:
        reference = _object(raw_reference, "QA gate result reference")
        _exact_keys(
            reference,
            {"gate_attempt_id", "result_digest"},
            "QA gate result reference",
        )
        references.append(
            QaGateReference(
                _string(reference["gate_attempt_id"], "gate_attempt_id"),
                _hex_digest(reference["result_digest"], "result_digest", _SHA256),
            )
        )
    gate_results = tuple(references)
    attempt_ids = [reference.gate_attempt_id for reference in gate_results]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ResultContractError("QA gate_results contain duplicate attempt IDs")
    if verdict is QaVerdict.APPROVE and not gate_results:
        raise ResultContractError("QA approval requires non-empty gate result references")
    if expected_gate_results is not None and dict(
        (reference.gate_attempt_id, reference.result_digest) for reference in gate_results
    ) != dict(expected_gate_results):
        raise ResultContractError(
            "QA gate result references differ from controller-ingested gate results"
        )
    evidence = _evidence(payload["evidence"], manifest.evidence, "QA evidence")
    if not evidence:
        raise ResultContractError("QA result requires immutable evidence")
    return QaResult(
        candidate_attempt_id,
        candidate_digest,
        verdict,
        findings,
        gate_results,
        evidence,
    )


def decode_resource_escalation(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    proposal: WorkItemProposal,
    *,
    allowed_resource_classes: Sequence[str],
) -> ResourceEscalationRequest:
    """Decode and policy-check a bounded resource request without granting it."""
    _manifest_attempt(manifest, attempt)
    if manifest.status is not WorkerResultStatus.RESOURCE_ESCALATION:
        raise ResultContractError("resource escalation requires matching worker status")
    if manifest.candidate_digest is not None:
        raise ResultContractError("resource escalation cannot assign candidate_digest")
    payload = _object(manifest.payload, "resource escalation payload")
    _exact_keys(
        payload,
        {"schema_version", "result_kind", "request"},
        "resource escalation payload",
    )
    _schema(payload)
    _literal(payload["result_kind"], "resource_escalation", "resource result_kind")
    request_data = _object(payload["request"], "resource escalation request")
    _exact_keys(
        request_data,
        {
            "request_id",
            "item_id",
            "attempt_id",
            "current_resource_class",
            "requested_resource_class",
            "reason",
        },
        "resource escalation request",
    )
    try:
        request = ResourceEscalationRequest(
            request_id=_string(request_data["request_id"], "request_id"),
            item_id=_string(request_data["item_id"], "item_id"),
            attempt_id=_string(request_data["attempt_id"], "attempt_id"),
            current_resource_class=_string(
                request_data["current_resource_class"], "current_resource_class"
            ),
            requested_resource_class=_string(
                request_data["requested_resource_class"], "requested_resource_class"
            ),
            reason=_string(request_data["reason"], "reason"),
        )
    except ValueError as error:
        raise ResultContractError(f"invalid resource escalation request: {error}") from error
    if request.item_id != manifest.item_id or request.attempt_id != manifest.attempt_id:
        raise ResultContractError("resource request identity differs from its manifest")
    if attempt.resource_class is None:
        raise ResultContractError("resource escalation source has no selected resource class")
    try:
        validate_resource_escalation(
            request,
            proposal,
            current_attempt_id=attempt.attempt_id,
            current_resource_class=attempt.resource_class,
            current_role=attempt.role,
            resource_class_history=(attempt.resource_class,),
            consumed_request_ids=(),
            allowed_resource_classes=allowed_resource_classes,
        )
    except ValueError as error:
        raise ResultContractError(f"invalid resource escalation policy: {error}") from error
    return request


def decode_worker_result(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    proposal: WorkItemProposal,
    *,
    candidate: CandidateReceipt | None = None,
    gate_spec: GateSpec | None = None,
    qa_gate_results: Mapping[str, str] | None = None,
    allowed_resource_classes: Sequence[str] = (),
) -> TypedWorkerResult:
    """Route a strict payload to its typed decoder; never inspect summary prose."""
    payload = _object(manifest.payload, "worker payload")
    kind = _string(payload.get("result_kind"), "result_kind")
    if kind == "coder":
        return decode_coder_result(manifest, attempt, proposal)
    if kind == "tuner_measurement":
        return decode_tuner_result(manifest, attempt, proposal)
    if kind in {"reviewer_analysis", "reviewer_rerun"}:
        return decode_reviewer_result(manifest, attempt, _required_candidate(candidate))
    if kind == "deterministic_gate":
        return decode_gate_result(
            manifest,
            attempt,
            _required_candidate(candidate),
            expected_spec=gate_spec,
        )
    if kind == "qa":
        return decode_qa_result(
            manifest,
            attempt,
            _required_candidate(candidate),
            expected_gate_results=qa_gate_results,
        )
    if kind == "resource_escalation":
        return decode_resource_escalation(
            manifest,
            attempt,
            proposal,
            allowed_resource_classes=allowed_resource_classes,
        )
    raise ResultContractError(f"unsupported worker result_kind {kind!r}")


def bind_coder_candidate(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    proposal: WorkItemProposal,
    *,
    worktree: Path,
    base_commit: str,
    attempt_dir: Path,
    git: ControllerGitOps,
) -> CandidateBinding:
    """Bind a metadata-free Coder overlay and publish a restart-safe receipt."""
    decoded = decode_coder_result(manifest, attempt, proposal)
    scanned_overlay_digest = manifest.candidate_digest
    if scanned_overlay_digest is None:
        raise CandidateReceiptError("Coder result lacks its scanned overlay digest")
    disposition = (
        CandidateDisposition.PATCH if proposal.modifies_files else CandidateDisposition.VERIFICATION
    )
    overlay = _canonical_candidate_directory(worktree, "candidate overlay")
    try:
        workspace = overlay.parents[2]
        relative = overlay.relative_to(workspace / "candidates")
    except (IndexError, ValueError) as error:
        raise CandidateReceiptError(
            "candidate overlay must be under workspace/candidates/<item>/<attempt>"
        ) from error
    if relative.parts != (manifest.item_id, manifest.attempt_id):
        raise CandidateReceiptError("candidate overlay differs from worker identity")
    repository = workspace / "controller-candidates" / relative
    branch = f"staircase/{manifest.run_id}/{manifest.attempt_id}"
    mailbox = _canonical_candidate_directory(attempt_dir, "attempt_dir")
    receipt_path = mailbox / CANDIDATE_RECEIPT_FILENAME
    if receipt_path.is_symlink():
        raise CandidateReceiptError("candidate receipt cannot be a symlink")
    if receipt_path.exists():
        receipt = load_candidate_receipt(receipt_path)
        if (
            (receipt.run_id, receipt.item_id, receipt.attempt_id)
            != (manifest.run_id, manifest.item_id, manifest.attempt_id)
            or receipt.disposition is not disposition
            or receipt.base_commit != base_commit
            or receipt.changed_paths != decoded.changed_paths
            or receipt.index_delta != decoded.index_delta
            or (
                disposition is CandidateDisposition.VERIFICATION
                and receipt.candidate_digest
                != _verification_candidate_digest(manifest, base_commit)
            )
        ):
            raise CandidateReceiptError("existing candidate receipt differs from Coder result")
        return reconstruct_candidate(
            receipt_path,
            worktree=repository,
            proposal=proposal,
            git=git,
        )

    try:
        with git.transaction() as transaction:
            bound = transaction.bind_candidate_overlay(
                overlay,
                repository=repository,
                branch=branch,
                base_commit=base_commit,
                expected_paths=decoded.changed_paths,
                expected_overlay_sha256=scanned_overlay_digest,
                message=f"staircase: bind {proposal.item_id} candidate",
            )
    except GitOpsError as error:
        raise CandidateReceiptError(f"candidate overlay binding failed: {error}") from error

    if disposition is CandidateDisposition.VERIFICATION:
        if bound.candidate_commit != base_commit or bound.changed_paths:
            raise CandidateReceiptError("verification candidate changed the frozen base")
        receipt = _new_candidate_receipt(
            manifest,
            disposition=disposition,
            base_commit=base_commit,
            candidate_commit=base_commit,
            candidate_digest=_verification_candidate_digest(manifest, base_commit),
            changed_paths=(),
            index_delta=None,
        )
        _write_receipt_once(receipt_path, receipt)
        return reconstruct_candidate(
            receipt_path,
            worktree=repository,
            proposal=proposal,
            git=git,
        )
    verification = git.verify_candidate(
        repository,
        base_commit=base_commit,
        candidate_commit=bound.candidate_commit,
        allowed_paths=proposal.allowed_paths,
    )
    if verification.changed_paths != decoded.changed_paths:
        raise CandidateReceiptError(
            "controller-computed candidate paths differ from the Coder declaration"
        )
    _verify_controller_commit(
        repository,
        verification.candidate_commit,
        item_id=proposal.item_id,
    )
    receipt = _new_candidate_receipt(
        manifest,
        disposition=disposition,
        base_commit=verification.base_commit,
        candidate_commit=verification.candidate_commit,
        candidate_digest=verification.patch_sha256,
        changed_paths=verification.changed_paths,
        index_delta=decoded.index_delta,
    )
    _write_receipt_once(receipt_path, receipt)
    return reconstruct_candidate(
        receipt_path,
        worktree=repository,
        proposal=proposal,
        git=git,
    )


def bind_tuner_candidate(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    proposal: WorkItemProposal,
    *,
    worktree: Path,
    base_commit: str,
    attempt_dir: Path,
    artifact_root: Path,
    git: ControllerGitOps,
) -> CandidateBinding:
    """Bind immutable Tuner evidence as an unchanged verification candidate.

    This grants no keep/reject authority. It only gives the normal
    Reviewer/gate/QA chain a content identity that covers the exact worker
    result, durable evidence-ready artifacts, and unchanged frozen Git base.
    """
    decoded = decode_tuner_result(manifest, attempt, proposal)
    scanned_overlay_digest = manifest.candidate_digest
    if scanned_overlay_digest is None:
        raise CandidateReceiptError("Tuner result lacks its scanned overlay digest")
    if proposal.modifies_files:
        raise CandidateReceiptError("Tuner verification cannot modify repository files")
    hypothesis = proposal.domain_input
    if not isinstance(hypothesis, TuningHypothesis):
        raise CandidateReceiptError("Tuner verification requires a typed tuning hypothesis")
    canonical_base = _hex_digest(base_commit, "base_commit", _SHA1)
    overlay = _canonical_candidate_directory(worktree, "candidate overlay")
    try:
        workspace = overlay.parents[2]
        relative = overlay.relative_to(workspace / "candidates")
    except (IndexError, ValueError) as error:
        raise CandidateReceiptError(
            "candidate overlay must be under workspace/candidates/<item>/<attempt>"
        ) from error
    if relative.parts != (manifest.item_id, manifest.attempt_id):
        raise CandidateReceiptError("candidate overlay differs from worker identity")
    repository = workspace / "controller-candidates" / relative
    branch = f"staircase/{manifest.run_id}/{manifest.attempt_id}"
    mailbox = _canonical_candidate_directory(attempt_dir, "attempt_dir")
    receipt_path = mailbox / CANDIDATE_RECEIPT_FILENAME
    if receipt_path.is_symlink():
        raise CandidateReceiptError("candidate receipt cannot be a symlink")

    artifacts = _canonical_candidate_directory(artifact_root, "Tuner artifact root")
    decision_path = artifacts / PROMOTION_DECISION_FILENAME
    if decision_path.exists() or decision_path.is_symlink():
        raise CandidateReceiptError("Tuner evidence cannot contain a promotion decision")
    try:
        baseline, baseline_digest = load_baseline_artifact(artifacts / BASELINE_FILENAME)
        candidate, candidate_digest = load_candidate_artifact(artifacts / CANDIDATE_FILENAME)
        evaluation, evaluation_digest = load_evaluation_artifact(artifacts / EVALUATION_FILENAME)
        campaign, campaign_digest = load_campaign_artifact(artifacts / CAMPAIGN_FILENAME)
        expected_evaluation = evaluate_tuning_campaign(
            proposal,
            hypothesis,
            decoded.baseline,
            decoded.candidate,
        )
        expected_campaign = record_tuning_evidence(
            new_tuning_campaign(proposal, hypothesis),
            expected_evaluation,
        )
    except (TuningArtifactError, TuningContractError) as error:
        raise CandidateReceiptError(f"invalid durable Tuner evidence: {error}") from error
    if (
        baseline != decoded.baseline
        or candidate != decoded.candidate
        or evaluation != expected_evaluation
        or campaign != expected_campaign
    ):
        raise CandidateReceiptError("durable Tuner evidence differs from the validated result")
    semantic_digest = _tuner_candidate_digest(
        manifest,
        base_commit=canonical_base,
        artifact_digests=(
            (BASELINE_FILENAME, baseline_digest),
            (CANDIDATE_FILENAME, candidate_digest),
            (EVALUATION_FILENAME, evaluation_digest),
            (CAMPAIGN_FILENAME, campaign_digest),
        ),
    )

    if receipt_path.exists():
        receipt = load_candidate_receipt(receipt_path)
        if (
            (receipt.run_id, receipt.item_id, receipt.attempt_id)
            != (manifest.run_id, manifest.item_id, manifest.attempt_id)
            or receipt.disposition is not CandidateDisposition.VERIFICATION
            or receipt.base_commit != canonical_base
            or receipt.candidate_commit != canonical_base
            or receipt.candidate_digest != semantic_digest
            or receipt.changed_paths
            or receipt.index_delta is not None
        ):
            raise CandidateReceiptError("existing Tuner receipt differs from durable evidence")
        return reconstruct_candidate(
            receipt_path,
            worktree=repository,
            proposal=proposal,
            git=git,
        )

    try:
        with git.transaction() as transaction:
            bound = transaction.bind_candidate_overlay(
                overlay,
                repository=repository,
                branch=branch,
                base_commit=canonical_base,
                expected_paths=(),
                expected_overlay_sha256=scanned_overlay_digest,
                message=f"staircase: bind {proposal.item_id} Tuner evidence",
            )
    except GitOpsError as error:
        raise CandidateReceiptError(f"Tuner overlay binding failed: {error}") from error
    if bound.candidate_commit != canonical_base or bound.changed_paths:
        raise CandidateReceiptError("Tuner verification changed the frozen base")
    receipt = _new_candidate_receipt(
        manifest,
        disposition=CandidateDisposition.VERIFICATION,
        base_commit=canonical_base,
        candidate_commit=canonical_base,
        candidate_digest=semantic_digest,
        changed_paths=(),
        index_delta=None,
    )
    _write_receipt_once(receipt_path, receipt)
    return reconstruct_candidate(
        receipt_path,
        worktree=repository,
        proposal=proposal,
        git=git,
    )


def load_candidate_receipt(path: Path) -> CandidateReceipt:
    """Strictly load and content-verify an immutable candidate receipt."""
    expanded = Path(os.path.abspath(path.expanduser()))
    if expanded.is_symlink():
        raise CandidateReceiptError("candidate receipt cannot be a symlink")
    resolved = expanded.resolve(strict=True)
    if resolved != expanded:
        raise CandidateReceiptError("candidate receipt path contains a symlink")
    if not resolved.is_file():
        raise CandidateReceiptError("candidate receipt must be a regular file")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateReceiptError(f"cannot parse candidate receipt: {error}") from error
    data = _object(raw, "candidate receipt")
    _exact_keys(
        data,
        {
            "schema_version",
            "run_id",
            "item_id",
            "attempt_id",
            "disposition",
            "base_commit",
            "candidate_commit",
            "candidate_digest",
            "changed_paths",
            "index_delta",
            "receipt_sha256",
        },
        "candidate receipt",
        error_type=CandidateReceiptError,
    )
    schema_version = _integer(data["schema_version"], "schema_version")
    if schema_version != CANDIDATE_RECEIPT_SCHEMA_VERSION:
        raise CandidateReceiptError(f"unsupported candidate receipt schema {schema_version!r}")
    raw_delta = data["index_delta"]
    try:
        index_delta = (
            None
            if raw_delta is None
            else IndexDelta.from_mapping(_object(raw_delta, "candidate receipt IndexDelta"))
        )
        receipt = CandidateReceipt(
            run_id=_string(data["run_id"], "run_id"),
            item_id=_string(data["item_id"], "item_id"),
            attempt_id=_string(data["attempt_id"], "attempt_id"),
            disposition=CandidateDisposition(_string(data["disposition"], "candidate disposition")),
            base_commit=_hex_digest(data["base_commit"], "base_commit", _SHA1),
            candidate_commit=_hex_digest(data["candidate_commit"], "candidate_commit", _SHA1),
            candidate_digest=_hex_digest(data["candidate_digest"], "candidate_digest", _SHA256),
            changed_paths=_string_sequence(data["changed_paths"], "changed_paths"),
            index_delta=index_delta,
            receipt_sha256=_hex_digest(data["receipt_sha256"], "receipt_sha256", _SHA256),
            schema_version=schema_version,
        )
    except (PolicyViolation, ValueError) as error:
        raise CandidateReceiptError(f"invalid candidate receipt: {error}") from error
    return receipt


def reconstruct_candidate(
    receipt_path: Path,
    *,
    worktree: Path,
    proposal: WorkItemProposal,
    git: ControllerGitOps,
) -> CandidateBinding:
    """Verify a persisted receipt, exact worktree HEAD, and one-parent patch."""
    receipt = load_candidate_receipt(receipt_path)
    if receipt.item_id != proposal.item_id:
        raise CandidateReceiptError("candidate receipt belongs to another WorkItem")
    if receipt.index_delta is not None:
        _validate_index_delta(receipt.index_delta, proposal)
    repository = worktree.expanduser().resolve(strict=True)
    inspection = git.inspect(repository)
    if inspection.head != receipt.candidate_commit:
        raise CandidateReceiptError(
            "candidate worktree HEAD differs from immutable candidate receipt"
        )
    if not inspection.clean:
        raise CandidateReceiptError("candidate worktree is not clean after binding")
    if receipt.disposition is CandidateDisposition.VERIFICATION:
        if proposal.modifies_files:
            raise CandidateReceiptError("verification receipt cannot satisfy modifying work")
        if receipt.candidate_commit != receipt.base_commit or inspection.changed_paths:
            raise CandidateReceiptError("verification receipt does not prove an unchanged worktree")
    else:
        if not proposal.modifies_files:
            raise CandidateReceiptError("patch receipt cannot satisfy read-only verification")
        try:
            verification = git.verify_candidate(
                repository,
                base_commit=receipt.base_commit,
                candidate_commit=receipt.candidate_commit,
                allowed_paths=proposal.allowed_paths,
                expected_patch_sha256=receipt.candidate_digest,
            )
        except GitOpsError as error:
            raise CandidateReceiptError(f"candidate Git verification failed: {error}") from error
        if verification.changed_paths != receipt.changed_paths:
            raise CandidateReceiptError("candidate receipt changed paths differ from Git")
        _verify_controller_commit(
            repository,
            verification.candidate_commit,
            item_id=proposal.item_id,
        )
    snapshot = CandidateSnapshot(
        item_id=receipt.item_id,
        coder_attempt_id=receipt.attempt_id,
        repository=repository,
        base_commit=receipt.base_commit,
        candidate_commit=receipt.candidate_commit,
        candidate_digest=receipt.candidate_digest,
        changed_paths=receipt.changed_paths,
    )
    return CandidateBinding(receipt, snapshot)


def _manifest_attempt(manifest: WorkerResultManifest, attempt: AttemptRecord) -> None:
    if (
        manifest.item_id != attempt.item_id
        or manifest.attempt_id != attempt.attempt_id
        or manifest.generation != attempt.generation
    ):
        raise ResultContractError("worker result identity differs from its attempt")
    if attempt.status is not AttemptStatus.VALIDATED:
        raise ResultContractError("worker result decoding requires a validated attempt")


def _candidate_pin(
    manifest: WorkerResultManifest,
    attempt: AttemptRecord,
    candidate: CandidateReceipt,
    candidate_attempt_id: str,
    candidate_digest: str,
) -> None:
    if manifest.candidate_digest is not None:
        raise ResultContractError("review/gate/QA worker cannot assign candidate_digest")
    expected = (candidate.attempt_id, candidate.candidate_digest)
    if (candidate_attempt_id, candidate_digest) != expected:
        raise ResultContractError("worker payload is not pinned to the candidate receipt")
    if attempt.attempt_id == candidate.attempt_id:
        raise ResultContractError("review/gate/QA must use a fresh attempt")
    if manifest.run_id != candidate.run_id:
        raise ResultContractError("candidate receipt belongs to another run")
    if manifest.item_id != candidate.item_id:
        raise ResultContractError("candidate receipt belongs to another WorkItem")
    if manifest.reviewed_candidate_digest != candidate.candidate_digest:
        raise ResultContractError("manifest reviewer linkage differs from candidate")
    if attempt.kind in {AttemptKind.REVIEWER_ANALYSIS, AttemptKind.REVIEWER_RERUN} and (
        attempt.review_of_attempt_id != candidate.attempt_id
        or attempt.reviewed_candidate_digest != candidate.candidate_digest
    ):
        raise ResultContractError("attempt or manifest reviewer linkage differs from candidate")


def _status_for_decision(status: WorkerResultStatus, decision: str, label: str) -> None:
    expected = {
        "approve": WorkerResultStatus.SUCCEEDED,
        "reject": WorkerResultStatus.REJECTED,
        "blocked": WorkerResultStatus.BLOCKED,
    }[decision]
    if status is not expected:
        raise ResultContractError(f"{label} worker status disagrees with typed verdict")


def _validate_index_delta(delta: IndexDelta, proposal: WorkItemProposal) -> None:
    if proposal.kind not in CATALOG_ITEM_KINDS or proposal.entry_id is None:
        raise ResultContractError("IndexDelta requires an atomic catalog WorkItem")
    if delta.item_id != proposal.item_id or delta.row.entry_id != proposal.entry_id:
        raise ResultContractError("IndexDelta identity differs from the frozen WorkItem")
    wrapper = f"tensorrt_llm/_torch/modeling_v2/catalog/{delta.row.path}"
    if wrapper not in proposal.allowed_paths:
        raise ResultContractError("IndexDelta wrapper is outside the frozen path claim")


def _validate_gate_spec(receipt: GateReceipt, spec: GateSpec) -> None:
    if receipt.gate_id != spec.gate_id or receipt.purpose is not spec.purpose:
        raise ResultContractError("gate receipt identity differs from frozen GateSpec")
    if receipt.accuracy != spec.accuracy:
        raise ResultContractError("gate receipt accuracy criteria differ from frozen GateSpec")


def _gate_receipt(value: object) -> GateReceipt:
    data = _object(value, "gate receipt")
    _exact_keys(
        data,
        {
            "gate_id",
            "purpose",
            "scope",
            "placements",
            "product_rank_body",
            "passed",
            "accuracy",
        },
        "gate receipt",
    )
    placements = tuple(_rank_placement(raw) for raw in _sequence(data["placements"], "placements"))
    raw_accuracy = data["accuracy"]
    accuracy: AccuracyCriteria | None
    if raw_accuracy is None:
        accuracy = None
    else:
        accuracy_data = _object(raw_accuracy, "accuracy criteria")
        _exact_keys(
            accuracy_data,
            {"selector", "reference", "protocol", "tolerance"},
            "accuracy criteria",
        )
        tolerance = accuracy_data["tolerance"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise ResultContractError("accuracy tolerance must be numeric")
        accuracy = AccuracyCriteria(
            _string(accuracy_data["selector"], "accuracy selector"),
            _string(accuracy_data["reference"], "accuracy reference"),
            _string(accuracy_data["protocol"], "accuracy protocol"),
            float(tolerance),
        )
    try:
        return GateReceipt(
            gate_id=_string(data["gate_id"], "gate_id"),
            purpose=_enum(data["purpose"], GatePurpose, "gate purpose"),
            scope=_enum(data["scope"], EvidenceScope, "gate scope"),
            placements=placements,
            product_rank_body=_boolean(data["product_rank_body"], "product_rank_body"),
            passed=_boolean(data["passed"], "passed"),
            accuracy=accuracy,
        )
    except ValueError as error:
        raise ResultContractError(f"invalid gate receipt: {error}") from error


def _rank_placement(value: object) -> RankPlacement:
    data = _object(value, "rank placement")
    _exact_keys(data, {"rank", "node", "local_rank"}, "rank placement")
    try:
        return RankPlacement(
            _integer(data["rank"], "rank"),
            _string(data["node"], "node"),
            _integer(data["local_rank"], "local_rank"),
        )
    except ValueError as error:
        raise ResultContractError(f"invalid rank placement: {error}") from error


def _evidence(
    value: object,
    expected: tuple[EvidenceFile, ...],
    label: str,
) -> tuple[EvidenceFile, ...]:
    raw_entries = _sequence(value, label)
    entries: list[EvidenceFile] = []
    for raw in raw_entries:
        data = _object(raw, label)
        _exact_keys(data, {"path", "sha256", "size_bytes"}, label)
        try:
            entries.append(
                EvidenceFile(
                    _string(data["path"], "evidence path"),
                    _hex_digest(data["sha256"], "evidence sha256", _SHA256),
                    _integer(data["size_bytes"], "evidence size_bytes"),
                )
            )
        except ValueError as error:
            raise ResultContractError(f"invalid {label}: {error}") from error
    result = tuple(entries)
    if result != expected:
        raise ResultContractError(f"{label} differs from WorkerResultManifest evidence")
    return result


def _new_candidate_receipt(
    manifest: WorkerResultManifest,
    *,
    disposition: CandidateDisposition,
    base_commit: str,
    candidate_commit: str,
    candidate_digest: str,
    changed_paths: tuple[str, ...],
    index_delta: IndexDelta | None,
) -> CandidateReceipt:
    payload = {
        "schema_version": CANDIDATE_RECEIPT_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "disposition": disposition.value,
        "base_commit": base_commit,
        "candidate_commit": candidate_commit,
        "candidate_digest": candidate_digest,
        "changed_paths": list(changed_paths),
        "index_delta": None if index_delta is None else _index_delta_payload(index_delta),
    }
    receipt_digest = _canonical_digest(payload)
    return CandidateReceipt(
        manifest.run_id,
        manifest.item_id,
        manifest.attempt_id,
        disposition,
        base_commit,
        candidate_commit,
        candidate_digest,
        changed_paths,
        index_delta,
        receipt_digest,
    )


def _receipt_digest(receipt: CandidateReceipt) -> str:
    payload = _receipt_payload(receipt, include_digest=False)
    return _canonical_digest(payload)


def _canonical_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verification_candidate_digest(
    manifest: WorkerResultManifest,
    base_commit: str,
) -> str:
    """Bind a no-change candidate to typed worker input and immutable evidence."""
    return _canonical_digest(
        {
            "kind": CandidateDisposition.VERIFICATION.value,
            "run_id": manifest.run_id,
            "item_id": manifest.item_id,
            "attempt_id": manifest.attempt_id,
            "task_digest": manifest.task_digest,
            "generation": manifest.generation,
            "input_digest": manifest.input_digest,
            "base_commit": base_commit,
            "payload": manifest.payload,
            "evidence": [
                {
                    "path": evidence.path,
                    "sha256": evidence.sha256,
                    "size_bytes": evidence.size_bytes,
                }
                for evidence in manifest.evidence
            ],
        }
    )


def _tuner_candidate_digest(
    manifest: WorkerResultManifest,
    *,
    base_commit: str,
    artifact_digests: tuple[tuple[str, str], ...],
) -> str:
    """Bind Tuner review identity to the exact result and evidence-ready artifacts."""
    return _canonical_digest(
        {
            "kind": "tuner_verification",
            "base_commit": base_commit,
            "manifest": {
                "run_id": manifest.run_id,
                "item_id": manifest.item_id,
                "attempt_id": manifest.attempt_id,
                "task_digest": manifest.task_digest,
                "generation": manifest.generation,
                "input_digest": manifest.input_digest,
                "status": manifest.status.value,
                "summary": manifest.summary,
                "payload": manifest.payload,
                "evidence": [
                    {
                        "path": evidence.path,
                        "sha256": evidence.sha256,
                        "size_bytes": evidence.size_bytes,
                    }
                    for evidence in manifest.evidence
                ],
            },
            "artifacts": [
                {"filename": filename, "sha256": digest} for filename, digest in artifact_digests
            ],
        }
    )


def _verify_controller_commit(
    repository: Path,
    candidate_commit: str,
    *,
    item_id: str,
) -> None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "show",
            "-s",
            "--format=%B",
            candidate_commit,
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
        raise CandidateReceiptError(f"cannot inspect candidate DCO trailer: {detail}")
    message = result.stdout.decode("utf-8", errors="strict")
    lines = message.splitlines()
    if not lines or lines[0] != f"staircase: bind {item_id} candidate":
        raise CandidateReceiptError("candidate commit lacks the controller-owned subject")
    if not any(re.fullmatch(r"Signed-off-by: .+ <[^<>\s]+@[^<>\s]+>", line) for line in lines):
        raise CandidateReceiptError("candidate commit lacks a valid DCO Signed-off-by trailer")


def _receipt_payload(receipt: CandidateReceipt, *, include_digest: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": receipt.schema_version,
        "run_id": receipt.run_id,
        "item_id": receipt.item_id,
        "attempt_id": receipt.attempt_id,
        "disposition": receipt.disposition.value,
        "base_commit": receipt.base_commit,
        "candidate_commit": receipt.candidate_commit,
        "candidate_digest": receipt.candidate_digest,
        "changed_paths": list(receipt.changed_paths),
        "index_delta": (
            None if receipt.index_delta is None else _index_delta_payload(receipt.index_delta)
        ),
    }
    if include_digest:
        payload["receipt_sha256"] = receipt.receipt_sha256
    return payload


def _index_delta_payload(delta: IndexDelta) -> dict[str, object]:
    return {
        "item_id": delta.item_id,
        "row": {
            "entry_id": delta.row.entry_id,
            "path": delta.row.path,
            "implementation": delta.row.implementation,
            "summary": delta.row.summary,
        },
        "certification_cells": [
            {
                "entry_path": cell.entry_path,
                "dimensions": dict(cell.dimensions),
            }
            for cell in delta.certification_cells
        ],
        "expected_row_sha256": delta.expected_row_sha256,
    }


def _write_receipt_once(path: Path, receipt: CandidateReceipt) -> None:
    parent = _canonical_candidate_directory(path.parent, "candidate receipt parent")
    if parent != path.parent or path.is_symlink():
        raise CandidateReceiptError("candidate receipt path contains a symlink")
    payload = _receipt_payload(receipt, include_digest=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, allow_nan=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = load_candidate_receipt(path)
            if existing != receipt:
                raise CandidateReceiptError("immutable candidate receipt already differs")
        _fsync_directory(parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_candidate_directory(path: Path, label: str) -> Path:
    """Require an existing directory whose leaf and ancestors are not symlinks."""
    expanded = Path(os.path.abspath(path.expanduser()))
    if expanded.is_symlink():
        raise CandidateReceiptError(f"{label} cannot be a symlink")
    try:
        resolved = expanded.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise CandidateReceiptError(f"{label} does not resolve: {error}") from error
    if resolved != expanded:
        raise CandidateReceiptError(f"{label} path contains a symlink")
    if not resolved.is_dir():
        raise CandidateReceiptError(f"{label} must be an existing directory")
    return resolved


def _required_candidate(candidate: CandidateReceipt | None) -> CandidateReceipt:
    if candidate is None:
        raise ResultContractError("worker result requires a candidate receipt")
    return candidate


def _object(
    value: object,
    label: str,
) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ResultContractError(f"{label} must be an object with string keys")
    return value


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    label: str,
    *,
    error_type: type[Exception] = ResultContractError,
) -> None:
    actual = set(value)
    if actual != expected:
        raise error_type(
            f"{label} keys invalid; missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _schema(value: Mapping[str, object]) -> None:
    if _integer(value["schema_version"], "schema_version") != WORKER_RESULT_SCHEMA_VERSION:
        raise ResultContractError("unsupported worker result schema_version")


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "\n" in value:
        raise ResultContractError(f"{label} must be a non-empty single-line string")
    return value


def _safe_id(value: object, label: str) -> str:
    text = _string(value, label)
    if not _SAFE_ID.fullmatch(text):
        raise ResultContractError(f"{label} must be a safe identifier")
    return text


def _hex_digest(value: object, label: str, pattern: re.Pattern[str]) -> str:
    text = _string(value, label)
    if not pattern.fullmatch(text):
        raise ResultContractError(f"{label} has an invalid digest")
    return text


def _digest(value: str, label: str, pattern: re.Pattern[str]) -> None:
    if not pattern.fullmatch(value):
        raise CandidateReceiptError(f"{label} has an invalid digest")


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultContractError(f"{label} must be a non-negative integer")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ResultContractError(f"{label} must be a boolean")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ResultContractError(f"{label} must be a list")
    return value


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    return tuple(_string(entry, label) for entry in _sequence(value, label))


def _nonempty_strings(value: object, label: str, *, allow_empty: bool) -> tuple[str, ...]:
    result = _string_sequence(value, label)
    if not allow_empty and not result:
        raise ResultContractError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise ResultContractError(f"{label} contains duplicates")
    return result


def _changed_paths(paths: tuple[str, ...]) -> None:
    if len(paths) != len(set(paths)):
        raise ResultContractError("changed_paths contains duplicates")
    for path in paths:
        pure = PurePosixPath(path)
        if (
            pure.is_absolute()
            or path != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise ResultContractError(f"unsafe changed path {path!r}")


def _literal(value: object, expected: str, label: str) -> None:
    if value != expected:
        raise ResultContractError(f"{label} must be {expected!r}")


def _enum(value: object, enum_type: type[Enum], label: str):  # type: ignore[no-untyped-def]
    text = _string(value, label)
    try:
        return enum_type(text)
    except ValueError as error:
        raise ResultContractError(f"unsupported {label} {text!r}") from error


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CANDIDATE_RECEIPT_FILENAME",
    "CandidateBinding",
    "CandidateDisposition",
    "CandidateReceipt",
    "CandidateReceiptError",
    "CoderResult",
    "GateResult",
    "QaGateReference",
    "QaResult",
    "QaVerdict",
    "ResultContractError",
    "ReviewerResult",
    "ReviewerVerdict",
    "TypedWorkerResult",
    "TunerResult",
    "bind_coder_candidate",
    "bind_tuner_candidate",
    "decode_coder_result",
    "decode_gate_result",
    "decode_qa_result",
    "decode_resource_escalation",
    "decode_reviewer_result",
    "decode_tuner_result",
    "decode_worker_result",
    "load_candidate_receipt",
    "reconstruct_candidate",
]
