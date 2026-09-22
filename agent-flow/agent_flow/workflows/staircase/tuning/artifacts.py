# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write-once, content-addressed artifacts for deterministic tuning replay."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, TypeVar, cast

from ..common.policy import WorkItemProposal
from .contracts import TuningHypothesis
from .workflow import (
    BaselineMeasurement,
    CampaignStatus,
    CandidateMeasurement,
    DecisionOwner,
    HardGateKind,
    HardGateResult,
    MeasurementCurve,
    MeasurementIdentity,
    PromotionAction,
    PromotionDecision,
    RouteIdentity,
    TopologyIdentity,
    TuningCampaign,
    TuningContractError,
    TuningEvaluation,
    TuningOutcome,
    evaluate_tuning_campaign,
    finalize_tuning_campaign,
    make_promotion_decision,
    new_tuning_campaign,
    record_tuning_evidence,
)

TUNING_ARTIFACT_SCHEMA_VERSION = 1
BASELINE_FILENAME = "baseline.json"
CANDIDATE_FILENAME = "candidate.json"
EVALUATION_FILENAME = "evaluation.json"
CAMPAIGN_FILENAME = "campaign.json"
PROMOTION_DECISION_FILENAME = "promotion-decision.json"

_T = TypeVar("_T")


class TuningArtifactError(RuntimeError):
    """Base error for an invalid tuning artifact or artifact operation."""


class ImmutableTuningArtifactError(TuningArtifactError):
    """Raised when a writer tries to replace a write-once artifact."""


class TuningArtifactDigestError(TuningArtifactError):
    """Raised when persisted content disagrees with its canonical digest."""


class TuningReplayError(TuningArtifactError):
    """Raised when durable tuning artifacts cannot reproduce their decision."""


class TuningArtifactKind(str, Enum):
    """Closed vocabulary for durable tuning artifact envelopes."""

    BASELINE = "tuning_baseline"
    CANDIDATE = "tuning_candidate"
    EVALUATION = "tuning_evaluation"
    CAMPAIGN = "tuning_campaign"
    PROMOTION_DECISION = "tuning_promotion_decision"


@dataclass(frozen=True, slots=True)
class TuningArtifactReceipt:
    """Path and canonical payload digest of one durable artifact."""

    kind: TuningArtifactKind
    path: Path
    digest: str


@dataclass(frozen=True, slots=True)
class PersistedTuningCampaign:
    """Evidence-ready campaign, terminal decision, and durable receipts."""

    campaign: TuningCampaign
    decision: PromotionDecision
    receipts: tuple[TuningArtifactReceipt, ...]


TuningArtifactValue = (
    BaselineMeasurement
    | CandidateMeasurement
    | TuningEvaluation
    | TuningCampaign
    | PromotionDecision
)


def canonical_tuning_artifact_bytes(value: TuningArtifactValue) -> bytes:
    """Return the exact canonical envelope bytes written for ``value``."""
    kind, payload = _kind_and_payload(value)
    envelope = _envelope(kind, payload)
    return _canonical_json(envelope) + b"\n"


def canonical_tuning_digest(value: TuningArtifactValue) -> str:
    """Return the canonical payload digest for ``value``."""
    _kind, payload = _kind_and_payload(value)
    return _canonical_digest(payload)


def write_baseline_artifact(path: Path, value: BaselineMeasurement) -> TuningArtifactReceipt:
    """Durably publish one immutable baseline artifact."""
    return _write_artifact(path, TuningArtifactKind.BASELINE, value)


def write_candidate_artifact(path: Path, value: CandidateMeasurement) -> TuningArtifactReceipt:
    """Durably publish one immutable candidate artifact."""
    return _write_artifact(path, TuningArtifactKind.CANDIDATE, value)


def write_evaluation_artifact(path: Path, value: TuningEvaluation) -> TuningArtifactReceipt:
    """Durably publish one immutable evaluation artifact."""
    return _write_artifact(path, TuningArtifactKind.EVALUATION, value)


def write_campaign_artifact(path: Path, value: TuningCampaign) -> TuningArtifactReceipt:
    """Durably publish one immutable campaign artifact."""
    return _write_artifact(path, TuningArtifactKind.CAMPAIGN, value)


def write_promotion_decision_artifact(
    path: Path,
    value: PromotionDecision,
    *,
    evaluation: TuningEvaluation,
    owner: DecisionOwner,
) -> TuningArtifactReceipt:
    """Durably publish an evidence-bound decision under controller authority."""
    if owner is not DecisionOwner.CONTROLLER or value.owner is not owner:
        raise TuningContractError("only the explicit controller owner may persist a decision")
    expected = make_promotion_decision(evaluation, owner=owner)
    if value != expected:
        raise TuningContractError("promotion decision disagrees with its immutable evaluation")
    return _write_artifact(path, TuningArtifactKind.PROMOTION_DECISION, value)


def load_baseline_artifact(path: Path) -> tuple[BaselineMeasurement, str]:
    """Load and verify one canonical baseline artifact."""
    return _load_artifact(path, TuningArtifactKind.BASELINE, _baseline_from_payload)


def load_candidate_artifact(path: Path) -> tuple[CandidateMeasurement, str]:
    """Load and verify one canonical candidate artifact."""
    return _load_artifact(path, TuningArtifactKind.CANDIDATE, _candidate_from_payload)


def parse_baseline_measurement_payload(value: object) -> BaselineMeasurement:
    """Strictly decode a worker-supplied baseline measurement payload."""
    return _baseline_from_payload(_object(value, "baseline measurement"))


def parse_candidate_measurement_payload(value: object) -> CandidateMeasurement:
    """Strictly decode a worker-supplied candidate measurement payload."""
    return _candidate_from_payload(_object(value, "candidate measurement"))


def load_evaluation_artifact(path: Path) -> tuple[TuningEvaluation, str]:
    """Load and verify one canonical evaluation artifact."""
    return _load_artifact(path, TuningArtifactKind.EVALUATION, _evaluation_from_payload)


def load_campaign_artifact(path: Path) -> tuple[TuningCampaign, str]:
    """Load and verify one canonical campaign artifact."""
    return _load_artifact(path, TuningArtifactKind.CAMPAIGN, _campaign_from_payload)


def load_promotion_decision_artifact(path: Path) -> tuple[PromotionDecision, str]:
    """Load and verify one canonical promotion-decision artifact."""
    return _load_artifact(path, TuningArtifactKind.PROMOTION_DECISION, _decision_from_payload)


def persist_terminal_tuning_campaign(
    root: Path,
    *,
    item: WorkItemProposal,
    hypothesis: TuningHypothesis,
    baseline: BaselineMeasurement,
    candidate: CandidateMeasurement,
    owner: DecisionOwner,
) -> PersistedTuningCampaign:
    """Evaluate and publish the immutable artifacts needed for restart replay.

    The campaign artifact intentionally records ``EVIDENCE_READY``. The
    controller-owned promotion decision is a separate artifact written last,
    so a restart can replay the exact transition to ``TERMINAL``.
    """
    evaluation = evaluate_tuning_campaign(item, hypothesis, baseline, candidate)
    evidence_ready = record_tuning_evidence(
        new_tuning_campaign(item, hypothesis),
        evaluation,
    )
    decision = make_promotion_decision(evaluation, owner=owner)
    receipts = (
        write_baseline_artifact(root / BASELINE_FILENAME, baseline),
        write_candidate_artifact(root / CANDIDATE_FILENAME, candidate),
        write_evaluation_artifact(root / EVALUATION_FILENAME, evaluation),
        write_campaign_artifact(root / CAMPAIGN_FILENAME, evidence_ready),
        write_promotion_decision_artifact(
            root / PROMOTION_DECISION_FILENAME,
            decision,
            evaluation=evaluation,
            owner=owner,
        ),
    )
    return PersistedTuningCampaign(
        campaign=finalize_tuning_campaign(evidence_ready, owner=owner),
        decision=decision,
        receipts=receipts,
    )


def replay_terminal_tuning_campaign(
    root: Path,
    *,
    item: WorkItemProposal,
    hypothesis: TuningHypothesis,
    owner: DecisionOwner,
) -> TuningCampaign:
    """Reload artifacts and deterministically replay the terminal decision."""
    baseline, _baseline_digest = load_baseline_artifact(root / BASELINE_FILENAME)
    candidate, _candidate_digest = load_candidate_artifact(root / CANDIDATE_FILENAME)
    recorded_evaluation, _evaluation_digest = load_evaluation_artifact(root / EVALUATION_FILENAME)
    recorded_campaign, _campaign_digest = load_campaign_artifact(root / CAMPAIGN_FILENAME)
    recorded_decision, _decision_digest = load_promotion_decision_artifact(
        root / PROMOTION_DECISION_FILENAME
    )

    replayed_evaluation = evaluate_tuning_campaign(item, hypothesis, baseline, candidate)
    if replayed_evaluation != recorded_evaluation:
        raise TuningReplayError("replayed evaluation differs from durable evaluation")
    replayed_campaign = record_tuning_evidence(
        new_tuning_campaign(item, hypothesis),
        replayed_evaluation,
    )
    if replayed_campaign != recorded_campaign:
        raise TuningReplayError("replayed campaign differs from durable campaign")
    replayed_decision = make_promotion_decision(replayed_evaluation, owner=owner)
    if replayed_decision != recorded_decision:
        raise TuningReplayError("replayed decision differs from durable promotion decision")
    decision_path = root / PROMOTION_DECISION_FILENAME
    if canonical_tuning_artifact_bytes(replayed_decision) != _read_artifact_bytes(decision_path):
        raise TuningReplayError("replayed promotion decision is not byte-identical")
    return finalize_tuning_campaign(replayed_campaign, owner=owner)


def _write_artifact(
    path: Path,
    expected_kind: TuningArtifactKind,
    value: TuningArtifactValue,
) -> TuningArtifactReceipt:
    kind, payload = _kind_and_payload(value)
    if kind is not expected_kind:
        raise TypeError(f"{expected_kind.value} writer received {kind.value}")
    digest = _canonical_digest(payload)
    _write_once_bytes(path, canonical_tuning_artifact_bytes(value))
    return TuningArtifactReceipt(kind=kind, path=path, digest=digest)


def _load_artifact(
    path: Path,
    expected_kind: TuningArtifactKind,
    parser: Callable[[dict[str, object]], _T],
) -> tuple[_T, str]:
    encoded = _read_artifact_bytes(path)
    raw = _parse_canonical_object(encoded, path)
    _expect_keys(raw, {"schema_version", "kind", "digest", "payload"}, "artifact envelope")
    if raw["schema_version"] != TUNING_ARTIFACT_SCHEMA_VERSION:
        raise TuningArtifactError(f"unsupported tuning artifact schema in {path}")
    if raw["kind"] != expected_kind.value:
        raise TuningArtifactError(
            f"expected {expected_kind.value!r} in {path}, got {raw['kind']!r}"
        )
    payload = _object(raw["payload"], "artifact payload")
    recorded_digest = _string(raw["digest"], "artifact digest")
    actual_digest = _canonical_digest(payload)
    if recorded_digest != actual_digest:
        raise TuningArtifactDigestError(f"artifact digest mismatch in {path}")
    value = parser(payload)
    kind, canonical_payload = _kind_and_payload(cast(TuningArtifactValue, value))
    if kind is not expected_kind or canonical_payload != payload:
        raise TuningArtifactError(f"non-canonical {expected_kind.value} payload in {path}")
    return value, recorded_digest


def _kind_and_payload(
    value: TuningArtifactValue,
) -> tuple[TuningArtifactKind, dict[str, object]]:
    if isinstance(value, BaselineMeasurement):
        return TuningArtifactKind.BASELINE, _measurement_to_payload(value)
    if isinstance(value, CandidateMeasurement):
        return TuningArtifactKind.CANDIDATE, _measurement_to_payload(value)
    if isinstance(value, TuningEvaluation):
        return TuningArtifactKind.EVALUATION, _evaluation_to_payload(value)
    if isinstance(value, TuningCampaign):
        return TuningArtifactKind.CAMPAIGN, _campaign_to_payload(value)
    if isinstance(value, PromotionDecision):
        return TuningArtifactKind.PROMOTION_DECISION, _decision_to_payload(value)
    raise TypeError(f"unsupported tuning artifact value {type(value).__name__}")


def _measurement_to_payload(
    value: BaselineMeasurement | CandidateMeasurement,
) -> dict[str, object]:
    return {
        "identity": _identity_to_payload(value.identity),
        "arm_digest": value.arm_digest,
        "curve": {
            "samples": list(value.curve.samples),
            "uncertainty": value.curve.uncertainty,
        },
        "gates": [
            {
                "name": gate.name,
                "kind": gate.kind.value,
                "passed": gate.passed,
                "evidence_digest": gate.evidence_digest,
            }
            for gate in value.gates
        ],
    }


def _identity_to_payload(value: MeasurementIdentity) -> dict[str, object]:
    return {
        "checkpoint": value.checkpoint,
        "target": value.target,
        "route": {
            "family": value.route.family,
            "architecture": value.route.architecture,
            "expected_route": value.route.expected_route,
            "synthetic_target": value.route.synthetic_target,
        },
        "workload": value.workload,
        "topology": {
            "world_size": value.topology.world_size,
            "tensor_parallel_size": value.topology.tensor_parallel_size,
            "pipeline_parallel_size": value.topology.pipeline_parallel_size,
            "moe_expert_parallel_size": value.topology.moe_expert_parallel_size,
            "moe_tensor_parallel_size": value.topology.moe_tensor_parallel_size,
            "attention_data_parallel_size": value.topology.attention_data_parallel_size,
        },
        "build": value.build,
        "protocol": value.protocol,
        "hardware": value.hardware,
    }


def _evaluation_to_payload(value: TuningEvaluation) -> dict[str, object]:
    return {
        "item_id": value.item_id,
        "hypothesis_id": value.hypothesis_id,
        "outcome": value.outcome.value,
        "baseline_mean": value.baseline_mean,
        "candidate_mean": value.candidate_mean,
        "oriented_effect": value.oriented_effect,
        "combined_uncertainty": value.combined_uncertainty,
        "decisive_effect": value.decisive_effect,
        "failing_gates": list(value.failing_gates),
        "reason": value.reason,
    }


def _decision_to_payload(value: PromotionDecision) -> dict[str, object]:
    return {
        "item_id": value.item_id,
        "hypothesis_id": value.hypothesis_id,
        "outcome": value.outcome.value,
        "action": value.action.value,
        "owner": value.owner.value,
        "reason": value.reason,
    }


def _campaign_to_payload(value: TuningCampaign) -> dict[str, object]:
    return {
        "item_id": value.item_id,
        "hypothesis_id": value.hypothesis_id,
        "status": value.status.value,
        "evaluation": (
            _evaluation_to_payload(value.evaluation) if value.evaluation is not None else None
        ),
        "decision": _decision_to_payload(value.decision) if value.decision is not None else None,
    }


def _baseline_from_payload(payload: dict[str, object]) -> BaselineMeasurement:
    identity, arm_digest, curve, gates = _measurement_from_payload(payload, "baseline")
    return BaselineMeasurement(
        identity=identity,
        arm_digest=arm_digest,
        curve=curve,
        gates=gates,
    )


def _candidate_from_payload(payload: dict[str, object]) -> CandidateMeasurement:
    identity, arm_digest, curve, gates = _measurement_from_payload(payload, "candidate")
    return CandidateMeasurement(
        identity=identity,
        arm_digest=arm_digest,
        curve=curve,
        gates=gates,
    )


def _measurement_from_payload(
    payload: dict[str, object], name: str
) -> tuple[MeasurementIdentity, str, MeasurementCurve, tuple[HardGateResult, ...]]:
    _expect_keys(payload, {"identity", "arm_digest", "curve", "gates"}, name)
    curve_raw = _object(payload["curve"], f"{name} curve")
    _expect_keys(curve_raw, {"samples", "uncertainty"}, f"{name} curve")
    gates_raw = _list(payload["gates"], f"{name} gates")
    return (
        _identity_from_payload(_object(payload["identity"], f"{name} identity")),
        _string(payload["arm_digest"], f"{name} arm_digest"),
        MeasurementCurve(
            samples=tuple(
                _number(value, "measurement sample")
                for value in _list(curve_raw["samples"], "measurement samples")
            ),
            uncertainty=_number(curve_raw["uncertainty"], "measurement uncertainty"),
        ),
        tuple(_gate_from_payload(_object(value, "hard gate")) for value in gates_raw),
    )


def _identity_from_payload(payload: dict[str, object]) -> MeasurementIdentity:
    _expect_keys(
        payload,
        {
            "checkpoint",
            "target",
            "route",
            "workload",
            "topology",
            "build",
            "protocol",
            "hardware",
        },
        "measurement identity",
    )
    route_raw = _object(payload["route"], "route identity")
    _expect_keys(
        route_raw,
        {"family", "architecture", "expected_route", "synthetic_target"},
        "route identity",
    )
    topology_raw = _object(payload["topology"], "topology identity")
    _expect_keys(
        topology_raw,
        {
            "world_size",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "moe_expert_parallel_size",
            "moe_tensor_parallel_size",
            "attention_data_parallel_size",
        },
        "topology identity",
    )
    return MeasurementIdentity(
        checkpoint=_string(payload["checkpoint"], "checkpoint"),
        target=_string(payload["target"], "target"),
        route=RouteIdentity(
            family=_string(route_raw["family"], "route family"),
            architecture=_string(route_raw["architecture"], "route architecture"),
            expected_route=_string(route_raw["expected_route"], "expected route"),
            synthetic_target=_boolean(route_raw["synthetic_target"], "synthetic_target"),
        ),
        workload=_string(payload["workload"], "workload"),
        topology=TopologyIdentity(
            world_size=_integer(topology_raw["world_size"], "world_size"),
            tensor_parallel_size=_integer(
                topology_raw["tensor_parallel_size"], "tensor_parallel_size"
            ),
            pipeline_parallel_size=_integer(
                topology_raw["pipeline_parallel_size"], "pipeline_parallel_size"
            ),
            moe_expert_parallel_size=_integer(
                topology_raw["moe_expert_parallel_size"], "moe_expert_parallel_size"
            ),
            moe_tensor_parallel_size=_integer(
                topology_raw["moe_tensor_parallel_size"], "moe_tensor_parallel_size"
            ),
            attention_data_parallel_size=_integer(
                topology_raw["attention_data_parallel_size"],
                "attention_data_parallel_size",
            ),
        ),
        build=_string(payload["build"], "build"),
        protocol=_string(payload["protocol"], "protocol"),
        hardware=_string(payload["hardware"], "hardware"),
    )


def _gate_from_payload(payload: dict[str, object]) -> HardGateResult:
    _expect_keys(payload, {"name", "kind", "passed", "evidence_digest"}, "hard gate")
    return HardGateResult(
        name=_string(payload["name"], "gate name"),
        kind=_enum(HardGateKind, payload["kind"], "gate kind"),
        passed=_boolean(payload["passed"], "gate passed"),
        evidence_digest=_string(payload["evidence_digest"], "gate evidence_digest"),
    )


def _evaluation_from_payload(payload: dict[str, object]) -> TuningEvaluation:
    _expect_keys(
        payload,
        {
            "item_id",
            "hypothesis_id",
            "outcome",
            "baseline_mean",
            "candidate_mean",
            "oriented_effect",
            "combined_uncertainty",
            "decisive_effect",
            "failing_gates",
            "reason",
        },
        "tuning evaluation",
    )
    return TuningEvaluation(
        item_id=_string(payload["item_id"], "evaluation item_id"),
        hypothesis_id=_string(payload["hypothesis_id"], "evaluation hypothesis_id"),
        outcome=_enum(TuningOutcome, payload["outcome"], "evaluation outcome"),
        baseline_mean=_number(payload["baseline_mean"], "baseline_mean"),
        candidate_mean=_number(payload["candidate_mean"], "candidate_mean"),
        oriented_effect=_number(payload["oriented_effect"], "oriented_effect"),
        combined_uncertainty=_number(payload["combined_uncertainty"], "combined_uncertainty"),
        decisive_effect=_number(payload["decisive_effect"], "decisive_effect"),
        failing_gates=tuple(
            _string(value, "failing gate")
            for value in _list(payload["failing_gates"], "failing_gates")
        ),
        reason=_string(payload["reason"], "evaluation reason"),
    )


def _decision_from_payload(payload: dict[str, object]) -> PromotionDecision:
    _expect_keys(
        payload,
        {"item_id", "hypothesis_id", "outcome", "action", "owner", "reason"},
        "promotion decision",
    )
    return PromotionDecision(
        item_id=_string(payload["item_id"], "decision item_id"),
        hypothesis_id=_string(payload["hypothesis_id"], "decision hypothesis_id"),
        outcome=_enum(TuningOutcome, payload["outcome"], "decision outcome"),
        action=_enum(PromotionAction, payload["action"], "promotion action"),
        owner=_enum(DecisionOwner, payload["owner"], "decision owner"),
        reason=_string(payload["reason"], "decision reason"),
    )


def _campaign_from_payload(payload: dict[str, object]) -> TuningCampaign:
    _expect_keys(
        payload,
        {"item_id", "hypothesis_id", "status", "evaluation", "decision"},
        "tuning campaign",
    )
    evaluation_raw = payload["evaluation"]
    decision_raw = payload["decision"]
    return TuningCampaign(
        item_id=_string(payload["item_id"], "campaign item_id"),
        hypothesis_id=_string(payload["hypothesis_id"], "campaign hypothesis_id"),
        status=_enum(CampaignStatus, payload["status"], "campaign status"),
        evaluation=(
            _evaluation_from_payload(_object(evaluation_raw, "campaign evaluation"))
            if evaluation_raw is not None
            else None
        ),
        decision=(
            _decision_from_payload(_object(decision_raw, "campaign decision"))
            if decision_raw is not None
            else None
        ),
    )


def _envelope(kind: TuningArtifactKind, payload: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": TUNING_ARTIFACT_SCHEMA_VERSION,
        "kind": kind.value,
        "digest": _canonical_digest(payload),
        "payload": payload,
    }


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _parse_canonical_object(encoded: bytes, path: Path) -> dict[str, object]:
    try:
        text = encoded.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TuningArtifactError(f"invalid JSON in {path}: {error}") from error
    raw = _object(value, str(path))
    if encoded != _canonical_json(raw) + b"\n":
        raise TuningArtifactError(f"non-canonical JSON encoding in {path}")
    return raw


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TuningArtifactError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise TuningArtifactError(f"non-finite JSON constant {value!r}")


def _write_once_bytes(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as error:
            raise ImmutableTuningArtifactError(
                f"immutable tuning artifact already exists: {path}"
            ) from error
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _read_artifact_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise TuningArtifactError(f"artifact is not a regular non-symlink file: {path}")
    return path.read_bytes()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _expect_keys(raw: dict[str, object], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing or unknown:
        raise TuningArtifactError(
            f"{name} fields invalid; missing={missing!r}, unknown={unknown!r}"
        )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TuningArtifactError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise TuningArtifactError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TuningArtifactError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TuningArtifactError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TuningArtifactError(f"{name} must be a number")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TuningArtifactError(f"{name} must be a boolean")
    return value


def _enum(enum_type: type[_T], value: object, name: str) -> _T:
    text = _string(value, name)
    try:
        return enum_type(text)  # type: ignore[call-arg]
    except ValueError as error:
        raise TuningArtifactError(f"invalid {name} {text!r}") from error


__all__ = [
    "BASELINE_FILENAME",
    "CAMPAIGN_FILENAME",
    "CANDIDATE_FILENAME",
    "EVALUATION_FILENAME",
    "ImmutableTuningArtifactError",
    "PROMOTION_DECISION_FILENAME",
    "PersistedTuningCampaign",
    "TUNING_ARTIFACT_SCHEMA_VERSION",
    "TuningArtifactDigestError",
    "TuningArtifactError",
    "TuningArtifactKind",
    "TuningArtifactReceipt",
    "TuningReplayError",
    "canonical_tuning_artifact_bytes",
    "canonical_tuning_digest",
    "load_baseline_artifact",
    "load_campaign_artifact",
    "load_candidate_artifact",
    "load_evaluation_artifact",
    "load_promotion_decision_artifact",
    "parse_baseline_measurement_payload",
    "parse_candidate_measurement_payload",
    "persist_terminal_tuning_campaign",
    "replay_terminal_tuning_campaign",
    "write_baseline_artifact",
    "write_campaign_artifact",
    "write_candidate_artifact",
    "write_evaluation_artifact",
    "write_promotion_decision_artifact",
]
