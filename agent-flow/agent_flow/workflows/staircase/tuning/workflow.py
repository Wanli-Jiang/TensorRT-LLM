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

"""Pure contracts and decisions for one-variable Staircase tuning campaigns.

This module is deliberately below the scheduler, state store, worker mailbox,
and controller-owned Git layers.  It consumes their shared
:class:`~agent_flow.workflows.staircase.common.policy.WorkItemProposal`
identity instead of defining a parallel work queue.  Its immutable values can
be carried in the existing worker input/result manifests and candidate
worktrees without granting this layer scheduler, filesystem, or promotion
authority.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from enum import Enum
from statistics import fmean

from ..common.policy import WorkItemKind, WorkItemProposal
from .contracts import (
    JsonPrimitive,
    KnobChange,
    KnobKind,
    MetricDirection,
    TuningContractError,
    TuningHypothesis,
    UncertaintyRule,
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HardGateKind(str, Enum):
    """Correctness gates that performance can never override."""

    CORRECTNESS = "correctness"
    FEATURE = "feature"


class TuningOutcome(str, Enum):
    """Closed terminal vocabulary for a measured hypothesis."""

    KEEP = "keep"
    INVALID_BASELINE = "invalid_baseline"
    HARD_GATE_FAILURE = "hard_gate_failure"
    REGRESSION = "regression"
    NO_CHANGE = "no_change"
    NOISE = "noise"


class PromotionAction(str, Enum):
    """Controller-owned disposition of a measured candidate."""

    KEEP = "keep"
    REJECT = "reject"


class DecisionOwner(str, Enum):
    """Actor requesting a promotion decision."""

    CONTROLLER = "controller"
    CODER = "coder"
    REVIEWER = "reviewer"
    QA = "qa"


class CampaignStatus(str, Enum):
    """Pure tuning-specific projection onto the shared item lifecycle."""

    PLANNED = "planned"
    EVIDENCE_READY = "evidence_ready"
    TERMINAL = "terminal"


@dataclass(frozen=True, kw_only=True)
class RouteIdentity:
    """ModelingV2 route identity that a tuning knob may not alter."""

    family: str
    architecture: str
    expected_route: str
    synthetic_target: bool

    def __post_init__(self) -> None:
        _require_text("route family", self.family)
        _require_text("route architecture", self.architecture)
        _require_text("expected route", self.expected_route)
        if not isinstance(self.synthetic_target, bool):
            raise TuningContractError("synthetic_target must be a boolean")


@dataclass(frozen=True, kw_only=True)
class TopologyIdentity:
    """Full parallel mapping held constant across the A/B comparison."""

    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    moe_expert_parallel_size: int
    moe_tensor_parallel_size: int
    attention_data_parallel_size: int

    def __post_init__(self) -> None:
        for name, value in (
            ("world_size", self.world_size),
            ("tensor_parallel_size", self.tensor_parallel_size),
            ("pipeline_parallel_size", self.pipeline_parallel_size),
            ("moe_expert_parallel_size", self.moe_expert_parallel_size),
            ("moe_tensor_parallel_size", self.moe_tensor_parallel_size),
            ("attention_data_parallel_size", self.attention_data_parallel_size),
        ):
            _require_positive_integer(name, value)
        product = self.tensor_parallel_size * self.pipeline_parallel_size
        if self.world_size % product != 0:
            raise TuningContractError("world_size must be divisible by TP x PP")


@dataclass(frozen=True, kw_only=True)
class MeasurementIdentity:
    """Every experimental dimension that must match between A and B."""

    checkpoint: str
    target: str
    route: RouteIdentity
    workload: str
    topology: TopologyIdentity
    build: str
    protocol: str
    hardware: str

    def __post_init__(self) -> None:
        for name, value in (
            ("checkpoint", self.checkpoint),
            ("target", self.target),
            ("workload", self.workload),
            ("build", self.build),
            ("protocol", self.protocol),
            ("hardware", self.hardware),
        ):
            _require_text(name, value)
        if not isinstance(self.route, RouteIdentity):
            raise TuningContractError("route must be a RouteIdentity")
        if not isinstance(self.topology, TopologyIdentity):
            raise TuningContractError("topology must be a TopologyIdentity")


@dataclass(frozen=True, kw_only=True)
class HardGateResult:
    """One content-addressed correctness or feature gate result."""

    name: str
    kind: HardGateKind
    passed: bool
    evidence_digest: str

    def __post_init__(self) -> None:
        _require_text("gate name", self.name)
        if not isinstance(self.kind, HardGateKind):
            raise TuningContractError("gate kind must be correctness or feature")
        if not isinstance(self.passed, bool):
            raise TuningContractError("gate passed must be a boolean")
        _require_digest("gate evidence_digest", self.evidence_digest)

    @property
    def identity(self) -> tuple[HardGateKind, str]:
        """Return the stable matched-gate identity."""
        return (self.kind, self.name)


@dataclass(frozen=True, kw_only=True)
class MeasurementCurve:
    """Finite benchmark samples and their protocol-defined uncertainty."""

    samples: tuple[float, ...]
    uncertainty: float

    def __post_init__(self) -> None:
        if not self.samples:
            raise TuningContractError("a measurement curve requires at least one sample")
        for value in self.samples:
            _require_finite_number("measurement sample", value)
        _require_non_negative_finite("measurement uncertainty", self.uncertainty)

    @property
    def mean(self) -> float:
        """Return the arithmetic mean of the fake or measured curve."""
        return fmean(self.samples)


@dataclass(frozen=True, kw_only=True)
class BaselineMeasurement:
    """Frozen baseline arm bound to matched identity and hard gates."""

    identity: MeasurementIdentity
    arm_digest: str
    curve: MeasurementCurve
    gates: tuple[HardGateResult, ...]

    def __post_init__(self) -> None:
        _validate_measurement("baseline", self.identity, self.arm_digest, self.curve, self.gates)


@dataclass(frozen=True, kw_only=True)
class CandidateMeasurement:
    """Frozen candidate arm bound to matched identity and hard gates."""

    identity: MeasurementIdentity
    arm_digest: str
    curve: MeasurementCurve
    gates: tuple[HardGateResult, ...]

    def __post_init__(self) -> None:
        _validate_measurement("candidate", self.identity, self.arm_digest, self.curve, self.gates)


@dataclass(frozen=True, kw_only=True)
class TuningEvaluation:
    """Deterministic terminal evidence for one measured hypothesis."""

    item_id: str
    hypothesis_id: str
    outcome: TuningOutcome
    baseline_mean: float
    candidate_mean: float
    oriented_effect: float
    combined_uncertainty: float
    decisive_effect: float
    failing_gates: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        _require_safe_id("evaluation item_id", self.item_id)
        _require_safe_id("evaluation hypothesis_id", self.hypothesis_id)
        if not isinstance(self.outcome, TuningOutcome):
            raise TuningContractError("outcome must be a TuningOutcome")
        for name, value in (
            ("baseline_mean", self.baseline_mean),
            ("candidate_mean", self.candidate_mean),
            ("oriented_effect", self.oriented_effect),
            ("combined_uncertainty", self.combined_uncertainty),
            ("decisive_effect", self.decisive_effect),
        ):
            _require_finite_number(name, value)
        if self.combined_uncertainty < 0 or self.decisive_effect < 0:
            raise TuningContractError("evaluation thresholds must be non-negative")
        if len(set(self.failing_gates)) != len(self.failing_gates):
            raise TuningContractError("evaluation contains duplicate failing gates")
        gate_outcomes = {
            TuningOutcome.INVALID_BASELINE,
            TuningOutcome.HARD_GATE_FAILURE,
        }
        if self.outcome in gate_outcomes and not self.failing_gates:
            raise TuningContractError("a hard-gate outcome must identify failing gates")
        if self.outcome not in gate_outcomes and self.failing_gates:
            raise TuningContractError("a performance outcome cannot carry failing hard gates")
        _require_text("evaluation reason", self.reason)

    @property
    def recommends_keep(self) -> bool:
        """Whether the measured evidence qualifies for controller promotion."""
        return self.outcome is TuningOutcome.KEEP


@dataclass(frozen=True, kw_only=True)
class PromotionDecision:
    """Controller-owned keep/reject action bound to terminal evidence."""

    item_id: str
    hypothesis_id: str
    outcome: TuningOutcome
    action: PromotionAction
    owner: DecisionOwner
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, TuningOutcome):
            raise TuningContractError("decision outcome must be a TuningOutcome")
        if not isinstance(self.action, PromotionAction):
            raise TuningContractError("decision action must be a PromotionAction")
        if not isinstance(self.owner, DecisionOwner):
            raise TuningContractError("decision owner must be a DecisionOwner")
        if self.owner is not DecisionOwner.CONTROLLER:
            raise TuningContractError("only the controller may own a promotion decision")
        expected = (
            PromotionAction.KEEP if self.outcome is TuningOutcome.KEEP else PromotionAction.REJECT
        )
        if self.action is not expected:
            raise TuningContractError("promotion action disagrees with terminal evidence")
        _require_safe_id("decision item_id", self.item_id)
        _require_safe_id("decision hypothesis_id", self.hypothesis_id)
        _require_text("decision reason", self.reason)


@dataclass(frozen=True, kw_only=True)
class TuningCampaign:
    """Immutable tuning projection carried alongside shared controller state."""

    item_id: str
    hypothesis_id: str
    status: CampaignStatus = CampaignStatus.PLANNED
    evaluation: TuningEvaluation | None = None
    decision: PromotionDecision | None = None

    def __post_init__(self) -> None:
        _require_safe_id("campaign item_id", self.item_id)
        _require_safe_id("campaign hypothesis_id", self.hypothesis_id)
        if self.status is CampaignStatus.PLANNED:
            if self.evaluation is not None or self.decision is not None:
                raise TuningContractError(
                    "a planned campaign cannot contain evidence or a decision"
                )
        elif self.status is CampaignStatus.EVIDENCE_READY:
            if self.evaluation is None or self.decision is not None:
                raise TuningContractError("evidence-ready campaign requires only an evaluation")
        elif self.status is CampaignStatus.TERMINAL:
            if self.evaluation is None or self.decision is None:
                raise TuningContractError("a terminal campaign requires evidence and a decision")
        else:
            raise TuningContractError("unsupported campaign status")
        if self.evaluation is not None:
            _require_campaign_identity(self, self.evaluation.item_id, self.evaluation.hypothesis_id)
        if self.decision is not None:
            _require_campaign_identity(self, self.decision.item_id, self.decision.hypothesis_id)
        if self.evaluation is not None and self.decision is not None:
            if self.decision.outcome is not self.evaluation.outcome:
                raise TuningContractError(
                    "campaign decision outcome disagrees with its immutable evaluation"
                )
            if self.decision.reason != self.evaluation.reason:
                raise TuningContractError(
                    "campaign decision reason disagrees with its immutable evaluation"
                )


def new_tuning_campaign(item: WorkItemProposal, hypothesis: TuningHypothesis) -> TuningCampaign:
    """Validate shared work-item identity and create a planned campaign.

    Args:
        item: Reviewed plan item consumed by the shared dispatcher.
        hypothesis: One-variable tuning hypothesis for the item.

    Returns:
        A new immutable tuning-state projection.
    """
    _validate_tuning_item(item, hypothesis)
    return TuningCampaign(item_id=item.item_id, hypothesis_id=hypothesis.hypothesis_id)


def evaluate_tuning_campaign(
    item: WorkItemProposal,
    hypothesis: TuningHypothesis,
    baseline: BaselineMeasurement,
    candidate: CandidateMeasurement,
) -> TuningEvaluation:
    """Evaluate matched A/B curves without performing benchmark or scheduler work.

    Hard correctness and feature gates are evaluated before performance.  The
    reported uncertainty is treated conservatively as an absolute half-width;
    baseline and candidate half-widths are added for the effect interval.

    Args:
        item: Shared typed item admitted by the reviewed plan.
        hypothesis: One-variable tuning hypothesis.
        baseline: Frozen baseline evidence.
        candidate: Frozen candidate evidence.

    Returns:
        Terminal evidence with a closed keep/negative/no-change vocabulary.
    """
    _validate_tuning_item(item, hypothesis)
    mismatches = _identity_mismatches(baseline.identity, candidate.identity)
    if mismatches:
        raise TuningContractError("baseline and candidate must match: " + ", ".join(mismatches))
    if baseline.arm_digest == candidate.arm_digest:
        raise TuningContractError("baseline and candidate require distinct arm digests")
    _validate_matched_gates(baseline.gates, candidate.gates)

    baseline_mean = baseline.curve.mean
    candidate_mean = candidate.curve.mean
    raw_effect = candidate_mean - baseline_mean
    oriented_effect = (
        raw_effect if hypothesis.direction is MetricDirection.HIGHER_IS_BETTER else -raw_effect
    )
    combined_uncertainty = baseline.curve.uncertainty + candidate.curve.uncertainty
    common = {
        "item_id": item.item_id,
        "hypothesis_id": hypothesis.hypothesis_id,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "oriented_effect": oriented_effect,
        "combined_uncertainty": combined_uncertainty,
        "decisive_effect": hypothesis.uncertainty.decisive_effect,
    }

    baseline_failures = _failed_gates(baseline.gates)
    if baseline_failures:
        return TuningEvaluation(
            **common,
            outcome=TuningOutcome.INVALID_BASELINE,
            failing_gates=baseline_failures,
            reason="baseline hard gates failed, so the A/B comparison is invalid",
        )
    candidate_failures = _failed_gates(candidate.gates)
    if candidate_failures:
        return TuningEvaluation(
            **common,
            outcome=TuningOutcome.HARD_GATE_FAILURE,
            failing_gates=candidate_failures,
            reason="candidate correctness or feature gates failed",
        )
    if combined_uncertainty > hypothesis.uncertainty.maximum_combined_uncertainty:
        return TuningEvaluation(
            **common,
            outcome=TuningOutcome.NOISE,
            failing_gates=(),
            reason="combined uncertainty exceeds the benchmark-specific limit",
        )

    lower_effect = oriented_effect - combined_uncertainty
    upper_effect = oriented_effect + combined_uncertainty
    decisive_effect = hypothesis.uncertainty.decisive_effect
    if lower_effect >= decisive_effect:
        return TuningEvaluation(
            **common,
            outcome=TuningOutcome.KEEP,
            failing_gates=(),
            reason="the conservative improvement bound meets the decisive effect",
        )
    if upper_effect < -hypothesis.uncertainty.noise_threshold:
        return TuningEvaluation(
            **common,
            outcome=TuningOutcome.REGRESSION,
            failing_gates=(),
            reason="the conservative effect interval is a regression beyond noise",
        )
    if abs(oriented_effect) + combined_uncertainty <= decisive_effect:
        return TuningEvaluation(
            **common,
            outcome=TuningOutcome.NO_CHANGE,
            failing_gates=(),
            reason="the bounded effect does not reach the keep threshold",
        )
    return TuningEvaluation(
        **common,
        outcome=TuningOutcome.NOISE,
        failing_gates=(),
        reason="the effect interval crosses a decisive boundary",
    )


def record_tuning_evidence(
    campaign: TuningCampaign, evaluation: TuningEvaluation
) -> TuningCampaign:
    """Advance a planned immutable campaign to evidence-ready state."""
    if campaign.status is not CampaignStatus.PLANNED:
        raise TuningContractError("tuning evidence can only be recorded once from PLANNED")
    _require_campaign_identity(campaign, evaluation.item_id, evaluation.hypothesis_id)
    return replace(
        campaign,
        status=CampaignStatus.EVIDENCE_READY,
        evaluation=evaluation,
    )


def make_promotion_decision(
    evaluation: TuningEvaluation, *, owner: DecisionOwner
) -> PromotionDecision:
    """Map terminal evidence to keep/reject under explicit controller authority."""
    if owner is not DecisionOwner.CONTROLLER:
        raise TuningContractError("Coder, Reviewer, and QA cannot promote tuning candidates")
    action = PromotionAction.KEEP if evaluation.recommends_keep else PromotionAction.REJECT
    return PromotionDecision(
        item_id=evaluation.item_id,
        hypothesis_id=evaluation.hypothesis_id,
        outcome=evaluation.outcome,
        action=action,
        owner=owner,
        reason=evaluation.reason,
    )


def finalize_tuning_campaign(campaign: TuningCampaign, *, owner: DecisionOwner) -> TuningCampaign:
    """Close an evidence-ready campaign with a controller-owned disposition."""
    if campaign.status is not CampaignStatus.EVIDENCE_READY or campaign.evaluation is None:
        raise TuningContractError("only an evidence-ready campaign can be finalized")
    decision = make_promotion_decision(campaign.evaluation, owner=owner)
    return replace(campaign, status=CampaignStatus.TERMINAL, decision=decision)


def _validate_tuning_item(item: WorkItemProposal, hypothesis: TuningHypothesis) -> None:
    if not isinstance(item, WorkItemProposal):
        raise TuningContractError("tuning requires a shared WorkItemProposal")
    if item.kind is not WorkItemKind.TUNE_HYPOTHESIS:
        raise TuningContractError("tuning is only valid for a TUNE_HYPOTHESIS item")
    if item.item_id != hypothesis.item_id:
        raise TuningContractError("hypothesis identity does not match its work item")


def _validate_measurement(
    arm: str,
    identity: MeasurementIdentity,
    arm_digest: str,
    curve: MeasurementCurve,
    gates: tuple[HardGateResult, ...],
) -> None:
    if not isinstance(identity, MeasurementIdentity):
        raise TuningContractError(f"{arm} identity must be a MeasurementIdentity")
    _require_digest(f"{arm} arm_digest", arm_digest)
    if not isinstance(curve, MeasurementCurve):
        raise TuningContractError(f"{arm} curve must be a MeasurementCurve")
    if not gates:
        raise TuningContractError(f"{arm} requires at least one hard gate")
    if any(not isinstance(gate, HardGateResult) for gate in gates):
        raise TuningContractError(f"{arm} gates must be HardGateResult values")
    identities = [gate.identity for gate in gates]
    if len(identities) != len(set(identities)):
        raise TuningContractError(f"{arm} contains duplicate hard gates")
    if not any(gate.kind is HardGateKind.CORRECTNESS for gate in gates):
        raise TuningContractError(f"{arm} requires at least one correctness gate")


def _validate_matched_gates(
    baseline: tuple[HardGateResult, ...], candidate: tuple[HardGateResult, ...]
) -> None:
    baseline_contract = tuple(sorted(gate.identity for gate in baseline))
    candidate_contract = tuple(sorted(gate.identity for gate in candidate))
    if baseline_contract != candidate_contract:
        raise TuningContractError("baseline and candidate hard-gate contracts must match")


def _identity_mismatches(
    baseline: MeasurementIdentity, candidate: MeasurementIdentity
) -> tuple[str, ...]:
    mismatches: list[str] = []
    for name, baseline_value, candidate_value in (
        ("checkpoint", baseline.checkpoint, candidate.checkpoint),
        ("target", baseline.target, candidate.target),
        ("route", baseline.route, candidate.route),
        ("workload", baseline.workload, candidate.workload),
        ("topology", baseline.topology, candidate.topology),
        ("build", baseline.build, candidate.build),
        ("protocol", baseline.protocol, candidate.protocol),
        ("hardware", baseline.hardware, candidate.hardware),
    ):
        if baseline_value != candidate_value:
            mismatches.append(name)
    return tuple(mismatches)


def _failed_gates(gates: tuple[HardGateResult, ...]) -> tuple[str, ...]:
    return tuple(f"{gate.kind.value}:{gate.name}" for gate in gates if not gate.passed)


def _require_campaign_identity(campaign: TuningCampaign, item_id: str, hypothesis_id: str) -> None:
    if campaign.item_id != item_id or campaign.hypothesis_id != hypothesis_id:
        raise TuningContractError("campaign evidence or decision identity does not match")


def _require_safe_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise TuningContractError(f"{name} must be a safe non-empty identifier")


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TuningContractError(f"{name} must be non-empty")


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TuningContractError(f"{name} must be a lowercase SHA-256 digest")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TuningContractError(f"{name} must be a positive integer")


def _require_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TuningContractError(f"{name} must be a finite number")


def _require_non_negative_finite(name: str, value: float) -> None:
    _require_finite_number(name, value)
    if value < 0:
        raise TuningContractError(f"{name} must be non-negative")


__all__ = [
    "BaselineMeasurement",
    "CampaignStatus",
    "CandidateMeasurement",
    "DecisionOwner",
    "HardGateKind",
    "HardGateResult",
    "JsonPrimitive",
    "KnobChange",
    "KnobKind",
    "MeasurementCurve",
    "MeasurementIdentity",
    "MetricDirection",
    "PromotionAction",
    "PromotionDecision",
    "RouteIdentity",
    "TopologyIdentity",
    "TuningCampaign",
    "TuningContractError",
    "TuningEvaluation",
    "TuningHypothesis",
    "TuningOutcome",
    "UncertaintyRule",
    "evaluate_tuning_campaign",
    "finalize_tuning_campaign",
    "make_promotion_decision",
    "new_tuning_campaign",
    "record_tuning_evidence",
]
