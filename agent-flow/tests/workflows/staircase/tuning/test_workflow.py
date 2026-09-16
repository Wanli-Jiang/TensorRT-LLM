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

"""Tests for pure typed Staircase tuning campaign decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.policy import (
    ExecutionShape,
    WorkItemKind,
    WorkItemProposal,
)
from agent_flow.workflows.staircase.tuning import artifacts
from agent_flow.workflows.staircase.tuning.workflow import (
    BaselineMeasurement,
    CampaignStatus,
    CandidateMeasurement,
    DecisionOwner,
    HardGateKind,
    HardGateResult,
    KnobChange,
    KnobKind,
    MeasurementCurve,
    MeasurementIdentity,
    MetricDirection,
    PromotionAction,
    PromotionDecision,
    RouteIdentity,
    TopologyIdentity,
    TuningContractError,
    TuningHypothesis,
    TuningOutcome,
    UncertaintyRule,
    evaluate_tuning_campaign,
    finalize_tuning_campaign,
    make_promotion_decision,
    new_tuning_campaign,
    record_tuning_evidence,
)

_BASELINE_DIGEST = "a" * 64
_CANDIDATE_DIGEST = "b" * 64
_CORRECTNESS_DIGEST = "c" * 64
_FEATURE_DIGEST = "d" * 64


def _item(*, kind: WorkItemKind = WorkItemKind.TUNE_HYPOTHESIS) -> WorkItemProposal:
    return WorkItemProposal(
        item_id="tune-attention-backend",
        goal_id="attention-performance",
        kind=kind,
        resource_class="tuner-gpu",
        execution=ExecutionShape(gpus_per_node=1),
        modifies_files=False,
        domain_input=_hypothesis() if kind is WorkItemKind.TUNE_HYPOTHESIS else None,
    )


def _hypothesis(
    *,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    changes: tuple[KnobChange, ...] | None = None,
    maximum_uncertainty: float = 3.0,
) -> TuningHypothesis:
    return TuningHypothesis(
        hypothesis_id="hypothesis-1",
        item_id="tune-attention-backend",
        statement="The candidate attention backend improves matched throughput.",
        metric="tokens_per_second",
        direction=direction,
        changes=changes
        if changes is not None
        else (
            KnobChange(
                name="attention.backend",
                kind=KnobKind.CONFIGURATION,
                baseline_value="baseline",
                candidate_value="candidate",
            ),
        ),
        uncertainty=UncertaintyRule(
            minimum_effect=5.0,
            noise_threshold=1.0,
            maximum_combined_uncertainty=maximum_uncertainty,
        ),
    )


def _identity() -> MeasurementIdentity:
    return MeasurementIdentity(
        checkpoint="Qwen3-30B-A3B@revision",
        target="qwen3_moe/sm100/tp8",
        route=RouteIdentity(
            family="qwen3_moe",
            architecture="Qwen3MoeForCausalLM",
            expected_route="tensorrt_llm._torch.modeling_v2.models.qwen3_moe",
            synthetic_target=False,
        ),
        workload="isl8192-osl1024-concurrency64",
        topology=TopologyIdentity(
            world_size=8,
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            moe_expert_parallel_size=8,
            moe_tensor_parallel_size=1,
            attention_data_parallel_size=1,
        ),
        build="trtllm-local-dev-2026-09-16",
        protocol="paired-abba-five-repetitions",
        hardware="8xGB300-exclusive",
    )


def _gates(*, correctness: bool = True, feature: bool = True) -> tuple[HardGateResult, ...]:
    return (
        HardGateResult(
            name="gsm8k-no-regression",
            kind=HardGateKind.CORRECTNESS,
            passed=correctness,
            evidence_digest=_CORRECTNESS_DIGEST,
        ),
        HardGateResult(
            name="modeling-v2-route-require",
            kind=HardGateKind.FEATURE,
            passed=feature,
            evidence_digest=_FEATURE_DIGEST,
        ),
    )


def _measurements(
    *,
    baseline_samples: tuple[float, ...] = (99.0, 100.0, 101.0),
    candidate_samples: tuple[float, ...] = (109.0, 110.0, 111.0),
    baseline_uncertainty: float = 0.5,
    candidate_uncertainty: float = 0.5,
    baseline_gates: tuple[HardGateResult, ...] | None = None,
    candidate_gates: tuple[HardGateResult, ...] | None = None,
    candidate_identity: MeasurementIdentity | None = None,
) -> tuple[BaselineMeasurement, CandidateMeasurement]:
    identity = _identity()
    return (
        BaselineMeasurement(
            identity=identity,
            arm_digest=_BASELINE_DIGEST,
            curve=MeasurementCurve(
                samples=baseline_samples,
                uncertainty=baseline_uncertainty,
            ),
            gates=baseline_gates if baseline_gates is not None else _gates(),
        ),
        CandidateMeasurement(
            identity=candidate_identity or identity,
            arm_digest=_CANDIDATE_DIGEST,
            curve=MeasurementCurve(
                samples=candidate_samples,
                uncertainty=candidate_uncertainty,
            ),
            gates=candidate_gates if candidate_gates is not None else _gates(),
        ),
    )


def test_tuning_contracts_are_frozen_and_change_exactly_one_variable() -> None:
    hypothesis = _hypothesis()
    with pytest.raises(FrozenInstanceError):
        hypothesis.metric = "latency"  # type: ignore[misc]

    with pytest.raises(TuningContractError, match="exactly one variable"):
        _hypothesis(changes=())
    second = KnobChange(
        name="attention.block_size",
        kind=KnobKind.CONFIGURATION,
        baseline_value=64,
        candidate_value=128,
    )
    with pytest.raises(TuningContractError, match="exactly one variable"):
        _hypothesis(changes=(_hypothesis().change, second))


@pytest.mark.parametrize(
    "knob_name",
    [
        "target.expected_route",
        "model.architecture",
        "parallel.topology",
        "parallel.tensor_parallel_size",
        "reference.checkpoint",
    ],
)
def test_route_and_execution_identity_cannot_be_tuning_knobs(knob_name: str) -> None:
    with pytest.raises(TuningContractError, match="cannot change route"):
        KnobChange(
            name=knob_name,
            kind=KnobKind.CONFIGURATION,
            baseline_value="a",
            candidate_value="b",
        )


def test_tuning_requires_the_shared_typed_tune_hypothesis_item() -> None:
    with pytest.raises(TuningContractError, match="TUNE_HYPOTHESIS"):
        new_tuning_campaign(_item(kind=WorkItemKind.SEARCH), _hypothesis())

    campaign = new_tuning_campaign(_item(), _hypothesis())
    assert campaign.status is CampaignStatus.PLANNED


def test_keep_curve_requires_conservative_improvement() -> None:
    baseline, candidate = _measurements()
    evaluation = evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)
    assert evaluation.outcome is TuningOutcome.KEEP
    assert evaluation.oriented_effect == pytest.approx(10.0)
    assert evaluation.combined_uncertainty == pytest.approx(1.0)
    assert evaluation.recommends_keep


def test_lower_is_better_metric_uses_the_same_keep_rule() -> None:
    baseline, candidate = _measurements(
        baseline_samples=(100.0, 100.0),
        candidate_samples=(90.0, 90.0),
    )
    evaluation = evaluate_tuning_campaign(
        _item(),
        _hypothesis(direction=MetricDirection.LOWER_IS_BETTER),
        baseline,
        candidate,
    )
    assert evaluation.outcome is TuningOutcome.KEEP
    assert evaluation.oriented_effect == pytest.approx(10.0)


@pytest.mark.parametrize("failed_kind", [HardGateKind.CORRECTNESS, HardGateKind.FEATURE])
def test_candidate_hard_gate_failure_overrides_a_large_speedup(
    failed_kind: HardGateKind,
) -> None:
    candidate_gates = _gates(
        correctness=failed_kind is not HardGateKind.CORRECTNESS,
        feature=failed_kind is not HardGateKind.FEATURE,
    )
    baseline, candidate = _measurements(
        candidate_samples=(1000.0, 1000.0),
        candidate_gates=candidate_gates,
    )
    evaluation = evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)
    assert evaluation.outcome is TuningOutcome.HARD_GATE_FAILURE
    assert not evaluation.recommends_keep
    assert any(failed_kind.value in gate for gate in evaluation.failing_gates)


def test_invalid_baseline_is_terminal_negative_evidence() -> None:
    baseline, candidate = _measurements(baseline_gates=_gates(correctness=False))
    evaluation = evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)
    assert evaluation.outcome is TuningOutcome.INVALID_BASELINE
    campaign = record_tuning_evidence(new_tuning_campaign(_item(), _hypothesis()), evaluation)
    terminal = finalize_tuning_campaign(campaign, owner=DecisionOwner.CONTROLLER)
    assert terminal.status is CampaignStatus.TERMINAL
    assert terminal.decision is not None
    assert terminal.decision.action is PromotionAction.REJECT


def test_negative_no_change_and_noise_curves_are_terminal_evidence() -> None:
    cases = (
        ((94.0, 94.0), 0.2, TuningOutcome.REGRESSION),
        ((103.0, 103.0), 0.2, TuningOutcome.NO_CHANGE),
        ((106.0, 106.0), 1.0, TuningOutcome.NOISE),
    )
    for samples, uncertainty, expected in cases:
        baseline, candidate = _measurements(
            baseline_samples=(100.0, 100.0),
            candidate_samples=samples,
            baseline_uncertainty=uncertainty,
            candidate_uncertainty=uncertainty,
        )
        evaluation = evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)
        assert evaluation.outcome is expected
        evidence_ready = record_tuning_evidence(
            new_tuning_campaign(_item(), _hypothesis()), evaluation
        )
        terminal = finalize_tuning_campaign(evidence_ready, owner=DecisionOwner.CONTROLLER)
        assert terminal.status is CampaignStatus.TERMINAL
        assert terminal.decision is not None
        assert terminal.decision.action is PromotionAction.REJECT


def test_excessive_protocol_uncertainty_is_noise_even_for_positive_mean() -> None:
    baseline, candidate = _measurements(
        candidate_samples=(120.0, 120.0),
        baseline_uncertainty=2.0,
        candidate_uncertainty=2.0,
    )
    evaluation = evaluate_tuning_campaign(
        _item(), _hypothesis(maximum_uncertainty=3.0), baseline, candidate
    )
    assert evaluation.outcome is TuningOutcome.NOISE


def test_every_matched_identity_dimension_is_enforced() -> None:
    identity = _identity()
    mismatches = (
        ("checkpoint", replace(identity, checkpoint="different-checkpoint")),
        ("target", replace(identity, target="different-target")),
        (
            "route",
            replace(identity, route=replace(identity.route, expected_route="different.route")),
        ),
        ("workload", replace(identity, workload="different-workload")),
        (
            "topology",
            replace(
                identity,
                topology=TopologyIdentity(
                    world_size=4,
                    tensor_parallel_size=4,
                    pipeline_parallel_size=1,
                    moe_expert_parallel_size=4,
                    moe_tensor_parallel_size=1,
                    attention_data_parallel_size=1,
                ),
            ),
        ),
        ("build", replace(identity, build="different-build")),
        ("protocol", replace(identity, protocol="different-protocol")),
        ("hardware", replace(identity, hardware="different-hardware")),
    )
    for dimension, candidate_identity in mismatches:
        baseline, candidate = _measurements(candidate_identity=candidate_identity)
        with pytest.raises(TuningContractError, match=dimension):
            evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)


def test_matched_hard_gate_contract_is_required() -> None:
    baseline, candidate = _measurements(
        candidate_gates=(
            HardGateResult(
                name="different-correctness-gate",
                kind=HardGateKind.CORRECTNESS,
                passed=True,
                evidence_digest=_CORRECTNESS_DIGEST,
            ),
        )
    )
    with pytest.raises(TuningContractError, match="hard-gate contracts must match"):
        evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)


@pytest.mark.parametrize("owner", [DecisionOwner.CODER, DecisionOwner.REVIEWER, DecisionOwner.QA])
def test_only_controller_can_make_keep_or_reject_decision(owner: DecisionOwner) -> None:
    baseline, candidate = _measurements()
    evaluation = evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)
    with pytest.raises(TuningContractError, match="cannot promote"):
        make_promotion_decision(evaluation, owner=owner)

    decision = make_promotion_decision(evaluation, owner=DecisionOwner.CONTROLLER)
    assert decision.action is PromotionAction.KEEP
    assert decision.owner is DecisionOwner.CONTROLLER


def test_campaign_state_is_single_write_and_identity_bound() -> None:
    baseline, candidate = _measurements()
    evaluation = evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)
    campaign = new_tuning_campaign(_item(), _hypothesis())
    evidence_ready = record_tuning_evidence(campaign, evaluation)

    with pytest.raises(TuningContractError, match="only be recorded once"):
        record_tuning_evidence(evidence_ready, evaluation)
    terminal = finalize_tuning_campaign(evidence_ready, owner=DecisionOwner.CONTROLLER)
    assert terminal.status is CampaignStatus.TERMINAL
    with pytest.raises(TuningContractError, match="evidence-ready"):
        finalize_tuning_campaign(terminal, owner=DecisionOwner.CONTROLLER)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_curves_and_knobs_are_rejected(bad_value: float) -> None:
    with pytest.raises(TuningContractError, match="finite"):
        MeasurementCurve(samples=(bad_value,), uncertainty=0.0)
    with pytest.raises(TuningContractError, match="NaN or infinity"):
        KnobChange(
            name="attention.scale",
            kind=KnobKind.CONFIGURATION,
            baseline_value=1.0,
            candidate_value=bad_value,
        )


def test_tuning_artifacts_are_canonical_write_once_and_reloadable(tmp_path: Path) -> None:
    baseline, candidate = _measurements()
    persisted = artifacts.persist_terminal_tuning_campaign(
        tmp_path,
        item=_item(),
        hypothesis=_hypothesis(),
        baseline=baseline,
        candidate=candidate,
        owner=DecisionOwner.CONTROLLER,
    )

    assert persisted.campaign.status is CampaignStatus.TERMINAL
    assert persisted.decision.action is PromotionAction.KEEP
    assert len(persisted.receipts) == 5
    loaded_baseline, baseline_digest = artifacts.load_baseline_artifact(
        tmp_path / artifacts.BASELINE_FILENAME
    )
    loaded_candidate, candidate_digest = artifacts.load_candidate_artifact(
        tmp_path / artifacts.CANDIDATE_FILENAME
    )
    loaded_evaluation, evaluation_digest = artifacts.load_evaluation_artifact(
        tmp_path / artifacts.EVALUATION_FILENAME
    )
    loaded_campaign, campaign_digest = artifacts.load_campaign_artifact(
        tmp_path / artifacts.CAMPAIGN_FILENAME
    )
    loaded_decision, decision_digest = artifacts.load_promotion_decision_artifact(
        tmp_path / artifacts.PROMOTION_DECISION_FILENAME
    )
    assert loaded_baseline == baseline
    assert loaded_candidate == candidate
    assert loaded_campaign.status is CampaignStatus.EVIDENCE_READY
    assert loaded_campaign.evaluation == loaded_evaluation
    assert loaded_decision == persisted.decision
    assert (
        baseline_digest,
        candidate_digest,
        evaluation_digest,
        campaign_digest,
        decision_digest,
    ) == tuple(receipt.digest for receipt in persisted.receipts)
    assert (tmp_path / artifacts.PROMOTION_DECISION_FILENAME).read_bytes() == (
        artifacts.canonical_tuning_artifact_bytes(loaded_decision)
    )

    with pytest.raises(artifacts.ImmutableTuningArtifactError, match="already exists"):
        artifacts.write_baseline_artifact(tmp_path / artifacts.BASELINE_FILENAME, baseline)


def test_restart_replay_yields_byte_identical_controller_decision(tmp_path: Path) -> None:
    baseline, candidate = _measurements()
    persisted = artifacts.persist_terminal_tuning_campaign(
        tmp_path,
        item=_item(),
        hypothesis=_hypothesis(),
        baseline=baseline,
        candidate=candidate,
        owner=DecisionOwner.CONTROLLER,
    )
    before = (tmp_path / artifacts.PROMOTION_DECISION_FILENAME).read_bytes()

    replayed = artifacts.replay_terminal_tuning_campaign(
        tmp_path,
        item=_item(),
        hypothesis=_hypothesis(),
        owner=DecisionOwner.CONTROLLER,
    )

    assert replayed == persisted.campaign
    assert replayed.decision is not None
    assert artifacts.canonical_tuning_artifact_bytes(replayed.decision) == before
    assert (tmp_path / artifacts.PROMOTION_DECISION_FILENAME).read_bytes() == before


def test_hard_gate_failure_is_durably_rejected_despite_speedup(tmp_path: Path) -> None:
    baseline, candidate = _measurements(
        candidate_samples=(10_000.0, 10_000.0),
        candidate_gates=_gates(correctness=False),
    )
    persisted = artifacts.persist_terminal_tuning_campaign(
        tmp_path,
        item=_item(),
        hypothesis=_hypothesis(),
        baseline=baseline,
        candidate=candidate,
        owner=DecisionOwner.CONTROLLER,
    )

    assert persisted.decision.outcome is TuningOutcome.HARD_GATE_FAILURE
    assert persisted.decision.action is PromotionAction.REJECT
    replayed = artifacts.replay_terminal_tuning_campaign(
        tmp_path,
        item=_item(),
        hypothesis=_hypothesis(),
        owner=DecisionOwner.CONTROLLER,
    )
    assert replayed.decision is not None
    assert replayed.decision.action is PromotionAction.REJECT

    forged_keep = PromotionDecision(
        item_id=persisted.decision.item_id,
        hypothesis_id=persisted.decision.hypothesis_id,
        outcome=TuningOutcome.KEEP,
        action=PromotionAction.KEEP,
        owner=DecisionOwner.CONTROLLER,
        reason="performance overrides the hard gate",
    )
    evaluation, _digest = artifacts.load_evaluation_artifact(
        tmp_path / artifacts.EVALUATION_FILENAME
    )
    with pytest.raises(TuningContractError, match="immutable evaluation"):
        artifacts.write_promotion_decision_artifact(
            tmp_path / "forged-decision.json",
            forged_keep,
            evaluation=evaluation,
            owner=DecisionOwner.CONTROLLER,
        )


@pytest.mark.parametrize("owner", [DecisionOwner.CODER, DecisionOwner.REVIEWER, DecisionOwner.QA])
def test_artifact_promotion_owner_must_be_explicit_controller(
    tmp_path: Path, owner: DecisionOwner
) -> None:
    baseline, candidate = _measurements()
    with pytest.raises(TuningContractError, match="cannot promote"):
        artifacts.persist_terminal_tuning_campaign(
            tmp_path,
            item=_item(),
            hypothesis=_hypothesis(),
            baseline=baseline,
            candidate=candidate,
            owner=owner,
        )
    assert not tmp_path.exists() or not tuple(tmp_path.iterdir())


def test_loader_rejects_noncanonical_and_unknown_json(tmp_path: Path) -> None:
    baseline, _candidate = _measurements()
    path = tmp_path / artifacts.BASELINE_FILENAME
    artifacts.write_baseline_artifact(path, baseline)
    raw = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(artifacts.TuningArtifactError, match="non-canonical JSON"):
        artifacts.load_baseline_artifact(path)

    raw["payload"]["unknown"] = True
    raw["digest"] = hashlib.sha256(
        json.dumps(
            raw["payload"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(
        json.dumps(
            raw,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(artifacts.TuningArtifactError, match="unknown"):
        artifacts.load_baseline_artifact(path)


def test_loader_rejects_canonical_payload_digest_mismatch(tmp_path: Path) -> None:
    baseline, _candidate = _measurements()
    path = tmp_path / artifacts.BASELINE_FILENAME
    artifacts.write_baseline_artifact(path, baseline)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload"]["arm_digest"] = "e" * 64
    path.write_text(
        json.dumps(
            raw,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(artifacts.TuningArtifactDigestError, match="digest mismatch"):
        artifacts.load_baseline_artifact(path)


def test_replay_rejects_self_consistent_but_different_evaluation(tmp_path: Path) -> None:
    baseline, candidate = _measurements()
    artifacts.persist_terminal_tuning_campaign(
        tmp_path,
        item=_item(),
        hypothesis=_hypothesis(),
        baseline=baseline,
        candidate=candidate,
        owner=DecisionOwner.CONTROLLER,
    )
    evaluation_path = tmp_path / artifacts.EVALUATION_FILENAME
    raw = json.loads(evaluation_path.read_text(encoding="utf-8"))
    raw["payload"]["reason"] = "tampered but internally valid reason"
    raw["digest"] = hashlib.sha256(
        json.dumps(
            raw["payload"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    evaluation_path.write_text(
        json.dumps(
            raw,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(artifacts.TuningReplayError, match="evaluation differs"):
        artifacts.replay_terminal_tuning_campaign(
            tmp_path,
            item=_item(),
            hypothesis=_hypothesis(),
            owner=DecisionOwner.CONTROLLER,
        )


def test_terminal_campaign_rejects_decision_from_different_evidence() -> None:
    baseline, candidate = _measurements()
    evaluation = evaluate_tuning_campaign(_item(), _hypothesis(), baseline, candidate)
    campaign = record_tuning_evidence(new_tuning_campaign(_item(), _hypothesis()), evaluation)
    mismatched = replace(
        make_promotion_decision(evaluation, owner=DecisionOwner.CONTROLLER),
        outcome=TuningOutcome.NO_CHANGE,
        action=PromotionAction.REJECT,
        reason="different terminal evidence",
    )
    with pytest.raises(TuningContractError, match="immutable evaluation"):
        replace(
            campaign,
            status=CampaignStatus.TERMINAL,
            decision=mismatched,
        )
