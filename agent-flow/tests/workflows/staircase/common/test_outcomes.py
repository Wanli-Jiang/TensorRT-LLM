# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed planning-role outcomes."""

from __future__ import annotations

import json

import pytest

from agent_flow.workflows.staircase.common.outcomes import (
    CoderDecision,
    GateDecision,
    OutcomeError,
    PlanDraftBlockedOutcome,
    PlanDraftDecision,
    PlanReviewDecision,
    QaDecision,
    ReviewerDecision,
    parse_plan_draft,
    parse_plan_review,
    parse_role_decision,
    planning_outcome_instruction,
    role_outcome_instruction,
)


def _plan() -> dict[str, object]:
    return {
        "schema_version": 2,
        "outcome": "DRAFTED",
        "stages": [{"stage_id": "stage-1", "goal_ids": ["goal-1"], "exit_gates": ["gate"]}],
        "goals": [
            {
                "goal_id": "goal-1",
                "stage_id": "stage-1",
                "capability": "attention module",
                "item_ids": ["entry-1"],
            }
        ],
        "items": [
            {
                "item_id": "entry-1",
                "goal_id": "goal-1",
                "kind": "catalog_verify",
                "resource_class": "deterministic_gate",
                "execution": {
                    "nodes": 1,
                    "ranks_per_node": 1,
                    "gpus_per_node": 1,
                    "array_element": False,
                    "verdict_scope": "item",
                },
                "modifies_files": False,
                "domain_input": None,
                "dependencies": [],
                "entry_ids": ["attention-entry"],
                "allowed_paths": ["tensorrt_llm/_torch/modeling_v2/catalog/attention/entry.py"],
                "certified_claim_cells": [{"entry_id": "attention-entry", "cell_id": "sm100:bf16"}],
            }
        ],
    }


def _parse(payload: dict[str, object]):
    return parse_plan_draft(
        json.dumps(payload),
        allowed_resource_classes={"deterministic_gate"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="onboard",
        target_features=("mtp",),
    )


def test_plan_draft_is_typed_and_digest_stable() -> None:
    first = _parse(_plan())
    second = _parse(dict(reversed(list(_plan().items()))))
    assert first == second
    assert first.outcome is PlanDraftDecision.DRAFTED
    assert len(first.digest) == 64


def test_plan_draft_can_return_a_typed_blocker() -> None:
    blocked = _parse(
        {
            "schema_version": 2,
            "outcome": "BLOCKED",
            "reason": "checkpoint config is unavailable",
        }
    )
    assert isinstance(blocked, PlanDraftBlockedOutcome)
    assert blocked.outcome is PlanDraftDecision.BLOCKED


def test_plan_draft_rejects_markdown_wrapping() -> None:
    with pytest.raises(OutcomeError, match="exactly one JSON"):
        parse_plan_draft(
            f"```json\n{json.dumps(_plan())}\n```",
            allowed_resource_classes={"deterministic_gate"},
            allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
            workflow_mode="onboard",
            target_features=("mtp",),
        )


def test_plan_draft_rejects_unknown_keys() -> None:
    payload = _plan()
    payload["narrative"] = "not control state"
    with pytest.raises(OutcomeError, match="unknown"):
        _parse(payload)


def test_plan_draft_rejects_goal_as_unowned_item() -> None:
    payload = _plan()
    cast_goals = payload["goals"]
    assert isinstance(cast_goals, list)
    cast_goals[0]["item_ids"] = []
    with pytest.raises(OutcomeError, match="requires at least one WorkItem"):
        _parse(payload)


def test_plan_draft_rejects_unknown_resource_class() -> None:
    payload = _plan()
    items = payload["items"]
    assert isinstance(items, list)
    items[0]["resource_class"] = "unbounded"
    with pytest.raises(OutcomeError, match="unknown resource class"):
        _parse(payload)


def test_plan_draft_binds_assembler_feature_to_frozen_target() -> None:
    payload = _plan()
    items = payload["items"]
    assert isinstance(items, list)
    items[0].update(
        {
            "kind": "assemble_feature",
            "entry_ids": [],
            "certified_claim_cells": [],
            "domain_input": {"feature": "mtp"},
        }
    )
    outcome = _parse(payload)
    assert outcome.items[0].domain_input is not None

    items[0]["domain_input"] = {"feature": "lora"}
    changed = parse_plan_draft(
        json.dumps(payload),
        allowed_resource_classes={"deterministic_gate"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="onboard",
        target_features=("mtp", "lora"),
    )
    assert changed.digest != outcome.digest

    items[0]["domain_input"] = {"feature": "unknown"}
    with pytest.raises(OutcomeError, match="unknown target feature"):
        _parse(payload)


def test_plan_draft_requires_null_domain_input_for_other_kinds() -> None:
    payload = _plan()
    items = payload["items"]
    assert isinstance(items, list)
    items[0]["domain_input"] = {"feature": "mtp"}
    with pytest.raises(OutcomeError, match="null domain_input"):
        _parse(payload)


def test_plan_draft_parses_complete_tuning_hypothesis_in_tune_mode() -> None:
    payload = _plan()
    items = payload["items"]
    assert isinstance(items, list)
    items[0].update(
        {
            "kind": "tune_hypothesis",
            "entry_ids": [],
            "certified_claim_cells": [],
            "domain_input": {
                "hypothesis_id": "hypothesis-1",
                "item_id": "entry-1",
                "statement": "larger tile improves throughput",
                "metric": "tokens_per_second",
                "direction": "higher_is_better",
                "changes": [
                    {
                        "name": "kernel.tile_size",
                        "kind": "configuration",
                        "baseline_value": 64,
                        "candidate_value": 128,
                    }
                ],
                "uncertainty": {
                    "minimum_effect": 1.0,
                    "noise_threshold": 0.5,
                    "maximum_combined_uncertainty": 0.25,
                },
            },
        }
    )
    outcome = parse_plan_draft(
        json.dumps(payload),
        allowed_resource_classes={"deterministic_gate"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="tune",
        target_features=("mtp",),
    )
    hypothesis = outcome.items[0].domain_input
    assert hypothesis is not None
    assert hypothesis.item_id == "entry-1"

    items[0]["domain_input"]["item_id"] = "another-item"
    with pytest.raises(OutcomeError, match="must match its WorkItem"):
        parse_plan_draft(
            json.dumps(payload),
            allowed_resource_classes={"deterministic_gate"},
            allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
            workflow_mode="tune",
            target_features=("mtp",),
        )


def test_plan_review_is_pinned_to_frozen_digest() -> None:
    outcome = parse_plan_review(
        json.dumps(
            {
                "schema_version": 2,
                "outcome": "ACCEPT",
                "plan_digest": "a" * 64,
                "corrections": [],
            }
        ),
        expected_plan_digest="a" * 64,
    )
    assert outcome.outcome is PlanReviewDecision.ACCEPT


def test_plan_review_rejects_digest_drift() -> None:
    with pytest.raises(OutcomeError, match="frozen plan digest"):
        parse_plan_review(
            json.dumps(
                {
                    "schema_version": 2,
                    "outcome": "ACCEPT",
                    "plan_digest": "b" * 64,
                    "corrections": [],
                }
            ),
            expected_plan_digest="a" * 64,
        )


def test_revised_or_blocked_review_requires_details() -> None:
    with pytest.raises(OutcomeError, match="at least one correction"):
        parse_plan_review(
            json.dumps(
                {
                    "schema_version": 2,
                    "outcome": "REVISE",
                    "plan_digest": "a" * 64,
                    "corrections": [],
                }
            ),
            expected_plan_digest="a" * 64,
        )

    blocked = parse_plan_review(
        json.dumps(
            {
                "schema_version": 2,
                "outcome": "BLOCK",
                "plan_digest": "a" * 64,
                "corrections": ["checkpoint metadata requires owner input"],
            }
        ),
        expected_plan_digest="a" * 64,
    )
    assert blocked.outcome is PlanReviewDecision.BLOCK


@pytest.mark.parametrize(
    ("role", "value", "expected"),
    [
        ("plan_drafter", "DRAFTED", PlanDraftDecision.DRAFTED),
        ("plan_reviewer", "ACCEPT", PlanReviewDecision.ACCEPT),
        ("coder", "CANDIDATE", CoderDecision.CANDIDATE),
        ("coder", "RESOURCE_ESCALATION", CoderDecision.RESOURCE_ESCALATION),
        ("gate", "PASS", GateDecision.PASS),
        ("reviewer", "APPROVE", ReviewerDecision.APPROVE),
        ("qa", "FAIL", QaDecision.FAIL),
    ],
)
def test_role_outcomes_are_exact(role: str, value: str, expected: object) -> None:
    assert parse_role_decision(role, value) is expected
    assert value in role_outcome_instruction(role)


@pytest.mark.parametrize("legacy", ["approve", "reject", "success", "failure"])
def test_role_outcomes_do_not_alias_legacy_values(legacy: str) -> None:
    with pytest.raises(OutcomeError, match="outcome must be one of"):
        parse_role_decision("reviewer", legacy)


def test_old_planning_wire_schema_is_rejected() -> None:
    legacy = _plan()
    legacy["schema_version"] = 1
    legacy["outcome"] = "plan"
    with pytest.raises(OutcomeError, match="schema_version must be 2"):
        _parse(legacy)


def test_plan_reviewer_instruction_attests_domain_input_review() -> None:
    instruction = planning_outcome_instruction("plan_reviewer", plan_digest="a" * 64)
    assert "every special WorkItem domain_input" in instruction
    assert "plan_digest" in instruction
