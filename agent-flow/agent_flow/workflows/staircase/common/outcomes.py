# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict typed outcomes accepted from Staircase planning roles."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Collection, Sequence, cast

from ..tuning.contracts import (
    KnobChange,
    KnobKind,
    MetricDirection,
    TuningContractError,
    TuningHypothesis,
    UncertaintyRule,
)
from .policy import (
    AssemblerFeatureInput,
    CertifiedClaimCell,
    ExecutionShape,
    PolicyViolation,
    VerdictScope,
    WorkItemKind,
    WorkItemProposal,
    validate_plan_domain_inputs,
    validate_work_item_proposals,
)

OUTCOME_SCHEMA_VERSION = 2
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class OutcomeError(ValueError):
    """Raised when an agent response violates its typed outcome contract."""


class PlanDraftDecision(str, Enum):
    """Closed PlanDrafter outcome vocabulary."""

    DRAFTED = "DRAFTED"
    BLOCKED = "BLOCKED"


class PlanReviewDecision(str, Enum):
    """Closed PlanReviewer outcome vocabulary."""

    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    BLOCK = "BLOCK"


class CoderDecision(str, Enum):
    """Closed Coder outcome vocabulary."""

    CANDIDATE = "CANDIDATE"
    RESOURCE_ESCALATION = "RESOURCE_ESCALATION"
    BLOCKED = "BLOCKED"


class GateDecision(str, Enum):
    """Closed deterministic-gate outcome vocabulary."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


class ReviewerDecision(str, Enum):
    """Closed Reviewer outcome vocabulary."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    BLOCK = "BLOCK"


class QaDecision(str, Enum):
    """Closed QA outcome vocabulary."""

    PASS = "PASS"
    FAIL = "FAIL"
    BLOCK = "BLOCK"


@dataclass(frozen=True)
class StageProposal:
    """One independently QA-able proposed Stage."""

    stage_id: str
    goal_ids: tuple[str, ...]
    exit_gates: tuple[str, ...]


@dataclass(frozen=True)
class GoalProposal:
    """One module/capability Goal and its required atomic items."""

    goal_id: str
    stage_id: str
    capability: str
    item_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlanDraftOutcome:
    """Strict reviewed-plan input emitted by PlanDrafter."""

    stages: tuple[StageProposal, ...]
    goals: tuple[GoalProposal, ...]
    items: tuple[WorkItemProposal, ...]
    digest: str

    @property
    def outcome(self) -> PlanDraftDecision:
        """Return the normative outcome represented by this payload."""
        return PlanDraftDecision.DRAFTED


@dataclass(frozen=True)
class PlanDraftBlockedOutcome:
    """PlanDrafter result when required planning input is unavailable."""

    reason: str

    @property
    def outcome(self) -> PlanDraftDecision:
        """Return the normative outcome represented by this payload."""
        return PlanDraftDecision.BLOCKED


@dataclass(frozen=True)
class PlanReviewOutcome:
    """PlanReviewer verdict pinned to an immutable plan digest."""

    outcome: PlanReviewDecision
    plan_digest: str
    corrections: tuple[str, ...]


def parse_plan_draft(
    response: str,
    *,
    allowed_resource_classes: Collection[str],
    allowed_path_roots: Sequence[str],
    workflow_mode: str,
    target_features: Collection[str],
) -> PlanDraftOutcome | PlanDraftBlockedOutcome:
    """Parse and validate a pure-JSON PlanDrafter response."""
    root = _json_object(response)
    _schema_version(root)
    try:
        outcome = PlanDraftDecision(_string(root.get("outcome"), "outcome"))
    except ValueError as exc:
        raise OutcomeError("PlanDrafter outcome must be DRAFTED or BLOCKED") from exc
    if outcome is PlanDraftDecision.BLOCKED:
        _exact_keys(root, {"schema_version", "outcome", "reason"}, "blocked plan")
        return PlanDraftBlockedOutcome(reason=_string(root["reason"], "reason"))
    _exact_keys(root, {"schema_version", "outcome", "stages", "goals", "items"}, "plan")
    stages = tuple(_parse_stage(value) for value in _list(root["stages"], "stages"))
    goals = tuple(_parse_goal(value) for value in _list(root["goals"], "goals"))
    items = tuple(_parse_item(value) for value in _list(root["items"], "items"))
    if not stages or not goals or not items:
        raise OutcomeError("a plan must contain at least one Stage, Goal, and WorkItem")
    try:
        validate_work_item_proposals(
            items,
            allowed_resource_classes=allowed_resource_classes,
            allowed_path_roots=allowed_path_roots,
        )
        validate_plan_domain_inputs(
            items,
            workflow_mode=workflow_mode,
            target_features=target_features,
        )
    except PolicyViolation as exc:
        raise OutcomeError(str(exc)) from exc
    _validate_hierarchy(stages, goals, items)
    canonical = json.dumps(root, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return PlanDraftOutcome(
        stages=stages,
        goals=goals,
        items=items,
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def parse_plan_review(response: str, *, expected_plan_digest: str) -> PlanReviewOutcome:
    """Parse a pure-JSON PlanReviewer verdict for one frozen plan."""
    root = _json_object(response)
    _exact_keys(
        root,
        {"schema_version", "outcome", "plan_digest", "corrections"},
        "plan review",
    )
    _schema_version(root)
    plan_digest = _string(root["plan_digest"], "plan_digest")
    if plan_digest != expected_plan_digest:
        raise OutcomeError("PlanReviewer verdict does not match the frozen plan digest")
    try:
        outcome = PlanReviewDecision(_string(root["outcome"], "outcome"))
    except ValueError as exc:
        raise OutcomeError("PlanReviewer outcome must be ACCEPT, REVISE, or BLOCK") from exc
    corrections = _strings(root["corrections"], "corrections")
    if outcome is PlanReviewDecision.ACCEPT and corrections:
        raise OutcomeError("an accepted plan cannot carry corrections")
    if outcome in {PlanReviewDecision.REVISE, PlanReviewDecision.BLOCK} and not corrections:
        raise OutcomeError(f"{outcome.value} requires at least one correction or blocker")
    return PlanReviewOutcome(outcome, plan_digest, corrections)


def planning_outcome_instruction(role: str, *, plan_digest: str | None = None) -> str:
    """Return the controller-owned JSON outcome contract for a planning role."""
    if role == "plan_drafter":
        return (
            "Return exactly one JSON object with schema_version=2. Use outcome='DRAFTED' with "
            "keys stages, goals, and items, or outcome='BLOCKED' with the single additional key "
            "reason. Stage keys: stage_id, goal_ids, exit_gates. Goal keys: goal_id, "
            "stage_id, capability, item_ids. Item keys: item_id, goal_id, kind, resource_class, "
            "execution, modifies_files, domain_input, dependencies, entry_ids, allowed_paths, "
            "certified_claim_cells. Execution keys: nodes, ranks_per_node, gpus_per_node, "
            "array_element, verdict_scope. Claim keys: entry_id, cell_id. domain_input must be "
            "null except: assemble_feature requires exactly {'feature': <one exact task target "
            "feature>}; tune_hypothesis requires exactly hypothesis_id, item_id matching the "
            "WorkItem, statement, metric, direction, changes (exactly one typed change with name, "
            "kind, baseline_value, candidate_value), and uncertainty (minimum_effect, "
            "noise_threshold, maximum_combined_uncertainty). No Markdown or extra keys."
        )
    if role == "plan_reviewer" and plan_digest is not None:
        return (
            "Return exactly one JSON object with keys schema_version=2, outcome, "
            f"plan_digest='{plan_digest}', and corrections. Outcome is ACCEPT, REVISE, or BLOCK. "
            "ACCEPT attests that every special WorkItem domain_input in the plan pinned by "
            "plan_digest was checked against mode, target-feature, item-identity, one-change, "
            "and uncertainty boundaries. ACCEPT requires no corrections; REVISE and BLOCK "
            "require at least one exact correction or external blocker. No Markdown."
        )
    raise ValueError("unsupported planning role or missing plan digest")


_ROLE_DECISIONS: dict[str, type[Enum]] = {
    "plan_drafter": PlanDraftDecision,
    "plan_reviewer": PlanReviewDecision,
    "coder": CoderDecision,
    "gate": GateDecision,
    "reviewer": ReviewerDecision,
    "qa": QaDecision,
}


def parse_role_decision(role: str, value: object) -> Enum:
    """Parse one exact normative role outcome without legacy aliases."""
    decision_type = _ROLE_DECISIONS.get(role)
    if decision_type is None:
        raise ValueError(f"unsupported outcome role {role!r}")
    text = _string(value, "outcome")
    try:
        return decision_type(text)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in decision_type)
        raise OutcomeError(f"{role} outcome must be one of: {allowed}") from exc


def role_outcome_instruction(role: str) -> str:
    """Return the exact schema/version vocabulary for one execution role."""
    decision_type = _ROLE_DECISIONS.get(role)
    if decision_type is None:
        raise ValueError(f"unsupported outcome role {role!r}")
    allowed = ", ".join(member.value for member in decision_type)
    return (
        f"Return schema_version={OUTCOME_SCHEMA_VERSION} and exactly one {role} outcome from: "
        f"{allowed}. Lowercase or generic success/failure aliases are invalid."
    )


def _parse_stage(value: object) -> StageProposal:
    data = _mapping(value, "stage")
    _exact_keys(data, {"stage_id", "goal_ids", "exit_gates"}, "stage")
    stage = StageProposal(
        stage_id=_identifier(data["stage_id"], "stage_id"),
        goal_ids=_identifiers(data["goal_ids"], "goal_ids"),
        exit_gates=_strings(data["exit_gates"], "exit_gates"),
    )
    if not stage.goal_ids or not stage.exit_gates:
        raise OutcomeError("each Stage requires Goals and exit gates")
    return stage


def _parse_goal(value: object) -> GoalProposal:
    data = _mapping(value, "goal")
    _exact_keys(data, {"goal_id", "stage_id", "capability", "item_ids"}, "goal")
    goal = GoalProposal(
        goal_id=_identifier(data["goal_id"], "goal_id"),
        stage_id=_identifier(data["stage_id"], "stage_id"),
        capability=_string(data["capability"], "capability"),
        item_ids=_identifiers(data["item_ids"], "item_ids"),
    )
    if not goal.item_ids:
        raise OutcomeError("each Goal requires at least one WorkItem")
    return goal


def _parse_item(value: object) -> WorkItemProposal:
    data = _mapping(value, "work item")
    _exact_keys(
        data,
        {
            "item_id",
            "goal_id",
            "kind",
            "resource_class",
            "execution",
            "modifies_files",
            "domain_input",
            "dependencies",
            "entry_ids",
            "allowed_paths",
            "certified_claim_cells",
        },
        "work item",
    )
    execution_data = _mapping(data["execution"], "execution")
    _exact_keys(
        execution_data,
        {"nodes", "ranks_per_node", "gpus_per_node", "array_element", "verdict_scope"},
        "execution",
    )
    try:
        kind = WorkItemKind(_string(data["kind"], "kind"))
        item_id = _identifier(data["item_id"], "item_id")
        verdict_scope = VerdictScope(_string(execution_data["verdict_scope"], "verdict_scope"))
        claims = tuple(
            _parse_claim(entry)
            for entry in _list(data["certified_claim_cells"], "certified_claim_cells")
        )
        return WorkItemProposal(
            item_id=item_id,
            goal_id=_identifier(data["goal_id"], "goal_id"),
            kind=kind,
            resource_class=_identifier(data["resource_class"], "resource_class"),
            execution=ExecutionShape(
                nodes=_integer(execution_data["nodes"], "nodes", minimum=1),
                ranks_per_node=_integer(
                    execution_data["ranks_per_node"], "ranks_per_node", minimum=1
                ),
                gpus_per_node=_integer(execution_data["gpus_per_node"], "gpus_per_node", minimum=0),
                array_element=_boolean(execution_data["array_element"], "array_element"),
                verdict_scope=verdict_scope,
            ),
            modifies_files=_boolean(data["modifies_files"], "modifies_files"),
            domain_input=_parse_domain_input(data["domain_input"], kind, item_id),
            dependencies=_identifiers(data["dependencies"], "dependencies"),
            entry_ids=_identifiers(data["entry_ids"], "entry_ids"),
            allowed_paths=_strings(data["allowed_paths"], "allowed_paths"),
            certified_claim_cells=claims,
        )
    except (PolicyViolation, TuningContractError, ValueError) as exc:
        raise OutcomeError(str(exc)) from exc


def _parse_domain_input(
    value: object, kind: WorkItemKind, item_id: str
) -> AssemblerFeatureInput | TuningHypothesis | None:
    if kind is WorkItemKind.ASSEMBLE_FEATURE:
        data = _mapping(value, "assemble_feature domain_input")
        _exact_keys(data, {"feature"}, "assemble_feature domain_input")
        return AssemblerFeatureInput(feature=_string(data["feature"], "feature"))
    if kind is WorkItemKind.TUNE_HYPOTHESIS:
        data = _mapping(value, "tune_hypothesis domain_input")
        _exact_keys(
            data,
            {
                "hypothesis_id",
                "item_id",
                "statement",
                "metric",
                "direction",
                "changes",
                "uncertainty",
            },
            "tune_hypothesis domain_input",
        )
        hypothesis_item_id = _identifier(data["item_id"], "hypothesis item_id")
        if hypothesis_item_id != item_id:
            raise OutcomeError("tuning hypothesis item_id must match its WorkItem identity")
        changes = tuple(_parse_knob_change(change) for change in _list(data["changes"], "changes"))
        uncertainty_data = _mapping(data["uncertainty"], "uncertainty")
        _exact_keys(
            uncertainty_data,
            {"minimum_effect", "noise_threshold", "maximum_combined_uncertainty"},
            "uncertainty",
        )
        return TuningHypothesis(
            hypothesis_id=_identifier(data["hypothesis_id"], "hypothesis_id"),
            item_id=hypothesis_item_id,
            statement=_string(data["statement"], "statement"),
            metric=_string(data["metric"], "metric"),
            direction=MetricDirection(_string(data["direction"], "direction")),
            changes=changes,
            uncertainty=UncertaintyRule(
                minimum_effect=_number(uncertainty_data["minimum_effect"], "minimum_effect"),
                noise_threshold=_number(uncertainty_data["noise_threshold"], "noise_threshold"),
                maximum_combined_uncertainty=_number(
                    uncertainty_data["maximum_combined_uncertainty"],
                    "maximum_combined_uncertainty",
                ),
            ),
        )
    if value is not None:
        raise OutcomeError(f"{kind.value} work requires a null domain_input")
    return None


def _parse_knob_change(value: object) -> KnobChange:
    data = _mapping(value, "tuning change")
    _exact_keys(
        data,
        {"name", "kind", "baseline_value", "candidate_value"},
        "tuning change",
    )
    return KnobChange(
        name=_string(data["name"], "knob name"),
        kind=KnobKind(_string(data["kind"], "knob kind")),
        baseline_value=_json_primitive(data["baseline_value"], "baseline_value"),
        candidate_value=_json_primitive(data["candidate_value"], "candidate_value"),
    )


def _parse_claim(value: object) -> CertifiedClaimCell:
    data = _mapping(value, "certified claim cell")
    _exact_keys(data, {"entry_id", "cell_id"}, "certified claim cell")
    try:
        return CertifiedClaimCell(
            entry_id=_identifier(data["entry_id"], "entry_id"),
            cell_id=_string(data["cell_id"], "cell_id"),
        )
    except PolicyViolation as exc:
        raise OutcomeError(str(exc)) from exc


def _validate_hierarchy(
    stages: Sequence[StageProposal],
    goals: Sequence[GoalProposal],
    items: Sequence[WorkItemProposal],
) -> None:
    stage_ids = [stage.stage_id for stage in stages]
    goal_ids = [goal.goal_id for goal in goals]
    item_ids = [item.item_id for item in items]
    if len(set(stage_ids)) != len(stage_ids):
        raise OutcomeError("duplicate Stage IDs are not allowed")
    if len(set(goal_ids)) != len(goal_ids):
        raise OutcomeError("duplicate Goal IDs are not allowed")
    goals_by_id = {goal.goal_id: goal for goal in goals}
    items_by_id = {item.item_id: item for item in items}
    claimed_goals: set[str] = set()
    for stage in stages:
        for goal_id in stage.goal_ids:
            if goal_id not in goals_by_id or goals_by_id[goal_id].stage_id != stage.stage_id:
                raise OutcomeError(f"Stage {stage.stage_id!r} has an invalid Goal reference")
            if goal_id in claimed_goals:
                raise OutcomeError(f"Goal {goal_id!r} belongs to more than one Stage")
            claimed_goals.add(goal_id)
    if claimed_goals != set(goal_ids):
        raise OutcomeError("every Goal must belong to exactly one Stage")
    claimed_items: set[str] = set()
    for goal in goals:
        for item_id in goal.item_ids:
            if item_id not in items_by_id or items_by_id[item_id].goal_id != goal.goal_id:
                raise OutcomeError(f"Goal {goal.goal_id!r} has an invalid WorkItem reference")
            if item_id in claimed_items:
                raise OutcomeError(f"WorkItem {item_id!r} belongs to more than one Goal")
            claimed_items.add(item_id)
    if claimed_items != set(item_ids):
        raise OutcomeError("every WorkItem must belong to exactly one Goal")


def _json_object(response: str) -> dict[str, object]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise OutcomeError(
            "agent outcome must be exactly one JSON object without Markdown"
        ) from exc
    return _mapping(value, "outcome")


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise OutcomeError(f"{name} must be an object with string keys")
    return cast(dict[str, object], value)


def _exact_keys(data: dict[str, object], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing or unknown:
        raise OutcomeError(f"{name} keys invalid; missing={missing}, unknown={unknown}")


def _schema_version(data: dict[str, object]) -> None:
    if data.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        raise OutcomeError(f"schema_version must be {OUTCOME_SCHEMA_VERSION}")


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise OutcomeError(f"{name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OutcomeError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier(value: object, name: str) -> str:
    result = _string(value, name)
    if not _SAFE_ID.fullmatch(result):
        raise OutcomeError(f"{name} must be a safe identifier")
    return result


def _strings(value: object, name: str) -> tuple[str, ...]:
    result = tuple(_string(entry, name) for entry in _list(value, name))
    if len(set(result)) != len(result):
        raise OutcomeError(f"{name} cannot contain duplicates")
    return result


def _identifiers(value: object, name: str) -> tuple[str, ...]:
    return tuple(_identifier(entry, name) for entry in _list(value, name))


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise OutcomeError(f"{name} must be an integer >= {minimum}")
    return value


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise OutcomeError(f"{name} must be a boolean")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise OutcomeError(f"{name} must be a finite number")
    return float(value)


def _json_primitive(value: object, name: str) -> str | int | float | bool | None:
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise OutcomeError(f"{name} must be a JSON primitive")
    if isinstance(value, float) and not math.isfinite(value):
        raise OutcomeError(f"{name} cannot contain NaN or infinity")
    return cast(str | int | float | bool | None, value)
