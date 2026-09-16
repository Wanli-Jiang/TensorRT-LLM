# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure action planning for Assembler, Tuner, gate, Reviewer, and QA work.

The functions in this module only validate controller-owned facts and return
immutable actions.  They do not create candidate overlays, write mailboxes, mutate Git,
or contact Slurm.  A caller must persist the returned state before it
materializes or dispatches the corresponding action.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from ..common.artifacts import JsonValue, WorkerInputManifest
from ..common.gates import EvidenceScope, GatePhase, GateSpec
from ..common.policy import ASSEMBLER_ITEM_KINDS, CATALOG_ITEM_KINDS, WorkItemProposal
from ..common.policy import WorkItemKind as PolicyWorkItemKind
from ..common.runners import BackendKind
from ..onboarding.workflow import (
    ROUTER_INDEX_PATH,
    AssemblerInput,
    AssemblyPreparation,
    CoreAssemblerInput,
    FeatureAssemblerInput,
    RoutingAssemblerInput,
    WeightsAssemblerInput,
    validate_assembler_output,
)
from ..state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    Role,
    RunState,
    WorkItemRecord,
    WorkItemStatus,
)
from ..task_schema import CertificationMode, ResourceClass
from ..tuning.workflow import TuningHypothesis, new_tuning_campaign

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DomainActionKind(str, Enum):
    """Closed vocabulary of non-Smith worker actions."""

    ASSEMBLER_CODER = "assembler_coder"
    TUNER_CODER = "tuner_coder"
    DETERMINISTIC_GATE = "deterministic_gate"
    REVIEWER_ANALYSIS = "reviewer_analysis"
    REVIEWER_RERUN = "reviewer_rerun"
    QA = "qa"


class DomainBlockCode(str, Enum):
    """Machine-readable reason why a safe action could not be formed."""

    INVALID_STATE = "invalid_state"
    PROFILE_OR_KIND = "profile_or_kind"
    DEPENDENCY_NOT_INTEGRATED = "dependency_not_integrated"
    CATALOG_GAP = "catalog_gap"
    ALLOWED_PATHS = "allowed_paths"
    TUNING_CONTRACT = "tuning_contract"
    CANDIDATE_MISMATCH = "candidate_mismatch"
    GATE_CONTRACT = "gate_contract"
    RESOURCE_CONTRACT = "resource_contract"
    RANK_LAUNCHER_REQUIRED = "rank_launcher_required"


@dataclass(frozen=True, slots=True)
class AgentSettings:
    """Controller-selected backend and model for one fresh agent process."""

    backend_kind: BackendKind
    model: str

    def __post_init__(self) -> None:
        if self.backend_kind not in {"claude-code", "codex"}:
            raise ValueError(f"unsupported agent backend {self.backend_kind!r}")
        if not self.model.strip():
            raise ValueError("agent model must be non-empty")


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    """Controller-owned candidate identity consumed by read-only attempts."""

    item_id: str
    coder_attempt_id: str
    commit: str
    digest: str
    changed_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("item_id", self.item_id),
            ("coder_attempt_id", self.coder_attempt_id),
        ):
            if _SAFE_ID.fullmatch(value) is None:
                raise ValueError(f"{name} must be a safe non-empty identifier")
        if _COMMIT.fullmatch(self.commit) is None:
            raise ValueError("candidate commit must be a lowercase 40-character Git hash")
        if _DIGEST.fullmatch(self.digest) is None:
            raise ValueError("candidate digest must be a lowercase SHA-256 digest")
        _validate_paths(self.changed_paths, "candidate changed paths")


@dataclass(frozen=True, slots=True)
class GateExecutionContract:
    """Controller-knowable shape expected from one deterministic gate job.

    Generic gate workers support only CPU-static or one-rank product commands.
    Other valid scopes remain representable so the planner can return the
    typed ``RANK_LAUNCHER_REQUIRED`` block.  Concrete nodes and GPU identities
    are intentionally absent because they do not exist until Slurm allocates
    the worker.
    """

    scope: EvidenceScope
    expected_world_size: int
    expected_rank: int | None
    expected_local_rank: int | None
    product_rank_body: bool

    def __post_init__(self) -> None:
        if not isinstance(self.scope, EvidenceScope):
            raise TypeError("gate execution scope must use EvidenceScope")
        if (
            isinstance(self.expected_world_size, bool)
            or not isinstance(self.expected_world_size, int)
            or self.expected_world_size < 0
        ):
            raise ValueError("expected_world_size must be a non-negative integer")
        for name, value in (
            ("expected_rank", self.expected_rank),
            ("expected_local_rank", self.expected_local_rank),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be null or a non-negative integer")
        if not isinstance(self.product_rank_body, bool):
            raise TypeError("product_rank_body must be a boolean")
        if self.scope is EvidenceScope.CPU_STATIC:
            expected = (0, None, None, False)
        elif self.scope is EvidenceScope.SINGLE_GPU_PRODUCT:
            expected = (1, 0, 0, True)
        elif self.scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
            expected = (4, None, None, True)
        elif self.scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER:
            expected = (self.expected_world_size, None, None, False)
            if self.expected_world_size < 2:
                raise ValueError("synthetic multi-node execution requires at least two ranks")
        else:
            expected = (self.expected_world_size, None, None, True)
            if self.expected_world_size < 2:
                raise ValueError("real multi-node execution requires at least two ranks")
        observed = (
            self.expected_world_size,
            self.expected_rank,
            self.expected_local_rank,
            self.product_rank_body,
        )
        if observed != expected:
            raise ValueError(
                f"{self.scope.value} requires execution shape {expected!r}, got {observed!r}"
            )


@dataclass(frozen=True, slots=True)
class DomainActionBlock:
    """Typed fail-closed result from action planning."""

    code: DomainBlockCode
    item_id: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, DomainBlockCode):
            raise TypeError("block code must use DomainBlockCode")
        if _SAFE_ID.fullmatch(self.item_id) is None:
            raise ValueError("blocked item_id must be a safe non-empty identifier")
        if not self.reason.strip():
            raise ValueError("blocked action reason must be non-empty")


@dataclass(frozen=True, slots=True)
class DomainAttemptAction:
    """One persisted attempt that is ready for later materialization."""

    kind: DomainActionKind
    proposal: WorkItemProposal
    attempt: AttemptRecord
    resource: ResourceClass
    worktree: Path
    branch: str
    base_commit: str
    attempt_dir: Path
    read_only: bool
    manifest: WorkerInputManifest

    def __post_init__(self) -> None:
        if self.proposal.item_id != self.attempt.item_id:
            raise ValueError("domain action proposal and attempt identities differ")
        if self.manifest.item_id != self.attempt.item_id:
            raise ValueError("domain action manifest belongs to another item")
        if self.manifest.attempt_id != self.attempt.attempt_id:
            raise ValueError("domain action manifest belongs to another attempt")
        if self.manifest.role is not self.attempt.role:
            raise ValueError("domain action manifest role differs from its attempt")
        if self.manifest.profile is not self.attempt.profile:
            raise ValueError("domain action manifest profile differs from its attempt")
        if self.manifest.worktree != str(self.worktree):
            raise ValueError("domain action manifest does not name its worktree")
        if self.read_only != (self.attempt.role is not Role.CODER):
            raise ValueError("only Coder domain actions may receive a writable worktree")
        if set(self.manifest.payload) != {"runtime", "context"}:
            raise ValueError("domain action must emit the generic {runtime, context} envelope")


@dataclass(frozen=True, slots=True)
class DomainActionPlan:
    """Either one immutable dispatch action or one typed BLOCKED result."""

    state: RunState
    action: DomainAttemptAction | None = None
    blocked: DomainActionBlock | None = None

    def __post_init__(self) -> None:
        if (self.action is None) == (self.blocked is None):
            raise ValueError("domain plan must contain exactly one action or BLOCKED result")

    @property
    def ready(self) -> bool:
        """Whether the action may be materialized after the state is persisted."""
        return self.action is not None


def plan_assembler_coder_action(
    state: RunState,
    proposal: WorkItemProposal,
    preparation: AssemblyPreparation,
    *,
    resource: ResourceClass,
    workspace: Path,
    settings: AgentSettings,
    evidence_paths: Sequence[str],
) -> DomainActionPlan:
    """Plan one Assembler Coder without performing filesystem or scheduler work."""
    try:
        item = _validate_ready_item(state, proposal, DomainProfile.ASSEMBLER)
        if proposal.kind not in ASSEMBLER_ITEM_KINDS:
            raise _Blocked(DomainBlockCode.PROFILE_OR_KIND, "not an Assembler work-item kind")
        _validate_coder_resource(proposal, resource)
        if preparation.catalog_gaps:
            entries = sorted(gap.entry_id for gap in preparation.catalog_gaps)
            raise _Blocked(
                DomainBlockCode.CATALOG_GAP,
                f"Assembler requires Smith catalog gaps to integrate first: {entries!r}",
            )
        if preparation.assembler_input is None:
            raise _Blocked(DomainBlockCode.INVALID_STATE, "Assembler preparation has no input")
        assembler_input = preparation.assembler_input
        _validate_assembler_paths(assembler_input, proposal.allowed_paths)
        integration_head = _coder_integration_head(state, item)
        attempt = _new_attempt(
            state,
            item,
            Role.CODER,
            AttemptKind.ROLE,
            resource_class=resource.name,
        )
        desired = state.replace_item(item.add_attempt(attempt).transition(WorkItemStatus.CODING))
        context: dict[str, JsonValue] = {
            "action": DomainActionKind.ASSEMBLER_CODER.value,
            "proposal": _proposal_context(proposal),
            "assembler": _assembler_context(assembler_input),
        }
        action = _agent_action(
            desired,
            proposal,
            attempt,
            resource,
            workspace=workspace,
            base_commit=integration_head,
            settings=settings,
            evidence_paths=evidence_paths,
            candidate=None,
            prompt=(
                "Implement exactly the controller-authorized Assembler unit using only the "
                "integrated dependencies in context. Stay within allowed_paths and publish the "
                "authorized evidence."
            ),
            context=context,
            kind=DomainActionKind.ASSEMBLER_CODER,
        )
        return DomainActionPlan(desired, action=action)
    except _Blocked as error:
        return _blocked(state, proposal.item_id, error)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked_error(state, proposal.item_id, DomainBlockCode.INVALID_STATE, error)


def plan_tuner_coder_action(
    state: RunState,
    proposal: WorkItemProposal,
    hypothesis: TuningHypothesis,
    *,
    resource: ResourceClass,
    workspace: Path,
    settings: AgentSettings,
    evidence_paths: Sequence[str],
) -> DomainActionPlan:
    """Plan one-variable Tuner Coder work with a matched typed hypothesis."""
    try:
        item = _validate_ready_item(state, proposal, DomainProfile.TUNER)
        if proposal.kind is not PolicyWorkItemKind.TUNE_HYPOTHESIS:
            raise _Blocked(DomainBlockCode.PROFILE_OR_KIND, "not a tuning work-item kind")
        _validate_coder_resource(proposal, resource)
        integration_head = _coder_integration_head(state, item)
        campaign = new_tuning_campaign(proposal, hypothesis)
        attempt = _new_attempt(
            state,
            item,
            Role.CODER,
            AttemptKind.ROLE,
            resource_class=resource.name,
        )
        desired = state.replace_item(item.add_attempt(attempt).transition(WorkItemStatus.CODING))
        context: dict[str, JsonValue] = {
            "action": DomainActionKind.TUNER_CODER.value,
            "proposal": _proposal_context(proposal),
            "campaign": {
                "item_id": campaign.item_id,
                "hypothesis_id": campaign.hypothesis_id,
                "status": campaign.status.value,
            },
            "hypothesis": _hypothesis_context(hypothesis),
        }
        action = _agent_action(
            desired,
            proposal,
            attempt,
            resource,
            workspace=workspace,
            base_commit=integration_head,
            settings=settings,
            evidence_paths=evidence_paths,
            candidate=None,
            prompt=(
                "Execute exactly the one-variable tuning hypothesis in context. Keep every "
                "baseline/candidate identity matched, run hard gates first, and publish measured "
                "evidence without promoting the result."
            ),
            context=context,
            kind=DomainActionKind.TUNER_CODER,
        )
        return DomainActionPlan(desired, action=action)
    except _Blocked as error:
        return _blocked(state, proposal.item_id, error)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked_error(state, proposal.item_id, DomainBlockCode.TUNING_CONTRACT, error)


def plan_deterministic_gate_action(
    state: RunState,
    proposal: WorkItemProposal,
    candidate: FrozenCandidate,
    gate: GateSpec,
    execution: GateExecutionContract,
    *,
    resource: ResourceClass,
    workspace: Path,
) -> DomainActionPlan:
    """Plan one shell-free command in one gate attempt for a frozen candidate."""
    try:
        item = _validate_item_identity(state, proposal)
        _validate_candidate_profile(item, proposal)
        if item.status is not WorkItemStatus.CODING:
            raise _Blocked(
                DomainBlockCode.INVALID_STATE,
                "deterministic gate requires a WorkItem in CODING",
            )
        _validate_gate_execution(gate, execution, resource)
        _validate_candidate(
            item,
            proposal,
            candidate,
            require_frozen=item.candidate_attempt_id is not None,
        )
        gate_context = _single_gate_context(gate)
        attempt = _new_attempt(
            state,
            item,
            Role.GATE,
            AttemptKind.DETERMINISTIC_GATE,
            resource_class=resource.name,
        )
        desired_item = item
        if item.candidate_attempt_id is None:
            desired_item = replace(
                item,
                candidate_attempt_id=candidate.coder_attempt_id,
                candidate_digest=candidate.digest,
            )
        desired_item = desired_item.add_attempt(attempt)
        desired = state.replace_item(desired_item)
        action = _gate_action(
            desired,
            proposal,
            attempt,
            resource,
            workspace=workspace,
            base_commit=candidate.commit,
            candidate=candidate,
            gate=gate,
            execution=execution,
            context={
                "action": DomainActionKind.DETERMINISTIC_GATE.value,
                "proposal": _proposal_context(proposal),
                "candidate": _candidate_context(candidate),
                "gate": gate_context,
                "execution": _gate_execution_context(execution),
            },
        )
        return DomainActionPlan(desired, action=action)
    except _Blocked as error:
        return _blocked(state, proposal.item_id, error)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked_error(state, proposal.item_id, DomainBlockCode.GATE_CONTRACT, error)


def plan_reviewer_analysis_action(
    state: RunState,
    proposal: WorkItemProposal,
    candidate: FrozenCandidate,
    gates: Sequence[GateSpec],
    *,
    gate_attempt_ids: Sequence[str],
    resource: ResourceClass,
    workspace: Path,
    settings: AgentSettings,
    evidence_paths: Sequence[str],
    candidate_evidence: JsonValue | None = None,
) -> DomainActionPlan:
    """Plan Reviewer analysis after every ordered gate receipt was validated."""
    try:
        item = _validate_item_identity(state, proposal)
        _validate_candidate_profile(item, proposal)
        if item.status is not WorkItemStatus.CANDIDATE_READY:
            raise _Blocked(
                DomainBlockCode.INVALID_STATE,
                "Reviewer analysis requires a CANDIDATE_READY WorkItem",
            )
        _validate_candidate(item, proposal, candidate, require_frozen=True)
        validated_gate_ids = _validated_gate_attempt_ids(
            item,
            candidate,
            gates,
            gate_attempt_ids,
        )
        attempt = _new_attempt(
            state,
            item,
            Role.REVIEWER,
            AttemptKind.REVIEWER_ANALYSIS,
            resource_class=resource.name,
            review_of_attempt_id=candidate.coder_attempt_id,
            reviewed_candidate_digest=candidate.digest,
        )
        desired = state.replace_item(item.transition(WorkItemStatus.REVIEWING).add_attempt(attempt))
        action = _review_action(
            desired,
            proposal,
            attempt,
            resource,
            workspace=workspace,
            candidate=candidate,
            settings=settings,
            evidence_paths=evidence_paths,
            gates=gates,
            gate_attempt_ids=validated_gate_ids,
            kind=DomainActionKind.REVIEWER_ANALYSIS,
            candidate_evidence=candidate_evidence,
        )
        return DomainActionPlan(desired, action=action)
    except _Blocked as error:
        return _blocked(state, proposal.item_id, error)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked_error(state, proposal.item_id, DomainBlockCode.CANDIDATE_MISMATCH, error)


def plan_reviewer_rerun_action(
    state: RunState,
    proposal: WorkItemProposal,
    candidate: FrozenCandidate,
    gates: Sequence[GateSpec],
    *,
    analysis_attempt_id: str,
    gate_attempt_ids: Sequence[str],
    resource: ResourceClass,
    workspace: Path,
    settings: AgentSettings,
    evidence_paths: Sequence[str],
    candidate_evidence: JsonValue | None = None,
) -> DomainActionPlan:
    """Plan a distinct Reviewer rerun after fresh analysis succeeded."""
    try:
        item = _validate_item_identity(state, proposal)
        _validate_candidate_profile(item, proposal)
        if item.status is not WorkItemStatus.REVIEWING:
            raise _Blocked(
                DomainBlockCode.INVALID_STATE,
                "Reviewer rerun requires a WorkItem in REVIEWING",
            )
        _validate_candidate(item, proposal, candidate, require_frozen=True)
        validated_gate_ids = _validated_gate_attempt_ids(
            item,
            candidate,
            gates,
            gate_attempt_ids,
        )
        analysis = _attempt(item, analysis_attempt_id)
        if (
            analysis.kind is not AttemptKind.REVIEWER_ANALYSIS
            or analysis.role is not Role.REVIEWER
            or analysis.status is not AttemptStatus.VALIDATED
            or analysis.review_of_attempt_id != candidate.coder_attempt_id
            or analysis.reviewed_candidate_digest != candidate.digest
            or analysis.candidate_digest != candidate.digest
        ):
            raise _Blocked(
                DomainBlockCode.CANDIDATE_MISMATCH,
                "Reviewer rerun requires validated analysis of the same frozen candidate",
            )
        attempt = _new_attempt(
            state,
            item,
            Role.REVIEWER,
            AttemptKind.REVIEWER_RERUN,
            resource_class=resource.name,
            review_of_attempt_id=candidate.coder_attempt_id,
            reviewed_candidate_digest=candidate.digest,
        )
        desired = state.replace_item(item.add_attempt(attempt))
        action = _review_action(
            desired,
            proposal,
            attempt,
            resource,
            workspace=workspace,
            candidate=candidate,
            settings=settings,
            evidence_paths=evidence_paths,
            gates=gates,
            gate_attempt_ids=validated_gate_ids,
            kind=DomainActionKind.REVIEWER_RERUN,
            candidate_evidence=candidate_evidence,
        )
        return DomainActionPlan(desired, action=action)
    except _Blocked as error:
        return _blocked(state, proposal.item_id, error)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked_error(state, proposal.item_id, DomainBlockCode.CANDIDATE_MISMATCH, error)


def plan_qa_action(
    state: RunState,
    proposal: WorkItemProposal,
    candidate: FrozenCandidate,
    gates: Sequence[GateSpec],
    *,
    gate_attempt_ids: Sequence[str],
    gate_result_digests: Mapping[str, str],
    resource: ResourceClass,
    workspace: Path,
    settings: AgentSettings,
    evidence_paths: Sequence[str],
    candidate_evidence: JsonValue | None = None,
) -> DomainActionPlan:
    """Plan an independent QA audit after Reviewer approval and before integration."""
    try:
        item = _validate_item_identity(state, proposal)
        _validate_candidate_profile(item, proposal)
        if item.status is not WorkItemStatus.APPROVED or item.reviewer_attempt_id is None:
            raise _Blocked(
                DomainBlockCode.INVALID_STATE,
                "QA requires a Reviewer-approved frozen candidate",
            )
        _validate_candidate(item, proposal, candidate, require_frozen=True)
        validated_gate_ids = _validated_gate_attempt_ids(
            item,
            candidate,
            gates,
            gate_attempt_ids,
        )
        if set(gate_result_digests) != set(validated_gate_ids):
            raise _Blocked(
                DomainBlockCode.GATE_CONTRACT,
                "QA gate result references differ from validated gate attempts",
            )
        for gate_attempt_id in validated_gate_ids:
            gate_attempt = _attempt(item, gate_attempt_id)
            if gate_attempt.result_digest != gate_result_digests[gate_attempt_id]:
                raise _Blocked(
                    DomainBlockCode.GATE_CONTRACT,
                    "QA gate result digest differs from controller-ingested state",
                )
        reviewer = _attempt(item, item.reviewer_attempt_id)
        if reviewer.status is not AttemptStatus.VALIDATED:
            raise _Blocked(
                DomainBlockCode.CANDIDATE_MISMATCH,
                "QA requires validated Reviewer evidence",
            )
        _gate_context(gates)
        attempt = _new_attempt(
            state,
            item,
            Role.QA,
            AttemptKind.QA,
            resource_class=resource.name,
        )
        desired = state.replace_item(item.add_attempt(attempt))
        context: dict[str, JsonValue] = {
            "action": DomainActionKind.QA.value,
            "proposal": _proposal_context(proposal),
            "candidate": _candidate_context(candidate),
            "stage_id": item.stage_id,
            "reviewer_attempt_id": reviewer.attempt_id,
            "gate_results": [
                {
                    "gate_attempt_id": gate_attempt_id,
                    "result_digest": gate_result_digests[gate_attempt_id],
                }
                for gate_attempt_id in validated_gate_ids
            ],
            "gates": _gate_context(gates),
        }
        if candidate_evidence is not None:
            context["candidate_evidence"] = candidate_evidence
        prompt = (
            "Independently audit the frozen target candidate and the complete ordered gate "
            "matrix in context. Hard-gate failures must reject; performance cannot override "
            "correctness."
        )
        if candidate_evidence is not None:
            prompt += (
                " Independently validate the exact immutable tuning measurement envelopes "
                "and reject any identity, hard-gate, or uncertainty mismatch."
            )
        action = _agent_action(
            desired,
            proposal,
            attempt,
            resource,
            workspace=workspace,
            base_commit=candidate.commit,
            settings=settings,
            evidence_paths=evidence_paths,
            candidate=candidate,
            prompt=prompt,
            context=context,
            kind=DomainActionKind.QA,
        )
        return DomainActionPlan(desired, action=action)
    except _Blocked as error:
        return _blocked(state, proposal.item_id, error)
    except (KeyError, TypeError, ValueError) as error:
        return _blocked_error(state, proposal.item_id, DomainBlockCode.GATE_CONTRACT, error)


class _Blocked(ValueError):
    def __init__(self, code: DomainBlockCode, reason: str) -> None:
        super().__init__(reason)
        self.code = code


def _blocked(state: RunState, item_id: str, error: _Blocked) -> DomainActionPlan:
    return DomainActionPlan(
        state,
        blocked=DomainActionBlock(error.code, item_id, str(error)),
    )


def _blocked_error(
    state: RunState,
    item_id: str,
    code: DomainBlockCode,
    error: Exception,
) -> DomainActionPlan:
    return DomainActionPlan(
        state,
        blocked=DomainActionBlock(code, item_id, f"action validation failed: {error}"),
    )


def _validate_ready_item(
    state: RunState,
    proposal: WorkItemProposal,
    profile: DomainProfile,
) -> WorkItemRecord:
    item = _validate_item_identity(state, proposal)
    if item.profile is not profile:
        raise _Blocked(
            DomainBlockCode.PROFILE_OR_KIND,
            f"work item requires {profile.value!r} profile",
        )
    not_integrated = sorted(
        dependency
        for dependency in item.dependencies
        if state.item(dependency).status is not WorkItemStatus.INTEGRATED
    )
    if not_integrated:
        raise _Blocked(
            DomainBlockCode.DEPENDENCY_NOT_INTEGRATED,
            f"dependencies are not INTEGRATED: {not_integrated!r}",
        )
    if item.status is not WorkItemStatus.READY:
        raise _Blocked(DomainBlockCode.INVALID_STATE, "Coder action requires a READY WorkItem")
    return item


def _coder_integration_head(state: RunState, item: WorkItemRecord) -> str:
    integrated_ids = {record.item_id for record in state.integration_history}
    missing = sorted(set(item.dependencies) - integrated_ids)
    if missing:
        raise _Blocked(
            DomainBlockCode.DEPENDENCY_NOT_INTEGRATED,
            f"integrated dependencies lack persisted fan-in records: {missing!r}",
        )
    return state.integration_head


def _validate_item_identity(
    state: RunState,
    proposal: WorkItemProposal,
) -> WorkItemRecord:
    item = state.item(proposal.item_id)
    if (
        item.goal_id != proposal.goal_id
        or item.kind.value != proposal.kind.value
        or item.dependencies != proposal.dependencies
    ):
        raise _Blocked(
            DomainBlockCode.PROFILE_OR_KIND,
            "authoritative WorkItem identity differs from its admitted proposal",
        )
    return item


def _validate_coder_resource(proposal: WorkItemProposal, resource: ResourceClass) -> None:
    if resource.name != proposal.resource_class:
        raise _Blocked(
            DomainBlockCode.RESOURCE_CONTRACT,
            "Coder resource class differs from the admitted proposal",
        )
    if (
        proposal.execution.nodes != resource.nodes
        or proposal.execution.ranks_per_node != resource.tasks_per_node
        or proposal.execution.gpus_per_node != resource.gpus_per_node
    ):
        raise _Blocked(
            DomainBlockCode.RESOURCE_CONTRACT,
            "Coder execution shape differs from the selected resource class",
        )


def _validate_candidate_profile(
    item: WorkItemRecord,
    proposal: WorkItemProposal,
) -> None:
    if proposal.kind in CATALOG_ITEM_KINDS:
        expected = DomainProfile.SMITH
    elif proposal.kind is PolicyWorkItemKind.TUNE_HYPOTHESIS:
        expected = DomainProfile.TUNER
    elif proposal.kind in {*ASSEMBLER_ITEM_KINDS, PolicyWorkItemKind.GATE}:
        expected = DomainProfile.ASSEMBLER
    else:
        raise _Blocked(
            DomainBlockCode.PROFILE_OR_KIND,
            "candidate action planner does not accept this work-item kind",
        )
    if item.profile is not expected:
        raise _Blocked(
            DomainBlockCode.PROFILE_OR_KIND,
            f"work-item kind {proposal.kind.value!r} requires profile {expected.value!r}",
        )


def _validate_non_smith_profile(
    item: WorkItemRecord,
    proposal: WorkItemProposal,
) -> None:
    """Retain the collective planner's private compatibility import."""
    _validate_candidate_profile(item, proposal)


def _validate_resource_name(resource: ResourceClass, expected: str) -> None:
    """Retain the collective adapter's private exact-resource guard."""
    if resource.name != expected:
        raise _Blocked(
            DomainBlockCode.RESOURCE_CONTRACT,
            f"action requires resource class {expected!r}",
        )


def _validate_gate_execution(
    gate: GateSpec,
    execution: GateExecutionContract,
    resource: ResourceClass,
) -> None:
    if not isinstance(gate, GateSpec):
        raise _Blocked(
            DomainBlockCode.GATE_CONTRACT,
            "one deterministic gate attempt requires exactly one GateSpec",
        )
    if not isinstance(execution, GateExecutionContract):
        raise _Blocked(
            DomainBlockCode.GATE_CONTRACT,
            "deterministic gate requires a typed GateExecutionContract",
        )
    if gate.phase is GatePhase.COLLECTIVE or execution.scope not in {
        EvidenceScope.CPU_STATIC,
        EvidenceScope.SINGLE_GPU_PRODUCT,
    }:
        raise _Blocked(
            DomainBlockCode.RANK_LAUNCHER_REQUIRED,
            "collective and multi-rank gates require the trusted rank launcher/supervisor",
        )
    if execution.scope is EvidenceScope.CPU_STATIC:
        if resource.total_tasks != 1 or resource.total_gpus != 0:
            raise _Blocked(
                DomainBlockCode.RESOURCE_CONTRACT,
                "CPU-static gate resource must request one task and zero GPUs",
            )
        return
    if (resource.nodes, resource.tasks_per_node, resource.gpus_per_node) != (1, 1, 1):
        raise _Blocked(
            DomainBlockCode.RESOURCE_CONTRACT,
            "single-GPU product gate requires exactly one node, task, and GPU",
        )


def _validate_candidate(
    item: WorkItemRecord,
    proposal: WorkItemProposal,
    candidate: FrozenCandidate,
    *,
    require_frozen: bool,
) -> None:
    if candidate.item_id != item.item_id:
        raise _Blocked(DomainBlockCode.CANDIDATE_MISMATCH, "candidate belongs to another item")
    coder = _attempt(item, candidate.coder_attempt_id)
    if (
        coder.role is not Role.CODER
        or coder.kind is not AttemptKind.ROLE
        or coder.status is not AttemptStatus.VALIDATED
        or coder.candidate_digest != candidate.digest
    ):
        raise _Blocked(
            DomainBlockCode.CANDIDATE_MISMATCH,
            "candidate must come from the validated Coder attempt",
        )
    if require_frozen and (
        item.candidate_attempt_id != candidate.coder_attempt_id
        or item.candidate_digest != candidate.digest
    ):
        raise _Blocked(
            DomainBlockCode.CANDIDATE_MISMATCH,
            "candidate differs from the authoritative frozen candidate",
        )
    if proposal.modifies_files:
        if not candidate.changed_paths or not set(candidate.changed_paths).issubset(
            proposal.allowed_paths
        ):
            raise _Blocked(
                DomainBlockCode.ALLOWED_PATHS,
                "candidate paths are not a non-empty subset of allowed_paths",
            )
    elif candidate.changed_paths:
        raise _Blocked(
            DomainBlockCode.ALLOWED_PATHS,
            "a non-modifying proposal cannot carry changed candidate paths",
        )


def _validate_assembler_paths(
    assembler_input: AssemblerInput,
    allowed_paths: Sequence[str],
) -> None:
    if not allowed_paths:
        raise _Blocked(
            DomainBlockCode.ALLOWED_PATHS,
            "Assembler Coder requires explicit product allowed_paths",
        )
    paths = tuple(PurePosixPath(path) for path in allowed_paths)
    updates_router_index = ROUTER_INDEX_PATH in paths
    try:
        validate_assembler_output(
            assembler_input,
            paths,
            updates_router_index=updates_router_index,
        )
    except ValueError as error:
        raise _Blocked(DomainBlockCode.ALLOWED_PATHS, str(error)) from error


def _new_attempt(
    state: RunState,
    item: WorkItemRecord,
    role: Role,
    kind: AttemptKind,
    *,
    resource_class: str,
    review_of_attempt_id: str | None = None,
    reviewed_candidate_digest: str | None = None,
) -> AttemptRecord:
    if item.attempts and not item.attempts[-1].terminal:
        raise _Blocked(
            DomainBlockCode.INVALID_STATE,
            "a separate attempt cannot start while the prior attempt is active",
        )
    scheduler_ids = tuple(
        attempt.job.scheduler_id for attempt in item.attempts if attempt.job is not None
    )
    if len(scheduler_ids) != len(set(scheduler_ids)):
        raise _Blocked(
            DomainBlockCode.INVALID_STATE,
            "prior role, gate, Reviewer, and QA attempts must use distinct scheduler jobs",
        )
    sequence = len(item.attempts) + 1
    suffix = f".{sequence:04d}.{kind.value}"
    attempt_id = f"{item.item_id}{suffix}"
    if len(attempt_id) > 128:
        digest = hashlib.sha256(item.item_id.encode("utf-8")).hexdigest()[:16]
        prefix_length = 128 - len(suffix) - len(digest) - 1
        attempt_id = f"{item.item_id[:prefix_length]}.{digest}{suffix}"
    return AttemptRecord(
        attempt_id=attempt_id,
        item_id=item.item_id,
        sequence=sequence,
        role=role,
        kind=kind,
        generation=state.generation,
        profile=item.profile,
        resource_class=resource_class,
        review_of_attempt_id=review_of_attempt_id,
        reviewed_candidate_digest=reviewed_candidate_digest,
    )


def _agent_action(
    state: RunState,
    proposal: WorkItemProposal,
    attempt: AttemptRecord,
    resource: ResourceClass,
    *,
    workspace: Path,
    base_commit: str,
    settings: AgentSettings,
    evidence_paths: Sequence[str],
    candidate: FrozenCandidate | None,
    prompt: str,
    context: dict[str, JsonValue],
    kind: DomainActionKind,
) -> DomainAttemptAction:
    authorized_evidence = _validate_paths(evidence_paths, "authorized evidence paths")
    runtime: dict[str, JsonValue] = {
        "schema_version": 1,
        "kind": "agent",
        "backend_kind": settings.backend_kind,
        "model": settings.model,
        "prompt_id": attempt.attempt_id,
        "prompt": prompt,
        "evidence_paths": list(authorized_evidence),
        "candidate_digest": candidate.digest if candidate is not None else None,
    }
    return _action(
        state,
        proposal,
        attempt,
        resource,
        workspace=workspace,
        base_commit=base_commit,
        payload={"runtime": runtime, "context": context},
        allowed_paths=proposal.allowed_paths if attempt.role is Role.CODER else (),
        kind=kind,
    )


def _gate_action(
    state: RunState,
    proposal: WorkItemProposal,
    attempt: AttemptRecord,
    resource: ResourceClass,
    *,
    workspace: Path,
    base_commit: str,
    candidate: FrozenCandidate,
    gate: GateSpec,
    execution: GateExecutionContract,
    context: dict[str, JsonValue],
) -> DomainAttemptAction:
    runtime: dict[str, JsonValue] = {
        "schema_version": 1,
        "kind": "gate",
        "command": {
            "command_id": gate.gate_id,
            "argv": list(gate.command.argv),
            "environment": dict(gate.command.environment),
            "cwd": ".",
        },
        "candidate_attempt_id": candidate.coder_attempt_id,
        "candidate_digest": candidate.digest,
        "receipt": {
            "gate_id": gate.gate_id,
            "purpose": gate.purpose.value,
            "scope": execution.scope.value,
            "certification_mode": _certification_mode(execution.scope).value,
            "expected_world_size": execution.expected_world_size,
            "expected_rank": execution.expected_rank,
            "expected_local_rank": execution.expected_local_rank,
            "product_rank_body": execution.product_rank_body,
            "accuracy": _accuracy_context(gate),
        },
    }
    return _action(
        state,
        proposal,
        attempt,
        resource,
        workspace=workspace,
        base_commit=base_commit,
        payload={"runtime": runtime, "context": context},
        allowed_paths=(),
        kind=DomainActionKind.DETERMINISTIC_GATE,
    )


def _certification_mode(scope: EvidenceScope) -> CertificationMode:
    if scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER:
        return CertificationMode.SYNTHETIC
    if scope is EvidenceScope.REAL_MULTI_NODE_PRODUCT:
        return CertificationMode.REAL
    return CertificationMode.LOCAL


def _review_action(
    state: RunState,
    proposal: WorkItemProposal,
    attempt: AttemptRecord,
    resource: ResourceClass,
    *,
    workspace: Path,
    candidate: FrozenCandidate,
    settings: AgentSettings,
    evidence_paths: Sequence[str],
    gates: Sequence[GateSpec],
    gate_attempt_ids: Sequence[str],
    kind: DomainActionKind,
    candidate_evidence: JsonValue | None,
) -> DomainAttemptAction:
    prompt = (
        "Review the exact frozen candidate in a fresh snapshot and independently rerun the "
        "smallest decisive evidence from the structured gate matrix. Return only the typed "
        f"{kind.value} result."
    )
    if candidate_evidence is not None:
        prompt += (
            " Independently validate the exact immutable tuning measurement envelopes and "
            "reject any identity, hard-gate, or uncertainty mismatch."
        )
    context: dict[str, JsonValue] = {
        "action": kind.value,
        "proposal": _proposal_context(proposal),
        "candidate": _candidate_context(candidate),
        "gate_attempt_ids": list(gate_attempt_ids),
        "gates": _gate_context(gates),
    }
    if candidate_evidence is not None:
        context["candidate_evidence"] = candidate_evidence
    return _agent_action(
        state,
        proposal,
        attempt,
        resource,
        workspace=workspace,
        base_commit=candidate.commit,
        settings=settings,
        evidence_paths=evidence_paths,
        candidate=candidate,
        prompt=prompt,
        context=context,
        kind=kind,
    )


def _action(
    state: RunState,
    proposal: WorkItemProposal,
    attempt: AttemptRecord,
    resource: ResourceClass,
    *,
    workspace: Path,
    base_commit: str,
    payload: dict[str, JsonValue],
    allowed_paths: Sequence[str],
    kind: DomainActionKind,
) -> DomainAttemptAction:
    root = workspace.expanduser().resolve()
    worktree = root / "candidates" / proposal.item_id / attempt.attempt_id
    attempt_dir = root / "items" / proposal.item_id / "attempts" / f"{attempt.sequence:04d}"
    manifest = WorkerInputManifest(
        run_id=state.run_id,
        item_id=proposal.item_id,
        attempt_id=attempt.attempt_id,
        task_digest=state.task_digest,
        generation=state.generation,
        role=attempt.role,
        profile=attempt.profile,
        worktree=str(worktree),
        allowed_paths=tuple(allowed_paths),
        payload=payload,
    )
    return DomainAttemptAction(
        kind=kind,
        proposal=proposal,
        attempt=attempt,
        resource=resource,
        worktree=worktree,
        branch=f"staircase/{state.run_id}/{attempt.attempt_id}",
        base_commit=base_commit,
        attempt_dir=attempt_dir,
        read_only=attempt.role is not Role.CODER,
        manifest=manifest,
    )


def _proposal_context(proposal: WorkItemProposal) -> dict[str, JsonValue]:
    return {
        "item_id": proposal.item_id,
        "goal_id": proposal.goal_id,
        "kind": proposal.kind.value,
        "resource_class": proposal.resource_class,
        "modifies_files": proposal.modifies_files,
        "dependencies": list(proposal.dependencies),
        "allowed_paths": list(proposal.allowed_paths),
        "execution": {
            "nodes": proposal.execution.nodes,
            "ranks_per_node": proposal.execution.ranks_per_node,
            "gpus_per_node": proposal.execution.gpus_per_node,
            "array_element": proposal.execution.array_element,
            "verdict_scope": proposal.execution.verdict_scope.value,
        },
    }


def _assembler_context(assembler_input: AssemblerInput) -> dict[str, JsonValue]:
    scope = assembler_input.scope
    if isinstance(assembler_input, CoreAssemblerInput):
        unit = "core"
        extension: dict[str, JsonValue] = {}
    elif isinstance(assembler_input, FeatureAssemblerInput):
        unit = "feature"
        extension = {"feature": assembler_input.feature}
    elif isinstance(assembler_input, WeightsAssemblerInput):
        unit = "weights"
        extension = {}
    elif isinstance(assembler_input, RoutingAssemblerInput):
        unit = "routing"
        extension = {"new_architecture_family": assembler_input.new_architecture_family}
    else:
        raise TypeError(f"unsupported Assembler input {type(assembler_input).__name__}")
    route = scope.route
    return {
        "unit": unit,
        "route": {
            "architecture": route.architecture,
            "family": route.family,
            "checkpoint_id": route.checkpoint_id,
            "sm": route.sm,
            "structural_features": list(route.structural_features),
            "target_directory": str(route.target_directory),
            "target_module": route.target_module,
            "synthetic_class": route.synthetic_class,
            "mapping": {
                "world_size": route.mapping.world_size,
                "tensor_parallel_size": route.mapping.tensor_parallel_size,
                "pipeline_parallel_size": route.mapping.pipeline_parallel_size,
                "moe_expert_parallel_size": route.mapping.moe_expert_parallel_size,
                "moe_tensor_parallel_size": route.mapping.moe_tensor_parallel_size,
                "attention_data_parallel_size": route.mapping.attention_data_parallel_size,
            },
        },
        "dependency_item_ids": list(scope.dependency_item_ids),
        "catalog_surfaces": [
            {
                "source_item_id": surface.source_item_id,
                "entry_id": surface.entry_id,
                "catalog_path": str(surface.catalog_path),
            }
            for surface in scope.catalog_surfaces
        ],
        **extension,
    }


def _hypothesis_context(hypothesis: TuningHypothesis) -> dict[str, JsonValue]:
    change = hypothesis.change
    return {
        "hypothesis_id": hypothesis.hypothesis_id,
        "item_id": hypothesis.item_id,
        "statement": hypothesis.statement,
        "metric": hypothesis.metric,
        "direction": hypothesis.direction.value,
        "change": {
            "name": change.name,
            "kind": change.kind.value,
            "baseline_value": change.baseline_value,
            "candidate_value": change.candidate_value,
        },
        "uncertainty": {
            "minimum_effect": hypothesis.uncertainty.minimum_effect,
            "noise_threshold": hypothesis.uncertainty.noise_threshold,
            "maximum_combined_uncertainty": (hypothesis.uncertainty.maximum_combined_uncertainty),
        },
    }


def _gate_context(gates: Sequence[GateSpec]) -> list[JsonValue]:
    if not gates:
        raise _Blocked(DomainBlockCode.GATE_CONTRACT, "gate matrix cannot be empty")
    if not all(isinstance(gate, GateSpec) for gate in gates):
        raise _Blocked(DomainBlockCode.GATE_CONTRACT, "gate matrix must contain GateSpec values")
    phases = [gate.phase for gate in gates]
    identifiers = [gate.gate_id for gate in gates]
    if phases != sorted(phases) or len(phases) != len(set(phases)):
        raise _Blocked(DomainBlockCode.GATE_CONTRACT, "gate phases must be unique and ordered")
    if len(identifiers) != len(set(identifiers)):
        raise _Blocked(DomainBlockCode.GATE_CONTRACT, "gate IDs must be unique")
    return [
        {
            "gate_id": gate.gate_id,
            "phase": int(gate.phase),
            "purpose": gate.purpose.value,
            "hard_gate": gate.hard_gate,
            "command": {
                "argv": list(gate.command.argv),
                "environment": dict(gate.command.environment),
                "cwd": ".",
            },
        }
        for gate in gates
    ]


def _single_gate_context(gate: GateSpec) -> JsonValue:
    if not isinstance(gate, GateSpec):
        raise _Blocked(
            DomainBlockCode.GATE_CONTRACT,
            "one deterministic gate attempt requires exactly one GateSpec",
        )
    return _gate_context((gate,))[0]


def _gate_execution_context(execution: GateExecutionContract) -> dict[str, JsonValue]:
    return {
        "scope": execution.scope.value,
        "expected_world_size": execution.expected_world_size,
        "expected_rank": execution.expected_rank,
        "expected_local_rank": execution.expected_local_rank,
        "product_rank_body": execution.product_rank_body,
    }


def _accuracy_context(gate: GateSpec) -> JsonValue:
    if gate.accuracy is None:
        return None
    return {
        "selector": gate.accuracy.selector,
        "reference": gate.accuracy.reference,
        "protocol": gate.accuracy.protocol,
        "tolerance": gate.accuracy.tolerance,
    }


def _candidate_context(candidate: FrozenCandidate) -> dict[str, JsonValue]:
    return {
        "item_id": candidate.item_id,
        "coder_attempt_id": candidate.coder_attempt_id,
        "commit": candidate.commit,
        "digest": candidate.digest,
        "changed_paths": list(candidate.changed_paths),
    }


def _validated_gate_attempt_ids(
    item: WorkItemRecord,
    candidate: FrozenCandidate,
    gates: Sequence[GateSpec],
    gate_attempt_ids: Sequence[str],
) -> tuple[str, ...]:
    _gate_context(gates)
    identifiers = tuple(gate_attempt_ids)
    if len(identifiers) != len(gates) or len(set(identifiers)) != len(identifiers):
        raise _Blocked(
            DomainBlockCode.GATE_CONTRACT,
            "Reviewer and QA require one distinct validated receipt per ordered GateSpec",
        )
    attempts = tuple(_attempt(item, attempt_id) for attempt_id in identifiers)
    if [attempt.sequence for attempt in attempts] != sorted(
        attempt.sequence for attempt in attempts
    ):
        raise _Blocked(
            DomainBlockCode.GATE_CONTRACT,
            "validated gate attempt IDs must follow the ordered GateSpec suite",
        )
    if any(
        attempt.kind is not AttemptKind.DETERMINISTIC_GATE
        or attempt.role is not Role.GATE
        or attempt.status is not AttemptStatus.VALIDATED
        or attempt.candidate_digest != candidate.digest
        for attempt in attempts
    ):
        raise _Blocked(
            DomainBlockCode.GATE_CONTRACT,
            "Reviewer and QA require validated candidate-pinned gate attempts",
        )
    return identifiers


def _attempt(item: WorkItemRecord, attempt_id: str) -> AttemptRecord:
    for attempt in item.attempts:
        if attempt.attempt_id == attempt_id:
            return attempt
    raise _Blocked(DomainBlockCode.INVALID_STATE, f"unknown attempt {attempt_id!r}")


def _validate_paths(paths: Sequence[str], name: str) -> tuple[str, ...]:
    normalized = tuple(paths)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} contain duplicates")
    for value in normalized:
        path = PurePosixPath(value)
        if (
            not isinstance(value, str)
            or not value
            or path.is_absolute()
            or str(path) != value
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError(f"{name} contain an unsafe relative path: {value!r}")
    return normalized


__all__ = [
    "AgentSettings",
    "DomainActionBlock",
    "DomainActionKind",
    "DomainActionPlan",
    "DomainAttemptAction",
    "DomainBlockCode",
    "FrozenCandidate",
    "GateExecutionContract",
    "plan_assembler_coder_action",
    "plan_deterministic_gate_action",
    "plan_qa_action",
    "plan_reviewer_analysis_action",
    "plan_reviewer_rerun_action",
    "plan_tuner_coder_action",
]
