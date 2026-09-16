# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure resolution of reviewed plans into non-secret production inputs.

The resolver intentionally does not import the runtime, scheduler, worker, or
Slurm layers.  Concrete allocation, node placement, credentials, and launcher
contracts remain controller-owned decisions made after this immutable intent
has been admitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ..common.gates import AccuracyCriteria, GatePhase, build_gate_suite
from ..common.outcomes import PlanDraftOutcome, PlanReviewDecision, PlanReviewOutcome
from ..common.policy import (
    ASSEMBLER_ITEM_KINDS,
    AssemblerFeatureInput,
    PolicyViolation,
    WorkItemKind,
    validate_plan_domain_inputs,
)
from ..state import WorkflowMode
from ..task_schema import NormalizedTask
from ..tuning.contracts import TuningHypothesis

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProductionInputError(ValueError):
    """Raised when reviewed evidence cannot produce unambiguous inputs."""


class GateExecutionIntent(str, Enum):
    """Allocation-independent shape from which runtime builds a gate contract."""

    CPU_STATIC = "cpu_static"
    SINGLE_GPU_PRODUCT = "single_gpu_product"
    TARGET_PRODUCT = "target_product"


@dataclass(frozen=True, slots=True)
class GateResourceIntent:
    """Exact task-envelope subrequest without allocation or placement facts."""

    resource_class: str
    nodes: int
    tasks_per_node: int
    gpus_per_node: int
    cpus_per_task: int
    memory_mib: int
    time_limit_seconds: int

    def __post_init__(self) -> None:
        if self.resource_class != "deterministic_gate":
            raise ProductionInputError("gate intent must retain deterministic_gate identity")
        for name, value, minimum in (
            ("nodes", self.nodes, 1),
            ("tasks_per_node", self.tasks_per_node, 1),
            ("gpus_per_node", self.gpus_per_node, 0),
            ("cpus_per_task", self.cpus_per_task, 1),
            ("memory_mib", self.memory_mib, 1),
            ("time_limit_seconds", self.time_limit_seconds, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ProductionInputError(f"gate resource {name} must be >= {minimum}")


@dataclass(frozen=True, slots=True)
class GateScopeIntent:
    """One ordered gate's evidence shape and exact resource intent."""

    gate_id: str
    phase: int
    execution: GateExecutionIntent
    expected_world_size: int
    product_rank_body: bool
    requires_rank_supervisor: bool
    resource: GateResourceIntent

    def __post_init__(self) -> None:
        if not self.gate_id or not isinstance(self.phase, int):
            raise ProductionInputError("gate intent requires an identity and numeric phase")
        if not isinstance(self.execution, GateExecutionIntent):
            raise ProductionInputError("gate intent requires a typed execution shape")
        if (
            isinstance(self.expected_world_size, bool)
            or not isinstance(self.expected_world_size, int)
            or self.expected_world_size < 0
        ):
            raise ProductionInputError("gate expected_world_size must be non-negative")
        expected = {
            GateExecutionIntent.CPU_STATIC: 0,
            GateExecutionIntent.SINGLE_GPU_PRODUCT: 1,
        }.get(self.execution)
        if expected is not None and self.expected_world_size != expected:
            raise ProductionInputError("gate execution intent has an inconsistent world size")
        if self.execution is GateExecutionIntent.TARGET_PRODUCT and self.expected_world_size < 1:
            raise ProductionInputError("target-product gates require a positive world size")
        if not isinstance(self.product_rank_body, bool) or not isinstance(
            self.requires_rank_supervisor, bool
        ):
            raise ProductionInputError("gate product and supervisor intent must be boolean")
        expected_product = self.execution is not GateExecutionIntent.CPU_STATIC
        if self.product_rank_body is not expected_product:
            raise ProductionInputError("gate product-rank intent disagrees with execution")
        if self.requires_rank_supervisor != (self.expected_world_size > 1):
            raise ProductionInputError("multi-rank gate supervisor intent is inconsistent")
        if not isinstance(self.resource, GateResourceIntent):
            raise ProductionInputError("gate scope requires a typed resource intent")
        if self.resource.nodes * self.resource.tasks_per_node != max(1, self.expected_world_size):
            raise ProductionInputError("gate resource tasks disagree with expected world size")
        expected_gpus = 0 if self.execution is GateExecutionIntent.CPU_STATIC else 1
        if self.resource.gpus_per_node != expected_gpus * self.resource.tasks_per_node:
            raise ProductionInputError("gate resource GPUs disagree with execution intent")


@dataclass(frozen=True, slots=True)
class RoleEvidencePolicy:
    """Least-privilege relative artifact paths authorized for one role."""

    role: str
    paths: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in {"coder", "qa", "reviewer"}:
            raise ProductionInputError("evidence policy contains an unknown role")
        if not self.paths:
            raise ProductionInputError("every evidence role requires at least one path")


@dataclass(frozen=True, slots=True)
class EvidencePathPolicy:
    """Controller-owned, allocation-independent evidence artifact policy."""

    by_role: tuple[RoleEvidencePolicy, ...]

    def __post_init__(self) -> None:
        roles = [entry.role for entry in self.by_role]
        if roles != sorted(roles) or len(roles) != len(set(roles)):
            raise ProductionInputError("evidence roles must be sorted and unique")
        paths = [path for entry in self.by_role for path in entry.paths]
        if len(paths) != len(set(paths)):
            raise ProductionInputError("evidence paths must be unique across roles")
        if any(
            not path
            or path.startswith("/")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            for path in paths
        ):
            raise ProductionInputError("evidence paths must be canonical relative paths")

    def paths_for(self, role: str) -> tuple[str, ...]:
        """Return the exact paths authorized for ``role`` without a fallback."""
        for entry in self.by_role:
            if entry.role == role:
                return entry.paths
        raise KeyError(role)


@dataclass(frozen=True, slots=True)
class ProductionInputs:
    """Immutable bridge from approved plan facts to production construction."""

    plan_digest: str
    assembler_features: tuple[tuple[str, str], ...]
    tuning_hypotheses: tuple[TuningHypothesis, ...]
    qa_required_item_ids: frozenset[str]
    ordered_gate_ids: tuple[str, ...]
    gate_scope_intents: tuple[GateScopeIntent, ...]
    evidence_path_policy: EvidencePathPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.plan_digest, str) or not _SHA256.fullmatch(self.plan_digest):
            raise ProductionInputError("production inputs require a plan SHA-256 digest")
        feature_items = [item_id for item_id, _feature in self.assembler_features]
        hypothesis_items = [hypothesis.item_id for hypothesis in self.tuning_hypotheses]
        if len(feature_items) != len(set(feature_items)):
            raise ProductionInputError("production inputs contain duplicate feature items")
        if len(hypothesis_items) != len(set(hypothesis_items)):
            raise ProductionInputError("production inputs contain duplicate hypothesis items")
        intent_ids = tuple(intent.gate_id for intent in self.gate_scope_intents)
        if intent_ids != self.ordered_gate_ids:
            raise ProductionInputError("gate scope intents must exactly preserve gate order")


def resolve_production_inputs(
    task: NormalizedTask,
    workflow_mode: WorkflowMode,
    plan: PlanDraftOutcome,
    review: PlanReviewOutcome,
) -> ProductionInputs:
    """Resolve only facts pinned by an approved plan and normalized task.

    No value is recovered from prose, filenames, process environment, node
    placement, credentials, or a caller-provided default.
    """
    if not isinstance(task, NormalizedTask):
        raise ProductionInputError("production input resolution requires a NormalizedTask")
    if not isinstance(workflow_mode, WorkflowMode):
        raise ProductionInputError("production input resolution requires a WorkflowMode")
    if not isinstance(plan, PlanDraftOutcome) or not isinstance(review, PlanReviewOutcome):
        raise ProductionInputError("production input resolution requires typed planning evidence")
    if review.outcome is not PlanReviewDecision.ACCEPT:
        raise ProductionInputError("production inputs require an approved plan")
    if review.plan_digest != plan.digest:
        raise ProductionInputError("PlanReviewer evidence does not pin this plan")
    try:
        validate_plan_domain_inputs(
            plan.items,
            workflow_mode=workflow_mode.value,
            target_features=task.target.features,
        )
    except PolicyViolation as error:
        raise ProductionInputError(str(error)) from error

    try:
        gate_envelope = task.execution.slurm.smith.resource_class("deterministic_gate")
    except KeyError:
        raise ProductionInputError("task lacks the deterministic_gate resource class")

    gates = build_gate_suite(
        entry_gpu_tests=task.gates.component_tests,
        collective_tests=task.gates.collective_tests,
        boot_tests=task.gates.boot_tests,
        accuracy=AccuracyCriteria(
            selector=task.gates.accuracy.selector,
            reference=task.gates.accuracy.reference,
            protocol=task.gates.accuracy.protocol,
            tolerance=task.gates.accuracy.tolerance,
        ),
        feature_signal_tests=task.gates.feature_signal_tests,
        base_environment=dict(task.execution.slurm.controller.environment),
    )
    ordered_gate_ids = tuple(gate.gate_id for gate in gates)
    declared_exit_gates = tuple(gate for stage in plan.stages for gate in stage.exit_gates)
    if len(declared_exit_gates) != len(set(declared_exit_gates)):
        raise ProductionInputError("Stage exit gates contain ambiguous duplicate identities")
    if declared_exit_gates != ordered_gate_ids:
        missing = sorted(set(ordered_gate_ids) - set(declared_exit_gates))
        unknown = sorted(set(declared_exit_gates) - set(ordered_gate_ids))
        raise ProductionInputError(
            "Stage exit gates must exactly preserve the task gate suite; "
            f"missing={missing}, unknown={unknown}"
        )

    assembler_features: list[tuple[str, str]] = []
    tuning_hypotheses: list[TuningHypothesis] = []
    for item in plan.items:
        if item.kind is WorkItemKind.ASSEMBLE_FEATURE:
            domain_input = item.domain_input
            if not isinstance(domain_input, AssemblerFeatureInput):
                raise ProductionInputError("assemble_feature input changed after plan validation")
            assembler_features.append((item.item_id, domain_input.feature))
        elif item.kind is WorkItemKind.TUNE_HYPOTHESIS:
            domain_input = item.domain_input
            if not isinstance(domain_input, TuningHypothesis):
                raise ProductionInputError("tuning input changed after plan validation")
            tuning_hypotheses.append(domain_input)

    qa_required_item_ids = frozenset(
        item.item_id
        for item in plan.items
        if item.modifies_files
        or item.kind in ASSEMBLER_ITEM_KINDS
        or item.kind is WorkItemKind.TUNE_HYPOTHESIS
    )
    gate_scope_intents = tuple(
        _gate_scope_intent(
            gate.phase,
            gate.gate_id,
            task.target.world_size,
            envelope_nodes=gate_envelope.nodes,
            envelope_tasks_per_node=gate_envelope.tasks_per_node,
            envelope_gpus_per_node=gate_envelope.gpus_per_node,
            envelope_cpus_per_task=gate_envelope.cpus_per_task,
            envelope_memory_mib=gate_envelope.memory_mib,
            envelope_time_limit_seconds=gate_envelope.time_limit_seconds,
        )
        for gate in gates
    )
    evidence_path_policy = EvidencePathPolicy(
        by_role=(
            RoleEvidencePolicy("coder", ("reports/coder.json",)),
            RoleEvidencePolicy("qa", ("reports/qa.json",)),
            RoleEvidencePolicy("reviewer", ("reports/reviewer.json",)),
        )
    )
    return ProductionInputs(
        plan_digest=plan.digest,
        assembler_features=tuple(assembler_features),
        tuning_hypotheses=tuple(tuning_hypotheses),
        qa_required_item_ids=qa_required_item_ids,
        ordered_gate_ids=ordered_gate_ids,
        gate_scope_intents=gate_scope_intents,
        evidence_path_policy=evidence_path_policy,
    )


def _gate_scope_intent(
    phase: GatePhase,
    gate_id: str,
    target_world_size: int,
    *,
    envelope_nodes: int,
    envelope_tasks_per_node: int,
    envelope_gpus_per_node: int,
    envelope_cpus_per_task: int,
    envelope_memory_mib: int,
    envelope_time_limit_seconds: int,
) -> GateScopeIntent:
    if phase in {GatePhase.CLAIMS_ROUTING_NO_STALE, GatePhase.NATIVE_CONTRACT}:
        execution = GateExecutionIntent.CPU_STATIC
        world_size = 0
        topology = (1, 1, 0)
    elif phase is GatePhase.ENTRY_GPU:
        execution = GateExecutionIntent.SINGLE_GPU_PRODUCT
        world_size = 1
        topology = (1, 1, 1)
    else:
        execution = GateExecutionIntent.TARGET_PRODUCT
        world_size = target_world_size
        if (
            envelope_nodes * envelope_tasks_per_node != target_world_size
            or envelope_gpus_per_node != envelope_tasks_per_node
        ):
            raise ProductionInputError(
                "deterministic_gate envelope must exactly represent target product topology"
            )
        topology = (
            envelope_nodes,
            envelope_tasks_per_node,
            envelope_gpus_per_node,
        )
    resource = GateResourceIntent(
        "deterministic_gate",
        *topology,
        envelope_cpus_per_task,
        envelope_memory_mib,
        envelope_time_limit_seconds,
    )
    return GateScopeIntent(
        gate_id=gate_id,
        phase=int(phase),
        execution=execution,
        expected_world_size=world_size,
        product_rank_body=execution is not GateExecutionIntent.CPU_STATIC,
        requires_rank_supervisor=world_size > 1,
        resource=resource,
    )


__all__ = [
    "EvidencePathPolicy",
    "GateExecutionIntent",
    "GateResourceIntent",
    "GateScopeIntent",
    "ProductionInputError",
    "ProductionInputs",
    "RoleEvidencePolicy",
    "resolve_production_inputs",
]
