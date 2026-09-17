# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, deterministic controller ticks for the Staircase workflow.

The runtime deliberately does not contain a sleep loop.  Its caller owns
polling, signal handling, and lease lifetime; one call to :meth:`tick` makes at
most one authoritative state transition or dispatches one newly persisted
Smith action.  This keeps re-entry after controller preemption deterministic.

Planning retains its role-process transport.  Admitted work uses generic
``WorkerInputManifest``/``WorkerResultManifest`` mailboxes, with state persisted
before a worktree or mailbox is materialized.  Missing domain facts and
execution adapters produce explicit typed blockers, never inferred work or a
fabricated successful WorkItem.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from ..common.artifacts import (
    INPUT_FILENAME,
    ResultExpectation,
    WorkerInputManifest,
    WorkerResultManifest,
    WorkerResultStatus,
    digest_file,
    load_input_manifest,
    load_result_manifest,
    worker_output_directory,
)
from ..common.credentials import CredentialBroker
from ..common.dispatch import AttemptReservation, ResourceCaps, select_ready_wave
from ..common.gates import (
    AccuracyCriteria,
    ClaimScope,
    EvidenceScope,
    GatePhase,
    GatePurpose,
    GateSpec,
    build_gate_suite,
    validate_receipt_claim,
)
from ..common.gitops import (
    ControllerGitOps,
    DeliveryCheckout,
    GitOpsError,
    IntegratedPatchExpectation,
    prepare_diff_delivery_checkout,
    prepare_isolated_delivery_checkout,
    publish_delivery_patch,
)
from ..common.index import load_catalog_index, render_index_deltas, update_catalog_index
from ..common.outcomes import (
    PlanDraftOutcome,
    PlanReviewDecision,
    PlanReviewOutcome,
    parse_plan_draft,
    parse_plan_review,
)
from ..common.placement import PLACEMENT_RECEIPT_FILENAME, PlacementError
from ..common.policy import CATALOG_ITEM_KINDS, PolicyViolation, WorkItemProposal
from ..common.policy import WorkItemKind as PolicyWorkItemKind
from ..common.runners import RoleProcessResult, load_role_spec
from ..common.slurm import JobIdentity, JobStatus, Scheduler
from ..onboarding.workflow import (
    AssemblerUnit,
    CatalogRequirement,
    IntegratedCatalogSurface,
    prepare_assembler_input,
    route_identity_from_task,
)
from ..state import (
    PLANNING_ITEM_ID,
    STATE_FILENAME,
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    GoalRecord,
    HierarchyStatus,
    IntegrationEvidenceKind,
    JobReference,
    Role,
    RunState,
    RunTerminalStatus,
    StageRecord,
    StageStatus,
    WorkflowMode,
    WorkItemKind,
    WorkItemRecord,
    WorkItemStatus,
    load_state,
    save_state,
)
from ..task_schema import NormalizedTask, ParallelMapping, ResourceClass, SlurmRole, SmithConfig
from ..tuning.artifacts import (
    BASELINE_FILENAME,
    CAMPAIGN_FILENAME,
    CANDIDATE_FILENAME,
    EVALUATION_FILENAME,
    PROMOTION_DECISION_FILENAME,
    load_baseline_artifact,
    load_campaign_artifact,
    load_candidate_artifact,
    load_evaluation_artifact,
    load_promotion_decision_artifact,
    replay_terminal_tuning_campaign,
    write_baseline_artifact,
    write_campaign_artifact,
    write_candidate_artifact,
    write_evaluation_artifact,
    write_promotion_decision_artifact,
)
from ..tuning.workflow import (
    DecisionOwner,
    PromotionAction,
    TuningHypothesis,
    evaluate_tuning_campaign,
    make_promotion_decision,
    new_tuning_campaign,
    record_tuning_evidence,
)
from .attempts import (
    AttemptAction,
    AttemptEngineError,
    AttemptExecution,
    AttemptPolicy,
    RetryLimitExceeded,
    cancel_owned_attempt,
    cancel_submitting_intent,
    logical_attempt_ordinal,
    logical_attempt_root,
    plan_infrastructure_retry,
    tick_attempt,
)
from .collective import (
    CollectiveAdapterError,
    TrustedCollectiveGateExecutionAdapter,
    build_local_product_certifications,
)
from .domain import (
    AgentSettings,
    DomainActionPlan,
    DomainAttemptAction,
    FrozenCandidate,
    GateExecutionContract,
    plan_assembler_coder_action,
    plan_deterministic_gate_action,
    plan_qa_action,
    plan_reviewer_analysis_action,
    plan_reviewer_rerun_action,
    plan_tuner_coder_action,
)
from .execution import ExecutionAdapterError, GenericWorkerExecutionAdapter, MaterializableAction
from .fan_in import ApprovedSmithCandidate, execute_fan_in, plan_fan_in
from .planning import PlanningAction, PlanningConfig, PlanningEngine, PlanningTickResult
from .production_inputs import GateExecutionIntent, ProductionInputs, resolve_production_inputs
from .recovery import plan_attempt_adoption
from .results import (
    CANDIDATE_RECEIPT_FILENAME,
    CandidateBinding,
    CandidateDisposition,
    CandidateReceipt,
    GateResult,
    QaResult,
    QaVerdict,
    ResultContractError,
    ReviewerResult,
    ReviewerVerdict,
    TunerResult,
    bind_coder_candidate,
    bind_tuner_candidate,
    decode_worker_result,
    reconstruct_candidate,
)
from .smith import (
    SmithAttemptAction,
    horizontal_smith_wave_for_item,
    mark_ready_after_integrated_dependencies,
    materialize_attempts,
    plan_coder_wave,
    plan_resource_escalation_attempt,
    validate_smith_role_wave_placement,
)

_ROLE_INPUT_FILENAME = "role-input.json"
_ROLE_RESULT_FILENAME = "role-result.json"
_ACTIVE_ATTEMPT_STATUSES = frozenset(
    {
        AttemptStatus.PREPARED,
        AttemptStatus.SUBMITTING,
        AttemptStatus.SUBMITTED,
        AttemptStatus.PENDING,
        AttemptStatus.RUNNING,
        AttemptStatus.TERMINAL_OBSERVED,
        AttemptStatus.COLLECTING,
    }
)
_PRODUCTION_WORK_ITEM_KINDS = frozenset(
    {
        PolicyWorkItemKind.CATALOG_VERIFY,
        PolicyWorkItemKind.CATALOG_ONBOARD,
        PolicyWorkItemKind.ASSEMBLE_CORE,
        PolicyWorkItemKind.ASSEMBLE_FEATURE,
        PolicyWorkItemKind.ROUTING,
        PolicyWorkItemKind.TUNE_HYPOTHESIS,
    }
)


class RuntimeErrorBase(RuntimeError):
    """Base class for deterministic controller-runtime errors."""


class FrozenPlanError(RuntimeErrorBase):
    """Raised when admitted state cannot be tied to approved planning evidence."""


class RuntimeDisposition(str, Enum):
    """Instruction returned to the outer, signal-aware controller process."""

    CONTINUE = "continue"
    WAIT = "wait"
    TERMINAL = "terminal"
    REQUEUE = "requeue"
    WAITING_INPUT = "waiting_input"


class RuntimeEvent(str, Enum):
    """One typed event completed, or blocker observed, by a runtime tick."""

    PLANNING = "planning"
    ITEM_READY = "item_ready"
    ATTEMPT_UPDATED = "attempt_updated"
    ATTEMPT_ADOPTED = "attempt_adopted"
    ACTION_PERSISTED = "action_persisted"
    WORKER_ACTION_MATERIALIZED = "worker_action_materialized"
    CREDENTIALS_REVOKED = "credentials_revoked"
    CANDIDATE_MATERIALIZED = "candidate_materialized"
    FAN_IN_INTENT_PERSISTED = "fan_in_intent_persisted"
    FAN_IN_MATERIALIZED = "fan_in_materialized"
    TUNING_CAMPAIGN_MATERIALIZED = "tuning_campaign_materialized"
    DELIVERY_PUBLISHED = "delivery_published"
    ITEM_INTEGRATED = "item_integrated"
    SMITH_ACTION_MATERIALIZED = "smith_action_materialized"
    HIERARCHY_UPDATED = "hierarchy_updated"
    CANCELLATION_UPDATED = "cancellation_updated"
    CANCEL_SIGNALLED = "cancel_signalled"
    STOP_DISPATCH = "stop_dispatch"
    BLOCKED = "blocked"
    NO_CHANGE = "no_change"
    TERMINAL = "terminal"
    REQUEUE = "requeue"
    WAITING_INPUT = "waiting_input"


@dataclass(frozen=True, slots=True)
class RuntimeTickResult:
    """Observable result of one bounded controller tick."""

    disposition: RuntimeDisposition
    event: RuntimeEvent
    revision: int
    reason: str
    item_id: str | None = None
    attempt_id: str | None = None
    terminal_status: RunTerminalStatus | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("runtime tick reason must be non-empty")
        if self.revision < 0:
            raise ValueError("runtime tick revision must be non-negative")
        if self.disposition is RuntimeDisposition.TERMINAL and self.terminal_status is None:
            raise ValueError("a terminal runtime disposition requires terminal_status")
        if self.disposition is not RuntimeDisposition.TERMINAL and self.terminal_status is not None:
            raise ValueError("only a terminal runtime disposition may carry terminal_status")


class AttemptExecutionFactory(Protocol):
    """Build the immutable scheduler/mailbox contract for one persisted attempt."""

    def __call__(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        attempt: AttemptRecord,
    ) -> AttemptExecution:
        """Return an execution whose identity exactly matches ``attempt``."""


class SmithMaterializer(Protocol):
    """Materialize controller-authored Smith actions after state persistence."""

    def __call__(self, actions: Sequence[SmithAttemptAction]) -> object:
        """Materialize immutable inputs without changing authoritative state."""


class WorkerExecutionAdapter(Protocol):
    """Materialize and reconstruct admitted generic worker actions."""

    def materialize(self, action: MaterializableAction) -> object:
        """Publish one immutable worker input after its state reservation."""

    def execution(self, state: RunState, action: MaterializableAction) -> AttemptExecution:
        """Rebuild the exact scheduler contract from persisted input bytes."""

    def revoke(self, state: RunState, attempt: AttemptRecord) -> bool:
        """Revoke exact terminal credentials; return whether a new receipt was written."""


class CollectiveGateExecutionAdapter(WorkerExecutionAdapter, Protocol):
    """Bridge a COLLECTIVE GateSpec to the trusted rank-supervisor runtime.

    The adapter must plan exactly one fresh deterministic-gate attempt and
    preserve the supplied candidate receipt, GateSpec, placement contract,
    task mapping, and resource class without weakening them.  Its materialize
    and execution methods must use the fixed internal rank-supervisor worker;
    a generic shell or agent worker is not an acceptable implementation.
    """

    def plan(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        candidate: FrozenCandidate,
        candidate_receipt: CandidateReceipt,
        gate: GateSpec,
        execution: GateExecutionContract,
        *,
        resource: ResourceClass,
        mapping: ParallelMapping,
        workspace: Path,
    ) -> DomainActionPlan:
        """Return one pure rank-gate reservation or a typed domain blocker."""


class RuntimeDomainInputResolver(Protocol):
    """Resolve approved immutable planning evidence into runtime-owned facts."""

    def __call__(
        self,
        task: NormalizedTask,
        workflow_mode: WorkflowMode,
        plan: PlanDraftOutcome,
        review: PlanReviewOutcome,
        backend_environment: tuple[tuple[str, str], ...],
    ) -> RuntimeDomainInputs:
        """Return inputs bound to the exact task and reviewed plan."""


@dataclass(frozen=True, slots=True)
class RuntimeDomainInputs:
    """Workflow-construction facts that cannot be inferred from a reviewed plan."""

    tuning_hypotheses: tuple[TuningHypothesis, ...] = ()
    assembler_features: tuple[tuple[str, str], ...] = ()
    gate_executions: tuple[tuple[str, GateExecutionContract], ...] = ()
    gate_resources: tuple[tuple[str, ResourceClass], ...] = ()
    qa_required_items: frozenset[str] | None = None
    evidence_paths: tuple[str, ...] = ()
    role_evidence_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()
    task_digest: str | None = None
    plan_digest: str | None = None

    def __post_init__(self) -> None:
        hypothesis_items = [hypothesis.item_id for hypothesis in self.tuning_hypotheses]
        if len(hypothesis_items) != len(set(hypothesis_items)):
            raise ValueError("runtime inputs contain duplicate tuning hypotheses")
        feature_items = [item_id for item_id, _feature in self.assembler_features]
        if len(feature_items) != len(set(feature_items)):
            raise ValueError("runtime inputs contain duplicate Assembler feature identities")
        gate_ids = [gate_id for gate_id, _execution in self.gate_executions]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("runtime inputs contain duplicate gate execution contracts")
        resource_gate_ids = [gate_id for gate_id, _resource in self.gate_resources]
        if len(resource_gate_ids) != len(set(resource_gate_ids)):
            raise ValueError("runtime inputs contain duplicate gate resource requests")
        if any(resource.name != "deterministic_gate" for _gate_id, resource in self.gate_resources):
            raise ValueError("every gate subrequest must retain deterministic_gate identity")
        role_names = [role for role, _paths in self.role_evidence_paths]
        if role_names != sorted(role_names) or len(role_names) != len(set(role_names)):
            raise ValueError("role evidence paths must have sorted unique role names")
        scoped_paths = [path for _role, paths in self.role_evidence_paths for path in paths]
        if len(scoped_paths) != len(set(scoped_paths)):
            raise ValueError("role evidence paths must be unique across roles")
        for name, digest in (("task_digest", self.task_digest), ("plan_digest", self.plan_digest)):
            if digest is not None and (
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    @property
    def hypotheses(self) -> Mapping[str, TuningHypothesis]:
        """Return hypotheses keyed by their immutable WorkItem identity."""
        return {hypothesis.item_id: hypothesis for hypothesis in self.tuning_hypotheses}

    @property
    def features(self) -> Mapping[str, str]:
        """Return explicit feature identities keyed by Assembler WorkItem."""
        return dict(self.assembler_features)

    @property
    def gate_execution_by_id(self) -> Mapping[str, GateExecutionContract]:
        """Return exact evidence topology keyed by frozen GateSpec identity."""
        return dict(self.gate_executions)

    @property
    def gate_resource_by_id(self) -> Mapping[str, ResourceClass]:
        """Return exact per-gate requests keyed by frozen GateSpec identity."""
        return dict(self.gate_resources)

    def evidence_paths_for(self, role: Role) -> tuple[str, ...]:
        """Return the least-privilege evidence policy for one agent role."""
        configured = dict(self.role_evidence_paths)
        if configured:
            try:
                return configured[role.value]
            except KeyError as error:
                raise RuntimeErrorBase(
                    f"runtime inputs have no evidence policy for role {role.value!r}"
                ) from error
        return self.evidence_paths


def resolve_runtime_domain_inputs(
    task: NormalizedTask,
    workflow_mode: WorkflowMode,
    plan: PlanDraftOutcome,
    review: PlanReviewOutcome,
    backend_environment: tuple[tuple[str, str], ...],
) -> RuntimeDomainInputs:
    """Translate neutral reviewed-plan facts into exact runtime contracts."""
    if backend_environment:
        raise ValueError(
            "runtime domain resolution cannot receive backend credential values; "
            "configure a trusted CredentialBroker instead"
        )
    resolved = resolve_production_inputs(task, workflow_mode, plan, review)
    return _runtime_inputs_from_production(task, resolved)


def _runtime_inputs_from_production(
    task: NormalizedTask,
    resolved: ProductionInputs,
) -> RuntimeDomainInputs:
    executions: list[tuple[str, GateExecutionContract]] = []
    resources: list[tuple[str, ResourceClass]] = []
    for intent in resolved.gate_scope_intents:
        resource = ResourceClass(
            intent.resource.resource_class,
            intent.resource.nodes,
            intent.resource.tasks_per_node,
            intent.resource.gpus_per_node,
            intent.resource.cpus_per_task,
            intent.resource.memory_mib,
            intent.resource.time_limit_seconds,
            intent.resource.gpu_allocation_padding,
            intent.resource.partition,
            intent.resource.qos,
        )
        if intent.execution is GateExecutionIntent.CPU_STATIC:
            execution = GateExecutionContract(
                EvidenceScope.CPU_STATIC,
                expected_world_size=0,
                expected_rank=None,
                expected_local_rank=None,
                product_rank_body=False,
            )
        elif intent.execution is GateExecutionIntent.SINGLE_GPU_PRODUCT:
            execution = GateExecutionContract(
                EvidenceScope.SINGLE_GPU_PRODUCT,
                expected_world_size=1,
                expected_rank=0,
                expected_local_rank=0,
                product_rank_body=True,
            )
        else:
            execution = _target_product_gate_contract(
                intent.expected_world_size,
                resource,
            )
        executions.append((intent.gate_id, execution))
        resources.append((intent.gate_id, resource))

    return RuntimeDomainInputs(
        tuning_hypotheses=resolved.tuning_hypotheses,
        assembler_features=resolved.assembler_features,
        gate_executions=tuple(executions),
        gate_resources=tuple(resources),
        qa_required_items=resolved.qa_required_item_ids,
        role_evidence_paths=tuple(
            (entry.role, entry.paths) for entry in resolved.evidence_path_policy.by_role
        ),
        task_digest=task.digest,
        plan_digest=resolved.plan_digest,
    )


def _target_product_gate_contract(
    world_size: int,
    resource: ResourceClass,
) -> GateExecutionContract:
    if world_size == 1:
        exact = (resource.nodes, resource.tasks_per_node, resource.gpus_per_node) == (1, 1, 1)
        padded = (
            resource.gpu_allocation_padding
            and resource.nodes == 1
            and resource.tasks_per_node == 1
            and resource.gpus_per_node > 1
        )
        if exact or padded:
            return GateExecutionContract(
                EvidenceScope.SINGLE_GPU_PRODUCT,
                expected_world_size=1,
                expected_rank=0,
                expected_local_rank=0,
                product_rank_body=True,
            )
    if (
        world_size == 4
        and resource.nodes == 1
        and resource.tasks_per_node == 4
        and resource.gpus_per_node == 4
    ):
        return GateExecutionContract(
            EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
            expected_world_size=4,
            expected_rank=None,
            expected_local_rank=None,
            product_rank_body=True,
        )
    if (
        resource.nodes >= 2
        and resource.total_tasks == world_size
        and resource.tasks_per_node == resource.gpus_per_node
    ):
        return GateExecutionContract(
            EvidenceScope.REAL_MULTI_NODE_PRODUCT,
            expected_world_size=world_size,
            expected_rank=None,
            expected_local_rank=None,
            product_rank_body=True,
        )
    raise RuntimeErrorBase(
        "target-product gate topology cannot be represented by the frozen "
        "deterministic_gate resource envelope"
    )


@dataclass(frozen=True, slots=True)
class _PersistedWorkerAction:
    """Generation-stable action projection recovered from immutable input bytes."""

    attempt: AttemptRecord
    resource: ResourceClass
    worktree: Path
    branch: str
    base_commit: str
    attempt_dir: Path
    manifest: WorkerInputManifest


@dataclass(frozen=True, slots=True)
class RuntimeCallbacks:
    """Durable outer-controller signals sampled at deterministic boundaries."""

    heartbeat: Callable[[], object]
    cancellation_reason: Callable[[], str | None]
    stop_dispatch_requested: Callable[[], bool]
    requeue_requested: Callable[[], bool]
    waiting_input_reason: Callable[[], str | None]


def _adopt_journaled_delivery_checkout(
    workspace: Path,
    task: NormalizedTask,
    state: RunState,
    *,
    generation: int,
) -> DeliveryCheckout:
    """Adopt a clean advanced checkout only when one pending fan-in journal binds it."""
    root = workspace.expanduser().resolve(strict=True)
    signed = task.delivery.mode == "signed_commits"
    repository = root / "delivery" / ("repository" if signed else "diff-repository")
    if not repository.is_dir() or repository.is_symlink():
        raise RuntimeErrorBase("advanced delivery checkout is unavailable for recovery")
    owner = f"{state.run_id}:generation-{generation}"
    git = ControllerGitOps(repository, root / "locks" / "git.lock", owner)
    inspection = git.inspect()
    expected_branch = task.delivery.branch if signed else None
    if inspection.branch != expected_branch or not inspection.clean:
        raise RuntimeErrorBase("advanced delivery checkout has invalid branch or dirty state")
    if inspection.head == state.integration_head:
        raise RuntimeErrorBase("delivery recovery requires an advanced checkout head")

    matches: list[str] = []
    items_root = root / "items"
    for item in state.items:
        if item.status is not WorkItemStatus.APPROVED:
            continue
        item_root = items_root / item.item_id
        receipt_path = item_root / "fan-in-receipt.json"
        intent_path = item_root / "fan-in-intent.json"
        for path, terminal in ((receipt_path, True), (intent_path, False)):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeErrorBase(f"cannot parse pending fan-in journal: {error}") from error
            if not isinstance(value, dict):
                raise RuntimeErrorBase("pending fan-in journal must be an object")
            expected_identity = (
                state.run_id,
                state.task_digest,
                item.item_id,
                state.integration_head,
            )
            observed_identity = (
                value.get("run_id"),
                value.get("task_digest"),
                value.get("item_id"),
                value.get("previous_head"),
            )
            if observed_identity != expected_identity:
                continue
            if terminal and value.get("final_head") != inspection.head:
                continue
            if not terminal and value.get("candidate_disposition") != "patch":
                continue
            matches.append(item.item_id)
            break
    if len(matches) != 1:
        raise RuntimeErrorBase(
            "advanced delivery checkout does not have one exact pending fan-in journal"
        )
    return DeliveryCheckout(repository, expected_branch, inspection.head, git)


@dataclass(frozen=True, slots=True)
class ProductionRuntimeFactory:
    """Workflow-compatible constructor carrying required production facts."""

    domain_inputs: RuntimeDomainInputs | None = None
    domain_input_resolver: RuntimeDomainInputResolver = resolve_runtime_domain_inputs
    backend_environment: tuple[tuple[str, str], ...] = ()
    collective_gate_adapter: CollectiveGateExecutionAdapter | None = None
    attempt_policy: AttemptPolicy | None = None
    credential_broker: CredentialBroker | None = None

    def __post_init__(self) -> None:
        if self.backend_environment:
            raise ValueError(
                "production runtime cannot receive backend credential values; "
                "configure a trusted CredentialBroker instead"
            )

    def __call__(
        self,
        *,
        workspace: Path,
        task: NormalizedTask,
        scheduler: Scheduler,
        generation: int,
        callbacks: RuntimeCallbacks,
    ) -> ControllerRuntime:
        """Build one fully configured runtime for ``workflow.run_controller``."""
        state = load_state(workspace / STATE_FILENAME)
        try:
            if task.delivery.mode == "signed_commits":
                if task.delivery.branch is None:
                    raise RuntimeErrorBase("signed-commit delivery requires an isolated branch")
                delivery = prepare_isolated_delivery_checkout(
                    task.repository.root,
                    workspace=workspace,
                    branch=task.delivery.branch,
                    base_commit=task.repository.base_commit,
                    expected_head=state.integration_head,
                    lock_path=workspace / "locks" / "git.lock",
                    owner=f"{state.run_id}:generation-{generation}",
                )
            else:
                delivery = prepare_diff_delivery_checkout(
                    task.repository.root,
                    workspace=workspace,
                    base_commit=task.repository.base_commit,
                    expected_head=state.integration_head,
                    lock_path=workspace / "locks" / "git.lock",
                    owner=f"{state.run_id}:generation-{generation}",
                )
        except (GitOpsError, OSError, ValueError) as error:
            try:
                delivery = _adopt_journaled_delivery_checkout(
                    workspace,
                    task,
                    state,
                    generation=generation,
                )
            except (GitOpsError, OSError, RuntimeError, ValueError) as recovery_error:
                raise RuntimeErrorBase(
                    "isolated delivery checkout is unavailable and cannot be recovered: "
                    f"{error}; recovery: {recovery_error}"
                ) from recovery_error
        runtime = ControllerRuntime(
            workspace=workspace,
            task=task,
            scheduler=scheduler,
            generation=generation,
            callbacks=callbacks,
            git=delivery.git,
            delivery=delivery,
            domain_inputs=self.domain_inputs,
            domain_input_resolver=self.domain_input_resolver,
            collective_gate_adapter=self.collective_gate_adapter,
            attempt_policy=self.attempt_policy,
            credential_broker=self.credential_broker,
        )
        runtime.validate_production_construction()
        return runtime


class ControllerRuntime:
    """Compose planning, dispatch, attempt reconciliation, and hierarchy closure."""

    def __init__(
        self,
        *,
        workspace: Path,
        task: NormalizedTask,
        scheduler: Scheduler,
        generation: int,
        callbacks: RuntimeCallbacks,
        execution_factory: AttemptExecutionFactory | None = None,
        smith_materializer: SmithMaterializer | None = None,
        domain_inputs: RuntimeDomainInputs | None = None,
        domain_input_resolver: RuntimeDomainInputResolver | None = None,
        backend_environment: tuple[tuple[str, str], ...] = (),
        worker_adapter: WorkerExecutionAdapter | None = None,
        collective_gate_adapter: CollectiveGateExecutionAdapter | None = None,
        attempt_policy: AttemptPolicy | None = None,
        git: ControllerGitOps | None = None,
        delivery: DeliveryCheckout | None = None,
        credential_broker: CredentialBroker | None = None,
    ) -> None:
        self._workspace = workspace.expanduser().resolve(strict=True)
        self._task = task
        self._scheduler = scheduler
        self._generation = generation
        self._callbacks = callbacks
        self._execution_factory = execution_factory
        self._domain_inputs = domain_inputs
        self._domain_input_resolver = domain_input_resolver
        if backend_environment:
            raise ValueError(
                "controller runtime cannot receive backend credential values; "
                "configure a trusted CredentialBroker instead"
            )
        self._collective_gate_adapter = collective_gate_adapter
        self._credential_broker = credential_broker
        self._attempt_policy = attempt_policy or AttemptPolicy(
            max_preempted_retries=task.execution.slurm.smith.retry_policy.preempted,
            max_node_fail_retries=task.execution.slurm.smith.retry_policy.node_failure,
        )
        self._unknown_observations: dict[str, int] = {}
        self._state_path = self._workspace / STATE_FILENAME
        self._git = git or ControllerGitOps(
            task.repository.root,
            self._workspace / "locks" / "git.lock",
            f"staircase-{self._workspace.name}",
        )
        if delivery is not None and delivery.git is not self._git:
            raise ValueError("delivery checkout and runtime Git authority must be identical")
        self._delivery = delivery
        self._worker_adapter_was_supplied = worker_adapter is not None
        if worker_adapter is not None:
            self._worker_adapter: WorkerExecutionAdapter | None = worker_adapter
        elif domain_inputs is not None:
            self._worker_adapter = GenericWorkerExecutionAdapter(
                workspace=self._workspace,
                task=task,
                git=self._git,
                credential_broker=self._credential_broker,
            )
        else:
            self._worker_adapter = None
        if smith_materializer is None:
            self._smith_materializer: SmithMaterializer = lambda actions: materialize_attempts(
                actions, git=self._git
            )
        else:
            self._smith_materializer = smith_materializer

    def validate_production_construction(self) -> None:
        """Fail before the outer controller loop if required facts are absent."""
        state = self._load_owned_state()
        if self._domain_inputs is None:
            if self._domain_input_resolver is None:
                raise RuntimeErrorBase(
                    "production runtime requires a deterministic domain-input resolver"
                )
            if not state.items:
                return
            plan, review = recover_approved_plan(self._workspace, self._task, state)
            if self._domain_input_resolver is not None:
                self._resolve_domain_inputs(state, plan, review)
        else:
            if state.items:
                plan, review = recover_approved_plan(self._workspace, self._task, state)
                self._validate_domain_input_binding(
                    state, self._domain_inputs, plan=plan, review=review
                )
            else:
                self._validate_domain_input_binding(state, self._domain_inputs)
        self._validate_resolved_production_inputs()

    def _validate_resolved_production_inputs(self) -> None:
        if self._domain_inputs is None:
            raise RuntimeErrorBase("production runtime inputs were not resolved")
        if self._domain_inputs.qa_required_items is None:
            raise RuntimeErrorBase("production runtime requires an explicit terminal QA policy")
        if {role for role, _paths in self._domain_inputs.role_evidence_paths} != {
            Role.CODER.value,
            Role.QA.value,
            Role.REVIEWER.value,
        }:
            raise RuntimeErrorBase(
                "production runtime requires exact role-scoped evidence path policies"
            )
        gates = self._gate_suite()
        gate_ids = {gate.gate_id for gate in gates}
        if set(self._domain_inputs.gate_execution_by_id) != gate_ids:
            raise RuntimeErrorBase(
                "production gate execution contracts must exactly cover the frozen gate suite"
            )
        if set(self._domain_inputs.gate_resource_by_id) != gate_ids:
            raise RuntimeErrorBase(
                "production gate resource subrequests must exactly cover the frozen gate suite"
            )
        self._install_local_collective_adapter(gates)
        for gate in gates:
            execution = self._gate_execution(gate.gate_id)
            self._gate_resource(gate.gate_id)
            if (
                self._requires_collective_gate(gate, execution)
                and self._collective_gate_adapter is None
            ):
                raise RuntimeErrorBase(
                    f"collective GateSpec {gate.gate_id!r} requires a trusted "
                    "rank-supervisor adapter"
                )

    def _install_local_collective_adapter(self, gates: Sequence[GateSpec]) -> None:
        """Install only the canonical local-four supervisor without guessing identity."""
        if self._collective_gate_adapter is not None or self._domain_inputs is None:
            return
        try:
            certifications = build_local_product_certifications(
                gates=gates,
                executions=self._domain_inputs.gate_execution_by_id,
                resources=self._domain_inputs.gate_resource_by_id,
                mapping=self._task.target.mapping,
            )
        except CollectiveAdapterError:
            return
        if certifications:
            self._collective_gate_adapter = TrustedCollectiveGateExecutionAdapter(
                workspace=self._workspace,
                task=self._task,
                git=self._git,
                certifications=certifications,
            )

    def tick(self) -> RuntimeTickResult:
        """Make one bounded, non-sleeping controller reconciliation step."""
        self._callbacks.heartbeat()
        state = self._load_owned_state()
        if state.terminal_status is not RunTerminalStatus.ACTIVE:
            return self._terminal(state)

        cancellation_reason = self._callbacks.cancellation_reason()
        if cancellation_reason is not None:
            return self._tick_cancellation(state, cancellation_reason)
        if self._callbacks.requeue_requested():
            return self._result(
                state,
                RuntimeDisposition.REQUEUE,
                RuntimeEvent.REQUEUE,
                "outer controller requested a generation-fenced requeue",
            )
        waiting_reason = self._callbacks.waiting_input_reason()
        if waiting_reason is not None:
            return self._result(
                state,
                RuntimeDisposition.WAITING_INPUT,
                RuntimeEvent.WAITING_INPUT,
                waiting_reason,
            )

        adoption = plan_attempt_adoption(state)
        if adoption.pending:
            target = adoption.pending[0]
            desired = state.record_attempt_adoption(
                attempt_id=target.attempt_id,
                attempt_generation=target.attempt_generation,
                status=target.status,
                submission_token=target.submission_token,
                job=target.job,
            )
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ATTEMPT_ADOPTED,
                "successor recorded one exact old-generation attempt adoption",
                attempt_id=target.attempt_id,
            )
        if not adoption.new_work_allowed:
            return self._blocked(
                state,
                "new work is gated until predecessor accounting is reconciled",
            )

        if not state.items:
            planning = PlanningEngine(
                workspace=self._workspace,
                task=self._task,
                scheduler=self._scheduler,
                generation=self._generation,
                config=self._planning_config(),
                credential_broker=self._credential_broker,
            ).tick()
            return self._planning_result(planning)

        try:
            plan, review = recover_approved_plan(self._workspace, self._task, state)
            if self._domain_input_resolver is not None:
                self._resolve_domain_inputs(state, plan, review)
            self._validate_admitted_state(state, plan)
            self._validate_gate_policy()
            self._validate_resource_envelope(state, plan)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as error:
            return self._blocked(state, f"admitted workflow validation failed: {error}")

        active_wait = self._reconcile_active_attempts(state, plan)
        if active_wait is not None and active_wait.event is not RuntimeEvent.NO_CHANGE:
            return active_wait

        typed_outcome = self._close_one_typed_attempt_outcome(state, plan)
        if typed_outcome is not None:
            return typed_outcome

        integration = self._complete_one_integration(state)
        if integration is not None:
            return integration

        hierarchy = self._advance_hierarchy_once(state, cancelling=False)
        if hierarchy is not None:
            return hierarchy

        ready = self._mark_one_ready(state, plan)
        if ready is not None:
            return ready

        if self._callbacks.stop_dispatch_requested():
            return self._result(
                state,
                RuntimeDisposition.WAIT,
                RuntimeEvent.STOP_DISPATCH,
                "dispatch is stopped; active attempts were reconciled before this boundary",
            )

        approved_wait: RuntimeTickResult | None = None
        approved = next(
            (
                item
                for item in sorted(state.items, key=lambda entry: entry.item_id)
                if item.status is WorkItemStatus.APPROVED
            ),
            None,
        )
        if approved is not None:
            approved_result = self._handle_approved(state, plan, review, approved)
            if approved_result.event is not RuntimeEvent.NO_CHANGE:
                return approved_result
            approved_wait = approved_result

        ready_item = next(
            (
                item
                for item in sorted(state.items, key=lambda entry: entry.item_id)
                if item.status is WorkItemStatus.READY
            ),
            None,
        )
        if ready_item is not None:
            return self._dispatch_ready(state, plan, review, ready_item)

        if active_wait is not None:
            return active_wait

        if approved_wait is not None:
            return approved_wait

        return self._blocked(
            state,
            "no deterministic transition is available for the admitted typed state",
        )

    def _reconcile_active_attempts(
        self,
        state: RunState,
        plan: PlanDraftOutcome,
    ) -> RuntimeTickResult | None:
        """Reconcile every active attempt until one authoritative action is available.

        Scheduler observations that leave an attempt unchanged do not globally
        serialize independent Smith siblings.  The tick still returns after the
        first materialization or persisted transition, preserving the
        one-authoritative-action boundary.
        """
        waiting: RuntimeTickResult | None = None
        for item in sorted(state.items, key=lambda entry: entry.item_id):
            proposal = _proposal(plan, item.item_id)
            for attempt in sorted(item.attempts, key=lambda entry: entry.sequence):
                if attempt.status not in _ACTIVE_ATTEMPT_STATUSES:
                    continue
                result = self._reconcile_attempt(state, proposal, item, attempt)
                if result.event is not RuntimeEvent.NO_CHANGE:
                    return result
                if waiting is None:
                    waiting = result
        return waiting

    def _handle_approved(
        self,
        state: RunState,
        plan: PlanDraftOutcome,
        review: PlanReviewOutcome,
        item: WorkItemRecord,
    ) -> RuntimeTickResult:
        proposal = _proposal(plan, item.item_id)
        if self._domain_inputs is None or self._domain_inputs.qa_required_items is None:
            return self._blocked(
                state,
                "terminal QA policy was not supplied by workflow construction",
                item_id=item.item_id,
            )
        qa_attempts = [attempt for attempt in item.attempts if attempt.kind is AttemptKind.QA]
        qa_required = (
            item.item_id in self._domain_inputs.qa_required_items
            or item.profile is DomainProfile.TUNER
        )
        if qa_required:
            if not qa_attempts:
                binding = self._candidate_binding(state, proposal)
                domain = plan_qa_action(
                    state,
                    proposal,
                    self._frozen_from_binding(binding),
                    self._gate_suite(),
                    gate_attempt_ids=self._gate_attempt_ids(item),
                    gate_result_digests=self._qa_gate_result_digests(item),
                    resource=self._role_resource(SlurmRole.QA),
                    workspace=self._workspace,
                    settings=self._agent_settings(),
                    evidence_paths=self._domain_inputs.evidence_paths_for(Role.QA),
                    candidate_evidence=self._candidate_evidence_context(proposal),
                )
                return self._persist_domain_plan(
                    state,
                    domain,
                    "persisted independent terminal QA job",
                )
            qa = qa_attempts[-1]
            if qa.status is not AttemptStatus.VALIDATED:
                return self._blocked(
                    state,
                    "terminal QA did not produce validated exact-candidate evidence",
                    item_id=item.item_id,
                    attempt_id=qa.attempt_id,
                )
            manifest = self._validated_manifest(state, proposal, qa)
            if manifest is None:
                return self._blocked(
                    state,
                    "terminal QA result cannot be reconstructed from its immutable mailbox",
                    item_id=item.item_id,
                    attempt_id=qa.attempt_id,
                )
            try:
                decoded = decode_worker_result(
                    manifest,
                    qa,
                    proposal,
                    candidate=self._candidate_binding(state, proposal).receipt,
                    qa_gate_results=self._qa_gate_result_digests(item),
                )
            except ResultContractError as error:
                return self._blocked(
                    state,
                    f"terminal QA result is invalid: {error}",
                    item_id=item.item_id,
                    attempt_id=qa.attempt_id,
                )
            if not isinstance(decoded, QaResult):
                return self._blocked(
                    state,
                    "terminal QA decoder returned another result kind",
                    item_id=item.item_id,
                    attempt_id=qa.attempt_id,
                )
            try:
                self._reload_trusted_gate_results(state, proposal, item)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                return self._blocked(
                    state,
                    f"terminal QA referenced invalid controller gate evidence: {error}",
                    item_id=item.item_id,
                    attempt_id=qa.attempt_id,
                )
            if qa.candidate_digest is None:
                pinned = replace(qa, candidate_digest=decoded.candidate_digest)
                desired = state.replace_item(item.replace_attempt(pinned))
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.ATTEMPT_UPDATED,
                    "persisted controller-validated terminal QA candidate linkage",
                    item_id=item.item_id,
                    attempt_id=qa.attempt_id,
                )
            if decoded.verdict is not QaVerdict.APPROVE:
                return self._close_decision(
                    state,
                    item,
                    qa,
                    rejected=decoded.verdict is QaVerdict.REJECT,
                    summary="terminal QA did not approve the exact frozen candidate",
                )
        if item.profile is DomainProfile.TUNER:
            try:
                finalized = self._finalize_tuning_campaign(state, proposal, item)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                return self._blocked(
                    state,
                    f"controller-owned tuning promotion failed closed: {error}",
                    item_id=item.item_id,
                )
            if finalized is not None:
                return finalized
        return self._materialize_or_record_fan_in(state, plan, review, item, proposal)

    def _materialize_or_record_fan_in(
        self,
        state: RunState,
        plan: PlanDraftOutcome,
        review: PlanReviewOutcome,
        item: WorkItemRecord,
        proposal: WorkItemProposal,
    ) -> RuntimeTickResult:
        placement = self._horizontal_smith_placement_guard(state, plan, item, proposal)
        if placement is not None:
            return placement
        predecessor = self._integration_predecessor(state, item)
        if predecessor is not None:
            return self._result(
                state,
                RuntimeDisposition.WAIT,
                RuntimeEvent.NO_CHANGE,
                "stable fan-in order is waiting for earlier integrable WorkItem "
                f"{predecessor.item_id!r} in {predecessor.status.value!r}; terminal "
                "non-integrated predecessors are skipped",
                item_id=item.item_id,
            )
        receipt_path = self._fan_in_receipt_path(item.item_id)
        binding = self._candidate_binding(state, proposal)
        if not receipt_path.exists():
            try:
                intent_path = self._fan_in_intent_path(item.item_id)
                intent = self._fan_in_intent_payload(
                    state,
                    plan,
                    item,
                    proposal,
                    binding,
                )
                created = _write_atomic_json_once(intent_path, intent)
                observed_intent = _load_exact_json(intent_path, set(intent))
                if observed_intent != intent:
                    raise RuntimeErrorBase(
                        "fan-in intent differs from the exact approved candidate"
                    )
                if created:
                    return self._result(
                        state,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.FAN_IN_INTENT_PERSISTED,
                        "persisted immutable fan-in intent before Git mutation",
                        item_id=item.item_id,
                    )
                inspection = self._git.inspect()
                if not inspection.clean:
                    raise RuntimeErrorBase("integration checkout is dirty before controller fan-in")
                if inspection.head != state.integration_head:
                    return self._recover_interrupted_fan_in(
                        state,
                        item,
                        proposal,
                        binding,
                        observed_head=inspection.head,
                    )
                if proposal.kind in CATALOG_ITEM_KINDS:
                    batch = plan_fan_in(
                        state,
                        plan,
                        review,
                        (
                            ApprovedSmithCandidate(
                                proposal,
                                (
                                    binding.snapshot
                                    if binding.receipt.disposition is CandidateDisposition.PATCH
                                    else None
                                ),
                                binding.index_delta,
                            ),
                        ),
                    )
                    catalog_root = (
                        self._git.repository / "tensorrt_llm" / "_torch" / "modeling_v2" / "catalog"
                    )
                    result = execute_fan_in(
                        batch,
                        git=self._git,
                        index_path=catalog_root / "index.yaml",
                        catalog_root=catalog_root,
                    )
                    integrated = next(
                        entry for entry in result.items if entry.item_id == item.item_id
                    )
                    if not integrated.integrated or integrated.integrated_commit is None:
                        return self._blocked(
                            state,
                            f"candidate fan-in conflict: {integrated.reason}",
                            item_id=item.item_id,
                        )
                    final_head = result.final_head
                    index_relative = "tensorrt_llm/_torch/modeling_v2/catalog/index.yaml"
                    expectations = (
                        []
                        if binding.receipt.disposition is CandidateDisposition.VERIFICATION
                        else [
                            IntegratedPatchExpectation(
                                binding.receipt.candidate_digest,
                                binding.receipt.changed_paths,
                            )
                        ]
                    )
                    if final_head != integrated.integrated_commit:
                        index_patch = self._git.render_patch(
                            base_commit=integrated.integrated_commit,
                            head_commit=final_head,
                        )
                        expectations.append(
                            IntegratedPatchExpectation(
                                hashlib.sha256(index_patch).hexdigest(),
                                (index_relative,),
                            )
                        )
                else:
                    if binding.receipt.disposition is CandidateDisposition.VERIFICATION:
                        final_head = inspection.head
                        expectations = []
                    else:
                        with self._git.transaction() as transaction:
                            inspection = self._git.inspect()
                            if not inspection.clean:
                                raise RuntimeErrorBase(
                                    "non-catalog fan-in requires a clean integration checkout"
                                )
                            final_head = transaction.integrate_commit(
                                candidate_commit=binding.snapshot.candidate_commit,
                                expected_head=inspection.head,
                            )
                        expectations = [
                            IntegratedPatchExpectation(
                                binding.receipt.candidate_digest,
                                binding.receipt.changed_paths,
                            )
                        ]
                verification = self._git.verify_linear_patch_sequence(
                    previous_head=state.integration_head,
                    observed_head=final_head,
                    expectations=expectations,
                )
                self._write_fan_in_receipt(
                    receipt_path,
                    state=state,
                    item=item,
                    binding=binding,
                    previous_head=state.integration_head,
                    final_head=final_head,
                    expectations=expectations,
                    final_tree_hash=verification.final_tree_hash,
                    sequence_evidence_sha256=verification.evidence_sha256,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                return self._blocked(
                    state,
                    f"controller-owned fan-in failed closed: {error}",
                    item_id=item.item_id,
                )
            return self._result(
                state,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.FAN_IN_MATERIALIZED,
                "controller materialized an immutable fan-in receipt",
                item_id=item.item_id,
            )

        try:
            receipt = self._load_fan_in_receipt(receipt_path, state, item, binding)
            self._validate_fan_in_ledger(state, item, receipt)
            if self._git.inspect().head != receipt["final_head"]:
                raise RuntimeErrorBase(
                    "integration checkout HEAD differs from the pending fan-in receipt"
                )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            return self._blocked(
                state,
                f"pending fan-in receipt failed validation: {error}",
                item_id=item.item_id,
            )
        record = next(
            (entry for entry in state.integration_history if entry.item_id == item.item_id),
            None,
        )
        if record is None:
            if item.profile is DomainProfile.TUNER:
                desired = state.replace_item(item.transition(WorkItemStatus.INTEGRATING))
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.ATTEMPT_UPDATED,
                    "persisted no-change Tuner completion intent",
                    item_id=item.item_id,
                )
            desired = state.record_integration(
                item_id=item.item_id,
                previous_commit=cast(str, receipt["previous_head"]),
                new_commit=cast(str, receipt["final_head"]),
                evidence_kind=IntegrationEvidenceKind(cast(str, receipt["evidence_kind"])),
                evidence_digest=cast(str, receipt["evidence_digest"]),
            )
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ATTEMPT_UPDATED,
                "persisted controller-verified integration ledger record",
                item_id=item.item_id,
            )
        desired = state.replace_item(item.transition(WorkItemStatus.INTEGRATING))
        self._save(state, desired)
        return self._result(
            desired,
            RuntimeDisposition.CONTINUE,
            RuntimeEvent.ATTEMPT_UPDATED,
            "persisted fan-in completion intent",
            item_id=item.item_id,
        )

    def _horizontal_smith_placement_guard(
        self,
        state: RunState,
        plan: PlanDraftOutcome,
        item: WorkItemRecord,
        proposal: WorkItemProposal,
    ) -> RuntimeTickResult | None:
        """Require exact distinct-node evidence before a Smith catalog fan-in."""
        smith = self._task.execution.slurm.smith
        if not smith.distinct_nodes_required or proposal.kind not in CATALOG_ITEM_KINDS:
            return None
        try:
            wave = horizontal_smith_wave_for_item(plan, item.item_id, smith=smith)
        except (PlacementError, PolicyViolation) as error:
            return self._blocked(
                state,
                f"horizontal Smith placement cohort is invalid: {error}",
                item_id=item.item_id,
            )

        for stage_kind in (
            AttemptKind.ROLE,
            AttemptKind.REVIEWER_ANALYSIS,
            AttemptKind.REVIEWER_RERUN,
        ):
            attempt_ids: list[str] = []
            for peer_id in wave.item_ids:
                try:
                    peer = state.item(peer_id)
                except ValueError as error:
                    return self._blocked(
                        state,
                        f"horizontal Smith placement cohort is absent from state: {error}",
                        item_id=item.item_id,
                    )
                stage_attempts = tuple(
                    attempt
                    for attempt in peer.attempts
                    if attempt.kind is stage_kind
                    and attempt.profile is DomainProfile.SMITH
                    and (
                        attempt.role is Role.CODER
                        if stage_kind is AttemptKind.ROLE
                        else attempt.role is Role.REVIEWER
                    )
                )
                preferred_attempt_id = (
                    peer.candidate_attempt_id
                    if stage_kind is AttemptKind.ROLE
                    else (
                        peer.reviewer_attempt_id
                        if stage_kind is AttemptKind.REVIEWER_RERUN
                        else None
                    )
                )
                selected = (
                    next(
                        (
                            attempt
                            for attempt in stage_attempts
                            if attempt.attempt_id == preferred_attempt_id
                        ),
                        None,
                    )
                    if preferred_attempt_id is not None
                    else (stage_attempts[-1] if stage_attempts else None)
                )
                if selected is None:
                    if peer.terminal:
                        continue
                    return self._result(
                        state,
                        RuntimeDisposition.WAIT,
                        RuntimeEvent.NO_CHANGE,
                        "horizontal Smith placement is waiting for peer "
                        f"{peer_id!r} to close role stage {stage_kind.value!r}",
                        item_id=item.item_id,
                    )
                if selected.job is None:
                    if peer.terminal or selected.status is AttemptStatus.VALIDATED:
                        return self._blocked(
                            state,
                            "terminal horizontal Smith role lacks an exact Slurm job: "
                            f"{selected.attempt_id!r}",
                            item_id=item.item_id,
                            attempt_id=selected.attempt_id,
                        )
                    return self._result(
                        state,
                        RuntimeDisposition.WAIT,
                        RuntimeEvent.NO_CHANGE,
                        "horizontal Smith placement is waiting for an exact Slurm job for "
                        f"{selected.attempt_id!r}",
                        item_id=item.item_id,
                        attempt_id=selected.attempt_id,
                    )
                placement_path = self._attempt_dir(selected) / PLACEMENT_RECEIPT_FILENAME
                if not placement_path.exists() and not placement_path.is_symlink():
                    if peer.terminal or selected.status is AttemptStatus.VALIDATED:
                        return self._blocked(
                            state,
                            "terminal horizontal Smith role lacks its placement receipt: "
                            f"{selected.attempt_id!r}",
                            item_id=item.item_id,
                            attempt_id=selected.attempt_id,
                        )
                    return self._result(
                        state,
                        RuntimeDisposition.WAIT,
                        RuntimeEvent.NO_CHANGE,
                        "horizontal Smith placement is waiting for the active peer receipt for "
                        f"{selected.attempt_id!r}",
                        item_id=item.item_id,
                        attempt_id=selected.attempt_id,
                    )
                attempt_ids.append(selected.attempt_id)

            try:
                validate_smith_role_wave_placement(
                    state,
                    attempt_ids,
                    attempt_kind=stage_kind,
                    smith=smith,
                    workspace=self._workspace,
                    scheduler=self._scheduler,
                )
            except PlacementError as error:
                return self._blocked(
                    state,
                    "horizontal Smith placement proof failed closed for role stage "
                    f"{stage_kind.value!r}: {error}",
                    item_id=item.item_id,
                )
        return None

    def _recover_interrupted_fan_in(
        self,
        state: RunState,
        item: WorkItemRecord,
        proposal: WorkItemProposal,
        binding: CandidateBinding,
        *,
        observed_head: str,
    ) -> RuntimeTickResult:
        """Prove or finish one journaled fan-in after a pre-receipt crash."""
        if binding.receipt.disposition is not CandidateDisposition.PATCH:
            raise RuntimeErrorBase("a no-change fan-in unexpectedly advanced the checkout")
        candidate_expectation = IntegratedPatchExpectation(
            binding.receipt.candidate_digest,
            binding.receipt.changed_paths,
        )
        expectations = [candidate_expectation]
        final_head = observed_head
        candidate_only = False
        try:
            self._git.verify_linear_patch_sequence(
                previous_head=state.integration_head,
                observed_head=observed_head,
                expectations=expectations,
            )
            candidate_only = True
        except GitOpsError:
            if proposal.kind not in CATALOG_ITEM_KINDS:
                raise

        if proposal.kind in CATALOG_ITEM_KINDS:
            delta = binding.index_delta
            if delta is None:
                raise RuntimeErrorBase("catalog fan-in recovery requires its frozen IndexDelta")
            catalog_root = (
                self._git.repository / "tensorrt_llm" / "_torch" / "modeling_v2" / "catalog"
            )
            index_path = catalog_root / "index.yaml"
            index_relative = index_path.relative_to(self._git.repository).as_posix()
            if candidate_only:
                with self._git.transaction() as transaction:
                    update = update_catalog_index(
                        index_path,
                        catalog_root,
                        (delta,),
                        controller_lock=transaction.lock,
                    )
                    if update.changed:
                        transaction.stage_paths(self._git.repository, (index_relative,))
                        final_head = transaction.commit_signed_off(
                            self._git.repository,
                            message="agent-flow: integrate reviewed Smith catalog index deltas",
                            expected_paths=(index_relative,),
                        )
            else:
                recovery_repository = (
                    self._workspace / "recovery" / "fan-in" / item.item_id / state.integration_head
                )
                if not recovery_repository.exists():
                    with self._git.transaction() as transaction:
                        transaction.create_detached_worktree(
                            recovery_repository,
                            base_commit=state.integration_head,
                        )
                recovery_inspection = self._git.inspect(recovery_repository)
                if (
                    not recovery_inspection.clean
                    or recovery_inspection.branch is not None
                    or recovery_inspection.head != state.integration_head
                ):
                    raise RuntimeErrorBase(
                        "fan-in recovery snapshot differs from the journaled previous head"
                    )
                recovery_catalog = (
                    recovery_repository / "tensorrt_llm" / "_torch" / "modeling_v2" / "catalog"
                )
                previous_index = load_catalog_index(
                    recovery_catalog / "index.yaml",
                    recovery_catalog,
                )
                expected_update = render_index_deltas(previous_index, (delta,), catalog_root)
                if index_path.read_text(encoding="utf-8") != expected_update.text:
                    raise RuntimeErrorBase(
                        "advanced fan-in index differs from the deterministic IndexDelta replay"
                    )

            if final_head != observed_head or not candidate_only:
                index_patch = self._git.render_patch(
                    base_commit=f"{final_head}^",
                    head_commit=final_head,
                )
                expectations.append(
                    IntegratedPatchExpectation(
                        hashlib.sha256(index_patch).hexdigest(),
                        (index_relative,),
                    )
                )

        verification = self._git.verify_linear_patch_sequence(
            previous_head=state.integration_head,
            observed_head=final_head,
            expectations=expectations,
        )
        self._write_fan_in_receipt(
            self._fan_in_receipt_path(item.item_id),
            state=state,
            item=item,
            binding=binding,
            previous_head=state.integration_head,
            final_head=final_head,
            expectations=expectations,
            final_tree_hash=verification.final_tree_hash,
            sequence_evidence_sha256=verification.evidence_sha256,
        )
        return self._result(
            state,
            RuntimeDisposition.CONTINUE,
            RuntimeEvent.FAN_IN_MATERIALIZED,
            "recovered exact journaled fan-in and persisted its immutable receipt",
            item_id=item.item_id,
        )

    def _complete_one_integration(self, state: RunState) -> RuntimeTickResult | None:
        for item in sorted(state.items, key=lambda entry: entry.item_id):
            if item.status is not WorkItemStatus.INTEGRATING:
                continue
            predecessor = self._integration_predecessor(state, item)
            if predecessor is not None:
                return self._blocked(
                    state,
                    "cannot complete fan-in before earlier integrable WorkItem "
                    f"{predecessor.item_id!r}",
                    item_id=item.item_id,
                )
            proposal_state = self._load_owned_state()
            plan, _review = recover_approved_plan(self._workspace, self._task, proposal_state)
            proposal = _proposal(plan, item.item_id)
            binding = self._candidate_binding(state, proposal)
            receipt_path = self._fan_in_receipt_path(item.item_id)
            try:
                receipt = self._load_fan_in_receipt(receipt_path, state, item, binding)
                self._validate_fan_in_ledger(state, item, receipt)
                if self._git.inspect().head != receipt["final_head"]:
                    raise RuntimeErrorBase(
                        "integration checkout HEAD differs from its fan-in receipt"
                    )
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                return self._blocked(
                    state,
                    f"cannot complete fan-in: {error}",
                    item_id=item.item_id,
                )
            desired = state.replace_item(item.transition(WorkItemStatus.INTEGRATED))
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ITEM_INTEGRATED,
                "persisted controller-verified fan-in completion",
                item_id=item.item_id,
            )
        return None

    @staticmethod
    def _integration_predecessor(
        state: RunState,
        item: WorkItemRecord,
    ) -> WorkItemRecord | None:
        """Return the first earlier, currently integrable nonterminal item.

        Item IDs provide the stable global tie-break for independent siblings.
        An earlier item whose dependencies are not integrated cannot block the
        dependency that makes it runnable. Integrated and terminal failed,
        rejected, or cancelled items never block later fan-in.
        """
        for candidate in sorted(state.items, key=lambda entry: entry.item_id):
            if candidate.item_id >= item.item_id:
                break
            if candidate.terminal:
                continue
            if any(
                state.item(dependency).status is not WorkItemStatus.INTEGRATED
                for dependency in candidate.dependencies
            ):
                continue
            return candidate
        return None

    def _tuning_artifact_root(self, item_id: str) -> Path:
        return self._workspace / "items" / item_id / "tuning"

    def _candidate_evidence_context(
        self,
        proposal: WorkItemProposal,
    ) -> dict[str, object] | None:
        """Return exact immutable tuning envelopes for independent reviewers."""
        if proposal.kind is not PolicyWorkItemKind.TUNE_HYPOTHESIS:
            return None
        root = self._tuning_artifact_root(proposal.item_id)
        artifacts: dict[str, object] = {}
        for filename in (
            BASELINE_FILENAME,
            CANDIDATE_FILENAME,
            EVALUATION_FILENAME,
            CAMPAIGN_FILENAME,
        ):
            artifacts[filename] = _load_exact_json(
                root / filename,
                {"schema_version", "kind", "digest", "payload"},
            )
        return {"kind": "tuning_measurement", "artifacts": artifacts}

    def _fan_in_intent_path(self, item_id: str) -> Path:
        return self._workspace / "items" / item_id / "fan-in-intent.json"

    def _fan_in_receipt_path(self, item_id: str) -> Path:
        return self._workspace / "items" / item_id / "fan-in-receipt.json"

    @staticmethod
    def _fan_in_intent_payload(
        state: RunState,
        plan: PlanDraftOutcome,
        item: WorkItemRecord,
        proposal: WorkItemProposal,
        binding: CandidateBinding,
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": state.run_id,
            "task_digest": state.task_digest,
            "plan_digest": plan.digest,
            "generation": state.generation,
            "item_id": item.item_id,
            "candidate_attempt_id": binding.receipt.attempt_id,
            "candidate_commit": binding.receipt.candidate_commit,
            "candidate_digest": binding.receipt.candidate_digest,
            "candidate_receipt_sha256": binding.receipt.receipt_sha256,
            "candidate_disposition": binding.receipt.disposition.value,
            "previous_head": state.integration_head,
            "allowed_paths": list(proposal.allowed_paths),
        }

    @staticmethod
    def _write_fan_in_receipt(
        path: Path,
        *,
        state: RunState,
        item: WorkItemRecord,
        binding: CandidateBinding,
        previous_head: str,
        final_head: str,
        expectations: Sequence[IntegratedPatchExpectation],
        final_tree_hash: str,
        sequence_evidence_sha256: str,
    ) -> None:
        payload = {
            "schema_version": 1,
            "run_id": state.run_id,
            "task_digest": state.task_digest,
            "item_id": item.item_id,
            "candidate_attempt_id": binding.receipt.attempt_id,
            "candidate_digest": binding.receipt.candidate_digest,
            "previous_head": previous_head,
            "final_head": final_head,
            "evidence_kind": (
                IntegrationEvidenceKind.VERIFICATION.value
                if binding.receipt.disposition is CandidateDisposition.VERIFICATION
                else IntegrationEvidenceKind.CANDIDATE.value
            ),
            "evidence_digest": binding.receipt.candidate_digest,
            "patch_sequence": [
                {
                    "patch_sha256": expectation.patch_sha256,
                    "changed_paths": list(expectation.changed_paths),
                }
                for expectation in expectations
            ],
            "final_tree_hash": final_tree_hash,
            "sequence_evidence_sha256": sequence_evidence_sha256,
        }
        _write_atomic_json_once(path, payload)
        if _load_exact_json(path, set(payload)) != payload:
            raise RuntimeErrorBase("fan-in receipt differs from the exact integrated candidate")

    def _load_fan_in_receipt(
        self,
        path: Path,
        state: RunState,
        item: WorkItemRecord,
        binding: CandidateBinding,
    ) -> dict[str, object]:
        if path.is_symlink() or not path.is_file():
            raise RuntimeErrorBase("fan-in receipt is missing or not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
        expected_keys = {
            "schema_version",
            "run_id",
            "task_digest",
            "item_id",
            "candidate_attempt_id",
            "candidate_digest",
            "previous_head",
            "final_head",
            "evidence_kind",
            "evidence_digest",
            "patch_sequence",
            "final_tree_hash",
            "sequence_evidence_sha256",
        }
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise RuntimeErrorBase("fan-in receipt has an invalid schema")
        expected = (
            1,
            state.run_id,
            state.task_digest,
            item.item_id,
            binding.receipt.attempt_id,
            binding.receipt.candidate_digest,
        )
        actual = tuple(
            value[key]
            for key in (
                "schema_version",
                "run_id",
                "task_digest",
                "item_id",
                "candidate_attempt_id",
                "candidate_digest",
            )
        )
        if actual != expected:
            raise RuntimeErrorBase("fan-in receipt identity differs from authoritative state")
        for key in ("previous_head", "final_head"):
            commit = value[key]
            if (
                not isinstance(commit, str)
                or len(commit) != 40
                or any(character not in "0123456789abcdef" for character in commit)
            ):
                raise RuntimeErrorBase(f"fan-in receipt {key} is not a Git commit")
        try:
            evidence_kind = IntegrationEvidenceKind(cast(str, value["evidence_kind"]))
        except (TypeError, ValueError) as error:
            raise RuntimeErrorBase("fan-in receipt has an invalid evidence kind") from error
        expected_kind = (
            IntegrationEvidenceKind.VERIFICATION
            if binding.receipt.disposition is CandidateDisposition.VERIFICATION
            else IntegrationEvidenceKind.CANDIDATE
        )
        if evidence_kind is not expected_kind:
            raise RuntimeErrorBase("fan-in receipt evidence kind differs from its candidate")
        if value["evidence_digest"] != binding.receipt.candidate_digest:
            raise RuntimeErrorBase("fan-in receipt evidence digest differs from its candidate")
        raw_sequence = value["patch_sequence"]
        if not isinstance(raw_sequence, list):
            raise RuntimeErrorBase("fan-in receipt patch sequence must be a list")
        expectations: list[IntegratedPatchExpectation] = []
        for entry in raw_sequence:
            if not isinstance(entry, dict) or set(entry) != {
                "patch_sha256",
                "changed_paths",
            }:
                raise RuntimeErrorBase("fan-in receipt patch expectation has an invalid schema")
            patch_digest = entry["patch_sha256"]
            changed_paths = entry["changed_paths"]
            if not isinstance(patch_digest, str) or not isinstance(changed_paths, list):
                raise RuntimeErrorBase("fan-in receipt patch expectation has invalid types")
            if any(not isinstance(changed_path, str) for changed_path in changed_paths):
                raise RuntimeErrorBase("fan-in receipt changed paths must be strings")
            expectations.append(IntegratedPatchExpectation(patch_digest, tuple(changed_paths)))
        if binding.receipt.disposition is CandidateDisposition.VERIFICATION:
            if expectations:
                raise RuntimeErrorBase("verification fan-in cannot contain patch expectations")
        elif not expectations or expectations[0] != IntegratedPatchExpectation(
            binding.receipt.candidate_digest,
            binding.receipt.changed_paths,
        ):
            raise RuntimeErrorBase("fan-in sequence does not begin with the frozen candidate")
        verification = self._git.verify_linear_patch_sequence(
            previous_head=cast(str, value["previous_head"]),
            observed_head=cast(str, value["final_head"]),
            expectations=expectations,
        )
        if (
            value["final_tree_hash"] != verification.final_tree_hash
            or value["sequence_evidence_sha256"] != verification.evidence_sha256
        ):
            raise RuntimeErrorBase("fan-in sequence evidence differs from the exact Git history")
        return value

    @staticmethod
    def _validate_fan_in_ledger(
        state: RunState,
        item: WorkItemRecord,
        receipt: Mapping[str, object],
    ) -> None:
        record = next(
            (entry for entry in state.integration_history if entry.item_id == item.item_id),
            None,
        )
        if record is None:
            if receipt["previous_head"] != state.integration_head:
                raise RuntimeErrorBase("fan-in receipt does not extend the persisted head")
            return
        expected = (
            record.previous_commit,
            record.new_commit,
            record.evidence_kind.value,
            record.evidence_digest,
        )
        actual = (
            receipt["previous_head"],
            receipt["final_head"],
            receipt["evidence_kind"],
            receipt["evidence_digest"],
        )
        if actual != expected:
            raise RuntimeErrorBase("fan-in receipt differs from its persisted integration record")

    def _load_owned_state(self) -> RunState:
        state = load_state(self._state_path)
        if state.task_digest != self._task.digest:
            raise RuntimeErrorBase("normalized task digest differs from authoritative state")
        if state.generation != self._generation:
            raise RuntimeErrorBase(
                f"controller generation fence failed: state={state.generation}, "
                f"writer={self._generation}"
            )
        return state

    def _reconcile_attempt(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        item: WorkItemRecord,
        attempt: AttemptRecord,
    ) -> RuntimeTickResult:
        action: MaterializableAction | None = None
        try:
            adapter = self._execution_adapter_for_attempt(item, attempt)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            return self._blocked(
                state,
                f"worker execution adapter selection is blocked: {error}",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )
        if adapter is not None:
            try:
                input_path = self._attempt_dir(attempt) / INPUT_FILENAME
                action = (
                    self._persisted_attempt_action(state, proposal, item, attempt)
                    if input_path.exists()
                    else self._reconstruct_attempt_action(state, proposal, item, attempt)
                )
                if attempt.status is AttemptStatus.PREPARED and not input_path.exists():
                    self._callbacks.heartbeat()
                    adapter.materialize(action)
                    return self._result(
                        state,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.WORKER_ACTION_MATERIALIZED,
                        "materialized one persisted generic worker action",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                execution = adapter.execution(state, action)
            except (ExecutionAdapterError, KeyError, OSError, TypeError, ValueError) as error:
                return self._blocked(
                    state,
                    f"generic worker execution is blocked: {error}",
                    item_id=item.item_id,
                    attempt_id=attempt.attempt_id,
                )
        elif self._execution_factory is None:
            return self._blocked(
                state,
                "attempt execution is blocked: no generic worker execution adapter is configured",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )
        else:
            if (
                attempt.status is AttemptStatus.PREPARED
                and attempt.role is Role.CODER
                and attempt.profile is DomainProfile.SMITH
            ):
                smith_action = self._reconstruct_coder_action(state, proposal, attempt)
                self._callbacks.heartbeat()
                self._smith_materializer((smith_action,))
            execution = self._execution_factory(state, proposal, attempt)
        result = tick_attempt(
            attempt,
            scheduler=self._scheduler,
            execution=execution,
            policy=self._attempt_policy,
            unknown_observations=self._unknown_observations.get(attempt.attempt_id, 0),
        )
        self._unknown_observations[attempt.attempt_id] = result.unknown_observations
        if result.attempt == attempt:
            return self._result(
                state,
                RuntimeDisposition.WAIT,
                RuntimeEvent.NO_CHANGE,
                f"attempt {attempt.attempt_id} is waiting for scheduler/accounting evidence",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )
        desired_item = item.replace_attempt(result.attempt)
        desired = state.replace_item(desired_item)
        self._save(state, desired)
        disposition = (
            RuntimeDisposition.WAIT
            if result.action is AttemptAction.WAITING_FOR_ACCOUNTING
            else RuntimeDisposition.CONTINUE
        )
        return self._result(
            desired,
            disposition,
            RuntimeEvent.ATTEMPT_UPDATED,
            f"attempt reconciled with action {result.action.value}",
            item_id=item.item_id,
            attempt_id=attempt.attempt_id,
        )

    def _close_one_typed_attempt_outcome(
        self, state: RunState, plan: PlanDraftOutcome
    ) -> RuntimeTickResult | None:
        for item in sorted(state.items, key=lambda entry: entry.item_id):
            if item.terminal or not item.attempts:
                continue
            latest = item.attempts[-1]
            if not latest.terminal:
                continue
            revoked = self._revoke_terminal_attempt_credentials(state, item, latest)
            if revoked is not None:
                return revoked
            infrastructure_retry = self._retry_terminal_infrastructure_attempt(state, item, latest)
            if infrastructure_retry is not None:
                return infrastructure_retry
            if item.status in {
                WorkItemStatus.APPROVED,
                WorkItemStatus.INTEGRATING,
            } and latest.kind in {
                AttemptKind.REVIEWER_RERUN,
                AttemptKind.QA,
            }:
                continue
            if latest.status is not AttemptStatus.VALIDATED:
                status = _failed_item_status(item.status)
                desired_item = item.transition(
                    status,
                    terminal_reason=latest.terminal_reason
                    or "attempt failed without typed evidence",
                )
                desired = state.replace_item(desired_item)
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.ATTEMPT_UPDATED,
                    "typed terminal attempt failure blocked its WorkItem",
                    item_id=item.item_id,
                    attempt_id=latest.attempt_id,
                )
            manifest = self._validated_manifest(state, _proposal(plan, item.item_id), latest)
            if manifest is None:
                return self._blocked(
                    state,
                    "validated attempt cannot be closed because its typed result manifest "
                    "cannot be reconstructed by the configured execution adapter",
                    item_id=item.item_id,
                    attempt_id=latest.attempt_id,
                )
            if latest.kind in {
                AttemptKind.DETERMINISTIC_GATE,
                AttemptKind.REVIEWER_ANALYSIS,
                AttemptKind.REVIEWER_RERUN,
                AttemptKind.QA,
            } and manifest.status in {
                WorkerResultStatus.SUCCEEDED,
                WorkerResultStatus.REJECTED,
                WorkerResultStatus.BLOCKED,
            }:
                return self._close_successful_attempt(state, plan, item, latest, manifest)
            if manifest.status is WorkerResultStatus.BLOCKED:
                status = _failed_item_status(item.status)
            elif manifest.status is WorkerResultStatus.REJECTED:
                status = (
                    WorkItemStatus.REJECTED
                    if item.status
                    in {
                        WorkItemStatus.CODING,
                        WorkItemStatus.CANDIDATE_READY,
                        WorkItemStatus.REVIEWING,
                    }
                    else _failed_item_status(item.status)
                )
            elif manifest.status is WorkerResultStatus.RESOURCE_ESCALATION:
                try:
                    request = decode_worker_result(
                        manifest,
                        latest,
                        _proposal(plan, item.item_id),
                        allowed_resource_classes=tuple(
                            resource.name
                            for resource in self._task.execution.slurm.smith.resource_classes
                        ),
                    )
                    escalation = plan_resource_escalation_attempt(
                        state,
                        _proposal(plan, item.item_id),
                        request,
                        smith=self._task.execution.slurm.smith,
                        workspace=self._workspace,
                    )
                except (KeyError, ResultContractError, TypeError, ValueError) as error:
                    return self._blocked(
                        state,
                        f"invalid typed resource escalation: {error}",
                        item_id=item.item_id,
                        attempt_id=latest.attempt_id,
                    )
                if len(escalation.actions) != 1 or escalation.state.revision != state.revision + 1:
                    return self._blocked(
                        state,
                        "resource escalation did not produce exactly one successor attempt",
                        item_id=item.item_id,
                        attempt_id=latest.attempt_id,
                    )
                action = escalation.actions[0]
                self._save(state, escalation.state)
                return self._result(
                    escalation.state,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.ACTION_PERSISTED,
                    "persisted one controller-approved resource-escalation successor",
                    item_id=item.item_id,
                    attempt_id=action.attempt.attempt_id,
                )
            else:
                return self._close_successful_attempt(state, plan, item, latest, manifest)
            desired_item = item.transition(status, terminal_reason=manifest.summary)
            desired = state.replace_item(desired_item)
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ATTEMPT_UPDATED,
                f"typed worker result closed WorkItem as {status.value}",
                item_id=item.item_id,
                attempt_id=latest.attempt_id,
            )
        return None

    def _retry_terminal_infrastructure_attempt(
        self,
        state: RunState,
        item: WorkItemRecord,
        attempt: AttemptRecord,
    ) -> RuntimeTickResult | None:
        eligible = attempt.status is AttemptStatus.PREEMPTED or (
            attempt.status is AttemptStatus.RETRYABLE_FAILED
            and attempt.scheduler_state == JobStatus.NODE_FAIL.value
            and attempt.result_digest is None
        )
        if not eligible:
            return None
        sequence = len(item.attempts) + 1
        suffix = f".{sequence:04d}.{attempt.kind.value}"
        retry_id = f"{item.item_id}{suffix}"
        if len(retry_id) > 128:
            digest = hashlib.sha256(item.item_id.encode("utf-8")).hexdigest()[:16]
            prefix_length = 128 - len(suffix) - len(digest) - 1
            retry_id = f"{item.item_id[:prefix_length]}.{digest}{suffix}"
        try:
            retry = plan_infrastructure_retry(
                item,
                attempt.attempt_id,
                new_attempt_id=retry_id,
                generation=self._generation,
                policy=self._attempt_policy,
            )
        except RetryLimitExceeded:
            return None
        except AttemptEngineError as error:
            return self._blocked(
                state,
                f"infrastructure retry failed closed: {error}",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )
        desired_item = item if retry.already_appended else item.add_attempt(retry.attempt)
        if desired_item == item:
            return None
        desired = state.replace_item(desired_item)
        self._save(state, desired)
        return self._result(
            desired,
            RuntimeDisposition.CONTINUE,
            RuntimeEvent.ACTION_PERSISTED,
            f"persisted one bounded {retry.category.value} infrastructure retry",
            item_id=item.item_id,
            attempt_id=retry.attempt.attempt_id,
        )

    def _revoke_terminal_attempt_credentials(
        self,
        state: RunState,
        item: WorkItemRecord,
        attempt: AttemptRecord,
    ) -> RuntimeTickResult | None:
        if attempt.role not in {Role.CODER, Role.REVIEWER, Role.QA}:
            return None
        if self._worker_adapter is None:
            return None
        try:
            created = self._worker_adapter.revoke(state, attempt)
        except (ExecutionAdapterError, OSError, TypeError, ValueError) as error:
            return self._blocked(
                state,
                f"terminal credential revocation failed closed: {error}",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )
        if not created:
            return None
        return self._result(
            state,
            RuntimeDisposition.CONTINUE,
            RuntimeEvent.CREDENTIALS_REVOKED,
            "revoked exact terminal-attempt credential provision",
            item_id=item.item_id,
            attempt_id=attempt.attempt_id,
        )

    def _close_successful_attempt(
        self,
        state: RunState,
        plan: PlanDraftOutcome,
        item: WorkItemRecord,
        attempt: AttemptRecord,
        manifest: WorkerResultManifest,
    ) -> RuntimeTickResult:
        proposal = _proposal(plan, item.item_id)
        try:
            if attempt.role is Role.CODER and attempt.profile is DomainProfile.TUNER:
                return self._close_tuner_attempt(
                    state,
                    proposal,
                    item,
                    attempt,
                    manifest,
                )
            if attempt.role is Role.CODER:
                receipt_path = self._attempt_dir(attempt) / CANDIDATE_RECEIPT_FILENAME
                worktree = self._workspace / "candidates" / item.item_id / attempt.attempt_id
                if not receipt_path.exists():
                    bind_coder_candidate(
                        manifest,
                        attempt,
                        proposal,
                        worktree=worktree,
                        base_commit=self._coder_base_commit(state, proposal),
                        attempt_dir=self._attempt_dir(attempt),
                        git=self._git,
                    )
                    return self._result(
                        state,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.CANDIDATE_MATERIALIZED,
                        "controller bound the Coder result to an immutable candidate receipt",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                worktree = (
                    self._workspace / "controller-candidates" / item.item_id / attempt.attempt_id
                )
                binding = reconstruct_candidate(
                    receipt_path,
                    worktree=worktree,
                    proposal=proposal,
                    git=self._git,
                )
                if attempt.candidate_digest is None:
                    bound_attempt = replace(
                        attempt,
                        candidate_digest=binding.receipt.candidate_digest,
                    )
                    desired = state.replace_item(item.replace_attempt(bound_attempt))
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.ATTEMPT_UPDATED,
                        "persisted the controller-bound candidate digest",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                if attempt.candidate_digest != binding.receipt.candidate_digest:
                    raise ResultContractError(
                        "persisted Coder digest differs from the candidate receipt"
                    )
                return self._plan_gate(state, proposal, binding, gate_index=0)

            binding = self._candidate_binding(state, proposal)
            candidate = binding.receipt
            if attempt.kind is AttemptKind.DETERMINISTIC_GATE:
                gates = self._gate_suite()
                gate_index = logical_attempt_ordinal(item, attempt.attempt_id) - 1
                if gate_index >= len(gates):
                    raise ResultContractError("gate attempt exceeds the frozen gate suite")
                decoded = decode_worker_result(
                    manifest,
                    attempt,
                    proposal,
                    candidate=candidate,
                    gate_spec=gates[gate_index],
                )
                if not isinstance(decoded, GateResult):
                    raise ResultContractError("gate decoder returned another result kind")
                if attempt.candidate_digest is None:
                    pinned_attempt = replace(
                        attempt,
                        candidate_digest=decoded.candidate_digest,
                    )
                    desired = state.replace_item(item.replace_attempt(pinned_attempt))
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.ATTEMPT_UPDATED,
                        "persisted controller-validated gate candidate linkage",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                if attempt.candidate_digest != decoded.candidate_digest:
                    raise ResultContractError(
                        "persisted gate candidate digest differs from the typed receipt"
                    )
                if not decoded.receipt.passed:
                    desired = state.replace_item(
                        item.transition(
                            WorkItemStatus.REJECTED,
                            terminal_reason="deterministic hard gate rejected the candidate",
                        )
                    )
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.ATTEMPT_UPDATED,
                        "deterministic gate rejected the frozen candidate",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                if gate_index + 1 != len(gates):
                    return self._plan_gate(
                        state,
                        proposal,
                        binding,
                        gate_index=gate_index + 1,
                    )
                if item.status is WorkItemStatus.CODING:
                    desired = state.replace_item(item.transition(WorkItemStatus.CANDIDATE_READY))
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.ATTEMPT_UPDATED,
                        "all ordered deterministic gates validated the frozen candidate",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                return self._plan_reviewer_analysis(state, proposal, binding)

            if attempt.kind is AttemptKind.REVIEWER_ANALYSIS:
                decoded = decode_worker_result(
                    manifest,
                    attempt,
                    proposal,
                    candidate=candidate,
                )
                if not isinstance(decoded, ReviewerResult):
                    raise ResultContractError("Reviewer decoder returned another result kind")
                if attempt.candidate_digest is None:
                    pinned_attempt = replace(
                        attempt,
                        candidate_digest=decoded.candidate_digest,
                    )
                    desired = state.replace_item(item.replace_attempt(pinned_attempt))
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.ATTEMPT_UPDATED,
                        "persisted controller-validated Reviewer analysis candidate linkage",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                if decoded.verdict is ReviewerVerdict.APPROVE:
                    return self._plan_reviewer_rerun(
                        state,
                        proposal,
                        binding,
                        attempt.attempt_id,
                    )
                return self._close_decision(
                    state,
                    item,
                    attempt,
                    rejected=decoded.verdict is ReviewerVerdict.REJECT,
                    summary="Reviewer analysis did not approve the frozen candidate",
                )

            if attempt.kind is AttemptKind.REVIEWER_RERUN:
                decoded = decode_worker_result(
                    manifest,
                    attempt,
                    proposal,
                    candidate=candidate,
                )
                if not isinstance(decoded, ReviewerResult):
                    raise ResultContractError("Reviewer decoder returned another result kind")
                if attempt.candidate_digest is None:
                    pinned_attempt = replace(
                        attempt,
                        candidate_digest=decoded.candidate_digest,
                    )
                    desired = state.replace_item(item.replace_attempt(pinned_attempt))
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.ATTEMPT_UPDATED,
                        "persisted controller-validated Reviewer rerun candidate linkage",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                if not decoded.final_approval:
                    return self._close_decision(
                        state,
                        item,
                        attempt,
                        rejected=decoded.verdict is ReviewerVerdict.REJECT,
                        summary="fresh Reviewer rerun did not approve the frozen candidate",
                    )
                desired_item = item.transition(
                    WorkItemStatus.APPROVED,
                    reviewer_attempt_id=attempt.attempt_id,
                )
                desired = state.replace_item(desired_item)
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.ATTEMPT_UPDATED,
                    "fresh Reviewer rerun approved the exact frozen candidate",
                    item_id=item.item_id,
                    attempt_id=attempt.attempt_id,
                )

            if attempt.kind is AttemptKind.QA:
                decoded = decode_worker_result(
                    manifest,
                    attempt,
                    proposal,
                    candidate=candidate,
                    qa_gate_results=self._qa_gate_result_digests(item),
                )
                if not isinstance(decoded, QaResult):
                    raise ResultContractError("QA decoder returned another result kind")
                self._reload_trusted_gate_results(state, proposal, item)
                if attempt.candidate_digest is None:
                    pinned_attempt = replace(
                        attempt,
                        candidate_digest=decoded.candidate_digest,
                    )
                    desired = state.replace_item(item.replace_attempt(pinned_attempt))
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.ATTEMPT_UPDATED,
                        "persisted controller-validated QA candidate linkage",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                if decoded.verdict is not QaVerdict.APPROVE:
                    return self._close_decision(
                        state,
                        item,
                        attempt,
                        rejected=decoded.verdict is QaVerdict.REJECT,
                        summary="terminal QA did not approve the exact frozen candidate",
                    )
                return self._result(
                    state,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.NO_CHANGE,
                    "terminal QA validated the frozen candidate; fan-in is now eligible",
                    item_id=item.item_id,
                    attempt_id=attempt.attempt_id,
                )
            raise ResultContractError(f"unsupported successful attempt {attempt.kind.value!r}")
        except (ExecutionAdapterError, OSError, RuntimeError, TypeError, ValueError) as error:
            return self._blocked(
                state,
                f"typed successful result failed closed: {error}",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )

    def _close_tuner_attempt(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        item: WorkItemRecord,
        attempt: AttemptRecord,
        manifest: WorkerResultManifest,
    ) -> RuntimeTickResult:
        """Freeze Tuner evidence and enter the normal gate/review/QA chain."""
        decoded = decode_worker_result(manifest, attempt, proposal)
        if not isinstance(decoded, TunerResult):
            raise ResultContractError("Tuner decoder returned another result kind")
        if self._domain_inputs is None:
            raise RuntimeErrorBase("Tuner closure requires frozen runtime inputs")
        hypothesis = self._domain_inputs.hypotheses.get(item.item_id)
        if hypothesis is None or hypothesis != proposal.domain_input:
            raise RuntimeErrorBase("Tuner closure hypothesis differs from the admitted WorkItem")

        root = self._tuning_artifact_root(item.item_id)
        decision_path = root / PROMOTION_DECISION_FILENAME
        if decision_path.exists() or decision_path.is_symlink():
            raise RuntimeErrorBase(
                "Tuner promotion decision exists before Reviewer and QA approval"
            )
        evaluation = evaluate_tuning_campaign(
            proposal,
            hypothesis,
            decoded.baseline,
            decoded.candidate,
        )
        campaign = record_tuning_evidence(
            new_tuning_campaign(proposal, hypothesis),
            evaluation,
        )
        artifacts = (
            (
                root / BASELINE_FILENAME,
                decoded.baseline,
                load_baseline_artifact,
                write_baseline_artifact,
            ),
            (
                root / CANDIDATE_FILENAME,
                decoded.candidate,
                load_candidate_artifact,
                write_candidate_artifact,
            ),
            (
                root / EVALUATION_FILENAME,
                evaluation,
                load_evaluation_artifact,
                write_evaluation_artifact,
            ),
            (
                root / CAMPAIGN_FILENAME,
                campaign,
                load_campaign_artifact,
                write_campaign_artifact,
            ),
        )
        created_artifact = False
        for path, expected, loader, writer in artifacts:
            if path.exists() or path.is_symlink():
                observed, _digest = loader(path)
                if observed != expected:
                    raise RuntimeErrorBase(
                        f"durable Tuner artifact differs from measured evidence: {path.name}"
                    )
            else:
                writer(path, expected)
                created_artifact = True
        if created_artifact:
            return self._result(
                state,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.TUNING_CAMPAIGN_MATERIALIZED,
                "controller materialized immutable evidence-ready tuning artifacts",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )

        receipt_path = self._attempt_dir(attempt) / CANDIDATE_RECEIPT_FILENAME
        receipt_exists = receipt_path.exists()
        binding = bind_tuner_candidate(
            manifest,
            attempt,
            proposal,
            worktree=self._workspace / "candidates" / item.item_id / attempt.attempt_id,
            base_commit=self._coder_base_commit(state, proposal),
            attempt_dir=self._attempt_dir(attempt),
            artifact_root=root,
            git=self._git,
        )
        if not receipt_exists:
            return self._result(
                state,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.CANDIDATE_MATERIALIZED,
                "controller bound Tuner evidence to an immutable verification candidate",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )
        if attempt.candidate_digest is None:
            bound_attempt = replace(
                attempt,
                candidate_digest=binding.receipt.candidate_digest,
            )
            desired = state.replace_item(item.replace_attempt(bound_attempt))
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ATTEMPT_UPDATED,
                "persisted the controller-bound Tuner candidate digest",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )
        if attempt.candidate_digest != binding.receipt.candidate_digest:
            raise RuntimeErrorBase(
                "persisted Tuner digest differs from the immutable candidate receipt"
            )
        return self._plan_gate(state, proposal, binding, gate_index=0)

    def _finalize_tuning_campaign(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        item: WorkItemRecord,
    ) -> RuntimeTickResult | None:
        """Replay approved evidence and grant the sole controller promotion decision."""
        if self._domain_inputs is None:
            raise RuntimeErrorBase("Tuner promotion requires frozen runtime inputs")
        hypothesis = self._domain_inputs.hypotheses.get(item.item_id)
        if hypothesis is None or hypothesis != proposal.domain_input:
            raise RuntimeErrorBase("Tuner promotion hypothesis differs from the admitted item")
        root = self._tuning_artifact_root(item.item_id)
        baseline, _baseline_digest = load_baseline_artifact(root / BASELINE_FILENAME)
        candidate, _candidate_digest = load_candidate_artifact(root / CANDIDATE_FILENAME)
        evaluation, _evaluation_digest = load_evaluation_artifact(root / EVALUATION_FILENAME)
        campaign, _campaign_digest = load_campaign_artifact(root / CAMPAIGN_FILENAME)
        expected_evaluation = evaluate_tuning_campaign(
            proposal,
            hypothesis,
            baseline,
            candidate,
        )
        expected_campaign = record_tuning_evidence(
            new_tuning_campaign(proposal, hypothesis),
            expected_evaluation,
        )
        if evaluation != expected_evaluation or campaign != expected_campaign:
            raise RuntimeErrorBase(
                "immutable tuning artifacts do not replay to the admitted evidence"
            )

        expected_decision = make_promotion_decision(
            expected_evaluation,
            owner=DecisionOwner.CONTROLLER,
        )
        decision_path = root / PROMOTION_DECISION_FILENAME
        if not decision_path.exists() and not decision_path.is_symlink():
            candidate_attempt_id = item.candidate_attempt_id
            if candidate_attempt_id is None:
                raise RuntimeErrorBase("approved Tuner item has no frozen candidate attempt")
            candidate_attempt = next(
                attempt for attempt in item.attempts if attempt.attempt_id == candidate_attempt_id
            )
            manifest = self._validated_manifest(state, proposal, candidate_attempt)
            if manifest is None:
                raise RuntimeErrorBase("approved Tuner candidate manifest cannot be reconstructed")
            bind_tuner_candidate(
                manifest,
                candidate_attempt,
                proposal,
                worktree=(
                    self._workspace / "candidates" / item.item_id / candidate_attempt.attempt_id
                ),
                base_commit=self._coder_base_commit(state, proposal),
                attempt_dir=self._attempt_dir(candidate_attempt),
                artifact_root=root,
                git=self._git,
            )
            write_promotion_decision_artifact(
                decision_path,
                expected_decision,
                evaluation=expected_evaluation,
                owner=DecisionOwner.CONTROLLER,
            )
            return self._result(
                state,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.TUNING_CAMPAIGN_MATERIALIZED,
                "controller persisted the post-Reviewer/QA tuning promotion decision",
                item_id=item.item_id,
            )

        observed_decision, _decision_digest = load_promotion_decision_artifact(decision_path)
        if observed_decision != expected_decision:
            raise RuntimeErrorBase(
                "durable tuning promotion differs from the replayed controller decision"
            )
        terminal = replay_terminal_tuning_campaign(
            root,
            item=proposal,
            hypothesis=hypothesis,
            owner=DecisionOwner.CONTROLLER,
        )
        decision = terminal.decision
        if decision is None or decision != expected_decision:
            raise RuntimeErrorBase("terminal tuning replay lacks the exact controller decision")
        if decision.action is PromotionAction.REJECT:
            desired = state.replace_item(
                item.transition(
                    WorkItemStatus.REJECTED,
                    terminal_reason=(
                        f"controller rejected tuning outcome {decision.outcome.value}: "
                        f"{decision.reason}"
                    ),
                )
            )
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ATTEMPT_UPDATED,
                "controller rejected the independently approved tuning campaign",
                item_id=item.item_id,
            )
        if decision.action is not PromotionAction.KEEP:
            raise RuntimeErrorBase("replayed tuning campaign has an unsupported decision")
        return None

    def _validated_manifest(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        attempt: AttemptRecord,
    ) -> WorkerResultManifest | None:
        try:
            adapter = self._execution_adapter_for_attempt(
                state.item(attempt.item_id),
                attempt,
            )
            if adapter is not None:
                item = state.item(attempt.item_id)
                action = self._persisted_attempt_action(state, proposal, item, attempt)
                execution = adapter.execution(state, action)
            elif self._execution_factory is not None:
                execution = self._execution_factory(state, proposal, attempt)
            else:
                return None
        except (ExecutionAdapterError, KeyError, OSError, TypeError, ValueError):
            return None
        try:
            manifest, result_digest = load_result_manifest(execution.attempt_dir)
        except (OSError, ValueError):
            return None
        expectation: ResultExpectation = execution.expectation
        identity = (
            manifest.run_id,
            manifest.item_id,
            manifest.attempt_id,
            manifest.task_digest,
            manifest.generation,
            manifest.input_digest,
        )
        expected = (
            expectation.run_id,
            expectation.item_id,
            expectation.attempt_id,
            expectation.task_digest,
            expectation.generation,
            expectation.input_digest,
        )
        if identity != expected or result_digest != attempt.result_digest:
            return None
        return manifest

    def _mark_one_ready(self, state: RunState, plan: PlanDraftOutcome) -> RuntimeTickResult | None:
        proposals = {proposal.item_id: proposal for proposal in plan.items}
        for item in sorted(state.items, key=lambda entry: entry.item_id):
            if item.status is not WorkItemStatus.PLANNED:
                continue
            if any(
                state.item(dependency).status is not WorkItemStatus.INTEGRATED
                for dependency in item.dependencies
            ):
                continue
            proposal = proposals[item.item_id]
            if proposal.dependencies != item.dependencies:
                raise FrozenPlanError("state dependencies differ from the frozen plan")
            desired = mark_ready_after_integrated_dependencies(state, item.item_id)
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ITEM_READY,
                "all typed dependencies are integrated; WorkItem became READY",
                item_id=item.item_id,
            )
        return None

    def _dispatch_ready(
        self,
        state: RunState,
        plan: PlanDraftOutcome,
        review: PlanReviewOutcome,
        item: WorkItemRecord,
    ) -> RuntimeTickResult:
        proposal = _proposal(plan, item.item_id)
        if item.profile is DomainProfile.SMITH and proposal.kind in CATALOG_ITEM_KINDS:
            if self._execution_factory is None and self._worker_adapter is None:
                return self._blocked(
                    state,
                    "Smith dispatch is blocked before reservation because the generic worker "
                    "execution adapter is not configured",
                    item_id=item.item_id,
                )
            active_count = sum(
                attempt.status in _ACTIVE_ATTEMPT_STATUSES
                for entry in state.items
                for attempt in entry.attempts
            )
            smith = self._single_dispatch_config(self._task.execution.slurm.smith, active_count)
            tick = plan_coder_wave(
                state,
                plan,
                review,
                smith=smith,
                workspace=self._workspace,
                allowed_path_roots=_allowed_path_roots(self._task),
            )
            if not tick.actions:
                return self._blocked(
                    state,
                    "Smith ready item could not be selected within configured job/node/GPU "
                    "caps or immutable path/claim locks",
                    item_id=item.item_id,
                )
            if len(tick.actions) != 1 or tick.state.revision != state.revision + 1:
                raise RuntimeErrorBase(
                    "one runtime tick must persist exactly one Smith reservation"
                )
            action = tick.actions[0]
            self._save(state, tick.state)
            return self._result(
                tick.state,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.ACTION_PERSISTED,
                "controller persisted one bounded Smith Coder reservation",
                item_id=action.proposal.item_id,
                attempt_id=action.attempt.attempt_id,
            )
        if item.profile is DomainProfile.ASSEMBLER:
            if self._domain_inputs is None or self._worker_adapter is None:
                return self._blocked(
                    state,
                    "Assembler dispatch requires frozen runtime inputs and a generic worker adapter",
                    item_id=item.item_id,
                )
            try:
                preparation = self._assembler_preparation(state, proposal)
                domain = plan_assembler_coder_action(
                    state,
                    proposal,
                    preparation,
                    resource=self._role_resource(SlurmRole.ASSEMBLER_CODER),
                    workspace=self._workspace,
                    settings=self._agent_settings(),
                    evidence_paths=self._domain_inputs.evidence_paths_for(Role.CODER),
                )
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
                return self._blocked(
                    state,
                    f"Assembler input contract is incomplete: {error}",
                    item_id=item.item_id,
                )
            return self._persist_domain_plan(
                state,
                domain,
                "persisted one typed Assembler Coder action",
            )
        if item.profile is DomainProfile.TUNER:
            if self._domain_inputs is None or self._worker_adapter is None:
                return self._blocked(
                    state,
                    "Tuner dispatch requires frozen runtime inputs and a generic worker adapter",
                    item_id=item.item_id,
                )
            hypothesis = self._domain_inputs.hypotheses.get(item.item_id)
            if hypothesis is None:
                return self._blocked(
                    state,
                    "Tuner dispatch has no typed one-variable TuningHypothesis",
                    item_id=item.item_id,
                )
            domain = plan_tuner_coder_action(
                state,
                proposal,
                hypothesis,
                resource=self._role_resource(SlurmRole.TUNER_CODER),
                workspace=self._workspace,
                settings=self._agent_settings(),
                evidence_paths=self._domain_inputs.evidence_paths_for(Role.CODER),
            )
            return self._persist_domain_plan(
                state,
                domain,
                "persisted one typed Tuner Coder action",
            )
        return self._blocked(
            state,
            f"no controller adapter exists for ready WorkItem kind {item.kind.value!r}",
            item_id=item.item_id,
        )

    def _assembler_blocker(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        item: WorkItemRecord,
    ) -> RuntimeTickResult:
        if state.workflow_mode is not WorkflowMode.ONBOARD:
            return self._blocked(
                state,
                "Assembler work is not permitted in a tuning run",
                item_id=item.item_id,
            )
        route = route_identity_from_task(self._task)
        unit_by_kind = {
            WorkItemKind.ASSEMBLE_CORE: AssemblerUnit.CORE,
            WorkItemKind.ASSEMBLE_FEATURE: AssemblerUnit.FEATURE,
            WorkItemKind.ROUTING: AssemblerUnit.ROUTING,
        }
        unit = unit_by_kind.get(item.kind)
        if unit is None:
            return self._blocked(
                state,
                "deterministic gate execution has a typed suite but no persisted gate-attempt "
                "runner/result adapter",
                item_id=item.item_id,
            )
        feature: str | None = None
        if unit is AssemblerUnit.FEATURE:
            if len(self._task.target.features) != 1:
                return self._blocked(
                    state,
                    "feature Assembler WorkItem does not identify exactly one target feature",
                    item_id=item.item_id,
                )
            feature = self._task.target.features[0]
        prepare_assembler_input(
            state,
            proposal.item_id,
            route,
            unit,
            feature=feature,
        )
        return self._blocked(
            state,
            "Assembler input is valid, but no controller-owned Assembler role/output adapter "
            "persists and validates the candidate before review",
            item_id=item.item_id,
        )

    def _advance_hierarchy_once(
        self, state: RunState, *, cancelling: bool
    ) -> RuntimeTickResult | None:
        for goal in sorted(state.goals, key=lambda entry: entry.goal_id):
            if goal.terminal:
                continue
            items = tuple(state.item(item_id) for item_id in goal.required_item_ids)
            desired_status: HierarchyStatus | None = None
            reason: str | None = None
            if goal.status is HierarchyStatus.PLANNED and any(
                item.status is not WorkItemStatus.PLANNED for item in items
            ):
                desired_status = HierarchyStatus.ACTIVE
            elif goal.status is HierarchyStatus.ACTIVE and all(
                item.status is WorkItemStatus.INTEGRATED for item in items
            ):
                desired_status = HierarchyStatus.SUCCEEDED
            elif all(item.terminal for item in items) and any(
                item.status is not WorkItemStatus.INTEGRATED for item in items
            ):
                desired_status = (
                    HierarchyStatus.CANCELLED if cancelling else HierarchyStatus.BLOCKED
                )
                reason = "required WorkItems did not all integrate"
            if desired_status is not None:
                desired_goal = goal.transition(
                    desired_status,
                    **({"terminal_reason": reason} if reason is not None else {}),
                )
                desired = _replace_goal(state, desired_goal)
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.HIERARCHY_UPDATED,
                    f"Goal {goal.goal_id} advanced to {desired_status.value}",
                )

        for stage in sorted(state.stages, key=lambda entry: entry.stage_id):
            if stage.terminal:
                continue
            goals = tuple(_goal(state, goal_id) for goal_id in stage.required_goal_ids)
            desired_status: StageStatus | None = None
            reason = None
            if stage.status is StageStatus.PLANNED and any(
                goal.status is not HierarchyStatus.PLANNED for goal in goals
            ):
                desired_status = StageStatus.ACTIVE
            elif stage.status is StageStatus.ACTIVE and all(
                goal.status is HierarchyStatus.SUCCEEDED for goal in goals
            ):
                desired_status = StageStatus.READY_FOR_QA
            elif stage.status is StageStatus.READY_FOR_QA:
                qa_passed, qa_reason = self._stage_qa_verdict(state, stage)
                if qa_passed:
                    desired_status = StageStatus.QA_PASSED
                elif qa_reason is not None:
                    desired_status = StageStatus.QA_FAILED
            elif stage.status is StageStatus.QA_PASSED:
                desired_status = StageStatus.CLOSED
            elif stage.status is StageStatus.QA_FAILED:
                desired_status = StageStatus.BLOCKED
                reason = "independent Stage QA failed"
            elif all(goal.terminal for goal in goals) and any(
                goal.status is not HierarchyStatus.SUCCEEDED for goal in goals
            ):
                desired_status = StageStatus.CANCELLED if cancelling else StageStatus.BLOCKED
                reason = "required Goals did not all succeed"
            if desired_status is not None:
                desired_stage = stage.transition(
                    desired_status,
                    **({"terminal_reason": reason} if reason is not None else {}),
                )
                desired = _replace_stage(state, desired_stage)
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.HIERARCHY_UPDATED,
                    f"Stage {stage.stage_id} advanced to {desired_status.value}",
                )

        if state.stages and all(stage.terminal for stage in state.stages):
            if all(stage.status is StageStatus.CLOSED for stage in state.stages) and all(
                item.status is WorkItemStatus.INTEGRATED for item in state.items
            ):
                status = RunTerminalStatus.SUCCEEDED
                reason = "all typed WorkItems integrated and every Stage passed QA"
                delivery = self._publish_terminal_delivery(state)
                if delivery is not None:
                    return delivery
            elif cancelling:
                status = RunTerminalStatus.CANCELLED
                reason = "durable cancellation request closed all owned work"
            else:
                status = RunTerminalStatus.BLOCKED
                reason = "one or more typed WorkItems could not be integrated"
            desired = state.finish(status, reason)
            self._save(state, desired)
            return self._terminal(desired)
        return None

    def _publish_terminal_delivery(self, state: RunState) -> RuntimeTickResult | None:
        """Publish and verify the immutable diff-only terminal artifact once."""
        if self._task.delivery.mode != "diff_only":
            return None
        try:
            inspection = self._git.inspect()
            if not inspection.clean or inspection.head != state.integration_head:
                raise RuntimeErrorBase(
                    "terminal delivery checkout differs from the persisted integration head"
                )
            delivery = self._delivery or DeliveryCheckout(
                self._git.repository,
                inspection.branch,
                inspection.head,
                self._git,
            )
            current = replace(delivery, head=inspection.head)
            patch = publish_delivery_patch(
                current,
                base_commit=state.base_commit,
                output_path=self._workspace / "delivery" / "staircase.patch",
            )
            receipt_path = self._workspace / "delivery" / "patch-receipt.json"
            payload = {
                "schema_version": 1,
                "run_id": state.run_id,
                "task_digest": state.task_digest,
                "base_commit": patch.base_commit,
                "head_commit": patch.head_commit,
                "patch_sha256": patch.sha256,
                "size_bytes": patch.size_bytes,
                "path": "delivery/staircase.patch",
            }
            created = _write_atomic_json_once(receipt_path, payload)
            observed = _load_exact_json(receipt_path, set(payload))
            if observed != payload:
                raise RuntimeErrorBase(
                    "terminal delivery receipt differs from the exact published patch"
                )
        except (GitOpsError, OSError, RuntimeError, TypeError, ValueError) as error:
            return self._blocked(state, f"terminal diff publication failed closed: {error}")
        if created:
            return self._result(
                state,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.DELIVERY_PUBLISHED,
                "published immutable diff-only delivery patch and receipt",
            )
        return None

    def _stage_qa_verdict(
        self,
        state: RunState,
        stage: StageRecord,
    ) -> tuple[bool, str | None]:
        """Project exact per-item QA evidence into one independent Stage verdict."""
        if self._domain_inputs is None or self._domain_inputs.qa_required_items is None:
            return False, "Stage QA policy is unavailable"
        goal_ids = set(stage.required_goal_ids)
        stage_items = tuple(item for item in state.items if item.goal_id in goal_ids)
        required = tuple(
            item for item in stage_items if item.item_id in self._domain_inputs.qa_required_items
        )
        for item in required:
            qa_attempts = tuple(
                attempt for attempt in item.attempts if attempt.kind is AttemptKind.QA
            )
            if not qa_attempts:
                return False, f"required item {item.item_id!r} has no QA attempt"
            qa = qa_attempts[-1]
            if qa.status is not AttemptStatus.VALIDATED:
                if qa.terminal:
                    return False, f"required item {item.item_id!r} QA did not validate"
                return False, None
            if (
                item.status is not WorkItemStatus.INTEGRATED
                or item.candidate_digest is None
                or qa.candidate_digest != item.candidate_digest
            ):
                return False, f"required item {item.item_id!r} QA is not integration-pinned"
        if any(item.status is not WorkItemStatus.INTEGRATED for item in stage_items):
            return False, "Stage contains a non-integrated WorkItem"
        return True, None

    def _tick_cancellation(self, state: RunState, reason: str) -> RuntimeTickResult:
        if not reason.strip():
            raise ValueError("cancellation reason must be non-empty")
        if not state.items:
            planning = PlanningEngine(
                workspace=self._workspace,
                task=self._task,
                scheduler=self._scheduler,
                generation=self._generation,
                config=self._planning_config(),
                credential_broker=self._credential_broker,
            ).tick(cancel_requested=True, cancellation_reason=reason)
            return self._planning_result(planning)

        active = self._first_active_attempt(state)
        if active is not None:
            item, attempt = active
            if attempt.status is AttemptStatus.PREPARED:
                cancelled = attempt.transition(
                    AttemptStatus.CANCELLED,
                    terminal_reason=reason,
                )
                desired = state.replace_item(item.replace_attempt(cancelled))
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.CANCELLATION_UPDATED,
                    "cancelled a prepared attempt before scheduler submission",
                    item_id=item.item_id,
                    attempt_id=attempt.attempt_id,
                )
            if attempt.status is AttemptStatus.SUBMITTING and attempt.job is None:
                try:
                    proposal = self._proposal_for_item(item.item_id)
                    adapter = self._execution_adapter_for_attempt(item, attempt)
                    if adapter is not None:
                        action = self._persisted_attempt_action(
                            state,
                            proposal,
                            item,
                            attempt,
                        )
                        execution = adapter.execution(state, action)
                    elif self._execution_factory is not None:
                        execution = self._execution_factory(state, proposal, attempt)
                    else:
                        raise RuntimeErrorBase("submission cancellation has no execution adapter")
                    result = cancel_submitting_intent(
                        attempt,
                        scheduler=self._scheduler,
                        execution=execution,
                        cancellation_reason=reason,
                    )
                except (
                    AttemptEngineError,
                    ExecutionAdapterError,
                    KeyError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    return self._blocked(
                        state,
                        f"submission intent cancellation failed closed: {error}",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                if result.attempt == attempt:
                    return self._result(
                        state,
                        RuntimeDisposition.WAIT,
                        RuntimeEvent.NO_CHANGE,
                        "waiting for bounded submission cancellation evidence",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                desired = state.replace_item(item.replace_attempt(result.attempt))
                self._save(state, desired)
                return self._result(
                    desired,
                    RuntimeDisposition.CONTINUE,
                    RuntimeEvent.CANCELLATION_UPDATED,
                    f"reconciled submission cancellation as {result.action.value}",
                    item_id=item.item_id,
                    attempt_id=attempt.attempt_id,
                )
            if attempt.status in {
                AttemptStatus.SUBMITTED,
                AttemptStatus.PENDING,
                AttemptStatus.RUNNING,
            }:
                identity = _scheduler_job(cast(JobReference, attempt.job))
                if attempt.submission_token is None:
                    raise RuntimeErrorBase("submitted cancellation target has no immutable token")
                observation = self._scheduler.observe_owned(
                    identity,
                    attempt.submission_token,
                )
                if observation.status.terminal:
                    observed = attempt.transition(
                        AttemptStatus.TERMINAL_OBSERVED,
                        scheduler_state=observation.status.value,
                        terminal_reason=(
                            reason
                            if observation.status is JobStatus.CANCELLED
                            else observation.reason
                        ),
                    )
                    desired = state.replace_item(item.replace_attempt(observed))
                    self._save(state, desired)
                    return self._result(
                        desired,
                        RuntimeDisposition.CONTINUE,
                        RuntimeEvent.CANCELLATION_UPDATED,
                        "persisted terminal scheduler observation during cancellation",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                self._callbacks.heartbeat()
                try:
                    cancel_owned_attempt(attempt, self._scheduler)
                except AttemptEngineError as error:
                    return self._blocked(
                        state,
                        f"cancellation ownership verification failed closed: {error}",
                        item_id=item.item_id,
                        attempt_id=attempt.attempt_id,
                    )
                return self._result(
                    state,
                    RuntimeDisposition.WAIT,
                    RuntimeEvent.CANCEL_SIGNALLED,
                    "sent cancellation to the exact persisted scheduler identity",
                    item_id=item.item_id,
                    attempt_id=attempt.attempt_id,
                )
            terminal_status = (
                AttemptStatus.CANCELLED
                if attempt.scheduler_state == JobStatus.CANCELLED.value
                or attempt.status is AttemptStatus.COLLECTING
                else AttemptStatus.FAILED
            )
            terminal = attempt.transition(
                terminal_status,
                terminal_reason=reason,
            )
            desired = state.replace_item(item.replace_attempt(terminal))
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.CANCELLATION_UPDATED,
                "closed one scheduler-terminal attempt during cancellation",
                item_id=item.item_id,
                attempt_id=attempt.attempt_id,
            )

        for item in sorted(state.items, key=lambda entry: entry.item_id):
            for attempt in item.attempts:
                if not attempt.terminal:
                    continue
                revoked = self._revoke_terminal_attempt_credentials(state, item, attempt)
                if revoked is not None:
                    return revoked

        for item in sorted(state.items, key=lambda entry: entry.item_id):
            if item.terminal:
                continue
            desired = state.replace_item(
                item.transition(WorkItemStatus.CANCELLED, terminal_reason=reason)
            )
            self._save(state, desired)
            return self._result(
                desired,
                RuntimeDisposition.CONTINUE,
                RuntimeEvent.CANCELLATION_UPDATED,
                "closed one WorkItem after all of its attempts became terminal",
                item_id=item.item_id,
            )
        hierarchy = self._advance_hierarchy_once(state, cancelling=True)
        if hierarchy is not None:
            return hierarchy
        return self._blocked(state, "cancellation could not derive a safe next transition")

    def _validate_admitted_state(self, state: RunState, plan: PlanDraftOutcome) -> None:
        proposals = {proposal.item_id: proposal for proposal in plan.items}
        unsupported = tuple(
            sorted(
                {
                    proposal.kind.value
                    for proposal in plan.items
                    if proposal.kind not in _PRODUCTION_WORK_ITEM_KINDS
                }
            )
        )
        if unsupported:
            raise FrozenPlanError(
                "admitted plan violates the production WorkItem adapter invariant: "
                f"unsupported kinds {list(unsupported)!r}"
            )
        if set(proposals) != {item.item_id for item in state.items}:
            raise FrozenPlanError("admitted WorkItem identities differ from frozen plan evidence")
        planned_stages = {stage.stage_id: stage.goal_ids for stage in plan.stages}
        if planned_stages != {stage.stage_id: stage.required_goal_ids for stage in state.stages}:
            raise FrozenPlanError("admitted Stage hierarchy differs from frozen plan evidence")
        planned_goals = {goal.goal_id: (goal.stage_id, goal.item_ids) for goal in plan.goals}
        if planned_goals != {
            goal.goal_id: (goal.stage_id, goal.required_item_ids) for goal in state.goals
        }:
            raise FrozenPlanError("admitted Goal hierarchy differs from frozen plan evidence")
        for item in state.items:
            proposal = proposals[item.item_id]
            if (
                item.goal_id != proposal.goal_id
                or item.kind.value != proposal.kind.value
                or item.dependencies != proposal.dependencies
            ):
                raise FrozenPlanError(
                    f"admitted WorkItem {item.item_id!r} differs from frozen plan evidence"
                )

    def _validate_gate_policy(self) -> None:
        build_gate_suite(
            entry_gpu_tests=self._task.gates.component_tests,
            collective_tests=self._task.gates.collective_tests,
            boot_tests=self._task.gates.boot_tests,
            accuracy=AccuracyCriteria(
                selector=self._task.gates.accuracy.selector,
                reference=self._task.gates.accuracy.reference,
                protocol=self._task.gates.accuracy.protocol,
                tolerance=self._task.gates.accuracy.tolerance,
            ),
            feature_signal_tests=self._task.gates.feature_signal_tests,
            base_environment=dict(self._task.execution.slurm.controller.environment),
        )

    def _validate_resource_envelope(self, state: RunState, plan: PlanDraftOutcome) -> None:
        proposals = {proposal.item_id: proposal for proposal in plan.items}
        reservations: list[AttemptReservation] = []
        for item in state.items:
            proposal = proposals[item.item_id]
            for attempt in item.attempts:
                resource = self._resource_for_attempt(item, proposal, attempt)
                reservations.append(
                    AttemptReservation(
                        attempt_id=attempt.attempt_id,
                        item_id=item.item_id,
                        status=attempt.status,
                        jobs=1,
                        nodes=resource.nodes,
                        gpus=resource.total_gpus,
                        path_locks=proposal.allowed_paths,
                        claim_locks=proposal.certified_claim_cells,
                    )
                )
        smith = self._task.execution.slurm.smith
        select_ready_wave(
            plan.items,
            item_statuses={item.item_id: item.status for item in state.items},
            attempts=reservations,
            caps=ResourceCaps(
                max_jobs=smith.max_parallel_items,
                max_nodes=smith.max_nodes_total,
                max_gpus=smith.max_gpus_total,
            ),
        )

    def _resource_for_attempt(
        self,
        item: WorkItemRecord,
        proposal: WorkItemProposal,
        attempt: AttemptRecord,
    ) -> ResourceClass:
        if attempt.kind is AttemptKind.DETERMINISTIC_GATE:
            gate_index = logical_attempt_ordinal(item, attempt.attempt_id) - 1
            gates = self._gate_suite()
            if gate_index >= len(gates):
                raise RuntimeErrorBase("gate attempt exceeds the frozen gate suite")
            return self._gate_resource(gates[gate_index].gate_id)
        input_path = self._attempt_dir(attempt) / INPUT_FILENAME
        resource_name = attempt.resource_class or {
            AttemptKind.REVIEWER_ANALYSIS: "reviewer_analysis",
            AttemptKind.REVIEWER_RERUN: "reviewer_rerun",
            AttemptKind.QA: "reviewer_analysis",
        }.get(attempt.kind, proposal.resource_class)
        if input_path.is_file() and not input_path.is_symlink():
            manifest, _digest = load_input_manifest(input_path)
            payload_resource = manifest.payload.get("resource_class")
            if isinstance(payload_resource, str):
                if attempt.resource_class is not None and payload_resource != resource_name:
                    raise RuntimeErrorBase(
                        "persisted input resource differs from durable attempt selection"
                    )
                resource_name = payload_resource
        return self._task.execution.slurm.smith.resource_class(resource_name)

    def _reconstruct_coder_action(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        attempt: AttemptRecord,
    ) -> SmithAttemptAction:
        item = state.item(proposal.item_id)
        if item.status is not WorkItemStatus.CODING or attempt not in item.attempts:
            raise RuntimeErrorBase("persisted Coder attempt is not owned by its WorkItem")
        if attempt.resource_class is None:
            raise RuntimeErrorBase("Smith attempt has no durable selected resource class")
        resource = self._task.execution.slurm.smith.resource_class(attempt.resource_class)
        worktree = self._workspace / "candidates" / proposal.item_id / attempt.attempt_id
        attempt_dir = self._attempt_dir(attempt)
        from ..common.artifacts import WorkerInputManifest

        manifest = WorkerInputManifest(
            run_id=state.run_id,
            item_id=proposal.item_id,
            attempt_id=attempt.attempt_id,
            task_digest=state.task_digest,
            generation=attempt.generation,
            role=attempt.role,
            profile=DomainProfile.SMITH,
            worktree=str(worktree),
            allowed_paths=proposal.allowed_paths,
            payload={
                "runtime": {
                    "schema_version": 1,
                    "kind": "agent",
                    "backend_kind": self._agent_settings().backend_kind,
                    "model": self._agent_settings().model,
                    "prompt_id": attempt.attempt_id,
                    "prompt": (
                        "Implement exactly the admitted atomic Smith catalog item. Stay within "
                        "allowed_paths and return the strict typed Coder result."
                    ),
                    "evidence_paths": list(
                        self._domain_inputs.evidence_paths_for(Role.CODER)
                        if self._domain_inputs is not None
                        else ()
                    ),
                    "candidate_digest": None,
                },
                "context": {
                    "action": "smith_coder",
                    "predecessor_attempt_id": attempt.predecessor_attempt_id,
                    "resource_escalation_request_id": (attempt.resource_escalation_request_id),
                    "proposal": {
                        "item_id": proposal.item_id,
                        "goal_id": proposal.goal_id,
                        "kind": proposal.kind.value,
                        "entry_id": proposal.entry_id,
                        "resource_class": resource.name,
                        "modifies_files": proposal.modifies_files,
                        "dependencies": list(proposal.dependencies),
                        "allowed_paths": list(proposal.allowed_paths),
                        "execution": {
                            "nodes": resource.nodes,
                            "tasks_per_node": resource.tasks_per_node,
                            "gpus_per_node": resource.gpus_per_node,
                        },
                        "certified_claim_cells": [
                            {"entry_id": claim.entry_id, "cell_id": claim.cell_id}
                            for claim in proposal.certified_claim_cells
                        ],
                    },
                },
            },
        )
        return SmithAttemptAction(
            proposal=proposal,
            attempt=attempt,
            resource=resource,
            worktree=worktree,
            branch=f"staircase/{state.run_id}/{attempt.attempt_id}",
            base_commit=state.base_commit,
            attempt_dir=attempt_dir,
            manifest=manifest,
        )

    def _persisted_attempt_action(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        item: WorkItemRecord,
        attempt: AttemptRecord,
    ) -> MaterializableAction:
        """Recover execution identity from input bytes across controller generations."""
        attempt_dir = self._attempt_dir(attempt)
        manifest, _input_digest = load_input_manifest(attempt_dir / INPUT_FILENAME)
        expected_identity = (
            state.run_id,
            attempt.item_id,
            attempt.attempt_id,
            state.task_digest,
            attempt.generation,
            attempt.role,
            attempt.profile,
        )
        observed_identity = (
            manifest.run_id,
            manifest.item_id,
            manifest.attempt_id,
            manifest.task_digest,
            manifest.generation,
            manifest.role,
            manifest.profile,
        )
        if observed_identity != expected_identity:
            raise RuntimeErrorBase(
                "persisted worker input differs from its generation-fenced attempt"
            )
        worktree = self._workspace / "candidates" / item.item_id / attempt.attempt_id
        return _PersistedWorkerAction(
            attempt=attempt,
            resource=self._resource_for_attempt(item, proposal, attempt),
            worktree=worktree,
            branch=f"staircase/{state.run_id}/{attempt.attempt_id}",
            base_commit=state.base_commit,
            attempt_dir=attempt_dir,
            manifest=manifest,
        )

    def _reconstruct_attempt_action(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        item: WorkItemRecord,
        attempt: AttemptRecord,
    ) -> MaterializableAction:
        """Re-run the pure planner against the exact pre-attempt state."""
        if attempt.role is Role.CODER and attempt.profile is DomainProfile.SMITH:
            return self._reconstruct_coder_action(state, proposal, attempt)
        if self._domain_inputs is None:
            raise RuntimeErrorBase("domain action reconstruction requires runtime inputs")
        if attempt != item.attempts[-1]:
            raise RuntimeErrorBase("only the latest persisted attempt may be reconstructed")

        prior_item = replace(item, attempts=item.attempts[:-1])
        plan: DomainActionPlan
        if attempt.role is Role.CODER and attempt.profile is DomainProfile.ASSEMBLER:
            prior_item = replace(prior_item, status=WorkItemStatus.READY)
            prior = _replace_item_without_revision(state, prior_item)
            preparation = self._assembler_preparation(prior, proposal)
            plan = plan_assembler_coder_action(
                prior,
                proposal,
                preparation,
                resource=self._role_resource(SlurmRole.ASSEMBLER_CODER),
                workspace=self._workspace,
                settings=self._agent_settings(),
                evidence_paths=self._domain_inputs.evidence_paths_for(Role.CODER),
            )
        elif attempt.role is Role.CODER and attempt.profile is DomainProfile.TUNER:
            prior_item = replace(prior_item, status=WorkItemStatus.READY)
            prior = _replace_item_without_revision(state, prior_item)
            hypothesis = self._domain_inputs.hypotheses.get(item.item_id)
            if hypothesis is None:
                raise RuntimeErrorBase("Tuner WorkItem has no typed TuningHypothesis")
            plan = plan_tuner_coder_action(
                prior,
                proposal,
                hypothesis,
                resource=self._role_resource(SlurmRole.TUNER_CODER),
                workspace=self._workspace,
                settings=self._agent_settings(),
                evidence_paths=self._domain_inputs.evidence_paths_for(Role.CODER),
            )
        elif attempt.kind is AttemptKind.DETERMINISTIC_GATE:
            binding = self._candidate_binding(state, proposal)
            candidate = self._frozen_from_binding(binding)
            prior_item = replace(
                prior_item,
                status=WorkItemStatus.CODING,
                candidate_attempt_id=None,
                candidate_digest=None,
            )
            prior = _replace_item_without_revision(state, prior_item)
            gates = self._gate_suite()
            gate_index = logical_attempt_ordinal(item, attempt.attempt_id) - 1
            if gate_index >= len(gates):
                raise RuntimeErrorBase("gate attempt exceeds the frozen ordered suite")
            gate = gates[gate_index]
            execution = self._gate_execution(gate.gate_id)
            if self._requires_collective_gate(gate, execution):
                if self._collective_gate_adapter is None:
                    raise RuntimeErrorBase(
                        "collective GateSpec requires a trusted rank-supervisor adapter"
                    )
                plan = self._collective_gate_adapter.plan(
                    prior,
                    proposal,
                    candidate,
                    binding.receipt,
                    gate,
                    execution,
                    resource=self._gate_resource(gate.gate_id),
                    mapping=self._task.target.mapping,
                    workspace=self._workspace,
                )
            else:
                plan = plan_deterministic_gate_action(
                    prior,
                    proposal,
                    candidate,
                    gate,
                    execution,
                    resource=self._gate_resource(gate.gate_id),
                    workspace=self._workspace,
                )
        elif attempt.kind is AttemptKind.REVIEWER_ANALYSIS:
            candidate = self._frozen_candidate(state, proposal)
            prior_item = replace(prior_item, status=WorkItemStatus.CANDIDATE_READY)
            prior = _replace_item_without_revision(state, prior_item)
            plan = plan_reviewer_analysis_action(
                prior,
                proposal,
                candidate,
                self._gate_suite(),
                gate_attempt_ids=self._gate_attempt_ids(prior_item),
                resource=self._role_resource(SlurmRole.REVIEWER),
                workspace=self._workspace,
                settings=self._agent_settings(),
                evidence_paths=self._domain_inputs.evidence_paths_for(Role.REVIEWER),
                candidate_evidence=self._candidate_evidence_context(proposal),
            )
        elif attempt.kind is AttemptKind.REVIEWER_RERUN:
            candidate = self._frozen_candidate(state, proposal)
            prior = _replace_item_without_revision(state, prior_item)
            analysis = self._latest_attempt(prior_item, AttemptKind.REVIEWER_ANALYSIS)
            plan = plan_reviewer_rerun_action(
                prior,
                proposal,
                candidate,
                self._gate_suite(),
                analysis_attempt_id=analysis.attempt_id,
                gate_attempt_ids=self._gate_attempt_ids(prior_item),
                resource=self._role_resource(SlurmRole.REVIEWER),
                workspace=self._workspace,
                settings=self._agent_settings(),
                evidence_paths=self._domain_inputs.evidence_paths_for(Role.REVIEWER),
                candidate_evidence=self._candidate_evidence_context(proposal),
            )
        elif attempt.kind is AttemptKind.QA:
            candidate = self._frozen_candidate(state, proposal)
            prior = _replace_item_without_revision(state, prior_item)
            plan = plan_qa_action(
                prior,
                proposal,
                candidate,
                self._gate_suite(),
                gate_attempt_ids=self._gate_attempt_ids(prior_item),
                gate_result_digests=self._qa_gate_result_digests(prior_item),
                resource=self._role_resource(SlurmRole.QA),
                workspace=self._workspace,
                settings=self._agent_settings(),
                evidence_paths=self._domain_inputs.evidence_paths_for(Role.QA),
                candidate_evidence=self._candidate_evidence_context(proposal),
            )
        else:
            raise RuntimeErrorBase(
                f"no domain action reconstruction exists for {attempt.kind.value!r}"
            )
        if plan.action is None:
            raise RuntimeErrorBase(
                f"domain action reconstruction blocked: {cast(object, plan.blocked)}"
            )
        planned_attempt = plan.action.attempt
        if (
            planned_attempt.attempt_id,
            planned_attempt.item_id,
            planned_attempt.sequence,
            planned_attempt.role,
            planned_attempt.kind,
            planned_attempt.generation,
            planned_attempt.profile,
            planned_attempt.review_of_attempt_id,
            planned_attempt.reviewed_candidate_digest,
        ) != (
            attempt.attempt_id,
            attempt.item_id,
            attempt.sequence,
            attempt.role,
            attempt.kind,
            attempt.generation,
            attempt.profile,
            attempt.review_of_attempt_id,
            attempt.reviewed_candidate_digest,
        ):
            raise RuntimeErrorBase("reconstructed domain action differs from persisted identity")
        planned_item = plan.state.item(item.item_id)
        if replace(planned_item, attempts=item.attempts) != item:
            raise RuntimeErrorBase("reconstructed domain reservation differs from persisted item")
        return replace(plan.action, attempt=attempt)

    def _assembler_preparation(
        self,
        state: RunState,
        proposal: WorkItemProposal,
    ):  # type: ignore[no-untyped-def]
        if self._domain_inputs is None:
            raise RuntimeErrorBase("Assembler preparation requires runtime inputs")
        item = state.item(proposal.item_id)
        unit_by_kind = {
            WorkItemKind.ASSEMBLE_CORE: AssemblerUnit.CORE,
            WorkItemKind.ASSEMBLE_FEATURE: AssemblerUnit.FEATURE,
            WorkItemKind.ROUTING: AssemblerUnit.ROUTING,
        }
        unit = unit_by_kind.get(item.kind)
        if unit is None:
            raise RuntimeErrorBase("WorkItem kind has no typed Assembler unit")
        feature = self._domain_inputs.features.get(item.item_id)
        if unit is AssemblerUnit.FEATURE and feature is None:
            raise RuntimeErrorBase("feature Assembler has no explicit feature identity")
        requirements: list[CatalogRequirement] = []
        surfaces: list[IntegratedCatalogSurface] = []
        for dependency_id in item.dependencies:
            dependency_proposal = self._proposal_for_item(dependency_id)
            binding = self._candidate_binding(state, dependency_proposal)
            if binding.index_delta is None:
                raise RuntimeErrorBase(
                    f"integrated catalog dependency {dependency_id!r} has no IndexDelta"
                )
            delta = binding.index_delta
            catalog_path = PurePosixPath(delta.row.path)
            requirements.append(
                CatalogRequirement(
                    delta.row.entry_id,
                    catalog_path,
                    f"admitted dependency {dependency_id}",
                )
            )
            surfaces.append(
                IntegratedCatalogSurface(
                    dependency_id,
                    delta.row.entry_id,
                    catalog_path,
                )
            )
        return prepare_assembler_input(
            state,
            proposal.item_id,
            route_identity_from_task(self._task),
            unit,
            tuple(requirements),
            tuple(surfaces),
            feature=feature,
        )

    def _plan_gate(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        binding: CandidateBinding,
        *,
        gate_index: int,
    ) -> RuntimeTickResult:
        gates = self._gate_suite()
        if gate_index >= len(gates):
            return self._blocked(state, "deterministic gate index exceeds the frozen suite")
        gate = gates[gate_index]
        execution = self._gate_execution(gate.gate_id)
        if self._requires_collective_gate(gate, execution):
            if self._collective_gate_adapter is None:
                return self._blocked(
                    state,
                    "collective GateSpec requires a trusted rank-supervisor adapter",
                    item_id=proposal.item_id,
                )
            plan = self._collective_gate_adapter.plan(
                state,
                proposal,
                self._frozen_from_binding(binding),
                binding.receipt,
                gate,
                execution,
                resource=self._gate_resource(gate.gate_id),
                mapping=self._task.target.mapping,
                workspace=self._workspace,
            )
        else:
            plan = plan_deterministic_gate_action(
                state,
                proposal,
                self._frozen_from_binding(binding),
                gate,
                execution,
                resource=self._gate_resource(gate.gate_id),
                workspace=self._workspace,
            )
        return self._persist_domain_plan(
            state,
            plan,
            f"persisted deterministic gate job {gate_index + 1}/{len(gates)}",
        )

    def _plan_reviewer_analysis(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        binding: CandidateBinding,
    ) -> RuntimeTickResult:
        if self._domain_inputs is None:
            return self._blocked(state, "Reviewer analysis requires runtime agent settings")
        plan = plan_reviewer_analysis_action(
            state,
            proposal,
            self._frozen_from_binding(binding),
            self._gate_suite(),
            gate_attempt_ids=self._gate_attempt_ids(state.item(proposal.item_id)),
            resource=self._role_resource(SlurmRole.REVIEWER),
            workspace=self._workspace,
            settings=self._agent_settings(),
            evidence_paths=self._domain_inputs.evidence_paths_for(Role.REVIEWER),
            candidate_evidence=self._candidate_evidence_context(proposal),
        )
        return self._persist_domain_plan(state, plan, "persisted fresh Reviewer analysis job")

    def _plan_reviewer_rerun(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        binding: CandidateBinding,
        analysis_attempt_id: str,
    ) -> RuntimeTickResult:
        if self._domain_inputs is None:
            return self._blocked(state, "Reviewer rerun requires runtime agent settings")
        plan = plan_reviewer_rerun_action(
            state,
            proposal,
            self._frozen_from_binding(binding),
            self._gate_suite(),
            analysis_attempt_id=analysis_attempt_id,
            gate_attempt_ids=self._gate_attempt_ids(state.item(proposal.item_id)),
            resource=self._role_resource(SlurmRole.REVIEWER),
            workspace=self._workspace,
            settings=self._agent_settings(),
            evidence_paths=self._domain_inputs.evidence_paths_for(Role.REVIEWER),
            candidate_evidence=self._candidate_evidence_context(proposal),
        )
        return self._persist_domain_plan(state, plan, "persisted fresh Reviewer rerun job")

    def _persist_domain_plan(
        self,
        state: RunState,
        plan: DomainActionPlan,
        reason: str,
    ) -> RuntimeTickResult:
        if plan.blocked is not None:
            return self._blocked(
                state,
                f"{plan.blocked.code.value}: {plan.blocked.reason}",
                item_id=plan.blocked.item_id,
            )
        action = cast(DomainAttemptAction, plan.action)
        if plan.state.revision != state.revision + 1:
            raise RuntimeErrorBase("domain planner must persist exactly one revision")
        self._save(state, plan.state)
        return self._result(
            plan.state,
            RuntimeDisposition.CONTINUE,
            RuntimeEvent.ACTION_PERSISTED,
            reason,
            item_id=action.attempt.item_id,
            attempt_id=action.attempt.attempt_id,
        )

    def _close_decision(
        self,
        state: RunState,
        item: WorkItemRecord,
        attempt: AttemptRecord,
        *,
        rejected: bool,
        summary: str,
    ) -> RuntimeTickResult:
        status = WorkItemStatus.REJECTED if rejected else WorkItemStatus.BLOCKED
        desired = state.replace_item(item.transition(status, terminal_reason=summary))
        self._save(state, desired)
        return self._result(
            desired,
            RuntimeDisposition.CONTINUE,
            RuntimeEvent.ATTEMPT_UPDATED,
            summary,
            item_id=item.item_id,
            attempt_id=attempt.attempt_id,
        )

    @staticmethod
    def _frozen_from_binding(binding: CandidateBinding) -> FrozenCandidate:
        snapshot = binding.snapshot
        return FrozenCandidate(
            snapshot.item_id,
            snapshot.coder_attempt_id,
            snapshot.candidate_commit,
            snapshot.candidate_digest,
            snapshot.changed_paths,
        )

    def _coder_base_commit(
        self,
        state: RunState,
        proposal: WorkItemProposal,
    ) -> str:
        item = state.item(proposal.item_id)
        if item.profile is DomainProfile.SMITH:
            return state.base_commit
        return state.integration_head

    def _candidate_binding(
        self,
        state: RunState,
        proposal: WorkItemProposal,
    ) -> CandidateBinding:
        item = state.item(proposal.item_id)
        candidate_id = item.candidate_attempt_id
        if candidate_id is None:
            validated_coders = [
                attempt
                for attempt in item.attempts
                if attempt.role is Role.CODER
                and attempt.status is AttemptStatus.VALIDATED
                and attempt.candidate_digest is not None
            ]
            if len(validated_coders) != 1:
                raise RuntimeErrorBase("WorkItem does not identify one bound Coder candidate")
            candidate = validated_coders[0]
        else:
            candidate = next(
                attempt for attempt in item.attempts if attempt.attempt_id == candidate_id
            )
        receipt_path = self._attempt_dir(candidate) / CANDIDATE_RECEIPT_FILENAME
        worktree = self._workspace / "controller-candidates" / item.item_id / candidate.attempt_id
        return reconstruct_candidate(
            receipt_path,
            worktree=worktree,
            proposal=proposal,
            git=self._git,
        )

    def _frozen_candidate(
        self,
        state: RunState,
        proposal: WorkItemProposal,
    ) -> FrozenCandidate:
        snapshot = self._candidate_binding(state, proposal).snapshot
        return FrozenCandidate(
            snapshot.item_id,
            snapshot.coder_attempt_id,
            snapshot.candidate_commit,
            snapshot.candidate_digest,
            snapshot.changed_paths,
        )

    def _gate_suite(self):  # type: ignore[no-untyped-def]
        return build_gate_suite(
            entry_gpu_tests=self._task.gates.component_tests,
            collective_tests=self._task.gates.collective_tests,
            boot_tests=self._task.gates.boot_tests,
            accuracy=AccuracyCriteria(
                selector=self._task.gates.accuracy.selector,
                reference=self._task.gates.accuracy.reference,
                protocol=self._task.gates.accuracy.protocol,
                tolerance=self._task.gates.accuracy.tolerance,
            ),
            feature_signal_tests=self._task.gates.feature_signal_tests,
            base_environment=dict(self._task.execution.slurm.controller.environment),
        )

    def _resource_named(self, name: str) -> ResourceClass:
        return self._task.execution.slurm.smith.resource_class(name)

    def _role_resource(self, role: SlurmRole) -> ResourceClass:
        """Resolve one agent attempt resource from the frozen role-class policy."""
        policy = self._task.execution.slurm.role_class(role)
        return self._resource_named(policy.resource_class)

    def _agent_settings(self) -> AgentSettings:
        agent = self._task.execution.agent
        return AgentSettings(agent.backend_kind, agent.model)

    def _planning_config(self) -> PlanningConfig:
        agent = self._task.execution.agent
        return PlanningConfig(backend_kind=agent.backend_kind, model=agent.model)

    def _resolve_domain_inputs(
        self,
        state: RunState,
        plan: PlanDraftOutcome,
        review: PlanReviewOutcome,
    ) -> None:
        """Resolve once after approval, then reject any task/plan/environment drift."""
        if self._domain_inputs is None:
            if self._domain_input_resolver is None:
                raise RuntimeErrorBase("runtime has no deterministic domain-input resolver")
            resolved = self._domain_input_resolver(
                self._task,
                state.workflow_mode,
                plan,
                review,
                (),
            )
            if not isinstance(resolved, RuntimeDomainInputs):
                raise RuntimeErrorBase("domain-input resolver returned an invalid value")
            self._validate_domain_input_binding(state, resolved, plan=plan, review=review)
            self._domain_inputs = resolved
            if self._worker_adapter is None and not self._worker_adapter_was_supplied:
                self._worker_adapter = GenericWorkerExecutionAdapter(
                    workspace=self._workspace,
                    task=self._task,
                    git=self._git,
                    credential_broker=self._credential_broker,
                )
            self._validate_resolved_production_inputs()
            return
        self._validate_domain_input_binding(state, self._domain_inputs, plan=plan, review=review)

    def _validate_domain_input_binding(
        self,
        state: RunState,
        inputs: RuntimeDomainInputs,
        *,
        plan: PlanDraftOutcome | None = None,
        review: PlanReviewOutcome | None = None,
    ) -> None:
        if inputs.task_digest != self._task.digest or inputs.task_digest != state.task_digest:
            raise RuntimeErrorBase("runtime inputs are not bound to the normalized task digest")
        if plan is not None and inputs.plan_digest != plan.digest:
            raise RuntimeErrorBase("runtime inputs are not bound to the approved plan digest")
        if review is not None and (
            review.outcome is not PlanReviewDecision.ACCEPT
            or inputs.plan_digest != review.plan_digest
        ):
            raise RuntimeErrorBase("runtime inputs are not bound to the PlanReviewer verdict")

    def _gate_execution(self, gate_id: str) -> GateExecutionContract:
        if self._domain_inputs is None:
            raise RuntimeErrorBase("gate execution topology was not supplied")
        try:
            return self._domain_inputs.gate_execution_by_id[gate_id]
        except KeyError as error:
            raise RuntimeErrorBase(
                f"GateSpec {gate_id!r} has no typed GateExecutionContract"
            ) from error

    def _gate_resource(self, gate_id: str) -> ResourceClass:
        if self._domain_inputs is None:
            raise RuntimeErrorBase("gate resource topology was not supplied")
        try:
            resource = self._domain_inputs.gate_resource_by_id[gate_id]
        except KeyError as error:
            raise RuntimeErrorBase(
                f"GateSpec {gate_id!r} has no explicit deterministic_gate subrequest"
            ) from error
        envelope = self._resource_named("deterministic_gate")
        requested = (
            resource.nodes,
            resource.tasks_per_node,
            resource.gpus_per_node,
            resource.cpus_per_task,
            resource.memory_mib,
            resource.time_limit_seconds,
        )
        limits = (
            envelope.nodes,
            envelope.tasks_per_node,
            envelope.gpus_per_node,
            envelope.cpus_per_task,
            envelope.memory_mib,
            envelope.time_limit_seconds,
        )
        if any(value > limit for value, limit in zip(requested, limits, strict=True)):
            raise RuntimeErrorBase(
                f"GateSpec {gate_id!r} subrequest exceeds the deterministic_gate envelope"
            )
        return resource

    @staticmethod
    def _requires_collective_gate(
        gate: GateSpec,
        execution: GateExecutionContract,
    ) -> bool:
        return gate.phase is GatePhase.COLLECTIVE or execution.scope not in {
            EvidenceScope.CPU_STATIC,
            EvidenceScope.SINGLE_GPU_PRODUCT,
        }

    @staticmethod
    def _gate_attempt_ids(item: WorkItemRecord) -> tuple[str, ...]:
        return tuple(
            attempt.attempt_id
            for attempt in ControllerRuntime._logical_attempt_heads(
                item, AttemptKind.DETERMINISTIC_GATE
            )
        )

    @staticmethod
    def _logical_attempt_heads(
        item: WorkItemRecord,
        kind: AttemptKind,
    ) -> tuple[AttemptRecord, ...]:
        """Return one latest physical attempt per ordered logical lineage root."""
        roots: dict[str, AttemptRecord] = {}
        heads: dict[str, AttemptRecord] = {}
        for attempt in item.attempts:
            if attempt.kind is not kind:
                continue
            root = logical_attempt_root(item, attempt.attempt_id)
            roots[root.attempt_id] = root
            previous = heads.get(root.attempt_id)
            if previous is None or attempt.sequence > previous.sequence:
                heads[root.attempt_id] = attempt
        return tuple(
            heads[root_id] for root_id in sorted(roots, key=lambda value: roots[value].sequence)
        )

    @staticmethod
    def _qa_gate_result_digests(item: WorkItemRecord) -> Mapping[str, str]:
        references: dict[str, str] = {}
        for attempt in ControllerRuntime._logical_attempt_heads(
            item, AttemptKind.DETERMINISTIC_GATE
        ):
            if attempt.status is not AttemptStatus.VALIDATED or attempt.result_digest is None:
                raise RuntimeErrorBase(
                    "terminal QA requires every gate result to be controller-ingested"
                )
            references[attempt.attempt_id] = attempt.result_digest
        if not references:
            raise RuntimeErrorBase("terminal QA requires non-empty gate result references")
        return references

    def _reload_trusted_gate_results(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        item: WorkItemRecord,
    ) -> tuple[GateResult, ...]:
        """Reload controller-ingested gate bytes and revalidate their claim scopes."""
        candidate = self._candidate_binding(state, proposal).receipt
        attempts = self._logical_attempt_heads(
            item,
            AttemptKind.DETERMINISTIC_GATE,
        )
        gates = self._gate_suite()
        if len(attempts) != len(gates):
            raise ResultContractError("gate attempt count differs from the frozen suite")
        decoded_results: list[GateResult] = []
        for attempt, gate in zip(attempts, gates, strict=True):
            manifest = self._validated_manifest(state, proposal, attempt)
            if manifest is None:
                raise ResultContractError(
                    f"gate result {attempt.attempt_id!r} cannot be reloaded byte-for-byte"
                )
            decoded = decode_worker_result(
                manifest,
                attempt,
                proposal,
                candidate=candidate,
                gate_spec=gate,
            )
            if not isinstance(decoded, GateResult):
                raise ResultContractError("gate decoder returned another result kind")
            validate_receipt_claim(
                decoded.receipt,
                self._claim_scope(decoded.receipt.scope, decoded.receipt.purpose),
                gate_spec=gate,
            )
            decoded_results.append(decoded)
        return tuple(decoded_results)

    @staticmethod
    def _claim_scope(scope: EvidenceScope, purpose: GatePurpose) -> ClaimScope:
        if purpose is GatePurpose.ACCURACY:
            return ClaimScope.ACCURACY
        if purpose is GatePurpose.PERFORMANCE:
            return ClaimScope.PERFORMANCE
        if scope is EvidenceScope.CPU_STATIC:
            return ClaimScope.STRUCTURAL
        if scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
            return ClaimScope.LOCAL_FOUR_GPU_PRODUCT
        if scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER:
            return ClaimScope.SYNTHETIC_RUNNER
        if scope is EvidenceScope.REAL_MULTI_NODE_PRODUCT:
            return ClaimScope.MULTI_NODE_PRODUCT
        return ClaimScope.PRODUCT_CORRECTNESS

    def _execution_adapter_for_attempt(
        self,
        item: WorkItemRecord,
        attempt: AttemptRecord,
    ) -> WorkerExecutionAdapter | None:
        if attempt.kind is not AttemptKind.DETERMINISTIC_GATE:
            return self._worker_adapter
        gate_index = logical_attempt_ordinal(item, attempt.attempt_id) - 1
        gates = self._gate_suite()
        if gate_index >= len(gates):
            raise RuntimeErrorBase("gate attempt exceeds the frozen gate suite")
        gate = gates[gate_index]
        execution = self._gate_execution(gate.gate_id)
        if self._requires_collective_gate(gate, execution):
            if self._collective_gate_adapter is None:
                raise RuntimeErrorBase(
                    "collective GateSpec requires a trusted rank-supervisor adapter"
                )
            return self._collective_gate_adapter
        return self._worker_adapter

    @staticmethod
    def _latest_attempt(item: WorkItemRecord, kind: AttemptKind) -> AttemptRecord:
        matches = [attempt for attempt in item.attempts if attempt.kind is kind]
        if not matches:
            raise RuntimeErrorBase(f"WorkItem has no {kind.value!r} attempt")
        return matches[-1]

    def _proposal_for_item(self, item_id: str) -> WorkItemProposal:
        state = self._load_owned_state()
        plan, _review = recover_approved_plan(self._workspace, self._task, state)
        return _proposal(plan, item_id)

    def _attempt_dir(self, attempt: AttemptRecord) -> Path:
        return self._workspace / "items" / attempt.item_id / "attempts" / f"{attempt.sequence:04d}"

    def _first_active_attempt(self, state: RunState) -> tuple[WorkItemRecord, AttemptRecord] | None:
        for item in sorted(state.items, key=lambda entry: entry.item_id):
            for attempt in sorted(item.attempts, key=lambda entry: entry.sequence):
                if attempt.status in _ACTIVE_ATTEMPT_STATUSES:
                    return item, attempt
        return None

    @staticmethod
    def _single_dispatch_config(smith: SmithConfig, active_count: int) -> SmithConfig:
        return replace(
            smith,
            max_parallel_items=min(smith.max_parallel_items, active_count + 1),
        )

    def _save(self, previous: RunState, desired: RunState) -> None:
        save_state(
            self._state_path,
            desired,
            expected_revision=previous.revision,
            expected_generation=self._generation,
        )

    def _planning_result(self, result: PlanningTickResult) -> RuntimeTickResult:
        if result.next_action is PlanningAction.STOP:
            state = self._load_owned_state()
            if state.terminal_status is not RunTerminalStatus.ACTIVE:
                return self._terminal(state)
            disposition = RuntimeDisposition.WAIT
        elif result.next_action is PlanningAction.WAIT_FOR_SCHEDULER:
            disposition = RuntimeDisposition.WAIT
        else:
            disposition = RuntimeDisposition.CONTINUE
        return RuntimeTickResult(
            disposition=disposition,
            event=RuntimeEvent.PLANNING,
            revision=result.revision,
            reason=f"planning phase {result.phase.value}: {result.event.value}",
            attempt_id=result.attempt_id,
        )

    def _terminal(self, state: RunState) -> RuntimeTickResult:
        return RuntimeTickResult(
            disposition=RuntimeDisposition.TERMINAL,
            event=RuntimeEvent.TERMINAL,
            revision=state.revision,
            reason=state.terminal_reason or f"run ended as {state.terminal_status.value}",
            terminal_status=state.terminal_status,
        )

    def _blocked(
        self,
        state: RunState,
        reason: str,
        *,
        item_id: str | None = None,
        attempt_id: str | None = None,
    ) -> RuntimeTickResult:
        return self._result(
            state,
            RuntimeDisposition.WAIT,
            RuntimeEvent.BLOCKED,
            f"BLOCKED: {reason}",
            item_id=item_id,
            attempt_id=attempt_id,
        )

    @staticmethod
    def _result(
        state: RunState,
        disposition: RuntimeDisposition,
        event: RuntimeEvent,
        reason: str,
        *,
        item_id: str | None = None,
        attempt_id: str | None = None,
    ) -> RuntimeTickResult:
        return RuntimeTickResult(
            disposition=disposition,
            event=event,
            revision=state.revision,
            reason=reason,
            item_id=item_id,
            attempt_id=attempt_id,
        )


def _write_atomic_json_once(path: Path, payload: Mapping[str, object]) -> bool:
    """Atomically create immutable JSON and report whether this call published it."""
    destination = path.expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise RuntimeErrorBase("immutable JSON destination is not a regular file")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if destination.exists() or destination.is_symlink():
            return False
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _load_exact_json(path: Path, expected_keys: set[str]) -> dict[str, object]:
    """Load one regular immutable JSON object with an exact schema."""
    if path.is_symlink() or not path.is_file():
        raise RuntimeErrorBase("immutable JSON artifact is missing or not a regular file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeErrorBase("immutable JSON artifact has an invalid schema")
    return value


def recover_approved_plan(
    workspace: Path,
    task: NormalizedTask,
    state: RunState | None = None,
) -> tuple[PlanDraftOutcome, PlanReviewOutcome]:
    """Recover the admitted plan and approval from immutable planning evidence."""
    root = workspace.expanduser().resolve(strict=True)
    current = state or load_state(root / STATE_FILENAME)
    if current.task_digest != task.digest:
        raise FrozenPlanError("normalized task digest differs from admitted state")
    if not current.planning_attempts:
        raise FrozenPlanError("admitted state has no planning evidence")
    reviewer = current.planning_attempts[-1]
    if (
        reviewer.role is not Role.PLAN_REVIEWER
        or reviewer.status is not AttemptStatus.VALIDATED
        or reviewer.review_of_attempt_id is None
        or reviewer.reviewed_candidate_digest is None
    ):
        raise FrozenPlanError("final planning attempt is not a validated PlanReviewer verdict")
    draft = next(
        (
            attempt
            for attempt in current.planning_attempts
            if attempt.attempt_id == reviewer.review_of_attempt_id
        ),
        None,
    )
    if (
        draft is None
        or draft.role is not Role.PLAN_DRAFTER
        or draft.status is not AttemptStatus.VALIDATED
    ):
        raise FrozenPlanError("PlanReviewer does not link a validated PlanDrafter attempt")
    draft_response = _load_planning_response(root, current, draft)
    plan = parse_plan_draft(
        draft_response,
        allowed_resource_classes=tuple(
            resource.name for resource in task.execution.slurm.smith.resource_classes
        ),
        allowed_path_roots=_allowed_path_roots(task),
        workflow_mode=current.workflow_mode.value,
        target_features=task.target.features,
    )
    if not isinstance(plan, PlanDraftOutcome):
        raise FrozenPlanError("admitted PlanDrafter evidence is BLOCKED, not DRAFTED")
    if (
        draft.candidate_digest != plan.digest
        or reviewer.candidate_digest != plan.digest
        or reviewer.reviewed_candidate_digest != plan.digest
    ):
        raise FrozenPlanError("planning evidence is not pinned to one frozen plan digest")
    review_response = _load_planning_response(root, current, reviewer)
    review = parse_plan_review(review_response, expected_plan_digest=plan.digest)
    if review.outcome is not PlanReviewDecision.ACCEPT:
        raise FrozenPlanError("final PlanReviewer verdict did not approve the frozen plan")
    return plan, review


def _load_planning_response(
    workspace: Path,
    state: RunState,
    attempt: AttemptRecord,
) -> str:
    attempt_dir = workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id
    result_path = worker_output_directory(attempt_dir) / _ROLE_RESULT_FILENAME
    if result_path.is_symlink() or not result_path.is_file():
        raise FrozenPlanError(f"planning result is missing for {attempt.attempt_id!r}")
    if digest_file(result_path) != attempt.result_digest:
        raise FrozenPlanError(f"planning result digest drift for {attempt.attempt_id!r}")
    value = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or set(value) != set(RoleProcessResult.__dataclass_fields__):
        raise FrozenPlanError("planning role result has an invalid schema")
    result = RoleProcessResult(**value)
    spec = load_role_spec(attempt_dir / _ROLE_INPUT_FILENAME)
    expected = (
        spec.schema_version,
        spec.run_id,
        spec.task_digest,
        spec.item_id,
        spec.attempt_id,
        spec.prompt_id,
        spec.role,
        spec.profile,
    )
    actual = (
        result.schema_version,
        result.run_id,
        result.task_digest,
        result.item_id,
        result.attempt_id,
        result.prompt_id,
        result.role,
        result.profile,
    )
    if actual != expected or result.run_id != state.run_id:
        raise FrozenPlanError("planning role result identity differs from immutable input")
    if hashlib.sha256(result.response.encode("utf-8")).hexdigest() != result.response_digest:
        raise FrozenPlanError("planning role response digest does not match response bytes")
    return result.response


def _allowed_path_roots(task: NormalizedTask) -> tuple[str, ...]:
    family = task.target.family
    return (
        "tensorrt_llm/_torch/modeling_v2/catalog",
        f"tensorrt_llm/_torch/modeling_v2/models/{family}",
        "tensorrt_llm/_torch/modeling_v2/_router_index.py",
        "tests/unittest/_torch/modeling_v2",
        "tests/integration/defs/accuracy",
        "tests/integration/test_lists/test-db",
    )


def _proposal(plan: PlanDraftOutcome, item_id: str) -> WorkItemProposal:
    for proposal in plan.items:
        if proposal.item_id == item_id:
            return proposal
    raise FrozenPlanError(f"frozen plan has no WorkItem {item_id!r}")


def _failed_item_status(status: WorkItemStatus) -> WorkItemStatus:
    """Choose a legal terminal failure state for the current domain phase."""
    if status is WorkItemStatus.CANDIDATE_READY:
        return WorkItemStatus.REJECTED
    if status in {
        WorkItemStatus.PLANNED,
        WorkItemStatus.READY,
        WorkItemStatus.CODING,
        WorkItemStatus.REVIEWING,
        WorkItemStatus.INTEGRATING,
    }:
        return WorkItemStatus.BLOCKED
    raise RuntimeErrorBase(f"cannot map an attempt failure from WorkItem state {status.value!r}")


def _goal(state: RunState, goal_id: str) -> GoalRecord:
    for goal in state.goals:
        if goal.goal_id == goal_id:
            return goal
    raise FrozenPlanError(f"authoritative state has no Goal {goal_id!r}")


def _replace_goal(state: RunState, desired: GoalRecord) -> RunState:
    if not any(goal.goal_id == desired.goal_id for goal in state.goals):
        raise FrozenPlanError(f"cannot replace unknown Goal {desired.goal_id!r}")
    return replace(
        state,
        goals=tuple(desired if goal.goal_id == desired.goal_id else goal for goal in state.goals),
        revision=state.revision + 1,
    )


def _replace_stage(state: RunState, desired: StageRecord) -> RunState:
    if not any(stage.stage_id == desired.stage_id for stage in state.stages):
        raise FrozenPlanError(f"cannot replace unknown Stage {desired.stage_id!r}")
    return replace(
        state,
        stages=tuple(
            desired if stage.stage_id == desired.stage_id else stage for stage in state.stages
        ),
        revision=state.revision + 1,
    )


def _replace_item_without_revision(state: RunState, desired: WorkItemRecord) -> RunState:
    """Reconstruct the exact state immediately before one persisted action."""
    if state.revision < 1:
        raise FrozenPlanError("persisted action has no preceding state revision")
    if not any(item.item_id == desired.item_id for item in state.items):
        raise FrozenPlanError(f"cannot reconstruct unknown WorkItem {desired.item_id!r}")
    return replace(
        state,
        items=tuple(desired if item.item_id == desired.item_id else item for item in state.items),
        revision=state.revision - 1,
    )


def _state_job(identity: JobIdentity) -> JobReference:
    return JobReference(
        identity.job_id,
        array_task_id=identity.array_task_id,
        cluster=identity.cluster,
    )


def _scheduler_job(reference: JobReference) -> JobIdentity:
    return JobIdentity(
        reference.job_id,
        array_task_id=reference.array_task_id,
        cluster=reference.cluster,
    )


__all__ = [
    "AttemptExecutionFactory",
    "CollectiveGateExecutionAdapter",
    "ControllerRuntime",
    "FrozenPlanError",
    "ProductionRuntimeFactory",
    "RuntimeCallbacks",
    "RuntimeDisposition",
    "RuntimeDomainInputResolver",
    "RuntimeDomainInputs",
    "RuntimeErrorBase",
    "RuntimeEvent",
    "RuntimeTickResult",
    "SmithMaterializer",
    "recover_approved_plan",
    "resolve_runtime_domain_inputs",
]
