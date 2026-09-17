# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for bounded Staircase controller runtime composition."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase.common.artifacts import (
    INPUT_FILENAME,
    ResultExpectation,
    WorkerInputManifest,
    WorkerResultManifest,
    WorkerResultStatus,
    describe_evidence,
    load_input_manifest,
    publish_result,
    write_input_manifest,
)
from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialDescriptor,
    NoCredentialBroker,
)
from agent_flow.workflows.staircase.common.gates import (
    AccuracyCriteria,
    EvidenceScope,
    build_gate_suite,
)
from agent_flow.workflows.staircase.common.isolation import digest_metadata_free_tree
from agent_flow.workflows.staircase.common.launch_policy import AgentLaunchPolicy
from agent_flow.workflows.staircase.common.outcomes import (
    PlanDraftOutcome,
    PlanReviewDecision,
    PlanReviewOutcome,
    parse_plan_draft,
)
from agent_flow.workflows.staircase.common.placement import (
    HorizontalPlacementContract,
    publish_worker_placement,
)
from agent_flow.workflows.staircase.common.runners import RoleProcessResult, RoleProcessSpec
from agent_flow.workflows.staircase.common.slurm import (
    FakeScheduler,
    InternalCommand,
    InternalEntrypoint,
    JobIdentity,
    JobOwnership,
    JobStatus,
    ResourceRequest,
)
from agent_flow.workflows.staircase.controller.attempts import AttemptExecution
from agent_flow.workflows.staircase.controller.domain import GateExecutionContract
from agent_flow.workflows.staircase.controller.runtime import (
    ControllerRuntime,
    FrozenPlanError,
    ProductionRuntimeFactory,
    RuntimeCallbacks,
    RuntimeDisposition,
    RuntimeDomainInputs,
    RuntimeEvent,
    recover_approved_plan,
    resolve_runtime_domain_inputs,
)
from agent_flow.workflows.staircase.state import (
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
    initialize_state,
    load_state,
    save_state,
)
from agent_flow.workflows.staircase.task_schema import (
    AccuracyGate,
    AgentExecutionConfig,
    AgentWorkerConfig,
    CertificationConfig,
    CertificationMode,
    ContainerLaunchMode,
    ControllerConfig,
    CredentialBrokerPolicy,
    DeliveryConfig,
    ExecutionConfig,
    GatesConfig,
    MountConfig,
    NetworkEnforcement,
    NormalizedTask,
    OverrideBounds,
    ParallelMapping,
    ReferenceConfig,
    RepositoryConfig,
    ResourceClass,
    RetryPolicy,
    RoleEnvironmentPolicy,
    RoleNetworkPolicy,
    SlurmConfig,
    SlurmRole,
    SlurmRoleClass,
    SmithConfig,
    TargetConfig,
)
from agent_flow.workflows.staircase.tuning.artifacts import (
    PROMOTION_DECISION_FILENAME,
    load_promotion_decision_artifact,
)
from agent_flow.workflows.staircase.tuning.contracts import TuningHypothesis
from agent_flow.workflows.staircase.tuning.workflow import PromotionAction

_TASK_DIGEST = "a" * 64


class OwnershipMismatchScheduler:
    """Expose scheduler state while denying immutable-token ownership."""

    def __init__(self, delegate: FakeScheduler) -> None:
        self._delegate = delegate

    def observe(self, identity: JobIdentity):  # type: ignore[no-untyped-def]
        return self._delegate.observe(identity)

    def observe_owned(
        self,
        identity: JobIdentity,
        submission_token: str,
    ):  # type: ignore[no-untyped-def]
        return self._delegate.observe_owned(identity, submission_token)

    def cancel(self, identity: JobIdentity) -> None:
        self._delegate.cancel(identity)

    def lookup_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.lookup_submission(submission_token)

    def verify_ownership(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobOwnership:
        observed = self._delegate.verify_ownership(identity, submission_token)
        return replace(observed, matched=False, reason="injected immutable-token mismatch")


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def _task(tmp_path: Path) -> tuple[NormalizedTask, Path]:
    repository = tmp_path / "repository"
    modeling_v2 = repository / "tensorrt_llm" / "_torch" / "modeling_v2"
    modeling_v2.mkdir(parents=True)
    (modeling_v2 / "README.md").write_text("modeling v2\n", encoding="utf-8")
    catalog = modeling_v2 / "catalog"
    catalog.mkdir()
    norm = catalog / "norm"
    norm.mkdir()
    (norm / "existing.py").write_text("VALUE = 'existing'\n", encoding="utf-8")
    (catalog / "index.yaml").write_text(
        "# ModelingV2 catalog\n"
        "entries:\n"
        "  # --- norm ---\n"
        "  - path: norm/existing.py\n"
        "    impl: torch.ops.trtllm.existing\n"
        "    summary: Existing contract\n",
        encoding="utf-8",
    )
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Staircase Tests")
    _git(repository, "config", "user.email", "staircase@example.com")
    _git(repository, "add", "--", ".")
    _git(repository, "commit", "-q", "-s", "-m", "base")
    base_commit = _git(repository, "rev-parse", "HEAD")

    workspace_root = tmp_path / "workspaces"
    checkpoint = tmp_path / "checkpoint"
    image = tmp_path / "trtllm.sqsh"
    workspace_root.mkdir()
    checkpoint.mkdir()
    image.write_bytes(b"image")
    resources = tuple(
        ResourceClass(name, 1, 1, 0, 2, 1024, 600)
        for name in (
            "coder_analysis",
            "exploratory_probe",
            "deterministic_gate",
            "reviewer_analysis",
            "reviewer_rerun",
        )
    )
    role_resources = {
        SlurmRole.PLAN_DRAFTER: "coder_analysis",
        SlurmRole.PLAN_REVIEWER: "reviewer_analysis",
        SlurmRole.SMITH_CODER: "coder_analysis",
        SlurmRole.ASSEMBLER_CODER: "coder_analysis",
        SlurmRole.TUNER_CODER: "exploratory_probe",
        SlurmRole.REVIEWER: "reviewer_analysis",
        SlurmRole.QA: "reviewer_rerun",
    }
    role_classes = tuple(
        SlurmRoleClass(
            role,
            resource,
            RoleNetworkPolicy.BACKEND_API_ONLY,
            RoleEnvironmentPolicy((), ()),
            CredentialBrokerPolicy("test-broker", (), 3600),
        )
        for role, resource in role_resources.items()
    )
    task = NormalizedTask(
        schema_version=1,
        repository=RepositoryConfig(repository, base_commit, "reject", workspace_root),
        reference=ReferenceConfig(checkpoint, "test", "TestForCausalLM", ()),
        target=TargetConfig(
            "test_family",
            "test_checkpoint",
            100,
            1,
            ParallelMapping(1, 1, 1, 1, 1),
            (),
            "tensorrt_llm/_torch/modeling_v2/models/test_family/routing.py",
            True,
        ),
        gates=GatesConfig(
            AccuracyGate("tests/accuracy.py", "reference", "exact", 0.0),
            (),
            ("tests/boot.py",),
            ("tests/component.py",),
            (),
        ),
        certification=CertificationConfig(CertificationMode.LOCAL, None),
        execution=ExecutionConfig(
            "slurm",
            SlurmConfig(
                controller=ControllerConfig(
                    account="coreai",
                    partition="batch",
                    qos=None,
                    reservation=None,
                    time_limit_seconds=3600,
                    cpus_per_task=2,
                    memory_mib=2048,
                    image=image,
                    scheduler_clients=True,
                    container_launch_mode=ContainerLaunchMode.IN_ALLOCATION_SRUN,
                    mounts=(),
                    environment=(),
                    build_identity="test-build",
                    dispatch_mode="nested_submission",
                    requeue=False,
                    advance_signal_lead_seconds=60,
                    heartbeat_timeout_seconds=120,
                    lease_timeout_seconds=300,
                    orphan_grace_seconds=300,
                ),
                agent_worker=AgentWorkerConfig(
                    image,
                    "test-agent-build",
                    False,
                    NetworkEnforcement.CERTIFIED_WORKER_IMAGE,
                ),
                smith=SmithConfig(
                    max_parallel_items=2,
                    max_nodes_total=2,
                    max_gpus_total=0,
                    resource_classes=resources,
                    per_item_override_bounds=OverrideBounds(2, 2, 1, 8, 8192, 3600),
                    retry_policy=RetryPolicy(1, 1),
                    distinct_nodes_required=False,
                    exclusive=False,
                ),
                role_classes=role_classes,
            ),
            AgentExecutionConfig("codex", "test-model"),
        ),
        delivery=DeliveryConfig("diff_only", None, False, False),
        digest=_TASK_DIGEST,
    )
    workspace = workspace_root / "run"
    workspace.mkdir()
    return task, workspace


def _plan_json() -> str:
    return json.dumps(
        {
            "schema_version": 2,
            "outcome": "DRAFTED",
            "stages": [
                {
                    "stage_id": "catalog-stage",
                    "goal_ids": ["catalog-goal"],
                    "exit_gates": ["catalog accepted"],
                }
            ],
            "goals": [
                {
                    "goal_id": "catalog-goal",
                    "stage_id": "catalog-stage",
                    "capability": "catalog",
                    "item_ids": ["catalog-item"],
                }
            ],
            "items": [
                {
                    "item_id": "catalog-item",
                    "goal_id": "catalog-goal",
                    "kind": "catalog_onboard",
                    "resource_class": "coder_analysis",
                    "execution": {
                        "nodes": 1,
                        "ranks_per_node": 1,
                        "gpus_per_node": 0,
                        "array_element": False,
                        "verdict_scope": "item",
                    },
                    "modifies_files": True,
                    "dependencies": [],
                    "entry_ids": ["test_entry"],
                    "allowed_paths": ["tensorrt_llm/_torch/modeling_v2/catalog/norm/test_entry.py"],
                    "certified_claim_cells": [],
                    "domain_input": None,
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _verify_plan_json() -> str:
    payload = json.loads(_plan_json())
    item = payload["items"][0]
    item["kind"] = "catalog_verify"
    item["modifies_files"] = False
    item["allowed_paths"] = []
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _tuning_plan_json() -> str:
    payload = json.loads(_plan_json())
    item = payload["items"][0]
    item.update(
        {
            "kind": "tune_hypothesis",
            "modifies_files": False,
            "entry_ids": [],
            "allowed_paths": [],
            "resource_class": "exploratory_probe",
            "domain_input": {
                "hypothesis_id": "hypothesis-1",
                "item_id": "catalog-item",
                "statement": "A bounded setting improves matched throughput.",
                "metric": "tokens_per_second",
                "direction": "higher_is_better",
                "changes": [
                    {
                        "name": "attention.backend",
                        "kind": "configuration",
                        "baseline_value": "a",
                        "candidate_value": "b",
                    }
                ],
                "uncertainty": {
                    "minimum_effect": 1.0,
                    "noise_threshold": 0.5,
                    "maximum_combined_uncertainty": 2.0,
                },
            },
        }
    )
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _tuning_measurement_payload(
    *, samples: list[float], gate_passed: bool = True
) -> dict[str, object]:
    return {
        "identity": {
            "checkpoint": "checkpoint@revision",
            "target": "target-sm100",
            "route": {
                "family": "example",
                "architecture": "ExampleForCausalLM",
                "expected_route": "agent_flow.test.example",
                "synthetic_target": True,
            },
            "workload": "fixed-workload",
            "topology": {
                "world_size": 1,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "moe_expert_parallel_size": 1,
                "moe_tensor_parallel_size": 1,
                "attention_data_parallel_size": 1,
            },
            "build": "build-identity",
            "protocol": "paired-three-repetitions",
            "hardware": "one-test-gpu",
        },
        "arm_digest": ("a" if samples[0] < 105 else "b") * 64,
        "curve": {"samples": samples, "uncertainty": 0.25},
        "gates": [
            {
                "name": "correctness",
                "kind": "correctness",
                "passed": gate_passed,
                "evidence_digest": "e" * 64,
            }
        ],
    }


def _two_item_plan_json() -> str:
    payload = json.loads(_plan_json())
    second = dict(payload["items"][0])
    second["item_id"] = "catalog-item-b"
    second["entry_ids"] = ["test_entry_b"]
    second["allowed_paths"] = ["tensorrt_llm/_torch/modeling_v2/catalog/norm/test_entry_b.py"]
    payload["items"].append(second)
    payload["goals"][0]["item_ids"].append("catalog-item-b")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _require_distinct_smith_nodes(task: NormalizedTask) -> NormalizedTask:
    smith = replace(
        task.execution.slurm.smith,
        distinct_nodes_required=True,
        exclusive=True,
    )
    slurm = replace(task.execution.slurm, smith=smith)
    return replace(task, execution=replace(task.execution, slurm=slurm))


def _smith_coder_attempt(
    item_id: str,
    *,
    job_id: str,
    status: AttemptStatus,
    digest_character: str,
) -> AttemptRecord:
    candidate_digest = digest_character * 64 if status is AttemptStatus.VALIDATED else None
    return AttemptRecord(
        f"{item_id}.0001.role",
        item_id,
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        status=status,
        profile=DomainProfile.SMITH,
        submission_token=f"token-{item_id}-coder",
        job=JobReference(job_id),
        result_digest="c" * 64 if status is AttemptStatus.VALIDATED else None,
        candidate_digest=candidate_digest,
        terminal_reason="injected terminal worker failure"
        if status is AttemptStatus.FAILED
        else None,
    )


def _approved_smith_item(item_id: str, *, job_id: str, digest_character: str) -> WorkItemRecord:
    coder = _smith_coder_attempt(
        item_id,
        job_id=job_id,
        status=AttemptStatus.VALIDATED,
        digest_character=digest_character,
    )
    candidate_digest = digest_character * 64
    analysis = AttemptRecord(
        f"{item_id}.0002.analysis",
        item_id,
        2,
        Role.REVIEWER,
        AttemptKind.REVIEWER_ANALYSIS,
        1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        submission_token=f"token-{item_id}-analysis",
        job=JobReference(str(int(job_id) + 1000)),
        result_digest="d" * 64,
        candidate_digest=candidate_digest,
        review_of_attempt_id=coder.attempt_id,
        reviewed_candidate_digest=candidate_digest,
    )
    rerun = AttemptRecord(
        f"{item_id}.0003.rerun",
        item_id,
        3,
        Role.REVIEWER,
        AttemptKind.REVIEWER_RERUN,
        1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        submission_token=f"token-{item_id}-rerun",
        job=JobReference(str(int(job_id) + 2000)),
        result_digest="e" * 64,
        candidate_digest=candidate_digest,
        review_of_attempt_id=coder.attempt_id,
        reviewed_candidate_digest=candidate_digest,
    )
    return WorkItemRecord(
        item_id,
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.CATALOG_ONBOARD,
        DomainProfile.SMITH,
        status=WorkItemStatus.APPROVED,
        attempts=(coder, analysis, rerun),
        candidate_attempt_id=coder.attempt_id,
        candidate_digest=candidate_digest,
        reviewer_attempt_id=rerun.attempt_id,
    )


def _publish_smith_placement(
    workspace: Path,
    state: RunState,
    attempt: AttemptRecord,
    *,
    hostname: str,
    scheduler: FakeScheduler,
    observed_job_id: str | None = None,
) -> None:
    attempt_dir = workspace / "items" / attempt.item_id / "attempts" / f"{attempt.sequence:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    manifest = WorkerInputManifest(
        state.run_id,
        attempt.item_id,
        attempt.attempt_id,
        state.task_digest,
        attempt.generation,
        attempt.role,
        attempt.profile,
        str(workspace / "candidates" / attempt.item_id / attempt.attempt_id),
    )
    publish_worker_placement(
        attempt_dir,
        manifest,
        HorizontalPlacementContract(True, True),
        environment={
            "SLURM_JOB_ID": observed_job_id or attempt.job.job_id,  # type: ignore[union-attr]
            "SLURMD_NODENAME": hostname,
            "SLURM_JOB_NODELIST": hostname,
            "SLURM_NNODES": "1",
        },
        hostname=hostname,
    )
    assert attempt.job is not None and attempt.submission_token is not None
    scheduler.set_placement(
        JobIdentity(
            attempt.job.job_id,
            array_task_id=attempt.job.array_task_id,
            cluster=attempt.job.cluster,
        ),
        attempt.submission_token,
        hostname,
    )


def _publish_all_smith_role_placements(
    workspace: Path,
    state: RunState,
    item: WorkItemRecord,
    *,
    hostname: str,
    scheduler: FakeScheduler,
    spoof_attempt_id: str | None = None,
) -> None:
    for attempt in item.attempts:
        if attempt.role not in {Role.CODER, Role.REVIEWER}:
            continue
        _publish_smith_placement(
            workspace,
            state,
            attempt,
            hostname=hostname,
            scheduler=scheduler,
            observed_job_id=("999999" if attempt.attempt_id == spoof_attempt_id else None),
        )


def _fake_scheduler(runtime: ControllerRuntime) -> FakeScheduler:
    scheduler = runtime._scheduler  # noqa: SLF001
    assert isinstance(scheduler, FakeScheduler)
    return scheduler


def _horizontal_placement_runtime(
    tmp_path: Path,
    *,
    peer: WorkItemRecord,
    reverse_items: bool = False,
) -> tuple[
    NormalizedTask,
    Path,
    RunState,
    ControllerRuntime,
    PlanDraftOutcome,
    PlanReviewOutcome,
]:
    task, workspace = _task(tmp_path)
    task = _require_distinct_smith_nodes(task)
    current = _approved_smith_item("catalog-item", job_id="101", digest_character="a")
    items = (peer, current) if reverse_items else (current, peer)
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=_two_item_plan_json(),
        items=items,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    state = load_state(workspace / STATE_FILENAME)
    plan, review = recover_approved_plan(workspace, task, state)
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    )
    return task, workspace, state, runtime, plan, review


def _write_role_evidence(
    workspace: Path,
    *,
    run_id: str,
    attempt_id: str,
    role: str,
    response: str,
) -> tuple[str, str]:
    attempt_dir = workspace / "items" / "__planning__" / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "output").mkdir()
    role_input = attempt_dir / "role-input.json"
    role_result = attempt_dir / "output" / "role-result.json"
    spec = RoleProcessSpec(
        schema_version=1,
        run_id=run_id,
        task_digest=_TASK_DIGEST,
        generation=1,
        item_id="planning",
        attempt_id=attempt_id,
        prompt_id=attempt_id,
        role=role,  # type: ignore[arg-type]
        profile="planner",
        backend_kind="codex",
        model="test-model",
        source_root=str(workspace.parent.parent / "repository"),
        cwd=str(workspace.parent.parent / "repository"),
        result_path=str(role_result),
        system_prompt="system",
        prompt="prompt",
        credential_descriptor=CredentialDescriptor.no_credentials(
            CredentialBinding(
                run_id=run_id,
                item_id="planning",
                attempt_id=attempt_id,
                task_digest=_TASK_DIGEST,
                generation=1,
                backend_kind="codex",
            )
        ).to_public_dict(),
        launch_policy=AgentLaunchPolicy.create(
            image=str(workspace.parent.parent / "trtllm.sqsh"),
            build_identity="test-agent-build",
        ).to_public_dict(),
    )
    role_input.write_text(json.dumps(asdict(spec), sort_keys=True) + "\n", encoding="utf-8")
    response_digest = hashlib.sha256(response.encode("utf-8")).hexdigest()
    result = RoleProcessResult(
        schema_version=1,
        run_id=run_id,
        task_digest=_TASK_DIGEST,
        generation=1,
        item_id="planning",
        attempt_id=attempt_id,
        prompt_id=attempt_id,
        role=role,  # type: ignore[arg-type]
        profile="planner",
        response=response,
        response_digest=response_digest,
    )
    role_result.write_text(json.dumps(asdict(result), sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(role_result.read_bytes()).hexdigest(), response_digest


def _initialize_admitted(
    task: NormalizedTask,
    workspace: Path,
    *,
    item_status: WorkItemStatus = WorkItemStatus.PLANNED,
    goal_status: HierarchyStatus = HierarchyStatus.PLANNED,
    stage_status: StageStatus = StageStatus.PLANNED,
    attempts: tuple[AttemptRecord, ...] = (),
    terminal_reason: str | None = None,
) -> None:
    plan_response = _plan_json()
    item = WorkItemRecord(
        "catalog-item",
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.CATALOG_ONBOARD,
        DomainProfile.SMITH,
        status=item_status,
        attempts=attempts,
        terminal_reason=terminal_reason,
    )
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=plan_response,
        items=(item,),
        goal_status=goal_status,
        stage_status=stage_status,
    )


def _initialize_custom_admitted(
    task: NormalizedTask,
    workspace: Path,
    *,
    plan_response: str,
    items: tuple[WorkItemRecord, ...],
    goal_status: HierarchyStatus,
    stage_status: StageStatus,
    workflow_mode: WorkflowMode = WorkflowMode.ONBOARD,
) -> None:
    run_id = "runtime-test-run"
    plan = parse_plan_draft(
        plan_response,
        allowed_resource_classes=tuple(
            resource.name for resource in task.execution.slurm.smith.resource_classes
        ),
        allowed_path_roots=(
            "tensorrt_llm/_torch/modeling_v2/catalog",
            "tensorrt_llm/_torch/modeling_v2/models/test_family",
            "tensorrt_llm/_torch/modeling_v2/_router_index.py",
            "tests/unittest/_torch/modeling_v2",
            "tests/integration/defs/accuracy",
            "tests/integration/test_lists/test-db",
        ),
        workflow_mode=workflow_mode.value,
        target_features=task.target.features,
    )
    draft_result_digest, _ = _write_role_evidence(
        workspace,
        run_id=run_id,
        attempt_id="plan-draft-0001",
        role="plan_drafter",
        response=plan_response,
    )
    review_response = json.dumps(
        {
            "schema_version": 2,
            "outcome": "ACCEPT",
            "plan_digest": plan.digest,
            "corrections": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    review_result_digest, _ = _write_role_evidence(
        workspace,
        run_id=run_id,
        attempt_id="plan-review-0002",
        role="plan_reviewer",
        response=review_response,
    )
    planning_attempts = (
        AttemptRecord(
            "plan-draft-0001",
            "__planning__",
            1,
            Role.PLAN_DRAFTER,
            AttemptKind.ROLE,
            1,
            status=AttemptStatus.VALIDATED,
            submission_token="draft-token",
            job=JobReference("101"),
            result_digest=draft_result_digest,
            candidate_digest=plan.digest,
        ),
        AttemptRecord(
            "plan-review-0002",
            "__planning__",
            2,
            Role.PLAN_REVIEWER,
            AttemptKind.REVIEWER_ANALYSIS,
            1,
            status=AttemptStatus.VALIDATED,
            submission_token="review-token",
            job=JobReference("102"),
            result_digest=review_result_digest,
            candidate_digest=plan.digest,
            review_of_attempt_id="plan-draft-0001",
            reviewed_candidate_digest=plan.digest,
        ),
    )
    initialize_state(
        workspace / STATE_FILENAME,
        RunState(
            run_id=run_id,
            task_digest=task.digest,
            base_commit=task.repository.base_commit,
            generation=1,
            stages=(StageRecord("catalog-stage", ("catalog-goal",), stage_status),),
            goals=(
                GoalRecord(
                    "catalog-goal",
                    "catalog-stage",
                    tuple(item.item_id for item in items),
                    goal_status,
                ),
            ),
            items=items,
            planning_attempts=planning_attempts,
            workflow_mode=workflow_mode,
        ),
    )


def _callbacks(
    heartbeats: list[str],
    *,
    cancel: str | None = None,
    stop: bool = False,
    requeue: bool = False,
    waiting: str | None = None,
) -> RuntimeCallbacks:
    return RuntimeCallbacks(
        heartbeat=lambda: heartbeats.append("heartbeat"),
        cancellation_reason=lambda: cancel,
        stop_dispatch_requested=lambda: stop,
        requeue_requested=lambda: requeue,
        waiting_input_reason=lambda: waiting,
    )


def _gate_specs(task: NormalizedTask):  # type: ignore[no-untyped-def]
    return build_gate_suite(
        entry_gpu_tests=task.gates.component_tests,
        collective_tests=task.gates.collective_tests,
        boot_tests=task.gates.boot_tests,
        accuracy=AccuracyCriteria(
            task.gates.accuracy.selector,
            task.gates.accuracy.reference,
            task.gates.accuracy.protocol,
            task.gates.accuracy.tolerance,
        ),
        feature_signal_tests=task.gates.feature_signal_tests,
        base_environment=dict(task.execution.slurm.controller.environment),
    )


def _publish_approved_attempt_result(
    workspace: Path,
    task: NormalizedTask,
    state: RunState,
    attempt: AttemptRecord,
) -> None:
    attempt_root = workspace / "items" / attempt.item_id / "attempts" / f"{attempt.sequence:04d}"
    attempt_dir = attempt_root / "output"
    input_manifest, input_digest = load_input_manifest(attempt_root / INPUT_FILENAME)
    item = state.item(attempt.item_id)
    if attempt.role is Role.CODER:
        changed_paths: list[str] = []
        index_delta: dict[str, object] | None = None
        if item.kind is WorkItemKind.CATALOG_ONBOARD:
            changed_path = input_manifest.allowed_paths[0]
            catalog_path = changed_path.removeprefix("tensorrt_llm/_torch/modeling_v2/catalog/")
            entry_id = Path(catalog_path).stem
            candidate_path = (
                workspace / "candidates" / attempt.item_id / attempt.attempt_id / changed_path
            )
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(f"VALUE = {entry_id!r}\n", encoding="utf-8")
            changed_paths.append(changed_path)
            index_delta = {
                "item_id": attempt.item_id,
                "row": {
                    "entry_id": entry_id,
                    "path": catalog_path,
                    "implementation": f"torch.ops.trtllm.{entry_id}",
                    "summary": "Test entry contract",
                },
                "certification_cells": [],
                "expected_row_sha256": None,
            }
        payload: dict[str, object] = {
            "schema_version": 1,
            "result_kind": "coder",
            "changed_paths": changed_paths,
            "index_delta": index_delta,
        }
        evidence = ()
        reviewed_candidate_digest = None
        candidate_overlay = workspace / "candidates" / attempt.item_id / attempt.attempt_id
        scanned_overlay_digest = digest_metadata_free_tree(
            candidate_overlay,
            secret_scan_paths=changed_paths,
        )
    else:
        assert item.candidate_attempt_id is not None
        assert item.candidate_digest is not None
        evidence_path = attempt_dir / "evidence" / "result.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps({"attempt_id": attempt.attempt_id}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        evidence = (describe_evidence(attempt_dir, "evidence/result.json"),)
        evidence_payload = [asdict(entry) for entry in evidence]
        reviewed_candidate_digest = item.candidate_digest
        candidate_fields = {
            "candidate_attempt_id": item.candidate_attempt_id,
            "candidate_digest": item.candidate_digest,
        }
        if attempt.kind is AttemptKind.DETERMINISTIC_GATE:
            gate_attempts = [
                entry for entry in item.attempts if entry.kind is AttemptKind.DETERMINISTIC_GATE
            ]
            gate_index = gate_attempts.index(attempt)
            gate = _gate_specs(task)[gate_index]
            runtime = input_manifest.payload["runtime"]
            assert isinstance(runtime, dict)
            receipt_input = runtime["receipt"]
            assert isinstance(receipt_input, dict)
            scope = receipt_input["scope"]
            placements = (
                []
                if scope == EvidenceScope.CPU_STATIC.value
                else [{"rank": 0, "node": "test-node", "local_rank": 0}]
            )
            payload = {
                "schema_version": 1,
                "result_kind": "deterministic_gate",
                **candidate_fields,
                "certification_mode": receipt_input["certification_mode"],
                "receipt": {
                    "gate_id": gate.gate_id,
                    "purpose": gate.purpose.value,
                    "scope": scope,
                    "placements": placements,
                    "product_rank_body": receipt_input["product_rank_body"],
                    "passed": True,
                    "accuracy": receipt_input["accuracy"],
                },
                "evidence": evidence_payload,
            }
        elif attempt.kind in {
            AttemptKind.REVIEWER_ANALYSIS,
            AttemptKind.REVIEWER_RERUN,
        }:
            payload = {
                "schema_version": 1,
                "result_kind": attempt.kind.value,
                **candidate_fields,
                "verdict": "approve",
                "findings": [],
                "evidence": evidence_payload,
            }
        else:
            assert attempt.kind is AttemptKind.QA
            gate_results = [
                {
                    "gate_attempt_id": entry.attempt_id,
                    "result_digest": entry.result_digest,
                }
                for entry in item.attempts
                if entry.kind is AttemptKind.DETERMINISTIC_GATE and entry.result_digest is not None
            ]
            payload = {
                "schema_version": 1,
                "result_kind": "qa",
                **candidate_fields,
                "verdict": "approve",
                "findings": [],
                "gate_results": gate_results,
                "evidence": evidence_payload,
            }
    publish_result(
        attempt_dir,
        WorkerResultManifest(
            state.run_id,
            attempt.item_id,
            attempt.attempt_id,
            state.task_digest,
            attempt.generation,
            input_digest,
            WorkerResultStatus.SUCCEEDED,
            f"approved {attempt.kind.value}",
            evidence=evidence,
            candidate_digest=(scanned_overlay_digest if attempt.role is Role.CODER else None),
            reviewed_candidate_digest=reviewed_candidate_digest,
            payload=payload,  # type: ignore[arg-type]
        ),
        input_path=attempt_root / INPUT_FILENAME,
    )


def _publish_tuner_result(
    workspace: Path,
    state: RunState,
    attempt: AttemptRecord,
    *,
    candidate_gate_passed: bool,
) -> None:
    attempt_root = workspace / "items" / attempt.item_id / "attempts" / f"{attempt.sequence:04d}"
    attempt_dir = attempt_root / "output"
    _input_manifest, input_digest = load_input_manifest(attempt_root / INPUT_FILENAME)
    evidence_path = attempt_dir / "evidence" / "measurement.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text('{"measurement":true}\n', encoding="utf-8")
    evidence = (describe_evidence(attempt_dir, "evidence/measurement.json"),)
    candidate_overlay = workspace / "candidates" / attempt.item_id / attempt.attempt_id
    scanned_overlay_digest = digest_metadata_free_tree(candidate_overlay)
    publish_result(
        attempt_dir,
        WorkerResultManifest(
            state.run_id,
            attempt.item_id,
            attempt.attempt_id,
            state.task_digest,
            attempt.generation,
            input_digest,
            WorkerResultStatus.SUCCEEDED,
            "measured matched tuning arms",
            evidence=evidence,
            candidate_digest=scanned_overlay_digest,
            payload={
                "schema_version": 1,
                "result_kind": "tuner_measurement",
                "baseline": _tuning_measurement_payload(samples=[99.0, 100.0, 101.0]),
                "candidate": _tuning_measurement_payload(
                    samples=[109.0, 110.0, 111.0],
                    gate_passed=candidate_gate_passed,
                ),
                "evidence": [asdict(entry) for entry in evidence],
            },
        ),
        input_path=attempt_root / INPUT_FILENAME,
    )


def test_recovers_only_digest_pinned_approved_plan(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    _initialize_admitted(task, workspace)
    state = load_state(workspace / STATE_FILENAME)

    plan, review = recover_approved_plan(workspace, task, state)

    assert plan.items[0].item_id == "catalog-item"
    assert review.plan_digest == plan.digest
    result_path = (
        workspace
        / "items"
        / "__planning__"
        / "attempts"
        / "plan-draft-0001"
        / "output"
        / "role-result.json"
    )
    result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(FrozenPlanError, match="digest drift"):
        recover_approved_plan(workspace, task, state)


def test_tick_marks_one_item_ready_and_heartbeats(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    _initialize_admitted(task, workspace)
    heartbeats: list[str] = []
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks(heartbeats),
    )

    result = runtime.tick()

    assert result.event is RuntimeEvent.ITEM_READY
    assert result.disposition is RuntimeDisposition.CONTINUE
    assert (
        load_state(workspace / STATE_FILENAME).item("catalog-item").status is WorkItemStatus.READY
    )
    assert heartbeats == ["heartbeat"]


def test_runtime_rejects_legacy_admitted_kind_without_production_adapter(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    payload = json.loads(_plan_json())
    payload["items"][0].update(
        {
            "kind": "search",
            "modifies_files": False,
            "entry_ids": [],
            "allowed_paths": [],
        }
    )
    plan_response = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    item = WorkItemRecord(
        "catalog-item",
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.SEARCH,
        DomainProfile.SMITH,
        status=WorkItemStatus.READY,
    )
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=plan_response,
        items=(item,),
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    scheduler = FakeScheduler()
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([]),
    )

    result = runtime.tick()

    assert result.event is RuntimeEvent.BLOCKED
    assert "production WorkItem adapter invariant" in result.reason
    assert scheduler.submissions == ()


def test_production_factory_requires_and_wires_exact_gate_inputs(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    source_head = _git(task.repository.root, "rev-parse", "HEAD")
    _initialize_admitted(task, workspace)
    plan, _review = recover_approved_plan(workspace, task)
    gates = build_gate_suite(
        entry_gpu_tests=task.gates.component_tests,
        collective_tests=task.gates.collective_tests,
        boot_tests=task.gates.boot_tests,
        accuracy=AccuracyCriteria(
            task.gates.accuracy.selector,
            task.gates.accuracy.reference,
            task.gates.accuracy.protocol,
            task.gates.accuracy.tolerance,
        ),
        feature_signal_tests=task.gates.feature_signal_tests,
        base_environment=dict(task.execution.slurm.controller.environment),
    )
    gate_resource = task.execution.slurm.smith.resource_class("deterministic_gate")
    inputs = RuntimeDomainInputs(
        gate_executions=tuple(
            (
                gate.gate_id,
                GateExecutionContract(
                    EvidenceScope.CPU_STATIC,
                    0,
                    None,
                    None,
                    False,
                ),
            )
            for gate in gates
        ),
        gate_resources=tuple((gate.gate_id, gate_resource) for gate in gates),
        qa_required_items=frozenset(),
        role_evidence_paths=(
            ("coder", ("reports/coder.json",)),
            ("qa", ("reports/qa.json",)),
            ("reviewer", ("reports/reviewer.json",)),
        ),
        task_digest=task.digest,
        plan_digest=plan.digest,
    )

    runtime = ProductionRuntimeFactory(inputs, credential_broker=NoCredentialBroker())(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    )

    assert isinstance(runtime, ControllerRuntime)
    assert runtime._git.repository == workspace / "delivery" / "diff-repository"  # noqa: SLF001
    assert _git(task.repository.root, "rev-parse", "HEAD") == source_head
    signed_task = replace(
        task,
        delivery=DeliveryConfig(
            "signed_commits",
            "staircase/runtime-test/delivery",
            False,
            False,
        ),
    )
    signed_runtime = ProductionRuntimeFactory(inputs, credential_broker=NoCredentialBroker())(
        workspace=workspace,
        task=signed_task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    )
    assert signed_runtime._git.repository == workspace / "delivery" / "repository"  # noqa: SLF001
    assert _git(task.repository.root, "rev-parse", "HEAD") == source_head
    with pytest.raises(RuntimeError, match="exactly cover"):
        ProductionRuntimeFactory(
            replace(inputs, gate_executions=(), gate_resources=()),
            credential_broker=NoCredentialBroker(),
        )(
            workspace=workspace,
            task=task,
            scheduler=FakeScheduler(),
            generation=1,
            callbacks=_callbacks([]),
        )


def test_default_production_resolver_binds_review_and_exact_gate_subrequests(
    tmp_path: Path,
) -> None:
    task, _workspace = _task(tmp_path)
    gate_resource = replace(
        task.execution.slurm.smith.resource_class("deterministic_gate"),
        gpus_per_node=1,
    )
    resources = tuple(
        gate_resource if resource.name == "deterministic_gate" else resource
        for resource in task.execution.slurm.smith.resource_classes
    )
    task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(
                task.execution.slurm,
                smith=replace(
                    task.execution.slurm.smith,
                    max_gpus_total=1,
                    resource_classes=resources,
                ),
            ),
        ),
    )
    gates = build_gate_suite(
        entry_gpu_tests=task.gates.component_tests,
        collective_tests=task.gates.collective_tests,
        boot_tests=task.gates.boot_tests,
        accuracy=AccuracyCriteria(
            task.gates.accuracy.selector,
            task.gates.accuracy.reference,
            task.gates.accuracy.protocol,
            task.gates.accuracy.tolerance,
        ),
        feature_signal_tests=task.gates.feature_signal_tests,
        base_environment=dict(task.execution.slurm.controller.environment),
    )
    plan_payload = json.loads(_plan_json())
    plan_payload["stages"][0]["exit_gates"] = [gate.gate_id for gate in gates]
    plan = parse_plan_draft(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")),
        allowed_resource_classes=tuple(
            resource.name for resource in task.execution.slurm.smith.resource_classes
        ),
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2/catalog",),
        workflow_mode=WorkflowMode.ONBOARD.value,
        target_features=task.target.features,
    )
    review = PlanReviewOutcome(PlanReviewDecision.ACCEPT, plan.digest, ())

    with pytest.raises(ValueError, match="CredentialBroker"):
        resolve_runtime_domain_inputs(
            task,
            WorkflowMode.ONBOARD,
            plan,
            review,
            (("OPENAI_API_KEY", "test-secret"),),
        )
    resolved = resolve_runtime_domain_inputs(
        task,
        WorkflowMode.ONBOARD,
        plan,
        review,
        (),
    )

    assert resolved.task_digest == task.digest
    assert resolved.plan_digest == plan.digest
    assert tuple(gate_id for gate_id, _contract in resolved.gate_executions) == tuple(
        gate.gate_id for gate in gates
    )
    assert resolved.gate_executions[0][1].scope is EvidenceScope.CPU_STATIC
    assert resolved.gate_resources[0][1].gpus_per_node == 0
    assert all(
        contract.scope is EvidenceScope.SINGLE_GPU_PRODUCT
        for _gate_id, contract in resolved.gate_executions[2:]
    )
    assert all(resource.gpus_per_node == 1 for _gate_id, resource in resolved.gate_resources[2:])
    assert resolved.evidence_paths_for(Role.CODER) == ("reports/coder.json",)
    assert resolved.evidence_paths_for(Role.REVIEWER) == ("reports/reviewer.json",)
    assert resolved.evidence_paths_for(Role.QA) == ("reports/qa.json",)


def test_production_factory_defers_resolution_during_first_run_planning(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    initialize_state(
        workspace / STATE_FILENAME,
        RunState(
            run_id="planning-only-run",
            task_digest=task.digest,
            base_commit=task.repository.base_commit,
            generation=1,
        ),
    )

    runtime = ProductionRuntimeFactory(credential_broker=NoCredentialBroker())(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    )

    assert isinstance(runtime, ControllerRuntime)


def test_stop_dispatch_and_missing_transport_are_explicit(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.READY,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    stopped = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([], stop=True),
    ).tick()
    assert stopped.event is RuntimeEvent.STOP_DISPATCH
    assert load_state(workspace / STATE_FILENAME).item("catalog-item").attempts == ()

    blocked = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    ).tick()
    assert blocked.event is RuntimeEvent.BLOCKED
    assert "generic worker execution adapter" in blocked.reason
    assert load_state(workspace / STATE_FILENAME).item("catalog-item").attempts == ()


def test_outer_requeue_and_waiting_input_are_typed_without_mutation(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    _initialize_admitted(task, workspace)

    requeue = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([], requeue=True),
    ).tick()
    waiting = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([], waiting="operator response required"),
    ).tick()

    assert requeue.disposition is RuntimeDisposition.REQUEUE
    assert waiting.disposition is RuntimeDisposition.WAITING_INPUT
    assert load_state(workspace / STATE_FILENAME).revision == 0


def test_attempt_intent_is_persisted_before_scheduler_submit(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    attempt = AttemptRecord(
        "catalog-item.0001.role",
        "catalog-item",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        profile=DomainProfile.SMITH,
        resource_class="coder_analysis",
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.CODING,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        attempts=(attempt,),
    )
    scheduler = FakeScheduler()

    def execution_factory(
        state: RunState, proposal: object, current: AttemptRecord
    ) -> AttemptExecution:
        del proposal
        attempt_dir = workspace / "items" / current.item_id / "attempts" / "0001"
        input_path = attempt_dir / INPUT_FILENAME
        if input_path.exists():
            _manifest, input_digest = load_input_manifest(input_path)
        else:
            input_digest = write_input_manifest(
                input_path,
                WorkerInputManifest(
                    state.run_id,
                    current.item_id,
                    current.attempt_id,
                    state.task_digest,
                    state.generation,
                    current.role,
                    current.profile,
                    str(workspace),
                ),
            )
        return AttemptExecution(
            resources=ResourceRequest(
                label=current.attempt_id,
                account="coreai",
                partition="batch",
                time_limit="00:10:00",
                output_path=workspace / "logs/%j.out",
                error_path=workspace / "logs/%j.err",
            ),
            command=InternalCommand(InternalEntrypoint.WORKER, input_bundle=input_path),
            submission_token="catalog-attempt-token",
            attempt_dir=attempt_dir,
            expectation=ResultExpectation(
                state.run_id,
                current.item_id,
                current.attempt_id,
                state.task_digest,
                state.generation,
                input_digest,
            ),
            receipt_root=workspace / "receipts",
            quarantine_root=workspace / "quarantine",
        )

    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([]),
        execution_factory=execution_factory,
        smith_materializer=lambda actions: tuple(actions),
    )

    intent = runtime.tick()

    assert intent.event is RuntimeEvent.ATTEMPT_UPDATED
    assert scheduler.submissions == ()
    persisted = load_state(workspace / STATE_FILENAME).item("catalog-item").attempts[0]
    assert persisted.status is AttemptStatus.SUBMITTING

    submitted = runtime.tick()
    assert submitted.event is RuntimeEvent.ATTEMPT_UPDATED
    assert len(scheduler.submissions) == 1
    persisted = load_state(workspace / STATE_FILENAME).item("catalog-item").attempts[0]
    assert persisted.status is AttemptStatus.SUBMITTED


def test_generic_worker_is_materialized_after_reservation_with_frozen_agent(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    mount_root = tmp_path.resolve()
    controller = replace(
        task.execution.slurm.controller,
        mounts=(MountConfig(mount_root, "/staircase-test", False),),
    )
    task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(task.execution.slurm, controller=controller),
        ),
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.READY,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    scheduler = FakeScheduler()
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([]),
        domain_inputs=RuntimeDomainInputs(qa_required_items=frozenset()),
        credential_broker=NoCredentialBroker(),
    )

    reserved = runtime.tick()
    assert reserved.event is RuntimeEvent.ACTION_PERSISTED
    assert not (workspace / "items/catalog-item/attempts/0001/input.json").exists()

    materialized = runtime.tick()
    assert materialized.event is RuntimeEvent.WORKER_ACTION_MATERIALIZED
    input_path = workspace / "items/catalog-item/attempts/0001/input.json"
    manifest, _digest = load_input_manifest(input_path)
    agent_runtime = manifest.payload["runtime"]
    assert isinstance(agent_runtime, dict)
    assert agent_runtime["backend_kind"] == "codex"
    assert agent_runtime["model"] == "test-model"
    descriptor = agent_runtime["credential_descriptor"]
    assert isinstance(descriptor, dict)
    assert descriptor["state"] == "none"
    assert manifest.worktree.startswith("/staircase-test/")
    assert scheduler.submissions == ()

    submitting = runtime.tick()
    assert submitting.event is RuntimeEvent.ATTEMPT_UPDATED
    assert scheduler.submissions == ()

    submitted = runtime.tick()
    assert submitted.event is RuntimeEvent.ATTEMPT_UPDATED
    assert len(scheduler.submissions) == 1
    assert scheduler.submissions[0].environment == ()


def test_generic_submitting_cancellation_certifies_absence_without_submission(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    mount_root = tmp_path.resolve()
    controller = replace(
        task.execution.slurm.controller,
        mounts=(MountConfig(mount_root, "/staircase-test", False),),
    )
    task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(task.execution.slurm, controller=controller),
        ),
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.READY,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    scheduler = FakeScheduler()
    kwargs = {
        "workspace": workspace,
        "task": task,
        "scheduler": scheduler,
        "generation": 1,
        "domain_inputs": RuntimeDomainInputs(qa_required_items=frozenset()),
        "credential_broker": NoCredentialBroker(),
    }
    runtime = ControllerRuntime(callbacks=_callbacks([]), **kwargs)
    runtime.tick()
    runtime.tick()
    runtime.tick()
    persisted = load_state(workspace / STATE_FILENAME).item("catalog-item").attempts[-1]
    assert persisted.status is AttemptStatus.SUBMITTING

    cancelling = ControllerRuntime(
        callbacks=_callbacks([], cancel="operator requested cancellation"),
        **kwargs,
    )
    result = cancelling.tick()

    assert result.event is RuntimeEvent.CANCELLATION_UPDATED
    assert scheduler.submissions == ()
    persisted = load_state(workspace / STATE_FILENAME).item("catalog-item").attempts[-1]
    assert persisted.status is AttemptStatus.CANCELLED
    assert (
        workspace / "receipts" / "submissions" / persisted.attempt_id / "cancellation.json"
    ).is_file()


def test_cancellation_targets_only_exact_persisted_job(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    scheduler = FakeScheduler()
    resources = ResourceRequest(
        label="catalog-item",
        account="coreai",
        partition="batch",
        time_limit="00:10:00",
        output_path=workspace / "logs/%j.out",
        error_path=workspace / "logs/%j.err",
    )
    command = InternalCommand(
        InternalEntrypoint.WORKER,
        input_bundle=workspace / "unused.json",
    )
    identity = scheduler.submit(resources, command, "owned-token")
    scheduler.transition(identity, JobStatus.RUNNING)
    attempt = AttemptRecord(
        "catalog-item.0001.role",
        "catalog-item",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        status=AttemptStatus.RUNNING,
        profile=DomainProfile.SMITH,
        submission_token="owned-token",
        job=JobReference(identity.job_id),
        scheduler_state=JobStatus.RUNNING.value,
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.CODING,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        attempts=(attempt,),
    )
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([], cancel="operator requested cancellation"),
    )

    result = runtime.tick()

    assert result.event is RuntimeEvent.CANCEL_SIGNALLED
    assert scheduler.cancelled == (identity,)


def test_runtime_cancellation_fails_closed_on_token_ownership_mismatch(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    delegate = FakeScheduler()
    resources = ResourceRequest(
        label="catalog-item",
        account="coreai",
        partition="batch",
        time_limit="00:10:00",
        output_path=workspace / "logs/%j.out",
        error_path=workspace / "logs/%j.err",
    )
    command = InternalCommand(
        InternalEntrypoint.WORKER,
        input_bundle=workspace / "unused.json",
    )
    identity = delegate.submit(resources, command, "owned-token")
    delegate.transition(identity, JobStatus.RUNNING)
    attempt = AttemptRecord(
        "catalog-item.0001.role",
        "catalog-item",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        status=AttemptStatus.RUNNING,
        profile=DomainProfile.SMITH,
        submission_token="owned-token",
        job=JobReference(identity.job_id),
        scheduler_state=JobStatus.RUNNING.value,
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.CODING,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        attempts=(attempt,),
    )
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=OwnershipMismatchScheduler(delegate),
        generation=1,
        callbacks=_callbacks([], cancel="operator requested cancellation"),
    )

    result = runtime.tick()

    assert result.event is RuntimeEvent.BLOCKED
    assert "ownership verification failed closed" in result.reason
    assert delegate.cancelled == ()
    assert load_state(workspace / STATE_FILENAME).item("catalog-item").attempts[0].status is (
        AttemptStatus.RUNNING
    )


def test_runtime_cancellation_rejects_reused_scheduler_job_id(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    resources = ResourceRequest(
        label="catalog-item",
        account="coreai",
        partition="batch",
        time_limit="00:10:00",
        output_path=workspace / "logs/%j.out",
        error_path=workspace / "logs/%j.err",
    )
    command = InternalCommand(
        InternalEntrypoint.WORKER,
        input_bundle=workspace / "unused.json",
    )
    recycled = FakeScheduler(first_job_id=10_000)
    identity = recycled.submit(resources, command, "recycled-token")
    recycled.transition(identity, JobStatus.RUNNING)
    attempt = AttemptRecord(
        "catalog-item.0001.role",
        "catalog-item",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        status=AttemptStatus.RUNNING,
        profile=DomainProfile.SMITH,
        submission_token="original-token",
        job=JobReference(identity.job_id),
        scheduler_state=JobStatus.RUNNING.value,
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.CODING,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        attempts=(attempt,),
    )
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=recycled,
        generation=1,
        callbacks=_callbacks([], cancel="operator requested cancellation"),
    )

    result = runtime.tick()

    assert result.event is RuntimeEvent.BLOCKED
    assert "ownership verification failed closed" in result.reason
    assert recycled.cancelled == ()


def test_terminal_agent_attempt_revokes_credentials_before_domain_closure(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    attempt = AttemptRecord(
        "catalog-item.0001.role",
        "catalog-item",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        status=AttemptStatus.FAILED,
        profile=DomainProfile.SMITH,
        resource_class="coder_analysis",
        submission_token="terminal-worker-token",
        terminal_reason="worker failed",
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.CODING,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        attempts=(attempt,),
    )

    class _RevokingAdapter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def materialize(self, action: object) -> object:
            raise AssertionError(f"unexpected materialization: {action!r}")

        def execution(self, state: RunState, action: object) -> AttemptExecution:
            raise AssertionError(f"unexpected execution: {state!r} {action!r}")

        def revoke(self, state: RunState, current: AttemptRecord) -> bool:
            assert state.run_id == "runtime-test-run"
            self.calls.append(current.attempt_id)
            return True

    adapter = _RevokingAdapter()
    result = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
        worker_adapter=adapter,  # type: ignore[arg-type]
    ).tick()

    assert result.event is RuntimeEvent.CREDENTIALS_REVOKED
    assert adapter.calls == [attempt.attempt_id]
    assert load_state(workspace / STATE_FILENAME).item("catalog-item").status is (
        WorkItemStatus.CODING
    )


def test_preempted_attempt_appends_one_bounded_lineage_retry(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    failed = AttemptRecord(
        "catalog-item.0001.role",
        "catalog-item",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        status=AttemptStatus.PREEMPTED,
        profile=DomainProfile.SMITH,
        resource_class="coder_analysis",
        submission_token="preempted-token",
        job=JobReference("101"),
        scheduler_state=JobStatus.PREEMPTED.value,
        terminal_reason="scheduler reported PREEMPTED",
    )
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.CODING,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        attempts=(failed,),
    )
    result = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    ).tick()

    item = load_state(workspace / STATE_FILENAME).item("catalog-item")
    assert result.event is RuntimeEvent.ACTION_PERSISTED
    assert len(item.attempts) == 2
    retry = item.attempts[-1]
    assert retry.predecessor_attempt_id == failed.attempt_id
    assert retry.resource_class == failed.resource_class
    assert retry.status is AttemptStatus.PREPARED


def test_gate_evidence_collapses_physical_retry_lineage() -> None:
    failed = AttemptRecord(
        "gate-item.0001.deterministic_gate",
        "gate-item",
        1,
        Role.GATE,
        AttemptKind.DETERMINISTIC_GATE,
        1,
        status=AttemptStatus.PREEMPTED,
        profile=DomainProfile.SMITH,
        resource_class="deterministic_gate",
        submission_token="gate-preempted",
        job=JobReference("201"),
        scheduler_state=JobStatus.PREEMPTED.value,
        terminal_reason="scheduler reported PREEMPTED",
    )
    retry = AttemptRecord(
        "gate-item.0002.deterministic_gate",
        "gate-item",
        2,
        Role.GATE,
        AttemptKind.DETERMINISTIC_GATE,
        1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        resource_class="deterministic_gate",
        predecessor_attempt_id=failed.attempt_id,
        submission_token="gate-retry",
        job=JobReference("202"),
        result_digest="f" * 64,
    )
    item = WorkItemRecord(
        "gate-item",
        "stage",
        "goal",
        WorkItemKind.CATALOG_ONBOARD,
        DomainProfile.SMITH,
        status=WorkItemStatus.CODING,
        attempts=(failed, retry),
    )

    assert ControllerRuntime._gate_attempt_ids(item) == (retry.attempt_id,)
    assert ControllerRuntime._qa_gate_result_digests(item) == {
        retry.attempt_id: retry.result_digest
    }


def test_blocked_item_closes_goal_stage_and_run_in_bounded_ticks(tmp_path: Path) -> None:
    task, workspace = _task(tmp_path)
    _initialize_admitted(
        task,
        workspace,
        item_status=WorkItemStatus.BLOCKED,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        terminal_reason="typed worker blocker",
    )
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    )

    first = runtime.tick()
    second = runtime.tick()
    third = runtime.tick()

    assert first.event is RuntimeEvent.HIERARCHY_UPDATED
    assert second.event is RuntimeEvent.HIERARCHY_UPDATED
    assert third.disposition is RuntimeDisposition.TERMINAL
    assert third.terminal_status is not None
    assert third.terminal_status.value == "blocked"


def test_missing_required_stage_qa_records_qa_failed_then_blocks_run(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    candidate_digest = "d" * 64
    coder = AttemptRecord(
        "catalog-item.0001.role",
        "catalog-item",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        resource_class="coder_analysis",
        submission_token="coder-token",
        job=JobReference("201"),
        result_digest="a" * 64,
        candidate_digest=candidate_digest,
    )
    reviewer = AttemptRecord(
        "catalog-item.0002.reviewer_rerun",
        "catalog-item",
        2,
        Role.REVIEWER,
        AttemptKind.REVIEWER_RERUN,
        1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        resource_class="reviewer_rerun",
        submission_token="reviewer-token",
        job=JobReference("202"),
        result_digest="b" * 64,
        review_of_attempt_id=coder.attempt_id,
        reviewed_candidate_digest=candidate_digest,
    )
    item = WorkItemRecord(
        "catalog-item",
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.CATALOG_ONBOARD,
        DomainProfile.SMITH,
        status=WorkItemStatus.INTEGRATED,
        attempts=(coder, reviewer),
        candidate_attempt_id=coder.attempt_id,
        candidate_digest=candidate_digest,
        reviewer_attempt_id=reviewer.attempt_id,
    )
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=_plan_json(),
        items=(item,),
        goal_status=HierarchyStatus.SUCCEEDED,
        stage_status=StageStatus.READY_FOR_QA,
    )
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
        domain_inputs=RuntimeDomainInputs(qa_required_items=frozenset({"catalog-item"})),
        credential_broker=NoCredentialBroker(),
    )

    failed = runtime.tick()
    blocked = runtime.tick()
    terminal = runtime.tick()

    assert failed.event is RuntimeEvent.HIERARCHY_UPDATED
    assert blocked.event is RuntimeEvent.HIERARCHY_UPDATED
    assert terminal.disposition is RuntimeDisposition.TERMINAL
    state = load_state(workspace / STATE_FILENAME)
    assert state.stages[0].status is StageStatus.BLOCKED
    assert state.terminal_status is RunTerminalStatus.BLOCKED


@pytest.mark.parametrize(
    ("candidate_gate_passed", "expected_item_status", "expected_action"),
    [
        (True, WorkItemStatus.INTEGRATED, PromotionAction.KEEP),
        (False, WorkItemStatus.REJECTED, PromotionAction.REJECT),
    ],
)
def test_tuner_runtime_replays_controller_owned_terminal_decision(
    tmp_path: Path,
    candidate_gate_passed: bool,
    expected_item_status: WorkItemStatus,
    expected_action: PromotionAction,
) -> None:
    task, workspace = _task(tmp_path)
    gate_envelope = replace(
        task.execution.slurm.smith.resource_class("deterministic_gate"),
        gpus_per_node=1,
    )
    resources = tuple(
        gate_envelope if resource.name == "deterministic_gate" else resource
        for resource in task.execution.slurm.smith.resource_classes
    )
    controller = replace(
        task.execution.slurm.controller,
        mounts=(MountConfig(tmp_path.resolve(), "/staircase-test", False),),
    )
    task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(
                task.execution.slurm,
                controller=controller,
                smith=replace(
                    task.execution.slurm.smith,
                    max_gpus_total=1,
                    resource_classes=resources,
                ),
            ),
        ),
    )
    plan_response = _tuning_plan_json()
    item = WorkItemRecord(
        "catalog-item",
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.TUNE_HYPOTHESIS,
        DomainProfile.TUNER,
        status=WorkItemStatus.READY,
    )
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=plan_response,
        items=(item,),
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
        workflow_mode=WorkflowMode.TUNE,
    )
    plan, _review = recover_approved_plan(workspace, task)
    hypothesis = plan.items[0].domain_input
    assert isinstance(hypothesis, TuningHypothesis)
    gates = _gate_specs(task)
    cpu_resource = replace(gate_envelope, gpus_per_node=0)
    domain_inputs = RuntimeDomainInputs(
        tuning_hypotheses=(hypothesis,),
        gate_executions=tuple(
            (
                gate.gate_id,
                (
                    GateExecutionContract(EvidenceScope.CPU_STATIC, 0, None, None, False)
                    if index < 2
                    else GateExecutionContract(
                        EvidenceScope.SINGLE_GPU_PRODUCT,
                        1,
                        0,
                        0,
                        True,
                    )
                ),
            )
            for index, gate in enumerate(gates)
        ),
        gate_resources=tuple(
            (gate.gate_id, cpu_resource if index < 2 else gate_envelope)
            for index, gate in enumerate(gates)
        ),
        qa_required_items=frozenset(),
        role_evidence_paths=(
            ("coder", ("reports/coder.json",)),
            ("qa", ("reports/qa.json",)),
            ("reviewer", ("reports/reviewer.json",)),
        ),
        task_digest=task.digest,
        plan_digest=plan.digest,
    )
    scheduler = FakeScheduler()
    runtime = ProductionRuntimeFactory(
        domain_inputs,
        credential_broker=NoCredentialBroker(),
    )(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([]),
    )
    published: set[str] = set()
    reviewed_evidence = 0

    for _ in range(300):
        result = runtime.tick()
        assert result.event is not RuntimeEvent.BLOCKED, result.reason
        state = load_state(workspace / STATE_FILENAME)
        current = state.item("catalog-item")
        if state.terminal_status is not RunTerminalStatus.ACTIVE:
            break
        for attempt in current.attempts:
            if attempt.job is None or attempt.attempt_id in published or attempt.terminal:
                continue
            attempt_dir = (
                workspace / "items" / attempt.item_id / "attempts" / f"{attempt.sequence:04d}"
            )
            if not (attempt_dir / INPUT_FILENAME).is_file():
                continue
            if attempt.role is Role.CODER:
                _publish_tuner_result(
                    workspace,
                    state,
                    attempt,
                    candidate_gate_passed=candidate_gate_passed,
                )
            else:
                input_manifest, _input_digest = load_input_manifest(attempt_dir / INPUT_FILENAME)
                if attempt.role in {Role.REVIEWER, Role.QA}:
                    context = input_manifest.payload["context"]
                    assert isinstance(context, dict)
                    evidence = context["candidate_evidence"]
                    assert isinstance(evidence, dict)
                    assert set(evidence["artifacts"]) == {
                        "baseline.json",
                        "candidate.json",
                        "evaluation.json",
                        "campaign.json",
                    }
                    reviewed_evidence += 1
                _publish_approved_attempt_result(workspace, task, state, attempt)
            scheduler.transition(
                JobIdentity(
                    attempt.job.job_id,
                    array_task_id=attempt.job.array_task_id,
                    cluster=attempt.job.cluster,
                ),
                JobStatus.COMPLETED,
            )
            published.add(attempt.attempt_id)

    state = load_state(workspace / STATE_FILENAME)
    assert state.item("catalog-item").status is expected_item_status
    decision, _digest = load_promotion_decision_artifact(
        workspace / "items" / "catalog-item" / "tuning" / PROMOTION_DECISION_FILENAME
    )
    assert decision.action is expected_action
    assert reviewed_evidence == 3
    assert [attempt.role for attempt in state.item("catalog-item").attempts] == [
        Role.CODER,
        *(Role.GATE for _gate in gates),
        Role.REVIEWER,
        Role.REVIEWER,
        Role.QA,
    ]
    if expected_action is PromotionAction.KEEP:
        assert state.terminal_status is RunTerminalStatus.SUCCEEDED
        assert state.integration_history == ()
        fan_in_receipt = json.loads(
            (workspace / "items" / "catalog-item" / "fan-in-receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert fan_in_receipt["previous_head"] == fan_in_receipt["final_head"]
    else:
        assert state.terminal_status is RunTerminalStatus.BLOCKED


def test_smith_runtime_runs_coder_gates_reviewers_qa_and_integration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, workspace = _task(tmp_path)
    gate_envelope = replace(
        task.execution.slurm.smith.resource_class("deterministic_gate"),
        gpus_per_node=1,
    )
    resources = tuple(
        gate_envelope if resource.name == "deterministic_gate" else resource
        for resource in task.execution.slurm.smith.resource_classes
    )
    controller = replace(
        task.execution.slurm.controller,
        mounts=(MountConfig(tmp_path.resolve(), "/staircase-test", False),),
    )
    task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(
                task.execution.slurm,
                controller=controller,
                smith=replace(
                    task.execution.slurm.smith,
                    max_gpus_total=1,
                    resource_classes=resources,
                ),
            ),
        ),
    )
    plan_response = _plan_json()
    item = WorkItemRecord(
        "catalog-item",
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.CATALOG_ONBOARD,
        DomainProfile.SMITH,
        status=WorkItemStatus.READY,
    )
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=plan_response,
        items=(item,),
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    plan, _review = recover_approved_plan(workspace, task)
    gates = _gate_specs(task)
    cpu_resource = replace(gate_envelope, gpus_per_node=0)
    domain_inputs = RuntimeDomainInputs(
        gate_executions=tuple(
            (
                gate.gate_id,
                (
                    GateExecutionContract(
                        EvidenceScope.CPU_STATIC,
                        0,
                        None,
                        None,
                        False,
                    )
                    if index < 2
                    else GateExecutionContract(
                        EvidenceScope.SINGLE_GPU_PRODUCT,
                        1,
                        0,
                        0,
                        True,
                    )
                ),
            )
            for index, gate in enumerate(gates)
        ),
        gate_resources=tuple(
            (
                gate.gate_id,
                cpu_resource if index < 2 else gate_envelope,
            )
            for index, gate in enumerate(gates)
        ),
        qa_required_items=frozenset({"catalog-item"}),
        role_evidence_paths=(
            ("coder", ("reports/coder.json",)),
            ("qa", ("reports/qa.json",)),
            ("reviewer", ("reports/reviewer.json",)),
        ),
        task_digest=task.digest,
        plan_digest=plan.digest,
    )
    scheduler = FakeScheduler()
    runtime = ProductionRuntimeFactory(
        domain_inputs,
        credential_broker=NoCredentialBroker(),
    )(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([]),
    )
    published: set[str] = set()
    events: list[RuntimeEvent] = []
    restarted_after_fan_in = False
    injected_pre_receipt_crash = False
    recovered_pre_receipt_crash = False
    original_receipt_writer = ControllerRuntime._write_fan_in_receipt

    for _ in range(300):
        result = runtime.tick()
        events.append(result.event)
        if result.event is RuntimeEvent.BLOCKED and injected_pre_receipt_crash:
            monkeypatch.setattr(
                ControllerRuntime,
                "_write_fan_in_receipt",
                staticmethod(original_receipt_writer),
            )
            runtime = ProductionRuntimeFactory(
                domain_inputs,
                credential_broker=NoCredentialBroker(),
            )(
                workspace=workspace,
                task=task,
                scheduler=scheduler,
                generation=1,
                callbacks=_callbacks([]),
            )
            recovered_pre_receipt_crash = True
            injected_pre_receipt_crash = False
            continue
        assert result.event is not RuntimeEvent.BLOCKED, result.reason
        if result.event is RuntimeEvent.FAN_IN_INTENT_PERSISTED:

            def fail_before_receipt(*_args: object, **_kwargs: object) -> None:
                raise OSError("simulated controller crash before fan-in receipt")

            monkeypatch.setattr(
                ControllerRuntime,
                "_write_fan_in_receipt",
                staticmethod(fail_before_receipt),
            )
            injected_pre_receipt_crash = True
        if result.event is RuntimeEvent.FAN_IN_MATERIALIZED and not restarted_after_fan_in:
            runtime = ProductionRuntimeFactory(
                domain_inputs,
                credential_broker=NoCredentialBroker(),
            )(
                workspace=workspace,
                task=task,
                scheduler=scheduler,
                generation=1,
                callbacks=_callbacks([]),
            )
            restarted_after_fan_in = True
        state = load_state(workspace / STATE_FILENAME)
        current = state.item("catalog-item")
        if state.terminal_status is not RunTerminalStatus.ACTIVE:
            break
        for attempt in current.attempts:
            if attempt.job is None or attempt.attempt_id in published or attempt.terminal:
                continue
            attempt_dir = (
                workspace / "items" / attempt.item_id / "attempts" / f"{attempt.sequence:04d}"
            )
            if not (attempt_dir / INPUT_FILENAME).is_file():
                continue
            _publish_approved_attempt_result(workspace, task, state, attempt)
            scheduler.transition(
                JobIdentity(
                    attempt.job.job_id,
                    array_task_id=attempt.job.array_task_id,
                    cluster=attempt.job.cluster,
                ),
                JobStatus.COMPLETED,
            )
            published.add(attempt.attempt_id)

    state = load_state(workspace / STATE_FILENAME)
    item = state.item("catalog-item")
    assert item.status is WorkItemStatus.INTEGRATED, (
        f"{result.reason}; integration checkout={runtime._git.inspect()!r}"
    )
    assert [attempt.role for attempt in item.attempts] == [
        Role.CODER,
        *(Role.GATE for _gate in gates),
        Role.REVIEWER,
        Role.REVIEWER,
        Role.QA,
    ]
    assert [attempt.kind for attempt in item.attempts] == [
        AttemptKind.ROLE,
        *(AttemptKind.DETERMINISTIC_GATE for _gate in gates),
        AttemptKind.REVIEWER_ANALYSIS,
        AttemptKind.REVIEWER_RERUN,
        AttemptKind.QA,
    ]
    assert all(attempt.profile is DomainProfile.SMITH for attempt in item.attempts)
    assert len({attempt.job.scheduler_id for attempt in item.attempts if attempt.job}) == len(
        item.attempts
    )
    assert RuntimeEvent.FAN_IN_MATERIALIZED in events
    assert restarted_after_fan_in
    assert recovered_pre_receipt_crash
    assert RuntimeEvent.ITEM_INTEGRATED in events
    assert state.stages[0].status is StageStatus.CLOSED
    assert state.terminal_status is RunTerminalStatus.SUCCEEDED
    assert not any((workspace / "candidates").rglob(".git"))
    assert (workspace / "delivery" / "diff-repository").is_dir()
    assert _git(task.repository.root, "rev-parse", "HEAD") == task.repository.base_commit
    delivery_repository = workspace / "delivery" / "diff-repository"
    assert (
        delivery_repository
        / "tensorrt_llm"
        / "_torch"
        / "modeling_v2"
        / "catalog"
        / "norm"
        / "test_entry.py"
    ).is_file()
    assert "norm/test_entry.py" in (
        delivery_repository / "tensorrt_llm" / "_torch" / "modeling_v2" / "catalog" / "index.yaml"
    ).read_text(encoding="utf-8")
    index = yaml.safe_load(
        (
            delivery_repository
            / "tensorrt_llm"
            / "_torch"
            / "modeling_v2"
            / "catalog"
            / "index.yaml"
        ).read_text(encoding="utf-8")
    )
    assert any(entry["path"] == "norm/test_entry.py" for entry in index["entries"])
    patch_receipt = json.loads(
        (workspace / "delivery" / "patch-receipt.json").read_text(encoding="utf-8")
    )
    assert patch_receipt["head_commit"] == state.integration_head
    assert (
        patch_receipt["patch_sha256"]
        == hashlib.sha256((workspace / "delivery" / "staircase.patch").read_bytes()).hexdigest()
    )


def test_active_reconciliation_does_not_serialize_on_first_waiting_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, workspace = _task(tmp_path)
    plan_payload = json.loads(_plan_json())
    second = dict(plan_payload["items"][0])
    second["item_id"] = "catalog-item-b"
    second["entry_ids"] = ["test_entry_b"]
    second["allowed_paths"] = ["tensorrt_llm/_torch/modeling_v2/catalog/norm/test_entry_b.py"]
    plan_payload["items"].append(second)
    plan_payload["goals"][0]["item_ids"].append("catalog-item-b")
    plan = parse_plan_draft(
        json.dumps(plan_payload, sort_keys=True, separators=(",", ":")),
        allowed_resource_classes=tuple(
            resource.name for resource in task.execution.slurm.smith.resource_classes
        ),
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2/catalog",),
        workflow_mode=WorkflowMode.ONBOARD.value,
        target_features=task.target.features,
    )
    attempts = tuple(
        AttemptRecord(
            f"{item_id}.0001.role",
            item_id,
            1,
            Role.CODER,
            AttemptKind.ROLE,
            1,
            status=AttemptStatus.RUNNING,
            profile=DomainProfile.SMITH,
            submission_token=f"token-{item_id}",
            job=JobReference(str(index)),
        )
        for index, item_id in enumerate(("catalog-item", "catalog-item-b"), start=101)
    )
    state = RunState(
        "runtime-test-run",
        task.digest,
        task.repository.base_commit,
        1,
        stages=(StageRecord("catalog-stage", ("catalog-goal",), StageStatus.ACTIVE),),
        goals=(
            GoalRecord(
                "catalog-goal",
                "catalog-stage",
                ("catalog-item", "catalog-item-b"),
                HierarchyStatus.ACTIVE,
            ),
        ),
        items=tuple(
            WorkItemRecord(
                item_id,
                "catalog-stage",
                "catalog-goal",
                WorkItemKind.CATALOG_ONBOARD,
                DomainProfile.SMITH,
                status=WorkItemStatus.CODING,
                attempts=(attempt,),
            )
            for item_id, attempt in zip(("catalog-item", "catalog-item-b"), attempts, strict=True)
        ),
    )
    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        callbacks=_callbacks([]),
    )
    visited: list[str] = []

    def reconcile(
        current: RunState,
        proposal: object,
        item: WorkItemRecord,
        attempt: AttemptRecord,
    ) -> object:
        del proposal
        visited.append(item.item_id)
        event = (
            RuntimeEvent.NO_CHANGE
            if item.item_id == "catalog-item"
            else RuntimeEvent.ATTEMPT_UPDATED
        )
        return runtime._result(  # noqa: SLF001 - focused controller fairness test
            current,
            RuntimeDisposition.WAIT,
            event,
            "scheduler observation",
            item_id=item.item_id,
            attempt_id=attempt.attempt_id,
        )

    monkeypatch.setattr(runtime, "_reconcile_attempt", reconcile)
    result = runtime._reconcile_active_attempts(  # noqa: SLF001
        state,
        plan,
    )

    assert visited == ["catalog-item", "catalog-item-b"]
    assert result is not None and result.event is RuntimeEvent.ATTEMPT_UPDATED


def test_reverse_completion_still_journals_fan_in_in_stable_item_id_order_after_restart(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    plan_payload = json.loads(_two_item_plan_json())
    plan_payload["items"][0]["item_id"] = "a"
    plan_payload["items"][0]["kind"] = "catalog_verify"
    plan_payload["items"][0]["modifies_files"] = False
    plan_payload["items"][0]["entry_ids"] = ["entry_a"]
    plan_payload["items"][0]["allowed_paths"] = []
    plan_payload["items"][1]["item_id"] = "b"
    plan_payload["items"][1]["kind"] = "catalog_verify"
    plan_payload["items"][1]["modifies_files"] = False
    plan_payload["items"][1]["entry_ids"] = ["entry_b"]
    plan_payload["items"][1]["allowed_paths"] = []
    plan_payload["goals"][0]["item_ids"] = ["b", "a"]
    plan_payload["items"] = [plan_payload["items"][1], plan_payload["items"][0]]
    plan_response = json.dumps(plan_payload, sort_keys=True, separators=(",", ":"))

    def approved_item(item_id: str, digest_character: str) -> WorkItemRecord:
        candidate_digest = digest_character * 64
        coder = AttemptRecord(
            f"{item_id}.0001.role",
            item_id,
            1,
            Role.CODER,
            AttemptKind.ROLE,
            1,
            status=AttemptStatus.VALIDATED,
            profile=DomainProfile.SMITH,
            submission_token=f"token-{item_id}-coder",
            job=JobReference("101" if item_id == "a" else "201"),
            result_digest="c" * 64,
            candidate_digest=candidate_digest,
        )
        reviewer = AttemptRecord(
            f"{item_id}.0002.review",
            item_id,
            2,
            Role.REVIEWER,
            AttemptKind.REVIEWER_RERUN,
            1,
            status=AttemptStatus.VALIDATED,
            profile=DomainProfile.SMITH,
            submission_token=f"token-{item_id}-review",
            job=JobReference("102" if item_id == "a" else "202"),
            result_digest="d" * 64,
            candidate_digest=candidate_digest,
            review_of_attempt_id=coder.attempt_id,
            reviewed_candidate_digest=candidate_digest,
        )
        return WorkItemRecord(
            item_id,
            "catalog-stage",
            "catalog-goal",
            WorkItemKind.CATALOG_VERIFY,
            DomainProfile.SMITH,
            status=WorkItemStatus.APPROVED,
            attempts=(coder, reviewer),
            candidate_attempt_id=coder.attempt_id,
            candidate_digest=candidate_digest,
            reviewer_attempt_id=reviewer.attempt_id,
        )

    b = approved_item("b", "b")
    a = approved_item("a", "a")
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=plan_response,
        items=(b, a),
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    state = load_state(workspace / STATE_FILENAME)

    assert ControllerRuntime._integration_predecessor(state, b) == a  # noqa: SLF001
    assert ControllerRuntime._integration_predecessor(state, a) is None  # noqa: SLF001
    failed_predecessor = state.replace_item(
        a.transition(WorkItemStatus.REJECTED, terminal_reason="independent review rejected")
    )
    assert (
        ControllerRuntime._integration_predecessor(failed_predecessor, b) is None  # noqa: SLF001
    )

    desired = state.record_integration(
        item_id="a",
        previous_commit=state.integration_head,
        new_commit=state.integration_head,
        evidence_kind=IntegrationEvidenceKind.VERIFICATION,
        evidence_digest="a" * 64,
    )
    save_state(
        workspace / STATE_FILENAME,
        desired,
        expected_revision=state.revision,
        expected_generation=1,
    )
    restarted = load_state(workspace / STATE_FILENAME)
    assert ControllerRuntime._integration_predecessor(restarted, b).item_id == "a"  # noqa: SLF001

    integrating = restarted.replace_item(restarted.item("a").transition(WorkItemStatus.INTEGRATING))
    save_state(
        workspace / STATE_FILENAME,
        integrating,
        expected_revision=restarted.revision,
        expected_generation=1,
    )
    restarted = load_state(workspace / STATE_FILENAME)
    integrated = restarted.replace_item(restarted.item("a").transition(WorkItemStatus.INTEGRATED))
    save_state(
        workspace / STATE_FILENAME,
        integrated,
        expected_revision=restarted.revision,
        expected_generation=1,
    )
    restarted = load_state(workspace / STATE_FILENAME)
    assert ControllerRuntime._integration_predecessor(restarted, b) is None  # noqa: SLF001

    desired = restarted.record_integration(
        item_id="b",
        previous_commit=restarted.integration_head,
        new_commit=restarted.integration_head,
        evidence_kind=IntegrationEvidenceKind.VERIFICATION,
        evidence_digest="b" * 64,
    )
    save_state(
        workspace / STATE_FILENAME,
        desired,
        expected_revision=restarted.revision,
        expected_generation=1,
    )
    final = load_state(workspace / STATE_FILENAME)
    assert [record.item_id for record in final.integration_history] == ["a", "b"]


def test_horizontal_smith_fan_in_waits_for_active_peer_placement_receipt(
    tmp_path: Path,
) -> None:
    peer_attempt = _smith_coder_attempt(
        "catalog-item-b",
        job_id="201",
        status=AttemptStatus.RUNNING,
        digest_character="b",
    )
    peer = WorkItemRecord(
        "catalog-item-b",
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.CATALOG_ONBOARD,
        DomainProfile.SMITH,
        status=WorkItemStatus.CODING,
        attempts=(peer_attempt,),
    )
    _task_value, workspace, state, runtime, plan, review = _horizontal_placement_runtime(
        tmp_path,
        peer=peer,
    )
    current = state.item("catalog-item")
    _publish_all_smith_role_placements(
        workspace,
        state,
        current,
        hostname="node-current",
        scheduler=_fake_scheduler(runtime),
    )

    result = runtime._materialize_or_record_fan_in(  # noqa: SLF001
        state,
        plan,
        review,
        current,
        next(item for item in plan.items if item.item_id == current.item_id),
    )

    assert result.disposition is RuntimeDisposition.WAIT
    assert result.event is RuntimeEvent.NO_CHANGE
    assert result.attempt_id == peer_attempt.attempt_id
    assert "active peer receipt" in result.reason
    assert not (workspace / "items" / current.item_id / "fan-in-intent.json").exists()


def test_horizontal_smith_fan_in_blocks_terminal_attempt_without_receipt(
    tmp_path: Path,
) -> None:
    peer_attempt = _smith_coder_attempt(
        "catalog-item-b",
        job_id="201",
        status=AttemptStatus.FAILED,
        digest_character="b",
    )
    peer = WorkItemRecord(
        "catalog-item-b",
        "catalog-stage",
        "catalog-goal",
        WorkItemKind.CATALOG_ONBOARD,
        DomainProfile.SMITH,
        status=WorkItemStatus.BLOCKED,
        attempts=(peer_attempt,),
        terminal_reason="worker failed",
    )
    _task_value, workspace, state, runtime, plan, review = _horizontal_placement_runtime(
        tmp_path,
        peer=peer,
    )
    current = state.item("catalog-item")
    _publish_all_smith_role_placements(
        workspace,
        state,
        current,
        hostname="node-current",
        scheduler=_fake_scheduler(runtime),
    )

    result = runtime._materialize_or_record_fan_in(  # noqa: SLF001
        state,
        plan,
        review,
        current,
        next(item for item in plan.items if item.item_id == current.item_id),
    )

    assert result.event is RuntimeEvent.BLOCKED
    assert result.attempt_id == peer_attempt.attempt_id
    assert "lacks its placement receipt" in result.reason
    assert not (workspace / "items" / current.item_id / "fan-in-intent.json").exists()


def test_horizontal_smith_reverse_completion_and_restart_validate_each_role_stage(
    tmp_path: Path,
) -> None:
    peer = _approved_smith_item("catalog-item-b", job_id="201", digest_character="b")
    task, workspace, state, runtime, plan, _review = _horizontal_placement_runtime(
        tmp_path,
        peer=peer,
        reverse_items=True,
    )
    current = state.item("catalog-item")
    # Complete the lexical successor first. Reusing its node across role stages is
    # intentional: uniqueness is enforced within, not across, role-stage cohorts.
    scheduler = _fake_scheduler(runtime)
    _publish_all_smith_role_placements(
        workspace, state, peer, hostname="node-peer", scheduler=scheduler
    )
    _publish_all_smith_role_placements(
        workspace, state, current, hostname="node-current", scheduler=scheduler
    )
    proposal = next(item for item in plan.items if item.item_id == current.item_id)

    assert (
        runtime._horizontal_smith_placement_guard(  # noqa: SLF001
            state, plan, current, proposal
        )
        is None
    )

    restarted_state = load_state(workspace / STATE_FILENAME)
    restarted_plan, _restarted_review = recover_approved_plan(workspace, task, restarted_state)
    restarted = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([]),
    )
    assert (
        restarted._horizontal_smith_placement_guard(  # noqa: SLF001
            restarted_state,
            restarted_plan,
            restarted_state.item(current.item_id),
            next(item for item in restarted_plan.items if item.item_id == current.item_id),
        )
        is None
    )


def test_horizontal_smith_fan_in_rejects_duplicate_host_within_role_stage(
    tmp_path: Path,
) -> None:
    peer = _approved_smith_item("catalog-item-b", job_id="201", digest_character="b")
    _task_value, workspace, state, runtime, plan, _review = _horizontal_placement_runtime(
        tmp_path,
        peer=peer,
    )
    current = state.item("catalog-item")
    scheduler = _fake_scheduler(runtime)
    _publish_all_smith_role_placements(
        workspace, state, current, hostname="reused-node", scheduler=scheduler
    )
    _publish_all_smith_role_placements(
        workspace, state, peer, hostname="reused-node", scheduler=scheduler
    )

    result = runtime._horizontal_smith_placement_guard(  # noqa: SLF001
        state,
        plan,
        current,
        next(item for item in plan.items if item.item_id == current.item_id),
    )

    assert result is not None and result.event is RuntimeEvent.BLOCKED
    assert "reused a trusted scheduler node" in result.reason
    assert "role stage 'role'" in result.reason


def test_horizontal_smith_fan_in_rejects_spoofed_reviewer_job(
    tmp_path: Path,
) -> None:
    peer = _approved_smith_item("catalog-item-b", job_id="201", digest_character="b")
    _task_value, workspace, state, runtime, plan, _review = _horizontal_placement_runtime(
        tmp_path,
        peer=peer,
    )
    current = state.item("catalog-item")
    scheduler = _fake_scheduler(runtime)
    _publish_all_smith_role_placements(
        workspace, state, current, hostname="node-current", scheduler=scheduler
    )
    peer_analysis = next(
        attempt for attempt in peer.attempts if attempt.kind is AttemptKind.REVIEWER_ANALYSIS
    )
    _publish_all_smith_role_placements(
        workspace,
        state,
        peer,
        hostname="node-peer",
        scheduler=scheduler,
        spoof_attempt_id=peer_analysis.attempt_id,
    )

    result = runtime._horizontal_smith_placement_guard(  # noqa: SLF001
        state,
        plan,
        current,
        next(item for item in plan.items if item.item_id == current.item_id),
    )

    assert result is not None and result.event is RuntimeEvent.BLOCKED
    assert "identity mismatch" in result.reason
    assert "role stage 'reviewer_analysis'" in result.reason


def test_two_ready_disjoint_smith_siblings_hold_concurrent_nonterminal_jobs(
    tmp_path: Path,
) -> None:
    task, workspace = _task(tmp_path)
    item_ids = ("catalog-item", "catalog-item-b")
    items = tuple(
        WorkItemRecord(
            item_id,
            "catalog-stage",
            "catalog-goal",
            WorkItemKind.CATALOG_ONBOARD,
            DomainProfile.SMITH,
            status=WorkItemStatus.READY,
        )
        for item_id in item_ids
    )
    _initialize_custom_admitted(
        task,
        workspace,
        plan_response=_two_item_plan_json(),
        items=items,
        goal_status=HierarchyStatus.ACTIVE,
        stage_status=StageStatus.ACTIVE,
    )
    scheduler = FakeScheduler()

    def execution_factory(
        state: RunState,
        proposal: object,
        attempt: AttemptRecord,
    ) -> AttemptExecution:
        del proposal
        attempt_dir = workspace / "items" / attempt.item_id / "attempts" / "0001"
        input_path = attempt_dir / INPUT_FILENAME
        if input_path.exists():
            _manifest, input_digest = load_input_manifest(input_path)
        else:
            input_digest = write_input_manifest(
                input_path,
                WorkerInputManifest(
                    state.run_id,
                    attempt.item_id,
                    attempt.attempt_id,
                    state.task_digest,
                    state.generation,
                    attempt.role,
                    attempt.profile,
                    str(workspace / "candidates" / attempt.item_id / attempt.attempt_id),
                ),
            )
        return AttemptExecution(
            resources=ResourceRequest(
                label=attempt.attempt_id,
                account="coreai",
                partition="batch",
                time_limit="00:10:00",
                output_path=workspace / "logs/%j.out",
                error_path=workspace / "logs/%j.err",
            ),
            command=InternalCommand(InternalEntrypoint.WORKER, input_bundle=input_path),
            submission_token=f"token-{attempt.attempt_id}",
            attempt_dir=attempt_dir,
            expectation=ResultExpectation(
                state.run_id,
                attempt.item_id,
                attempt.attempt_id,
                state.task_digest,
                state.generation,
                input_digest,
            ),
            receipt_root=workspace / "receipts",
            quarantine_root=workspace / "quarantine",
        )

    runtime = ControllerRuntime(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        callbacks=_callbacks([]),
        execution_factory=execution_factory,
        smith_materializer=lambda actions: tuple(actions),
    )

    for _ in range(20):
        runtime.tick()
        if len(scheduler.submissions) == 2:
            break

    state = load_state(workspace / STATE_FILENAME)
    attempts = tuple(state.item(item_id).attempts[-1] for item_id in item_ids)
    assert len(scheduler.submissions) == 2
    assert all(not attempt.terminal for attempt in attempts)
    assert all(attempt.job is not None for attempt in attempts)
    assert len({attempt.job.scheduler_id for attempt in attempts if attempt.job is not None}) == 2
