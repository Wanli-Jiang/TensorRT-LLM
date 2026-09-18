# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the durable Staircase planning controller."""

from __future__ import annotations

import hashlib
import json
import stat
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Mapping

import pytest

from agent_flow.workflows.staircase.common.credentials import (
    CREDENTIAL_MOUNT_PATH,
    CredentialBinding,
    CredentialDescriptor,
    CredentialHandle,
    CredentialProvision,
    CredentialRevocationCause,
    CredentialRevocationReceipt,
    CredentialState,
    NoCredentialBroker,
    descriptor_from_public_dict,
    locate_credential_provision,
    materialize_credential_provision,
    revoke_terminal_attempt_credentials,
)
from agent_flow.workflows.staircase.common.outcomes import parse_plan_draft, parse_plan_review
from agent_flow.workflows.staircase.common.runners import RoleProcessResult, load_role_spec
from agent_flow.workflows.staircase.common.slurm import (
    FakeScheduler,
    InternalCommand,
    JobIdentity,
    JobObservation,
    JobOwnership,
    JobStatus,
    ObservationSource,
    ResourceRequest,
    SchedulerError,
)
from agent_flow.workflows.staircase.common.submission_recovery import SubmissionRecoveryPolicy
from agent_flow.workflows.staircase.controller.planning import (
    PlanningAction,
    PlanningConfig,
    PlanningEngine,
    PlanningError,
    PlanningEvent,
    PlanningPhase,
)
from agent_flow.workflows.staircase.controller.production_inputs import (
    GateExecutionIntent,
    ProductionInputError,
    resolve_production_inputs,
)
from agent_flow.workflows.staircase.state import (
    PLANNING_ITEM_ID,
    STATE_FILENAME,
    AttemptStatus,
    DomainProfile,
    RunState,
    RunTerminalStatus,
    WorkflowMode,
    WorkItemKind,
    initialize_state,
    load_state,
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

_TASK_DIGEST = "a" * 64


class InjectedCrash(RuntimeError):
    """Crash injected after the fake scheduler has accepted a job."""


class TransientObservationScheduler(FakeScheduler):
    """Fail one owned observation without changing authoritative job state."""

    def __init__(self) -> None:
        super().__init__()
        self.failures_remaining = 1

    def observe_owned(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobObservation:
        if self.failures_remaining:
            self.failures_remaining -= 1
            raise SchedulerError("transient scheduler read failure")
        return super().observe_owned(identity, submission_token)


class CrashAfterSubmitScheduler:
    """Leave authoritative state in SUBMITTING after a successful submit."""

    def __init__(self, delegate: FakeScheduler) -> None:
        self._delegate = delegate

    def submit(
        self,
        resources: ResourceRequest,
        command: InternalCommand,
        submission_token: str,
        environment: Mapping[str, str] | None = None,
    ) -> JobIdentity:
        self._delegate.submit(resources, command, submission_token, environment)
        raise InjectedCrash("controller crashed after scheduler acceptance")

    def observe(self, identity: JobIdentity):  # type: ignore[no-untyped-def]
        return self._delegate.observe(identity)

    def cancel(self, identity: JobIdentity) -> None:
        self._delegate.cancel(identity)

    def lookup_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.lookup_submission(submission_token)

    def probe_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.probe_submission(submission_token)

    def verify_ownership(self, identity: JobIdentity, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.verify_ownership(identity, submission_token)


class CrashBeforeSubmitScheduler:
    """Crash after a durable claim but before the scheduler accepts a job."""

    def __init__(self, delegate: FakeScheduler) -> None:
        self._delegate = delegate
        self.submit_calls = 0

    def submit(self, *args: object, **kwargs: object) -> JobIdentity:
        self.submit_calls += 1
        raise InjectedCrash("controller crashed before scheduler acceptance")

    def observe(self, identity: JobIdentity):  # type: ignore[no-untyped-def]
        return self._delegate.observe(identity)

    def cancel(self, identity: JobIdentity) -> None:
        self._delegate.cancel(identity)

    def lookup_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.lookup_submission(submission_token)

    def probe_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.probe_submission(submission_token)

    def verify_ownership(self, identity: JobIdentity, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.verify_ownership(identity, submission_token)


class MutableClock:
    """Trusted deterministic clock for bounded submission-recovery tests."""

    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class OwnershipMismatchScheduler:
    """Expose an exact job while denying its immutable token ownership."""

    def __init__(self, delegate: FakeScheduler) -> None:
        self._delegate = delegate
        self._verification_count = 0

    def submit(self, *args: object, **kwargs: object) -> JobIdentity:
        return self._delegate.submit(*args, **kwargs)  # type: ignore[arg-type]

    def observe(self, identity: JobIdentity):  # type: ignore[no-untyped-def]
        return self._delegate.observe(identity)

    def cancel(self, identity: JobIdentity) -> None:
        self._delegate.cancel(identity)

    def lookup_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.lookup_submission(submission_token)

    def probe_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
        return self._delegate.probe_submission(submission_token)

    def verify_ownership(
        self,
        identity: JobIdentity,
        submission_token: str,
    ) -> JobOwnership:
        observed = self._delegate.verify_ownership(identity, submission_token)
        self._verification_count += 1
        if self._verification_count == 1:
            return observed
        return replace(observed, matched=False, reason="injected immutable-token mismatch")


class FakeCredentialBroker:
    """Test-only trusted boundary; controller calls expose no secret values."""

    def __init__(self, secret: str, *, broker_id: str = "test-broker") -> None:
        self._secret = secret
        self._broker_id = broker_id
        self.described: list[str] = []
        self.injected: list[str] = []
        self.revocations: list[tuple[str, CredentialRevocationCause]] = []

    def describe(self, binding: CredentialBinding) -> CredentialDescriptor:
        self.described.append(binding.attempt_id)
        return CredentialDescriptor.create(
            binding,
            CredentialState.BUNDLE,
            ("OPENAI_API_KEY",),
            handle=CredentialHandle(
                self._broker_id,
                binding.attempt_id,
                int(time.time()) + 3_600,
            ),
        )

    def inject_after_admission(
        self,
        *,
        workspace: Path,
        descriptor: CredentialDescriptor,
    ) -> CredentialProvision:
        self.injected.append(descriptor.binding.attempt_id)
        return materialize_credential_provision(
            workspace=workspace,
            descriptor=descriptor,
            credential_values={"OPENAI_API_KEY": self._secret},
        )

    def revoke(
        self,
        *,
        workspace: Path,
        expected_binding: CredentialBinding,
        descriptor: CredentialDescriptor,
        cause: CredentialRevocationCause,
    ) -> CredentialRevocationReceipt:
        self.revocations.append((expected_binding.attempt_id, cause))
        return revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=expected_binding,
            provision=locate_credential_provision(
                workspace=workspace,
                descriptor=descriptor,
            ),
            cause=cause,
        )


@pytest.fixture
def task_and_workspace(tmp_path: Path) -> tuple[NormalizedTask, Path]:
    """Create a normalized task and revision-zero planning workspace."""
    repository = tmp_path / "repository"
    workspace_root = tmp_path / "workspaces"
    checkpoint = tmp_path / "checkpoint"
    image = tmp_path / "trtllm.sqsh"
    repository.mkdir()
    workspace_root.mkdir()
    checkpoint.mkdir()
    image.write_bytes(b"image")
    resource_classes = tuple(
        ResourceClass(
            name=name,
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=2,
            memory_mib=1_024,
            time_limit_seconds=600,
            partition="cpu",
            qos="cpu-short",
        )
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
            RoleEnvironmentPolicy((), (("PYTHONNOUSERSITE", "1"),)),
            CredentialBrokerPolicy("test-broker", ("OPENAI_API_KEY",), 7_200),
        )
        for role, resource in role_resources.items()
    )
    task = NormalizedTask(
        schema_version=1,
        repository=RepositoryConfig(
            root=repository,
            base_commit="1" * 40,
            dirty_policy="reject",
            workspace_root=workspace_root,
        ),
        reference=ReferenceConfig(
            checkpoint=checkpoint,
            provenance="test",
            architecture="TestForCausalLM",
            additional_sources=(),
        ),
        target=TargetConfig(
            family="test_family",
            checkpoint_id="test_checkpoint",
            sm=100,
            world_size=1,
            mapping=ParallelMapping(1, 1, 1, 1, 1),
            features=(),
            expected_route="tensorrt_llm/_torch/modeling_v2/models/test_family/routing.py",
            synthetic_target=True,
        ),
        gates=GatesConfig(
            accuracy=AccuracyGate("tests/accuracy.py::test_model", "reference", "exact", 0.0),
            feature_signal_tests=(),
            boot_tests=("tests/boot.py::test_model",),
            component_tests=("tests/component.py::test_model",),
            collective_tests=(),
        ),
        certification=CertificationConfig(CertificationMode.LOCAL, None),
        execution=ExecutionConfig(
            mode="slurm",
            slurm=SlurmConfig(
                controller=ControllerConfig(
                    account="coreai",
                    partition="batch",
                    qos=None,
                    reservation=None,
                    time_limit_seconds=3_600,
                    cpus_per_task=2,
                    memory_mib=2_048,
                    image=image,
                    scheduler_clients=True,
                    container_launch_mode=ContainerLaunchMode.IN_ALLOCATION_SRUN,
                    mounts=(MountConfig(tmp_path, str(tmp_path), False),),
                    environment=(),
                    build_identity="test-build",
                    dispatch_mode="nested_submission",
                    requeue=False,
                    advance_signal_lead_seconds=60,
                    heartbeat_timeout_seconds=120,
                    lease_timeout_seconds=300,
                    orphan_grace_seconds=600,
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
                    resource_classes=resource_classes,
                    per_item_override_bounds=OverrideBounds(2, 2, 2, 8, 8_192, 3_600),
                    retry_policy=RetryPolicy(preempted=1, node_failure=1),
                    distinct_nodes_required=False,
                    exclusive=False,
                ),
                role_classes=role_classes,
            ),
            agent=AgentExecutionConfig("codex", "test-model"),
        ),
        delivery=DeliveryConfig(mode="diff_only", branch=None, merge=False, push=False),
        digest=_TASK_DIGEST,
    )
    workspace = workspace_root / "run"
    workspace.mkdir()
    initialize_state(
        workspace / STATE_FILENAME,
        RunState(
            run_id="staircase-test-run",
            task_digest=task.digest,
            base_commit=task.repository.base_commit,
            generation=1,
        ),
    )
    return task, workspace


def _plan_json() -> str:
    return json.dumps(
        {
            "schema_version": 2,
            "outcome": "DRAFTED",
            "stages": [
                {
                    "stage_id": "stage-discovery",
                    "goal_ids": ["goal-attention"],
                    "exit_gates": ["attention semantics derived"],
                }
            ],
            "goals": [
                {
                    "goal_id": "goal-attention",
                    "stage_id": "stage-discovery",
                    "capability": "attention",
                    "item_ids": ["search-attention"],
                }
            ],
            "items": [
                {
                    "item_id": "search-attention",
                    "goal_id": "goal-attention",
                    "kind": "catalog_verify",
                    "resource_class": "coder_analysis",
                    "execution": {
                        "nodes": 1,
                        "ranks_per_node": 1,
                        "gpus_per_node": 0,
                        "array_element": False,
                        "verdict_scope": "item",
                    },
                    "modifies_files": False,
                    "domain_input": None,
                    "dependencies": [],
                    "entry_ids": ["attention"],
                    "allowed_paths": [],
                    "certified_claim_cells": [],
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _production_gate_task(task: NormalizedTask) -> NormalizedTask:
    smith = task.execution.slurm.smith
    resource_classes = tuple(
        replace(resource, gpus_per_node=1, partition="batch", qos="normal")
        if resource.name == "deterministic_gate"
        else resource
        for resource in smith.resource_classes
    )
    return replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(
                task.execution.slurm,
                smith=replace(
                    smith,
                    max_gpus_total=max(1, smith.max_gpus_total),
                    resource_classes=resource_classes,
                ),
            ),
        ),
    )


def _publish_response(command: InternalCommand, response: str) -> None:
    input_path = command.input_bundle
    assert input_path is not None
    spec = load_role_spec(input_path)
    result = RoleProcessResult(
        schema_version=spec.schema_version,
        run_id=spec.run_id,
        task_digest=spec.task_digest,
        generation=spec.generation,
        item_id=spec.item_id,
        attempt_id=spec.attempt_id,
        prompt_id=spec.prompt_id,
        role=spec.role,
        profile=spec.profile,
        response=response,
        response_digest=hashlib.sha256(response.encode("utf-8")).hexdigest(),
    )
    Path(spec.result_path).write_text(
        json.dumps(asdict(result), sort_keys=True) + "\n", encoding="utf-8"
    )


def _finish_current_attempt(
    engine: PlanningEngine,
    scheduler: FakeScheduler,
    response: str | None,
) -> None:
    submission = scheduler.submissions[-1]
    if response is not None:
        _publish_response(submission.command, response)
    scheduler.transition(
        submission.identity,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    observed = engine.tick()
    assert observed.event is PlanningEvent.TERMINAL_OBSERVED
    engine.tick()


def _engine(
    task: NormalizedTask,
    workspace: Path,
    scheduler: FakeScheduler,
    *,
    config: PlanningConfig | None = None,
    generation: int = 1,
    credential_broker: FakeCredentialBroker | None = None,
    backend_environment: Mapping[str, str] | None = None,
) -> PlanningEngine:
    broker = credential_broker or FakeCredentialBroker("default-test-secret")
    return PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=generation,
        config=config,
        credential_broker=broker,
        backend_environment=backend_environment,
    )


def test_approved_plan_is_digest_pinned_and_admitted_only_after_review(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    engine = _engine(task, workspace, scheduler)

    prepared = engine.tick()
    assert prepared.event is PlanningEvent.DRAFT_PREPARED
    assert not load_state(workspace / STATE_FILENAME).items
    submitted = engine.tick()
    assert submitted.event is PlanningEvent.SUBMITTED
    assert scheduler.submissions[-1].command.entrypoint.value == "role-worker"
    assert scheduler.submissions[-1].resources.partition == "cpu"
    assert scheduler.submissions[-1].resources.qos == "cpu-short"
    _finish_current_attempt(engine, scheduler, _plan_json())

    draft_state = load_state(workspace / STATE_FILENAME)
    draft = draft_state.planning_attempts[-1]
    assert draft.status is AttemptStatus.VALIDATED
    assert draft.candidate_digest is not None
    assert not draft_state.items

    reviewer_prepared = engine.tick()
    assert reviewer_prepared.event is PlanningEvent.REVIEW_PREPARED
    review_attempt = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    assert review_attempt.review_of_attempt_id == draft.attempt_id
    assert review_attempt.reviewed_candidate_digest == draft.candidate_digest
    review_spec = load_role_spec(
        workspace
        / "items"
        / PLANNING_ITEM_ID
        / "attempts"
        / review_attempt.attempt_id
        / "role-input.json"
    )
    assert draft.candidate_digest in review_spec.prompt

    engine.tick()
    review_json = json.dumps(
        {
            "schema_version": 2,
            "outcome": "ACCEPT",
            "plan_digest": draft.candidate_digest,
            "corrections": [],
        }
    )
    _finish_current_attempt(engine, scheduler, review_json)
    assert not load_state(workspace / STATE_FILENAME).items

    admitted = engine.tick()
    assert admitted.phase is PlanningPhase.ADMITTED
    assert admitted.event is PlanningEvent.PLAN_ADMITTED
    assert admitted.next_action is PlanningAction.START_WORKFLOW
    state = load_state(workspace / STATE_FILENAME)
    assert state.stages[0].stage_id == "stage-discovery"
    assert state.goals[0].goal_id == "goal-attention"
    assert state.items[0].kind is WorkItemKind.CATALOG_VERIFY
    assert state.items[0].profile is DomainProfile.SMITH


@pytest.mark.parametrize("unsupported_kind", ["feasibility", "search", "gate"])
def test_unsupported_work_item_kind_is_rejected_before_review_or_admission(
    task_and_workspace: tuple[NormalizedTask, Path],
    unsupported_kind: str,
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    engine = _engine(task, workspace, scheduler)
    payload = json.loads(_plan_json())
    payload["items"][0]["kind"] = unsupported_kind
    payload["items"][0]["entry_ids"] = []

    engine.tick()
    engine.tick()
    _finish_current_attempt(engine, scheduler, json.dumps(payload))

    state = load_state(workspace / STATE_FILENAME)
    assert state.items == ()
    assert len(state.planning_attempts) == 1
    assert state.planning_attempts[0].status is AttemptStatus.FAILED
    assert state.planning_attempts[0].terminal_reason == "invalid_role_result:PlanningError"
    retry = engine.tick()
    assert retry.event is PlanningEvent.DRAFT_PREPARED


def test_crash_after_submit_is_adopted_by_token_without_duplicate(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    engine = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=CrashAfterSubmitScheduler(scheduler),
        generation=1,
        credential_broker=FakeCredentialBroker("crash-test-secret"),
    )
    engine.tick()

    with pytest.raises(InjectedCrash):
        engine.tick()
    state = load_state(workspace / STATE_FILENAME)
    assert state.planning_attempts[-1].status is AttemptStatus.SUBMITTING
    assert len(scheduler.submissions) == 1

    adopted = _engine(task, workspace, scheduler).tick()
    assert adopted.event is PlanningEvent.ADOPTED
    assert len(scheduler.submissions) == 1
    assert load_state(workspace / STATE_FILENAME).planning_attempts[-1].job is not None


def test_transient_observation_failure_waits_without_mutating_state(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = TransientObservationScheduler()
    engine = _engine(task, workspace, scheduler)

    engine.tick()
    submitted = engine.tick()
    before = load_state(workspace / STATE_FILENAME)

    waiting = engine.tick()

    assert waiting.event is PlanningEvent.NO_CHANGE
    assert waiting.next_action is PlanningAction.WAIT_FOR_SCHEDULER
    assert waiting.scheduler_status is JobStatus.UNKNOWN
    assert load_state(workspace / STATE_FILENAME) == before

    recovered = engine.tick()
    assert recovered.event is PlanningEvent.SCHEDULER_UPDATED
    assert recovered.scheduler_status is JobStatus.PENDING
    assert recovered.attempt_id == submitted.attempt_id


def test_crash_before_submit_retries_only_after_bounded_absence_proof(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    crashing = CrashBeforeSubmitScheduler(scheduler)
    clock = MutableClock()
    recovery_policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=1,
        absence_samples_required=2,
        absence_sample_interval_seconds=1,
        max_submit_calls=2,
    )
    broker = FakeCredentialBroker("crash-before-submit-secret")
    engine = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=crashing,
        generation=1,
        credential_broker=broker,
        submission_clock=clock,
        submission_recovery_policy=recovery_policy,
    )
    engine.tick()
    with pytest.raises(InjectedCrash, match="before scheduler acceptance"):
        engine.tick()
    assert scheduler.submissions == ()

    recovered = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        credential_broker=broker,
        submission_clock=clock,
        submission_recovery_policy=recovery_policy,
    )
    clock.advance(2)
    assert recovered.tick().next_action is PlanningAction.WAIT_FOR_SCHEDULER
    assert scheduler.submissions == ()
    clock.advance(1)
    assert recovered.tick().event is PlanningEvent.SUBMITTED
    assert len(scheduler.submissions) == 1


def test_planning_submission_retry_exhaustion_is_durable_manual_failure(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    delegate = FakeScheduler()
    scheduler = CrashBeforeSubmitScheduler(delegate)
    clock = MutableClock()
    recovery_policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=1,
        absence_samples_required=2,
        absence_sample_interval_seconds=1,
        max_submit_calls=2,
    )
    broker = FakeCredentialBroker("retry-exhaustion-secret")
    engine = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        credential_broker=broker,
        submission_clock=clock,
        submission_recovery_policy=recovery_policy,
    )
    engine.tick()
    with pytest.raises(InjectedCrash):
        engine.tick()
    clock.advance(2)
    assert engine.tick().next_action is PlanningAction.WAIT_FOR_SCHEDULER
    clock.advance(1)
    with pytest.raises(InjectedCrash):
        engine.tick()
    clock.advance(2)
    assert engine.tick().next_action is PlanningAction.WAIT_FOR_SCHEDULER
    clock.advance(1)
    exhausted = engine.tick()

    assert exhausted.event is PlanningEvent.MANUAL_RECOVERY_REQUIRED
    assert exhausted.next_action is PlanningAction.STOP
    attempt = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    assert attempt.status is AttemptStatus.FAILED
    assert attempt.scheduler_state == "SUBMISSION_MANUAL_RECOVERY"
    assert scheduler.submit_calls == 2
    manual = (
        workspace
        / "receipts"
        / "submissions"
        / "planning"
        / attempt.attempt_id
        / "manual-recovery.json"
    )
    assert manual.is_file()

    attempt_count = len(load_state(workspace / STATE_FILENAME).planning_attempts)
    injection_count = len(broker.injected)
    restarted = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        credential_broker=broker,
        submission_clock=clock,
        submission_recovery_policy=recovery_policy,
    )
    stable = restarted.tick()
    assert stable.event is PlanningEvent.MANUAL_RECOVERY_REQUIRED
    assert stable.next_action is PlanningAction.STOP
    assert len(load_state(workspace / STATE_FILENAME).planning_attempts) == attempt_count
    assert len(broker.injected) == injection_count
    assert scheduler.submit_calls == 2


def test_completed_without_valid_result_is_application_failure(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    broker = FakeCredentialBroker("sk-missing-result-secret")
    engine = _engine(
        task,
        workspace,
        scheduler,
        credential_broker=broker,
    )
    prepared = engine.tick()
    assert prepared.event is PlanningEvent.DRAFT_PREPARED
    assert broker.described == [prepared.attempt_id]
    assert broker.injected == []
    engine.tick()
    assert broker.injected == [prepared.attempt_id]
    credential_mount = next(
        mount
        for mount in scheduler.submissions[-1].resources.mounts
        if mount.target == CREDENTIAL_MOUNT_PATH
    )

    _finish_current_attempt(engine, scheduler, None)

    attempt = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    assert attempt.status is AttemptStatus.FAILED
    assert attempt.terminal_reason == "invalid_role_result:PlanningError"
    assert not credential_mount.source.exists()
    assert broker.revocations == [(attempt.attempt_id, CredentialRevocationCause.TERMINAL)]


def test_scheduler_terminal_failure_revokes_exact_planning_bundle(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    broker = FakeCredentialBroker("sk-scheduler-failure-secret")
    engine = _engine(
        task,
        workspace,
        scheduler,
        credential_broker=broker,
    )
    engine.tick()
    engine.tick()
    submission = scheduler.submissions[-1]
    bundle = next(
        mount.source
        for mount in submission.resources.mounts
        if mount.target == CREDENTIAL_MOUNT_PATH
    )
    scheduler.transition(
        submission.identity,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )

    assert engine.tick().event is PlanningEvent.TERMINAL_OBSERVED
    assert engine.tick().event is PlanningEvent.RESULT_REJECTED
    assert not bundle.exists()
    attempt = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    assert broker.revocations == [(attempt.attempt_id, CredentialRevocationCause.TERMINAL)]


def test_preempted_planning_attempt_revokes_handle_before_replacement(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    broker = FakeCredentialBroker("sk-replaced-planning-secret")
    engine = _engine(
        task,
        workspace,
        scheduler,
        credential_broker=broker,
    )
    engine.tick()
    engine.tick()
    submission = scheduler.submissions[-1]
    scheduler.transition(
        submission.identity,
        JobStatus.PREEMPTED,
        source=ObservationSource.ACCOUNTING,
    )

    assert engine.tick().event is PlanningEvent.TERMINAL_OBSERVED
    assert engine.tick().event is PlanningEvent.RESULT_REJECTED
    failed = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    assert failed.status is AttemptStatus.PREEMPTED
    assert broker.revocations == [(failed.attempt_id, CredentialRevocationCause.REPLACED)]

    replacement = engine.tick()
    assert replacement.event is PlanningEvent.DRAFT_PREPARED
    assert replacement.attempt_id != failed.attempt_id


def test_rejected_plan_revision_limit_is_bounded(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    engine = _engine(
        task,
        workspace,
        scheduler,
        config=PlanningConfig(
            backend_kind=task.execution.agent.backend_kind,
            model=task.execution.agent.model,
            max_draft_attempts=1,
        ),
    )
    engine.tick()
    engine.tick()
    _finish_current_attempt(engine, scheduler, _plan_json())
    draft = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    engine.tick()
    engine.tick()
    rejection = json.dumps(
        {
            "schema_version": 2,
            "outcome": "REVISE",
            "plan_digest": draft.candidate_digest,
            "corrections": ["split the attention capability"],
        }
    )
    _finish_current_attempt(engine, scheduler, rejection)

    exhausted = engine.tick()
    assert exhausted.phase is PlanningPhase.EXHAUSTED
    assert exhausted.event is PlanningEvent.RUN_EXHAUSTED
    state = load_state(workspace / STATE_FILENAME)
    assert state.terminal_status is RunTerminalStatus.EXHAUSTED
    assert not state.items


def test_plan_drafter_blocked_outcome_stops_as_blocked_input(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    engine = _engine(task, workspace, scheduler)
    engine.tick()
    engine.tick()
    _finish_current_attempt(
        engine,
        scheduler,
        json.dumps(
            {
                "schema_version": 2,
                "outcome": "BLOCKED",
                "reason": "checkpoint metadata is unavailable",
            }
        ),
    )

    blocked = engine.tick()

    assert blocked.phase is PlanningPhase.BLOCKED_INPUT
    assert blocked.event is PlanningEvent.RUN_BLOCKED
    state = load_state(workspace / STATE_FILENAME)
    assert state.terminal_status is RunTerminalStatus.BLOCKED_INPUT
    assert state.terminal_reason == "PlanDrafter: checkpoint metadata is unavailable"


def test_plan_reviewer_block_outcome_stops_without_drafting_replacement(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    engine = _engine(task, workspace, scheduler)
    engine.tick()
    engine.tick()
    _finish_current_attempt(engine, scheduler, _plan_json())
    draft = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    engine.tick()
    engine.tick()
    _finish_current_attempt(
        engine,
        scheduler,
        json.dumps(
            {
                "schema_version": 2,
                "outcome": "BLOCK",
                "plan_digest": draft.candidate_digest,
                "corrections": ["independent reference source is unavailable"],
            }
        ),
    )

    blocked = engine.tick()

    assert blocked.event is PlanningEvent.RUN_BLOCKED
    state = load_state(workspace / STATE_FILENAME)
    assert state.terminal_status is RunTerminalStatus.BLOCKED_INPUT
    assert len(state.planning_attempts) == 2


def test_pending_planning_job_is_cancelled_by_exact_identity(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    scheduler = FakeScheduler()
    broker = FakeCredentialBroker("sk-cancelled-planning-secret")
    engine = _engine(task, workspace, scheduler, credential_broker=broker)
    engine.tick()
    engine.tick()
    identity = scheduler.submissions[-1].identity

    cancelled = engine.tick(cancel_requested=True, cancellation_reason="operator request")

    assert cancelled.event is PlanningEvent.RUN_CANCELLED
    assert scheduler.cancelled == (identity,)
    state = load_state(workspace / STATE_FILENAME)
    assert state.planning_attempts[-1].status is AttemptStatus.CANCELLED
    assert state.terminal_status is RunTerminalStatus.CANCELLED
    assert broker.revocations == [
        (state.planning_attempts[-1].attempt_id, CredentialRevocationCause.CANCELLED)
    ]


def test_planning_submitting_cancellation_bounds_claim_and_survives_restart(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    delegate = FakeScheduler()
    scheduler = CrashBeforeSubmitScheduler(delegate)
    broker = FakeCredentialBroker("planning-cancel-secret")
    clock = MutableClock()
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=1,
        absence_samples_required=2,
        absence_sample_interval_seconds=1,
    )
    engine = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        credential_broker=broker,
        submission_clock=clock,
        submission_recovery_policy=policy,
    )
    engine.tick()
    with pytest.raises(InjectedCrash):
        engine.tick()

    clock.advance(1)
    first = engine.tick(cancel_requested=True, cancellation_reason="operator request")
    assert first.next_action is PlanningAction.WAIT_FOR_SCHEDULER
    restarted = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        credential_broker=broker,
        submission_clock=clock,
        submission_recovery_policy=policy,
    )
    clock.advance(1)
    cancelled_intent = restarted.tick(
        cancel_requested=True,
        cancellation_reason="operator request",
    )
    assert cancelled_intent.event is PlanningEvent.RUN_CANCELLED
    assert cancelled_intent.next_action is PlanningAction.TICK
    assert delegate.submissions == ()
    assert scheduler.submit_calls == 1
    state = load_state(workspace / STATE_FILENAME)
    assert state.planning_attempts[-1].status is AttemptStatus.CANCELLED
    cancellation = (
        workspace
        / "receipts"
        / "submissions"
        / "planning"
        / state.planning_attempts[-1].attempt_id
        / "cancellation.json"
    )
    assert cancellation.is_file()

    terminal = restarted.tick(cancel_requested=True, cancellation_reason="operator request")
    assert terminal.next_action is PlanningAction.STOP
    assert load_state(workspace / STATE_FILENAME).terminal_status is RunTerminalStatus.CANCELLED


def test_planning_cancellation_fails_closed_on_token_ownership_mismatch(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    delegate = FakeScheduler()
    scheduler = OwnershipMismatchScheduler(delegate)
    engine = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=scheduler,
        generation=1,
        credential_broker=FakeCredentialBroker("ownership-mismatch-secret"),
    )
    engine.tick()
    engine.tick()

    with pytest.raises(PlanningError, match="ownership verification failed closed"):
        engine.tick(cancel_requested=True, cancellation_reason="operator request")

    assert delegate.cancelled == ()
    state = load_state(workspace / STATE_FILENAME)
    assert state.terminal_status is RunTerminalStatus.ACTIVE
    assert state.planning_attempts[-1].status is AttemptStatus.SUBMITTED


def test_planning_cancellation_rejects_reused_scheduler_job_id(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    original = FakeScheduler(first_job_id=10_000)
    engine = _engine(task, workspace, original)
    engine.tick()
    engine.tick()
    original_submission = original.submissions[-1]

    recycled = FakeScheduler(first_job_id=10_000)
    recycled_identity = recycled.submit(
        original_submission.resources,
        original_submission.command,
        "recycled-token",
        dict(original_submission.environment),
    )
    assert recycled_identity == original_submission.identity
    restarted = _engine(task, workspace, recycled)

    with pytest.raises(PlanningError, match="ownership verification failed closed"):
        restarted.tick(cancel_requested=True, cancellation_reason="operator request")

    assert recycled.cancelled == ()
    assert load_state(workspace / STATE_FILENAME).terminal_status is RunTerminalStatus.ACTIVE


def test_generation_fence_rejects_stale_controller(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    engine = _engine(task, workspace, FakeScheduler(), generation=2)

    with pytest.raises(PlanningError, match="generation fence"):
        engine.tick()


def test_planning_config_must_match_frozen_task_agent_identity(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    with pytest.raises(PlanningError, match="frozen task.execution.agent"):
        _engine(
            task,
            workspace,
            FakeScheduler(),
            config=PlanningConfig(backend_kind="codex", model="different-model"),
        )


def test_planning_controller_rejects_backend_secret_environment(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace

    with pytest.raises(PlanningError, match="cannot receive backend credential values"):
        _engine(
            task,
            workspace,
            FakeScheduler(),
            backend_environment={"OPENAI_API_KEY": "must-not-enter-controller"},
        )


def test_planning_requires_an_explicit_credential_broker(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace

    with pytest.raises(PlanningError, match="explicit CredentialBroker"):
        PlanningEngine(
            workspace=workspace,
            task=task,
            scheduler=FakeScheduler(),
            generation=1,
        )


def test_planning_descriptor_must_match_selected_role_broker_policy(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    engine = _engine(
        task,
        workspace,
        FakeScheduler(),
        credential_broker=FakeCredentialBroker(
            "mismatched-broker-secret",
            broker_id="other-broker",
        ),
    )

    with pytest.raises(PlanningError, match="broker differs from task role policy"):
        engine.tick()


def test_explicit_empty_planning_policy_accepts_no_credential_broker(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    role_classes = tuple(
        replace(
            role_class,
            credential_broker=CredentialBrokerPolicy("preauthenticated", (), 7_200),
        )
        if role_class.role in {SlurmRole.PLAN_DRAFTER, SlurmRole.PLAN_REVIEWER}
        else role_class
        for role_class in task.execution.slurm.role_classes
    )
    task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(task.execution.slurm, role_classes=role_classes),
        ),
    )
    engine = PlanningEngine(
        workspace=workspace,
        task=task,
        scheduler=FakeScheduler(),
        generation=1,
        credential_broker=NoCredentialBroker(),
    )

    engine.tick()
    attempt = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    spec = load_role_spec(
        workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id / "role-input.json"
    )
    descriptor = descriptor_from_public_dict(spec.credential_descriptor)
    assert descriptor.state is CredentialState.NONE
    assert descriptor.handle is None


def test_planning_submission_uses_narrow_container_visible_worker_paths(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    host_root = task.repository.root.parent
    controller = replace(
        task.execution.slurm.controller,
        mounts=(MountConfig(host_root, "/container/shared", False),),
    )
    task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(task.execution.slurm, controller=controller),
        ),
    )
    scheduler = FakeScheduler()
    engine = _engine(task, workspace, scheduler)

    engine.tick()
    engine.tick()

    submission = scheduler.submissions[-1]
    attempt = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    mailbox = workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id
    assert (
        submission.command.input_bundle
        == Path("/container/shared/workspaces/run/items/__planning__/attempts")
        / attempt.attempt_id
        / "role-input.json"
    )
    code_mounts = [
        mount for mount in submission.resources.mounts if mount.target != CREDENTIAL_MOUNT_PATH
    ]
    assert [mount.source for mount in code_mounts] == [
        task.repository.root,
        mailbox / "output",
        mailbox / "role-input.json",
        task.reference.checkpoint,
        workspace / "worker-surfaces" / "empty-git",
    ]
    metadata_mask = code_mounts[-1]
    assert metadata_mask.read_only
    assert metadata_mask.target == Path("/container/shared/repository/.git")
    assert not any(metadata_mask.source.iterdir())
    assert [mount.read_only for mount in code_mounts] == [True, False, True, True, True]
    role_input = json.loads((mailbox / "role-input.json").read_text(encoding="utf-8"))
    assert role_input["source_root"] == "/container/shared/repository"
    assert role_input["cwd"] == "/container/shared/repository"
    assert role_input["result_path"].startswith(
        "/container/shared/workspaces/run/items/__planning__/attempts/"
    )
    assert "/output/role-result.json" in role_input["result_path"]


def test_planning_credentials_use_private_bundle_and_public_descriptor_only(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, workspace = task_and_workspace
    secret = "sk-planning-controller-secret"
    scheduler = FakeScheduler()
    broker = FakeCredentialBroker(secret)
    engine = _engine(
        task,
        workspace,
        scheduler,
        credential_broker=broker,
        backend_environment={
            "SLURM_JWT": "must-not-propagate",
        },
    )

    engine.tick()
    engine.tick()

    attempt = load_state(workspace / STATE_FILENAME).planning_attempts[-1]
    mailbox = workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id
    role_input_path = mailbox / "role-input.json"
    role_input_text = role_input_path.read_text(encoding="utf-8")
    role_input = json.loads(role_input_text)
    descriptor = descriptor_from_public_dict(role_input["credential_descriptor"])
    assert descriptor.state is CredentialState.BUNDLE
    assert descriptor.binding.run_id == "staircase-test-run"
    assert descriptor.binding.item_id == "planning"
    assert descriptor.binding.attempt_id == attempt.attempt_id
    assert descriptor.binding.task_digest == task.digest
    assert descriptor.binding.generation == 1
    assert descriptor.binding.backend_kind == task.execution.agent.backend_kind
    assert descriptor.credential_names == ("OPENAI_API_KEY",)
    assert descriptor.handle is not None
    assert descriptor.handle.broker_id == "test-broker"
    assert descriptor.handle.handle_id == attempt.attempt_id
    assert int(time.time()) < descriptor.handle.expires_at_epoch_seconds <= int(time.time()) + 7_200
    assert secret not in role_input_text

    submission = scheduler.submissions[-1]
    assert submission.environment == (("PYTHONNOUSERSITE", "1"),)
    credential_mounts = [
        mount for mount in submission.resources.mounts if mount.target == CREDENTIAL_MOUNT_PATH
    ]
    assert len(credential_mounts) == 1
    assert credential_mounts[0].read_only
    assert credential_mounts[0].source.is_relative_to(workspace / "controller-secrets")
    assert stat.S_IMODE(credential_mounts[0].source.stat().st_mode) == 0o600
    assert secret not in json.dumps(asdict(submission.command), default=str)

    bundle_path = credential_mounts[0].source
    _finish_current_attempt(engine, scheduler, _plan_json())
    assert not bundle_path.exists()
    receipts = tuple(
        (workspace / "controller-secrets" / "credential-revocation-receipts").rglob(
            f"{attempt.attempt_id}.json"
        )
    )
    assert len(receipts) == 1
    assert broker.revocations == [(attempt.attempt_id, CredentialRevocationCause.TERMINAL)]


def test_strict_draft_digest_matches_controller_parser() -> None:
    outcome = parse_plan_draft(
        _plan_json(),
        allowed_resource_classes={"coder_analysis"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="onboard",
        target_features=(),
    )
    assert len(outcome.digest) == 64


def test_production_input_resolver_is_approved_task_bound_and_ordered(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, _workspace = task_and_workspace
    task = _production_gate_task(task)
    payload = json.loads(_plan_json())
    gate_ids = (
        "claims-routing-no-stale",
        "native-contract",
        "entry-gpu",
        "boot",
        "accuracy",
    )
    payload["stages"][0]["exit_gates"] = list(gate_ids)
    plan = parse_plan_draft(
        json.dumps(payload),
        allowed_resource_classes={"coder_analysis"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="onboard",
        target_features=(),
    )
    review = parse_plan_review(
        json.dumps(
            {
                "schema_version": 2,
                "outcome": "ACCEPT",
                "plan_digest": plan.digest,
                "corrections": [],
            }
        ),
        expected_plan_digest=plan.digest,
    )

    production = resolve_production_inputs(task, WorkflowMode.ONBOARD, plan, review)

    assert production.ordered_gate_ids == gate_ids
    assert production.assembler_features == ()
    assert production.tuning_hypotheses == ()
    assert production.qa_required_item_ids == frozenset()
    assert production.gate_scope_intents[0].execution is GateExecutionIntent.CPU_STATIC
    assert production.gate_scope_intents[0].resource.gpus_per_node == 0
    assert production.gate_scope_intents[0].resource.partition is None
    assert production.gate_scope_intents[0].resource.qos is None
    assert production.gate_scope_intents[2].execution is GateExecutionIntent.SINGLE_GPU_PRODUCT
    assert production.gate_scope_intents[2].resource.gpus_per_node == 1
    assert production.gate_scope_intents[2].resource.partition == "batch"
    assert production.gate_scope_intents[2].resource.qos == "normal"
    assert production.gate_scope_intents[-1].execution is GateExecutionIntent.TARGET_PRODUCT
    assert production.gate_scope_intents[-1].resource.tasks_per_node == 1
    assert production.gate_scope_intents[-1].resource.partition == "batch"
    assert production.gate_scope_intents[-1].resource.qos == "normal"
    assert production.evidence_path_policy.paths_for("qa") == ("reports/qa.json",)

    smith = task.execution.slurm.smith
    padded_resources = tuple(
        replace(
            resource,
            gpus_per_node=4,
            gpu_allocation_padding=True,
        )
        if resource.name == "deterministic_gate"
        else resource
        for resource in smith.resource_classes
    )
    padded_task = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(
                task.execution.slurm,
                smith=replace(smith, resource_classes=padded_resources),
            ),
        ),
    )

    padded_production = resolve_production_inputs(
        padded_task,
        WorkflowMode.ONBOARD,
        plan,
        review,
    )

    entry_gpu = padded_production.gate_scope_intents[2].resource
    target_product = padded_production.gate_scope_intents[-1].resource
    assert (entry_gpu.gpus_per_node, entry_gpu.gpu_allocation_padding) == (4, True)
    assert (target_product.gpus_per_node, target_product.gpu_allocation_padding) == (4, True)

    payload["stages"][0]["exit_gates"] = ["boot"]
    incomplete = parse_plan_draft(
        json.dumps(payload),
        allowed_resource_classes={"coder_analysis"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="onboard",
        target_features=(),
    )
    incomplete_review = replace(review, plan_digest=incomplete.digest)
    with pytest.raises(ProductionInputError, match="exactly preserve"):
        resolve_production_inputs(task, WorkflowMode.ONBOARD, incomplete, incomplete_review)


def test_production_input_resolver_preserves_feature_mapping(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, _workspace = task_and_workspace
    task = _production_gate_task(task)
    task = replace(task, target=replace(task.target, features=("mtp",)))
    payload = json.loads(_plan_json())
    payload["stages"][0]["exit_gates"] = [
        "claims-routing-no-stale",
        "native-contract",
        "entry-gpu",
        "boot",
        "accuracy",
    ]
    payload["items"][0].update(
        {
            "kind": "assemble_feature",
            "domain_input": {"feature": "mtp"},
            "entry_ids": [],
        }
    )
    plan = parse_plan_draft(
        json.dumps(payload),
        allowed_resource_classes={"coder_analysis"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="onboard",
        target_features=task.target.features,
    )
    review = parse_plan_review(
        json.dumps(
            {
                "schema_version": 2,
                "outcome": "ACCEPT",
                "plan_digest": plan.digest,
                "corrections": [],
            }
        ),
        expected_plan_digest=plan.digest,
    )

    production = resolve_production_inputs(task, WorkflowMode.ONBOARD, plan, review)

    assert production.assembler_features == (("search-attention", "mtp"),)
    assert production.qa_required_item_ids == frozenset({"search-attention"})


def test_production_input_resolver_preserves_tuning_hypothesis(
    task_and_workspace: tuple[NormalizedTask, Path],
) -> None:
    task, _workspace = task_and_workspace
    task = _production_gate_task(task)
    payload = json.loads(_plan_json())
    payload["stages"][0]["exit_gates"] = [
        "claims-routing-no-stale",
        "native-contract",
        "entry-gpu",
        "boot",
        "accuracy",
    ]
    payload["items"][0].update(
        {
            "kind": "tune_hypothesis",
            "entry_ids": [],
            "domain_input": {
                "hypothesis_id": "attention-tile",
                "item_id": "search-attention",
                "statement": "A larger tile improves throughput.",
                "metric": "tokens_per_second",
                "direction": "higher_is_better",
                "changes": [
                    {
                        "name": "attention.tile_size",
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
    plan = parse_plan_draft(
        json.dumps(payload),
        allowed_resource_classes={"coder_analysis"},
        allowed_path_roots=("tensorrt_llm/_torch/modeling_v2",),
        workflow_mode="tune",
        target_features=(),
    )
    review = parse_plan_review(
        json.dumps(
            {
                "schema_version": 2,
                "outcome": "ACCEPT",
                "plan_digest": plan.digest,
                "corrections": [],
            }
        ),
        expected_plan_digest=plan.digest,
    )

    production = resolve_production_inputs(task, WorkflowMode.TUNE, plan, review)

    assert len(production.tuning_hypotheses) == 1
    assert production.tuning_hypotheses[0].item_id == "search-attention"
    assert production.tuning_hypotheses[0].change.baseline_value == 64
    assert production.qa_required_item_ids == frozenset({"search-attention"})
