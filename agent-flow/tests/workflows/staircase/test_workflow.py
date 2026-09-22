# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Staircase bootstrap and public workflow operations."""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest
import yaml

from agent_flow.workflows.staircase import workflow as staircase_workflow
from agent_flow.workflows.staircase.common.artifacts import (
    WorkerInputManifest,
    write_input_manifest,
)
from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialDescriptor,
    CredentialHandle,
    CredentialRevocationCause,
    CredentialRevocationReceipt,
    CredentialState,
    prepare_credential_provision,
    revoke_terminal_attempt_credentials,
)
from agent_flow.workflows.staircase.common.launch_policy import AgentLaunchPolicy
from agent_flow.workflows.staircase.common.runners import RoleProcessSpec
from agent_flow.workflows.staircase.common.slurm import (
    DependencyType,
    FakeScheduler,
    InternalCommand,
    InternalEntrypoint,
    JobDependency,
    JobIdentity,
    JobStatus,
    ObservationSource,
    ResourceRequest,
    SchedulerError,
    SubmissionProbe,
)
from agent_flow.workflows.staircase.common.submission_recovery import (
    SubmissionRecoveryAction,
    SubmissionRecoveryPolicy,
    decide_submission_recovery,
    submission_intent_digest,
)
from agent_flow.workflows.staircase.controller.runtime import (
    RuntimeDisposition,
    RuntimeEvent,
    RuntimeTickResult,
)
from agent_flow.workflows.staircase.state import (
    PLANNING_ITEM_ID,
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    ControllerBootstrapStatus,
    ControllerLease,
    ControllerLifecycle,
    DomainProfile,
    GoalRecord,
    HierarchyStatus,
    JobReference,
    LeaseConflictError,
    OrphanReceiptAction,
    Role,
    RunState,
    RunTerminalStatus,
    StageRecord,
    StageStatus,
    WorkItemKind,
    WorkItemRecord,
    WorkItemStatus,
    initialize_state,
    load_state,
    save_state,
)
from agent_flow.workflows.staircase.task_schema import load_and_normalize_task
from agent_flow.workflows.staircase.workflow import (
    WorkflowError,
    _adopt_orphaned_attempts,
    _bootstrap_initial_controller,
    _ControllerSignalFlags,
    _fence_unadopted_orphan_credentials,
    _gate_spec_for_terminal_attempt,
    _WorkspaceRecoveryCredentialBroker,
    load_observed_preflight_facts,
    record_human_response,
    render_status,
    request_cancellation,
    run_controller,
    start_run,
)

from .test_task_schema import _valid_task


def _task(
    tmp_path: Path,
    *,
    execution: str = "slurm",
    credentialed: bool = True,
):
    task_path = tmp_path / "task.yaml"
    raw = _valid_task(tmp_path)
    if not credentialed:
        slurm = raw["execution"]["slurm"]  # type: ignore[index]
        controller = slurm["controller"]  # type: ignore[index]
        del controller["credential_broker_socket"]  # type: ignore[index]
        for role in slurm["role_classes"].values():  # type: ignore[index,union-attr]
            role["credential_broker"] = {  # type: ignore[index]
                "broker_id": "preauthenticated",
                "allowed_credential_names": [],
                "per_attempt_ttl_seconds": 7_200,
            }
    task_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_and_normalize_task(task_path, execution_override=execution)


def _skip_preflight(_task: object, _phase: str) -> None:
    """Explicit CPU-test seam that never probes the host Slurm installation."""


def _start(
    *,
    mode: str,
    task: Any,
    workspace: Path,
    scheduler: FakeScheduler | None = None,
):
    return start_run(
        mode=mode,
        task=task,
        workspace=workspace,
        scheduler=scheduler,
        preflight_check=_skip_preflight,
    )


def _bootstrap_state(task: Any, workspace: Path, scheduler: FakeScheduler, job: object) -> None:
    _bootstrap_initial_controller(
        workspace=workspace,
        state_path=workspace / "state.json",
        task=task,
        generation=1,
        current_job=job,  # type: ignore[arg-type]
        scheduler=scheduler,
    )


def _credentialed_emergency_fixture(
    tmp_path: Path,
    *,
    hidden_role: Role | None = None,
) -> tuple[
    Any,
    Path,
    FakeScheduler,
    tuple[JobIdentity, ...],
    tuple[CredentialDescriptor, ...],
]:
    """Create active planning/Coder/Reviewer/QA attempts with public handles."""
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    state_path = workspace / "state.json"
    state = load_state(state_path)
    resources = scheduler.submissions[0].resources
    roles = (
        (PLANNING_ITEM_ID, "planning-active", Role.PLAN_DRAFTER, None),
        ("coder-item", "coder-active", Role.CODER, DomainProfile.SMITH),
        ("reviewer-item", "reviewer-active", Role.REVIEWER, DomainProfile.SMITH),
        ("qa-item", "qa-active", Role.QA, DomainProfile.SMITH),
    )
    attempts: list[AttemptRecord] = []
    jobs: list[JobIdentity] = []
    descriptors: list[CredentialDescriptor] = []
    items: list[WorkItemRecord] = []
    for sequence, (item_id, attempt_id, role, profile) in enumerate(roles, start=1):
        token = f"emergency-token-{sequence}"
        job = scheduler.submit(
            resources,
            InternalCommand(
                InternalEntrypoint.WORKER,
                input_bundle=(workspace / f"emergency-input-{sequence}.json").resolve(),
            ),
            token,
        )
        binding = CredentialBinding(
            run_id=state.run_id,
            item_id="planning" if item_id == PLANNING_ITEM_ID else item_id,
            attempt_id=attempt_id,
            task_digest=state.task_digest,
            generation=state.generation,
            backend_kind=task.execution.agent.backend_kind,
        )
        descriptor = CredentialDescriptor.create(
            binding,
            CredentialState.BUNDLE,
            ("OPENAI_API_KEY",),
            handle=CredentialHandle(
                "site-codex-broker",
                f"emergency-handle-{sequence}",
                int(time.time()) + 7_200,
            ),
        )
        hidden = role is hidden_role
        attempt = AttemptRecord(
            attempt_id,
            item_id,
            1,
            role,
            AttemptKind.ROLE,
            state.generation,
            AttemptStatus.SUBMITTING if hidden else AttemptStatus.RUNNING,
            profile,
            token,
            None if hidden else JobReference(job.job_id, cluster=job.cluster),
            resource_class="coder_analysis",
        )
        attempt_root = workspace / "items" / item_id / "attempts"
        if item_id == PLANNING_ITEM_ID:
            input_path = attempt_root / attempt_id / "role-input.json"
            input_path.parent.mkdir(parents=True)
            input_path.write_text(
                json.dumps(
                    {"credential_descriptor": descriptor.to_public_dict()},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            input_path = attempt_root / "0001" / "input.json"
            write_input_manifest(
                input_path,
                WorkerInputManifest(
                    state.run_id,
                    item_id,
                    attempt_id,
                    state.task_digest,
                    state.generation,
                    role,
                    profile,
                    str(workspace / "candidates" / item_id / attempt_id),
                    payload={
                        "runtime": {
                            "kind": "agent",
                            "credential_descriptor": descriptor.to_public_dict(),
                        },
                        "context": {},
                    },
                ),
            )
            items.append(
                WorkItemRecord(
                    item_id,
                    "emergency-stage",
                    "emergency-goal",
                    WorkItemKind.CATALOG_VERIFY,
                    DomainProfile.SMITH,
                    status=WorkItemStatus.CODING,
                    attempts=(attempt,),
                )
            )
        attempts.append(attempt)
        jobs.append(job)
        descriptors.append(descriptor)
    desired = dataclasses.replace(
        state,
        revision=state.revision + 1,
        stages=(StageRecord("emergency-stage", ("emergency-goal",)),),
        goals=(
            GoalRecord(
                "emergency-goal",
                "emergency-stage",
                tuple(item.item_id for item in items),
            ),
        ),
        items=tuple(items),
        planning_attempts=(attempts[0],),
    )
    save_state(
        state_path,
        desired,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )
    return task, workspace, scheduler, tuple(jobs), tuple(descriptors)


def _hidden_submitting_emergency_fixture(
    tmp_path: Path,
    *,
    scheduler: FakeScheduler | None = None,
) -> tuple[Any, Path, FakeScheduler, AttemptRecord]:
    task = _task(tmp_path, credentialed=False)
    active_scheduler = scheduler or FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(
        mode="onboard",
        task=task,
        workspace=workspace,
        scheduler=active_scheduler,
    )
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, active_scheduler, started.controller_job)
    state_path = workspace / "state.json"
    state = load_state(state_path)
    attempt = AttemptRecord(
        "hidden-submitting",
        "item-1",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        state.generation,
        AttemptStatus.SUBMITTING,
        DomainProfile.SMITH,
        "hidden-submission-token",
        resource_class="coder_analysis",
    )
    desired = dataclasses.replace(
        state,
        revision=state.revision + 1,
        stages=(StageRecord("stage-1", ("goal-1",)),),
        goals=(GoalRecord("goal-1", "stage-1", ("item-1",)),),
        items=(
            WorkItemRecord(
                "item-1",
                "stage-1",
                "goal-1",
                WorkItemKind.CATALOG_VERIFY,
                DomainProfile.SMITH,
                status=WorkItemStatus.CODING,
                attempts=(attempt,),
            ),
        ),
    )
    save_state(
        state_path,
        desired,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    active_scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )
    return task, workspace, active_scheduler, attempt


def _write_hidden_submission_claim(
    workspace: Path,
    attempt: AttemptRecord,
    *,
    claimed_at: datetime,
) -> None:
    token = attempt.submission_token
    assert token is not None
    intent_revision = f"{attempt.generation}:{attempt.sequence}:{attempt.attempt_id}"
    intent_digest = submission_intent_digest("test-hidden-submission", intent_revision, token)
    journal_root = workspace / "receipts" / "submissions"
    if attempt.item_id == PLANNING_ITEM_ID:
        journal_root = journal_root / "planning"
    decision = decide_submission_recovery(
        journal_dir=journal_root / attempt.attempt_id,
        submission_token=token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
        probe=FakeScheduler(cluster="alpha").probe_submission(token),
        clock=lambda: claimed_at,
        policy=SubmissionRecoveryPolicy(),
    )
    assert decision.action is SubmissionRecoveryAction.SUBMIT


def _submit_hidden_attempt(
    scheduler: FakeScheduler,
    workspace: Path,
    attempt: AttemptRecord,
) -> JobIdentity:
    token = attempt.submission_token
    assert token is not None
    return scheduler.submit(
        scheduler.submissions[0].resources,
        InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=(workspace / f"{attempt.attempt_id}.json").resolve(),
        ),
        token,
    )


def _gate_attempt(
    attempt_id: str,
    sequence: int,
    status: AttemptStatus,
    *,
    predecessor_attempt_id: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        item_id="item-1",
        sequence=sequence,
        role=Role.GATE,
        kind=AttemptKind.DETERMINISTIC_GATE,
        generation=1,
        status=status,
        profile=DomainProfile.SMITH,
        submission_token=f"submit-{attempt_id}",
        job=JobReference(str(800 + sequence)),
        result_digest=(f"{sequence:x}" * 64)[:64] if status is AttemptStatus.VALIDATED else None,
        terminal_reason=None
        if status is AttemptStatus.VALIDATED
        else f"scheduler_{status.value.lower()}",
        resource_class="deterministic_gate",
        predecessor_attempt_id=predecessor_attempt_id,
    )


def _gate_item(*attempts: AttemptRecord) -> WorkItemRecord:
    return WorkItemRecord(
        "item-1",
        "stage-1",
        "goal-1",
        WorkItemKind.CATALOG_VERIFY,
        DomainProfile.SMITH,
        status=WorkItemStatus.CODING,
        attempts=attempts,
    )


def test_slurm_start_writes_only_launcher_mailboxes_until_controller_bootstrap(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler()
    workspace = task.repository.workspace_root / "run"
    result = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert result.controller_job is not None
    assert not (workspace / "state.json").exists()
    intent = json.loads((workspace / "launcher/initial.intent.json").read_text())
    receipt = json.loads((workspace / "launcher/initial.receipt.json").read_text())
    assert intent["submission_token"] == scheduler.submissions[0].submission_token
    assert receipt["job"]["job_id"] == result.controller_job.job_id

    _bootstrap_state(task, workspace, scheduler, result.controller_job)
    state = load_state(workspace / "state.json")
    assert state.controller_bootstrap_status is ControllerBootstrapStatus.SUBMITTED
    assert state.controller_submission_token == scheduler.submissions[0].submission_token
    assert state.controller_job is not None
    assert state.controller_job.job_id == result.controller_job.job_id
    assert state.workflow_mode.value == "onboard"


def test_start_rejects_legacy_agent_team_workspace_without_mutating_it(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, execution="local")
    workspace = task.repository.workspace_root / "legacy-run"
    workspace.mkdir(parents=True)
    legacy_state = workspace / ".agent_team_state.json"
    legacy_state.write_text('{"done": true}\n', encoding="utf-8")
    before = legacy_state.read_bytes()

    with pytest.raises(WorkflowError, match="legacy AgentTeam workspace"):
        _start(mode="onboard", task=task, workspace=workspace)

    assert legacy_state.read_bytes() == before
    assert sorted(path.name for path in workspace.iterdir()) == [legacy_state.name]


def test_terminal_gate_retry_maps_to_original_logical_gate_across_restart() -> None:
    first = _gate_attempt("gate-1", 1, AttemptStatus.PREEMPTED)
    retry = _gate_attempt(
        "gate-1-retry",
        2,
        AttemptStatus.VALIDATED,
        predecessor_attempt_id=first.attempt_id,
    )
    item = _gate_item(first, retry)
    gates = (object(), object())

    resolved = _gate_spec_for_terminal_attempt(item, retry, gates)

    assert resolved is gates[0]
    assert _gate_spec_for_terminal_attempt(item, retry, gates) is resolved


def test_terminal_gate_exhausted_retries_do_not_shift_following_gate() -> None:
    first = _gate_attempt("gate-1", 1, AttemptStatus.PREEMPTED)
    exhausted = _gate_attempt(
        "gate-1-retry",
        2,
        AttemptStatus.RETRYABLE_FAILED,
        predecessor_attempt_id=first.attempt_id,
    )
    second_gate = _gate_attempt("gate-2", 3, AttemptStatus.VALIDATED)
    item = _gate_item(first, exhausted, second_gate)
    gates = (object(), object())

    assert _gate_spec_for_terminal_attempt(item, second_gate, gates) is gates[1]


def test_terminal_gate_multiple_retry_lineages_never_cross_bind() -> None:
    first = _gate_attempt("gate-1", 1, AttemptStatus.PREEMPTED)
    second = _gate_attempt("gate-2", 2, AttemptStatus.PREEMPTED)
    first_retry = _gate_attempt(
        "gate-1-retry",
        3,
        AttemptStatus.VALIDATED,
        predecessor_attempt_id=first.attempt_id,
    )
    second_retry = _gate_attempt(
        "gate-2-retry",
        4,
        AttemptStatus.VALIDATED,
        predecessor_attempt_id=second.attempt_id,
    )
    item = _gate_item(first, second, first_retry, second_retry)
    gates = (object(), object())

    assert _gate_spec_for_terminal_attempt(item, first_retry, gates) is gates[0]
    assert _gate_spec_for_terminal_attempt(item, second_retry, gates) is gates[1]


def test_terminal_gate_branching_retry_lineage_fails_closed() -> None:
    first = _gate_attempt("gate-1", 1, AttemptStatus.PREEMPTED)
    retry = _gate_attempt(
        "gate-1-retry-a",
        2,
        AttemptStatus.VALIDATED,
        predecessor_attempt_id=first.attempt_id,
    )
    branch = _gate_attempt(
        "gate-1-retry-b",
        3,
        AttemptStatus.PREEMPTED,
        predecessor_attempt_id=first.attempt_id,
    )

    with pytest.raises(ValueError, match="only one child"):
        _gate_item(first, retry, branch)


def test_resume_rejects_a_different_workflow_mode(tmp_path: Path) -> None:
    task = _task(tmp_path)
    workspace = task.repository.workspace_root / "run"
    _start(mode="onboard", task=task, workspace=workspace, scheduler=FakeScheduler())
    with pytest.raises(WorkflowError, match="workspace launcher"):
        _start(mode="tune", task=task, workspace=workspace, scheduler=FakeScheduler())


def test_resume_adopts_missing_launcher_receipt_by_token_without_state_write(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler()
    workspace = task.repository.workspace_root / "run"
    result = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    (workspace / "launcher/initial.receipt.json").unlink()
    adopted = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert adopted.adopted is True
    assert adopted.controller_job == result.controller_job
    assert len(scheduler.submissions) == 1
    assert not (workspace / "state.json").exists()
    assert (workspace / "launcher/initial.receipt.json").is_file()


def test_ambiguous_submission_never_resubmits(tmp_path: Path) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler()
    workspace = task.repository.workspace_root / "run"
    _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    (workspace / "launcher/initial.receipt.json").unlink()
    first = scheduler.submissions[0]
    scheduler.submit(
        first.resources, first.command, first.submission_token, dict(first.environment)
    )
    with pytest.raises(WorkflowError, match="quarantined"):
        _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert len(scheduler.submissions) == 2


def test_resume_rejects_task_digest_drift(tmp_path: Path) -> None:
    task = _task(tmp_path)
    workspace = task.repository.workspace_root / "run"
    _start(mode="onboard", task=task, workspace=workspace, scheduler=FakeScheduler())
    drifted = dataclasses.replace(task, digest="b" * 64)
    with pytest.raises(WorkflowError, match="task digest"):
        _start(mode="onboard", task=drifted, workspace=workspace, scheduler=FakeScheduler())


def test_status_reconciles_without_mutating_state(tmp_path: Path) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler()
    workspace = task.repository.workspace_root / "run"
    result = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert result.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, result.controller_job)
    before = (workspace / "state.json").read_bytes()
    assert result.controller_job is not None
    scheduler.transition(result.controller_job, JobStatus.RUNNING)
    payload = json.loads(render_status(workspace, as_json=True, scheduler=scheduler))
    assert payload["scheduler_observation"]["status"] == "RUNNING"
    assert (workspace / "state.json").read_bytes() == before


def test_status_projects_launcher_receipt_before_controller_bootstrap(tmp_path: Path) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    result = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert result.controller_job is not None

    payload = json.loads(render_status(workspace, as_json=True, scheduler=scheduler))

    assert payload["authoritative_state"] == "not_bootstrapped"
    assert payload["controller_job"]["cluster"] == "alpha"
    assert payload["scheduler_observation"]["status"] == "PENDING"
    assert not (workspace / "state.json").exists()


def test_status_projects_child_scheduler_mailboxes_lease_and_drift_read_only(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    child = scheduler.submit(
        scheduler.submissions[0].resources,
        InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=(workspace / "child-input.json").resolve(),
        ),
        "worker-status-0001",
    )
    scheduler.transition(child, JobStatus.COMPLETED, source=ObservationSource.ACCOUNTING)
    state_path = workspace / "state.json"
    state = load_state(state_path)
    attempt = AttemptRecord(
        "attempt-status",
        "item-status",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        AttemptStatus.RUNNING,
        DomainProfile.SMITH,
        "worker-status-0001",
        JobReference(child.job_id, cluster=child.cluster),
        resource_class="coder_analysis",
    )
    desired = dataclasses.replace(
        state,
        revision=state.revision + 1,
        stages=(StageRecord("stage-status", ("goal-status",)),),
        goals=(GoalRecord("goal-status", "stage-status", ("item-status",)),),
        items=(
            WorkItemRecord(
                "item-status",
                "stage-status",
                "goal-status",
                WorkItemKind.CATALOG_VERIFY,
                DomainProfile.SMITH,
                status=WorkItemStatus.CODING,
                attempts=(attempt,),
            ),
        ),
    )
    save_state(
        state_path,
        desired,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    mailbox = workspace / "items/item-status/attempts/0001"
    mailbox.mkdir(parents=True)
    (mailbox / "input.json").write_text("{}\n", encoding="utf-8")
    output = mailbox / "output"
    output.mkdir()
    for name in ("COMPLETE", "result.json"):
        (output / name).write_text("{}\n", encoding="utf-8")
    lease = ControllerLease.acquire(
        workspace / "controller-lease.json",
        run_id=desired.run_id,
        generation=desired.generation,
        controller_job=desired.controller_job,
        owner_nonce="status-live-owner",
    )
    before = state_path.read_bytes()
    try:
        payload = json.loads(render_status(workspace, as_json=True, scheduler=scheduler))
    finally:
        lease.release(remove_record=False)

    projected = payload["projection"]
    assert projected["lease"]["lock_status"] == "held"
    assert projected["lease"]["consistent"] is True
    projected_attempt = projected["attempts"][0]
    assert projected_attempt["scheduler_status"] == "COMPLETED"
    assert all(
        projected_attempt["mailbox"][name]["present"] for name in ("input", "complete", "result")
    )
    assert {entry["kind"] for entry in projected["drift"]} == {
        "scheduler_terminal_state_active",
        "result_ready_unadopted",
    }
    assert state_path.read_bytes() == before

    class _ReadErrorScheduler:
        def observe_owned(
            self,
            identity: JobIdentity,
            submission_token: str,
        ):
            if identity == child:
                raise SchedulerError("simulated child observation failure")
            return scheduler.observe_owned(identity, submission_token)

    failed_projection = json.loads(
        render_status(
            workspace,
            as_json=True,
            scheduler=_ReadErrorScheduler(),  # type: ignore[arg-type]
        )
    )["projection"]
    assert failed_projection["attempts"][0]["scheduler_status"] == "ERROR"
    assert failed_projection["read_errors"][0]["surface"] == ("attempt.attempt-status.scheduler")


def test_cancel_writes_durable_request_without_calling_scheduler(tmp_path: Path) -> None:
    task = _task(tmp_path, execution="local")
    workspace = task.repository.workspace_root / "run"
    _start(mode="onboard", task=task, workspace=workspace)
    request_cancellation(workspace, reason="operator")
    request = json.loads((workspace / "requests/cancel.json").read_text(encoding="utf-8"))
    assert request["reason"] == "operator"
    assert load_state(workspace / "state.json").terminal_status is RunTerminalStatus.ACTIVE


def test_human_response_requires_matching_durable_request(tmp_path: Path) -> None:
    task = _task(tmp_path, execution="local")
    workspace = task.repository.workspace_root / "run"
    result = _start(mode="onboard", task=task, workspace=workspace)
    with pytest.raises(WorkflowError, match="unknown human-input request"):
        record_human_response(workspace, request_id="request-1", response="answer")
    request_path = workspace / "requests/human/request-1.request.json"
    request_path.write_text(
        json.dumps(
            {"run_id": result.run_id, "task_digest": task.digest, "request_id": "request-1"}
        ),
        encoding="utf-8",
    )
    with pytest.raises(WorkflowError, match="absent from authoritative state"):
        record_human_response(workspace, request_id="request-1", response="answer")
    assert not (workspace / "requests/human/request-1.response.json").exists()

    state_path = workspace / "state.json"
    state = load_state(state_path)
    waiting = state.wait_for_human_input(
        request_id="request-1",
        prompt_digest="a" * 64,
        controller_requeue=False,
        reason="operator decision required",
    )
    save_state(
        state_path,
        waiting,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    record_human_response(workspace, request_id="request-1", response="answer")
    response = json.loads(
        (workspace / "requests/human/request-1.response.json").read_text(encoding="utf-8")
    )
    assert response["response"] == "answer"


def test_workspace_must_be_inside_declared_shared_root(tmp_path: Path) -> None:
    task = _task(tmp_path, execution="local")
    with pytest.raises(WorkflowError, match="workspace_root"):
        _start(mode="onboard", task=task, workspace=tmp_path / "outside")


def test_slurm_start_fails_closed_without_observed_preflight_facts(tmp_path: Path) -> None:
    task = _task(tmp_path)
    workspace = task.repository.workspace_root / "run"

    with pytest.raises(WorkflowError, match="independently observed"):
        start_run(
            mode="onboard",
            task=task,
            workspace=workspace,
            scheduler=FakeScheduler(),
        )

    assert not workspace.exists()


def test_observed_preflight_facts_manifest_is_strict_and_independent(tmp_path: Path) -> None:
    task = _task(tmp_path)
    facts_path = tmp_path / "observed.json"
    facts_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository_root": str(task.repository.root),
                "workspace_root": str(task.repository.workspace_root),
                "container_image": str(task.execution.slurm.controller.image),
                "build_identity": "independently-observed-build",
                "mounts": [
                    {
                        "host_path": str(mount.host_path),
                        "container_path": mount.container_path,
                        "read_only": mount.read_only,
                    }
                    for mount in task.execution.slurm.controller.mounts
                ],
            }
        ),
        encoding="utf-8",
    )

    facts = load_observed_preflight_facts(facts_path)

    assert facts.build_identity == "independently-observed-build"
    assert facts.repository_root == task.repository.root
    raw = json.loads(facts_path.read_text())
    raw["untrusted_extra"] = True
    facts_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(WorkflowError, match="unknown fields"):
        load_observed_preflight_facts(facts_path)


class _WaitingRuntime:
    def __init__(self, *, workspace: Path, **_kwargs: object) -> None:
        self._workspace = workspace

    def tick(self) -> RuntimeTickResult:
        state = load_state(self._workspace / "state.json")
        return RuntimeTickResult(
            RuntimeDisposition.WAITING_INPUT,
            RuntimeEvent.WAITING_INPUT,
            state.revision,
            "choose the validated architecture mapping",
        )


class _TerminalRuntime:
    def __init__(self, *, workspace: Path, **_kwargs: object) -> None:
        self._workspace = workspace

    def tick(self) -> RuntimeTickResult:
        state_path = self._workspace / "state.json"
        state = load_state(state_path)
        terminal = state.finish(RunTerminalStatus.CANCELLED, "test terminal outcome")
        save_state(
            state_path,
            terminal,
            expected_revision=state.revision,
            expected_generation=state.generation,
        )
        return RuntimeTickResult(
            RuntimeDisposition.TERMINAL,
            RuntimeEvent.TERMINAL,
            terminal.revision,
            "test terminal outcome",
            terminal_status=RunTerminalStatus.CANCELLED,
        )


class _CrashAfterAcceptedSubmit(FakeScheduler):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.crash_after_next_submit = False

    def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        identity = super().submit(*args, **kwargs)  # type: ignore[arg-type]
        if self.crash_after_next_submit:
            self.crash_after_next_submit = False
            raise SchedulerError("simulated crash after scheduler acceptance")
        return identity


class _CrashBeforeLauncherSubmit(FakeScheduler):
    crash_before_submit = True

    def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self.crash_before_submit:
            self.crash_before_submit = False
            raise RuntimeError("simulated launcher crash before scheduler call")
        return super().submit(*args, **kwargs)  # type: ignore[arg-type]


def test_initial_launcher_adopts_acceptance_after_lost_submit_reply(tmp_path: Path) -> None:
    task = _task(tmp_path)
    scheduler = _CrashAfterAcceptedSubmit(cluster="alpha")
    scheduler.crash_after_next_submit = True
    workspace = task.repository.workspace_root / "run"

    with pytest.raises(WorkflowError, match="bounded token reconciliation"):
        _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)

    assert len(scheduler.submissions) == 1
    assert not (workspace / "state.json").exists()
    adopted = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert adopted.controller_job == scheduler.submissions[0].identity
    assert len(scheduler.submissions) == 1
    assert (workspace / "launcher/initial.receipt.json").is_file()


def test_initial_launcher_claim_crash_retries_after_bounded_absence_proof(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    scheduler = _CrashBeforeLauncherSubmit(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=10,
        absence_samples_required=2,
        absence_sample_interval_seconds=2,
        max_submit_calls=2,
    )

    with pytest.raises(RuntimeError, match="launcher crash"):
        start_run(
            mode="onboard",
            task=task,
            workspace=workspace,
            scheduler=scheduler,
            preflight_check=_skip_preflight,
            submission_clock=lambda: now,
            submission_recovery_policy=policy,
        )
    assert scheduler.submissions == ()

    with pytest.raises(WorkflowError, match="absence evidence"):
        start_run(
            mode="onboard",
            task=task,
            workspace=workspace,
            scheduler=scheduler,
            preflight_check=_skip_preflight,
            submission_clock=lambda: now + timedelta(seconds=10),
            submission_recovery_policy=policy,
        )
    recovered = start_run(
        mode="onboard",
        task=task,
        workspace=workspace,
        scheduler=scheduler,
        preflight_check=_skip_preflight,
        submission_clock=lambda: now + timedelta(seconds=12),
        submission_recovery_policy=policy,
    )
    assert recovered.controller_job is not None
    assert len(scheduler.submissions) == 1


def test_initial_launcher_second_claim_crash_ends_in_manual_recovery(tmp_path: Path) -> None:
    class _TwoCrashes(FakeScheduler):
        crashes_remaining = 2

        def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.crashes_remaining:
                self.crashes_remaining -= 1
                raise RuntimeError("simulated launcher process crash")
            return super().submit(*args, **kwargs)  # type: ignore[arg-type]

    task = _task(tmp_path)
    scheduler = _TwoCrashes(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=10,
        absence_samples_required=2,
        absence_sample_interval_seconds=2,
        max_submit_calls=2,
    )

    def start_at(offset: int):
        return start_run(
            mode="onboard",
            task=task,
            workspace=workspace,
            scheduler=scheduler,
            preflight_check=_skip_preflight,
            submission_clock=lambda: now + timedelta(seconds=offset),
            submission_recovery_policy=policy,
        )

    with pytest.raises(RuntimeError, match="process crash"):
        start_at(0)
    with pytest.raises(WorkflowError):
        start_at(10)
    with pytest.raises(RuntimeError, match="process crash"):
        start_at(12)
    with pytest.raises(WorkflowError):
        start_at(22)
    with pytest.raises(WorkflowError, match="exhausted"):
        start_at(24)

    manual = tuple((workspace / "launcher/submission-journals").glob("*/manual-recovery.json"))
    assert len(manual) == 1
    assert scheduler.submissions == ()


def test_signal_checkpoints_with_live_workers_and_successor_finishes_terminal_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)

    child = scheduler.submit(
        scheduler.submissions[0].resources,
        InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=(workspace / "child-input.json").resolve(),
        ),
        "worker-child-0001",
    )
    state_path = workspace / "state.json"
    state = load_state(state_path)
    attempt = AttemptRecord(
        attempt_id="attempt-1",
        item_id="item-1",
        sequence=1,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.RUNNING,
        profile=DomainProfile.SMITH,
        submission_token="worker-child-0001",
        job=JobReference(child.job_id, cluster=child.cluster),
    )
    admitted = dataclasses.replace(
        state,
        revision=state.revision + 1,
        stages=(StageRecord("stage-1", ("goal-1",), StageStatus.ACTIVE),),
        goals=(
            GoalRecord(
                "goal-1",
                "stage-1",
                ("item-1",),
                HierarchyStatus.ACTIVE,
            ),
        ),
        items=(
            WorkItemRecord(
                "item-1",
                "stage-1",
                "goal-1",
                WorkItemKind.CATALOG_VERIFY,
                DomainProfile.SMITH,
                status=WorkItemStatus.CODING,
                attempts=(attempt,),
            ),
        ),
    )
    save_state(
        state_path,
        admitted,
        expected_revision=state.revision,
        expected_generation=1,
    )
    monkeypatch.setenv("SLURM_JOB_ID", started.controller_job.job_id)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "alpha")
    flags = _ControllerSignalFlags(advance_requested=True)

    run_controller(
        workspace,
        generation=1,
        owner_nonce="owner-generation-1",
        scheduler=scheduler,
        runtime_factory=_TerminalRuntime,
        signal_flags=flags,
        sleeper=lambda _seconds: None,
        preflight_check=_skip_preflight,
    )

    advanced = load_state(state_path)
    assert advanced.generation == 2
    assert advanced.controller_lifecycle is ControllerLifecycle.AWAITING_PREDECESSOR
    assert advanced.items[0].attempts[0].status is AttemptStatus.RUNNING
    successor = scheduler.submissions[-1]
    assert successor.dependency is not None
    assert successor.dependency.kind is DependencyType.AFTERANY
    assert successor.dependency.job == started.controller_job
    assert successor.resources.requeue is False

    scheduler.transition(
        started.controller_job,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    assert advanced.controller_job is not None
    monkeypatch.setenv("SLURM_JOB_ID", advanced.controller_job.job_id)

    class _CloseAdmittedRuntime(_TerminalRuntime):
        def tick(self) -> RuntimeTickResult:
            state_path = self._workspace / "state.json"
            state = load_state(state_path)
            assert state.attempt_adoptions[-1].attempt_id == "attempt-1"
            assert state.orphan_action_receipts[-1].job == JobReference(
                child.job_id,
                cluster=child.cluster,
            )
            scheduler.transition(
                child,
                JobStatus.CANCELLED,
                source=ObservationSource.ACCOUNTING,
            )
            cancelled_attempt = dataclasses.replace(
                state.items[0].attempts[0],
                status=AttemptStatus.CANCELLED,
                terminal_reason="cancelled for terminal fixture",
            )
            cancelled_item = dataclasses.replace(
                state.items[0],
                status=WorkItemStatus.CANCELLED,
                attempts=(cancelled_attempt,),
                terminal_reason="cancelled for terminal fixture",
            )
            terminal = dataclasses.replace(
                state,
                revision=state.revision + 1,
                stages=(
                    dataclasses.replace(
                        state.stages[0],
                        status=StageStatus.CANCELLED,
                        terminal_reason="cancelled for terminal fixture",
                    ),
                ),
                goals=(
                    dataclasses.replace(
                        state.goals[0],
                        status=HierarchyStatus.CANCELLED,
                        terminal_reason="cancelled for terminal fixture",
                    ),
                ),
                items=(cancelled_item,),
                terminal_status=RunTerminalStatus.CANCELLED,
                terminal_reason="test terminal outcome",
            )
            save_state(
                state_path,
                terminal,
                expected_revision=state.revision,
                expected_generation=state.generation,
            )
            return RuntimeTickResult(
                RuntimeDisposition.TERMINAL,
                RuntimeEvent.TERMINAL,
                terminal.revision,
                "test terminal outcome",
                terminal_status=RunTerminalStatus.CANCELLED,
            )

    run_controller(
        workspace,
        generation=2,
        owner_nonce="owner-generation-2",
        scheduler=scheduler,
        runtime_factory=_CloseAdmittedRuntime,
        signal_flags=_ControllerSignalFlags(),
        sleeper=lambda _seconds: None,
        preflight_check=_skip_preflight,
    )

    report = json.loads((workspace / "reports/terminal-report.json").read_text())
    assert report["outcome"] == "cancelled"
    observations = {entry["scheduler_id"]: entry["observation"] for entry in report["jobs"]}
    assert observations[child.scheduler_id]["source"] == "ACCOUNTING"
    assert not (workspace / "controller-lease.json").exists()


def test_generation_fence_keeps_adopted_credentials_and_revokes_terminal_orphan(
    tmp_path: Path,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    source = (tmp_path / "source").resolve()
    source.mkdir()
    attempts = tuple(
        AttemptRecord(
            attempt_id=attempt_id,
            item_id=PLANNING_ITEM_ID,
            sequence=sequence,
            role=Role.PLAN_DRAFTER,
            kind=AttemptKind.ROLE,
            generation=1,
            status=AttemptStatus.RUNNING,
            submission_token=f"token-{attempt_id}",
            job=JobReference(str(700 + sequence), cluster="alpha"),
            resource_class="plan_drafter",
        )
        for sequence, attempt_id in enumerate(("adopted", "terminal"), start=1)
    )
    predecessor = RunState(
        run_id="run-workflow-fence",
        task_digest="a" * 64,
        base_commit="abc",
        generation=1,
        planning_attempts=attempts,
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-token",
        controller_job=JobReference("600", cluster="alpha"),
    )
    state = predecessor.resume_after_reconciliation(
        predecessor_job=JobReference("600", cluster="alpha"),
        scheduler_state=JobStatus.PREEMPTED.value,
    )
    state = state.start_orphan_grace(
        started_at="2026-09-16T12:00:00+00:00",
        deadline_at="2026-09-16T12:01:00+00:00",
    )
    assert attempts[0].job is not None
    state = state.record_orphan_action(
        action=OrphanReceiptAction.ADOPTED,
        job=attempts[0].job,
        scheduler_state=JobStatus.RUNNING.value,
    )
    for attempt in attempts:
        state = state.record_attempt_adoption(
            attempt_id=attempt.attempt_id,
            attempt_generation=attempt.generation,
            status=attempt.status,
            submission_token=attempt.submission_token,
            job=attempt.job,
        )

    provisions = []
    for attempt in attempts:
        provision = prepare_credential_provision(
            workspace=workspace,
            binding=CredentialBinding(
                run_id=state.run_id,
                item_id="planning",
                attempt_id=attempt.attempt_id,
                task_digest=state.task_digest,
                generation=attempt.generation,
                backend_kind="codex",
            ),
            ambient_environment={"OPENAI_API_KEY": "test-secret"},
        )
        provisions.append(provision)
        input_path = (
            workspace
            / "items"
            / PLANNING_ITEM_ID
            / "attempts"
            / attempt.attempt_id
            / "role-input.json"
        )
        input_path.parent.mkdir(parents=True)
        role_input = RoleProcessSpec(
            schema_version=1,
            run_id=state.run_id,
            task_digest=state.task_digest,
            generation=attempt.generation,
            item_id="planning",
            attempt_id=attempt.attempt_id,
            prompt_id=attempt.attempt_id,
            role="plan_drafter",
            profile="planner",
            backend_kind="codex",
            model="test-model",
            source_root=str(source),
            cwd=str(source),
            result_path=str(input_path.with_name("role-result.json")),
            system_prompt="system",
            prompt="prompt",
            credential_descriptor=provision.descriptor.to_public_dict(),
            launch_policy=AgentLaunchPolicy.create(
                image="/images/agent-worker.sqsh",
                build_identity="agent-worker-test",
            ).to_public_dict(),
        )
        input_path.write_text(
            json.dumps(dataclasses.asdict(role_input), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    class _ExplicitBroker:
        def revoke(self, **kwargs: object):
            expected_binding = kwargs["expected_binding"]
            descriptor = kwargs["descriptor"]
            cause = kwargs["cause"]
            matching = next(entry for entry in provisions if entry.descriptor == descriptor)
            return revoke_terminal_attempt_credentials(
                workspace=workspace,
                expected_binding=expected_binding,  # type: ignore[arg-type]
                provision=matching,
                cause=cause,  # type: ignore[arg-type]
            )

    _fence_unadopted_orphan_credentials(
        workspace=workspace,
        state=state,
        credential_broker=_ExplicitBroker(),  # type: ignore[arg-type]
    )

    assert provisions[0].bundle_path is not None
    assert not provisions[0].bundle_path.exists()
    assert provisions[1].bundle_path is not None
    assert not provisions[1].bundle_path.exists()


def test_workspace_recovery_broker_refuses_bundle_before_reading_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path.resolve()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=CredentialBinding(
            run_id="run-direct-recovery",
            item_id="item",
            attempt_id="attempt",
            task_digest="a" * 64,
            generation=1,
            backend_kind="codex",
        ),
        ambient_environment={"OPENAI_API_KEY": "must-not-be-read"},
    )
    assert provision.descriptor.state is CredentialState.BUNDLE
    assert provision.bundle_path is not None
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == provision.bundle_path:
            raise AssertionError("controller attempted to read credential bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    with pytest.raises(WorkflowError, match="requires an explicit credential broker"):
        _WorkspaceRecoveryCredentialBroker().revoke(
            workspace=workspace,
            expected_binding=provision.descriptor.binding,
            descriptor=provision.descriptor,
            cause=CredentialRevocationCause.ORPHAN_EXPIRED,
        )

    assert provision.bundle_path.is_file()


def test_workspace_recovery_broker_allows_no_credential_cleanup(tmp_path: Path) -> None:
    workspace = tmp_path.resolve()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=CredentialBinding(
            run_id="run-direct-recovery",
            item_id="item",
            attempt_id="attempt-none",
            task_digest="a" * 64,
            generation=1,
            backend_kind="codex",
        ),
        ambient_environment={},
    )
    assert provision.descriptor.state is CredentialState.NONE

    receipt = _WorkspaceRecoveryCredentialBroker().revoke(
        workspace=workspace,
        expected_binding=provision.descriptor.binding,
        descriptor=provision.descriptor,
        cause=CredentialRevocationCause.ORPHAN_EXPIRED,
    )

    assert receipt.cause is CredentialRevocationCause.ORPHAN_EXPIRED
    assert receipt.receipt_path.is_file()


def test_late_successor_cleans_exact_orphan_and_revokes_idempotently_across_restart(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    workspace.mkdir(parents=True)
    child = scheduler.submit(
        ResourceRequest(
            label="orphan-test",
            account="account",
            partition="partition",
            time_limit="00:01:00",
            output_path=(workspace / "orphan.out").resolve(),
            error_path=(workspace / "orphan.err").resolve(),
        ),
        InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=(workspace / "orphan-input.json").resolve(),
        ),
        "orphan-worker-token",
    )
    attempt = AttemptRecord(
        attempt_id="orphan-plan",
        item_id=PLANNING_ITEM_ID,
        sequence=1,
        role=Role.PLAN_DRAFTER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.RUNNING,
        submission_token="orphan-worker-token",
        job=JobReference(child.job_id, cluster=child.cluster),
        resource_class="plan_drafter",
    )
    state_path = workspace / "state.json"
    predecessor = RunState(
        run_id="run-orphan-expiry",
        task_digest=task.digest,
        base_commit=task.repository.base_commit,
        generation=1,
        planning_attempts=(attempt,),
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-token",
        controller_job=JobReference("999", cluster="alpha"),
    )
    assert predecessor.controller_job is not None
    successor = predecessor.resume_after_reconciliation(
        predecessor_job=predecessor.controller_job,
        scheduler_state=JobStatus.COMPLETED.value,
    )
    successor = successor.start_orphan_grace(
        started_at="2026-09-16T12:00:00+00:00",
        deadline_at="2026-09-16T12:01:00+00:00",
    )
    successor = dataclasses.replace(successor, revision=0)
    initialize_state(state_path, successor)

    source = (tmp_path / "source").resolve()
    source.mkdir()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=CredentialBinding(
            run_id=successor.run_id,
            item_id="planning",
            attempt_id=attempt.attempt_id,
            task_digest=successor.task_digest,
            generation=1,
            backend_kind="codex",
        ),
        ambient_environment={},
    )
    input_path = (
        workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id / "role-input.json"
    )
    input_path.parent.mkdir(parents=True)
    role_input = RoleProcessSpec(
        schema_version=1,
        run_id=successor.run_id,
        task_digest=successor.task_digest,
        generation=1,
        item_id="planning",
        attempt_id=attempt.attempt_id,
        prompt_id=attempt.attempt_id,
        role="plan_drafter",
        profile="planner",
        backend_kind="codex",
        model="test-model",
        source_root=str(source),
        cwd=str(source),
        result_path=str(input_path.with_name("role-result.json")),
        system_prompt="system",
        prompt="prompt",
        credential_descriptor=provision.descriptor.to_public_dict(),
        launch_policy=AgentLaunchPolicy.create(
            image="/images/agent-worker.sqsh",
            build_identity="agent-worker-test",
        ).to_public_dict(),
    )
    input_path.write_text(
        json.dumps(dataclasses.asdict(role_input), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    causes: list[CredentialRevocationCause] = []

    class _Broker:
        def revoke(self, **kwargs: object):
            cause = kwargs["cause"]
            assert isinstance(cause, CredentialRevocationCause)
            causes.append(cause)
            return revoke_terminal_attempt_credentials(
                workspace=workspace,
                expected_binding=provision.descriptor.binding,
                provision=provision,
                cause=cause,
            )

    recovered = _adopt_orphaned_attempts(
        workspace=workspace,
        state=successor,
        task=task,
        scheduler=scheduler,
        credential_broker=_Broker(),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 9, 16, 12, 3, tzinfo=UTC),
    )

    assert scheduler.observe(child).status is JobStatus.CANCELLED
    assert recovered.orphan_action_receipts[-1].action is OrphanReceiptAction.CLEANED_UP
    assert recovered.orphan_action_receipts[-1].job == attempt.job
    assert recovered.attempt_adoptions[-1].attempt_id == attempt.attempt_id
    assert causes == [CredentialRevocationCause.GENERATION_FENCE]

    restarted = _adopt_orphaned_attempts(
        workspace=workspace,
        state=load_state(state_path),
        task=task,
        scheduler=scheduler,
        credential_broker=_Broker(),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 9, 16, 12, 4, tzinfo=UTC),
    )
    assert restarted == recovered
    assert causes == [
        CredentialRevocationCause.GENERATION_FENCE,
        CredentialRevocationCause.GENERATION_FENCE,
    ]


@pytest.mark.parametrize("status", [AttemptStatus.PREPARED, AttemptStatus.SUBMITTING])
@pytest.mark.parametrize("credentialed", [False, True])
def test_expired_no_job_orphan_is_terminally_fenced_and_never_submitted(
    tmp_path: Path,
    status: AttemptStatus,
    credentialed: bool,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    workspace.mkdir(parents=True)
    token = "orphan-no-job-token" if status is AttemptStatus.SUBMITTING else None
    attempt = AttemptRecord(
        attempt_id=f"orphan-{status.value}",
        item_id=PLANNING_ITEM_ID,
        sequence=1,
        role=Role.PLAN_DRAFTER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=status,
        submission_token=token,
        resource_class="plan_drafter",
    )
    predecessor = RunState(
        run_id=f"run-{status.value}-{'credential' if credentialed else 'none'}",
        task_digest=task.digest,
        base_commit=task.repository.base_commit,
        generation=1,
        planning_attempts=(attempt,),
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-token",
        controller_job=JobReference("999", cluster="alpha"),
    )
    assert predecessor.controller_job is not None
    successor = predecessor.resume_after_reconciliation(
        predecessor_job=predecessor.controller_job,
        scheduler_state=JobStatus.COMPLETED.value,
    )
    successor = successor.start_orphan_grace(
        started_at="2026-09-16T12:00:00+00:00",
        deadline_at="2026-09-16T12:01:00+00:00",
    )
    successor = dataclasses.replace(successor, revision=0)
    state_path = workspace / "state.json"
    initialize_state(state_path, successor)

    source = (tmp_path / "source").resolve()
    source.mkdir()
    provision = prepare_credential_provision(
        workspace=workspace,
        binding=CredentialBinding(
            run_id=successor.run_id,
            item_id="planning",
            attempt_id=attempt.attempt_id,
            task_digest=successor.task_digest,
            generation=1,
            backend_kind="codex",
        ),
        ambient_environment={"OPENAI_API_KEY": "orphan-secret"} if credentialed else {},
    )
    input_path = (
        workspace / "items" / PLANNING_ITEM_ID / "attempts" / attempt.attempt_id / "role-input.json"
    )
    input_path.parent.mkdir(parents=True)
    role_input = RoleProcessSpec(
        schema_version=1,
        run_id=successor.run_id,
        task_digest=successor.task_digest,
        generation=1,
        item_id="planning",
        attempt_id=attempt.attempt_id,
        prompt_id=attempt.attempt_id,
        role="plan_drafter",
        profile="planner",
        backend_kind="codex",
        model="test-model",
        source_root=str(source),
        cwd=str(source),
        result_path=str(input_path.with_name("role-result.json")),
        system_prompt="system",
        prompt="prompt",
        credential_descriptor=provision.descriptor.to_public_dict(),
        launch_policy=AgentLaunchPolicy.create(
            image="/images/agent-worker.sqsh",
            build_identity="agent-worker-test",
        ).to_public_dict(),
    )
    input_path.write_text(
        json.dumps(dataclasses.asdict(role_input), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    causes: list[CredentialRevocationCause] = []

    class _Broker:
        def revoke(self, **kwargs: object):
            cause = kwargs["cause"]
            assert isinstance(cause, CredentialRevocationCause)
            causes.append(cause)
            return revoke_terminal_attempt_credentials(
                workspace=workspace,
                expected_binding=provision.descriptor.binding,
                provision=provision,
                cause=cause,
            )

    recovered = _adopt_orphaned_attempts(
        workspace=workspace,
        state=successor,
        task=task,
        scheduler=scheduler,
        credential_broker=_Broker(),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 9, 16, 12, 3, tzinfo=UTC),
    )

    cleaned = recovered.planning_attempts[0]
    assert cleaned.status is AttemptStatus.CANCELLED
    assert cleaned.generation == 1
    assert cleaned.job is None
    assert cleaned.terminal_reason == "old-generation attempt expired before scheduler submission"
    assert recovered.attempt_adoptions[-1].status is status
    assert recovered.attempt_adoptions[-1].attempt_id == attempt.attempt_id
    assert causes == [CredentialRevocationCause.ORPHAN_EXPIRED]
    assert scheduler.submissions == ()
    if provision.bundle_path is not None:
        assert not provision.bundle_path.exists()

    restarted = _adopt_orphaned_attempts(
        workspace=workspace,
        state=load_state(state_path),
        task=task,
        scheduler=scheduler,
        credential_broker=_Broker(),  # type: ignore[arg-type]
        clock=lambda: datetime(2026, 9, 16, 12, 4, tzinfo=UTC),
    )
    assert restarted == recovered
    assert causes == [CredentialRevocationCause.ORPHAN_EXPIRED]


def test_successor_adopts_delayed_job_for_old_generation_submitting_intent(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    workspace.mkdir(parents=True)
    token = "orphan-delayed-token"
    attempt = AttemptRecord(
        attempt_id="orphan-delayed",
        item_id=PLANNING_ITEM_ID,
        sequence=1,
        role=Role.PLAN_DRAFTER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.SUBMITTING,
        submission_token=token,
        resource_class="plan_drafter",
    )
    predecessor = RunState(
        run_id="run-delayed-orphan",
        task_digest=task.digest,
        base_commit=task.repository.base_commit,
        generation=1,
        planning_attempts=(attempt,),
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-token",
        controller_job=JobReference("999", cluster="alpha"),
    )
    successor = predecessor.resume_after_reconciliation(
        predecessor_job=JobReference("999", cluster="alpha"),
        scheduler_state=JobStatus.COMPLETED.value,
    ).start_orphan_grace(
        started_at="2026-09-16T12:00:00+00:00",
        deadline_at="2026-09-16T12:01:00+00:00",
    )
    successor = dataclasses.replace(successor, revision=0)
    initialize_state(workspace / "state.json", successor)
    delayed = scheduler.submit(
        ResourceRequest(
            label="orphan",
            account="coreai",
            partition="batch",
            time_limit="00:10:00",
            output_path=workspace / "logs/%j.out",
            error_path=workspace / "logs/%j.err",
        ),
        InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=(workspace / "delayed-input.json").resolve(),
        ),
        token,
    )

    adopted = _adopt_orphaned_attempts(
        workspace=workspace,
        state=successor,
        task=task,
        scheduler=scheduler,
        credential_broker=_WorkspaceRecoveryCredentialBroker(),
        clock=lambda: datetime(2026, 9, 16, 12, 0, 30, tzinfo=UTC),
    )

    persisted = adopted.planning_attempts[0]
    assert persisted.status is AttemptStatus.SUBMITTED
    assert persisted.job == JobReference(delayed.job_id, cluster="alpha")
    assert adopted.attempt_adoptions[-1].job == persisted.job
    assert scheduler.cancelled == ()


def test_detached_response_adopts_scheduler_acceptance_after_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path)
    scheduler = _CrashAfterAcceptedSubmit(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    monkeypatch.setenv("SLURM_JOB_ID", started.controller_job.job_id)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "alpha")

    run_controller(
        workspace,
        generation=1,
        owner_nonce="owner-waiting",
        scheduler=scheduler,
        runtime_factory=_WaitingRuntime,
        signal_flags=_ControllerSignalFlags(),
        sleeper=lambda _seconds: None,
        preflight_check=_skip_preflight,
    )
    waiting = load_state(workspace / "state.json")
    assert waiting.controller_lifecycle is ControllerLifecycle.WAITING_FOR_INPUT
    request_id = waiting.human_requests[-1].request_id

    scheduler.crash_after_next_submit = True
    with pytest.raises(WorkflowError, match="launcher submit returned"):
        record_human_response(
            workspace,
            request_id=request_id,
            response="use the public ModelingV2 route",
            scheduler=scheduler,
            preflight_check=_skip_preflight,
        )
    assert len(scheduler.submissions) == 2

    adopted = record_human_response(
        workspace,
        request_id=request_id,
        response="use the public ModelingV2 route",
        scheduler=scheduler,
        preflight_check=_skip_preflight,
    )
    assert adopted == scheduler.submissions[-1].identity
    assert len(scheduler.submissions) == 2
    still_waiting = load_state(workspace / "state.json")
    assert still_waiting.generation == 1
    assert still_waiting.human_requests[-1].response_digest is None

    scheduler.transition(
        started.controller_job,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    monkeypatch.setenv("SLURM_JOB_ID", adopted.job_id)
    run_controller(
        workspace,
        generation=2,
        owner_nonce="owner-resumed",
        scheduler=scheduler,
        runtime_factory=_TerminalRuntime,
        signal_flags=_ControllerSignalFlags(),
        sleeper=lambda _seconds: None,
        preflight_check=_skip_preflight,
    )
    resumed = load_state(workspace / "state.json")
    assert resumed.generation == 2
    assert resumed.human_requests[-1].response_digest is not None
    before_replay = (workspace / "state.json").read_bytes()
    replayed = record_human_response(
        workspace,
        request_id=request_id,
        response="use the public ModelingV2 route",
        scheduler=scheduler,
        preflight_check=_skip_preflight,
    )
    assert replayed == adopted
    assert (workspace / "state.json").read_bytes() == before_replay


def test_human_successor_submission_is_fenced_from_emergency_cancel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path)
    workspace = task.repository.workspace_root / "run"

    class _CancelBeforeAcceptedSubmit(FakeScheduler):
        trigger_cancel = False

        def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.trigger_cancel:
                self.trigger_cancel = False
                request_cancellation(
                    workspace,
                    reason="operator stop",
                    scheduler=self,
                )
            return super().submit(*args, **kwargs)  # type: ignore[arg-type]

    scheduler = _CancelBeforeAcceptedSubmit(cluster="alpha")
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    monkeypatch.setenv("SLURM_JOB_ID", started.controller_job.job_id)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "alpha")
    run_controller(
        workspace,
        generation=1,
        owner_nonce="owner-waiting",
        scheduler=scheduler,
        runtime_factory=_WaitingRuntime,
        signal_flags=_ControllerSignalFlags(),
        sleeper=lambda _seconds: None,
        preflight_check=_skip_preflight,
    )
    waiting = load_state(workspace / "state.json")
    request_id = waiting.human_requests[-1].request_id
    scheduler.transition(
        started.controller_job,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    scheduler.trigger_cancel = True

    successor = record_human_response(
        workspace,
        request_id=request_id,
        response="resume with the frozen decision",
        scheduler=scheduler,
        preflight_check=_skip_preflight,
    )

    assert successor is not None
    assert not (workspace / "requests/cancel.receipt.json").exists()
    request_cancellation(workspace, reason="operator stop", scheduler=scheduler)
    assert scheduler.cancelled == (successor,)


def test_dead_controller_cancellation_uses_only_exact_persisted_jobs(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, credentialed=False)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    child = scheduler.submit(
        scheduler.submissions[0].resources,
        InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=(workspace / "child-input.json").resolve(),
        ),
        "worker-cancel-0001",
    )
    state_path = workspace / "state.json"
    state = load_state(state_path)
    attempt = AttemptRecord(
        "attempt-1",
        "item-1",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        AttemptStatus.RUNNING,
        DomainProfile.SMITH,
        "worker-cancel-0001",
        JobReference(child.job_id, cluster=child.cluster),
    )
    admitted = dataclasses.replace(
        state,
        revision=state.revision + 1,
        stages=(StageRecord("stage-1", ("goal-1",)),),
        goals=(GoalRecord("goal-1", "stage-1", ("item-1",)),),
        items=(
            WorkItemRecord(
                "item-1",
                "stage-1",
                "goal-1",
                WorkItemKind.CATALOG_VERIFY,
                DomainProfile.SMITH,
                status=WorkItemStatus.CODING,
                attempts=(attempt,),
            ),
        ),
    )
    save_state(
        state_path,
        admitted,
        expected_revision=state.revision,
        expected_generation=1,
    )
    scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )

    request_cancellation(workspace, reason="operator stop", scheduler=scheduler)

    assert scheduler.cancelled == (child,)
    receipt = json.loads((workspace / "requests/cancel.receipt.json").read_text())
    assert receipt["controller_job"]["job_id"] == started.controller_job.job_id
    assert receipt["child_actions"] == [
        {
            "action": "cancelled",
            "job": {
                "array_task_id": None,
                "cluster": "alpha",
                "job_id": child.job_id,
            },
            "observation_source": "QUEUE",
            "observed_status": "PENDING",
            "ownership_user": "fake-user",
            "submission_token": "worker-cancel-0001",
        }
    ]


def test_emergency_cancel_refuses_cleanup_while_controller_lease_is_live(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    state = load_state(workspace / "state.json")
    lease = ControllerLease.acquire(
        workspace / "controller-lease.json",
        run_id=state.run_id,
        generation=state.generation,
        controller_job=state.controller_job,
        owner_nonce="live-controller-owner",
    )
    scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )
    try:
        request_cancellation(workspace, reason="operator stop", scheduler=scheduler)
        assert not (workspace / "requests/cancel.receipt.json").exists()
        assert scheduler.cancelled == ()
    finally:
        lease.release()


def test_emergency_cancel_holds_lease_lock_until_child_receipt_is_published(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, credentialed=False)
    workspace = task.repository.workspace_root / "run"

    class _RacingScheduler(FakeScheduler):
        def __init__(self) -> None:
            super().__init__(cluster="alpha")
            self.child: JobIdentity | None = None
            self.state: RunState | None = None
            self.successor_blocked = False

        def verify_ownership(self, identity: JobIdentity, submission_token: str):
            evidence = super().verify_ownership(identity, submission_token)
            if self.child == identity and not self.successor_blocked:
                assert self.state is not None and self.state.controller_job is not None
                with pytest.raises(LeaseConflictError):
                    ControllerLease.recover(
                        workspace / "controller-lease.json",
                        run_id=self.state.run_id,
                        generation=self.state.generation + 1,
                        controller_job=JobReference("9999", cluster="alpha"),
                        scheduler_reconciled=True,
                        expected_predecessor_job=self.state.controller_job,
                        owner_nonce="racing-successor",
                    )
                self.successor_blocked = True
            return evidence

    scheduler = _RacingScheduler()
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    child = scheduler.submit(
        scheduler.submissions[0].resources,
        InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=(workspace / "child-race.json").resolve(),
        ),
        "worker-race-0001",
    )
    state_path = workspace / "state.json"
    state = load_state(state_path)
    attempt = AttemptRecord(
        "attempt-race",
        "item-race",
        1,
        Role.CODER,
        AttemptKind.ROLE,
        1,
        AttemptStatus.RUNNING,
        DomainProfile.SMITH,
        "worker-race-0001",
        JobReference(child.job_id, cluster=child.cluster),
        resource_class="coder_analysis",
    )
    desired = dataclasses.replace(
        state,
        revision=state.revision + 1,
        stages=(StageRecord("stage-race", ("goal-race",)),),
        goals=(GoalRecord("goal-race", "stage-race", ("item-race",)),),
        items=(
            WorkItemRecord(
                "item-race",
                "stage-race",
                "goal-race",
                WorkItemKind.CATALOG_VERIFY,
                DomainProfile.SMITH,
                status=WorkItemStatus.CODING,
                attempts=(attempt,),
            ),
        ),
    )
    save_state(
        state_path,
        desired,
        expected_revision=state.revision,
        expected_generation=state.generation,
    )
    stale = ControllerLease.acquire(
        workspace / "controller-lease.json",
        run_id=desired.run_id,
        generation=desired.generation,
        controller_job=desired.controller_job,
        owner_nonce="dead-controller",
    )
    stale.release(remove_record=False)
    scheduler.child = child
    scheduler.state = desired
    scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )

    request_cancellation(workspace, reason="operator stop", scheduler=scheduler)

    assert scheduler.successor_blocked
    assert scheduler.cancelled == (child,)
    assert (workspace / "requests/cancel.receipt.json").is_file()


def test_emergency_cancel_fences_unbootstrapped_recovery_successor_before_cleanup(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, credentialed=False)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    state = load_state(workspace / "state.json")
    stale = ControllerLease.acquire(
        workspace / "controller-lease.json",
        run_id=state.run_id,
        generation=state.generation,
        controller_job=state.controller_job,
        owner_nonce="dead-controller",
    )
    stale.release(remove_record=False)
    scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )
    recovery = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert recovery.controller_job is not None
    assert load_state(workspace / "state.json").generation == 1

    request_cancellation(workspace, reason="operator stop", scheduler=scheduler)

    assert scheduler.cancelled == (recovery.controller_job,)
    assert not (workspace / "requests/cancel.receipt.json").exists()

    request_cancellation(workspace, reason="operator stop", scheduler=scheduler)

    receipt = json.loads((workspace / "requests/cancel.receipt.json").read_text())
    assert receipt["successor_actions"] == [
        {
            "action": "already_terminal",
            "job": {
                "array_task_id": None,
                "cluster": "alpha",
                "job_id": recovery.controller_job.job_id,
            },
            "observation_source": "ACCOUNTING",
            "observed_status": "CANCELLED",
            "ownership_user": "fake-user",
            "submission_token": scheduler.submissions[-1].submission_token,
        }
    ]


def test_recovery_successor_submission_is_fenced_from_emergency_cancel(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, credentialed=False)
    workspace = task.repository.workspace_root / "run"

    class _CancelBeforeAcceptedSubmit(FakeScheduler):
        trigger_cancel = False

        def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.trigger_cancel:
                self.trigger_cancel = False
                request_cancellation(
                    workspace,
                    reason="operator stop",
                    scheduler=self,
                )
            return super().submit(*args, **kwargs)  # type: ignore[arg-type]

    scheduler = _CancelBeforeAcceptedSubmit(cluster="alpha")
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    state = load_state(workspace / "state.json")
    stale = ControllerLease.acquire(
        workspace / "controller-lease.json",
        run_id=state.run_id,
        generation=state.generation,
        controller_job=state.controller_job,
        owner_nonce="dead-controller",
    )
    stale.release(remove_record=False)
    scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )
    scheduler.trigger_cancel = True

    recovery = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)

    assert recovery.controller_job is not None
    assert not (workspace / "requests/cancel.receipt.json").exists()
    request_cancellation(workspace, reason="operator stop", scheduler=scheduler)
    assert scheduler.cancelled == (recovery.controller_job,)


def test_emergency_cancel_revokes_all_agent_roles_before_children_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task_value, workspace, scheduler, jobs, descriptors = _credentialed_emergency_fixture(tmp_path)

    class _Broker:
        def __init__(self) -> None:
            self.calls: list[CredentialDescriptor] = []

        def revoke(
            self,
            *,
            workspace: Path,
            expected_binding: CredentialBinding,
            descriptor: CredentialDescriptor,
            cause: CredentialRevocationCause,
        ) -> CredentialRevocationReceipt:
            assert expected_binding == descriptor.binding
            assert cause is CredentialRevocationCause.CANCELLED
            assert scheduler.cancelled == ()
            self.calls.append(descriptor)
            receipt_path = (
                workspace
                / "receipts"
                / "emergency-credentials"
                / f"{descriptor.binding.attempt_id}.json"
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text("{}\n", encoding="utf-8")
            return CredentialRevocationReceipt(descriptor, cause, receipt_path)

    broker = _Broker()
    monkeypatch.setattr(
        staircase_workflow,
        "_build_emergency_credential_broker",
        lambda *_args, **_kwargs: broker,
    )

    request_cancellation(workspace, reason="credentialed stop", scheduler=scheduler)

    assert set(broker.calls) == set(descriptors)
    assert set(scheduler.cancelled) == set(jobs)
    receipt_path = workspace / "requests/cancel.receipt.json"
    receipt_before = receipt_path.read_bytes()
    receipt = json.loads(receipt_before)
    assert {entry["descriptor_digest"] for entry in receipt["credential_actions"]} == {
        descriptor.descriptor_digest for descriptor in descriptors
    }
    assert {entry["cause"] for entry in receipt["credential_actions"]} == {
        CredentialRevocationCause.CANCELLED.value
    }

    request_cancellation(workspace, reason="credentialed stop", scheduler=scheduler)

    assert set(broker.calls) == set(descriptors)
    assert set(scheduler.cancelled) == set(jobs)
    assert receipt_path.read_bytes() == receipt_before


@pytest.mark.parametrize(
    ("hidden_role", "hidden_job_index"),
    [(Role.PLAN_DRAFTER, 0), (Role.CODER, 1)],
)
def test_emergency_cancel_adopts_hidden_job_revokes_descriptor_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hidden_role: Role,
    hidden_job_index: int,
) -> None:
    _task_value, workspace, scheduler, jobs, descriptors = _credentialed_emergency_fixture(
        tmp_path,
        hidden_role=hidden_role,
    )
    state = load_state(workspace / "state.json")
    attempts = [*state.planning_attempts]
    attempts.extend(attempt for item in state.items for attempt in item.attempts)
    hidden_attempt = next(attempt for attempt in attempts if attempt.role is hidden_role)
    _write_hidden_submission_claim(
        workspace,
        hidden_attempt,
        claimed_at=datetime(2026, 9, 16, 17, 0, tzinfo=UTC),
    )

    class _Broker:
        def __init__(self) -> None:
            self.calls: list[CredentialDescriptor] = []

        def revoke(
            self,
            *,
            workspace: Path,
            expected_binding: CredentialBinding,
            descriptor: CredentialDescriptor,
            cause: CredentialRevocationCause,
        ) -> CredentialRevocationReceipt:
            assert expected_binding == descriptor.binding
            assert cause is CredentialRevocationCause.CANCELLED
            self.calls.append(descriptor)
            receipt_path = (
                workspace / "receipts" / "hidden-revoke" / f"{descriptor.binding.attempt_id}.json"
            )
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text("{}\n", encoding="utf-8")
            return CredentialRevocationReceipt(descriptor, cause, receipt_path)

    broker = _Broker()
    monkeypatch.setattr(
        staircase_workflow,
        "_build_emergency_credential_broker",
        lambda *_args, **_kwargs: broker,
    )

    request_cancellation(workspace, reason="hidden accepted stop", scheduler=scheduler)

    persisted = load_state(workspace / "state.json")
    persisted_attempts = [*persisted.planning_attempts]
    persisted_attempts.extend(attempt for item in persisted.items for attempt in item.attempts)
    adopted = next(
        attempt for attempt in persisted_attempts if attempt.attempt_id == hidden_attempt.attempt_id
    )
    assert adopted.status is AttemptStatus.SUBMITTED
    assert adopted.job == JobReference(jobs[hidden_job_index].job_id, cluster="alpha")
    assert set(scheduler.cancelled) == set(jobs)
    assert set(broker.calls) == set(descriptors)
    receipt_path = workspace / "requests/cancel.receipt.json"
    receipt_before = receipt_path.read_bytes()
    receipt = json.loads(receipt_before)
    hidden_actions = [
        action
        for action in receipt["child_actions"]
        if action.get("submission_token") == hidden_attempt.submission_token
    ]
    assert hidden_actions == [
        {
            "action": "cancelled",
            "job": {
                "array_task_id": None,
                "cluster": "alpha",
                "job_id": jobs[hidden_job_index].job_id,
            },
            "observation_source": "QUEUE",
            "observed_status": "PENDING",
            "ownership_user": "fake-user",
            "submission_token": hidden_attempt.submission_token,
        }
    ]

    request_cancellation(workspace, reason="hidden accepted stop", scheduler=scheduler)

    assert set(scheduler.cancelled) == set(jobs)
    assert set(broker.calls) == set(descriptors)
    assert receipt_path.read_bytes() == receipt_before


def test_emergency_cancel_claimed_certified_absence_is_bounded_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task_value, workspace, scheduler, attempt = _hidden_submitting_emergency_fixture(tmp_path)
    claimed_at = datetime(2026, 9, 16, 17, 0, tzinfo=UTC)
    _write_hidden_submission_claim(workspace, attempt, claimed_at=claimed_at)
    now = [claimed_at + timedelta(seconds=120)]
    monkeypatch.setattr(staircase_workflow, "_emergency_cancellation_clock", lambda: now[0])

    with pytest.raises(WorkflowError, match="waiting for bounded evidence"):
        request_cancellation(workspace, reason="certified absence", scheduler=scheduler)
    assert not (workspace / "requests/cancel.receipt.json").exists()

    now[0] += timedelta(seconds=5)
    write_once = staircase_workflow._write_once_json

    def crash_before_receipt(path: Path, payload: Mapping[str, object]) -> None:
        if path.name == "cancel.receipt.json":
            raise RuntimeError("crash before full receipt")
        write_once(path, payload)

    monkeypatch.setattr(staircase_workflow, "_write_once_json", crash_before_receipt)
    with pytest.raises(RuntimeError, match="crash before full receipt"):
        request_cancellation(workspace, reason="certified absence", scheduler=scheduler)
    assert (
        load_state(workspace / "state.json").items[0].attempts[0].status is AttemptStatus.CANCELLED
    )
    assert not (workspace / "requests/cancel.receipt.json").exists()

    monkeypatch.setattr(staircase_workflow, "_write_once_json", write_once)
    request_cancellation(workspace, reason="certified absence", scheduler=scheduler)

    cancelled = load_state(workspace / "state.json").items[0].attempts[0]
    assert cancelled.status is AttemptStatus.CANCELLED
    assert cancelled.job is None
    assert scheduler.cancelled == ()
    receipt_path = workspace / "requests/cancel.receipt.json"
    receipt_before = receipt_path.read_bytes()
    receipt = json.loads(receipt_before)
    assert receipt["child_actions"] == [
        {
            "action": "cancelled_intent",
            "attempt_id": attempt.attempt_id,
            "job": None,
            "observation_source": "UNKNOWN",
            "observed_status": "UNKNOWN",
            "ownership_user": None,
            "submission_token": attempt.submission_token,
        }
    ]

    request_cancellation(workspace, reason="certified absence", scheduler=scheduler)
    assert receipt_path.read_bytes() == receipt_before


def test_emergency_cancel_delayed_uncertified_job_remains_recoverable(
    tmp_path: Path,
) -> None:
    class _DelayedEvidenceScheduler(FakeScheduler):
        delayed = True

        def probe_submission(self, submission_token: str) -> SubmissionProbe:
            if self.delayed:
                return SubmissionProbe(
                    submission_token=submission_token,
                    matches=(),
                    status=JobStatus.UNKNOWN,
                    reason="accounting retention is not certified",
                    source=ObservationSource.UNKNOWN,
                    queue_complete=True,
                    accounting_complete=True,
                    trustworthy=False,
                )
            return super().probe_submission(submission_token)

    scheduler = _DelayedEvidenceScheduler(cluster="alpha")
    _task_value, workspace, scheduler, attempt = _hidden_submitting_emergency_fixture(
        tmp_path,
        scheduler=scheduler,
    )
    _write_hidden_submission_claim(
        workspace,
        attempt,
        claimed_at=datetime(2026, 9, 16, 17, 0, tzinfo=UTC),
    )
    hidden_job = _submit_hidden_attempt(scheduler, workspace, attempt)

    with pytest.raises(WorkflowError, match="manual reconciliation"):
        request_cancellation(workspace, reason="delayed visibility", scheduler=scheduler)
    assert not (workspace / "requests/cancel.receipt.json").exists()
    assert (
        workspace / "receipts" / "submissions" / attempt.attempt_id / "manual-recovery.json"
    ).is_file()

    scheduler.delayed = False
    request_cancellation(workspace, reason="delayed visibility", scheduler=scheduler)

    assert scheduler.cancelled == (hidden_job,)
    assert (workspace / "requests/cancel.receipt.json").is_file()


def test_emergency_cancel_ambiguity_persists_manual_but_late_unique_match_recovers(
    tmp_path: Path,
) -> None:
    _task_value, workspace, scheduler, attempt = _hidden_submitting_emergency_fixture(tmp_path)
    _write_hidden_submission_claim(
        workspace,
        attempt,
        claimed_at=datetime(2026, 9, 16, 17, 0, tzinfo=UTC),
    )
    first = _submit_hidden_attempt(scheduler, workspace, attempt)
    second = _submit_hidden_attempt(scheduler, workspace, attempt)

    with pytest.raises(WorkflowError, match="ambiguous"):
        request_cancellation(workspace, reason="ambiguous hidden job", scheduler=scheduler)
    assert not (workspace / "requests/cancel.receipt.json").exists()

    scheduler.transition(second, JobStatus.PENDING, visible=False)
    request_cancellation(workspace, reason="ambiguous hidden job", scheduler=scheduler)

    assert scheduler.cancelled == (first,)
    assert (workspace / "requests/cancel.receipt.json").is_file()


def test_emergency_cancel_probe_error_persists_manual_and_late_job_recovers(
    tmp_path: Path,
) -> None:
    class _ProbeErrorScheduler(FakeScheduler):
        fail_probe = False

        def probe_submission(self, submission_token: str) -> SubmissionProbe:
            if self.fail_probe:
                raise SchedulerError("accounting unavailable")
            return super().probe_submission(submission_token)

    scheduler = _ProbeErrorScheduler(cluster="alpha")
    _task_value, workspace, scheduler, attempt = _hidden_submitting_emergency_fixture(
        tmp_path,
        scheduler=scheduler,
    )
    _write_hidden_submission_claim(
        workspace,
        attempt,
        claimed_at=datetime(2026, 9, 16, 17, 0, tzinfo=UTC),
    )
    hidden_job = _submit_hidden_attempt(scheduler, workspace, attempt)
    scheduler.fail_probe = True

    with pytest.raises(WorkflowError, match="probe failed"):
        request_cancellation(workspace, reason="probe error", scheduler=scheduler)
    assert not (workspace / "requests/cancel.receipt.json").exists()

    scheduler.fail_probe = False
    request_cancellation(workspace, reason="probe error", scheduler=scheduler)

    assert scheduler.cancelled == (hidden_job,)
    assert (workspace / "requests/cancel.receipt.json").is_file()


def test_emergency_cancel_broker_failure_requires_manual_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _task_value, workspace, scheduler, _jobs, _descriptors = _credentialed_emergency_fixture(
        tmp_path
    )

    class _FailingBroker:
        def revoke(self, **_kwargs: object) -> CredentialRevocationReceipt:
            raise RuntimeError("injected broker outage")

    monkeypatch.setattr(
        staircase_workflow,
        "_build_emergency_credential_broker",
        lambda *_args, **_kwargs: _FailingBroker(),
    )

    with pytest.raises(WorkflowError, match="manual reconciliation"):
        request_cancellation(workspace, reason="credentialed stop", scheduler=scheduler)

    assert scheduler.cancelled == ()
    assert not (workspace / "requests/cancel.receipt.json").exists()


def test_unexpected_controller_death_launches_and_self_bootstraps_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _task(tmp_path)
    scheduler = FakeScheduler(cluster="alpha")
    workspace = task.repository.workspace_root / "run"
    started = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)
    assert started.controller_job is not None
    _bootstrap_state(task, workspace, scheduler, started.controller_job)
    state = load_state(workspace / "state.json")
    lease = ControllerLease.acquire(
        workspace / "controller-lease.json",
        run_id=state.run_id,
        generation=1,
        controller_job=state.controller_job,
        owner_nonce="dead-controller-owner",
    )
    lease.release(remove_record=False)
    scheduler.transition(
        started.controller_job,
        JobStatus.FAILED,
        source=ObservationSource.ACCOUNTING,
    )

    recovered = _start(mode="onboard", task=task, workspace=workspace, scheduler=scheduler)

    assert recovered.controller_job is not None
    assert recovered.controller_job.cluster == "alpha"
    assert load_state(workspace / "state.json").generation == 1
    submission = scheduler.submissions[-1]
    assert submission.dependency == JobDependency(
        DependencyType.AFTERANY,
        started.controller_job,
    )
    monkeypatch.setenv("SLURM_JOB_ID", recovered.controller_job.job_id)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "alpha")
    run_controller(
        workspace,
        generation=2,
        owner_nonce="recovery-controller-owner",
        scheduler=scheduler,
        runtime_factory=_TerminalRuntime,
        signal_flags=_ControllerSignalFlags(),
        sleeper=lambda _seconds: None,
        preflight_check=_skip_preflight,
    )

    advanced = load_state(workspace / "state.json")
    assert advanced.generation == 2
    assert advanced.controller_job == JobReference(
        recovered.controller_job.job_id,
        cluster="alpha",
    )


def test_local_cancellation_reaches_terminal_report_with_injected_runtime(
    tmp_path: Path,
) -> None:
    task = _task(tmp_path, execution="local")
    scheduler = FakeScheduler()
    workspace = task.repository.workspace_root / "run"
    _start(mode="onboard", task=task, workspace=workspace)
    request_cancellation(workspace, reason="operator stop")

    class _CancellationRuntime:
        def __init__(self, *, workspace: Path, callbacks: Any, **_kwargs: object) -> None:
            self._workspace = workspace
            self._callbacks = callbacks

        def tick(self) -> RuntimeTickResult:
            assert self._callbacks.cancellation_reason() == "operator stop"
            return _TerminalRuntime(workspace=self._workspace).tick()

    run_controller(
        workspace,
        generation=1,
        owner_nonce="local-owner",
        scheduler=scheduler,
        runtime_factory=_CancellationRuntime,
        signal_flags=_ControllerSignalFlags(),
        sleeper=lambda _seconds: None,
    )

    state = load_state(workspace / "state.json")
    assert state.terminal_status is RunTerminalStatus.CANCELLED
    assert (workspace / "reports/terminal-report.json").is_file()
