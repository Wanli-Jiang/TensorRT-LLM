# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only crash-boundary tests for controller successor handoff."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import state
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
)
from agent_flow.workflows.staircase.common.submission_recovery import SubmissionRecoveryPolicy
from agent_flow.workflows.staircase.controller.successor import (
    SuccessorAction,
    SuccessorDisposition,
    SuccessorProtocolError,
    activate_successor,
    bootstrap_successor_process,
    recover_successor_lease,
    tick_successor_submission,
)

_TASK_DIGEST = "a" * 64
_RESPONSE_DIGEST = "b" * 64
_REASON = "advance before controller time limit"


class _AcceptedThenErroredScheduler(FakeScheduler):
    """Model loss of the ``sbatch`` reply after scheduler acceptance."""

    fail_after_accept = False

    def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        identity = super().submit(*args, **kwargs)  # type: ignore[arg-type]
        if self.fail_after_accept:
            self.fail_after_accept = False
            raise SchedulerError("accepted but reply was lost")
        return identity


class _CrashBeforeSubmitScheduler(FakeScheduler):
    """Model process death after claim persistence but before ``sbatch``."""

    crash_before_submit = False

    def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if self.crash_before_submit:
            self.crash_before_submit = False
            raise RuntimeError("simulated successor crash before scheduler call")
        return super().submit(*args, **kwargs)  # type: ignore[arg-type]


def _resources(workspace: Path) -> ResourceRequest:
    return ResourceRequest(
        label="controller-successor",
        account="coreai",
        partition="batch",
        time_limit="00:20:00",
        output_path=workspace / "logs" / "%j.out",
        error_path=workspace / "logs" / "%j.err",
        requeue=False,
    )


def _command(workspace: Path, generation: int = 2) -> InternalCommand:
    return InternalCommand(
        InternalEntrypoint.CONTROLLER,
        workspace=workspace,
        generation=generation,
        owner_nonce=f"controller-{generation:04d}",
    )


def _initialized_run(
    workspace: Path,
) -> tuple[Path, FakeScheduler, ResourceRequest, InternalCommand, JobIdentity]:
    scheduler = FakeScheduler(first_job_id=100, cluster="test")
    resources = _resources(workspace)
    predecessor = scheduler.submit(
        resources,
        _command(workspace, generation=1),
        "controller-bootstrap-0001",
    )
    run = state.RunState(
        run_id="run-1",
        task_digest=_TASK_DIGEST,
        base_commit="0123456789abcdef",
        generation=1,
        controller_bootstrap_status=state.ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-bootstrap-0001",
        controller_job=state.JobReference(
            predecessor.job_id,
            cluster=predecessor.cluster,
        ),
    )
    state_path = workspace / state.STATE_FILENAME
    state.initialize_state(state_path, run)
    return state_path, scheduler, resources, _command(workspace), predecessor


def _persist_intent(
    state_path: Path,
    scheduler: FakeScheduler,
    resources: ResourceRequest,
    command: InternalCommand,
):
    result = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
    )
    assert result.action is SuccessorAction.INTENT_PERSISTED
    assert result.permit is not None
    return result


def _submit_and_fence(
    state_path: Path,
    scheduler: FakeScheduler,
    resources: ResourceRequest,
    command: InternalCommand,
):
    intent = _persist_intent(state_path, scheduler, resources, command)
    submitted = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
        permit=intent.permit,
    )
    assert submitted.action is SuccessorAction.SUBMITTED
    fenced = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
    )
    assert fenced.action is SuccessorAction.FENCED
    assert fenced.disposition is SuccessorDisposition.EXIT
    assert fenced.job is not None
    return submitted, fenced


def test_successor_handoff_uses_exact_afterany_and_accounting_gate(
    tmp_path: Path,
) -> None:
    state_path, scheduler, resources, command, predecessor = _initialized_run(tmp_path)
    submitted, fenced = _submit_and_fence(state_path, scheduler, resources, command)

    successor_submission = scheduler.submissions[-1]
    assert successor_submission.identity == submitted.job
    assert successor_submission.dependency == JobDependency(
        DependencyType.AFTERANY,
        predecessor,
    )
    assert not successor_submission.resources.requeue
    persisted = state.load_state(state_path)
    assert persisted.generation == 2
    assert persisted.controller_lifecycle is state.ControllerLifecycle.AWAITING_PREDECESSOR
    assert tuple(record.generation for record in persisted.controller_generation_history) == (1, 2)

    waiting = activate_successor(
        state_path=state_path,
        scheduler=scheduler,
        generation=2,
        current_job=fenced.job,
    )
    assert waiting.action is SuccessorAction.WAITING_PREDECESSOR
    assert waiting.disposition is SuccessorDisposition.WAIT
    assert not waiting.state.predecessor_reconciliations

    scheduler.transition(
        predecessor,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    reconciled = activate_successor(
        state_path=state_path,
        scheduler=scheduler,
        generation=2,
        current_job=fenced.job,
    )
    assert reconciled.action is SuccessorAction.PREDECESSOR_RECONCILED
    receipt = state.load_state(state_path).predecessor_reconciliations[-1]
    assert receipt.predecessor_job.job_id == predecessor.job_id
    assert receipt.successor_job == state.JobReference("101", cluster="test")
    assert receipt.observation_source == "ACCOUNTING"

    ready = activate_successor(
        state_path=state_path,
        scheduler=scheduler,
        generation=2,
        current_job=fenced.job,
    )
    assert ready.action is SuccessorAction.READY
    assert ready.disposition is SuccessorDisposition.READY


def test_restart_consumes_durable_unclaimed_permit_exactly_once(tmp_path: Path) -> None:
    state_path, scheduler, resources, command, _predecessor = _initialized_run(tmp_path)
    _persist_intent(state_path, scheduler, resources, command)
    submission_count = len(scheduler.submissions)

    restarted = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
    )

    assert restarted.action is SuccessorAction.SUBMITTED
    assert len(scheduler.submissions) == submission_count + 1

    state_path, scheduler, resources, command, _predecessor = _initialized_run(tmp_path / "claimed")
    scheduler.__class__ = _AcceptedThenErroredScheduler
    intent = _persist_intent(state_path, scheduler, resources, command)
    scheduler.fail_after_accept = True
    uncertain = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
        permit=intent.permit,
    )
    assert uncertain.action is SuccessorAction.WAITING_ACCOUNTING
    assert uncertain.disposition is SuccessorDisposition.WAIT
    scheduler.transition(scheduler.submissions[-1].identity, JobStatus.UNKNOWN, visible=False)
    count_after_claim = len(scheduler.submissions)
    waiting = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
    )
    assert waiting.action is SuccessorAction.WAITING_ACCOUNTING
    assert waiting.disposition is SuccessorDisposition.WAIT
    assert len(scheduler.submissions) == count_after_claim


def test_successor_claim_crash_retries_after_bounded_absence_proof(tmp_path: Path) -> None:
    state_path, scheduler, resources, command, _predecessor = _initialized_run(tmp_path)
    scheduler.__class__ = _CrashBeforeSubmitScheduler
    intent = _persist_intent(state_path, scheduler, resources, command)
    scheduler.crash_before_submit = True  # type: ignore[attr-defined]
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=10,
        absence_samples_required=2,
        absence_sample_interval_seconds=2,
        max_submit_calls=2,
    )

    with pytest.raises(RuntimeError, match="successor crash"):
        tick_successor_submission(
            state_path=state_path,
            scheduler=scheduler,
            resources=resources,
            command=command,
            expected_generation=1,
            reason=_REASON,
            permit=intent.permit,
            submission_clock=lambda: now,
            submission_recovery_policy=policy,
        )
    submission_count = len(scheduler.submissions)

    waiting = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
        submission_clock=lambda: now + timedelta(seconds=10),
        submission_recovery_policy=policy,
    )
    assert waiting.action is SuccessorAction.WAITING_ACCOUNTING
    retried = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
        submission_clock=lambda: now + timedelta(seconds=12),
        submission_recovery_policy=policy,
    )
    assert retried.action is SuccessorAction.SUBMITTED
    assert len(scheduler.submissions) == submission_count + 1


def test_successor_second_claim_crash_becomes_manual_recovery(tmp_path: Path) -> None:
    class _TwoCrashes(FakeScheduler):
        crashes_remaining = 2

        def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.crashes_remaining:
                self.crashes_remaining -= 1
                raise RuntimeError("simulated successor process crash")
            return super().submit(*args, **kwargs)  # type: ignore[arg-type]

    state_path, scheduler, resources, command, _predecessor = _initialized_run(tmp_path)
    scheduler.__class__ = _TwoCrashes
    intent = _persist_intent(state_path, scheduler, resources, command)
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=10,
        absence_samples_required=2,
        absence_sample_interval_seconds=2,
        max_submit_calls=2,
    )

    def tick_at(offset: int):
        return tick_successor_submission(
            state_path=state_path,
            scheduler=scheduler,
            resources=resources,
            command=command,
            expected_generation=1,
            reason=_REASON,
            permit=intent.permit,
            submission_clock=lambda: now + timedelta(seconds=offset),
            submission_recovery_policy=policy,
        )

    with pytest.raises(RuntimeError, match="process crash"):
        tick_at(0)
    assert tick_at(10).action is SuccessorAction.WAITING_ACCOUNTING
    with pytest.raises(RuntimeError, match="process crash"):
        tick_at(12)
    assert tick_at(22).action is SuccessorAction.WAITING_ACCOUNTING
    exhausted = tick_at(24)
    assert exhausted.action is SuccessorAction.MANUAL_RECOVERY
    assert exhausted.disposition is SuccessorDisposition.BLOCKED


def test_crash_after_scheduler_acceptance_adopts_by_token_without_duplicate(
    tmp_path: Path,
) -> None:
    state_path, scheduler, resources, command, predecessor = _initialized_run(tmp_path)
    intent = _persist_intent(state_path, scheduler, resources, command)
    token = intent.state.controller_successors[-1].submission_token
    accepted = scheduler.submit(
        resources,
        command,
        token,
        dependency=JobDependency(DependencyType.AFTERANY, predecessor),
    )
    submission_count = len(scheduler.submissions)

    adopted = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
    )

    assert adopted.action is SuccessorAction.ADOPTED
    assert adopted.job == accepted
    assert len(scheduler.submissions) == submission_count
    assert state.load_state(state_path).controller_successors[-1].successor_job == (
        state.JobReference(accepted.job_id, cluster="test")
    )


def test_accepted_successor_self_bootstraps_from_exact_job_and_token(
    tmp_path: Path,
) -> None:
    state_path, scheduler, resources, command, predecessor = _initialized_run(tmp_path)
    intent = _persist_intent(state_path, scheduler, resources, command)
    token = intent.state.controller_successors[-1].submission_token
    accepted = scheduler.submit(
        resources,
        command,
        token,
        dependency=JobDependency(DependencyType.AFTERANY, predecessor),
    )

    bootstrapped = bootstrap_successor_process(
        state_path=state_path,
        scheduler=scheduler,
        generation=2,
        current_job=accepted,
    )

    assert bootstrapped.generation == 2
    assert bootstrapped.controller_job == state.JobReference("101", cluster="test")
    assert bootstrapped.controller_lifecycle is state.ControllerLifecycle.AWAITING_PREDECESSOR
    assert len(scheduler.submissions) == 2


def test_ambiguous_token_blocks_without_scheduler_mutation(tmp_path: Path) -> None:
    state_path, scheduler, resources, command, predecessor = _initialized_run(tmp_path)
    intent = _persist_intent(state_path, scheduler, resources, command)
    token = intent.state.controller_successors[-1].submission_token
    dependency = JobDependency(DependencyType.AFTERANY, predecessor)
    scheduler.submit(resources, command, token, dependency=dependency)
    scheduler.submit(resources, command, token, dependency=dependency)
    submission_count = len(scheduler.submissions)

    blocked = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
    )

    assert blocked.action is SuccessorAction.BLOCKED
    assert "ambiguous" in blocked.reason
    assert len(scheduler.submissions) == submission_count
    assert not scheduler.cancelled
    assert (tmp_path / "quarantine/submissions" / f"successor-{token}.json").is_file()


def test_fence_is_idempotent_for_restarted_predecessor_and_fences_old_writes(
    tmp_path: Path,
) -> None:
    state_path, scheduler, resources, command, _predecessor = _initialized_run(tmp_path)
    submitted, _fenced = _submit_and_fence(state_path, scheduler, resources, command)

    restarted = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason=_REASON,
    )
    assert restarted.action is SuccessorAction.FENCED
    assert restarted.disposition is SuccessorDisposition.EXIT

    stale = submitted.state.checkpoint_controller(
        reason="stale predecessor write",
        requeue_requested=False,
    )
    with pytest.raises(state.StateConflictError):
        state.save_state(
            state_path,
            stale,
            expected_revision=submitted.state.revision,
            expected_generation=1,
        )


def test_unknown_and_non_accounting_terminal_predecessor_never_activate(
    tmp_path: Path,
) -> None:
    state_path, scheduler, resources, command, predecessor = _initialized_run(tmp_path)
    _submitted, fenced = _submit_and_fence(state_path, scheduler, resources, command)
    scheduler.transition(predecessor, JobStatus.COMPLETED, visible=False)

    unknown = activate_successor(
        state_path=state_path,
        scheduler=scheduler,
        generation=2,
        current_job=fenced.job,
    )
    assert unknown.action is SuccessorAction.WAITING_PREDECESSOR
    assert not state.load_state(state_path).predecessor_reconciliations

    scheduler.transition(
        predecessor,
        JobStatus.COMPLETED,
        source=ObservationSource.QUEUE,
        visible=True,
    )
    queued = activate_successor(
        state_path=state_path,
        scheduler=scheduler,
        generation=2,
        current_job=fenced.job,
    )
    assert queued.action is SuccessorAction.WAITING_PREDECESSOR
    assert "not an accounting receipt" in queued.reason
    assert not state.load_state(state_path).predecessor_reconciliations


def test_successor_lease_requires_durable_exact_receipt(tmp_path: Path) -> None:
    state_path, scheduler, resources, command, predecessor = _initialized_run(tmp_path)
    lease_path = tmp_path / state.LEASE_FILENAME
    lease = state.ControllerLease.acquire(
        lease_path,
        run_id="run-1",
        generation=1,
        controller_job=state.JobReference("100", cluster="test"),
        owner_nonce="predecessor-owner",
    )
    lease.release(remove_record=False)
    _submitted, fenced = _submit_and_fence(state_path, scheduler, resources, command)

    with pytest.raises(SuccessorProtocolError, match="accounting receipt"):
        recover_successor_lease(
            lease_path=lease_path,
            state_path=state_path,
            generation=2,
            current_job=fenced.job,
            owner_nonce="successor-owner",
        )

    scheduler.transition(
        predecessor,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    activate_successor(
        state_path=state_path,
        scheduler=scheduler,
        generation=2,
        current_job=fenced.job,
    )
    successor_lease = recover_successor_lease(
        lease_path=lease_path,
        state_path=state_path,
        generation=2,
        current_job=fenced.job,
        owner_nonce="successor-owner",
    )
    try:
        assert successor_lease.record.generation == 2
        assert successor_lease.record.controller_job == state.JobReference("101", cluster="test")
    finally:
        successor_lease.release()


def test_detached_human_successor_requires_frozen_response(tmp_path: Path) -> None:
    state_path, scheduler, resources, command, _predecessor = _initialized_run(tmp_path)
    original = state.load_state(state_path)
    waiting = original.wait_for_human_input(
        request_id="approval-1",
        prompt_digest=_TASK_DIGEST,
        controller_requeue=True,
        reason="operator approval required",
    )
    state.save_state(state_path, waiting, expected_revision=0, expected_generation=1)

    blocked = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason="resume after operator response",
        human_request_id="approval-1",
    )
    assert blocked.action is SuccessorAction.BLOCKED
    assert "human response" in blocked.reason

    answered = waiting.record_human_response(
        request_id="approval-1",
        response_digest=_RESPONSE_DIGEST,
    )
    state.save_state(state_path, answered, expected_revision=1, expected_generation=1)
    intent = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=resources,
        command=command,
        expected_generation=1,
        reason="resume after operator response",
        human_request_id="approval-1",
    )
    assert intent.action is SuccessorAction.INTENT_PERSISTED
    assert intent.state.controller_successors[-1].human_request_id == "approval-1"


def test_successor_submission_rejects_automatic_requeue(tmp_path: Path) -> None:
    state_path, scheduler, resources, command, _predecessor = _initialized_run(tmp_path)

    blocked = tick_successor_submission(
        state_path=state_path,
        scheduler=scheduler,
        resources=replace(resources, requeue=True),
        command=command,
        expected_generation=1,
        reason=_REASON,
    )

    assert blocked.action is SuccessorAction.BLOCKED
    assert "disable Slurm automatic requeue" in blocked.reason
    assert not state.load_state(state_path).controller_successors
