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

"""CPU-only tests for deterministic Staircase attempt reconciliation."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

import pytest

from agent_flow.workflows.staircase.common.artifacts import (
    INPUT_FILENAME,
    ResultExpectation,
    WorkerInputManifest,
    WorkerResultManifest,
    WorkerResultStatus,
    publish_result,
    write_input_manifest,
)
from agent_flow.workflows.staircase.common.slurm import (
    CommandResult,
    FakeScheduler,
    InternalCommand,
    InternalEntrypoint,
    JobIdentity,
    JobStatus,
    ObservationSource,
    ResourceRequest,
    SchedulerError,
    SlurmScheduler,
)
from agent_flow.workflows.staircase.common.submission_recovery import SubmissionRecoveryPolicy
from agent_flow.workflows.staircase.controller.attempts import (
    AttemptAction,
    AttemptEngineError,
    AttemptExecution,
    AttemptPolicy,
    RetryLimitExceeded,
    cancel_owned_attempt,
    cancel_owned_attempts,
    cancel_submitting_intent,
    create_infrastructure_retry,
    logical_attempt_ordinal,
    logical_attempt_root,
    plan_infrastructure_retry,
    tick_attempt,
)
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    JobReference,
    Role,
    WorkItemKind,
    WorkItemRecord,
)

TASK_DIGEST = "a" * 64
CANDIDATE_DIGEST = "b" * 64
TOKEN = "attempt-token-0001"


def _attempt(
    *,
    attempt_id: str = "attempt-1",
    sequence: int = 1,
    status: AttemptStatus = AttemptStatus.PREPARED,
    job: JobIdentity | None = None,
    scheduler_state: str | None = None,
    terminal_reason: str | None = None,
) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt_id,
        item_id="item-1",
        sequence=sequence,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=status,
        profile=DomainProfile.SMITH,
        resource_class="smith_agent",
        submission_token=(TOKEN if status is not AttemptStatus.PREPARED else None),
        job=(
            JobReference(
                job.job_id,
                array_task_id=job.array_task_id,
                cluster=job.cluster,
            )
            if job is not None
            else None
        ),
        scheduler_state=scheduler_state,
        terminal_reason=terminal_reason,
    )


def _resources(tmp_path: Path, *, requeue: bool = False) -> ResourceRequest:
    return ResourceRequest(
        label="smith-entry",
        account="coreai",
        partition="batch",
        time_limit="00:10:00",
        output_path=tmp_path / "logs/%j.out",
        error_path=tmp_path / "logs/%j.err",
        requeue=requeue,
    )


def _execution(
    tmp_path: Path,
    attempt: AttemptRecord,
    *,
    resources: ResourceRequest | None = None,
) -> AttemptExecution:
    attempt_dir = tmp_path / "items" / attempt.item_id / "attempts" / attempt.attempt_id
    attempt_dir.mkdir(parents=True)
    input_digest = write_input_manifest(
        attempt_dir / INPUT_FILENAME,
        WorkerInputManifest(
            run_id="run-1",
            item_id=attempt.item_id,
            attempt_id=attempt.attempt_id,
            task_digest=TASK_DIGEST,
            generation=attempt.generation,
            role=attempt.role,
            profile=attempt.profile,
            worktree=str(tmp_path / "worktrees" / attempt.item_id),
        ),
    )
    return AttemptExecution(
        resources=resources or _resources(tmp_path),
        command=InternalCommand(
            InternalEntrypoint.WORKER,
            input_bundle=attempt_dir / INPUT_FILENAME,
        ),
        submission_token=TOKEN,
        attempt_dir=attempt_dir,
        expectation=ResultExpectation(
            run_id="run-1",
            item_id=attempt.item_id,
            attempt_id=attempt.attempt_id,
            task_digest=TASK_DIGEST,
            generation=attempt.generation,
            input_digest=input_digest,
        ),
        receipt_root=tmp_path / "receipts",
        quarantine_root=tmp_path / "quarantine",
    )


def _publish(
    execution: AttemptExecution,
    *,
    status: WorkerResultStatus = WorkerResultStatus.SUCCEEDED,
) -> str:
    expectation = execution.expectation
    return publish_result(
        execution.attempt_dir,
        WorkerResultManifest(
            run_id=expectation.run_id,
            item_id=expectation.item_id,
            attempt_id=expectation.attempt_id,
            task_digest=expectation.task_digest,
            generation=expectation.generation,
            input_digest=expectation.input_digest,
            status=status,
            summary=f"worker returned {status.value}",
            candidate_digest=(CANDIDATE_DIGEST if status is WorkerResultStatus.SUCCEEDED else None),
        ),
    )


def _scheduled_attempt(
    tmp_path: Path,
    scheduler: FakeScheduler,
) -> tuple[AttemptRecord, AttemptExecution, JobIdentity]:
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    identity = scheduler.submit(
        execution.resources,
        execution.command,
        TOKEN,
    )
    running = _attempt(
        status=AttemptStatus.RUNNING,
        job=identity,
        scheduler_state=JobStatus.RUNNING.value,
    )
    scheduler.transition(identity, JobStatus.RUNNING)
    return running, execution, identity


def test_prepared_tick_persists_intent_without_submitting(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)

    result = tick_attempt(prepared, scheduler=scheduler, execution=execution)

    assert result.action is AttemptAction.PERSIST_SUBMITTING
    assert result.attempt.status is AttemptStatus.SUBMITTING
    assert result.attempt.submission_token == TOKEN
    assert scheduler.submissions == ()


def test_submitting_tick_adopts_crash_window_job_by_token(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(
        prepared,
        scheduler=scheduler,
        execution=execution,
    ).attempt
    previously_submitted = scheduler.submit(
        execution.resources,
        execution.command,
        TOKEN,
    )

    result = tick_attempt(submitting, scheduler=scheduler, execution=execution)

    assert result.action is AttemptAction.ADOPTED
    assert result.attempt.status is AttemptStatus.SUBMITTED
    assert result.attempt.job == JobReference(job_id=previously_submitted.job_id)
    assert len(scheduler.submissions) == 1


def test_real_strict_probe_argv_adopts_an_accepted_lost_reply(tmp_path: Path) -> None:
    class _Executor:
        def __init__(self, results: Sequence[CommandResult]) -> None:
            self.results = list(results)
            self.calls: list[tuple[str, ...]] = []

        def run(self, argv: Sequence[str], *, input_text: str | None = None) -> CommandResult:
            assert input_text is None
            self.calls.append(tuple(argv))
            return self.results.pop(0)

    job_name = f"staircase-{TOKEN}"
    queue_row = f"43210|PENDING|Resources|staircase:{TOKEN};label=x|tester|{job_name}\n"
    ownership_row = f"43210|tester|staircase:{TOKEN};label=x\n"
    executor = _Executor(
        (
            CommandResult(0, queue_row),
            CommandResult(0, ""),
            CommandResult(0, ownership_row),
        )
    )
    scheduler = SlurmScheduler(executor=executor, username="tester")
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt

    adopted = tick_attempt(submitting, scheduler=scheduler, execution=execution)

    assert adopted.action is AttemptAction.ADOPTED
    assert adopted.attempt.job == JobReference("43210")
    assert "--format=%i|%T|%R|%k|%u|%j" in executor.calls[0]
    assert "--format=JobIDRaw,State,Reason,Comment,User,JobName" in executor.calls[1]
    assert "--format=%i|%u|%k" in executor.calls[2]


def test_submitting_tick_persists_exact_new_job_and_disables_requeue(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(
        prepared,
        scheduler=scheduler,
        execution=execution,
    ).attempt

    result = tick_attempt(submitting, scheduler=scheduler, execution=execution)

    assert result.action is AttemptAction.SUBMITTED
    assert result.attempt.job == JobReference(job_id="10000")
    assert scheduler.submissions[0].resources.requeue is False

    requeue_execution = replace(execution, resources=_resources(tmp_path, requeue=True))
    with pytest.raises(AttemptEngineError, match="requeue is disabled"):
        tick_attempt(submitting, scheduler=FakeScheduler(), execution=requeue_execution)


def test_expired_submitting_credential_allows_adoption_but_never_launches(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    prepared = _attempt()
    execution = replace(
        _execution(tmp_path, prepared),
        credential_expires_at_epoch_seconds=int(now.timestamp()) - 1,
    )
    submitting = tick_attempt(prepared, scheduler=FakeScheduler(), execution=execution).attempt

    empty_scheduler = FakeScheduler()
    blocked = tick_attempt(
        submitting,
        scheduler=empty_scheduler,
        execution=execution,
        submission_clock=lambda: now,
    )
    assert blocked.attempt.status is AttemptStatus.FAILED
    assert blocked.attempt.scheduler_state == "SUBMISSION_MANUAL_RECOVERY"
    assert empty_scheduler.submissions == ()

    visible_scheduler = FakeScheduler()
    identity = visible_scheduler.submit(execution.resources, execution.command, TOKEN)
    adopted = tick_attempt(
        submitting,
        scheduler=visible_scheduler,
        execution=execution,
        submission_clock=lambda: now,
    )
    assert adopted.action is AttemptAction.ADOPTED
    assert adopted.attempt.job == JobReference(job_id=identity.job_id)
    assert len(visible_scheduler.submissions) == 1


def test_submitting_intent_cancellation_is_durable_and_never_submits(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt

    cancelled = cancel_submitting_intent(
        submitting,
        scheduler=scheduler,
        execution=execution,
        cancellation_reason="operator request",
    )
    replayed = cancel_submitting_intent(
        submitting,
        scheduler=scheduler,
        execution=execution,
        cancellation_reason="operator request",
    )

    assert cancelled.action is AttemptAction.CANCELLED_INTENT
    assert cancelled.attempt.status is AttemptStatus.CANCELLED
    assert replayed.attempt == cancelled.attempt
    assert scheduler.submissions == ()
    assert (
        execution.receipt_root / "submissions" / submitting.attempt_id / "cancellation.json"
    ).is_file()


def test_submitting_intent_cancellation_adopts_lost_reply_before_exact_cancel(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt
    identity = scheduler.submit(execution.resources, execution.command, TOKEN)

    adopted = cancel_submitting_intent(
        submitting,
        scheduler=scheduler,
        execution=execution,
        cancellation_reason="operator request",
    )
    cancel_owned_attempt(adopted.attempt, scheduler)

    assert adopted.action is AttemptAction.ADOPTED
    assert adopted.attempt.job == JobReference(job_id=identity.job_id)
    assert scheduler.cancelled == (identity,)


def test_submitting_intent_cancellation_persists_ambiguous_manual_boundary(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt
    scheduler.submit(execution.resources, execution.command, TOKEN)
    scheduler.submit(execution.resources, execution.command, TOKEN)

    blocked = cancel_submitting_intent(
        submitting,
        scheduler=scheduler,
        execution=execution,
        cancellation_reason="operator request",
    )

    assert blocked.attempt.status is AttemptStatus.FAILED
    assert blocked.attempt.scheduler_state == "SUBMISSION_MANUAL_RECOVERY"
    assert scheduler.cancelled == ()
    assert (
        execution.receipt_root / "submissions" / submitting.attempt_id / "manual-recovery.json"
    ).is_file()


def test_submitting_intent_cancellation_bounds_claim_and_probe_error(
    tmp_path: Path,
) -> None:
    class _CrashBeforeCall(FakeScheduler):
        def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            raise RuntimeError("crash after claim")

    scheduler = _CrashBeforeCall()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=1,
        absence_samples_required=2,
        absence_sample_interval_seconds=1,
    )
    with pytest.raises(RuntimeError, match="crash after claim"):
        tick_attempt(
            submitting,
            scheduler=scheduler,
            execution=execution,
            submission_clock=lambda: now,
            submission_recovery_policy=policy,
        )
    first = cancel_submitting_intent(
        submitting,
        scheduler=scheduler,
        execution=execution,
        cancellation_reason="operator request",
        submission_clock=lambda: now + timedelta(seconds=1),
        submission_recovery_policy=policy,
    )
    second = cancel_submitting_intent(
        first.attempt,
        scheduler=scheduler,
        execution=execution,
        cancellation_reason="operator request",
        submission_clock=lambda: now + timedelta(seconds=2),
        submission_recovery_policy=policy,
    )
    assert first.action is AttemptAction.WAITING_FOR_ACCOUNTING
    assert second.action is AttemptAction.CANCELLED_INTENT
    assert scheduler.submissions == ()

    class _ProbeError(FakeScheduler):
        def probe_submission(self, submission_token: str):  # type: ignore[no-untyped-def]
            raise SchedulerError("accounting unavailable")

    other_prepared = replace(prepared, attempt_id="attempt-0002", sequence=2)
    other_execution = replace(
        _execution(tmp_path, other_prepared),
        submission_token="attempt-00000002",
    )
    other_submitting = tick_attempt(
        other_prepared,
        scheduler=_ProbeError(),
        execution=other_execution,
    ).attempt
    failed = cancel_submitting_intent(
        other_submitting,
        scheduler=_ProbeError(),
        execution=other_execution,
        cancellation_reason="operator request",
    )
    assert failed.attempt.scheduler_state == "SUBMISSION_MANUAL_RECOVERY"


def test_claimed_submitting_unknown_waits_without_duplicate(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt

    accepted = tick_attempt(submitting, scheduler=scheduler, execution=execution)
    assert accepted.action is AttemptAction.SUBMITTED
    assert accepted.attempt.job is not None
    scheduler.transition(
        JobIdentity(accepted.attempt.job.job_id),
        JobStatus.UNKNOWN,
        visible=False,
    )

    waiting = tick_attempt(submitting, scheduler=scheduler, execution=execution)

    assert waiting.action is AttemptAction.WAITING_FOR_ACCOUNTING
    assert waiting.attempt.status is AttemptStatus.SUBMITTING
    assert len(scheduler.submissions) == 1


def test_crash_after_attempt_claim_retries_only_after_durable_absence_proof(
    tmp_path: Path,
) -> None:
    class _CrashBeforeSubmit(FakeScheduler):
        crash = True

        def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.crash:
                self.crash = False
                raise RuntimeError("simulated process crash before scheduler call")
            return super().submit(*args, **kwargs)  # type: ignore[arg-type]

    scheduler = _CrashBeforeSubmit()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=10,
        absence_samples_required=2,
        absence_sample_interval_seconds=2,
        max_submit_calls=2,
    )

    with pytest.raises(RuntimeError, match="process crash"):
        tick_attempt(
            submitting,
            scheduler=scheduler,
            execution=execution,
            submission_clock=lambda: now,
            submission_recovery_policy=policy,
        )
    assert scheduler.submissions == ()

    waiting = tick_attempt(
        submitting,
        scheduler=scheduler,
        execution=execution,
        submission_clock=lambda: now + timedelta(seconds=10),
        submission_recovery_policy=policy,
    )
    assert waiting.action is AttemptAction.WAITING_FOR_ACCOUNTING
    retried = tick_attempt(
        submitting,
        scheduler=scheduler,
        execution=execution,
        submission_clock=lambda: now + timedelta(seconds=12),
        submission_recovery_policy=policy,
    )
    assert retried.action is AttemptAction.SUBMITTED
    assert len(scheduler.submissions) == 1


def test_attempt_second_claim_crash_becomes_terminal_manual_recovery(tmp_path: Path) -> None:
    class _TwoCrashes(FakeScheduler):
        crashes_remaining = 2

        def submit(self, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            if self.crashes_remaining:
                self.crashes_remaining -= 1
                raise RuntimeError("simulated attempt process crash")
            return super().submit(*args, **kwargs)  # type: ignore[arg-type]

    scheduler = _TwoCrashes()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    submitting = tick_attempt(prepared, scheduler=scheduler, execution=execution).attempt
    now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)
    policy = SubmissionRecoveryPolicy(
        visibility_grace_seconds=10,
        absence_samples_required=2,
        absence_sample_interval_seconds=2,
        max_submit_calls=2,
    )

    def tick_at(offset: int):
        return tick_attempt(
            submitting,
            scheduler=scheduler,
            execution=execution,
            submission_clock=lambda: now + timedelta(seconds=offset),
            submission_recovery_policy=policy,
        )

    with pytest.raises(RuntimeError, match="process crash"):
        tick_at(0)
    assert tick_at(10).action is AttemptAction.WAITING_FOR_ACCOUNTING
    with pytest.raises(RuntimeError, match="process crash"):
        tick_at(12)
    assert tick_at(22).action is AttemptAction.WAITING_FOR_ACCOUNTING
    exhausted = tick_at(24)
    assert exhausted.action is AttemptAction.TERMINAL_FAILURE
    assert exhausted.attempt.status is AttemptStatus.FAILED
    assert exhausted.attempt.scheduler_state == "SUBMISSION_MANUAL_RECOVERY"


def test_ambiguous_submission_token_is_detected_without_wildcard_action(
    tmp_path: Path,
) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    scheduler.submit(execution.resources, execution.command, TOKEN)
    scheduler.submit(execution.resources, execution.command, TOKEN)
    submitting = prepared.transition(
        AttemptStatus.SUBMITTING,
        submission_token=TOKEN,
    )

    result = tick_attempt(submitting, scheduler=scheduler, execution=execution)

    assert result.action is AttemptAction.DUPLICATE_SUBMISSION
    assert result.attempt.status is AttemptStatus.FAILED
    assert tuple(identity.scheduler_id for identity in result.duplicate_jobs) == (
        "10000",
        "10001",
    )
    assert result.quarantined_path is not None
    assert result.quarantined_path.is_file()
    assert scheduler.cancelled == ()


def test_queue_to_accounting_unknown_has_bounded_grace(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    running, execution, identity = _scheduled_attempt(tmp_path, scheduler)
    scheduler.transition(identity, JobStatus.RUNNING, source=ObservationSource.QUEUE)
    visible = tick_attempt(running, scheduler=scheduler, execution=execution)
    assert visible.action is AttemptAction.OBSERVED
    assert visible.unknown_observations == 0

    scheduler.transition(identity, JobStatus.UNKNOWN, visible=False)
    first = tick_attempt(
        visible.attempt,
        scheduler=scheduler,
        execution=execution,
        policy=AttemptPolicy(accounting_unknown_grace_ticks=2),
    )
    second = tick_attempt(
        first.attempt,
        scheduler=scheduler,
        execution=execution,
        policy=AttemptPolicy(accounting_unknown_grace_ticks=2),
        unknown_observations=first.unknown_observations,
    )
    assert first.action is AttemptAction.WAITING_FOR_ACCOUNTING
    assert second.action is AttemptAction.WAITING_FOR_ACCOUNTING
    assert second.attempt.status is AttemptStatus.RUNNING

    scheduler.transition(
        identity,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    recovered = tick_attempt(
        second.attempt,
        scheduler=scheduler,
        execution=execution,
        policy=AttemptPolicy(accounting_unknown_grace_ticks=2),
        unknown_observations=second.unknown_observations,
    )
    assert recovered.attempt.status is AttemptStatus.TERMINAL_OBSERVED
    assert recovered.unknown_observations == 0

    scheduler.transition(identity, JobStatus.UNKNOWN, visible=False)
    cursor = 0
    current = running
    for _ in range(3):
        lost = tick_attempt(
            current,
            scheduler=scheduler,
            execution=execution,
            policy=AttemptPolicy(accounting_unknown_grace_ticks=2),
            unknown_observations=cursor,
        )
        current = lost.attempt
        cursor = lost.unknown_observations
    assert lost.attempt.status is AttemptStatus.LOST
    assert lost.attempt.scheduler_state == JobStatus.UNKNOWN.value


@pytest.mark.parametrize(
    ("scheduler_status", "expected_attempt_status"),
    [
        (JobStatus.COMPLETED, AttemptStatus.COLLECTING),
        (JobStatus.CANCELLED, AttemptStatus.CANCELLED),
        (JobStatus.FAILED, AttemptStatus.FAILED),
        (JobStatus.PREEMPTED, AttemptStatus.PREEMPTED),
        (JobStatus.NODE_FAIL, AttemptStatus.RETRYABLE_FAILED),
        (JobStatus.OUT_OF_MEMORY, AttemptStatus.RETRYABLE_FAILED),
        (JobStatus.TIMEOUT, AttemptStatus.RETRYABLE_FAILED),
    ],
)
def test_every_scheduler_terminal_state_has_an_explicit_mapping(
    tmp_path: Path,
    scheduler_status: JobStatus,
    expected_attempt_status: AttemptStatus,
) -> None:
    scheduler = FakeScheduler()
    running, execution, identity = _scheduled_attempt(tmp_path, scheduler)
    scheduler.transition(
        identity,
        scheduler_status,
        reason=f"reason for {scheduler_status.value}",
        source=ObservationSource.ACCOUNTING,
    )

    observed = tick_attempt(running, scheduler=scheduler, execution=execution)
    terminal = tick_attempt(observed.attempt, scheduler=scheduler, execution=execution)

    assert observed.attempt.status is AttemptStatus.TERMINAL_OBSERVED
    assert observed.attempt.scheduler_state == scheduler_status.value
    assert terminal.attempt.status is expected_attempt_status
    if scheduler_status is not JobStatus.COMPLETED:
        assert scheduler_status.value in (terminal.attempt.terminal_reason or "")


def test_completed_job_requires_complete_valid_result(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    running, execution, identity = _scheduled_attempt(tmp_path, scheduler)
    scheduler.transition(identity, JobStatus.COMPLETED, source=ObservationSource.ACCOUNTING)
    observed = tick_attempt(running, scheduler=scheduler, execution=execution)
    collecting = tick_attempt(observed.attempt, scheduler=scheduler, execution=execution)

    missing = tick_attempt(collecting.attempt, scheduler=scheduler, execution=execution)

    assert missing.attempt.status is AttemptStatus.FAILED
    assert "application failure" in (missing.attempt.terminal_reason or "")
    assert "COMPLETE/result" in (missing.attempt.terminal_reason or "")


def test_completed_valid_result_is_ingested_idempotently(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    running, execution, identity = _scheduled_attempt(tmp_path, scheduler)
    expected_digest = _publish(execution)
    scheduler.transition(identity, JobStatus.COMPLETED, source=ObservationSource.ACCOUNTING)
    observed = tick_attempt(running, scheduler=scheduler, execution=execution)
    collecting = tick_attempt(observed.attempt, scheduler=scheduler, execution=execution)

    validated = tick_attempt(collecting.attempt, scheduler=scheduler, execution=execution)
    replayed = tick_attempt(collecting.attempt, scheduler=scheduler, execution=execution)

    assert validated.action is AttemptAction.VALIDATED
    assert validated.attempt.status is AttemptStatus.VALIDATED
    assert validated.attempt.result_digest == expected_digest
    assert validated.attempt.candidate_digest is None
    assert replayed.attempt == validated.attempt


def test_worker_failure_remains_application_failure(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    running, execution, identity = _scheduled_attempt(tmp_path, scheduler)
    result_digest = _publish(execution, status=WorkerResultStatus.FAILED)
    scheduler.transition(identity, JobStatus.COMPLETED, source=ObservationSource.ACCOUNTING)
    observed = tick_attempt(running, scheduler=scheduler, execution=execution)
    collecting = tick_attempt(observed.attempt, scheduler=scheduler, execution=execution)

    failed = tick_attempt(collecting.attempt, scheduler=scheduler, execution=execution)

    assert failed.attempt.status is AttemptStatus.FAILED
    assert failed.attempt.scheduler_state == JobStatus.COMPLETED.value
    assert failed.attempt.result_digest == result_digest
    assert "application failure" in (failed.attempt.terminal_reason or "")


def test_late_result_for_terminal_attempt_is_quarantined(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    identity = JobIdentity("12345")
    preempted = _attempt(
        status=AttemptStatus.PREEMPTED,
        job=identity,
        scheduler_state=JobStatus.PREEMPTED.value,
        terminal_reason="PREEMPTED: reclaimed",
    )
    execution = _execution(tmp_path, preempted)
    _publish(execution)

    result = tick_attempt(preempted, scheduler=scheduler, execution=execution)

    assert result.action is AttemptAction.QUARANTINED
    assert result.quarantined_path is not None
    assert result.quarantined_path.is_dir()
    assert not execution.attempt_dir.exists()


def test_bounded_infrastructure_retry_preserves_history() -> None:
    failed = _attempt(
        status=AttemptStatus.RETRYABLE_FAILED,
        job=JobIdentity("10000"),
        scheduler_state=JobStatus.NODE_FAIL.value,
        terminal_reason="NODE_FAIL: node unavailable",
    )
    item = WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
        attempts=(failed,),
    )

    retried = create_infrastructure_retry(
        item,
        failed.attempt_id,
        new_attempt_id="attempt-2",
        generation=2,
        policy=AttemptPolicy(max_node_fail_retries=1),
    )

    assert retried.attempts[0] == failed
    assert retried.attempts[1].status is AttemptStatus.PREPARED
    assert retried.attempts[1].generation == 2
    assert retried.attempts[1].job is None
    assert retried.attempts[1].submission_token is None
    assert retried.attempts[1].resource_class == failed.resource_class
    assert retried.attempts[1].predecessor_attempt_id == failed.attempt_id

    second_failure = replace(
        retried.attempts[1],
        status=AttemptStatus.RETRYABLE_FAILED,
        submission_token="attempt-token-0002",
        job=JobReference("10001"),
        scheduler_state=JobStatus.NODE_FAIL.value,
        terminal_reason="NODE_FAIL: another node unavailable",
    )
    exhausted = replace(retried, attempts=(failed, second_failure))
    with pytest.raises(RetryLimitExceeded, match="exhausted 1"):
        create_infrastructure_retry(
            exhausted,
            second_failure.attempt_id,
            new_attempt_id="attempt-3",
            generation=2,
            policy=AttemptPolicy(max_node_fail_retries=1),
        )


def test_application_failure_is_not_infrastructure_retry() -> None:
    failed = _attempt(
        status=AttemptStatus.FAILED,
        job=JobIdentity("10000"),
        scheduler_state=JobStatus.COMPLETED.value,
        terminal_reason="application failure: tests failed",
    )
    item = WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
        attempts=(failed,),
    )

    with pytest.raises(AttemptEngineError, match="not eligible for automatic retry"):
        create_infrastructure_retry(
            item,
            failed.attempt_id,
            new_attempt_id="attempt-2",
            generation=1,
        )


@pytest.mark.parametrize(
    ("status", "scheduler_state", "result_digest"),
    [
        (AttemptStatus.LOST, JobStatus.UNKNOWN.value, None),
        (AttemptStatus.RETRYABLE_FAILED, "SUBMIT_ERROR", None),
        (AttemptStatus.RETRYABLE_FAILED, JobStatus.OUT_OF_MEMORY.value, None),
        (AttemptStatus.RETRYABLE_FAILED, JobStatus.TIMEOUT.value, None),
        (AttemptStatus.RETRYABLE_FAILED, JobStatus.COMPLETED.value, TASK_DIGEST),
    ],
)
def test_automatic_retry_rejects_non_policy_categories(
    status: AttemptStatus,
    scheduler_state: str,
    result_digest: str | None,
) -> None:
    failed = replace(
        _attempt(
            status=status,
            job=(
                JobIdentity("10000")
                if scheduler_state not in {"SUBMIT_ERROR", JobStatus.UNKNOWN.value}
                else None
            ),
            scheduler_state=scheduler_state,
            terminal_reason="not an eligible automatic retry",
        ),
        result_digest=result_digest,
    )
    item = WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
        attempts=(failed,),
    )

    with pytest.raises(AttemptEngineError, match="only PREEMPTED and scheduler NODE_FAIL"):
        create_infrastructure_retry(
            item,
            failed.attempt_id,
            new_attempt_id="attempt-2",
            generation=1,
        )


def test_retry_plan_replay_is_idempotent_and_ordinals_collapse_lineage() -> None:
    failed = _attempt(
        status=AttemptStatus.RETRYABLE_FAILED,
        job=JobIdentity("10000"),
        scheduler_state=JobStatus.NODE_FAIL.value,
        terminal_reason="NODE_FAIL: node unavailable",
    )
    item = WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
        attempts=(failed,),
    )
    retried = create_infrastructure_retry(
        item,
        failed.attempt_id,
        new_attempt_id="attempt-2",
        generation=2,
    )

    replay = plan_infrastructure_retry(
        retried,
        failed.attempt_id,
        new_attempt_id="attempt-2",
        generation=2,
    )
    assert replay.already_appended
    assert (
        create_infrastructure_retry(
            retried,
            failed.attempt_id,
            new_attempt_id="attempt-2",
            generation=2,
        )
        is retried
    )
    assert logical_attempt_root(retried, "attempt-2") == failed
    assert logical_attempt_ordinal(retried, "attempt-1") == 1
    assert logical_attempt_ordinal(retried, "attempt-2") == 1

    second_root = replace(
        retried.attempts[1],
        attempt_id="attempt-3",
        sequence=3,
        predecessor_attempt_id=None,
    )
    with_second_root = retried.add_attempt(second_root)
    assert logical_attempt_ordinal(with_second_root, "attempt-3") == 2


def test_cancellation_uses_only_exact_persisted_owned_jobs(tmp_path: Path) -> None:
    scheduler = FakeScheduler()
    prepared = _attempt()
    execution = _execution(tmp_path, prepared)
    first_identity = scheduler.submit(execution.resources, execution.command, TOKEN)
    second_identity = scheduler.submit(
        execution.resources,
        execution.command,
        "attempt-token-0002",
    )
    first = _attempt(
        attempt_id="attempt-1",
        sequence=1,
        status=AttemptStatus.RUNNING,
        job=first_identity,
        scheduler_state=JobStatus.RUNNING.value,
    )
    second = replace(
        first,
        attempt_id="attempt-2",
        sequence=2,
        submission_token="attempt-token-0002",
        job=JobReference(second_identity.job_id),
    )

    cancelled = cancel_owned_attempts((first, second), scheduler)

    assert cancelled == (first_identity, second_identity)
    assert scheduler.cancelled == (first_identity, second_identity)
    assert all(identity.scheduler_id.isdecimal() for identity in scheduler.cancelled)

    with pytest.raises(AttemptEngineError, match="without a persisted job"):
        cancel_owned_attempt(prepared, scheduler)
