# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic Staircase controller recovery policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialDescriptor,
    CredentialRevocationCause,
    CredentialRevocationReceipt,
    locate_credential_provision,
    prepare_credential_provision,
    revoke_terminal_attempt_credentials,
)
from agent_flow.workflows.staircase.common.slurm import (
    JobIdentity,
    JobObservation,
    JobStatus,
    ObservationSource,
)
from agent_flow.workflows.staircase.controller.recovery import (
    ControllerSignal,
    HumanInputStatus,
    OrphanAction,
    RecoveryError,
    decide_generation_advance,
    human_input_recovery_intent,
    plan_attempt_adoption,
    plan_orphan_recovery,
    revoke_reconciled_generation_credentials,
    signal_recovery_intent,
)
from agent_flow.workflows.staircase.state import (
    PLANNING_ITEM_ID,
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    ControllerBootstrapStatus,
    JobReference,
    Role,
    RunState,
    StateConflictError,
)


def _observation(job_id: str, status: JobStatus) -> JobObservation:
    return JobObservation(
        JobIdentity(job_id),
        status,
        None,
        ObservationSource.ACCOUNTING,
        status.value,
    )


@pytest.mark.parametrize("signal", [ControllerSignal.ADVANCE, ControllerSignal.PREEMPTION])
def test_signal_stops_dispatch_and_checkpoints_without_worker_requeue(
    signal: ControllerSignal,
) -> None:
    intent = signal_recovery_intent(signal, controller_requeue_enabled=True)

    assert intent.stop_dispatch
    assert intent.checkpoint_required
    assert intent.leave_workers_running
    assert intent.controller_requeue
    assert not intent.worker_requeue


@pytest.mark.parametrize(
    "observation",
    [None, _observation("41", JobStatus.RUNNING), _observation("41", JobStatus.UNKNOWN)],
)
def test_generation_advance_rejects_missing_or_nonterminal_observation(
    observation: JobObservation | None,
) -> None:
    decision = decide_generation_advance(JobReference("41"), observation)

    assert not decision.allowed
    with pytest.raises(RecoveryError):
        decision.require_allowed()


def test_generation_advance_requires_exact_predecessor_identity() -> None:
    mismatched = decide_generation_advance(
        JobReference("41"),
        _observation("42", JobStatus.PREEMPTED),
    )
    exact = decide_generation_advance(
        JobReference("41"),
        _observation("41", JobStatus.PREEMPTED),
    )

    assert not mismatched.allowed
    assert exact.allowed
    exact.require_allowed()


def test_successor_adopts_all_old_generation_intents_before_new_work() -> None:
    prepared = AttemptRecord(
        attempt_id="plan-prepared",
        item_id=PLANNING_ITEM_ID,
        sequence=1,
        role=Role.PLAN_DRAFTER,
        kind=AttemptKind.ROLE,
        generation=1,
        resource_class="plan_drafter",
    )
    submitting = AttemptRecord(
        attempt_id="plan-submitting",
        item_id=PLANNING_ITEM_ID,
        sequence=2,
        role=Role.PLAN_REVIEWER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.SUBMITTING,
        submission_token="plan-token",
        resource_class="plan_reviewer",
    )
    running = AttemptRecord(
        attempt_id="plan-running",
        item_id=PLANNING_ITEM_ID,
        sequence=3,
        role=Role.PLAN_REVIEWER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.RUNNING,
        submission_token="running-token",
        job=JobReference("501"),
        scheduler_state=JobStatus.RUNNING.value,
        resource_class="plan_reviewer",
    )
    predecessor_job = JobReference("400")
    predecessor = RunState(
        run_id="run-adoption",
        task_digest="a" * 64,
        base_commit="abc",
        generation=1,
        planning_attempts=(prepared, submitting, running),
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-token",
        controller_job=predecessor_job,
    )
    successor = predecessor.resume_after_reconciliation(
        predecessor_job=predecessor_job,
        scheduler_state=JobStatus.PREEMPTED.value,
    )

    plan = plan_attempt_adoption(successor)
    assert [target.attempt_id for target in plan.pending] == [
        "plan-prepared",
        "plan-submitting",
        "plan-running",
    ]
    assert not plan.new_work_allowed
    with pytest.raises(RecoveryError, match="all old-generation attempts"):
        plan.require_new_work_allowed()

    running_target = next(target for target in plan.pending if target.attempt_id == "plan-running")
    with pytest.raises(StateConflictError, match="exact persisted identity"):
        successor.record_attempt_adoption(
            attempt_id=running_target.attempt_id,
            attempt_generation=running_target.attempt_generation,
            status=running_target.status,
            submission_token=running_target.submission_token,
            job=JobReference("999"),
        )

    current = successor
    for target in plan.pending:
        current = current.record_attempt_adoption(
            attempt_id=target.attempt_id,
            attempt_generation=target.attempt_generation,
            status=target.status,
            submission_token=target.submission_token,
            job=target.job,
        )
    replayed = current.record_attempt_adoption(
        attempt_id=running_target.attempt_id,
        attempt_generation=running_target.attempt_generation,
        status=running_target.status,
        submission_token=running_target.submission_token,
        job=running_target.job,
    )
    assert replayed is current
    complete = plan_attempt_adoption(current)
    assert complete.pending == ()
    assert complete.new_work_allowed
    complete.require_new_work_allowed()


def test_generation_credentials_are_fenced_only_after_exact_adoption_receipts(
    tmp_path: Path,
) -> None:
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
            job=JobReference(str(500 + sequence)),
            resource_class="plan_drafter",
        )
        for sequence, attempt_id in enumerate(("adopted", "fenced"), start=1)
    )
    predecessor = RunState(
        run_id="run-credential-fence",
        task_digest="a" * 64,
        base_commit="abc",
        generation=1,
        planning_attempts=attempts,
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-token",
        controller_job=JobReference("400"),
    )
    successor = predecessor.resume_after_reconciliation(
        predecessor_job=JobReference("400"),
        scheduler_state=JobStatus.PREEMPTED.value,
    )
    provisions = tuple(
        prepare_credential_provision(
            workspace=tmp_path.resolve(),
            binding=CredentialBinding(
                run_id=successor.run_id,
                item_id="planning",
                attempt_id=attempt.attempt_id,
                task_digest=successor.task_digest,
                generation=attempt.generation,
                backend_kind="codex",
            ),
            ambient_environment={"OPENAI_API_KEY": "test-secret"},
        )
        for attempt in attempts
    )
    calls: list[CredentialBinding] = []

    class Broker:
        def revoke(
            self,
            *,
            workspace: Path,
            expected_binding: CredentialBinding,
            descriptor: CredentialDescriptor,
            cause: CredentialRevocationCause,
        ) -> CredentialRevocationReceipt:
            calls.append(expected_binding)
            return revoke_terminal_attempt_credentials(
                workspace=workspace,
                expected_binding=expected_binding,
                provision=locate_credential_provision(
                    workspace=workspace,
                    descriptor=descriptor,
                ),
                cause=cause,
            )

    with pytest.raises(RecoveryError, match="all old-generation attempts"):
        revoke_reconciled_generation_credentials(
            successor,
            broker=Broker(),  # type: ignore[arg-type]
            workspace=tmp_path.resolve(),
            descriptors=tuple(provision.descriptor for provision in provisions),
            adopted_bindings=(provisions[0].descriptor.binding,),
        )
    assert calls == []

    reconciled = successor
    for attempt in attempts:
        reconciled = reconciled.record_attempt_adoption(
            attempt_id=attempt.attempt_id,
            attempt_generation=attempt.generation,
            status=attempt.status,
            submission_token=attempt.submission_token,
            job=attempt.job,
        )
    receipts = revoke_reconciled_generation_credentials(
        reconciled,
        broker=Broker(),  # type: ignore[arg-type]
        workspace=tmp_path.resolve(),
        descriptors=tuple(provision.descriptor for provision in provisions),
        adopted_bindings=(provisions[0].descriptor.binding,),
    )

    assert calls == [
        provisions[0].descriptor.binding,
        provisions[1].descriptor.binding,
    ]
    assert len(receipts) == 2
    assert all(receipt.receipt_path.is_file() for receipt in receipts)
    assert provisions[0].bundle_path is not None and not provisions[0].bundle_path.exists()
    assert provisions[1].bundle_path is not None and not provisions[1].bundle_path.exists()


def test_orphan_grace_waits_then_cleans_only_exact_active_children() -> None:
    observations = (
        _observation("9", JobStatus.COMPLETED),
        _observation("7", JobStatus.RUNNING),
        _observation("8", JobStatus.UNKNOWN),
    )

    waiting = plan_orphan_recovery(
        observations,
        successor_present=False,
        elapsed_seconds=20,
        grace_seconds=60,
    )
    cleanup = plan_orphan_recovery(
        observations,
        successor_present=False,
        elapsed_seconds=60,
        grace_seconds=60,
    )

    assert waiting.action is OrphanAction.WAIT_FOR_SUCCESSOR
    assert waiting.grace_remaining_seconds == 40
    assert not waiting.cancel
    assert cleanup.action is OrphanAction.CLEAN_UP
    assert tuple(job.scheduler_id for job in cleanup.cancel) == ("7", "8")
    assert tuple(job.scheduler_id for job in cleanup.terminal) == ("9",)


def test_successor_adopts_nonterminal_children_without_cleanup() -> None:
    plan = plan_orphan_recovery(
        (_observation("7", JobStatus.RUNNING),),
        successor_present=True,
        elapsed_seconds=100,
        grace_seconds=60,
    )

    assert plan.action is OrphanAction.ADOPT
    assert tuple(job.scheduler_id for job in plan.adopt) == ("7",)
    assert not plan.cancel


def test_orphan_plan_rejects_duplicate_exact_identity() -> None:
    with pytest.raises(ValueError, match="duplicate exact job identities"):
        plan_orphan_recovery(
            (
                _observation("7", JobStatus.RUNNING),
                _observation("7", JobStatus.UNKNOWN),
            ),
            successor_present=False,
            elapsed_seconds=0,
            grace_seconds=60,
        )


def test_human_input_is_detached_and_never_requeues_workers() -> None:
    intent = human_input_recovery_intent(
        "approval-1",
        "a" * 64,
        controller_requeue_enabled=True,
    )

    assert intent.status is HumanInputStatus.WAITING_FOR_INPUT
    assert intent.checkpoint_required
    assert intent.release_controller
    assert intent.controller_requeue
    assert not intent.read_stdin
    assert not intent.worker_requeue


def test_human_input_requires_stable_prompt_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        human_input_recovery_intent(
            "approval-1",
            "not-a-digest",
            controller_requeue_enabled=False,
        )
