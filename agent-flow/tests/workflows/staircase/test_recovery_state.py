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

"""Focused persistence tests for Staircase controller recovery state."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import state

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
CONTROLLER_JOB = state.JobReference("100", cluster="test")
CHILD_JOB = state.JobReference("101", cluster="test")


def _run() -> state.RunState:
    attempt = state.AttemptRecord(
        attempt_id="gate-1",
        item_id="item-1",
        sequence=1,
        role=state.Role.GATE,
        kind=state.AttemptKind.DETERMINISTIC_GATE,
        profile=state.DomainProfile.ASSEMBLER,
        generation=1,
        status=state.AttemptStatus.RUNNING,
        submission_token="submit-gate-1",
        job=CHILD_JOB,
    )
    item = state.WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.GATE,
        profile=state.DomainProfile.ASSEMBLER,
        status=state.WorkItemStatus.CODING,
        attempts=(attempt,),
    )
    return state.RunState(
        run_id="run-1",
        task_digest=DIGEST_A,
        base_commit="0123456789abcdef",
        generation=1,
        stages=(state.StageRecord("stage-1", ("goal-1",)),),
        goals=(state.GoalRecord("goal-1", "stage-1", ("item-1",)),),
        items=(item,),
        controller_bootstrap_status=state.ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-submit-token",
        controller_job=CONTROLLER_JOB,
    )


def test_signal_checkpoint_is_append_only_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    checkpointed = original.checkpoint_controller(
        reason="Slurm advance signal",
        requeue_requested=True,
        signal=state.ControllerCheckpointSignal.ADVANCE,
    )
    state.save_state(path, checkpointed, expected_revision=0, expected_generation=1)

    assert state.load_state(path) == checkpointed
    assert checkpointed.controller_lifecycle is state.ControllerLifecycle.CHECKPOINTED
    assert checkpointed.controller_checkpoints[0].signal is state.ControllerCheckpointSignal.ADVANCE
    with pytest.raises(state.InvalidTransitionError, match="only a running controller"):
        checkpointed.checkpoint_controller(
            reason="duplicate signal",
            requeue_requested=True,
        )

    rewritten_checkpoint = replace(
        checkpointed.controller_checkpoints[0], reason="rewritten checkpoint"
    )
    rewritten = replace(
        checkpointed,
        revision=checkpointed.revision + 1,
        controller_checkpoints=(rewritten_checkpoint,),
    )
    with pytest.raises(state.StateConflictError, match="append-only"):
        state.save_state(path, rewritten, expected_revision=1, expected_generation=1)


def test_human_request_response_and_generation_resume_are_durable(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    waiting = original.wait_for_human_input(
        request_id="approval-1",
        prompt_digest=DIGEST_B,
        controller_requeue=True,
        reason="operator approval required",
    )
    state.save_state(path, waiting, expected_revision=0, expected_generation=1)
    assert waiting.controller_lifecycle is state.ControllerLifecycle.WAITING_FOR_INPUT
    assert waiting.human_requests[0].response_digest is None

    with pytest.raises(state.InvalidTransitionError, match="before human input"):
        waiting.resume_after_reconciliation(
            predecessor_job=CONTROLLER_JOB,
            scheduler_state="PREEMPTED",
        )

    answered = waiting.record_human_response(request_id="approval-1", response_digest=DIGEST_C)
    state.save_state(path, answered, expected_revision=1, expected_generation=1)
    recovered = answered.resume_after_reconciliation(
        predecessor_job=CONTROLLER_JOB,
        scheduler_state="PREEMPTED",
    )
    state.advance_generation(path, recovered, expected_revision=2, expected_generation=1)

    loaded = state.load_state(path)
    assert loaded.controller_lifecycle is state.ControllerLifecycle.RUNNING
    assert loaded.generation == 2
    assert loaded.human_requests[0].response_digest == DIGEST_C
    assert loaded.predecessor_reconciliations == (
        state.PredecessorReconciliationReceipt(1, 2, CONTROLLER_JOB, "PREEMPTED"),
    )


def test_human_request_identity_prompt_response_and_requeue_are_immutable(
    tmp_path: Path,
) -> None:
    path = tmp_path / state.STATE_FILENAME
    waiting = _run().wait_for_human_input(
        request_id="approval-1",
        prompt_digest=DIGEST_A,
        controller_requeue=False,
        reason="approval",
    )
    state.initialize_state(path, replace(waiting, revision=0))
    answered = waiting.record_human_response(request_id="approval-1", response_digest=DIGEST_B)
    answered = replace(answered, revision=1)
    state.save_state(path, answered, expected_revision=0, expected_generation=1)

    rewritten_request = replace(answered.human_requests[0], response_digest=DIGEST_C)
    rewritten = replace(
        answered,
        revision=2,
        human_requests=(rewritten_request,),
    )
    with pytest.raises(state.StateConflictError, match="response digest is immutable"):
        state.save_state(path, rewritten, expected_revision=1, expected_generation=1)

    rewritten_prompt = replace(answered.human_requests[0], prompt_digest=DIGEST_C)
    rewritten = replace(answered, revision=2, human_requests=(rewritten_prompt,))
    with pytest.raises(state.StateConflictError, match="prompt.*immutable"):
        state.save_state(path, rewritten, expected_revision=1, expected_generation=1)


def test_generation_advance_requires_exact_job_and_receipt() -> None:
    original = _run().checkpoint_controller(
        reason="preemption",
        requeue_requested=True,
        signal=state.ControllerCheckpointSignal.PREEMPTION,
    )
    with pytest.raises(state.InvalidTransitionError, match="exact controller job"):
        original.resume_after_reconciliation(
            predecessor_job=state.JobReference("999", cluster="test"),
            scheduler_state="PREEMPTED",
        )

    with pytest.raises(ValueError, match="cover every advanced generation"):
        replace(_run(), generation=2)


def test_orphan_grace_and_exact_action_receipts_are_append_only(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    grace = original.start_orphan_grace(
        started_at="2026-09-16T12:00:00Z",
        deadline_at="2026-09-16T12:05:00Z",
    )
    state.save_state(path, grace, expected_revision=0, expected_generation=1)
    adopted = grace.record_orphan_action(
        action=state.OrphanReceiptAction.ADOPTED,
        job=CHILD_JOB,
        scheduler_state="RUNNING",
    )
    state.save_state(path, adopted, expected_revision=1, expected_generation=1)
    assert state.load_state(path).orphan_action_receipts[0].job == CHILD_JOB

    with pytest.raises(ValueError, match="exact owned child job"):
        grace.record_orphan_action(
            action=state.OrphanReceiptAction.CLEANED_UP,
            job=state.JobReference("999", cluster="test"),
            scheduler_state="CANCELLED",
        )
    with pytest.raises(ValueError, match="only one action receipt"):
        adopted.record_orphan_action(
            action=state.OrphanReceiptAction.CLEANED_UP,
            job=CHILD_JOB,
            scheduler_state="CANCELLED",
        )
    with pytest.raises(ValueError, match="later than"):
        state.OrphanGraceWindow(
            generation=1,
            started_at="2026-09-16T12:05:00Z",
            deadline_at="2026-09-16T12:00:00Z",
        )


def test_nested_recovery_json_is_strict(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    checkpointed = _run().checkpoint_controller(
        reason="advance",
        requeue_requested=True,
        signal=state.ControllerCheckpointSignal.ADVANCE,
    )
    state.initialize_state(path, replace(checkpointed, revision=0))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["controller_checkpoints"][0]["surprise"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown controller checkpoint fields"):
        state.load_state(path)
