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

"""Tests for Staircase authoritative workflow state."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import state

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _validated_coder() -> state.AttemptRecord:
    return state.AttemptRecord(
        attempt_id="coder-1",
        item_id="item-1",
        sequence=1,
        role=state.Role.CODER,
        kind=state.AttemptKind.ROLE,
        profile=state.DomainProfile.SMITH,
        generation=1,
        status=state.AttemptStatus.VALIDATED,
        submission_token="submit-coder-1",
        job=state.JobReference(job_id="101"),
        result_digest=DIGEST_A,
        candidate_digest=DIGEST_B,
    )


def _validated_reviewer() -> state.AttemptRecord:
    return state.AttemptRecord(
        attempt_id="reviewer-2",
        item_id="item-1",
        sequence=2,
        role=state.Role.REVIEWER,
        kind=state.AttemptKind.REVIEWER_RERUN,
        profile=state.DomainProfile.SMITH,
        generation=1,
        status=state.AttemptStatus.VALIDATED,
        submission_token="submit-reviewer-2",
        job=state.JobReference(job_id="102"),
        result_digest=DIGEST_C,
        review_of_attempt_id="coder-1",
        reviewed_candidate_digest=DIGEST_B,
    )


def _integrated_item() -> state.WorkItemRecord:
    return state.WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-attention",
        kind=state.WorkItemKind.CATALOG_ONBOARD,
        profile=state.DomainProfile.SMITH,
        status=state.WorkItemStatus.INTEGRATED,
        attempts=(_validated_coder(), _validated_reviewer()),
        candidate_attempt_id="coder-1",
        candidate_digest=DIGEST_B,
        reviewer_attempt_id="reviewer-2",
    )


def _run(*, revision: int = 0, generation: int = 1) -> state.RunState:
    return state.RunState(
        run_id="run-1",
        task_digest=DIGEST_A,
        base_commit="0123456789abcdef",
        generation=generation,
        revision=revision,
        stages=(
            state.StageRecord(
                stage_id="stage-1",
                required_goal_ids=("goal-attention",),
                status=state.StageStatus.CLOSED,
            ),
        ),
        goals=(
            state.GoalRecord(
                goal_id="goal-attention",
                stage_id="stage-1",
                required_item_ids=("item-1",),
                status=state.HierarchyStatus.SUCCEEDED,
            ),
        ),
        items=(_integrated_item(),),
        controller_bootstrap_status=state.ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-submit-token",
        controller_job=state.JobReference(job_id="100", cluster="test"),
    )


def test_work_item_and_attempt_lifecycles_are_orthogonal() -> None:
    attempt = state.AttemptRecord(
        attempt_id="attempt-1",
        item_id="item-1",
        sequence=1,
        role=state.Role.CODER,
        kind=state.AttemptKind.ROLE,
        profile=state.DomainProfile.SMITH,
        generation=1,
    )
    submitting = attempt.transition(
        state.AttemptStatus.SUBMITTING,
        submission_token="token-1",
    )
    submitted = submitting.transition(
        state.AttemptStatus.SUBMITTED,
        job=state.JobReference(job_id="22"),
    )
    running = submitted.transition(state.AttemptStatus.RUNNING)
    assert running.status is state.AttemptStatus.RUNNING

    item = state.WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.CATALOG_VERIFY,
        profile=state.DomainProfile.SMITH,
    )
    assert item.transition(state.WorkItemStatus.READY).status is state.WorkItemStatus.READY
    assert item.status is state.WorkItemStatus.PLANNED


def test_illegal_transitions_and_terminal_rewrites_fail() -> None:
    attempt = _validated_coder()
    with pytest.raises(state.InvalidTransitionError, match="illegal attempt transition"):
        attempt.transition(state.AttemptStatus.RUNNING)

    item = _integrated_item()
    with pytest.raises(state.InvalidTransitionError, match="illegal work item transition"):
        item.transition(state.WorkItemStatus.READY)


def test_candidate_and_reviewer_linkage_is_frozen() -> None:
    with pytest.raises(ValueError, match="frozen candidate digest"):
        state.WorkItemRecord(
            item_id="item-1",
            stage_id="stage-1",
            goal_id="goal-1",
            kind=state.WorkItemKind.CATALOG_ONBOARD,
            profile=state.DomainProfile.SMITH,
            status=state.WorkItemStatus.APPROVED,
            attempts=(_validated_coder(), _validated_reviewer()),
            candidate_attempt_id="coder-1",
            candidate_digest=DIGEST_C,
            reviewer_attempt_id="reviewer-2",
        )

    wrong_reviewer = replace(_validated_reviewer(), reviewed_candidate_digest=DIGEST_C)
    with pytest.raises(ValueError, match="did not inspect"):
        state.WorkItemRecord(
            item_id="item-1",
            stage_id="stage-1",
            goal_id="goal-1",
            kind=state.WorkItemKind.CATALOG_ONBOARD,
            profile=state.DomainProfile.SMITH,
            status=state.WorkItemStatus.APPROVED,
            attempts=(_validated_coder(), wrong_reviewer),
            candidate_attempt_id="coder-1",
            candidate_digest=DIGEST_B,
            reviewer_attempt_id="reviewer-2",
        )


def test_dependencies_must_be_integrated_before_ready() -> None:
    dependency = state.WorkItemRecord(
        item_id="dependency",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.SEARCH,
        profile=state.DomainProfile.SMITH,
    )
    dependent = state.WorkItemRecord(
        item_id="dependent",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.ASSEMBLE_CORE,
        profile=state.DomainProfile.ASSEMBLER,
        status=state.WorkItemStatus.READY,
        dependencies=("dependency",),
    )
    with pytest.raises(ValueError, match="before all dependencies"):
        state.RunState(
            run_id="run-1",
            task_digest=DIGEST_A,
            base_commit="abc",
            generation=1,
            stages=(state.StageRecord(stage_id="stage-1", required_goal_ids=("goal-1",)),),
            goals=(
                state.GoalRecord(
                    goal_id="goal-1",
                    stage_id="stage-1",
                    required_item_ids=("dependency", "dependent"),
                ),
            ),
            items=(dependency, dependent),
        )


def test_dependency_cycles_are_rejected() -> None:
    first = state.WorkItemRecord(
        item_id="first",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.SEARCH,
        profile=state.DomainProfile.SMITH,
        dependencies=("second",),
    )
    second = state.WorkItemRecord(
        item_id="second",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.SEARCH,
        profile=state.DomainProfile.SMITH,
        dependencies=("first",),
    )
    with pytest.raises(ValueError, match="dependencies contain a cycle"):
        state.RunState(
            run_id="run-cycle",
            task_digest=DIGEST_A,
            base_commit="abc",
            generation=1,
            stages=(state.StageRecord(stage_id="stage-1", required_goal_ids=("goal-1",)),),
            goals=(
                state.GoalRecord(
                    goal_id="goal-1",
                    stage_id="stage-1",
                    required_item_ids=("first", "second"),
                ),
            ),
            items=(first, second),
        )


def test_atomic_state_round_trip_and_unknown_field_rejection(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    assert state.load_state(path) == original

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["surprise"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown run state fields"):
        state.load_state(path)


def test_revision_and_generation_fencing(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    next_state = replace(original, revision=1)
    state.save_state(path, next_state, expected_revision=0, expected_generation=1)

    with pytest.raises(state.StateConflictError, match="revision changed"):
        state.save_state(path, next_state, expected_revision=0, expected_generation=1)

    wrong_generation = next_state.resume_after_reconciliation(
        predecessor_job=state.JobReference(job_id="100", cluster="test"),
        scheduler_state="PREEMPTED",
    )
    with pytest.raises(state.StateConflictError, match="generation fence"):
        state.save_state(path, wrong_generation, expected_revision=1, expected_generation=1)


def test_generation_advance_is_separate_cas(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    recovered = original.resume_after_reconciliation(
        predecessor_job=state.JobReference(job_id="100", cluster="test"),
        scheduler_state="PREEMPTED",
    )
    state.advance_generation(path, recovered, expected_revision=0, expected_generation=1)
    assert state.load_state(path).generation == 2

    with pytest.raises(state.StateConflictError, match="changed"):
        state.advance_generation(path, recovered, expected_revision=0, expected_generation=1)


def test_persistence_rejects_historical_attempt_rewrite(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    item = original.items[0]
    rewritten_coder = replace(item.attempts[0], job=state.JobReference(job_id="999"))
    rewritten_item = replace(item, attempts=(rewritten_coder, item.attempts[1]))
    rewritten = replace(original, revision=1, items=(rewritten_item,))

    with pytest.raises(state.StateConflictError, match="job is immutable"):
        state.save_state(path, rewritten, expected_revision=0, expected_generation=1)


def test_attempt_lineage_requires_terminal_unique_logical_predecessor() -> None:
    predecessor = state.AttemptRecord(
        attempt_id="attempt-1",
        item_id="lineage-item",
        sequence=1,
        role=state.Role.CODER,
        kind=state.AttemptKind.ROLE,
        profile=state.DomainProfile.SMITH,
        generation=1,
        status=state.AttemptStatus.PREEMPTED,
        submission_token="token-1",
        job=state.JobReference("201"),
        terminal_reason="PREEMPTED",
        resource_class="smith_agent",
    )
    child = state.AttemptRecord(
        attempt_id="attempt-2",
        item_id="lineage-item",
        sequence=2,
        role=state.Role.CODER,
        kind=state.AttemptKind.ROLE,
        profile=state.DomainProfile.SMITH,
        generation=2,
        resource_class="smith_agent",
        predecessor_attempt_id="attempt-1",
    )
    base = dict(
        item_id="lineage-item",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.CATALOG_ONBOARD,
        profile=state.DomainProfile.SMITH,
    )
    assert state.WorkItemRecord(**base, attempts=(predecessor, child)).attempts[-1] == child

    nonterminal = replace(
        predecessor,
        status=state.AttemptStatus.RUNNING,
        terminal_reason=None,
    )
    with pytest.raises(ValueError, match="predecessor must be terminal"):
        state.WorkItemRecord(**base, attempts=(nonterminal, child))

    fork = replace(child, attempt_id="attempt-3", sequence=3)
    with pytest.raises(ValueError, match="only one child"):
        state.WorkItemRecord(**base, attempts=(predecessor, child, fork))

    changed = replace(child, role=state.Role.QA)
    with pytest.raises(ValueError, match="preserve immutable logical fields"):
        state.WorkItemRecord(**base, attempts=(predecessor, changed))


def test_new_attempt_append_requires_current_generation_and_resource(
    tmp_path: Path,
) -> None:
    recovered = replace(
        _run().resume_after_reconciliation(
            predecessor_job=state.JobReference(job_id="100", cluster="test"),
            scheduler_state="PREEMPTED",
        ),
        revision=0,
    )
    path = tmp_path / state.STATE_FILENAME
    state.initialize_state(path, recovered)
    stale = state.AttemptRecord(
        attempt_id="plan-stale",
        item_id=state.PLANNING_ITEM_ID,
        sequence=1,
        role=state.Role.PLAN_DRAFTER,
        kind=state.AttemptKind.ROLE,
        generation=1,
        resource_class="plan_drafter",
    )
    with pytest.raises(state.StateConflictError, match="active controller generation"):
        state.save_state(
            path,
            replace(recovered, revision=1, planning_attempts=(stale,)),
            expected_revision=0,
            expected_generation=2,
        )

    missing_resource = replace(stale, generation=2, resource_class=None)
    with pytest.raises(state.StateConflictError, match="selected resource_class"):
        state.save_state(
            path,
            replace(recovered, revision=1, planning_attempts=(missing_resource,)),
            expected_revision=0,
            expected_generation=2,
        )


def test_success_requires_every_item_integrated() -> None:
    cancelled = state.WorkItemRecord(
        item_id="cancelled",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.GATE,
        profile=state.DomainProfile.ASSEMBLER,
        status=state.WorkItemStatus.CANCELLED,
        terminal_reason="operator cancelled",
    )
    with pytest.raises(ValueError, match="successful run"):
        state.RunState(
            run_id="run-1",
            task_digest=DIGEST_A,
            base_commit="abc",
            generation=1,
            stages=(
                state.StageRecord(
                    stage_id="stage-1",
                    required_goal_ids=("goal-1",),
                    status=state.StageStatus.BLOCKED,
                    terminal_reason="item cancelled",
                ),
            ),
            goals=(
                state.GoalRecord(
                    goal_id="goal-1",
                    stage_id="stage-1",
                    required_item_ids=("cancelled",),
                    status=state.HierarchyStatus.BLOCKED,
                    terminal_reason="item cancelled",
                ),
            ),
            items=(cancelled,),
            terminal_status=state.RunTerminalStatus.SUCCEEDED,
            terminal_reason="done",
        )


def test_goal_and_stage_cannot_close_before_required_children() -> None:
    planned = state.WorkItemRecord(
        item_id="planned",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=state.WorkItemKind.SEARCH,
        profile=state.DomainProfile.SMITH,
    )
    with pytest.raises(ValueError, match="succeeded before required items"):
        state.RunState(
            run_id="run-1",
            task_digest=DIGEST_A,
            base_commit="abc",
            generation=1,
            stages=(
                state.StageRecord(
                    stage_id="stage-1",
                    required_goal_ids=("goal-1",),
                ),
            ),
            goals=(
                state.GoalRecord(
                    goal_id="goal-1",
                    stage_id="stage-1",
                    required_item_ids=("planned",),
                    status=state.HierarchyStatus.SUCCEEDED,
                ),
            ),
            items=(planned,),
        )


def test_stage_qa_lifecycle_has_no_success_shortcut() -> None:
    planned = state.StageRecord(stage_id="stage-1", required_goal_ids=("goal-1",))
    active = planned.transition(state.StageStatus.ACTIVE)
    with pytest.raises(state.InvalidTransitionError, match="illegal Stage transition"):
        active.transition(state.StageStatus.CLOSED)

    ready = active.transition(state.StageStatus.READY_FOR_QA)
    failed = ready.transition(state.StageStatus.QA_FAILED)
    retrying = failed.transition(state.StageStatus.ACTIVE)
    passed = retrying.transition(state.StageStatus.READY_FOR_QA).transition(
        state.StageStatus.QA_PASSED
    )
    assert passed.transition(state.StageStatus.CLOSED).terminal


def test_stage_qa_status_round_trip(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    failed = replace(
        _run(),
        stages=(
            state.StageRecord(
                stage_id="stage-1",
                required_goal_ids=("goal-attention",),
                status=state.StageStatus.QA_FAILED,
            ),
        ),
    )
    state.initialize_state(path, failed)
    assert state.load_state(path).stages[0].status is state.StageStatus.QA_FAILED


def test_schema_v1_stage_migration_is_explicit_and_evidence_safe(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    succeeded = replace(
        _run(),
        terminal_status=state.RunTerminalStatus.SUCCEEDED,
        terminal_reason="all gates passed",
    )
    state.initialize_state(path, succeeded)
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["schema_version"] = state.LEGACY_SCHEMA_VERSION
    legacy["stages"][0]["status"] = state.HierarchyStatus.SUCCEEDED.value

    with pytest.raises(state.StateMigrationError, match="lacks immutable QA evidence"):
        state.migrate_state_v1(legacy)
    assert legacy["schema_version"] == state.LEGACY_SCHEMA_VERSION
    assert legacy["stages"][0]["status"] == state.HierarchyStatus.SUCCEEDED.value

    migrated = state.migrate_state_v1(
        legacy,
        succeeded_stage_policy=state.LegacySucceededStagePolicy.BLOCK,
    )
    assert migrated["schema_version"] == state.SCHEMA_VERSION
    assert migrated["stages"][0]["status"] == state.StageStatus.BLOCKED.value
    assert migrated["terminal_status"] == state.RunTerminalStatus.BLOCKED.value
    path.write_text(json.dumps(migrated), encoding="utf-8")
    loaded = state.load_state(path)
    assert loaded.stages[0].status is state.StageStatus.BLOCKED
    assert loaded.terminal_status is state.RunTerminalStatus.BLOCKED


def test_load_rejects_implicit_schema_v1_migration(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    state.initialize_state(path, _run())
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["schema_version"] = state.LEGACY_SCHEMA_VERSION
    path.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(state.StateMigrationError, match="explicit migrate_state_v1"):
        state.load_state(path)


def test_schema_v1_open_stage_migrates_without_status_aliasing(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    active = replace(
        _run(),
        stages=(
            state.StageRecord(
                stage_id="stage-1",
                required_goal_ids=("goal-attention",),
                status=state.StageStatus.ACTIVE,
            ),
        ),
    )
    state.initialize_state(path, active)
    legacy = json.loads(path.read_text(encoding="utf-8"))
    legacy["schema_version"] = state.LEGACY_SCHEMA_VERSION

    migrated = state.migrate_state_v1(legacy)
    path.write_text(json.dumps(migrated), encoding="utf-8")
    assert state.load_state(path).stages[0].status is state.StageStatus.ACTIVE


def test_waiting_for_input_is_recoverable_not_terminal_blocked_input() -> None:
    waiting = _run().wait_for_human_input(
        request_id="request-1",
        prompt_digest=DIGEST_B,
        controller_requeue=False,
        reason="need checkpoint metadata",
    )
    assert waiting.controller_lifecycle is state.ControllerLifecycle.WAITING_FOR_INPUT
    assert waiting.terminal_status is state.RunTerminalStatus.ACTIVE
    with pytest.raises(ValueError, match="WAITING_FOR_INPUT is recoverable"):
        replace(
            waiting,
            terminal_status=state.RunTerminalStatus.BLOCKED_INPUT,
            terminal_reason="input will not be provided",
        )

    blocked = _run().finish(
        state.RunTerminalStatus.BLOCKED_INPUT,
        "required input is permanently unavailable",
    )
    assert blocked.terminal_status is state.RunTerminalStatus.BLOCKED_INPUT
    assert blocked.controller_lifecycle is state.ControllerLifecycle.RUNNING


def test_planning_attempts_have_typed_run_level_ownership() -> None:
    planning_attempt = state.AttemptRecord(
        attempt_id="plan-1",
        item_id=state.PLANNING_ITEM_ID,
        sequence=1,
        role=state.Role.PLAN_DRAFTER,
        kind=state.AttemptKind.ROLE,
        generation=1,
    )
    run = replace(_run(), planning_attempts=(planning_attempt,))
    assert run.planning_attempts[0].role is state.Role.PLAN_DRAFTER

    with pytest.raises(ValueError, match="PlanDrafter or PlanReviewer"):
        replace(
            _run(),
            planning_attempts=(replace(planning_attempt, role=state.Role.CODER),),
        )


def test_controller_submission_token_supports_crash_adoption(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    submitting = state.RunState(
        run_id="run-bootstrap",
        task_digest=DIGEST_A,
        base_commit="abc",
        generation=1,
        controller_bootstrap_status=state.ControllerBootstrapStatus.SUBMITTING,
        controller_submission_token="staircase-run-bootstrap-generation-1",
    )
    state.initialize_state(path, submitting)
    reloaded = state.load_state(path)
    assert reloaded.controller_job is None
    assert reloaded.controller_submission_token == "staircase-run-bootstrap-generation-1"

    adopted = reloaded.record_controller_submitted(state.JobReference(job_id="404"))
    state.save_state(path, adopted, expected_revision=0, expected_generation=1)
    assert state.load_state(path).controller_job == state.JobReference(job_id="404")

    with pytest.raises(state.StateConflictError, match="submission token is immutable"):
        state.save_state(
            path,
            replace(adopted, revision=2, controller_submission_token="replacement-token"),
            expected_revision=1,
            expected_generation=1,
        )


def test_controller_lease_requires_reconciliation_not_elapsed_time(tmp_path: Path) -> None:
    path = tmp_path / state.LEASE_FILENAME
    lease = state.ControllerLease.acquire(
        path,
        run_id="run-1",
        generation=1,
        controller_job=state.JobReference(job_id="100"),
        owner_nonce="owner-1",
    )
    with pytest.raises(state.LeaseConflictError, match="lock is held"):
        state.ControllerLease.acquire(
            path,
            run_id="run-1",
            generation=2,
            controller_job=state.JobReference(job_id="200"),
        )
    first_heartbeat = lease.record.heartbeat_at
    assert lease.heartbeat().heartbeat_at >= first_heartbeat
    lease.release(remove_record=False)

    with pytest.raises(state.LeaseConflictError, match="already exists"):
        state.ControllerLease.acquire(
            path,
            run_id="run-1",
            generation=2,
            controller_job=state.JobReference(job_id="200"),
        )
    with pytest.raises(state.LeaseConflictError, match="scheduler reconciliation"):
        state.ControllerLease.recover(
            path,
            run_id="run-1",
            generation=2,
            controller_job=state.JobReference(job_id="200"),
            scheduler_reconciled=False,
        )

    recovered = state.ControllerLease.recover(
        path,
        run_id="run-1",
        generation=2,
        controller_job=state.JobReference(job_id="200"),
        scheduler_reconciled=True,
        owner_nonce="owner-2",
    )
    assert state.load_lease_record(path).owner_nonce == "owner-2"
    recovered.release()
    assert not path.exists()
