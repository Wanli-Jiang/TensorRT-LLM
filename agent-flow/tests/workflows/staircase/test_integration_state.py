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

"""Focused tests for durable integration history and pre-integration QA outcomes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import state

BASE_COMMIT = "a" * 40
FIRST_COMMIT = "b" * 40
SECOND_COMMIT = "c" * 40
CANDIDATE_DIGEST = "d" * 64
VERIFY_DIGEST = "e" * 64
TASK_DIGEST = "f" * 64


def _approved_item(item_id: str, kind: state.WorkItemKind, digest: str) -> state.WorkItemRecord:
    coder_id = f"{item_id}-coder"
    reviewer_id = f"{item_id}-reviewer"
    coder = state.AttemptRecord(
        attempt_id=coder_id,
        item_id=item_id,
        sequence=1,
        role=state.Role.CODER,
        kind=state.AttemptKind.ROLE,
        generation=1,
        profile=state.DomainProfile.SMITH,
        status=state.AttemptStatus.VALIDATED,
        submission_token=f"submit-{coder_id}",
        job=state.JobReference("101" if kind is state.WorkItemKind.CATALOG_ONBOARD else "201"),
        result_digest="1" * 64,
        candidate_digest=digest,
    )
    reviewer = state.AttemptRecord(
        attempt_id=reviewer_id,
        item_id=item_id,
        sequence=2,
        role=state.Role.REVIEWER,
        kind=state.AttemptKind.REVIEWER_RERUN,
        generation=1,
        profile=state.DomainProfile.SMITH,
        status=state.AttemptStatus.VALIDATED,
        submission_token=f"submit-{reviewer_id}",
        job=state.JobReference("102" if kind is state.WorkItemKind.CATALOG_ONBOARD else "202"),
        result_digest="2" * 64,
        candidate_digest=digest,
        review_of_attempt_id=coder_id,
        reviewed_candidate_digest=digest,
    )
    return state.WorkItemRecord(
        item_id=item_id,
        stage_id="stage-1",
        goal_id="goal-1",
        kind=kind,
        profile=state.DomainProfile.SMITH,
        status=state.WorkItemStatus.APPROVED,
        attempts=(coder, reviewer),
        candidate_attempt_id=coder_id,
        candidate_digest=digest,
        reviewer_attempt_id=reviewer_id,
    )


def _run() -> state.RunState:
    items = (
        _approved_item("catalog-onboard", state.WorkItemKind.CATALOG_ONBOARD, CANDIDATE_DIGEST),
        _approved_item("catalog-verify", state.WorkItemKind.CATALOG_VERIFY, "9" * 64),
    )
    return state.RunState(
        run_id="integration-run",
        task_digest=TASK_DIGEST,
        base_commit=BASE_COMMIT,
        generation=1,
        stages=(state.StageRecord("stage-1", ("goal-1",)),),
        goals=(state.GoalRecord("goal-1", "stage-1", tuple(item.item_id for item in items)),),
        items=items,
    )


def test_integration_history_round_trip_and_no_change_verification(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    integrated = original.record_integration(
        item_id="catalog-onboard",
        previous_commit=BASE_COMMIT,
        new_commit=FIRST_COMMIT,
        evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
        evidence_digest=CANDIDATE_DIGEST,
    )
    state.save_state(path, integrated, expected_revision=0, expected_generation=1)
    verified = integrated.record_integration(
        item_id="catalog-verify",
        previous_commit=FIRST_COMMIT,
        new_commit=FIRST_COMMIT,
        evidence_kind=state.IntegrationEvidenceKind.VERIFICATION,
        evidence_digest=VERIFY_DIGEST,
    )
    state.save_state(path, verified, expected_revision=1, expected_generation=1)

    loaded = state.load_state(path)
    assert loaded == verified
    assert loaded.base_commit == BASE_COMMIT
    assert loaded.integration_head == FIRST_COMMIT
    assert [record.item_id for record in loaded.integration_history] == [
        "catalog-onboard",
        "catalog-verify",
    ]


def test_integration_history_rejects_invalid_chain_duplicate_and_fake_commit() -> None:
    original = _run()
    with pytest.raises(state.InvalidTransitionError, match="previous_commit is stale"):
        original.record_integration(
            item_id="catalog-onboard",
            previous_commit=SECOND_COMMIT,
            new_commit=FIRST_COMMIT,
            evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
            evidence_digest=CANDIDATE_DIGEST,
        )

    integrated = original.record_integration(
        item_id="catalog-onboard",
        previous_commit=BASE_COMMIT,
        new_commit=FIRST_COMMIT,
        evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
        evidence_digest=CANDIDATE_DIGEST,
    )
    with pytest.raises(state.InvalidTransitionError, match="already has"):
        integrated.record_integration(
            item_id="catalog-onboard",
            previous_commit=FIRST_COMMIT,
            new_commit=SECOND_COMMIT,
            evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
            evidence_digest=CANDIDATE_DIGEST,
        )
    with pytest.raises(ValueError, match="must advance"):
        original.record_integration(
            item_id="catalog-onboard",
            previous_commit=BASE_COMMIT,
            new_commit=BASE_COMMIT,
            evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
            evidence_digest=CANDIDATE_DIGEST,
        )
    with pytest.raises(ValueError, match="no-change"):
        original.record_integration(
            item_id="catalog-verify",
            previous_commit=BASE_COMMIT,
            new_commit=FIRST_COMMIT,
            evidence_kind=state.IntegrationEvidenceKind.VERIFICATION,
            evidence_digest=VERIFY_DIGEST,
        )
    with pytest.raises(ValueError, match="verification evidence"):
        original.record_integration(
            item_id="catalog-verify",
            previous_commit=BASE_COMMIT,
            new_commit=BASE_COMMIT,
            evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
            evidence_digest="9" * 64,
        )


def test_integration_save_is_revision_and_generation_fenced(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    desired = original.record_integration(
        item_id="catalog-onboard",
        previous_commit=BASE_COMMIT,
        new_commit=FIRST_COMMIT,
        evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
        evidence_digest=CANDIDATE_DIGEST,
    )
    state.save_state(path, desired, expected_revision=0, expected_generation=1)

    with pytest.raises(state.StateConflictError, match="revision changed"):
        state.save_state(path, desired, expected_revision=0, expected_generation=1)

    with pytest.raises(state.StateConflictError, match="generation fence failed"):
        state.save_state(
            path, replace(desired, revision=2), expected_revision=1, expected_generation=2
        )


def test_integration_json_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / state.STATE_FILENAME
    desired = _run().record_integration(
        item_id="catalog-onboard",
        previous_commit=BASE_COMMIT,
        new_commit=FIRST_COMMIT,
        evidence_kind=state.IntegrationEvidenceKind.CANDIDATE,
        evidence_digest=CANDIDATE_DIGEST,
    )
    state.initialize_state(path, replace(desired, revision=0))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["integration_history"][0]["surprise"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown integration record fields"):
        state.load_state(path)


@pytest.mark.parametrize(
    "status",
    [state.WorkItemStatus.REJECTED, state.WorkItemStatus.BLOCKED],
)
def test_qa_can_close_approved_item_and_reason_is_immutable(
    tmp_path: Path,
    status: state.WorkItemStatus,
) -> None:
    path = tmp_path / state.STATE_FILENAME
    original = _run()
    state.initialize_state(path, original)
    item = original.item("catalog-onboard")
    closed_item = item.transition(status, terminal_reason="QA rejected frozen candidate")
    closed = original.replace_item(closed_item)
    state.save_state(path, closed, expected_revision=0, expected_generation=1)

    loaded = state.load_state(path)
    assert loaded.item("catalog-onboard").terminal_reason == "QA rejected frozen candidate"
    with pytest.raises(state.InvalidTransitionError, match="illegal work item transition"):
        loaded.item("catalog-onboard").transition(state.WorkItemStatus.INTEGRATING)
    rewritten_item = replace(
        loaded.item("catalog-onboard"),
        terminal_reason="rewritten reason",
    )
    with pytest.raises(state.StateConflictError, match="reason is immutable"):
        loaded.replace_item(rewritten_item)
