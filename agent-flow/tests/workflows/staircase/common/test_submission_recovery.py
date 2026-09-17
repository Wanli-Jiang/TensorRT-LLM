# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent_flow.workflows.staircase.common.slurm import (
    JobStatus,
    ObservationSource,
    SubmissionProbe,
)
from agent_flow.workflows.staircase.common.submission_recovery import (
    SubmissionRecoveryAction,
    SubmissionRecoveryPolicy,
    decide_submission_recovery,
    record_probe_error,
    submission_recovery_lock,
)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 16, 16, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


def _absence(token: str = "submission-token-0001") -> SubmissionProbe:
    return SubmissionProbe(
        token,
        (),
        JobStatus.UNKNOWN,
        "clean absence",
        ObservationSource.UNKNOWN,
        True,
        True,
        True,
    )


def _policy() -> SubmissionRecoveryPolicy:
    return SubmissionRecoveryPolicy(
        visibility_grace_seconds=10,
        absence_samples_required=2,
        absence_sample_interval_seconds=2,
        max_submit_calls=2,
        max_probe_errors=2,
    )


def _decide(root: Path, clock: _Clock):
    return decide_submission_recovery(
        journal_dir=root,
        submission_token="submission-token-0001",
        intent_revision="revision-1",
        intent_digest="a" * 64,
        probe=_absence(),
        clock=clock,
        policy=_policy(),
    )


def test_claim_crash_recovers_only_after_grace_and_repeated_absence(tmp_path: Path) -> None:
    clock = _Clock()
    first = _decide(tmp_path, clock)
    assert first.action is SubmissionRecoveryAction.SUBMIT
    assert first.claim_sequence == 1

    assert _decide(tmp_path, clock).action is SubmissionRecoveryAction.WAIT
    clock.advance(10)
    assert _decide(tmp_path, clock).action is SubmissionRecoveryAction.WAIT
    clock.advance(2)
    retry = _decide(tmp_path, clock)
    assert retry.action is SubmissionRecoveryAction.SUBMIT
    assert retry.claim_sequence == 2

    clock.advance(10)
    assert _decide(tmp_path, clock).action is SubmissionRecoveryAction.WAIT
    clock.advance(2)
    exhausted = _decide(tmp_path, clock)
    assert exhausted.action is SubmissionRecoveryAction.MANUAL_RECOVERY
    assert (tmp_path / "manual-recovery.json").is_file()
    assert len(tuple(tmp_path.glob("claim-*.json"))) == 2
    assert len(tuple(tmp_path.glob("absence-*.json"))) == 4


def test_untrusted_absence_and_repeated_probe_errors_end_in_manual_recovery(
    tmp_path: Path,
) -> None:
    clock = _Clock()
    unsafe = SubmissionProbe(
        "submission-token-0001",
        (),
        JobStatus.UNKNOWN,
        "malformed accounting row",
        ObservationSource.UNKNOWN,
        True,
        False,
        False,
    )
    decision = decide_submission_recovery(
        journal_dir=tmp_path / "unsafe",
        submission_token=unsafe.submission_token,
        intent_revision="revision-1",
        intent_digest="a" * 64,
        probe=unsafe,
        clock=clock,
        policy=_policy(),
    )
    assert decision.action is SubmissionRecoveryAction.MANUAL_RECOVERY

    errors = tmp_path / "errors"
    first = record_probe_error(
        journal_dir=errors,
        submission_token=unsafe.submission_token,
        intent_revision="revision-1",
        intent_digest="a" * 64,
        reason="sacct timeout",
        clock=clock,
        policy=_policy(),
    )
    second = record_probe_error(
        journal_dir=errors,
        submission_token=unsafe.submission_token,
        intent_revision="revision-1",
        intent_digest="a" * 64,
        reason="sacct timeout",
        clock=clock,
        policy=_policy(),
    )
    assert first.action is SubmissionRecoveryAction.WAIT
    assert second.action is SubmissionRecoveryAction.MANUAL_RECOVERY


def test_concurrent_recovery_serializes_the_scheduler_call_claim(tmp_path: Path) -> None:
    clock = _Clock()

    def recover() -> SubmissionRecoveryAction:
        with submission_recovery_lock(tmp_path):
            return _decide(tmp_path, clock).action

    with ThreadPoolExecutor(max_workers=2) as pool:
        actions = tuple(pool.map(lambda _index: recover(), range(2)))

    assert actions.count(SubmissionRecoveryAction.SUBMIT) == 1
    assert actions.count(SubmissionRecoveryAction.WAIT) == 1
    assert len(tuple(tmp_path.glob("claim-*.json"))) == 1
