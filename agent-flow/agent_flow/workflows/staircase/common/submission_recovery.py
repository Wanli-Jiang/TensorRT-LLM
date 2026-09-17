# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Append-only, bounded recovery for crash-interrupted scheduler submissions."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Callable

from .slurm import SubmissionProbe


class SubmissionRecoveryError(RuntimeError):
    """Raised when durable submission recovery evidence is invalid."""


class SubmissionRecoveryAction(str, Enum):
    """Safe next action selected from an append-only recovery journal."""

    SUBMIT = "submit"
    WAIT = "wait"
    MANUAL_RECOVERY = "manual_recovery"


class SubmissionCancellationAction(str, Enum):
    """Safe cancellation decisions for one durable scheduler-call intent."""

    WAIT = "wait"
    CANCEL_INTENT = "cancel_intent"
    MANUAL_RECOVERY = "manual_recovery"


@dataclass(frozen=True, slots=True)
class SubmissionRecoveryPolicy:
    """Bounds for visibility grace, absence sampling, and scheduler calls."""

    visibility_grace_seconds: int = 120
    absence_samples_required: int = 2
    absence_sample_interval_seconds: int = 5
    max_submit_calls: int = 2
    max_probe_errors: int = 3

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SubmissionRecoveryDecision:
    """One journal-backed recovery decision."""

    action: SubmissionRecoveryAction
    reason: str
    claim_sequence: int | None = None


@dataclass(frozen=True, slots=True)
class SubmissionCancellationDecision:
    """One journal-backed decision that never authorizes scheduler submission."""

    action: SubmissionCancellationAction
    reason: str


Clock = Callable[[], datetime]


@contextmanager
def submission_recovery_lock(journal_dir: Path) -> Iterator[None]:
    """Serialize token probing, journal updates, and the scheduler call."""
    journal_dir.mkdir(parents=True, exist_ok=True)
    lock_path = journal_dir / ".lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


def submission_intent_digest(*parts: object) -> str:
    """Hash canonical scheduler intent values, including paths and enums."""
    encoded = json.dumps(
        [_json_safe(part) for part in parts],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decide_submission_recovery(
    *,
    journal_dir: Path,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    probe: SubmissionProbe,
    clock: Clock,
    policy: SubmissionRecoveryPolicy,
) -> SubmissionRecoveryDecision:
    """Select a first submit, delayed retry, wait, or durable manual boundary."""
    now = _trusted_now(clock)
    manual_path = journal_dir / "manual-recovery.json"
    if manual_path.is_file():
        _validate_manual_record(
            manual_path,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
        )
        return SubmissionRecoveryDecision(
            SubmissionRecoveryAction.MANUAL_RECOVERY,
            "submission recovery is durably exhausted; operator reconciliation is required",
        )
    if probe.matches:
        raise SubmissionRecoveryError("recovery decision requires zero scheduler matches")
    claims = _load_claims(
        journal_dir,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
    )
    if not claims:
        if not probe.queue_complete or not probe.accounting_complete:
            reason = probe.reason or "initial scheduler token probe is incomplete"
            return _manual_decision(
                journal_dir,
                submission_token,
                intent_revision,
                intent_digest,
                f"initial scheduler token evidence is unsafe: {reason}",
                clock,
            )
        sequence = 1
        _write_claim(
            journal_dir,
            sequence=sequence,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
            observed_at=now,
        )
        return SubmissionRecoveryDecision(
            SubmissionRecoveryAction.SUBMIT,
            "initial durable scheduler-call claim was appended",
            sequence,
        )

    if not probe.absence_proven:
        reason = probe.reason or "scheduler adapter did not prove queue and accounting absence"
        record_manual_recovery(
            journal_dir=journal_dir,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
            reason=reason,
            clock=clock,
        )
        return SubmissionRecoveryDecision(
            SubmissionRecoveryAction.MANUAL_RECOVERY,
            f"scheduler absence evidence is unsafe: {reason}",
        )

    latest = claims[-1]
    claimed_at = _parse_timestamp(latest["claimed_at"], "submission claim")
    if now < claimed_at:
        return _manual_decision(
            journal_dir,
            submission_token,
            intent_revision,
            intent_digest,
            "trusted clock moved backwards relative to the durable claim",
            clock,
        )
    age_seconds = (now - claimed_at).total_seconds()
    if age_seconds < policy.visibility_grace_seconds:
        return SubmissionRecoveryDecision(
            SubmissionRecoveryAction.WAIT,
            "scheduler visibility grace has not elapsed",
        )

    claim_sequence = int(latest["claim_sequence"])
    absences = _load_absences(
        journal_dir,
        claim_sequence=claim_sequence,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
    )
    if absences:
        previous = _parse_timestamp(absences[-1]["observed_at"], "absence sample")
        if now < previous:
            return _manual_decision(
                journal_dir,
                submission_token,
                intent_revision,
                intent_digest,
                "trusted clock moved backwards relative to an absence sample",
                clock,
            )
        if (now - previous).total_seconds() < policy.absence_sample_interval_seconds:
            return SubmissionRecoveryDecision(
                SubmissionRecoveryAction.WAIT,
                "another independent absence sample is not yet due",
            )
    sample_sequence = len(absences) + 1
    _write_absence(
        journal_dir,
        claim_sequence=claim_sequence,
        sample_sequence=sample_sequence,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
        observed_at=now,
    )
    if sample_sequence < policy.absence_samples_required:
        return SubmissionRecoveryDecision(
            SubmissionRecoveryAction.WAIT,
            "additional successful queue and accounting absence evidence is required",
        )
    if claim_sequence >= policy.max_submit_calls:
        return _manual_decision(
            journal_dir,
            submission_token,
            intent_revision,
            intent_digest,
            f"exhausted {policy.max_submit_calls} bounded scheduler submit calls",
            clock,
        )
    next_sequence = claim_sequence + 1
    _write_claim(
        journal_dir,
        sequence=next_sequence,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
        observed_at=now,
    )
    return SubmissionRecoveryDecision(
        SubmissionRecoveryAction.SUBMIT,
        "repeated post-grace queue and accounting absence proved a bounded retry safe",
        next_sequence,
    )


def decide_submission_cancellation(
    *,
    journal_dir: Path,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    probe: SubmissionProbe,
    clock: Clock,
    policy: SubmissionRecoveryPolicy,
) -> SubmissionCancellationDecision:
    """Cancel an unlaunched intent after bounded evidence, without ever submitting it."""
    now = _trusted_now(clock)
    recorded = load_submission_cancellation(
        journal_dir=journal_dir,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
    )
    if recorded is not None:
        return recorded
    cancellation_path = journal_dir / "cancellation.json"
    if probe.matches:
        raise SubmissionRecoveryError("cancellation absence decision requires zero matches")
    if not probe.absence_proven:
        reason = probe.reason or "scheduler did not prove queue and accounting absence"
        record_manual_recovery(
            journal_dir=journal_dir,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
            reason=f"submission cancellation evidence is unsafe: {reason}",
            clock=clock,
        )
        return SubmissionCancellationDecision(
            SubmissionCancellationAction.MANUAL_RECOVERY,
            f"submission cancellation evidence is unsafe: {reason}",
        )

    claims = _load_claims(
        journal_dir,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
    )
    if not claims:
        _write_cancellation(
            cancellation_path,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
            observed_at=now,
        )
        return SubmissionCancellationDecision(
            SubmissionCancellationAction.CANCEL_INTENT,
            "certified absence proved the unclaimed submission intent safe to cancel",
        )

    latest = claims[-1]
    claimed_at = _parse_timestamp(latest["claimed_at"], "submission claim")
    if now < claimed_at:
        return _cancellation_manual_decision(
            journal_dir,
            submission_token,
            intent_revision,
            intent_digest,
            "trusted clock moved backwards relative to the durable claim",
            clock,
        )
    if (now - claimed_at).total_seconds() < policy.visibility_grace_seconds:
        return SubmissionCancellationDecision(
            SubmissionCancellationAction.WAIT,
            "scheduler visibility grace has not elapsed before intent cancellation",
        )

    claim_sequence = int(latest["claim_sequence"])
    absences = _load_absences(
        journal_dir,
        claim_sequence=claim_sequence,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
    )
    if absences:
        previous = _parse_timestamp(absences[-1]["observed_at"], "absence sample")
        if now < previous:
            return _cancellation_manual_decision(
                journal_dir,
                submission_token,
                intent_revision,
                intent_digest,
                "trusted clock moved backwards relative to an absence sample",
                clock,
            )
        if (now - previous).total_seconds() < policy.absence_sample_interval_seconds:
            return SubmissionCancellationDecision(
                SubmissionCancellationAction.WAIT,
                "another independent cancellation absence sample is not yet due",
            )
    sample_sequence = len(absences) + 1
    _write_absence(
        journal_dir,
        claim_sequence=claim_sequence,
        sample_sequence=sample_sequence,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
        observed_at=now,
    )
    if sample_sequence < policy.absence_samples_required:
        return SubmissionCancellationDecision(
            SubmissionCancellationAction.WAIT,
            "additional certified cancellation absence evidence is required",
        )
    _write_cancellation(
        cancellation_path,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
        observed_at=now,
    )
    return SubmissionCancellationDecision(
        SubmissionCancellationAction.CANCEL_INTENT,
        "bounded queue and accounting absence proved the claimed intent safe to cancel",
    )


def load_submission_cancellation(
    *,
    journal_dir: Path,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
) -> SubmissionCancellationDecision | None:
    """Load a terminal cancellation/manual receipt before another scheduler query."""
    manual_path = journal_dir / "manual-recovery.json"
    if manual_path.is_file():
        _validate_manual_record(
            manual_path,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
        )
        return SubmissionCancellationDecision(
            SubmissionCancellationAction.MANUAL_RECOVERY,
            "submission cancellation requires operator reconciliation",
        )
    cancellation_path = journal_dir / "cancellation.json"
    if cancellation_path.is_file():
        _validate_cancellation_record(
            cancellation_path,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
        )
        return SubmissionCancellationDecision(
            SubmissionCancellationAction.CANCEL_INTENT,
            "submission intent was durably cancelled without a scheduler job",
        )
    return None


def record_submission_outcome(
    *,
    journal_dir: Path,
    claim_sequence: int,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    outcome: str,
    clock: Clock,
    scheduler_id: str | None = None,
) -> Path:
    """Append an immutable return-path receipt for one scheduler call."""
    if outcome not in {"accepted", "uncertain"}:
        raise ValueError("submission outcome must be accepted or uncertain")
    payload: dict[str, object] = {
        "schema_version": 1,
        "claim_sequence": claim_sequence,
        "submission_token": submission_token,
        "intent_revision": intent_revision,
        "intent_digest": intent_digest,
        "outcome": outcome,
        "observed_at": _format_timestamp(_trusted_now(clock)),
        "scheduler_id": scheduler_id,
    }
    path = journal_dir / f"outcome-{claim_sequence:04d}-{outcome}.json"
    _write_once_json(path, payload)
    return path


def record_probe_error(
    *,
    journal_dir: Path,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    reason: str,
    clock: Clock,
    policy: SubmissionRecoveryPolicy,
) -> SubmissionRecoveryDecision:
    """Append one failed scheduler-probe receipt and bound repeated errors."""
    manual_path = journal_dir / "manual-recovery.json"
    if manual_path.is_file():
        _validate_manual_record(
            manual_path,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
        )
        return SubmissionRecoveryDecision(
            SubmissionRecoveryAction.MANUAL_RECOVERY,
            "submission recovery is durably exhausted; operator reconciliation is required",
        )
    records = sorted(journal_dir.glob("probe-error-*.json"))
    for sequence, path in enumerate(records, start=1):
        record = _load_json(path)
        expected = (sequence, submission_token, intent_revision, intent_digest)
        actual = (
            record.get("probe_sequence"),
            record.get("submission_token"),
            record.get("intent_revision"),
            record.get("intent_digest"),
        )
        if actual != expected:
            raise SubmissionRecoveryError("scheduler probe-error history is malformed")
    sequence = len(records) + 1
    _write_once_json(
        journal_dir / f"probe-error-{sequence:04d}.json",
        {
            "schema_version": 1,
            "probe_sequence": sequence,
            "submission_token": submission_token,
            "intent_revision": intent_revision,
            "intent_digest": intent_digest,
            "reason": reason[:512],
            "observed_at": _format_timestamp(_trusted_now(clock)),
        },
    )
    if sequence < policy.max_probe_errors:
        return SubmissionRecoveryDecision(
            SubmissionRecoveryAction.WAIT,
            f"scheduler recovery probe failed ({sequence}/{policy.max_probe_errors})",
        )
    return _manual_decision(
        journal_dir,
        submission_token,
        intent_revision,
        intent_digest,
        f"scheduler recovery probe failed {sequence} consecutive times: {reason}",
        clock,
    )


def record_manual_recovery(
    *,
    journal_dir: Path,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    reason: str,
    clock: Clock,
) -> Path:
    """Persist one terminal, actionable fail-closed recovery receipt."""
    if not reason.strip():
        raise ValueError("manual recovery reason must be non-empty")
    path = journal_dir / "manual-recovery.json"
    if path.is_file():
        _validate_manual_record(
            path,
            submission_token=submission_token,
            intent_revision=intent_revision,
            intent_digest=intent_digest,
        )
        return path
    _write_once_json(
        path,
        {
            "schema_version": 1,
            "submission_token": submission_token,
            "intent_revision": intent_revision,
            "intent_digest": intent_digest,
            "status": "manual_recovery_required",
            "reason": reason.strip(),
            "recorded_at": _format_timestamp(_trusted_now(clock)),
        },
    )
    return path


def _manual_decision(
    journal_dir: Path,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    reason: str,
    clock: Clock,
) -> SubmissionRecoveryDecision:
    record_manual_recovery(
        journal_dir=journal_dir,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
        reason=reason,
        clock=clock,
    )
    return SubmissionRecoveryDecision(SubmissionRecoveryAction.MANUAL_RECOVERY, reason)


def _cancellation_manual_decision(
    journal_dir: Path,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    reason: str,
    clock: Clock,
) -> SubmissionCancellationDecision:
    record_manual_recovery(
        journal_dir=journal_dir,
        submission_token=submission_token,
        intent_revision=intent_revision,
        intent_digest=intent_digest,
        reason=reason,
        clock=clock,
    )
    return SubmissionCancellationDecision(SubmissionCancellationAction.MANUAL_RECOVERY, reason)


def _write_cancellation(
    path: Path,
    *,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    observed_at: datetime,
) -> None:
    _write_once_json(
        path,
        {
            "schema_version": 1,
            "submission_token": submission_token,
            "intent_revision": intent_revision,
            "intent_digest": intent_digest,
            "status": "submission_intent_cancelled",
            "observed_at": _format_timestamp(observed_at),
        },
    )


def _validate_cancellation_record(
    path: Path,
    *,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
) -> None:
    record = _load_json(path)
    expected = (
        submission_token,
        intent_revision,
        intent_digest,
        "submission_intent_cancelled",
    )
    actual = (
        record.get("submission_token"),
        record.get("intent_revision"),
        record.get("intent_digest"),
        record.get("status"),
    )
    if actual != expected:
        raise SubmissionRecoveryError("cancellation record conflicts with submission intent")
    _parse_timestamp(record.get("observed_at"), "submission cancellation")


def _write_claim(
    journal_dir: Path,
    *,
    sequence: int,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    observed_at: datetime,
) -> None:
    _write_once_json(
        journal_dir / f"claim-{sequence:04d}.json",
        {
            "schema_version": 1,
            "claim_sequence": sequence,
            "submission_token": submission_token,
            "intent_revision": intent_revision,
            "intent_digest": intent_digest,
            "status": "claimed",
            "claimed_at": _format_timestamp(observed_at),
        },
    )


def _write_absence(
    journal_dir: Path,
    *,
    claim_sequence: int,
    sample_sequence: int,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
    observed_at: datetime,
) -> None:
    _write_once_json(
        journal_dir / f"absence-{claim_sequence:04d}-{sample_sequence:04d}.json",
        {
            "schema_version": 1,
            "claim_sequence": claim_sequence,
            "sample_sequence": sample_sequence,
            "submission_token": submission_token,
            "intent_revision": intent_revision,
            "intent_digest": intent_digest,
            "queue_complete": True,
            "accounting_complete": True,
            "observed_at": _format_timestamp(observed_at),
        },
    )


def _load_claims(
    journal_dir: Path,
    *,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
) -> list[dict[str, object]]:
    claims = [_load_json(path) for path in sorted(journal_dir.glob("claim-*.json"))]
    for sequence, claim in enumerate(claims, start=1):
        expected = {
            "schema_version": 1,
            "claim_sequence": sequence,
            "submission_token": submission_token,
            "intent_revision": intent_revision,
            "intent_digest": intent_digest,
            "status": "claimed",
        }
        if {key: claim.get(key) for key in expected} != expected:
            raise SubmissionRecoveryError("submission claim history is malformed or conflicting")
        _parse_timestamp(claim.get("claimed_at"), "submission claim")
    return claims


def _load_absences(
    journal_dir: Path,
    *,
    claim_sequence: int,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
) -> list[dict[str, object]]:
    pattern = f"absence-{claim_sequence:04d}-*.json"
    records = [_load_json(path) for path in sorted(journal_dir.glob(pattern))]
    for sequence, record in enumerate(records, start=1):
        expected = {
            "schema_version": 1,
            "claim_sequence": claim_sequence,
            "sample_sequence": sequence,
            "submission_token": submission_token,
            "intent_revision": intent_revision,
            "intent_digest": intent_digest,
            "queue_complete": True,
            "accounting_complete": True,
        }
        if {key: record.get(key) for key in expected} != expected:
            raise SubmissionRecoveryError("submission absence history is malformed or conflicting")
        _parse_timestamp(record.get("observed_at"), "absence sample")
    return records


def _validate_manual_record(
    path: Path,
    *,
    submission_token: str,
    intent_revision: str,
    intent_digest: str,
) -> None:
    record = _load_json(path)
    expected = (
        submission_token,
        intent_revision,
        intent_digest,
        "manual_recovery_required",
    )
    actual = (
        record.get("submission_token"),
        record.get("intent_revision"),
        record.get("intent_digest"),
        record.get("status"),
    )
    if actual != expected:
        raise SubmissionRecoveryError("manual recovery record conflicts with submission intent")


def _trusted_now(clock: Clock) -> datetime:
    now = clock()
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise SubmissionRecoveryError("submission recovery clock must return timezone-aware time")
    return now.astimezone(UTC)


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise SubmissionRecoveryError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SubmissionRecoveryError(f"{label} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise SubmissionRecoveryError(f"{label} timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SubmissionRecoveryError(
            f"submission recovery record is unreadable: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SubmissionRecoveryError(f"submission recovery record is not an object: {path}")
    return value


def _write_once_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != encoded:
            raise SubmissionRecoveryError(f"immutable recovery record conflicts: {path}")
        return
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(entry) for key, entry in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(entry) for entry in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported submission intent value: {type(value).__name__}")


__all__ = [
    "Clock",
    "SubmissionCancellationAction",
    "SubmissionCancellationDecision",
    "SubmissionRecoveryAction",
    "SubmissionRecoveryDecision",
    "SubmissionRecoveryError",
    "SubmissionRecoveryPolicy",
    "decide_submission_cancellation",
    "decide_submission_recovery",
    "load_submission_cancellation",
    "record_manual_recovery",
    "record_probe_error",
    "record_submission_outcome",
    "submission_intent_digest",
    "submission_recovery_lock",
]
