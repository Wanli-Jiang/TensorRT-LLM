# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, evidence-bounded terminal reports for Staircase."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Sequence

from agent_flow.workflows.staircase.common.artifacts import (
    MANIFEST_SCHEMA_VERSION,
    IngestedResult,
    JsonValue,
)
from agent_flow.workflows.staircase.common.gates import EvidenceScope, GatePurpose, GateReceipt
from agent_flow.workflows.staircase.common.slurm import JobObservation
from agent_flow.workflows.staircase.state import (
    PLANNING_ITEM_ID,
    AttemptRecord,
    AttemptStatus,
    JobReference,
    Role,
    RunState,
    RunTerminalStatus,
)
from agent_flow.workflows.staircase.task_schema import NormalizedTask

REPORT_SCHEMA_VERSION = 2
JSON_REPORT_FILENAME = "terminal-report.json"
MARKDOWN_REPORT_FILENAME = "terminal-report.md"
PRODUCTION_CERTIFICATION_LEVELS = tuple(EvidenceScope)
DELIVERY_RECEIPT_SCHEMA_VERSION = 1
DELIVERY_RECEIPT_RELATIVE_PATH = PurePosixPath("delivery/patch-receipt.json")
DELIVERY_PATCH_RELATIVE_PATH = PurePosixPath("delivery/staircase.patch")
MAX_DELIVERY_RECEIPT_BYTES = 65_536
MAX_PLANNING_RECEIPT_BYTES = 65_536
PLANNING_RECEIPT_SCHEMA_VERSION = 1

_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(auth|cookie|credential|key|pass(?:word)?|secret|token)(\s*[:=]\s*)([^\s,;]+)"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_URL_CREDENTIALS = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)[^/@\s]+@")


class TerminalReportError(RuntimeError):
    """Raised when terminal evidence is invalid or a report would be overwritten."""


class ImmutableReportError(TerminalReportError):
    """Raised when an existing terminal report differs from requested content."""


@dataclass(frozen=True, slots=True)
class GateEvidence:
    """A typed gate receipt pinned to one immutable ingested result."""

    attempt_id: str
    result_digest: str
    receipt: GateReceipt

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError("gate evidence attempt_id must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.result_digest):
            raise ValueError("gate evidence result_digest must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class PlanningEvidence:
    """A typed accepted planning result and its immutable ingestion receipt."""

    attempt_id: str
    role: Role
    outcome: str
    plan_digest: str
    attempt_input_digest: str
    role_input_digest: str
    result_digest: str
    response_digest: str
    receipt_path: Path
    receipt_digest: str
    review_of_attempt_id: str | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError("planning evidence attempt_id must be non-empty")
        if self.role not in {Role.PLAN_DRAFTER, Role.PLAN_REVIEWER}:
            raise ValueError("planning evidence requires a planning role")
        if self.role is Role.PLAN_DRAFTER:
            if self.outcome != "DRAFTED" or self.review_of_attempt_id is not None:
                raise ValueError("PlanDrafter evidence must describe one DRAFTED result")
        elif self.outcome != "ACCEPT" or not self.review_of_attempt_id:
            raise ValueError("PlanReviewer evidence must describe one linked ACCEPT result")
        for name in (
            "plan_digest",
            "attempt_input_digest",
            "role_input_digest",
            "result_digest",
            "response_digest",
            "receipt_digest",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"planning evidence {name} must be a lowercase SHA-256 digest")
        if not isinstance(self.receipt_path, Path):
            raise TypeError("planning evidence receipt_path must be a Path")


@dataclass(frozen=True, slots=True)
class TerminalReport:
    """Byte-stable JSON and Markdown renderings of one terminal run."""

    outcome: RunTerminalStatus
    json_text: str
    markdown_text: str


@dataclass(frozen=True, slots=True)
class WrittenTerminalReport:
    """Workspace-relative paths of an immutable report pair."""

    json_path: str
    markdown_path: str


def build_terminal_report(
    state: RunState,
    task: NormalizedTask,
    *,
    workspace: Path,
    ingested_results: Sequence[IngestedResult] = (),
    planning_evidence: Sequence[PlanningEvidence] = (),
    gate_evidence: Sequence[GateEvidence] = (),
    scheduler_observations: Sequence[JobObservation] = (),
) -> TerminalReport:
    """Build terminal JSON and Markdown without inferring unproven claims.

    Args:
        state: Validated authoritative terminal controller state.
        task: Immutable normalized task that created the run.
        workspace: Canonical run workspace containing ingestion receipts.
        ingested_results: Results already accepted by the controller.
        planning_evidence: Accepted PlanDrafter/PlanReviewer results and receipts.
        gate_evidence: Typed gate receipts pinned to accepted result digests.
        scheduler_observations: Optional observations of exact owned jobs.

    Returns:
        Deterministic report bytes ready for immutable publication.

    Raises:
        TerminalReportError: If identities, receipts, paths, or evidence conflict.
    """
    if state.terminal_status is RunTerminalStatus.ACTIVE:
        raise TerminalReportError("active state cannot be emitted as a terminal report")
    if state.task_digest != task.digest:
        raise TerminalReportError("state and normalized task digest do not match")
    if state.base_commit != task.repository.base_commit:
        raise TerminalReportError("state and normalized task base commit do not match")

    workspace_root = _canonical_workspace(workspace)
    attempts = _attempts_by_id(state)
    results = _validated_results(
        state,
        attempts,
        workspace_root=workspace_root,
        ingested_results=ingested_results,
    )
    planning = _validated_planning_evidence(
        state,
        workspace_root=workspace_root,
        planning_evidence=planning_evidence,
    )
    gates = _validated_gate_evidence(gate_evidence, results)
    jobs = _job_records(state, scheduler_observations)
    delivery_boundary = _delivery_boundary(state, task, workspace_root)

    payload: dict[str, JsonValue] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "outcome": state.terminal_status.value,
        "terminal_reason": _redact(state.terminal_reason or ""),
        "identity": {
            "run_id": state.run_id,
            "task_digest": state.task_digest,
            "base_commit": state.base_commit,
            "workflow_mode": state.workflow_mode.value,
            "generation": state.generation,
            "state_revision": state.revision,
            "target": {
                "family": task.target.family,
                "checkpoint_id": task.target.checkpoint_id,
                "architecture": task.reference.architecture,
                "sm": task.target.sm,
                "world_size": task.target.world_size,
                "synthetic_target": task.target.synthetic_target,
            },
        },
        "build": {
            "identity": task.execution.slurm.controller.build_identity,
            "execution_mode": task.execution.mode,
        },
        "delivery_boundary": delivery_boundary,
        "hierarchy": _hierarchy_records(state, results, planning),
        "jobs": jobs,
        "evidence": _evidence_records(results, gates, planning),
    }
    json_text = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return TerminalReport(
        outcome=state.terminal_status,
        json_text=json_text,
        markdown_text=_render_markdown(payload),
    )


def _delivery_boundary(
    state: RunState,
    task: NormalizedTask,
    workspace_root: Path,
) -> dict[str, JsonValue]:
    """Return delivery claims bounded by persisted state and verified files."""
    boundary: dict[str, JsonValue] = {
        "mode": task.delivery.mode,
        "branch": task.delivery.branch,
        "merge": task.delivery.merge,
        "push": task.delivery.push,
        "base_commit": state.base_commit,
        "integration_head": state.integration_head,
    }
    if task.delivery.mode == "signed_commits":
        if task.delivery.branch is None:
            raise TerminalReportError("signed-commit delivery requires a configured branch")
        checkout = workspace_root / "delivery" / "repository"
        if checkout.exists() or checkout.is_symlink():
            _verify_delivery_checkout(
                workspace_root,
                state,
                mode="signed_commits",
                branch=task.delivery.branch,
            )
            boundary["repository_status"] = "verified"
        elif state.terminal_status is RunTerminalStatus.SUCCEEDED:
            raise TerminalReportError("successful signed-commit delivery lacks its checkout")
        else:
            boundary["repository_status"] = "missing"
        return boundary
    if task.delivery.mode != "diff_only":
        raise TerminalReportError(f"unsupported delivery mode {task.delivery.mode!r}")
    patch = _verified_delivery_patch(
        state,
        workspace_root,
        required=state.terminal_status is RunTerminalStatus.SUCCEEDED,
    )
    if patch["status"] == "verified":
        patch_bytes = _read_confined_regular_file(
            workspace_root,
            DELIVERY_PATCH_RELATIVE_PATH,
            label="delivery patch",
            maximum_bytes=max(int(patch["size_bytes"]), 1),
        )
        _verify_delivery_checkout(
            workspace_root,
            state,
            mode="diff_only",
            branch=None,
            patch_bytes=patch_bytes,
        )
        patch["repository_status"] = "verified"
    boundary["patch"] = patch
    return boundary


def _verify_delivery_checkout(
    workspace_root: Path,
    state: RunState,
    *,
    mode: str,
    branch: str | None,
    patch_bytes: bytes | None = None,
) -> None:
    """Reverify delivery bytes and branch identity against the exact Git history."""
    from agent_flow.workflows.staircase.common.gitops import ControllerGitOps, GitOpsError

    relative = PurePosixPath(
        "delivery/repository" if mode == "signed_commits" else "delivery/diff-repository"
    )
    checkout = workspace_root.joinpath(*relative.parts)
    if checkout.is_symlink() or not checkout.is_dir():
        raise TerminalReportError(f"delivery checkout is missing or unsafe: {checkout}")
    try:
        if checkout.resolve(strict=True) != checkout:
            raise TerminalReportError("delivery checkout does not have its exact canonical path")
        git = ControllerGitOps(
            checkout,
            workspace_root / "locks" / "terminal-report-git.lock",
            "staircase-terminal-report",
        )
        inspection = git.inspect()
        if inspection.head != state.integration_head:
            raise TerminalReportError("delivery checkout HEAD differs from integration state")
        if inspection.branch != branch:
            raise TerminalReportError("delivery checkout branch identity differs from task")
        if not inspection.clean:
            raise TerminalReportError("delivery checkout is not clean")
        rendered = git.render_patch(
            base_commit=state.base_commit,
            head_commit=state.integration_head,
        )
    except GitOpsError as error:
        raise TerminalReportError(f"delivery Git verification failed: {error}") from error
    if patch_bytes is not None and rendered != patch_bytes:
        raise TerminalReportError(
            "delivery patch bytes do not encode the exact base-to-integration-head diff"
        )


def _verified_delivery_patch(
    state: RunState,
    workspace_root: Path,
    *,
    required: bool,
) -> dict[str, JsonValue]:
    receipt_path = workspace_root.joinpath(*DELIVERY_RECEIPT_RELATIVE_PATH.parts)
    patch_path = workspace_root.joinpath(*DELIVERY_PATCH_RELATIVE_PATH.parts)
    receipt_present = receipt_path.exists() or receipt_path.is_symlink()
    patch_present = patch_path.exists() or patch_path.is_symlink()
    if not receipt_present and not patch_present:
        if required:
            raise TerminalReportError(
                "successful diff-only delivery lacks its patch receipt and artifact"
            )
        return {"status": "missing"}
    if not receipt_present or not patch_present:
        raise TerminalReportError("diff-only delivery has an incomplete patch artifact pair")

    receipt_bytes = _read_confined_regular_file(
        workspace_root,
        DELIVERY_RECEIPT_RELATIVE_PATH,
        label="delivery patch receipt",
        maximum_bytes=MAX_DELIVERY_RECEIPT_BYTES,
    )
    patch_sha256, patch_size = _digest_confined_regular_file(
        workspace_root,
        DELIVERY_PATCH_RELATIVE_PATH,
        label="delivery patch",
    )
    receipt = _strict_json_object(receipt_bytes, label="delivery patch receipt")
    expected: dict[str, str | int] = {
        "schema_version": DELIVERY_RECEIPT_SCHEMA_VERSION,
        "run_id": state.run_id,
        "task_digest": state.task_digest,
        "base_commit": state.base_commit,
        "head_commit": state.integration_head,
        "patch_sha256": patch_sha256,
        "size_bytes": patch_size,
        "path": DELIVERY_PATCH_RELATIVE_PATH.as_posix(),
    }
    if set(receipt) != set(expected) or any(
        type(receipt.get(name)) is not type(value) or receipt.get(name) != value
        for name, value in expected.items()
    ):
        raise TerminalReportError(
            "delivery patch receipt differs from the exact run, integration head, or bytes"
        )
    return {
        "status": "verified",
        "receipt_path": DELIVERY_RECEIPT_RELATIVE_PATH.as_posix(),
        "path": DELIVERY_PATCH_RELATIVE_PATH.as_posix(),
        "sha256": patch_sha256,
        "size_bytes": patch_size,
    }


def write_terminal_report(
    workspace: Path,
    report: TerminalReport,
    *,
    relative_directory: str = "reports",
) -> WrittenTerminalReport:
    """Atomically publish an immutable JSON/Markdown report pair.

    Repeating the call with byte-identical content is a no-op. Existing
    different content is never replaced.

    Args:
        workspace: Canonical run workspace that owns the report.
        report: Fully rendered terminal report.
        relative_directory: Safe workspace-relative output directory.

    Returns:
        Workspace-relative report paths.
    """
    workspace_root = _canonical_workspace(workspace)
    relative = _safe_relative_path(relative_directory, "relative_directory")
    output_dir = _prepare_output_directory(workspace_root, relative)

    json_path = output_dir / JSON_REPORT_FILENAME
    markdown_path = output_dir / MARKDOWN_REPORT_FILENAME
    lock_path = output_dir / ".terminal-report.lock"
    if lock_path.is_symlink():
        raise ImmutableReportError(f"report lock cannot be a symlink: {lock_path}")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _preflight_immutable(json_path, report.json_text)
        _preflight_immutable(markdown_path, report.markdown_text)
        _write_once(json_path, report.json_text)
        _write_once(markdown_path, report.markdown_text)
        _fsync_directory(output_dir)

    return WrittenTerminalReport(
        json_path=json_path.relative_to(workspace_root).as_posix(),
        markdown_path=markdown_path.relative_to(workspace_root).as_posix(),
    )


def _attempts_by_id(state: RunState) -> dict[str, AttemptRecord]:
    attempts = {
        attempt.attempt_id: attempt
        for attempt in (
            *state.planning_attempts,
            *(attempt for item in state.items for attempt in item.attempts),
        )
    }
    return attempts


def _validated_results(
    state: RunState,
    attempts: dict[str, AttemptRecord],
    *,
    workspace_root: Path,
    ingested_results: Sequence[IngestedResult],
) -> dict[str, tuple[IngestedResult, str]]:
    results: dict[str, tuple[IngestedResult, str]] = {}
    for ingested in ingested_results:
        manifest = ingested.manifest
        if manifest.attempt_id in results:
            raise TerminalReportError(f"duplicate ingested result for {manifest.attempt_id!r}")
        attempt = attempts.get(manifest.attempt_id)
        if attempt is None:
            raise TerminalReportError(
                f"ingested result names unknown attempt {manifest.attempt_id!r}"
            )
        if (
            manifest.run_id != state.run_id
            or manifest.item_id != attempt.item_id
            or manifest.task_digest != state.task_digest
            or manifest.generation != attempt.generation
        ):
            raise TerminalReportError(
                f"ingested result identity does not match attempt {manifest.attempt_id!r}"
            )
        if attempt.result_digest != ingested.result_digest:
            raise TerminalReportError(
                f"ingested result digest does not match state for {manifest.attempt_id!r}"
            )
        if attempt.candidate_digest != manifest.candidate_digest:
            raise TerminalReportError(
                f"candidate digest does not match state for {manifest.attempt_id!r}"
            )
        if attempt.reviewed_candidate_digest != manifest.reviewed_candidate_digest:
            raise TerminalReportError(
                f"reviewed candidate digest does not match state for {manifest.attempt_id!r}"
            )
        receipt_path = _workspace_relative_file(
            workspace_root, ingested.receipt_path, label="ingestion receipt"
        )
        _validate_ingestion_receipt(ingested, workspace_root / receipt_path)
        results[manifest.attempt_id] = (ingested, receipt_path.as_posix())
    return results


def _validate_ingestion_receipt(ingested: IngestedResult, receipt_path: Path) -> None:
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise TerminalReportError(f"ingestion receipt is not a regular file: {receipt_path}")
    try:
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TerminalReportError(f"cannot read ingestion receipt: {receipt_path}") from error
    manifest = ingested.manifest
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "generation": manifest.generation,
        "task_digest": manifest.task_digest,
        "input_digest": manifest.input_digest,
        "result_digest": ingested.result_digest,
    }
    if value != expected:
        raise TerminalReportError(f"ingestion receipt content mismatch: {receipt_path}")


def _validated_planning_evidence(
    state: RunState,
    *,
    workspace_root: Path,
    planning_evidence: Sequence[PlanningEvidence],
) -> tuple[PlanningEvidence, ...]:
    """Validate the accepted planning pair against state and receipt bytes."""
    if not planning_evidence:
        if state.planning_attempts and state.items:
            raise TerminalReportError("admitted state lacks accepted planning evidence")
        return ()
    if len(planning_evidence) != 2:
        raise TerminalReportError("accepted planning evidence must contain exactly two results")
    by_role = {entry.role: entry for entry in planning_evidence}
    if set(by_role) != {Role.PLAN_DRAFTER, Role.PLAN_REVIEWER}:
        raise TerminalReportError("accepted planning evidence must contain one result per role")
    draft = by_role[Role.PLAN_DRAFTER]
    review = by_role[Role.PLAN_REVIEWER]
    if review.review_of_attempt_id != draft.attempt_id:
        raise TerminalReportError("PlanReviewer evidence does not link the accepted PlanDrafter")
    if review.plan_digest != draft.plan_digest:
        raise TerminalReportError("accepted planning evidence has mismatched plan digests")

    attempts = {attempt.attempt_id: attempt for attempt in state.planning_attempts}
    draft_attempt = attempts.get(draft.attempt_id)
    review_attempt = attempts.get(review.attempt_id)
    if draft_attempt is None or review_attempt is None:
        raise TerminalReportError("accepted planning evidence names an unknown planning attempt")
    if state.planning_attempts[-1] != review_attempt:
        raise TerminalReportError("accepted PlanReviewer is not the final planning attempt")
    if (
        draft_attempt.role is not Role.PLAN_DRAFTER
        or review_attempt.role is not Role.PLAN_REVIEWER
        or draft_attempt.status is not AttemptStatus.VALIDATED
        or review_attempt.status is not AttemptStatus.VALIDATED
        or review_attempt.review_of_attempt_id != draft_attempt.attempt_id
        or draft_attempt.candidate_digest != draft.plan_digest
        or review_attempt.candidate_digest != draft.plan_digest
        or review_attempt.reviewed_candidate_digest != draft.plan_digest
    ):
        raise TerminalReportError("accepted planning evidence differs from authoritative lineage")

    validated: list[PlanningEvidence] = []
    for evidence, attempt in ((draft, draft_attempt), (review, review_attempt)):
        if attempt.result_digest != evidence.result_digest:
            raise TerminalReportError(
                f"planning result digest differs from state for {evidence.attempt_id!r}"
            )
        receipt_relative = _workspace_relative_file(
            workspace_root,
            evidence.receipt_path,
            label="planning ingestion receipt",
        )
        expected_relative = PurePosixPath(
            "receipts",
            state.run_id,
            PLANNING_ITEM_ID,
            f"{attempt.attempt_id}.json",
        )
        if receipt_relative != expected_relative:
            raise TerminalReportError(
                f"planning ingestion receipt has the wrong path for {evidence.attempt_id!r}"
            )
        receipt_bytes = _read_confined_regular_file(
            workspace_root,
            expected_relative,
            label="planning ingestion receipt",
            maximum_bytes=MAX_PLANNING_RECEIPT_BYTES,
        )
        if hashlib.sha256(receipt_bytes).hexdigest() != evidence.receipt_digest:
            raise TerminalReportError(
                f"planning receipt digest mismatch for {evidence.attempt_id!r}"
            )
        receipt = _strict_json_object(receipt_bytes, label="planning ingestion receipt")
        expected: dict[str, object] = {
            "schema_version": PLANNING_RECEIPT_SCHEMA_VERSION,
            "run_id": state.run_id,
            "task_digest": state.task_digest,
            "item_id": PLANNING_ITEM_ID,
            "attempt_id": attempt.attempt_id,
            "generation": attempt.generation,
            "role": attempt.role.value,
            "outcome": evidence.outcome,
            "plan_digest": evidence.plan_digest,
            "attempt_input_digest": evidence.attempt_input_digest,
            "role_input_digest": evidence.role_input_digest,
            "result_digest": evidence.result_digest,
            "response_digest": evidence.response_digest,
            "review_of_attempt_id": evidence.review_of_attempt_id,
        }
        if set(receipt) != set(expected) or any(
            type(receipt.get(name)) is not type(value) or receipt.get(name) != value
            for name, value in expected.items()
        ):
            raise TerminalReportError(
                f"planning ingestion receipt content mismatch for {evidence.attempt_id!r}"
            )
        validated.append(replace(evidence, receipt_path=Path(receipt_relative.as_posix())))
    return tuple(validated)


def _validated_gate_evidence(
    gate_evidence: Sequence[GateEvidence],
    results: dict[str, tuple[IngestedResult, str]],
) -> tuple[GateEvidence, ...]:
    seen: set[tuple[str, str]] = set()
    validated: list[GateEvidence] = []
    for evidence in gate_evidence:
        result = results.get(evidence.attempt_id)
        if result is None or result[0].result_digest != evidence.result_digest:
            raise TerminalReportError(
                f"gate {evidence.receipt.gate_id!r} is not pinned to an ingested result"
            )
        identity = (evidence.attempt_id, evidence.receipt.gate_id)
        if identity in seen:
            raise TerminalReportError(f"duplicate gate evidence identity: {identity!r}")
        seen.add(identity)
        validated.append(evidence)
    return tuple(sorted(validated, key=lambda entry: (entry.attempt_id, entry.receipt.gate_id)))


def _hierarchy_records(
    state: RunState,
    results: dict[str, tuple[IngestedResult, str]],
    planning: tuple[PlanningEvidence, ...],
) -> dict[str, JsonValue]:
    planning_attempt_ids = {evidence.attempt_id for evidence in planning}
    return {
        "stages": [
            {
                "stage_id": stage.stage_id,
                "status": stage.status.value,
                "required_goal_ids": list(stage.required_goal_ids),
                "terminal_reason": _redact(stage.terminal_reason or "") or None,
            }
            for stage in sorted(state.stages, key=lambda entry: entry.stage_id)
        ],
        "goals": [
            {
                "goal_id": goal.goal_id,
                "stage_id": goal.stage_id,
                "status": goal.status.value,
                "required_item_ids": list(goal.required_item_ids),
                "terminal_reason": _redact(goal.terminal_reason or "") or None,
            }
            for goal in sorted(state.goals, key=lambda entry: entry.goal_id)
        ],
        "work_items": [
            {
                "item_id": item.item_id,
                "stage_id": item.stage_id,
                "goal_id": item.goal_id,
                "kind": item.kind.value,
                "profile": item.profile.value,
                "status": item.status.value,
                "dependencies": list(item.dependencies),
                "candidate_attempt_id": item.candidate_attempt_id,
                "candidate_digest": item.candidate_digest,
                "reviewer_attempt_id": item.reviewer_attempt_id,
                "terminal_reason": _redact(item.terminal_reason or "") or None,
                "attempts": [
                    _attempt_record(attempt, attempt.attempt_id in results)
                    for attempt in item.attempts
                ],
            }
            for item in sorted(state.items, key=lambda entry: entry.item_id)
        ],
        "planning_attempts": [
            _attempt_record(
                attempt,
                attempt.attempt_id in results or attempt.attempt_id in planning_attempt_ids,
            )
            for attempt in state.planning_attempts
        ],
    }


def _attempt_record(attempt: AttemptRecord, has_ingested_result: bool) -> dict[str, JsonValue]:
    return {
        "attempt_id": attempt.attempt_id,
        "sequence": attempt.sequence,
        "role": attempt.role.value,
        "kind": attempt.kind.value,
        "profile": attempt.profile.value if attempt.profile is not None else None,
        "generation": attempt.generation,
        "status": attempt.status.value,
        "scheduler_state": _redact(attempt.scheduler_state or "") or None,
        "result_digest": attempt.result_digest,
        "ingested_result_present": has_ingested_result,
        "candidate_digest": attempt.candidate_digest,
        "review_of_attempt_id": attempt.review_of_attempt_id,
        "reviewed_candidate_digest": attempt.reviewed_candidate_digest,
        "terminal_reason": _redact(attempt.terminal_reason or "") or None,
    }


def _job_records(
    state: RunState, scheduler_observations: Sequence[JobObservation]
) -> list[JsonValue]:
    owned: list[tuple[str, str, JobReference]] = []
    if state.controller_job is not None:
        owned.append(("controller", state.run_id, state.controller_job))
    owned.extend(
        ("attempt", attempt.attempt_id, attempt.job)
        for attempt in (
            *state.planning_attempts,
            *(attempt for item in state.items for attempt in item.attempts),
        )
        if attempt.job is not None
    )
    owned_by_identity: dict[tuple[str, str | None, str | None], tuple[str, str, JobReference]] = {}
    for owner in owned:
        identity = _job_key(owner[2])
        if identity in owned_by_identity:
            raise TerminalReportError(
                f"exact scheduler job {owner[2].scheduler_id!r} has multiple owners"
            )
        owned_by_identity[identity] = owner

    observations: dict[tuple[str, str | None, str | None], JobObservation] = {}
    for observation in scheduler_observations:
        identity = (
            observation.identity.job_id,
            observation.identity.array_task_id,
            observation.identity.cluster,
        )
        if identity not in owned_by_identity:
            raise TerminalReportError(
                f"scheduler observation is not for an owned exact job: "
                f"{observation.identity.scheduler_id}"
            )
        if identity in observations:
            raise TerminalReportError(
                f"duplicate scheduler observation: {observation.identity.scheduler_id}"
            )
        observations[identity] = observation

    records: list[JsonValue] = []
    for identity, (owner_type, owner_id, job) in sorted(
        owned_by_identity.items(), key=lambda entry: (entry[1][0], entry[1][1])
    ):
        observation = observations.get(identity)
        records.append(
            {
                "owner_type": owner_type,
                "owner_id": owner_id,
                "scheduler_id": job.scheduler_id,
                "job_id": job.job_id,
                "array_task_id": job.array_task_id,
                "cluster": job.cluster,
                "observation": (
                    {
                        "status": observation.status.value,
                        "source": observation.source.value,
                        "reason": _redact(observation.reason or "") or None,
                        "raw_state": _redact(observation.raw_state or "") or None,
                    }
                    if observation is not None
                    else None
                ),
            }
        )
    return records


def _job_key(job: JobReference) -> tuple[str, str | None, str | None]:
    return (job.job_id, job.array_task_id, job.cluster)


def _evidence_records(
    results: dict[str, tuple[IngestedResult, str]],
    gates: tuple[GateEvidence, ...],
    planning: tuple[PlanningEvidence, ...],
) -> dict[str, JsonValue]:
    gate_records: list[JsonValue] = []
    certified: set[EvidenceScope] = set()
    performance_scopes: set[EvidenceScope] = set()
    for evidence in gates:
        receipt = evidence.receipt
        if receipt.passed:
            if receipt.purpose is GatePurpose.PERFORMANCE:
                performance_scopes.add(receipt.scope)
            else:
                certified.add(receipt.scope)
        gate_records.append(
            {
                "attempt_id": evidence.attempt_id,
                "result_digest": evidence.result_digest,
                "gate_id": receipt.gate_id,
                "purpose": receipt.purpose.value,
                "scope": receipt.scope.value,
                "passed": receipt.passed,
                "product_rank_body": receipt.product_rank_body,
                "placements": [
                    {
                        "rank": placement.rank,
                        "node": placement.node,
                        "local_rank": placement.local_rank,
                    }
                    for placement in receipt.placements
                ],
            }
        )
    missing = [scope.value for scope in PRODUCTION_CERTIFICATION_LEVELS if scope not in certified]
    return {
        "planning_results": [
            {
                "attempt_id": evidence.attempt_id,
                "role": evidence.role.value,
                "outcome": evidence.outcome,
                "plan_digest": evidence.plan_digest,
                "attempt_input_digest": evidence.attempt_input_digest,
                "role_input_digest": evidence.role_input_digest,
                "result_digest": evidence.result_digest,
                "response_digest": evidence.response_digest,
                "receipt_path": evidence.receipt_path.as_posix(),
                "receipt_digest": evidence.receipt_digest,
                "review_of_attempt_id": evidence.review_of_attempt_id,
            }
            for evidence in planning
        ],
        "ingested_results": [
            {
                "attempt_id": attempt_id,
                "result_digest": ingested.result_digest,
                "receipt_path": receipt_path,
                "worker_status": ingested.manifest.status.value,
                "summary": _redact(ingested.manifest.summary),
                "files": [
                    {
                        "path": entry.path,
                        "sha256": entry.sha256,
                        "size_bytes": entry.size_bytes,
                    }
                    for entry in sorted(ingested.manifest.evidence, key=lambda item: item.path)
                ],
            }
            for attempt_id, (ingested, receipt_path) in sorted(results.items())
        ],
        "gate_receipts": gate_records,
        "certified_levels": [
            scope.value for scope in PRODUCTION_CERTIFICATION_LEVELS if scope in certified
        ],
        "performance_only_levels": [
            scope.value for scope in PRODUCTION_CERTIFICATION_LEVELS if scope in performance_scopes
        ],
        "missing_certification_levels": missing,
        "certification_complete": not missing,
        "claim_boundary": (
            "Only passed, non-performance typed gate receipts pinned to ingested result "
            "digests satisfy certification levels. Scheduler and worker status do not."
        ),
    }


def _render_markdown(payload: dict[str, JsonValue]) -> str:
    identity = _object(payload["identity"])
    target = _object(identity["target"])
    delivery = _object(payload["delivery_boundary"])
    build = _object(payload["build"])
    hierarchy = _object(payload["hierarchy"])
    evidence = _object(payload["evidence"])

    lines = [
        "# Staircase terminal report",
        "",
        f"- Outcome: `{_markdown(payload['outcome'])}`",
        f"- Reason: {_markdown(payload['terminal_reason'])}",
        f"- Run: `{_markdown(identity['run_id'])}`",
        f"- Workflow: `{_markdown(identity['workflow_mode'])}`",
        f"- Task digest: `{_markdown(identity['task_digest'])}`",
        f"- Base commit: `{_markdown(identity['base_commit'])}`",
        f"- Target: `{_markdown(target['family'])}/{_markdown(target['checkpoint_id'])}`",
        f"- Build identity: `{_markdown(build['identity'])}`",
        "",
        "## Delivery boundary",
        "",
        f"- Mode: `{_markdown(delivery['mode'])}`",
        f"- Branch: `{_markdown(delivery['branch'])}`",
        f"- Base commit: `{_markdown(delivery['base_commit'])}`",
        f"- Integration head: `{_markdown(delivery['integration_head'])}`",
        f"- Merge: `{_markdown(delivery['merge'])}`",
        f"- Push: `{_markdown(delivery['push'])}`",
    ]
    if "repository_status" in delivery:
        lines.append(f"- Repository status: `{_markdown(delivery['repository_status'])}`")
    if "patch" in delivery:
        patch = _object(delivery["patch"])
        lines.append(f"- Patch status: `{_markdown(patch['status'])}`")
        if patch["status"] == "verified":
            lines.extend(
                (
                    f"- Patch receipt: `{_markdown(patch['receipt_path'])}`",
                    f"- Patch path: `{_markdown(patch['path'])}`",
                    f"- Patch SHA256: `{_markdown(patch['sha256'])}`",
                    f"- Patch size: `{_markdown(patch['size_bytes'])}` bytes",
                    f"- Patch repository status: `{_markdown(patch['repository_status'])}`",
                )
            )
    lines.extend(
        (
            "",
            "## Certification",
            "",
            f"- Complete: `{_markdown(evidence['certification_complete'])}`",
            f"- Certified levels: {_markdown_list(evidence['certified_levels'])}",
            f"- Missing levels: {_markdown_list(evidence['missing_certification_levels'])}",
            f"- Performance-only levels: {_markdown_list(evidence['performance_only_levels'])}",
            f"- Boundary: {_markdown(evidence['claim_boundary'])}",
            "",
            "## Stages",
            "",
            "| Stage | Status | Required goals | Reason |",
            "| --- | --- | --- | --- |",
        )
    )
    for stage_value in _array(hierarchy["stages"]):
        stage = _object(stage_value)
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(stage["stage_id"]),
                    _markdown(stage["status"]),
                    _markdown_list(stage["required_goal_ids"]),
                    _markdown(stage["terminal_reason"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Goals",
            "",
            "| Goal | Stage | Status | Required items | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for goal_value in _array(hierarchy["goals"]):
        goal = _object(goal_value)
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(goal["goal_id"]),
                    _markdown(goal["stage_id"]),
                    _markdown(goal["status"]),
                    _markdown_list(goal["required_item_ids"]),
                    _markdown(goal["terminal_reason"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Work items",
            "",
            "| Item | Stage / Goal | Status | Candidate | Reviewer | Attempts |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item_value in _array(hierarchy["work_items"]):
        item = _object(item_value)
        attempt_text = ", ".join(
            f"{_markdown(_object(attempt)['attempt_id'])}:{_markdown(_object(attempt)['status'])}"
            for attempt in _array(item["attempts"])
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(item["item_id"]),
                    f"{_markdown(item['stage_id'])} / {_markdown(item['goal_id'])}",
                    _markdown(item["status"]),
                    _markdown(item["candidate_attempt_id"]),
                    _markdown(item["reviewer_attempt_id"]),
                    attempt_text or "none",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Planning attempts",
            "",
            "| Attempt | Role | Status | Result digest | Evidence ingested |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for attempt_value in _array(hierarchy["planning_attempts"]):
        attempt = _object(attempt_value)
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(attempt["attempt_id"]),
                    _markdown(attempt["role"]),
                    _markdown(attempt["status"]),
                    _markdown(attempt["result_digest"]),
                    _markdown(attempt["ingested_result_present"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Accepted planning evidence",
            "",
            "| Attempt | Role | Outcome | Plan digest | Result digest | Receipt digest |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result_value in _array(evidence["planning_results"]):
        result = _object(result_value)
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(result["attempt_id"]),
                    _markdown(result["role"]),
                    _markdown(result["outcome"]),
                    _markdown(result["plan_digest"]),
                    _markdown(result["result_digest"]),
                    _markdown(result["receipt_digest"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Exact scheduler jobs",
            "",
            "| Owner | Job | Observed status | Source | Reason |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for job_value in _array(payload["jobs"]):
        job = _object(job_value)
        observation = _object(job["observation"]) if job["observation"] is not None else {}
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{_markdown(job['owner_type'])}:{_markdown(job['owner_id'])}",
                    _markdown(job["scheduler_id"]),
                    _markdown(observation.get("status")),
                    _markdown(observation.get("source")),
                    _markdown(observation.get("reason")),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Ingested evidence",
            "",
            "| Attempt | Result digest | Worker status | Receipt | Files |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for result_value in _array(evidence["ingested_results"]):
        result = _object(result_value)
        files = ", ".join(_markdown(_object(entry)["path"]) for entry in _array(result["files"]))
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(result["attempt_id"]),
                    _markdown(result["result_digest"]),
                    _markdown(result["worker_status"]),
                    _markdown(result["receipt_path"]),
                    files or "none",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Gate receipts",
            "",
            "| Gate | Attempt | Purpose | Scope | Passed |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for gate_value in _array(evidence["gate_receipts"]):
        gate = _object(gate_value)
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown(gate["gate_id"]),
                    _markdown(gate["attempt_id"]),
                    _markdown(gate["purpose"]),
                    _markdown(gate["scope"]),
                    _markdown(gate["passed"]),
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _strict_json_object(content: bytes, *, label: str) -> dict[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for name, entry in pairs:
            if name in value:
                raise TerminalReportError(f"{label} contains duplicate keys")
            value[name] = entry
        return value

    try:
        decoded = content.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TerminalReportError(f"cannot parse {label}") from error
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        raise TerminalReportError(f"{label} must be a JSON object")
    return value


def _confined_regular_path(
    workspace: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> Path:
    current = workspace
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise TerminalReportError(
                f"{label} parent must be a confined regular directory: {current}"
            )
    path = current / relative.name
    if path.is_symlink() or not path.is_file():
        raise TerminalReportError(f"{label} must be a regular non-symlink file: {path}")
    try:
        if path.resolve(strict=True).relative_to(workspace) != Path(*relative.parts):
            raise TerminalReportError(f"{label} resolves outside its exact workspace path")
    except (OSError, ValueError) as error:
        raise TerminalReportError(f"{label} is outside the run workspace") from error
    return path


def _open_confined_regular_file(
    workspace: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> tuple[int, os.stat_result]:
    path = _confined_regular_path(workspace, relative, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        info = os.fstat(descriptor)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise TerminalReportError(f"cannot open {label}: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise TerminalReportError(f"{label} must be a regular file: {path}")
    return descriptor, info


def _stable_file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _read_confined_regular_file(
    workspace: Path,
    relative: PurePosixPath,
    *,
    label: str,
    maximum_bytes: int,
) -> bytes:
    descriptor, before = _open_confined_regular_file(workspace, relative, label=label)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            content = source.read(maximum_bytes + 1)
        after = os.fstat(descriptor)
    except OSError as error:
        raise TerminalReportError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)
    if len(content) > maximum_bytes:
        raise TerminalReportError(f"{label} exceeds {maximum_bytes} bytes")
    if _stable_file_identity(before) != _stable_file_identity(after):
        raise TerminalReportError(f"{label} changed while being verified")
    return content


def _digest_confined_regular_file(
    workspace: Path,
    relative: PurePosixPath,
    *,
    label: str,
) -> tuple[str, int]:
    descriptor, before = _open_confined_regular_file(workspace, relative, label=label)
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise TerminalReportError(f"cannot read {label}") from error
    finally:
        os.close(descriptor)
    if size_bytes != before.st_size or _stable_file_identity(before) != _stable_file_identity(
        after
    ):
        raise TerminalReportError(f"{label} changed while being verified")
    return digest.hexdigest(), size_bytes


def _canonical_workspace(workspace: Path) -> Path:
    try:
        root = workspace.resolve(strict=True)
    except OSError as error:
        raise TerminalReportError(f"workspace does not exist: {workspace}") from error
    if not root.is_dir():
        raise TerminalReportError(f"workspace is not a directory: {workspace}")
    return root


def _workspace_relative_file(workspace: Path, path: Path, *, label: str) -> PurePosixPath:
    try:
        resolved = path.resolve(strict=True)
        relative = resolved.relative_to(workspace)
    except (OSError, ValueError) as error:
        raise TerminalReportError(f"{label} is outside the run workspace: {path}") from error
    if path.is_symlink() or not resolved.is_file():
        raise TerminalReportError(f"{label} must be a regular non-symlink file: {path}")
    return PurePosixPath(relative.as_posix())


def _safe_relative_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise TerminalReportError(f"{label} must be a safe relative path")
    return path


def _prepare_output_directory(workspace: Path, relative: PurePosixPath) -> Path:
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise TerminalReportError(f"report directory cannot contain symlinks: {current}")
        if current.exists() and not current.is_dir():
            raise TerminalReportError(f"report directory component is not a directory: {current}")
        current.mkdir(exist_ok=True)
    try:
        current.resolve(strict=True).relative_to(workspace)
    except ValueError as error:
        raise TerminalReportError("report directory escapes the run workspace") from error
    return current


def _preflight_immutable(path: Path, content: str) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_file():
        raise ImmutableReportError(f"report target is not a regular file: {path}")
    if path.read_text(encoding="utf-8") != content:
        raise ImmutableReportError(f"refusing to replace different terminal report: {path}")


def _write_once(path: Path, content: str) -> None:
    if path.exists():
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            _preflight_immutable(path, content)
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _redact(value: str) -> str:
    redacted = _SENSITIVE_ASSIGNMENT.sub(
        lambda match: match.group(1) + match.group(2) + "<redacted>", value
    )
    redacted = _BEARER_VALUE.sub("Bearer <redacted>", redacted)
    return _URL_CREDENTIALS.sub(r"\g<scheme><redacted>@", redacted)


def _markdown(value: JsonValue | object) -> str:
    if value is None or value == "":
        return "none"
    if isinstance(value, bool):
        text = str(value).lower()
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _markdown_list(value: JsonValue) -> str:
    entries = _array(value)
    if not entries:
        return "none"
    return ", ".join(f"`{_markdown(entry)}`" for entry in entries)


def _object(value: JsonValue) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise TerminalReportError("internal report rendering expected an object")
    return value


def _array(value: JsonValue) -> list[JsonValue]:
    if not isinstance(value, list):
        raise TerminalReportError("internal report rendering expected an array")
    return value


__all__ = [
    "GateEvidence",
    "ImmutableReportError",
    "JSON_REPORT_FILENAME",
    "MARKDOWN_REPORT_FILENAME",
    "PRODUCTION_CERTIFICATION_LEVELS",
    "PlanningEvidence",
    "TerminalReport",
    "TerminalReportError",
    "WrittenTerminalReport",
    "build_terminal_report",
    "write_terminal_report",
]
