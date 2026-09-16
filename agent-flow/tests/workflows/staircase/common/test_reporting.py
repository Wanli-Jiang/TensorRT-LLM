# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic, evidence-bounded Staircase terminal reports."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.artifacts import (
    EvidenceFile,
    IngestedResult,
    WorkerResultManifest,
    WorkerResultStatus,
)
from agent_flow.workflows.staircase.common.gates import (
    EvidenceScope,
    GatePurpose,
    GateReceipt,
    RankPlacement,
)
from agent_flow.workflows.staircase.common.reporting import (
    PLANNING_RECEIPT_SCHEMA_VERSION,
    GateEvidence,
    ImmutableReportError,
    PlanningEvidence,
    TerminalReportError,
    build_terminal_report,
    write_terminal_report,
)
from agent_flow.workflows.staircase.common.slurm import (
    JobIdentity,
    JobObservation,
    JobStatus,
    ObservationSource,
)
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    ControllerBootstrapStatus,
    DomainProfile,
    GoalRecord,
    HierarchyStatus,
    IntegrationEvidenceKind,
    IntegrationRecord,
    JobReference,
    Role,
    RunState,
    RunTerminalStatus,
    StageRecord,
    StageStatus,
    WorkItemKind,
    WorkItemRecord,
    WorkItemStatus,
)
from agent_flow.workflows.staircase.task_schema import (
    AccuracyGate,
    AgentExecutionConfig,
    CertificationConfig,
    CertificationMode,
    ContainerLaunchMode,
    ControllerConfig,
    DeliveryConfig,
    ExecutionConfig,
    GatesConfig,
    MountConfig,
    NormalizedTask,
    OverrideBounds,
    ParallelMapping,
    ReferenceConfig,
    RepositoryConfig,
    ResourceClass,
    RetryPolicy,
    SlurmConfig,
    SmithConfig,
    TargetConfig,
)

TASK_DIGEST = "a" * 64
CODER_RESULT_DIGEST = "b" * 64
REVIEWER_RESULT_DIGEST = "c" * 64
CANDIDATE_DIGEST = "d" * 64
INPUT_DIGEST = "e" * 64


def _git(tmp_path: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(tmp_path), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _delivery_commits(tmp_path: Path) -> tuple[str, str]:
    repository = tmp_path / "repo"
    if not (repository / ".git").is_dir():
        repository.mkdir()
        _git(repository, "init", "-q")
        _git(repository, "config", "user.name", "Staircase Test")
        _git(repository, "config", "user.email", "staircase-test@example.com")
        (repository / "model.py").write_text("BASE = True\n", encoding="utf-8")
        _git(repository, "add", "model.py")
        _git(repository, "commit", "-q", "-m", "base")
        (repository / "model.py").write_text("BASE = True\nINTEGRATED = True\n", encoding="utf-8")
        _git(repository, "add", "model.py")
        _git(repository, "commit", "-q", "-m", "integrated")
    commits = _git(repository, "rev-list", "--reverse", "HEAD").splitlines()
    return commits[0], commits[-1]


def _task(tmp_path: Path) -> NormalizedTask:
    base_commit, _head_commit = _delivery_commits(tmp_path)
    resource = ResourceClass(
        name="coder_analysis",
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        cpus_per_task=1,
        memory_mib=1024,
        time_limit_seconds=60,
    )
    return NormalizedTask(
        schema_version=1,
        repository=RepositoryConfig(
            root=tmp_path / "repo",
            base_commit=base_commit,
            dirty_policy="reject",
            workspace_root=tmp_path,
        ),
        reference=ReferenceConfig(
            checkpoint=tmp_path / "checkpoint",
            provenance="example/model@revision",
            architecture="ExampleForCausalLM",
            additional_sources=(),
        ),
        target=TargetConfig(
            family="example",
            checkpoint_id="example_sm100",
            sm=100,
            world_size=1,
            mapping=ParallelMapping(1, 1, 1, 1, 1),
            features=(),
            expected_route="tensorrt_llm/_torch/modeling_v2/models/example",
            synthetic_target=False,
        ),
        gates=GatesConfig(
            accuracy=AccuracyGate("tests/accuracy.py::test_gsm8k", "reference", "greedy", 0.0),
            feature_signal_tests=(),
            boot_tests=("tests/boot.py::test_model",),
            component_tests=("tests/component.py::test_model",),
            collective_tests=(),
        ),
        certification=CertificationConfig(CertificationMode.LOCAL, None),
        execution=ExecutionConfig(
            mode="slurm",
            slurm=SlurmConfig(
                controller=ControllerConfig(
                    account="account",
                    partition="partition",
                    qos=None,
                    reservation=None,
                    time_limit_seconds=60,
                    cpus_per_task=1,
                    memory_mib=1024,
                    image=tmp_path / "image.sqsh",
                    scheduler_clients=True,
                    container_launch_mode=ContainerLaunchMode.IN_ALLOCATION_SRUN,
                    mounts=(MountConfig(tmp_path, str(tmp_path), False),),
                    environment=(("PYTHONNOUSERSITE", "1"),),
                    build_identity="local-dev-deadbeef",
                    dispatch_mode="nested_submission",
                    requeue=True,
                    advance_signal_lead_seconds=30,
                    heartbeat_timeout_seconds=30,
                    lease_timeout_seconds=90,
                    orphan_grace_seconds=120,
                ),
                smith=SmithConfig(
                    max_parallel_items=1,
                    max_nodes_total=1,
                    max_gpus_total=1,
                    distinct_nodes_required=False,
                    exclusive=False,
                    resource_classes=(resource,),
                    per_item_override_bounds=OverrideBounds(1, 1, 1, 1, 1024, 60),
                    retry_policy=RetryPolicy(1, 1),
                ),
            ),
            agent=AgentExecutionConfig("codex", "test-model"),
        ),
        delivery=DeliveryConfig(mode="diff_only", branch=None, merge=False, push=False),
        digest=TASK_DIGEST,
    )


def _state(
    tmp_path: Path,
    *,
    terminal_status: RunTerminalStatus = RunTerminalStatus.SUCCEEDED,
) -> RunState:
    base_commit, integration_head = _delivery_commits(tmp_path)
    coder = AttemptRecord(
        attempt_id="coder-1",
        item_id="item-1",
        sequence=1,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        submission_token="submit-coder",
        job=JobReference("101"),
        result_digest=CODER_RESULT_DIGEST,
        candidate_digest=CANDIDATE_DIGEST,
    )
    reviewer = AttemptRecord(
        attempt_id="reviewer-2",
        item_id="item-1",
        sequence=2,
        role=Role.REVIEWER,
        kind=AttemptKind.REVIEWER_RERUN,
        generation=1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.SMITH,
        submission_token="submit-reviewer",
        job=JobReference("102", cluster="cluster-a"),
        result_digest=REVIEWER_RESULT_DIGEST,
        review_of_attempt_id="coder-1",
        reviewed_candidate_digest=CANDIDATE_DIGEST,
    )
    item = WorkItemRecord(
        item_id="item-1",
        stage_id="stage-1",
        goal_id="goal-1",
        kind=WorkItemKind.CATALOG_ONBOARD,
        profile=DomainProfile.SMITH,
        status=WorkItemStatus.INTEGRATED,
        attempts=(coder, reviewer),
        candidate_attempt_id="coder-1",
        candidate_digest=CANDIDATE_DIGEST,
        reviewer_attempt_id="reviewer-2",
    )
    return RunState(
        run_id="run-1",
        task_digest=TASK_DIGEST,
        base_commit=base_commit,
        generation=1,
        revision=9,
        stages=(StageRecord("stage-1", ("goal-1",), status=StageStatus.CLOSED),),
        goals=(
            GoalRecord(
                "goal-1",
                "stage-1",
                ("item-1",),
                status=HierarchyStatus.SUCCEEDED,
            ),
        ),
        items=(item,),
        integration_history=(
            IntegrationRecord(
                1,
                1,
                "item-1",
                base_commit,
                integration_head,
                IntegrationEvidenceKind.CANDIDATE,
                CANDIDATE_DIGEST,
            ),
        ),
        controller_bootstrap_status=ControllerBootstrapStatus.SUBMITTED,
        controller_submission_token="controller-token",
        controller_job=JobReference("100"),
        terminal_status=terminal_status,
        terminal_reason="all required state transitions closed",
    )


def _write_patch_delivery(
    workspace: Path,
    state: RunState,
    *,
    patch: bytes | None = None,
) -> dict[str, object]:
    base_commit, integration_head = _delivery_commits(workspace)
    delivery = workspace / "delivery"
    delivery.mkdir(exist_ok=True)
    checkout = delivery / "diff-repository"
    if not checkout.exists():
        subprocess.run(
            ("git", "clone", "-q", str(workspace / "repo"), str(checkout)),
            check=True,
        )
        _git(checkout, "checkout", "-q", "--detach", integration_head)
    if patch is None:
        patch = subprocess.run(
            (
                "git",
                "-C",
                str(checkout),
                "diff",
                "--binary",
                "--full-index",
                base_commit,
                integration_head,
            ),
            check=True,
            capture_output=True,
        ).stdout
    patch_path = delivery / "staircase.patch"
    patch_path.write_bytes(patch)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "run_id": state.run_id,
        "task_digest": state.task_digest,
        "base_commit": state.base_commit,
        "head_commit": state.integration_head,
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "size_bytes": len(patch),
        "path": "delivery/staircase.patch",
    }
    (delivery / "patch-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )
    return receipt


def _ingested_reviewer(workspace: Path) -> IngestedResult:
    manifest = WorkerResultManifest(
        run_id="run-1",
        item_id="item-1",
        attempt_id="reviewer-2",
        task_digest=TASK_DIGEST,
        generation=1,
        input_digest=INPUT_DIGEST,
        status=WorkerResultStatus.SUCCEEDED,
        summary="reviewed immutable candidate",
        evidence=(EvidenceFile("logs/reviewer.txt", "f" * 64, 42),),
        reviewed_candidate_digest=CANDIDATE_DIGEST,
    )
    receipt_path = workspace / "receipts" / "run-1" / "item-1" / "reviewer-2.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": manifest.run_id,
                "item_id": manifest.item_id,
                "attempt_id": manifest.attempt_id,
                "generation": manifest.generation,
                "task_digest": manifest.task_digest,
                "input_digest": manifest.input_digest,
                "result_digest": REVIEWER_RESULT_DIGEST,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return IngestedResult(manifest, REVIEWER_RESULT_DIGEST, receipt_path)


def _gate(
    scope: EvidenceScope,
    purpose: GatePurpose,
    *,
    gate_id: str,
) -> GateEvidence:
    if scope is EvidenceScope.CPU_STATIC:
        placements = ()
        product_rank_body = False
    elif scope is EvidenceScope.SINGLE_GPU_PRODUCT:
        placements = (RankPlacement(0, "node-a", 0),)
        product_rank_body = True
    elif scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
        placements = tuple(RankPlacement(rank, "node-a", rank) for rank in range(4))
        product_rank_body = True
    elif scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER:
        placements = (RankPlacement(0, "node-a", 0), RankPlacement(1, "node-b", 0))
        product_rank_body = False
    else:
        placements = (RankPlacement(0, "node-a", 0), RankPlacement(1, "node-b", 0))
        product_rank_body = True
    return GateEvidence(
        attempt_id="reviewer-2",
        result_digest=REVIEWER_RESULT_DIGEST,
        receipt=GateReceipt(
            gate_id=gate_id,
            purpose=purpose,
            scope=scope,
            placements=placements,
            product_rank_body=product_rank_body,
            passed=True,
        ),
    )


def _planning_pair(
    workspace: Path,
    state: RunState,
) -> tuple[RunState, tuple[PlanningEvidence, PlanningEvidence]]:
    draft_digest = "6" * 64
    review_digest = "7" * 64
    plan_digest = "8" * 64
    draft = AttemptRecord(
        attempt_id="plan-draft-0001",
        item_id="__planning__",
        sequence=1,
        role=Role.PLAN_DRAFTER,
        kind=AttemptKind.ROLE,
        generation=1,
        status=AttemptStatus.VALIDATED,
        submission_token="submit-plan-draft",
        job=JobReference("201"),
        result_digest=draft_digest,
        candidate_digest=plan_digest,
    )
    review = AttemptRecord(
        attempt_id="plan-review-0002",
        item_id="__planning__",
        sequence=2,
        role=Role.PLAN_REVIEWER,
        kind=AttemptKind.REVIEWER_ANALYSIS,
        generation=1,
        status=AttemptStatus.VALIDATED,
        submission_token="submit-plan-review",
        job=JobReference("202"),
        result_digest=review_digest,
        candidate_digest=plan_digest,
        review_of_attempt_id=draft.attempt_id,
        reviewed_candidate_digest=plan_digest,
    )
    state = replace(state, planning_attempts=(draft, review))
    result: list[PlanningEvidence] = []
    for attempt, outcome, review_of, result_digest in (
        (draft, "DRAFTED", None, draft_digest),
        (review, "ACCEPT", draft.attempt_id, review_digest),
    ):
        receipt_path = (
            workspace / "receipts" / state.run_id / "__planning__" / f"{attempt.attempt_id}.json"
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": PLANNING_RECEIPT_SCHEMA_VERSION,
            "run_id": state.run_id,
            "task_digest": state.task_digest,
            "item_id": "__planning__",
            "attempt_id": attempt.attempt_id,
            "generation": attempt.generation,
            "role": attempt.role.value,
            "outcome": outcome,
            "plan_digest": plan_digest,
            "attempt_input_digest": "9" * 64,
            "role_input_digest": "a" * 64,
            "result_digest": result_digest,
            "response_digest": "f" * 64,
            "review_of_attempt_id": review_of,
        }
        receipt_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result.append(
            PlanningEvidence(
                attempt_id=attempt.attempt_id,
                role=attempt.role,
                outcome=outcome,
                plan_digest=plan_digest,
                attempt_input_digest="9" * 64,
                role_input_digest="a" * 64,
                result_digest=result_digest,
                response_digest="f" * 64,
                receipt_path=receipt_path,
                receipt_digest=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                review_of_attempt_id=review_of,
            )
        )
    return state, (result[0], result[1])


def test_report_is_deterministic_complete_and_write_once(tmp_path: Path) -> None:
    state = _state(tmp_path)
    task = _task(tmp_path)
    patch_receipt = _write_patch_delivery(tmp_path, state)
    ingested = _ingested_reviewer(tmp_path)
    gate_evidence = (
        _gate(
            EvidenceScope.SINGLE_GPU_PRODUCT,
            GatePurpose.CORRECTNESS,
            gate_id="single_gpu",
        ),
        _gate(
            EvidenceScope.REAL_MULTI_NODE_PRODUCT,
            GatePurpose.PERFORMANCE,
            gate_id="throughput",
        ),
    )
    observations = (
        JobObservation(
            JobIdentity("102", cluster="cluster-a"),
            JobStatus.COMPLETED,
            "completed",
            ObservationSource.ACCOUNTING,
            "COMPLETED",
        ),
    )

    first = build_terminal_report(
        state,
        task,
        workspace=tmp_path,
        ingested_results=(ingested,),
        gate_evidence=gate_evidence,
        scheduler_observations=observations,
    )
    second = build_terminal_report(
        state,
        task,
        workspace=tmp_path,
        ingested_results=(ingested,),
        gate_evidence=tuple(reversed(gate_evidence)),
        scheduler_observations=observations,
    )
    assert first == second

    payload = json.loads(first.json_text)
    assert payload["outcome"] == "succeeded"
    assert payload["build"]["identity"] == "local-dev-deadbeef"
    assert payload["delivery_boundary"] == {
        "base_commit": state.base_commit,
        "branch": None,
        "integration_head": state.integration_head,
        "merge": False,
        "mode": "diff_only",
        "patch": {
            "path": "delivery/staircase.patch",
            "receipt_path": "delivery/patch-receipt.json",
            "sha256": patch_receipt["patch_sha256"],
            "size_bytes": patch_receipt["size_bytes"],
            "status": "verified",
            "repository_status": "verified",
        },
        "push": False,
    }
    assert f"- Integration head: `{state.integration_head}`" in first.markdown_text
    assert "- Patch status: `verified`" in first.markdown_text
    work_item = payload["hierarchy"]["work_items"][0]
    assert work_item["candidate_attempt_id"] == "coder-1"
    assert work_item["reviewer_attempt_id"] == "reviewer-2"
    reviewer_job = next(job for job in payload["jobs"] if job["owner_id"] == "reviewer-2")
    assert reviewer_job["observation"]["status"] == "COMPLETED"
    assert payload["evidence"]["certified_levels"] == ["single_gpu_product"]
    assert "real_multi_node_product" in payload["evidence"]["missing_certification_levels"]
    assert payload["evidence"]["performance_only_levels"] == ["real_multi_node_product"]
    assert payload["evidence"]["ingested_results"][0]["receipt_path"].startswith("receipts/")

    written = write_terminal_report(tmp_path, first)
    assert written.json_path == "reports/terminal-report.json"
    assert write_terminal_report(tmp_path, first) == written
    different = build_terminal_report(
        replace(state, terminal_reason="a different terminal reason"),
        task,
        workspace=tmp_path,
        ingested_results=(ingested,),
    )
    with pytest.raises(ImmutableReportError, match="different terminal report"):
        write_terminal_report(tmp_path, different)


@pytest.mark.parametrize(
    "status",
    [
        RunTerminalStatus.SUCCEEDED,
        RunTerminalStatus.BLOCKED,
        RunTerminalStatus.BLOCKED_INPUT,
        RunTerminalStatus.EXHAUSTED,
        RunTerminalStatus.FAILED,
        RunTerminalStatus.INTERRUPTED,
        RunTerminalStatus.CANCELLED,
    ],
)
def test_all_terminal_outcomes_are_reported_verbatim(
    tmp_path: Path, status: RunTerminalStatus
) -> None:
    state = _state(tmp_path, terminal_status=status)
    if status is RunTerminalStatus.SUCCEEDED:
        _write_patch_delivery(tmp_path, state)
    report = build_terminal_report(state, _task(tmp_path), workspace=tmp_path)
    payload = json.loads(report.json_text)
    assert payload["outcome"] == status.value
    assert payload["delivery_boundary"]["patch"]["status"] == (
        "verified" if status is RunTerminalStatus.SUCCEEDED else "missing"
    )


def test_successful_diff_only_requires_complete_delivery_evidence(tmp_path: Path) -> None:
    state = _state(tmp_path)
    task = _task(tmp_path)
    with pytest.raises(TerminalReportError, match="lacks its patch receipt and artifact"):
        build_terminal_report(state, task, workspace=tmp_path)

    delivery = tmp_path / "delivery"
    delivery.mkdir()
    (delivery / "staircase.patch").write_bytes(b"orphan patch")
    with pytest.raises(TerminalReportError, match="incomplete patch artifact pair"):
        build_terminal_report(state, task, workspace=tmp_path)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", 2),
        ("run_id", "another-run"),
        ("task_digest", "9" * 64),
        ("base_commit", "3" * 40),
        ("head_commit", "4" * 40),
        ("patch_sha256", "5" * 64),
        ("size_bytes", 999),
        ("path", "../escaped.patch"),
    ],
)
def test_delivery_receipt_identity_and_bytes_are_exact(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    state = _state(tmp_path)
    receipt = _write_patch_delivery(tmp_path, state)
    receipt[field] = replacement
    (tmp_path / "delivery" / "patch-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(TerminalReportError, match="receipt differs"):
        build_terminal_report(state, _task(tmp_path), workspace=tmp_path)


def test_delivery_receipt_rejects_unknown_duplicate_and_tampered_evidence(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    receipt = _write_patch_delivery(tmp_path, state)
    receipt["unexpected"] = "not allowed"
    receipt_path = tmp_path / "delivery" / "patch-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(TerminalReportError, match="receipt differs"):
        build_terminal_report(state, _task(tmp_path), workspace=tmp_path)

    receipt.pop("unexpected")
    duplicate = json.dumps(receipt)[:-1] + ', "run_id": "run-1"}'
    receipt_path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(TerminalReportError, match="duplicate keys"):
        build_terminal_report(state, _task(tmp_path), workspace=tmp_path)

    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (tmp_path / "delivery" / "staircase.patch").write_bytes(b"tampered after receipt")
    with pytest.raises(TerminalReportError, match="receipt differs"):
        build_terminal_report(state, _task(tmp_path), workspace=tmp_path)


def test_delivery_patch_receipt_cannot_bless_different_git_semantics(tmp_path: Path) -> None:
    state = _state(tmp_path)
    receipt = _write_patch_delivery(tmp_path, state)
    substituted = b"diff --git a/other.py b/other.py\n"
    (tmp_path / "delivery" / "staircase.patch").write_bytes(substituted)
    receipt["patch_sha256"] = hashlib.sha256(substituted).hexdigest()
    receipt["size_bytes"] = len(substituted)
    (tmp_path / "delivery" / "patch-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(TerminalReportError, match="exact base-to-integration-head diff"):
        build_terminal_report(state, _task(tmp_path), workspace=tmp_path)


def test_delivery_checkout_head_identity_is_reverified(tmp_path: Path) -> None:
    state = _state(tmp_path)
    _write_patch_delivery(tmp_path, state)
    checkout = tmp_path / "delivery" / "diff-repository"
    _git(checkout, "checkout", "-q", "--detach", state.base_commit)

    with pytest.raises(TerminalReportError, match="HEAD differs"):
        build_terminal_report(state, _task(tmp_path), workspace=tmp_path)


@pytest.mark.parametrize("artifact", ["patch-receipt.json", "staircase.patch"])
def test_delivery_artifacts_reject_symlinks(tmp_path: Path, artifact: str) -> None:
    state = _state(tmp_path)
    _write_patch_delivery(tmp_path, state)
    target = tmp_path / "delivery" / artifact
    outside = tmp_path / f"outside-{artifact}"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)

    with pytest.raises(TerminalReportError, match="non-symlink"):
        build_terminal_report(state, _task(tmp_path), workspace=tmp_path)


def test_signed_commit_delivery_reports_branch_and_head_without_patch_fields(
    tmp_path: Path,
) -> None:
    state = _state(tmp_path)
    task = replace(
        _task(tmp_path),
        delivery=DeliveryConfig(
            mode="signed_commits",
            branch="staircase/run-1",
            merge=False,
            push=False,
        ),
    )
    checkout = tmp_path / "delivery" / "repository"
    checkout.parent.mkdir(exist_ok=True)
    subprocess.run(
        ("git", "clone", "-q", str(tmp_path / "repo"), str(checkout)),
        check=True,
    )
    _git(checkout, "checkout", "-q", "-b", "staircase/run-1", state.integration_head)

    report = build_terminal_report(state, task, workspace=tmp_path)
    delivery = json.loads(report.json_text)["delivery_boundary"]

    assert delivery == {
        "base_commit": state.base_commit,
        "branch": "staircase/run-1",
        "integration_head": state.integration_head,
        "repository_status": "verified",
        "merge": False,
        "mode": "signed_commits",
        "push": False,
    }
    assert "Patch path" not in report.markdown_text
    assert "- Branch: `staircase/run-1`" in report.markdown_text
    assert f"- Integration head: `{state.integration_head}`" in report.markdown_text


def test_failed_signed_commit_delivery_reports_missing_checkout(tmp_path: Path) -> None:
    state = _state(tmp_path, terminal_status=RunTerminalStatus.FAILED)
    task = replace(
        _task(tmp_path),
        delivery=DeliveryConfig(
            mode="signed_commits",
            branch="staircase/run-1",
            merge=False,
            push=False,
        ),
    )

    report = build_terminal_report(state, task, workspace=tmp_path)

    assert json.loads(report.json_text)["delivery_boundary"]["repository_status"] == "missing"


def test_active_state_and_identity_mismatch_are_rejected(tmp_path: Path) -> None:
    state = _state(tmp_path)
    with pytest.raises(TerminalReportError, match="active state"):
        build_terminal_report(
            replace(state, terminal_status=RunTerminalStatus.ACTIVE, terminal_reason=None),
            _task(tmp_path),
            workspace=tmp_path,
        )
    with pytest.raises(TerminalReportError, match="digest"):
        build_terminal_report(
            state,
            replace(_task(tmp_path), digest="9" * 64),
            workspace=tmp_path,
        )


def test_gate_must_be_pinned_to_ingested_result(tmp_path: Path) -> None:
    with pytest.raises(TerminalReportError, match="not pinned"):
        build_terminal_report(
            _state(tmp_path),
            _task(tmp_path),
            workspace=tmp_path,
            gate_evidence=(
                _gate(
                    EvidenceScope.CPU_STATIC,
                    GatePurpose.STRUCTURE,
                    gate_id="static_contract",
                ),
            ),
        )


def test_accepted_planning_evidence_is_deterministic_and_digest_bound(
    tmp_path: Path,
) -> None:
    state, planning = _planning_pair(tmp_path, _state(tmp_path))
    _write_patch_delivery(tmp_path, state)

    first = build_terminal_report(
        state,
        _task(tmp_path),
        workspace=tmp_path,
        planning_evidence=planning,
    )
    second = build_terminal_report(
        state,
        _task(tmp_path),
        workspace=tmp_path,
        planning_evidence=tuple(reversed(planning)),
    )

    assert first == second
    records = json.loads(first.json_text)["evidence"]["planning_results"]
    assert [record["role"] for record in records] == ["plan_drafter", "plan_reviewer"]
    assert [record["outcome"] for record in records] == ["DRAFTED", "ACCEPT"]
    assert all(record["receipt_path"].startswith("receipts/") for record in records)
    planning_attempts = json.loads(first.json_text)["hierarchy"]["planning_attempts"]
    assert all(attempt["ingested_result_present"] for attempt in planning_attempts)
    assert "## Accepted planning evidence" in first.markdown_text
    assert planning[0].receipt_digest in first.markdown_text


def test_admitted_planning_evidence_missing_tampered_or_mismatched_fails_closed(
    tmp_path: Path,
) -> None:
    state, planning = _planning_pair(tmp_path, _state(tmp_path))
    _write_patch_delivery(tmp_path, state)
    task = _task(tmp_path)

    with pytest.raises(TerminalReportError, match="lacks accepted planning evidence"):
        build_terminal_report(state, task, workspace=tmp_path)

    planning[0].receipt_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(TerminalReportError, match="receipt digest mismatch"):
        build_terminal_report(
            state,
            task,
            workspace=tmp_path,
            planning_evidence=planning,
        )

    state, planning = _planning_pair(tmp_path, state)
    mismatched = replace(planning[1], plan_digest="b" * 64)
    with pytest.raises(TerminalReportError, match="mismatched plan digests"):
        build_terminal_report(
            state,
            task,
            workspace=tmp_path,
            planning_evidence=(planning[0], mismatched),
        )


def test_receipt_and_report_paths_cannot_escape_workspace(tmp_path: Path) -> None:
    ingested = _ingested_reviewer(tmp_path)
    outside = tmp_path.parent / "outside-receipt.json"
    outside.write_text(ingested.receipt_path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(TerminalReportError, match="outside"):
        build_terminal_report(
            _state(tmp_path),
            _task(tmp_path),
            workspace=tmp_path,
            ingested_results=(replace(ingested, receipt_path=outside),),
        )

    state = _state(tmp_path)
    _write_patch_delivery(tmp_path, state)
    report = build_terminal_report(state, _task(tmp_path), workspace=tmp_path)
    with pytest.raises(TerminalReportError, match="safe relative"):
        write_terminal_report(tmp_path, report, relative_directory="../escape")


def test_free_form_secrets_are_redacted_and_unknown_jobs_rejected(tmp_path: Path) -> None:
    state = replace(
        _state(tmp_path, terminal_status=RunTerminalStatus.FAILED),
        terminal_reason="TOKEN=top-secret Bearer bearer-secret",
    )
    report = build_terminal_report(state, _task(tmp_path), workspace=tmp_path)
    assert "top-secret" not in report.json_text
    assert "bearer-secret" not in report.markdown_text
    assert "<redacted>" in report.json_text

    observation = JobObservation(
        JobIdentity("999"),
        JobStatus.COMPLETED,
        None,
        ObservationSource.ACCOUNTING,
    )
    with pytest.raises(TerminalReportError, match="not for an owned"):
        build_terminal_report(
            state,
            _task(tmp_path),
            workspace=tmp_path,
            scheduler_observations=(observation,),
        )
