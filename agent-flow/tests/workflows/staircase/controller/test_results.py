# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict worker results and restart-safe candidate receipts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.artifacts import (
    EvidenceFile,
    WorkerResultManifest,
    WorkerResultStatus,
)
from agent_flow.workflows.staircase.common.gates import (
    GateCommand,
    GatePhase,
    GatePurpose,
    GateSpec,
)
from agent_flow.workflows.staircase.common.gitops import ControllerGitOps
from agent_flow.workflows.staircase.common.isolation import digest_metadata_free_tree
from agent_flow.workflows.staircase.common.policy import ExecutionShape, WorkItemProposal
from agent_flow.workflows.staircase.common.policy import WorkItemKind as PolicyWorkItemKind
from agent_flow.workflows.staircase.controller.results import (
    CANDIDATE_RECEIPT_FILENAME,
    CandidateDisposition,
    CandidateReceipt,
    CandidateReceiptError,
    QaVerdict,
    ResultContractError,
    ReviewerVerdict,
    bind_coder_candidate,
    bind_tuner_candidate,
    decode_coder_result,
    decode_gate_result,
    decode_qa_result,
    decode_resource_escalation,
    decode_reviewer_result,
    decode_tuner_result,
    load_candidate_receipt,
    reconstruct_candidate,
)
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    JobReference,
    Role,
)
from agent_flow.workflows.staircase.tuning import artifacts as tuning_artifacts
from agent_flow.workflows.staircase.tuning.contracts import (
    KnobChange,
    KnobKind,
    MetricDirection,
    TuningHypothesis,
    UncertaintyRule,
)
from agent_flow.workflows.staircase.tuning.workflow import (
    evaluate_tuning_campaign,
    new_tuning_campaign,
    record_tuning_evidence,
)

_TASK_DIGEST = "a" * 64
_INPUT_DIGEST = "b" * 64
_RESULT_DIGEST = "c" * 64


def _git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def _repository(tmp_path: Path) -> tuple[Path, Path, str, ControllerGitOps]:
    repository = tmp_path / "repository"
    wrapper = (
        repository / "tensorrt_llm" / "_torch" / "modeling_v2" / "catalog" / "norm" / "alpha.py"
    )
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("VALUE = 0\n", encoding="utf-8")
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Staircase Tests")
    _git(repository, "config", "user.email", "staircase@example.com")
    _git(repository, "add", "--", ".")
    _git(repository, "commit", "-q", "-s", "-m", "base")
    base = _git(repository, "rev-parse", "HEAD")
    controller = ControllerGitOps(repository, tmp_path / "git.lock", "results-test")
    worktree = tmp_path / "workspace" / "candidates" / "catalog-item" / "catalog-item.0001.role"
    with controller.transaction() as transaction:
        transaction.create_candidate_overlay(worktree, base_commit=base)
    assert not (worktree / ".git").exists()
    return repository, worktree, base, controller


def _proposal(*, modifies_files: bool = True) -> WorkItemProposal:
    return WorkItemProposal(
        item_id="catalog-item",
        goal_id="catalog-goal",
        kind=(
            PolicyWorkItemKind.CATALOG_ONBOARD
            if modifies_files
            else PolicyWorkItemKind.CATALOG_VERIFY
        ),
        resource_class="coder_analysis",
        execution=ExecutionShape(1, 1, 0),
        modifies_files=modifies_files,
        entry_ids=("alpha",),
        allowed_paths=(
            ("tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py",) if modifies_files else ()
        ),
    )


def _attempt(
    role: Role = Role.CODER,
    kind: AttemptKind = AttemptKind.ROLE,
    *,
    attempt_id: str = "catalog-item.0001.role",
    sequence: int = 1,
    candidate: CandidateReceipt | None = None,
    profile: DomainProfile = DomainProfile.SMITH,
) -> AttemptRecord:
    review_fields: dict[str, object] = {}
    if kind in {AttemptKind.REVIEWER_ANALYSIS, AttemptKind.REVIEWER_RERUN}:
        assert candidate is not None
        review_fields = {
            "review_of_attempt_id": candidate.attempt_id,
            "reviewed_candidate_digest": candidate.candidate_digest,
        }
    return AttemptRecord(
        attempt_id,
        "catalog-item",
        sequence,
        role,
        kind,
        1,
        status=AttemptStatus.VALIDATED,
        profile=profile,
        resource_class="coder_analysis" if role is Role.CODER else None,
        submission_token=f"token-{sequence}",
        job=JobReference(str(100 + sequence)),
        result_digest=_RESULT_DIGEST,
        **review_fields,
    )


def _index_delta() -> dict[str, object]:
    return {
        "item_id": "catalog-item",
        "row": {
            "entry_id": "alpha",
            "path": "norm/alpha.py",
            "implementation": "torch.ops.trtllm.alpha",
            "summary": "Alpha contract",
        },
        "certification_cells": [],
        "expected_row_sha256": None,
    }


def _coder_manifest(
    *,
    payload: dict[str, object] | None = None,
    candidate_digest: str | None = "d" * 64,
) -> WorkerResultManifest:
    return WorkerResultManifest(
        "run-1",
        "catalog-item",
        "catalog-item.0001.role",
        _TASK_DIGEST,
        1,
        _INPUT_DIGEST,
        WorkerResultStatus.SUCCEEDED,
        "Coder completed",
        candidate_digest=candidate_digest,
        payload=(
            payload
            if payload is not None
            else {
                "schema_version": 1,
                "result_kind": "coder",
                "changed_paths": ["tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"],
                "index_delta": _index_delta(),
            }
        ),
    )


def _scanned_coder_manifest(
    worktree: Path,
    *,
    payload: dict[str, object] | None = None,
) -> WorkerResultManifest:
    return _coder_manifest(
        payload=payload,
        candidate_digest=digest_metadata_free_tree(worktree),
    )


def _fake_candidate() -> CandidateReceipt:
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "item_id": "catalog-item",
        "attempt_id": "catalog-item.0001.role",
        "disposition": "patch",
        "base_commit": "1" * 40,
        "candidate_commit": "2" * 40,
        "candidate_digest": "d" * 64,
        "changed_paths": ["tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"],
        "index_delta": None,
    }
    receipt_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CandidateReceipt(
        "run-1",
        "catalog-item",
        "catalog-item.0001.role",
        CandidateDisposition.PATCH,
        "1" * 40,
        "2" * 40,
        "d" * 64,
        ("tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py",),
        None,
        receipt_digest,
    )


def _evidence() -> tuple[EvidenceFile, ...]:
    return (EvidenceFile("evidence/result.json", "f" * 64, 17),)


def _evidence_payload() -> list[dict[str, object]]:
    return [
        {
            "path": "evidence/result.json",
            "sha256": "f" * 64,
            "size_bytes": 17,
        }
    ]


def _review_manifest(
    candidate: CandidateReceipt,
    *,
    kind: str,
    verdict: str,
    status: WorkerResultStatus,
    attempt_id: str,
) -> WorkerResultManifest:
    return WorkerResultManifest(
        "run-1",
        "catalog-item",
        attempt_id,
        _TASK_DIGEST,
        1,
        _INPUT_DIGEST,
        status,
        "Reviewer completed",
        evidence=_evidence(),
        reviewed_candidate_digest=candidate.candidate_digest,
        payload={
            "schema_version": 1,
            "result_kind": kind,
            "candidate_attempt_id": candidate.attempt_id,
            "candidate_digest": candidate.candidate_digest,
            "verdict": verdict,
            "findings": [],
            "evidence": _evidence_payload(),
        },
    )


def _gate_receipt(*, passed: bool) -> dict[str, object]:
    return {
        "gate_id": "entry-gpu",
        "purpose": "correctness",
        "scope": "single_gpu_product",
        "placements": [{"rank": 0, "node": "node-a", "local_rank": 0}],
        "product_rank_body": True,
        "passed": passed,
        "accuracy": None,
    }


def _tuning_proposal() -> WorkItemProposal:
    hypothesis = TuningHypothesis(
        hypothesis_id="hypothesis-1",
        item_id="catalog-item",
        statement="A bounded setting improves matched throughput.",
        metric="tokens_per_second",
        direction=MetricDirection.HIGHER_IS_BETTER,
        changes=(
            KnobChange(
                name="attention.backend",
                kind=KnobKind.CONFIGURATION,
                baseline_value="a",
                candidate_value="b",
            ),
        ),
        uncertainty=UncertaintyRule(
            minimum_effect=1.0,
            noise_threshold=0.5,
            maximum_combined_uncertainty=2.0,
        ),
    )
    return WorkItemProposal(
        item_id="catalog-item",
        goal_id="tuning-goal",
        kind=PolicyWorkItemKind.TUNE_HYPOTHESIS,
        resource_class="coder_analysis",
        execution=ExecutionShape(1, 1, 0),
        modifies_files=False,
        domain_input=hypothesis,
    )


def _measurement_payload(*, samples: list[float]) -> dict[str, object]:
    return {
        "identity": {
            "checkpoint": "checkpoint@revision",
            "target": "target-sm100",
            "route": {
                "family": "example",
                "architecture": "ExampleForCausalLM",
                "expected_route": "agent_flow.test.example",
                "synthetic_target": True,
            },
            "workload": "fixed-workload",
            "topology": {
                "world_size": 1,
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "moe_expert_parallel_size": 1,
                "moe_tensor_parallel_size": 1,
                "attention_data_parallel_size": 1,
            },
            "build": "build-identity",
            "protocol": "paired-three-repetitions",
            "hardware": "one-test-gpu",
        },
        "arm_digest": ("a" if samples[0] < 105 else "b") * 64,
        "curve": {"samples": samples, "uncertainty": 0.25},
        "gates": [
            {
                "name": "correctness",
                "kind": "correctness",
                "passed": True,
                "evidence_digest": "e" * 64,
            }
        ],
    }


def _tuner_manifest(attempt: AttemptRecord) -> WorkerResultManifest:
    return WorkerResultManifest(
        "run-1",
        "catalog-item",
        attempt.attempt_id,
        _TASK_DIGEST,
        1,
        _INPUT_DIGEST,
        WorkerResultStatus.SUCCEEDED,
        "Tuner measured matched arms",
        evidence=_evidence(),
        candidate_digest="d" * 64,
        payload={
            "schema_version": 1,
            "result_kind": "tuner_measurement",
            "baseline": _measurement_payload(samples=[99.0, 100.0, 101.0]),
            "candidate": _measurement_payload(samples=[109.0, 110.0, 111.0]),
            "evidence": _evidence_payload(),
        },
    )


def test_coder_decoder_requires_scanned_overlay_digest() -> None:
    attempt = _attempt()
    decoded = decode_coder_result(_coder_manifest(), attempt, _proposal())
    assert decoded.changed_paths == ("tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py",)
    assert decoded.index_delta is not None

    with pytest.raises(ResultContractError, match="requires a scanned"):
        decode_coder_result(_coder_manifest(candidate_digest=None), attempt, _proposal())
    payload = dict(_coder_manifest().payload)
    payload["unknown"] = True
    with pytest.raises(ResultContractError, match="unknown"):
        decode_coder_result(_coder_manifest(payload=payload), attempt, _proposal())


def test_tuner_decoder_accepts_measurements_but_rejects_worker_promotion() -> None:
    attempt = _attempt(profile=DomainProfile.TUNER)
    manifest = _tuner_manifest(attempt)
    payload = manifest.payload

    result = decode_tuner_result(manifest, attempt, _tuning_proposal())

    assert result.baseline.curve.mean == 100.0
    assert result.candidate.curve.mean == 110.0
    assert result.evidence == _evidence()

    worker_decision = dict(payload)
    worker_decision["promotion_decision"] = "keep"
    with pytest.raises(ResultContractError, match="unknown"):
        decode_tuner_result(replace(manifest, payload=worker_decision), attempt, _tuning_proposal())

    mismatched = dict(payload)
    candidate = dict(mismatched["candidate"])  # type: ignore[arg-type]
    identity = dict(candidate["identity"])  # type: ignore[arg-type]
    identity["build"] = "different-build"
    candidate["identity"] = identity
    mismatched["candidate"] = candidate
    with pytest.raises(ResultContractError, match="invalid Tuner measurement"):
        decode_tuner_result(replace(manifest, payload=mismatched), attempt, _tuning_proposal())


def test_tuner_candidate_binds_exact_evidence_ready_artifacts(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    attempt = _attempt(profile=DomainProfile.TUNER)
    proposal = _tuning_proposal()
    manifest = replace(
        _tuner_manifest(attempt),
        candidate_digest=digest_metadata_free_tree(worktree),
    )
    decoded = decode_tuner_result(manifest, attempt, proposal)
    assert isinstance(proposal.domain_input, TuningHypothesis)
    evaluation = evaluate_tuning_campaign(
        proposal,
        proposal.domain_input,
        decoded.baseline,
        decoded.candidate,
    )
    campaign = record_tuning_evidence(
        new_tuning_campaign(proposal, proposal.domain_input),
        evaluation,
    )
    artifact_root = tmp_path / "tuning-artifacts"
    artifact_root.mkdir()
    tuning_artifacts.write_baseline_artifact(
        artifact_root / tuning_artifacts.BASELINE_FILENAME, decoded.baseline
    )
    tuning_artifacts.write_candidate_artifact(
        artifact_root / tuning_artifacts.CANDIDATE_FILENAME, decoded.candidate
    )
    tuning_artifacts.write_evaluation_artifact(
        artifact_root / tuning_artifacts.EVALUATION_FILENAME, evaluation
    )
    tuning_artifacts.write_campaign_artifact(
        artifact_root / tuning_artifacts.CAMPAIGN_FILENAME, campaign
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    binding = bind_tuner_candidate(
        manifest,
        attempt,
        proposal,
        worktree=worktree,
        base_commit=base,
        attempt_dir=attempt_dir,
        artifact_root=artifact_root,
        git=controller,
    )

    assert binding.receipt.disposition is CandidateDisposition.VERIFICATION
    assert binding.receipt.base_commit == binding.receipt.candidate_commit == base
    assert binding.receipt.changed_paths == ()
    assert binding.receipt.index_delta is None
    assert len(binding.receipt.candidate_digest) == 64
    assert (
        bind_tuner_candidate(
            manifest,
            attempt,
            proposal,
            worktree=worktree,
            base_commit=base,
            attempt_dir=attempt_dir,
            artifact_root=artifact_root,
            git=controller,
        )
        == binding
    )

    with pytest.raises(CandidateReceiptError, match="differs from durable evidence"):
        bind_tuner_candidate(
            replace(manifest, summary="different valid summary"),
            attempt,
            proposal,
            worktree=worktree,
            base_commit=base,
            attempt_dir=attempt_dir,
            artifact_root=artifact_root,
            git=controller,
        )


def test_tuner_candidate_rejects_promotion_artifact(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    attempt = _attempt(profile=DomainProfile.TUNER)
    artifact_root = tmp_path / "tuning-artifacts"
    artifact_root.mkdir()
    (artifact_root / tuning_artifacts.PROMOTION_DECISION_FILENAME).write_text(
        "{}\n", encoding="utf-8"
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    with pytest.raises(CandidateReceiptError, match="cannot contain a promotion decision"):
        bind_tuner_candidate(
            replace(
                _tuner_manifest(attempt),
                candidate_digest=digest_metadata_free_tree(worktree),
            ),
            attempt,
            _tuning_proposal(),
            worktree=worktree,
            base_commit=base,
            attempt_dir=attempt_dir,
            artifact_root=artifact_root,
            git=controller,
        )


def test_controller_binds_dco_commit_and_reconstructs_after_restart(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    relative = "tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"
    (worktree / relative).write_text("VALUE = 1\n", encoding="utf-8")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    binding = bind_coder_candidate(
        _scanned_coder_manifest(worktree),
        _attempt(),
        _proposal(),
        worktree=worktree,
        base_commit=base,
        attempt_dir=attempt_dir,
        git=controller,
    )

    assert binding.receipt.candidate_digest == binding.snapshot.candidate_digest
    assert binding.receipt.candidate_commit == _git(
        binding.snapshot.repository, "rev-parse", "HEAD"
    )
    assert "Signed-off-by:" in _git(binding.snapshot.repository, "log", "-1", "--format=%B")
    receipt_path = attempt_dir / CANDIDATE_RECEIPT_FILENAME
    assert load_candidate_receipt(receipt_path) == binding.receipt
    assert (
        reconstruct_candidate(
            receipt_path,
            worktree=binding.snapshot.repository,
            proposal=_proposal(),
            git=controller,
        ).snapshot
        == binding.snapshot
    )
    repeated = bind_coder_candidate(
        _scanned_coder_manifest(worktree),
        _attempt(),
        _proposal(),
        worktree=worktree,
        base_commit=base,
        attempt_dir=attempt_dir,
        git=controller,
    )
    assert repeated == binding


def test_read_only_verification_receipt_freezes_clean_unchanged_candidate(
    tmp_path: Path,
) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    manifest = _scanned_coder_manifest(
        worktree,
        payload={
            "schema_version": 1,
            "result_kind": "coder",
            "changed_paths": [],
            "index_delta": None,
        },
    )

    binding = bind_coder_candidate(
        manifest,
        _attempt(),
        _proposal(modifies_files=False),
        worktree=worktree,
        base_commit=base,
        attempt_dir=attempt_dir,
        git=controller,
    )

    assert binding.receipt.disposition is CandidateDisposition.VERIFICATION
    assert binding.receipt.base_commit == binding.receipt.candidate_commit == base
    assert binding.receipt.changed_paths == ()
    assert binding.snapshot.changed_paths == ()
    assert _git(binding.snapshot.repository, "rev-list", "--count", "HEAD") == "1"
    assert (
        reconstruct_candidate(
            attempt_dir / CANDIDATE_RECEIPT_FILENAME,
            worktree=binding.snapshot.repository,
            proposal=_proposal(modifies_files=False),
            git=controller,
        )
        == binding
    )


def test_candidate_binding_rejects_undeclared_changes(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    relative = "tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"
    (worktree / relative).write_text("VALUE = 1\n", encoding="utf-8")
    (worktree / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    with pytest.raises(CandidateReceiptError, match="differ"):
        bind_coder_candidate(
            _scanned_coder_manifest(worktree),
            _attempt(),
            _proposal(),
            worktree=worktree,
            base_commit=base,
            attempt_dir=attempt_dir,
            git=controller,
        )
    assert controller.inspect().head == base


def test_candidate_binding_rejects_symlinked_attempt_parent(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    relative = "tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"
    (worktree / relative).write_text("VALUE = 1\n", encoding="utf-8")
    real_parent = tmp_path / "real-attempts"
    real_parent.mkdir()
    (tmp_path / "attempts-link").symlink_to(real_parent, target_is_directory=True)
    attempt_dir = tmp_path / "attempts-link" / "attempt"
    attempt_dir.mkdir()

    with pytest.raises(CandidateReceiptError, match="path contains a symlink"):
        bind_coder_candidate(
            _scanned_coder_manifest(worktree),
            _attempt(),
            _proposal(),
            worktree=worktree,
            base_commit=base,
            attempt_dir=attempt_dir,
            git=controller,
        )


def test_binding_recovers_controller_commit_created_before_receipt(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    relative = "tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"
    (worktree / relative).write_text("VALUE = 2\n", encoding="utf-8")
    controller_repository = (
        worktree.parents[2] / "controller-candidates" / "catalog-item" / "catalog-item.0001.role"
    )
    with controller.transaction() as transaction:
        bound = transaction.bind_candidate_overlay(
            worktree,
            repository=controller_repository,
            branch="staircase/run-1/catalog-item.0001.role",
            base_commit=base,
            expected_paths=(relative,),
            expected_overlay_sha256=digest_metadata_free_tree(worktree),
            message="staircase: bind catalog-item candidate",
        )
        committed = bound.candidate_commit
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    binding = bind_coder_candidate(
        _scanned_coder_manifest(worktree),
        _attempt(),
        _proposal(),
        worktree=worktree,
        base_commit=base,
        attempt_dir=attempt_dir,
        git=controller,
    )

    assert binding.receipt.candidate_commit == committed
    assert (attempt_dir / CANDIDATE_RECEIPT_FILENAME).is_file()


def test_binding_rejects_substituted_same_parent_and_path_commit(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    relative = "tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"
    (worktree / relative).write_text("VALUE = 2\n", encoding="utf-8")
    controller_repository = (
        worktree.parents[2] / "controller-candidates" / "catalog-item" / "catalog-item.0001.role"
    )
    substitute_repository = tmp_path / "substitute"
    with controller.transaction() as transaction:
        transaction.bind_candidate_overlay(
            worktree,
            repository=controller_repository,
            branch="staircase/run-1/catalog-item.0001.role",
            base_commit=base,
            expected_paths=(relative,),
            expected_overlay_sha256=digest_metadata_free_tree(worktree),
            message="staircase: bind catalog-item candidate",
        )
        transaction.create_candidate_worktree(
            substitute_repository,
            branch="staircase/run-1/substitute",
            base_commit=base,
        )
        (substitute_repository / relative).write_text("VALUE = 999\n", encoding="utf-8")
        transaction.stage_paths(substitute_repository, (relative,))
        substitute = transaction.commit_signed_off(
            substitute_repository,
            message="staircase: bind catalog-item candidate",
            expected_paths=(relative,),
        )
    _git(
        controller_repository,
        "update-ref",
        "refs/heads/staircase/run-1/catalog-item.0001.role",
        substitute,
    )
    _git(controller_repository, "checkout", "-q", "HEAD", "--", relative)
    assert not _git(controller_repository, "status", "--short")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    with pytest.raises(CandidateReceiptError, match="tree differs"):
        bind_coder_candidate(
            _scanned_coder_manifest(worktree),
            _attempt(),
            _proposal(),
            worktree=worktree,
            base_commit=base,
            attempt_dir=attempt_dir,
            git=controller,
        )
    assert not (attempt_dir / CANDIDATE_RECEIPT_FILENAME).exists()


def test_binding_rejects_recovered_commit_without_dco(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    relative = "tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"
    (worktree / relative).write_text("VALUE = 3\n", encoding="utf-8")
    controller_repository = (
        worktree.parents[2] / "controller-candidates" / "catalog-item" / "catalog-item.0001.role"
    )
    with controller.transaction() as transaction:
        bound = transaction.bind_candidate_overlay(
            worktree,
            repository=controller_repository,
            branch="staircase/run-1/catalog-item.0001.role",
            base_commit=base,
            expected_paths=(relative,),
            expected_overlay_sha256=digest_metadata_free_tree(worktree),
            message="staircase: bind catalog-item candidate",
        )
    tree = _git(controller_repository, "rev-parse", f"{bound.candidate_commit}^{{tree}}")
    unsigned = subprocess.run(
        ["git", "-C", str(controller_repository), "commit-tree", tree, "-p", base],
        input="staircase: bind catalog-item candidate\n",
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    _git(
        controller_repository,
        "update-ref",
        "refs/heads/staircase/run-1/catalog-item.0001.role",
        unsigned,
    )
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()

    with pytest.raises(CandidateReceiptError, match="lacks a valid DCO"):
        bind_coder_candidate(
            _scanned_coder_manifest(worktree),
            _attempt(),
            _proposal(),
            worktree=worktree,
            base_commit=base,
            attempt_dir=attempt_dir,
            git=controller,
        )
    assert not (attempt_dir / CANDIDATE_RECEIPT_FILENAME).exists()


def test_receipt_tamper_and_worktree_drift_are_detected(tmp_path: Path) -> None:
    _repository_root, worktree, base, controller = _repository(tmp_path)
    relative = "tensorrt_llm/_torch/modeling_v2/catalog/norm/alpha.py"
    (worktree / relative).write_text("VALUE = 1\n", encoding="utf-8")
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    binding = bind_coder_candidate(
        _scanned_coder_manifest(worktree),
        _attempt(),
        _proposal(),
        worktree=worktree,
        base_commit=base,
        attempt_dir=attempt_dir,
        git=controller,
    )
    receipt_path = attempt_dir / CANDIDATE_RECEIPT_FILENAME
    raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw["candidate_digest"] = "0" * 64
    receipt_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(CandidateReceiptError, match="digest mismatch"):
        load_candidate_receipt(receipt_path)

    receipt_path.write_text(
        json.dumps(
            {
                **raw,
                "candidate_digest": binding.receipt.candidate_digest,
                "receipt_sha256": binding.receipt.receipt_sha256,
            }
        ),
        encoding="utf-8",
    )
    (binding.snapshot.repository / "untracked.txt").write_text("drift\n", encoding="utf-8")
    with pytest.raises(CandidateReceiptError, match="not clean"):
        reconstruct_candidate(
            receipt_path,
            worktree=binding.snapshot.repository,
            proposal=_proposal(),
            git=controller,
        )


def test_reviewer_analysis_and_fresh_rerun_are_distinct() -> None:
    candidate = _fake_candidate()
    analysis_attempt = _attempt(
        Role.REVIEWER,
        AttemptKind.REVIEWER_ANALYSIS,
        attempt_id="catalog-item.0002.analysis",
        sequence=2,
        candidate=candidate,
    )
    analysis = decode_reviewer_result(
        _review_manifest(
            candidate,
            kind="reviewer_analysis",
            verdict="approve",
            status=WorkerResultStatus.SUCCEEDED,
            attempt_id=analysis_attempt.attempt_id,
        ),
        analysis_attempt,
        candidate,
    )
    assert analysis.verdict is ReviewerVerdict.APPROVE
    assert not analysis.final_approval

    rerun_attempt = _attempt(
        Role.REVIEWER,
        AttemptKind.REVIEWER_RERUN,
        attempt_id="catalog-item.0003.rerun",
        sequence=3,
        candidate=candidate,
    )
    rerun = decode_reviewer_result(
        _review_manifest(
            candidate,
            kind="reviewer_rerun",
            verdict="approve",
            status=WorkerResultStatus.SUCCEEDED,
            attempt_id=rerun_attempt.attempt_id,
        ),
        rerun_attempt,
        candidate,
    )
    assert rerun.final_approval


def test_reviewer_rejects_wrong_candidate_and_status() -> None:
    candidate = _fake_candidate()
    attempt = _attempt(
        Role.REVIEWER,
        AttemptKind.REVIEWER_RERUN,
        attempt_id="catalog-item.0003.rerun",
        sequence=3,
        candidate=candidate,
    )
    wrong_status = _review_manifest(
        candidate,
        kind="reviewer_rerun",
        verdict="reject",
        status=WorkerResultStatus.SUCCEEDED,
        attempt_id=attempt.attempt_id,
    )
    with pytest.raises(ResultContractError, match="status disagrees"):
        decode_reviewer_result(wrong_status, attempt, candidate)

    payload = dict(wrong_status.payload)
    payload["candidate_digest"] = "0" * 64
    wrong_candidate = replace(
        wrong_status,
        status=WorkerResultStatus.REJECTED,
        payload=payload,
    )
    with pytest.raises(ResultContractError, match="not pinned"):
        decode_reviewer_result(wrong_candidate, attempt, candidate)


def test_gate_and_qa_receipts_pin_candidate_and_evidence() -> None:
    candidate = _fake_candidate()
    gate_attempt = _attempt(
        Role.GATE,
        AttemptKind.DETERMINISTIC_GATE,
        attempt_id="catalog-item.0002.gate",
        sequence=2,
    )
    gate_manifest = WorkerResultManifest(
        "run-1",
        "catalog-item",
        gate_attempt.attempt_id,
        _TASK_DIGEST,
        1,
        _INPUT_DIGEST,
        WorkerResultStatus.SUCCEEDED,
        "Gate completed",
        evidence=_evidence(),
        reviewed_candidate_digest=candidate.candidate_digest,
        payload={
            "schema_version": 1,
            "result_kind": "deterministic_gate",
            "certification_mode": "LOCAL",
            "candidate_attempt_id": candidate.attempt_id,
            "candidate_digest": candidate.candidate_digest,
            "receipt": _gate_receipt(passed=True),
            "evidence": _evidence_payload(),
        },
    )
    spec = GateSpec(
        "entry-gpu",
        GatePhase.ENTRY_GPU,
        GatePurpose.CORRECTNESS,
        GateCommand(("python3", "-m", "pytest", "test_entry.py")),
    )
    gate = decode_gate_result(
        gate_manifest,
        gate_attempt,
        candidate,
        expected_spec=spec,
    )
    assert gate.receipt.passed
    assert gate.certification_mode.value == "LOCAL"

    qa_attempt = _attempt(
        Role.QA,
        AttemptKind.QA,
        attempt_id="catalog-item.0003.qa",
        sequence=3,
    )
    qa_manifest = WorkerResultManifest(
        "run-1",
        "catalog-item",
        qa_attempt.attempt_id,
        _TASK_DIGEST,
        1,
        _INPUT_DIGEST,
        WorkerResultStatus.SUCCEEDED,
        "QA completed",
        evidence=_evidence(),
        reviewed_candidate_digest=candidate.candidate_digest,
        payload={
            "schema_version": 1,
            "result_kind": "qa",
            "candidate_attempt_id": candidate.attempt_id,
            "candidate_digest": candidate.candidate_digest,
            "verdict": "approve",
            "findings": [],
            "gate_results": [
                {
                    "gate_attempt_id": gate_attempt.attempt_id,
                    "result_digest": _RESULT_DIGEST,
                }
            ],
            "evidence": _evidence_payload(),
        },
    )
    qa = decode_qa_result(
        qa_manifest,
        qa_attempt,
        candidate,
        expected_gate_results={gate_attempt.attempt_id: _RESULT_DIGEST},
    )
    assert qa.verdict is QaVerdict.APPROVE
    assert qa.gate_results[0].gate_attempt_id == gate_attempt.attempt_id

    forged_qa_payload = dict(qa_manifest.payload)
    forged_qa_payload.pop("gate_results")
    forged_qa_payload["gate_receipts"] = [_gate_receipt(passed=True)]
    with pytest.raises(ResultContractError, match="keys invalid"):
        decode_qa_result(
            replace(qa_manifest, payload=forged_qa_payload),
            qa_attempt,
            candidate,
            expected_gate_results={gate_attempt.attempt_id: _RESULT_DIGEST},
        )

    mismatched_references = dict(qa_manifest.payload)
    mismatched_references["gate_results"] = [
        {
            "gate_attempt_id": gate_attempt.attempt_id,
            "result_digest": "f" * 64,
        }
    ]
    with pytest.raises(ResultContractError, match="controller-ingested"):
        decode_qa_result(
            replace(qa_manifest, payload=mismatched_references),
            qa_attempt,
            candidate,
            expected_gate_results={gate_attempt.attempt_id: _RESULT_DIGEST},
        )

    bad_payload = dict(gate_manifest.payload)
    bad_payload["evidence"] = []
    with pytest.raises(ResultContractError, match="differs"):
        decode_gate_result(replace(gate_manifest, payload=bad_payload), gate_attempt, candidate)

    bad_certification = dict(gate_manifest.payload)
    bad_certification["certification_mode"] = "REAL"
    with pytest.raises(ResultContractError, match="evidence scope"):
        decode_gate_result(
            replace(gate_manifest, payload=bad_certification), gate_attempt, candidate
        )


def test_resource_escalation_is_strict_and_policy_checked() -> None:
    proposal = _proposal()
    attempt = _attempt()
    manifest = WorkerResultManifest(
        "run-1",
        "catalog-item",
        attempt.attempt_id,
        _TASK_DIGEST,
        1,
        _INPUT_DIGEST,
        WorkerResultStatus.RESOURCE_ESCALATION,
        "more resources required",
        payload={
            "schema_version": 1,
            "result_kind": "resource_escalation",
            "request": {
                "request_id": "request-1",
                "item_id": "catalog-item",
                "attempt_id": attempt.attempt_id,
                "current_resource_class": "coder_analysis",
                "requested_resource_class": "exploratory_probe",
                "reason": "single-node probe exhausted memory",
            },
        },
    )
    request = decode_resource_escalation(
        manifest,
        attempt,
        proposal,
        allowed_resource_classes=("coder_analysis", "exploratory_probe"),
    )
    assert request.requested_resource_class == "exploratory_probe"

    payload = dict(manifest.payload)
    request_payload = dict(payload["request"])  # type: ignore[arg-type]
    request_payload["extra"] = "not allowed"
    payload["request"] = request_payload
    with pytest.raises(ResultContractError, match="unknown"):
        decode_resource_escalation(
            replace(manifest, payload=payload),
            attempt,
            proposal,
            allowed_resource_classes=("coder_analysis", "exploratory_probe"),
        )
