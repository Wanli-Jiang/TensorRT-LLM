# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused pure-planning tests for the trusted collective adapter."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent_flow.workflows.staircase.common.gates import (
    EvidenceScope,
    GateCommand,
    GatePhase,
    GatePurpose,
    GateSpec,
)
from agent_flow.workflows.staircase.common.gitops import ControllerGitOps
from agent_flow.workflows.staircase.common.policy import ExecutionShape, WorkItemProposal
from agent_flow.workflows.staircase.common.policy import WorkItemKind as PolicyWorkItemKind
from agent_flow.workflows.staircase.controller.collective import (
    CollectiveAdapterError,
    TrustedCollectiveGateExecutionAdapter,
    build_local_product_certifications,
    build_task_collective_adapter,
)
from agent_flow.workflows.staircase.controller.domain import FrozenCandidate, GateExecutionContract
from agent_flow.workflows.staircase.controller.rank_gates import CollectiveCertification
from agent_flow.workflows.staircase.controller.results import CandidateDisposition, CandidateReceipt
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    GoalRecord,
    JobReference,
    Role,
    RunState,
    StageRecord,
    WorkItemKind,
    WorkItemRecord,
    WorkItemStatus,
)
from agent_flow.workflows.staircase.task_schema import (
    CertificationConfig,
    CertificationMode,
    GateClass,
    NormalizedTask,
    ParallelMapping,
    ResourceClass,
)


def _candidate_receipt(commit: str = "2" * 40) -> CandidateReceipt:
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "item_id": "gate-item",
        "attempt_id": "gate-item.0001.role",
        "disposition": "verification",
        "base_commit": commit,
        "candidate_commit": commit,
        "candidate_digest": "d" * 64,
        "changed_paths": [],
        "index_delta": None,
    }
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return CandidateReceipt(
        run_id="run-1",
        item_id="gate-item",
        attempt_id="gate-item.0001.role",
        disposition=CandidateDisposition.VERIFICATION,
        base_commit=commit,
        candidate_commit=commit,
        candidate_digest="d" * 64,
        changed_paths=(),
        index_delta=None,
        receipt_sha256=digest,
    )


def _state_and_proposal() -> tuple[RunState, WorkItemProposal, FrozenCandidate]:
    proposal = WorkItemProposal(
        item_id="gate-item",
        goal_id="goal",
        kind=PolicyWorkItemKind.GATE,
        resource_class="coder_analysis",
        execution=ExecutionShape(),
        modifies_files=False,
        domain_input=None,
        dependencies=(),
        allowed_paths=(),
    )
    coder = AttemptRecord(
        attempt_id="gate-item.0001.role",
        item_id="gate-item",
        sequence=1,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        generation=1,
        profile=DomainProfile.ASSEMBLER,
        status=AttemptStatus.VALIDATED,
        submission_token="coder-token",
        job=JobReference("123"),
        result_digest="c" * 64,
        candidate_digest="d" * 64,
    )
    item = WorkItemRecord(
        item_id="gate-item",
        stage_id="stage",
        goal_id="goal",
        kind=WorkItemKind.GATE,
        profile=DomainProfile.ASSEMBLER,
        status=WorkItemStatus.CODING,
        attempts=(coder,),
    )
    state = RunState(
        run_id="run-1",
        task_digest="a" * 64,
        base_commit="1" * 40,
        generation=1,
        stages=(StageRecord("stage", ("goal",)),),
        goals=(GoalRecord("goal", "stage", ("gate-item",)),),
        items=(item,),
    )
    candidate = FrozenCandidate(
        item_id="gate-item",
        coder_attempt_id=coder.attempt_id,
        commit="2" * 40,
        digest="d" * 64,
        changed_paths=(),
    )
    return state, proposal, candidate


def _adapter(tmp_path: Path) -> TrustedCollectiveGateExecutionAdapter:
    mapping = ParallelMapping(2, 1, 1, 2, 1)
    task = cast(NormalizedTask, SimpleNamespace(target=SimpleNamespace(mapping=mapping)))
    git = cast(ControllerGitOps, object())
    return TrustedCollectiveGateExecutionAdapter(
        workspace=tmp_path,
        task=task,
        git=git,
        certifications={"collective": CollectiveCertification.SYNTHETIC_RUNNER},
    )


def test_plan_binds_candidate_gate_mapping_resource_and_rank_paths(tmp_path: Path) -> None:
    state, proposal, candidate = _state_and_proposal()
    mapping = ParallelMapping(2, 1, 1, 2, 1)
    gate = GateSpec(
        "collective",
        GatePhase.COLLECTIVE,
        GatePurpose.CORRECTNESS,
        GateCommand(("python3", "runner.py")),
    )
    execution = GateExecutionContract(
        EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
        2,
        None,
        None,
        False,
    )
    resource = ResourceClass("deterministic_gate", 2, 1, 1, 4, 4096, 600)

    plan = _adapter(tmp_path).plan(
        state,
        proposal,
        candidate,
        _candidate_receipt(),
        gate,
        execution,
        resource=resource,
        mapping=mapping,
        workspace=tmp_path,
    )

    assert plan.action is not None
    assert plan.action.attempt.role is Role.GATE
    assert plan.action.attempt.kind is AttemptKind.DETERMINISTIC_GATE
    assert plan.action.worktree == (
        tmp_path / "candidates" / "gate-item" / plan.action.attempt.attempt_id
    )
    runtime = plan.action.manifest.payload["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["kind"] == "rank_supervisor"
    assert runtime["certification_mode"] == "SYNTHETIC"
    assert runtime["candidate"]["receipt_sha256"] == _candidate_receipt().receipt_sha256
    assert runtime["execution"] == {
        "scope": "synthetic_multi_node_runner",
        "expected_world_size": 2,
        "expected_rank": None,
        "expected_local_rank": None,
        "product_rank_body": False,
    }
    assert runtime["rank_input_path"] == str(plan.action.attempt_dir / "output" / "rank-input.json")
    assert runtime["rank_report_directory"] == str(
        plan.action.attempt_dir / "output" / "rank-reports"
    )


def test_real_constructor_fails_closed_without_expected_identity(tmp_path: Path) -> None:
    task = cast(
        NormalizedTask,
        SimpleNamespace(target=SimpleNamespace(mapping=ParallelMapping(2, 1, 1, 2, 1))),
    )

    with pytest.raises(CollectiveAdapterError, match="lacks expected product identity"):
        TrustedCollectiveGateExecutionAdapter(
            workspace=tmp_path,
            task=task,
            git=cast(ControllerGitOps, object()),
            certifications={"collective": CollectiveCertification.REAL_PRODUCT},
        )


def test_task_factory_constructs_only_explicit_nonlocal_adapter(tmp_path: Path) -> None:
    common = {
        "target": SimpleNamespace(mapping=ParallelMapping(2, 1, 1, 2, 1)),
        "execution": SimpleNamespace(
            slurm=SimpleNamespace(gate_classes=(GateClass("collective", "deterministic_gate"),))
        ),
    }
    local_task = cast(
        NormalizedTask,
        SimpleNamespace(
            **common,
            certification=CertificationConfig(CertificationMode.LOCAL, None),
        ),
    )
    synthetic_task = cast(
        NormalizedTask,
        SimpleNamespace(
            **common,
            certification=CertificationConfig(CertificationMode.SYNTHETIC, None),
        ),
    )
    git = cast(ControllerGitOps, object())

    assert build_task_collective_adapter(workspace=tmp_path, task=local_task, git=git) is None
    adapter = build_task_collective_adapter(workspace=tmp_path, task=synthetic_task, git=git)
    assert isinstance(adapter, TrustedCollectiveGateExecutionAdapter)


def test_default_helper_derives_only_canonical_local_product() -> None:
    gate = GateSpec(
        "collective",
        GatePhase.COLLECTIVE,
        GatePurpose.CORRECTNESS,
        GateCommand(("python3", "collective.py")),
    )
    mapping = ParallelMapping(4, 1, 1, 4, 1)
    execution = GateExecutionContract(
        EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
        4,
        None,
        None,
        True,
    )
    resource = ResourceClass("deterministic_gate", 1, 4, 4, 4, 4096, 600)

    assert build_local_product_certifications(
        gates=(gate,),
        executions={gate.gate_id: execution},
        resources={gate.gate_id: resource},
        mapping=mapping,
    ) == {gate.gate_id: CollectiveCertification.LOCAL_PRODUCT}

    with pytest.raises(CollectiveAdapterError, match="requires explicit"):
        build_local_product_certifications(
            gates=(gate,),
            executions={
                gate.gate_id: GateExecutionContract(
                    EvidenceScope.REAL_MULTI_NODE_PRODUCT,
                    4,
                    None,
                    None,
                    True,
                )
            },
            resources={gate.gate_id: ResourceClass("deterministic_gate", 2, 2, 2, 4, 4096, 600)},
            mapping=mapping,
        )


def test_collective_resources_use_gate_partition_and_qos_overrides(tmp_path: Path) -> None:
    controller = SimpleNamespace(
        account="coreai",
        partition="cpu",
        qos="cpu-short",
        reservation=None,
        image=tmp_path / "runtime.sqsh",
    )
    task = cast(
        NormalizedTask,
        SimpleNamespace(
            target=SimpleNamespace(mapping=ParallelMapping(2, 1, 1, 2, 1)),
            execution=SimpleNamespace(slurm=SimpleNamespace(controller=controller)),
        ),
    )
    adapter = TrustedCollectiveGateExecutionAdapter(
        workspace=tmp_path,
        task=task,
        git=cast(ControllerGitOps, object()),
        certifications={"collective": CollectiveCertification.SYNTHETIC_RUNNER},
    )
    resource = ResourceClass(
        "deterministic_gate",
        2,
        1,
        1,
        4,
        4_096,
        600,
        partition="batch",
        qos="normal",
    )
    attempt = SimpleNamespace(attempt_id="gate-item.0002.deterministic_gate")
    action = SimpleNamespace(attempt=attempt, resource=resource)
    (tmp_path / "slurm" / "workers" / attempt.attempt_id).mkdir(parents=True)

    request = adapter._resources(action, SimpleNamespace(mounts=()))

    assert request.partition == "batch"
    assert request.qos == "normal"


def test_collective_revoke_is_noop_without_credential_artifacts(tmp_path: Path) -> None:
    state, _proposal, _candidate = _state_and_proposal()
    attempt = state.item("gate-item").attempts[0]

    assert _adapter(tmp_path).revoke(state, attempt) is False
    assert list(tmp_path.iterdir()) == []


def test_collective_candidate_is_metadata_free_read_only_snapshot(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    repository.mkdir()
    workspace.mkdir()
    subprocess.run(("git", "init", str(repository)), check=True, capture_output=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Staircase Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.email", "test@example.com"),
        check=True,
    )
    source = repository / "candidate.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "candidate.py"), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-m", "candidate"),
        check=True,
        capture_output=True,
    )
    commit = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git = ControllerGitOps(repository, workspace / "locks" / "git.lock", "test")
    mapping = ParallelMapping(2, 1, 1, 2, 1)
    task = cast(NormalizedTask, SimpleNamespace(target=SimpleNamespace(mapping=mapping)))
    adapter = TrustedCollectiveGateExecutionAdapter(
        workspace=workspace,
        task=task,
        git=git,
        certifications={"collective": CollectiveCertification.SYNTHETIC_RUNNER},
    )
    state, proposal, candidate = _state_and_proposal()
    candidate = FrozenCandidate(
        candidate.item_id,
        candidate.coder_attempt_id,
        commit,
        candidate.digest,
        candidate.changed_paths,
    )
    gate = GateSpec(
        "collective",
        GatePhase.COLLECTIVE,
        GatePurpose.CORRECTNESS,
        GateCommand(("python3", "candidate.py")),
    )
    plan = adapter.plan(
        state,
        proposal,
        candidate,
        _candidate_receipt(commit),
        gate,
        GateExecutionContract(EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER, 2, None, None, False),
        resource=ResourceClass("deterministic_gate", 2, 1, 1, 4, 4096, 600),
        mapping=mapping,
        workspace=workspace,
    )
    assert plan.action is not None

    adapter._materialize_candidate_snapshot(plan.action)
    adapter._materialize_candidate_snapshot(plan.action)

    snapshot = plan.action.worktree
    assert snapshot == workspace / "candidates" / "gate-item" / plan.action.attempt.attempt_id
    assert (snapshot / "candidate.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert not any(".git" in path.relative_to(snapshot).parts for path in snapshot.rglob("*"))
    assert not ((snapshot / "candidate.py").stat().st_mode & stat.S_IWUSR)
