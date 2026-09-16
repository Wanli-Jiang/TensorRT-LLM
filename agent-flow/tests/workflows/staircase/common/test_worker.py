# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the isolated generic Staircase worker runtime."""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import internal
from agent_flow.workflows.staircase.common import artifacts, runners, worker
from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialDescriptor,
    CredentialHandle,
    CredentialState,
)
from agent_flow.workflows.staircase.common.isolation import digest_metadata_free_tree
from agent_flow.workflows.staircase.common.launch_policy import AgentLaunchPolicy
from agent_flow.workflows.staircase.common.placement import (
    PLACEMENT_RECEIPT_FILENAME,
    load_worker_placement,
)
from agent_flow.workflows.staircase.common.slurm import AGENT_POLICY_DIGEST_ENVIRONMENT
from agent_flow.workflows.staircase.state import DomainProfile, Role

TASK_DIGEST = "a" * 64
CANDIDATE_DIGEST = "b" * 64
LAUNCH_POLICY = AgentLaunchPolicy.create(
    image="/images/agent-worker.sqsh",
    build_identity="agent-worker-test",
)


@pytest.fixture(autouse=True)
def isolated_role_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model the production agent image, which contains no Slurm clients."""
    monkeypatch.setattr(worker, "_scheduler_client_path", lambda _name: None)
    monkeypatch.setenv(AGENT_POLICY_DIGEST_ENVIRONMENT, LAUNCH_POLICY.digest)


def _agent_runtime(
    *,
    candidate_digest: str | None = None,
    evidence_paths: list[str] | None = None,
    credential_descriptor: dict[str, object] | None = None,
    distinct_nodes_required: bool = False,
) -> dict[str, artifacts.JsonValue]:
    runtime: dict[str, artifacts.JsonValue] = {
        "schema_version": 1,
        "kind": "agent",
        "backend_kind": "codex",
        "model": "test-model",
        "prompt_id": "prompt-1",
        "prompt": "Perform exactly one catalog task.",
        "evidence_paths": (["reports/probe.json"] if evidence_paths is None else evidence_paths),
        "candidate_digest": candidate_digest,
        "launch_policy": LAUNCH_POLICY.to_public_dict(),
        "candidate_overlay_digest": None,
        "placement_contract": {
            "distinct_nodes_required": distinct_nodes_required,
            "exclusive": distinct_nodes_required,
        },
    }
    if credential_descriptor is not None:
        runtime["credential_descriptor"] = credential_descriptor  # type: ignore[assignment]
    return runtime


def _gate_runtime(*, gpu: bool, command_id: str = "probe") -> dict[str, artifacts.JsonValue]:
    return {
        "schema_version": 1,
        "kind": "gate",
        "candidate_attempt_id": "coder-1",
        "candidate_digest": CANDIDATE_DIGEST,
        "command": {
            "command_id": command_id,
            "argv": [sys.executable, "-c", "print('passed')"],
            "environment": {"TRTLLM_MODELING_V2": "require"},
            "cwd": ".",
        },
        "receipt": {
            "gate_id": command_id,
            "purpose": "correctness",
            "scope": "single_gpu_product" if gpu else "cpu_static",
            "certification_mode": "LOCAL",
            "expected_world_size": 1 if gpu else 0,
            "expected_rank": 0 if gpu else None,
            "expected_local_rank": 0 if gpu else None,
            "product_rank_body": gpu,
            "accuracy": None,
        },
    }


def _set_single_gpu_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", socket.gethostname())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")


def _publish_input(
    tmp_path: Path,
    *,
    role: Role,
    runtime: dict[str, artifacts.JsonValue],
    profile: DomainProfile | None = DomainProfile.SMITH,
) -> tuple[Path, Path]:
    worktree = (tmp_path / "worktree").resolve()
    worktree.mkdir()
    if role in {Role.REVIEWER, Role.QA}:
        runtime = dict(runtime)
        runtime["candidate_overlay_digest"] = digest_metadata_free_tree(worktree)
    attempt_dir = (tmp_path / "attempt").resolve()
    attempt_dir.mkdir()
    (attempt_dir / artifacts.OUTPUT_DIRECTORY).mkdir()
    manifest = artifacts.WorkerInputManifest(
        run_id="run-1",
        item_id="item-1",
        attempt_id="attempt-1",
        task_digest=TASK_DIGEST,
        generation=1,
        role=role,
        profile=profile,
        worktree=str(worktree),
        allowed_paths=("catalog/entry.py",) if role is Role.CODER else (),
        payload={
            "runtime": runtime,
            "context": {
                "entry_id": "attention.sdpa",
                "candidate": {
                    "coder_attempt_id": "coder-1",
                    "digest": CANDIDATE_DIGEST,
                },
            },
        },
    )
    input_path = attempt_dir / artifacts.INPUT_FILENAME
    artifacts.write_input_manifest(input_path, manifest)
    return input_path, worktree


def _agent_response(
    *,
    outcome: str = "coder_result",
    status: str = "succeeded",
    evidence_paths: list[str] | None = None,
) -> str:
    payloads: dict[str, dict[str, object]] = {
        "coder_result": {
            "schema_version": 1,
            "result_kind": "coder",
            "changed_paths": [],
            "index_delta": None,
        },
        "reviewer_result": {
            "schema_version": 1,
            "result_kind": "reviewer_analysis",
            "candidate_attempt_id": "coder-1",
            "candidate_digest": CANDIDATE_DIGEST,
            "verdict": "approve",
            "findings": [],
            "evidence": [],
        },
        "qa_result": {
            "schema_version": 1,
            "result_kind": "qa",
            "candidate_attempt_id": "coder-1",
            "candidate_digest": CANDIDATE_DIGEST,
            "verdict": "approve",
            "findings": [],
            "gate_results": [],
            "evidence": [],
        },
    }
    return json.dumps(
        {
            "schema_version": 1,
            "outcome": outcome,
            "status": status,
            "summary": "focused probe passed",
            "evidence_paths": evidence_paths or [],
            "payload": payloads[outcome],
        }
    )


def test_agent_uses_compact_prompt_and_collects_only_authorized_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(),
    )
    evidence = worktree / "reports" / "probe.json"
    evidence.parent.mkdir()
    evidence.write_text('{"passed": true}\n', encoding="utf-8")
    invocation: dict[str, object] = {}

    def invoke(**kwargs: object) -> str:
        invocation.update(kwargs)
        return _agent_response(evidence_paths=["reports/probe.json"])

    monkeypatch.setattr(worker, "_invoke_agent", invoke)
    result = worker.execute_worker(input_path)

    loaded, _digest = artifacts.load_result_manifest(input_path.parent / "output")
    assert loaded == result
    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert result.candidate_digest == digest_metadata_free_tree(worktree)
    assert [entry.path for entry in result.evidence] == ["evidence/agent/reports/probe.json"]
    assert (
        input_path.parent / "output" / result.evidence[0].path
    ).read_bytes() == evidence.read_bytes()
    assert invocation["cwd"] == worktree
    assert "Domain profile: Smith" in str(invocation["system_prompt"])
    assert "Return exactly one JSON object" in str(invocation["system_prompt"])
    assert "Controller context (JSON)" in str(invocation["prompt"])
    assert result.payload == {
        "schema_version": 1,
        "result_kind": "coder",
        "changed_paths": [],
        "index_delta": None,
    }


def test_agent_rejects_undeclared_evidence_and_publishes_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(),
    )
    (worktree / "secret.txt").write_text("do not collect\n", encoding="utf-8")
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: _agent_response(evidence_paths=["secret.txt"]),
    )

    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert not result.evidence
    assert (input_path.parent / "output" / artifacts.COMPLETE_FILENAME).is_file()
    assert not (input_path.parent / "output" / "evidence").exists()


def test_reviewer_result_is_pinned_to_controller_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.REVIEWER,
        runtime=_agent_runtime(candidate_digest=CANDIDATE_DIGEST, evidence_paths=[]),
    )
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: _agent_response(outcome="reviewer_result"),
    )

    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert result.candidate_digest is None
    assert result.reviewed_candidate_digest == CANDIDATE_DIGEST
    assert result.payload["candidate_attempt_id"] == "coder-1"
    assert result.payload["candidate_digest"] == CANDIDATE_DIGEST


def test_qa_result_never_claims_controller_owned_candidate_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.QA,
        runtime=_agent_runtime(candidate_digest=CANDIDATE_DIGEST, evidence_paths=[]),
        profile=None,
    )
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: _agent_response(outcome="qa_result"),
    )

    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert result.candidate_digest is None
    assert result.reviewed_candidate_digest == CANDIDATE_DIGEST
    assert result.payload["candidate_attempt_id"] == "coder-1"
    assert result.payload["candidate_digest"] == CANDIDATE_DIGEST


def test_agent_response_extra_key_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(),
    )
    response = json.loads(_agent_response())
    response["decision_in_prose"] = "approve"
    monkeypatch.setattr(worker, "_invoke_agent", lambda **_kwargs: json.dumps(response))

    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "keys invalid" in result.summary


def test_gate_executes_shell_free_and_captures_output(tmp_path: Path) -> None:
    runtime: dict[str, artifacts.JsonValue] = {
        "schema_version": 1,
        "kind": "gate",
        "candidate_attempt_id": "coder-1",
        "candidate_digest": CANDIDATE_DIGEST,
        "command": {
            "command_id": "probe",
            "argv": [
                sys.executable,
                "-c",
                "import os; print(os.environ['EXPECTED'])",
            ],
            "environment": {
                "EXPECTED": "observed",
                "TRTLLM_MODELING_V2": "require",
            },
            "cwd": ".",
        },
        "receipt": {
            "gate_id": "probe",
            "purpose": "structure",
            "scope": "cpu_static",
            "certification_mode": "LOCAL",
            "expected_world_size": 0,
            "expected_rank": None,
            "expected_local_rank": None,
            "product_rank_body": False,
            "accuracy": None,
        },
    }
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=runtime,
    )

    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert result.candidate_digest is None
    assert result.reviewed_candidate_digest == CANDIDATE_DIGEST
    assert len(result.evidence) == 1
    record = json.loads((input_path.parent / "output" / result.evidence[0].path).read_text())
    assert record["returncode"] == 0
    assert record["stdout"] == "observed\n"
    assert record["environment_names"] == ["EXPECTED", "TRTLLM_MODELING_V2"]
    assert "shell" not in record
    assert result.payload["result_kind"] == "deterministic_gate"
    assert result.payload["candidate_attempt_id"] == "coder-1"
    assert result.payload["receipt"]["passed"] is True
    assert result.payload["receipt"]["certification_mode"] == "LOCAL"
    assert result.payload["receipt"]["placements"] == []
    assert result.payload["evidence"][0]["path"] == result.evidence[0].path


def test_gate_failure_publishes_one_rejected_typed_receipt(tmp_path: Path) -> None:
    runtime: dict[str, artifacts.JsonValue] = {
        "schema_version": 1,
        "kind": "gate",
        "candidate_attempt_id": "coder-1",
        "candidate_digest": CANDIDATE_DIGEST,
        "command": {
            "command_id": "fails",
            "argv": [sys.executable, "-c", "raise SystemExit(7)"],
            "environment": {"TRTLLM_MODELING_V2": "require"},
            "cwd": ".",
        },
        "receipt": {
            "gate_id": "fails",
            "purpose": "correctness",
            "scope": "cpu_static",
            "certification_mode": "LOCAL",
            "expected_world_size": 0,
            "expected_rank": None,
            "expected_local_rank": None,
            "product_rank_body": False,
            "accuracy": None,
        },
    }
    input_path, worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=runtime,
    )

    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.REJECTED
    assert len(result.evidence) == 1
    assert not (worktree / "sentinel").exists()
    record = json.loads((input_path.parent / "output" / result.evidence[0].path).read_text())
    assert record["returncode"] == 7
    assert result.payload["receipt"]["passed"] is False


def test_gate_accepts_exact_single_gpu_placement_and_rejects_collective_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLURM_NTASKS", "1")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURMD_NODENAME", socket.gethostname())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("HOSTNAME", "task-spoofed-node")
    runtime: dict[str, artifacts.JsonValue] = {
        "schema_version": 1,
        "kind": "gate",
        "candidate_attempt_id": "coder-1",
        "candidate_digest": CANDIDATE_DIGEST,
        "command": {
            "command_id": "entry-gpu",
            "argv": [sys.executable, "-c", "print('passed')"],
            "environment": {"TRTLLM_MODELING_V2": "require"},
            "cwd": ".",
        },
        "receipt": {
            "gate_id": "entry-gpu",
            "purpose": "correctness",
            "scope": "single_gpu_product",
            "certification_mode": "LOCAL",
            "expected_world_size": 1,
            "expected_rank": 0,
            "expected_local_rank": 0,
            "product_rank_body": True,
            "accuracy": None,
        },
    }
    input_path, _worktree = _publish_input(tmp_path, role=Role.GATE, runtime=runtime)
    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert result.payload["receipt"]["scope"] == "single_gpu_product"
    assert result.payload["receipt"]["placements"] == [
        {"rank": 0, "node": socket.gethostname(), "local_rank": 0}
    ]
    record = json.loads((input_path.parent / "output" / result.evidence[0].path).read_text())
    assert record["environment_names"] == [
        "CUDA_VISIBLE_DEVICES",
        "SLURMD_NODENAME",
        "SLURM_LOCALID",
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "TRTLLM_MODELING_V2",
    ]

    collective_root = tmp_path / "collective"
    collective_root.mkdir()
    collective = json.loads(json.dumps(runtime))
    collective["receipt"]["scope"] = "local_four_gpu_product"
    collective["receipt"]["expected_world_size"] = 4
    collective["receipt"]["expected_rank"] = None
    collective["receipt"]["expected_local_rank"] = None
    input_path, _worktree = _publish_input(
        collective_root,
        role=Role.GATE,
        runtime=collective,
    )
    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "supports only CPU_STATIC or SINGLE_GPU_PRODUCT" in result.summary


@pytest.mark.parametrize(
    ("missing", "override", "expected"),
    [
        ("SLURM_NTASKS", None, "runtime SLURM_NTASKS must be"),
        (None, ("CUDA_VISIBLE_DEVICES", "0,1"), "exactly one numeric CUDA device"),
        (None, ("SLURM_PROCID", "1"), "differs from expected"),
    ],
)
def test_single_gpu_gate_rejects_missing_malformed_or_mismatched_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str | None,
    override: tuple[str, str] | None,
    expected: str,
) -> None:
    _set_single_gpu_runtime_environment(monkeypatch)
    if missing is not None:
        monkeypatch.delenv(missing)
    if override is not None:
        monkeypatch.setenv(*override)
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=_gate_runtime(gpu=True),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert expected in result.summary
    assert not result.evidence


def test_single_gpu_gate_rejects_ambient_node_spoof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_single_gpu_runtime_environment(monkeypatch)
    monkeypatch.setenv("SLURMD_NODENAME", "spoofed-node")
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=_gate_runtime(gpu=True),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "differs from the worker hostname" in result.summary


def test_tuner_worker_publishes_measurements_with_trusted_evidence_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(),
        profile=DomainProfile.TUNER,
    )
    evidence = worktree / "reports" / "probe.json"
    evidence.parent.mkdir()
    evidence.write_text('{"samples":[1.0]}\n', encoding="utf-8")
    response = {
        "schema_version": 1,
        "outcome": "coder_result",
        "status": "succeeded",
        "summary": "matched arms measured",
        "evidence_paths": ["reports/probe.json"],
        "payload": {
            "schema_version": 1,
            "result_kind": "tuner_measurement",
            "baseline": {},
            "candidate": {},
        },
    }
    monkeypatch.setattr(worker, "_invoke_agent", lambda **_kwargs: json.dumps(response))

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert result.payload["result_kind"] == "tuner_measurement"
    assert result.payload["evidence"] == [
        {
            "path": result.evidence[0].path,
            "sha256": result.evidence[0].sha256,
            "size_bytes": result.evidence[0].size_bytes,
        }
    ]

    second = tmp_path / "with-worker-decision"
    second.mkdir()
    input_path, worktree = _publish_input(
        second,
        role=Role.CODER,
        runtime=_agent_runtime(),
        profile=DomainProfile.TUNER,
    )
    evidence = worktree / "reports" / "probe.json"
    evidence.parent.mkdir()
    evidence.write_text('{"samples":[1.0]}\n', encoding="utf-8")
    response["payload"]["promotion_decision"] = "keep"  # type: ignore[index]
    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "Tuner agent payload keys invalid" in result.summary


def test_cpu_gate_accepts_exact_single_task_slurm_step_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in {
        "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_STEP_NUM_TASKS": "1",
        "SLURM_TASKS_PER_NODE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=_gate_runtime(gpu=False),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert result.payload["receipt"]["scope"] == "cpu_static"
    assert result.payload["receipt"]["placements"] == []
    record = json.loads((input_path.parent / "output" / result.evidence[0].path).read_text())
    assert record["environment_names"] == ["TRTLLM_MODELING_V2"]


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("NVIDIA_VISIBLE_DEVICES", "none"),
        ("WORLD_SIZE", "1"),
        ("RANK", "0"),
        ("LOCAL_RANK", "0"),
        ("SLURM_GPUS_ON_NODE", "0"),
    ],
)
def test_cpu_gate_rejects_gpu_or_distributed_runtime_contamination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=_gate_runtime(gpu=False),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "CPU-static gate received GPU/distributed placement environment" in result.summary
    assert not result.evidence


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SLURM_NTASKS", "2"),
        ("SLURM_NTASKS", "garbage"),
        ("SLURM_PROCID", "1"),
        ("SLURM_PROCID", "00"),
        ("SLURM_LOCALID", "1"),
        ("SLURM_STEP_NUM_TASKS", "2"),
        ("SLURM_TASKS_PER_NODE", "1(x1)"),
    ],
)
def test_cpu_gate_rejects_incomplete_or_noncanonical_slurm_step_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    environment = {
        "SLURM_NTASKS": "1",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
        "SLURM_STEP_NUM_TASKS": "1",
        "SLURM_TASKS_PER_NODE": "1",
    }
    environment[name] = value
    for env_name, env_value in environment.items():
        monkeypatch.setenv(env_name, env_value)
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=_gate_runtime(gpu=False),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "exact single-task Slurm launcher environment" in result.summary
    assert not result.evidence


@pytest.mark.parametrize("injection", ["receipt", "command"])
def test_gate_rejects_task_supplied_placement_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injection: str,
) -> None:
    _set_single_gpu_runtime_environment(monkeypatch)
    runtime = _gate_runtime(gpu=True)
    if injection == "receipt":
        receipt = runtime["receipt"]
        assert isinstance(receipt, dict)
        receipt["placements"] = [{"rank": 0, "node": socket.gethostname(), "local_rank": 0}]
    else:
        command = runtime["command"]
        assert isinstance(command, dict)
        environment = command["environment"]
        assert isinstance(environment, dict)
        environment["CUDA_VISIBLE_DEVICES"] = "7"
    input_path, _worktree = _publish_input(tmp_path, role=Role.GATE, runtime=runtime)

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    if injection == "receipt":
        assert "unknown=['placements']" in result.summary
    else:
        assert "cannot supply scheduler placement environment" in result.summary


def test_gate_rejects_credential_environment_without_running_command(tmp_path: Path) -> None:
    runtime: dict[str, artifacts.JsonValue] = {
        "schema_version": 1,
        "kind": "gate",
        "candidate_attempt_id": "coder-1",
        "candidate_digest": CANDIDATE_DIGEST,
        "command": {
            "command_id": "must-not-run",
            "argv": [sys.executable, "-c", "open('sentinel', 'w').write('bad')"],
            "environment": {
                "OPENAI_API_KEY": "secret",
                "TRTLLM_MODELING_V2": "require",
            },
            "cwd": ".",
        },
        "receipt": {
            "gate_id": "must-not-run",
            "purpose": "structure",
            "scope": "cpu_static",
            "certification_mode": "LOCAL",
            "expected_world_size": 0,
            "expected_rank": None,
            "expected_local_rank": None,
            "product_rank_body": False,
            "accuracy": None,
        },
    }
    input_path, worktree = _publish_input(tmp_path, role=Role.GATE, runtime=runtime)
    result = worker.execute_worker(input_path)
    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert not (worktree / "sentinel").exists()
    assert "cannot receive credentials" in result.summary


def test_gate_rejects_certification_mode_that_exceeds_evidence_scope(
    tmp_path: Path,
) -> None:
    runtime = _gate_runtime(gpu=False)
    receipt = runtime["receipt"]
    assert isinstance(receipt, dict)
    receipt["certification_mode"] = "REAL"
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.GATE,
        runtime=runtime,
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "requires certification_mode 'LOCAL'" in result.summary
    assert not result.evidence


def test_generic_manifest_detection_does_not_capture_legacy_role_input(tmp_path: Path) -> None:
    legacy = tmp_path / "role-input.json"
    legacy.write_text('{"schema_version": 1, "role": "coder"}\n', encoding="utf-8")
    assert not worker.is_worker_input_manifest(legacy)

    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(),
    )
    assert worker.is_worker_input_manifest(input_path)


def test_internal_routes_generic_manifest_to_worker_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(),
    )
    observed: list[Path] = []
    monkeypatch.setattr(worker, "execute_worker", lambda path: observed.append(path))

    internal.main(["role-worker", "--input", str(input_path)])
    assert observed == [input_path]


def test_internal_preserves_legacy_planning_role_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = (tmp_path / "role-input.json").resolve()
    input_path.write_text('{"schema_version": 1, "role": "plan_drafter"}\n', encoding="utf-8")
    sentinel = object()
    observed: list[object] = []
    monkeypatch.setattr(runners, "load_role_spec", lambda path: sentinel)
    monkeypatch.setattr(runners, "execute_role", lambda spec: observed.append(spec))

    internal.main(["role-worker", "--input", str(input_path)])
    assert observed == [sentinel]


def test_relative_input_path_is_rejected_before_execution(tmp_path: Path) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(),
    )
    assert input_path.name == artifacts.INPUT_FILENAME
    with pytest.raises(worker.WorkerRuntimeError, match="must be absolute"):
        worker.execute_worker(Path(artifacts.INPUT_FILENAME))


def test_no_credential_state_clears_ambient_for_agent_only_and_restores_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(evidence_paths=[]),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-worker-secret")

    def invoke(**_kwargs: object) -> str:
        assert "OPENAI_API_KEY" not in os.environ
        return _agent_response(evidence_paths=[])

    monkeypatch.setattr(worker, "_invoke_agent", invoke)
    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert os.environ["OPENAI_API_KEY"] == "ambient-worker-secret"
    assert "ambient-worker-secret" not in (
        input_path.parent / "output" / artifacts.RESULT_FILENAME
    ).read_text(encoding="utf-8")


def test_bundle_credentials_are_scoped_and_never_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-worker-secret-never-persist"
    binding = CredentialBinding(
        "run-1",
        "item-1",
        "attempt-1",
        TASK_DIGEST,
        1,
        "codex",
    )
    descriptor = CredentialDescriptor.create(
        binding,
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "attempt-1", 2_147_483_647),
    )
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(
            evidence_paths=[],
            credential_descriptor=descriptor.to_public_dict(),
        ),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-value")
    monkeypatch.setattr(
        worker,
        "load_worker_credentials",
        lambda *_args, **_kwargs: {"OPENAI_API_KEY": secret},
    )

    def invoke(**_kwargs: object) -> str:
        assert os.environ["OPENAI_API_KEY"] == secret
        return _agent_response(evidence_paths=[])

    monkeypatch.setattr(worker, "_invoke_agent", invoke)
    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert os.environ["OPENAI_API_KEY"] == "ambient-value"
    assert secret not in input_path.read_text(encoding="utf-8")
    assert secret not in (input_path.parent / "output" / artifacts.RESULT_FILENAME).read_text(
        encoding="utf-8"
    )
    assert descriptor.descriptor_digest in input_path.read_text(encoding="utf-8")


def test_credentialed_coder_cannot_put_secret_in_non_evidence_candidate_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "synthetic-exact-candidate-secret-47"
    binding = CredentialBinding(
        "run-1",
        "item-1",
        "attempt-1",
        TASK_DIGEST,
        1,
        "codex",
    )
    descriptor = CredentialDescriptor.create(
        binding,
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "attempt-1", 2_147_483_647),
    )
    input_path, worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(
            evidence_paths=[],
            credential_descriptor=descriptor.to_public_dict(),
        ),
    )
    monkeypatch.setattr(
        worker,
        "load_worker_credentials",
        lambda *_args, **_kwargs: {"OPENAI_API_KEY": secret},
    )

    def invoke(**_kwargs: object) -> str:
        candidate = worktree / "catalog" / "entry.py"
        candidate.parent.mkdir()
        prefix = b"\x00" * (65_536 - len(secret) // 2)
        candidate.write_bytes(prefix + secret.encode("utf-8") + b"\xffsuffix")
        response = json.loads(_agent_response(evidence_paths=[]))
        response["payload"]["changed_paths"] = ["catalog/entry.py"]
        return json.dumps(response)

    monkeypatch.setattr(worker, "_invoke_agent", invoke)
    result = worker.execute_worker(input_path)
    durable = (input_path.parent / "output" / artifacts.RESULT_FILENAME).read_text(encoding="utf-8")

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert result.candidate_digest is None
    assert secret not in result.summary
    assert secret not in durable
    assert not (tmp_path / "workspace" / "controller-candidates").exists()


@pytest.mark.parametrize(
    "secret_shape",
    [
        b"Authorization: Basic dXNlcjpwYXNzd29yZA==",
        b"endpoint=https://user:password@example.invalid/v1",
        b"OPENAI_API_KEY=opaqueSyntheticTokenValue123456",
    ],
)
def test_coder_candidate_rejects_central_secret_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secret_shape: bytes,
) -> None:
    input_path, worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(evidence_paths=[]),
    )

    def invoke(**_kwargs: object) -> str:
        candidate = worktree / "catalog" / "entry.py"
        candidate.parent.mkdir()
        candidate.write_bytes(b"\x00binary-prefix\n" + secret_shape + b"\n")
        response = json.loads(_agent_response(evidence_paths=[]))
        response["payload"]["changed_paths"] = ["catalog/entry.py"]
        return json.dumps(response)

    monkeypatch.setattr(worker, "_invoke_agent", invoke)
    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert result.candidate_digest is None
    assert "secret-shaped material" in result.summary


def test_agent_response_and_evidence_credential_leaks_fail_without_secret_in_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-response-and-evidence-secret"
    binding = CredentialBinding(
        "run-1",
        "item-1",
        "attempt-1",
        TASK_DIGEST,
        1,
        "codex",
    )
    descriptor = CredentialDescriptor.create(
        binding,
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "attempt-1", 2_147_483_647),
    )
    runtime = _agent_runtime(
        credential_descriptor=descriptor.to_public_dict(),
    )
    input_path, worktree = _publish_input(tmp_path, role=Role.CODER, runtime=runtime)
    monkeypatch.setattr(
        worker,
        "load_worker_credentials",
        lambda *_args, **_kwargs: {"OPENAI_API_KEY": secret},
    )
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: _agent_response(evidence_paths=["reports/probe.json"]).replace(
            "focused probe passed", secret
        ),
    )
    result = worker.execute_worker(input_path)
    result_text = (input_path.parent / "output" / artifacts.RESULT_FILENAME).read_text(
        encoding="utf-8"
    )
    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert secret not in result.summary
    assert secret not in result_text

    second = tmp_path / "second"
    second.mkdir()
    input_path, worktree = _publish_input(second, role=Role.CODER, runtime=runtime)
    evidence = worktree / "reports" / "probe.json"
    evidence.parent.mkdir()
    evidence.write_text(f"leaked={secret}\n", encoding="utf-8")
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: _agent_response(evidence_paths=["reports/probe.json"]),
    )
    result = worker.execute_worker(input_path)
    result_text = (input_path.parent / "output" / artifacts.RESULT_FILENAME).read_text(
        encoding="utf-8"
    )
    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert not result.evidence
    assert secret not in result.summary
    assert secret not in result_text


@pytest.mark.parametrize(
    ("client", "auth_name"),
    [("sbatch", None), (None, "SLURM_JWT")],
)
def test_agent_negative_preflight_rejects_scheduler_capability_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: str | None,
    auth_name: str | None,
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(evidence_paths=[]),
    )
    invoked = False

    def invoke(**_kwargs: object) -> str:
        nonlocal invoked
        invoked = True
        return _agent_response(evidence_paths=[])

    if client is not None:
        monkeypatch.setattr(
            worker,
            "_scheduler_client_path",
            lambda name: f"/usr/bin/{name}" if name == client else None,
        )
    if auth_name is not None:
        monkeypatch.setenv(auth_name, "scheduler-auth-must-not-persist")
    monkeypatch.setattr(worker, "_invoke_agent", invoke)

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert not invoked
    durable = (input_path.parent / "output" / artifacts.RESULT_FILENAME).read_text(encoding="utf-8")
    assert "scheduler-auth-must-not-persist" not in durable
    assert "negative preflight" in result.summary


def test_reviewer_rejects_candidate_overlay_mutation_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, worktree = _publish_input(
        tmp_path,
        role=Role.REVIEWER,
        runtime=_agent_runtime(candidate_digest=CANDIDATE_DIGEST, evidence_paths=[]),
    )
    (worktree / "model.py").write_text("MUTATED = True\n", encoding="utf-8")
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: pytest.fail("backend must not run after overlay mutation"),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "overlay digest mismatch" in result.summary


def test_agent_rejects_unbound_network_launch_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(evidence_paths=[]),
    )
    monkeypatch.setenv(AGENT_POLICY_DIGEST_ENVIRONMENT, "f" * 64)
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: pytest.fail("backend must not run without policy binding"),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "not bound by the trusted scheduler" in result.summary


@pytest.mark.parametrize(
    ("role", "outcome"),
    [(Role.CODER, "coder_result"), (Role.REVIEWER, "reviewer_result")],
)
def test_smith_worker_publishes_exact_job_and_node_placement_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: Role,
    outcome: str,
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=role,
        runtime=_agent_runtime(
            candidate_digest=(CANDIDATE_DIGEST if role is Role.REVIEWER else None),
            evidence_paths=[],
            distinct_nodes_required=True,
        ),
    )
    for name, value in {
        "SLURM_JOB_ID": "12345",
        "SLURM_CLUSTER_NAME": "alpha",
        "SLURMD_NODENAME": socket.gethostname(),
        "SLURM_JOB_NODELIST": socket.gethostname(),
        "SLURM_NNODES": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: _agent_response(outcome=outcome, evidence_paths=[]),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    receipt = load_worker_placement(input_path.parent / "output" / PLACEMENT_RECEIPT_FILENAME)
    assert (receipt.attempt_id, receipt.job_id, receipt.hostname) == (
        "attempt-1",
        "12345",
        socket.gethostname(),
    )


def test_qa_rejects_horizontal_smith_item_placement_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.QA,
        runtime=_agent_runtime(
            candidate_digest=CANDIDATE_DIGEST,
            evidence_paths=[],
            distinct_nodes_required=True,
        ),
    )
    monkeypatch.setattr(
        worker,
        "_invoke_agent",
        lambda **_kwargs: pytest.fail("QA must not join an item-level horizontal Smith wave"),
    )

    result = worker.execute_worker(input_path)

    assert result.status is artifacts.WorkerResultStatus.FAILED
    assert "Smith Coder/Reviewer jobs" in result.summary


def test_backend_standard_streams_are_redacted_before_slurm_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    secret = "sk-standard-stream-secret"
    binding = CredentialBinding(
        "run-1",
        "item-1",
        "attempt-1",
        TASK_DIGEST,
        1,
        "codex",
    )
    descriptor = CredentialDescriptor.create(
        binding,
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "attempt-1", 2_147_483_647),
    )
    input_path, _worktree = _publish_input(
        tmp_path,
        role=Role.CODER,
        runtime=_agent_runtime(
            evidence_paths=[],
            credential_descriptor=descriptor.to_public_dict(),
        ),
    )
    monkeypatch.setattr(
        worker,
        "load_worker_credentials",
        lambda *_args, **_kwargs: {"OPENAI_API_KEY": secret},
    )

    def invoke(**_kwargs: object) -> str:
        os.write(1, f"stdout={secret}\n".encode())
        os.write(2, f"stderr={secret}\n".encode())
        return _agent_response(evidence_paths=[])

    monkeypatch.setattr(worker, "_invoke_agent", invoke)
    result = worker.execute_worker(input_path)
    captured = capfd.readouterr()

    assert result.status is artifacts.WorkerResultStatus.SUCCEEDED
    assert secret not in captured.out
    assert secret not in captured.err
    assert "stdout=<redacted>" in captured.out
    assert "stderr=<redacted>" in captured.err
