# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the controller bridge to trusted collective rank evidence."""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.artifacts import (
    EvidenceFile,
    JsonValue,
    WorkerResultManifest,
    WorkerResultStatus,
)
from agent_flow.workflows.staircase.common.gates import (
    AccuracyCriteria,
    ClaimScope,
    GateCommand,
    GatePhase,
    GatePolicyError,
    GatePurpose,
    GateSpec,
    validate_receipt_claim,
)
from agent_flow.workflows.staircase.common.launchers import (
    AllocationNode,
    ExpectedProductIdentity,
    RankAllocation,
)
from agent_flow.workflows.staircase.common.rank_worker import (
    execute_rank,
    publish_product_identity_evidence,
)
from agent_flow.workflows.staircase.common.supervisor import (
    ProcessCapture,
    RankSupervisionResult,
    supervise_rank_launch,
)
from agent_flow.workflows.staircase.controller.rank_gates import (
    CollectiveCertification,
    RankGateAttempt,
    RankGateContractError,
    complete_rank_gate_attempt,
    prepare_rank_gate_attempt,
)
from agent_flow.workflows.staircase.controller.results import (
    CandidateDisposition,
    CandidateReceipt,
    GateResult,
    decode_gate_result,
)
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    JobReference,
    Role,
)
from agent_flow.workflows.staircase.task_schema import ParallelMapping, ResourceClass


def _candidate() -> CandidateReceipt:
    payload = {
        "schema_version": 1,
        "run_id": "run-1",
        "item_id": "collective-item",
        "attempt_id": "collective-item.0001.role",
        "disposition": "patch",
        "base_commit": "1" * 40,
        "candidate_commit": "2" * 40,
        "candidate_digest": "d" * 64,
        "changed_paths": ["tensorrt_llm/_torch/modeling_v2/model.py"],
        "index_delta": None,
    }
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return CandidateReceipt(
        run_id="run-1",
        item_id="collective-item",
        attempt_id="collective-item.0001.role",
        disposition=CandidateDisposition.PATCH,
        base_commit="1" * 40,
        candidate_commit="2" * 40,
        candidate_digest="d" * 64,
        changed_paths=("tensorrt_llm/_torch/modeling_v2/model.py",),
        index_delta=None,
        receipt_sha256=digest,
    )


def _resource(*, nodes: int = 2, tasks_per_node: int = 1, gpus_per_node: int = 1) -> ResourceClass:
    return ResourceClass(
        "deterministic_gate",
        nodes,
        tasks_per_node,
        gpus_per_node,
        4,
        4096,
        600,
    )


def _mapping(world_size: int = 2) -> ParallelMapping:
    return ParallelMapping(
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        moe_expert_parallel_size=1,
        moe_tensor_parallel_size=world_size,
        attention_data_parallel_size=1,
    )


def _allocation() -> RankAllocation:
    return RankAllocation(
        (AllocationNode("node-a", (0,)), AllocationNode("node-b", (0,))),
        slurm_job_id="1234",
    )


def _identity(tmp_path: Path) -> ExpectedProductIdentity:
    return ExpectedProductIdentity(
        repository_commit="2" * 40,
        python_tensorrt_llm_path=str(tmp_path / "tensorrt_llm" / "__init__.py"),
        native_build_identity="build-1",
        image_identity="image-1",
        compute_capability="10.0",
        collective_backend="nccl",
        transport="ib",
    )


def _spec(
    command: GateCommand | None = None,
    *,
    purpose: GatePurpose = GatePurpose.CORRECTNESS,
    accuracy: AccuracyCriteria | None = None,
) -> GateSpec:
    phase = GatePhase.ACCURACY if purpose is GatePurpose.ACCURACY else GatePhase.COLLECTIVE
    return GateSpec(
        "collective",
        phase,
        purpose,
        command or GateCommand(("python3", "-m", "pytest", "tests/collective.py")),
        accuracy=accuracy,
    )


def _prepare(
    tmp_path: Path,
    *,
    certification: CollectiveCertification = CollectiveCertification.REAL_PRODUCT,
    gate_spec: GateSpec | None = None,
    body_command: GateCommand | None = None,
    resource: ResourceClass | None = None,
    mapping: ParallelMapping | None = None,
) -> RankGateAttempt:
    spec = gate_spec or _spec()
    return prepare_rank_gate_attempt(
        attempt_id="collective-item.0002.gate",
        candidate=_candidate(),
        gate_spec=spec,
        certification=certification,
        resource=resource or _resource(),
        mapping=mapping or _mapping(),
        allocation=_allocation(),
        body_command=body_command or spec.command,
        cwd=tmp_path,
        rank_input_path=tmp_path / "rank-input.json",
        report_directory=tmp_path / "rank-reports",
        expected_product_identity=(
            _identity(tmp_path) if certification is CollectiveCertification.REAL_PRODUCT else None
        ),
    )


class _CollectiveExecutor:
    def __init__(self, attempt: RankGateAttempt) -> None:
        self.attempt = attempt
        self.calls: list[tuple[str, ...]] = []

    def __call__(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        environment: tuple[tuple[str, str], ...],
        shell: bool,
        capture_limit_bytes: int,
    ) -> ProcessCapture:
        self.calls.append(argv)
        plan = self.attempt.supervisor.plan
        for placement in plan.placements:

            def body_executor(
                _argv: tuple[str, ...],
                _cwd: Path,
                body_environment: tuple[tuple[str, str], ...],
                _shell: bool,
            ) -> int:
                evidence_path = dict(body_environment).get("STAIRCASE_PRODUCT_EVIDENCE_PATH")
                if evidence_path is not None:
                    publish_product_identity_evidence(
                        collective_backend="nccl",
                        transport="ib",
                        environment=dict(body_environment),
                    )
                return 0

            status = execute_rank(
                plan.rank_input_path,
                environment={
                    "CUDA_VISIBLE_DEVICES": str(placement.gpu_id),
                    "SLURM_PROCID": str(placement.rank),
                    "SLURM_LOCALID": str(placement.local_rank),
                    "SLURM_NTASKS": str(plan.world_size),
                },
                hostname_provider=lambda hostname=placement.hostname: hostname,
                executor=body_executor,
                identity_probe=lambda _cwd, binding: {
                    "python_tensorrt_llm_path": str(
                        self.attempt.supervisor.cwd / "tensorrt_llm" / "__init__.py"
                    ),
                    "compute_capability": "10.0",
                    "cuda_device_uuid": f"GPU-{binding.rank}",
                    "cuda_device_model": "Test GPU",
                },
            )
            assert status == 0
        return ProcessCapture(0, b"collective output", b"")


def _supervise(attempt: RankGateAttempt) -> RankSupervisionResult:
    executor = _CollectiveExecutor(attempt)
    result = supervise_rank_launch(
        attempt.supervisor.plan,
        cwd=attempt.supervisor.cwd,
        executor=executor,
    )
    assert executor.calls == [attempt.supervisor.argv]
    return result


def _evidence() -> tuple[EvidenceFile, ...]:
    return (EvidenceFile("evidence/collective.json", "f" * 64, 17),)


def _decode(attempt: RankGateAttempt, completion_payload: dict[str, JsonValue]) -> GateResult:
    candidate = attempt.candidate
    record = AttemptRecord(
        attempt.attempt_id,
        candidate.item_id,
        2,
        Role.GATE,
        AttemptKind.DETERMINISTIC_GATE,
        1,
        status=AttemptStatus.VALIDATED,
        profile=DomainProfile.ASSEMBLER,
        submission_token="gate-token",
        job=JobReference("1234"),
        result_digest="c" * 64,
    )
    manifest = WorkerResultManifest(
        candidate.run_id,
        candidate.item_id,
        attempt.attempt_id,
        "a" * 64,
        1,
        "b" * 64,
        WorkerResultStatus.SUCCEEDED,
        "collective gate completed",
        evidence=_evidence(),
        reviewed_candidate_digest=candidate.candidate_digest,
        payload=completion_payload,
    )
    return decode_gate_result(manifest, record, candidate, expected_spec=attempt.gate_spec)


def test_real_product_bridge_builds_fixed_contract_and_decodable_payload(
    tmp_path: Path,
) -> None:
    attempt = _prepare(tmp_path)

    assert attempt.supervisor.argv == attempt.supervisor.plan.argv
    assert attempt.supervisor.argv[-3:] == (
        "rank-worker",
        "--input",
        str(attempt.supervisor.rank_input_path),
    )
    assert attempt.rank_input.argv == attempt.gate_spec.command.argv
    assert attempt.rank_input.environment == tuple(sorted(attempt.gate_spec.command.environment))
    assert attempt.rank_input.plan_digest == attempt.supervisor.plan_digest
    assert attempt.rank_input.rank_command_digest == attempt.supervisor.rank_command_digest
    assert attempt.supervisor.rank_input_digest

    completion = complete_rank_gate_attempt(
        attempt,
        _supervise(attempt),
        evidence=_evidence(),
    )
    decoded = _decode(attempt, completion.payload)

    assert completion.status is WorkerResultStatus.SUCCEEDED
    assert completion.payload["certification_mode"] == "REAL"
    assert decoded.certification_mode.value == "REAL"
    assert decoded.receipt.scope.value == "real_multi_node_product"
    assert decoded.receipt.product_rank_body
    assert [placement.node for placement in decoded.receipt.placements] == ["node-a", "node-b"]


def test_synthetic_bridge_stays_runner_only_and_cannot_support_product_claim(
    tmp_path: Path,
) -> None:
    synthetic_spec = _spec(GateCommand(("python3", "synthetic_runner.py")))
    attempt = _prepare(
        tmp_path,
        certification=CollectiveCertification.SYNTHETIC_RUNNER,
        gate_spec=synthetic_spec,
    )

    completion = complete_rank_gate_attempt(
        attempt,
        _supervise(attempt),
        evidence=_evidence(),
    )

    assert completion.receipt.scope.value == "synthetic_multi_node_runner"
    assert not completion.receipt.product_rank_body
    validate_receipt_claim(completion.receipt, ClaimScope.SYNTHETIC_RUNNER)
    with pytest.raises(GatePolicyError, match="real topology-bearing"):
        validate_receipt_claim(completion.receipt, ClaimScope.MULTI_NODE_PRODUCT)


def test_synthetic_supervision_cannot_complete_frozen_product_contract(tmp_path: Path) -> None:
    product_root = tmp_path / "product"
    synthetic_root = tmp_path / "synthetic"
    product_root.mkdir()
    synthetic_root.mkdir()
    gate_spec = _spec(GateCommand(("python3", "collective_body.py")))
    product = _prepare(product_root, gate_spec=gate_spec)
    synthetic = _prepare(
        synthetic_root,
        certification=CollectiveCertification.SYNTHETIC_RUNNER,
        gate_spec=gate_spec,
    )

    with pytest.raises(RankGateContractError, match="differs from frozen"):
        complete_rank_gate_attempt(product, _supervise(synthetic), evidence=_evidence())


def test_product_accuracy_criteria_are_preserved_for_decoder(tmp_path: Path) -> None:
    criteria = AccuracyCriteria(
        "tests/accuracy.py::test_model",
        "golden-v1",
        "exact-match-v1",
        0.01,
    )
    attempt = _prepare(
        tmp_path,
        gate_spec=_spec(purpose=GatePurpose.ACCURACY, accuracy=criteria),
    )

    completion = complete_rank_gate_attempt(attempt, _supervise(attempt), evidence=_evidence())
    decoded = _decode(attempt, completion.payload)

    assert decoded.receipt.accuracy == criteria
    validate_receipt_claim(decoded.receipt, ClaimScope.ACCURACY, gate_spec=attempt.gate_spec)


def test_synthetic_accuracy_gate_is_rejected_before_input_publication(tmp_path: Path) -> None:
    criteria = AccuracyCriteria(
        "tests/accuracy.py::test_model",
        "golden-v1",
        "exact-match-v1",
        0.01,
    )
    with pytest.raises(RankGateContractError, match="cannot satisfy accuracy"):
        _prepare(
            tmp_path,
            certification=CollectiveCertification.SYNTHETIC_RUNNER,
            gate_spec=_spec(purpose=GatePurpose.ACCURACY, accuracy=criteria),
        )
    assert not (tmp_path / "rank-input.json").exists()


@pytest.mark.parametrize(
    ("resource", "mapping", "match"),
    [
        (_resource(nodes=1), _mapping(), "node counts"),
        (_resource(tasks_per_node=2), _mapping(), "one GPU per rank"),
        (_resource(gpus_per_node=2), _mapping(), "one GPU per rank"),
        (_resource(), _mapping(4), "Mapping world size"),
    ],
)
def test_resource_mapping_and_allocation_topology_must_agree(
    tmp_path: Path,
    resource: ResourceClass,
    mapping: ParallelMapping,
    match: str,
) -> None:
    with pytest.raises(RankGateContractError, match=match):
        _prepare(tmp_path, resource=resource, mapping=mapping)


def test_body_command_must_be_exact_frozen_gate_command(tmp_path: Path) -> None:
    with pytest.raises(RankGateContractError, match="differs from frozen GateSpec"):
        _prepare(tmp_path, body_command=GateCommand(("python3", "other.py")))


@pytest.mark.parametrize(
    "command",
    [
        GateCommand(("srun", "nested.py")),
        GateCommand(
            ("python3", "gate.py"),
            environment=(
                ("API_TOKEN", "secret"),
                ("TRTLLM_MODELING_V2", "require"),
            ),
        ),
    ],
)
def test_rank_body_rejects_nested_scheduler_or_credential_environment(
    tmp_path: Path, command: GateCommand
) -> None:
    with pytest.raises(RankGateContractError, match="could not prepare"):
        _prepare(tmp_path, gate_spec=_spec(command))


def test_completion_revalidates_immutable_rank_input_digest(tmp_path: Path) -> None:
    attempt = _prepare(tmp_path)
    supervision = _supervise(attempt)
    input_path = attempt.supervisor.rank_input_path
    input_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    payload = json.loads(input_path.read_text())
    payload["environment"].append(["EXTRA", "changed"])
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    input_path.chmod(stat.S_IRUSR)

    with pytest.raises(RankGateContractError, match="digest changed"):
        complete_rank_gate_attempt(attempt, supervision, evidence=_evidence())


def test_completion_rejects_receipt_for_another_plan(tmp_path: Path) -> None:
    attempt = _prepare(tmp_path)
    forged = replace(_supervise(attempt).receipt, plan_digest="0" * 64)

    with pytest.raises(RankGateContractError, match="differs from frozen"):
        complete_rank_gate_attempt(
            attempt,
            RankSupervisionResult(forged, b"", b""),
            evidence=_evidence(),
        )


def test_completion_requires_evidence_and_one_typed_gate_spec(tmp_path: Path) -> None:
    attempt = _prepare(tmp_path)
    supervision = _supervise(attempt)
    with pytest.raises(RankGateContractError, match="non-empty typed evidence"):
        complete_rank_gate_attempt(attempt, supervision, evidence=())

    with pytest.raises(RankGateContractError, match="exactly one GateSpec"):
        prepare_rank_gate_attempt(
            attempt_id="collective-item.0003.gate",
            candidate=_candidate(),
            gate_spec=(attempt.gate_spec,),
            certification=CollectiveCertification.REAL_PRODUCT,
            resource=_resource(),
            mapping=_mapping(),
            allocation=_allocation(),
            body_command=attempt.gate_spec.command,
            cwd=tmp_path,
            rank_input_path=tmp_path / "other-input.json",
            report_directory=tmp_path / "other-reports",
            expected_product_identity=_identity(tmp_path),
        )
