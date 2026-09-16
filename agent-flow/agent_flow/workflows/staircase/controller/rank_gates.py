# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller bridge from one collective GateSpec to trusted rank evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from ..common.artifacts import EvidenceFile, JsonValue, WorkerResultStatus
from ..common.gates import (
    ClaimScope,
    EvidenceScope,
    GateCommand,
    GatePolicyError,
    GatePurpose,
    GateReceipt,
    GateSpec,
    RankPlacement,
    validate_receipt_claim,
)
from ..common.launchers import (
    ExpectedProductIdentity,
    RankAllocation,
    RankLaunchPlan,
    RankPlacementReceipt,
    build_single_node_multi_rank_plan,
    build_slurm_multi_node_product_plan,
    build_synthetic_multi_node_plan,
)
from ..common.rank_worker import RankInput, RankWorkerError, load_rank_input, publish_rank_input
from ..common.supervisor import RankSupervisionResult
from ..task_schema import CertificationMode, ParallelMapping, ResourceClass
from .results import CandidateReceipt

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class RankGateContractError(RuntimeError):
    """Raised when collective gate preparation or completion fails closed."""


class CollectiveCertification(str, Enum):
    """Evidence boundary frozen before a collective attempt starts."""

    REAL_PRODUCT = "real_multi_node_product"
    LOCAL_PRODUCT = "local_four_gpu_product"
    SYNTHETIC_RUNNER = "synthetic_multi_node_runner"

    @property
    def certification_mode(self) -> CertificationMode:
        """Return the task/result-facing certification boundary."""
        return {
            CollectiveCertification.REAL_PRODUCT: CertificationMode.REAL,
            CollectiveCertification.LOCAL_PRODUCT: CertificationMode.LOCAL,
            CollectiveCertification.SYNTHETIC_RUNNER: CertificationMode.SYNTHETIC,
        }[self]


@dataclass(frozen=True, slots=True)
class RankGateSupervisorExecution:
    """Exact non-executing contract consumed by the trusted supervisor."""

    plan: RankLaunchPlan
    cwd: Path
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    plan_digest: str
    rank_command_digest: str
    rank_input_path: Path
    rank_input_digest: str
    report_directory: Path

    def __post_init__(self) -> None:
        if self.argv != self.plan.argv or self.environment != self.plan.environment:
            raise RankGateContractError("supervisor command differs from immutable rank plan")
        if self.plan_digest != self.plan.plan_digest:
            raise RankGateContractError("supervisor plan digest differs from immutable rank plan")
        if self.rank_command_digest != self.plan.rank_command_digest:
            raise RankGateContractError("supervisor command digest differs from rank body")
        if self.rank_input_path != self.plan.rank_input_path:
            raise RankGateContractError("supervisor rank input path differs from rank plan")


@dataclass(frozen=True, slots=True)
class RankGateAttempt:
    """Frozen controller identity and execution contract for one GateSpec."""

    attempt_id: str
    candidate: CandidateReceipt
    gate_spec: GateSpec
    certification: CollectiveCertification
    resource: ResourceClass
    rank_input: RankInput
    supervisor: RankGateSupervisorExecution

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.attempt_id) is None:
            raise RankGateContractError("rank gate attempt_id must be a safe identifier")
        if self.attempt_id == self.candidate.attempt_id:
            raise RankGateContractError("rank gate must use a fresh attempt identity")
        if not isinstance(self.gate_spec, GateSpec):
            raise RankGateContractError("rank gate attempt requires exactly one GateSpec")
        if not isinstance(self.certification, CollectiveCertification):
            raise RankGateContractError("rank gate certification must use its typed enum")
        _validate_rank_input_identity(self.rank_input, self.supervisor)


@dataclass(frozen=True, slots=True)
class RankGateCompletion:
    """Manifest-ready successful result fragment for ``decode_gate_result``."""

    status: WorkerResultStatus
    summary: str
    candidate_digest: None
    reviewed_candidate_digest: str
    receipt: GateReceipt
    evidence: tuple[EvidenceFile, ...]
    payload: dict[str, JsonValue]

    def __post_init__(self) -> None:
        if self.status is not WorkerResultStatus.SUCCEEDED or not self.receipt.passed:
            raise RankGateContractError("collective completion may represent only verified success")
        if not self.summary.strip():
            raise RankGateContractError("collective completion summary must be non-empty")
        if not self.evidence:
            raise RankGateContractError("collective completion requires immutable evidence")


def prepare_rank_gate_attempt(
    *,
    attempt_id: str,
    candidate: CandidateReceipt,
    gate_spec: GateSpec,
    certification: CollectiveCertification,
    resource: ResourceClass,
    mapping: ParallelMapping,
    allocation: RankAllocation,
    body_command: GateCommand,
    cwd: Path,
    rank_input_path: Path,
    report_directory: Path,
    expected_product_identity: ExpectedProductIdentity | None = None,
) -> RankGateAttempt:
    """Validate and publish one immutable collective gate execution contract.

    This function performs only controller-owned filesystem publication. It
    does not invoke the supervisor, ``srun``, or any scheduler operation.
    """
    if not isinstance(candidate, CandidateReceipt):
        raise RankGateContractError("candidate must use a verified CandidateReceipt")
    if not isinstance(gate_spec, GateSpec):
        raise RankGateContractError("collective attempt requires exactly one GateSpec")
    if not isinstance(certification, CollectiveCertification):
        raise RankGateContractError("certification must use CollectiveCertification")
    if not isinstance(body_command, GateCommand) or body_command != gate_spec.command:
        raise RankGateContractError("rank body command differs from frozen GateSpec")
    if _SAFE_ID.fullmatch(attempt_id) is None or attempt_id == candidate.attempt_id:
        raise RankGateContractError("collective gate requires a safe fresh attempt_id")
    _validate_resource_topology(resource, mapping, allocation)
    if certification is CollectiveCertification.SYNTHETIC_RUNNER and gate_spec.purpose in {
        GatePurpose.ACCURACY,
        GatePurpose.PERFORMANCE,
    }:
        raise RankGateContractError(
            "synthetic runner evidence cannot satisfy accuracy or performance GateSpec"
        )
    try:
        if certification is CollectiveCertification.REAL_PRODUCT:
            if expected_product_identity is None:
                raise RankGateContractError(
                    "real product certification requires expected execution identity"
                )
            plan = build_slurm_multi_node_product_plan(
                mapping=mapping,
                world_size=resource.total_tasks,
                allocation=allocation,
                command=body_command,
                rank_input_path=rank_input_path,
                expected_product_identity=expected_product_identity,
                cpus_per_rank=resource.cpus_per_task,
            )
        elif certification is CollectiveCertification.LOCAL_PRODUCT:
            if expected_product_identity is not None:
                raise RankGateContractError(
                    "local product certification cannot claim multi-node product identity"
                )
            plan = build_single_node_multi_rank_plan(
                mapping=mapping,
                world_size=resource.total_tasks,
                allocation=allocation,
                command=body_command,
                rank_input_path=rank_input_path,
                cpus_per_rank=resource.cpus_per_task,
            )
        else:
            if expected_product_identity is not None:
                raise RankGateContractError(
                    "synthetic runner certification cannot carry product identity"
                )
            plan = build_synthetic_multi_node_plan(
                mapping=mapping,
                world_size=resource.total_tasks,
                allocation=allocation,
                command=body_command,
                rank_input_path=rank_input_path,
                cpus_per_rank=resource.cpus_per_task,
            )
        rank_input, rank_input_digest = publish_rank_input(
            plan,
            cwd=cwd,
            report_directory=report_directory,
        )
    except (OSError, TypeError, ValueError, RankWorkerError) as error:
        raise RankGateContractError(f"could not prepare collective rank gate: {error}") from error
    supervisor = RankGateSupervisorExecution(
        plan=plan,
        cwd=rank_input.cwd,
        argv=plan.argv,
        environment=plan.environment,
        plan_digest=plan.plan_digest,
        rank_command_digest=plan.rank_command_digest,
        rank_input_path=plan.rank_input_path,
        rank_input_digest=rank_input_digest,
        report_directory=rank_input.report_directory,
    )
    return RankGateAttempt(
        attempt_id=attempt_id,
        candidate=candidate,
        gate_spec=gate_spec,
        certification=certification,
        resource=resource,
        rank_input=rank_input,
        supervisor=supervisor,
    )


def complete_rank_gate_attempt(
    attempt: RankGateAttempt,
    supervision: RankSupervisionResult,
    *,
    evidence: Sequence[EvidenceFile],
) -> RankGateCompletion:
    """Convert complete supervisor proof into the strict gate-result payload."""
    if not isinstance(attempt, RankGateAttempt):
        raise RankGateContractError("completion requires a RankGateAttempt")
    if not isinstance(supervision, RankSupervisionResult):
        raise RankGateContractError("completion requires a successful RankSupervisionResult")
    evidence_tuple = tuple(evidence)
    if not evidence_tuple or not all(isinstance(item, EvidenceFile) for item in evidence_tuple):
        raise RankGateContractError("collective gate requires non-empty typed evidence")
    if len({item.path for item in evidence_tuple}) != len(evidence_tuple):
        raise RankGateContractError("collective gate evidence paths must be unique")
    _revalidate_rank_input(attempt)
    placement_receipt = supervision.receipt
    _validate_supervisor_receipt(attempt, placement_receipt)
    gate_receipt = _gate_receipt(attempt, placement_receipt)
    evidence_payload: list[JsonValue] = [
        {
            "path": item.path,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in evidence_tuple
    ]
    payload: dict[str, JsonValue] = {
        "schema_version": 1,
        "result_kind": "deterministic_gate",
        "certification_mode": attempt.certification.certification_mode.value,
        "candidate_attempt_id": attempt.candidate.attempt_id,
        "candidate_digest": attempt.candidate.candidate_digest,
        "receipt": _gate_receipt_payload(gate_receipt),
        "evidence": evidence_payload,
    }
    return RankGateCompletion(
        status=WorkerResultStatus.SUCCEEDED,
        summary=(
            f"collective gate {attempt.gate_spec.gate_id!r} passed with "
            f"{attempt.certification.value} evidence"
        ),
        candidate_digest=None,
        reviewed_candidate_digest=attempt.candidate.candidate_digest,
        receipt=gate_receipt,
        evidence=evidence_tuple,
        payload=payload,
    )


def _validate_resource_topology(
    resource: ResourceClass,
    mapping: ParallelMapping,
    allocation: RankAllocation,
) -> None:
    if not isinstance(resource, ResourceClass) or resource.name != "deterministic_gate":
        raise RankGateContractError("collective gate requires deterministic_gate ResourceClass")
    values = (
        resource.nodes,
        resource.tasks_per_node,
        resource.gpus_per_node,
        resource.cpus_per_task,
        resource.memory_mib,
        resource.time_limit_seconds,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise RankGateContractError("collective resource values must be positive integers")
    if not isinstance(mapping, ParallelMapping) or not isinstance(allocation, RankAllocation):
        raise RankGateContractError("collective topology requires typed Mapping and allocation")
    if len(allocation.nodes) != resource.nodes:
        raise RankGateContractError("collective resource and allocation node counts differ")
    if resource.tasks_per_node != resource.gpus_per_node:
        raise RankGateContractError("collective execution requires exactly one GPU per rank")
    if any(len(node.gpu_ids) != resource.gpus_per_node for node in allocation.nodes):
        raise RankGateContractError("allocation GPUs per node differ from ResourceClass")
    if allocation.world_size != resource.total_tasks or resource.total_gpus != resource.total_tasks:
        raise RankGateContractError("collective world size differs from resource topology")
    mapping_world_size = mapping.tensor_parallel_size * mapping.pipeline_parallel_size
    if mapping_world_size != resource.total_tasks:
        raise RankGateContractError("Mapping world size differs from ResourceClass")
    if len({node.hostname for node in allocation.nodes}) != resource.nodes:
        raise RankGateContractError("collective allocation must use exact distinct nodes")


def _validate_rank_input_identity(
    rank_input: RankInput, supervisor: RankGateSupervisorExecution
) -> None:
    plan = supervisor.plan
    expected_body = "product" if plan.product_rank_body else "synthetic"
    if (
        rank_input.plan_digest != supervisor.plan_digest
        or rank_input.rank_command_digest != supervisor.rank_command_digest
        or rank_input.launch_kind is not plan.kind
        or rank_input.body_kind != expected_body
        or rank_input.argv != plan.rank_command.argv
        or rank_input.environment != tuple(sorted(plan.rank_command.environment))
        or rank_input.cwd != supervisor.cwd
        or rank_input.report_directory != supervisor.report_directory
        or rank_input.world_size != plan.world_size
        or rank_input.placements != plan.placements
        or rank_input.expected_product_identity != plan.expected_product_identity
    ):
        raise RankGateContractError("rank input identity differs from immutable supervisor plan")


def _revalidate_rank_input(attempt: RankGateAttempt) -> None:
    try:
        rank_input, rank_input_digest = load_rank_input(attempt.supervisor.rank_input_path)
    except RankWorkerError as error:
        raise RankGateContractError(f"rank input cannot be revalidated: {error}") from error
    if rank_input_digest != attempt.supervisor.rank_input_digest:
        raise RankGateContractError("rank input digest changed after attempt preparation")
    _validate_rank_input_identity(rank_input, attempt.supervisor)


def _validate_supervisor_receipt(attempt: RankGateAttempt, receipt: RankPlacementReceipt) -> None:
    plan = attempt.supervisor.plan
    if (
        receipt.plan_digest != attempt.supervisor.plan_digest
        or receipt.kind is not plan.kind
        or receipt.world_size != plan.world_size
        or receipt.placements != plan.placements
        or receipt.product_rank_body is not plan.product_rank_body
        or receipt.evidence_scope is not plan.evidence_scope
    ):
        raise RankGateContractError("supervisor receipt differs from frozen collective plan")
    expected_scope = {
        CollectiveCertification.REAL_PRODUCT: EvidenceScope.REAL_MULTI_NODE_PRODUCT,
        CollectiveCertification.LOCAL_PRODUCT: EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
        CollectiveCertification.SYNTHETIC_RUNNER: EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
    }[attempt.certification]
    expected_product_body = attempt.certification is not CollectiveCertification.SYNTHETIC_RUNNER
    if (
        receipt.evidence_scope is not expected_scope
        or receipt.product_rank_body is not expected_product_body
    ):
        raise RankGateContractError("supervisor receipt crosses the frozen certification boundary")
    if len({placement.hostname for placement in receipt.placements}) != attempt.resource.nodes:
        raise RankGateContractError(
            "supervisor receipt does not prove exact distinct-node placement"
        )


def _gate_receipt(attempt: RankGateAttempt, placement_receipt: RankPlacementReceipt) -> GateReceipt:
    receipt = GateReceipt(
        gate_id=attempt.gate_spec.gate_id,
        purpose=attempt.gate_spec.purpose,
        scope=placement_receipt.evidence_scope,
        placements=tuple(
            RankPlacement(
                rank=placement.rank,
                node=placement.hostname,
                local_rank=placement.local_rank,
            )
            for placement in placement_receipt.placements
        ),
        product_rank_body=placement_receipt.product_rank_body,
        passed=True,
        accuracy=attempt.gate_spec.accuracy,
    )
    claim = _claim_scope(attempt)
    try:
        validate_receipt_claim(receipt, claim, gate_spec=attempt.gate_spec)
    except GatePolicyError as error:
        raise RankGateContractError(
            f"collective receipt cannot support frozen claim: {error}"
        ) from error
    return receipt


def _claim_scope(attempt: RankGateAttempt) -> ClaimScope:
    if attempt.certification is CollectiveCertification.SYNTHETIC_RUNNER:
        return ClaimScope.SYNTHETIC_RUNNER
    if attempt.gate_spec.purpose is GatePurpose.ACCURACY:
        return ClaimScope.ACCURACY
    if attempt.gate_spec.purpose is GatePurpose.PERFORMANCE:
        return ClaimScope.PERFORMANCE
    if attempt.certification is CollectiveCertification.LOCAL_PRODUCT:
        return ClaimScope.LOCAL_FOUR_GPU_PRODUCT
    return ClaimScope.MULTI_NODE_PRODUCT


def _gate_receipt_payload(receipt: GateReceipt) -> dict[str, JsonValue]:
    accuracy: JsonValue
    if receipt.accuracy is None:
        accuracy = None
    else:
        accuracy = {
            "selector": receipt.accuracy.selector,
            "reference": receipt.accuracy.reference,
            "protocol": receipt.accuracy.protocol,
            "tolerance": receipt.accuracy.tolerance,
        }
    return {
        "gate_id": receipt.gate_id,
        "purpose": receipt.purpose.value,
        "scope": receipt.scope.value,
        "placements": [
            {
                "rank": placement.rank,
                "node": placement.node,
                "local_rank": placement.local_rank,
            }
            for placement in receipt.placements
        ],
        "product_rank_body": receipt.product_rank_body,
        "passed": receipt.passed,
        "accuracy": accuracy,
    }


__all__ = [
    "CollectiveCertification",
    "RankGateAttempt",
    "RankGateCompletion",
    "RankGateContractError",
    "RankGateSupervisorExecution",
    "complete_rank_gate_attempt",
    "prepare_rank_gate_attempt",
]
