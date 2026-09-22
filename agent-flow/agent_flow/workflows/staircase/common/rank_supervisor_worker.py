# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted in-allocation worker for one collective deterministic gate.

The controller submits exactly one instance of this worker.  It derives the
allocation from scheduler-owned facts, publishes the immutable rank input,
invokes one validated ``srun`` step, and publishes the ordinary Staircase
worker result consumed by the existing controller result decoder.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, cast

from ..state import Role
from ..task_schema import CertificationMode, ParallelMapping, ResourceClass
from .artifacts import (
    OUTPUT_DIRECTORY,
    EvidenceFile,
    JsonValue,
    WorkerInputManifest,
    WorkerResultManifest,
    WorkerResultStatus,
    describe_evidence,
    load_input_manifest,
    publish_result,
)
from .gates import (
    AccuracyCriteria,
    EvidenceScope,
    GateCommand,
    GatePhase,
    GatePurpose,
    GateReceipt,
    GateSpec,
    RankPlacement,
)
from .launchers import (
    AllocationNode,
    ExpectedProductIdentity,
    LaunchKind,
    RankAllocation,
    RankLaunchPlan,
    build_single_node_multi_rank_plan,
    build_slurm_multi_node_product_plan,
    build_synthetic_multi_node_plan,
)
from .rank_worker import publish_rank_input
from .supervisor import (
    RankSupervisionResult,
    RankSupervisorError,
    resolve_trusted_scheduler_client,
    supervise_rank_launch,
)

RANK_SUPERVISOR_SCHEMA_VERSION = 2

_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_MAX_DIAGNOSTIC_CHARS = 8_000
_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "kind",
        "certification_mode",
        "candidate",
        "gate",
        "execution",
        "mapping",
        "resource",
        "rank_input_path",
        "rank_report_directory",
        "expected_product_identity",
    }
)


class RankSupervisorWorkerError(RuntimeError):
    """Raised when collective execution cannot preserve its frozen contract."""


@dataclass(frozen=True, slots=True)
class RankSupervisorInput:
    """Strict projection of the immutable standard worker input manifest."""

    manifest: WorkerInputManifest
    input_digest: str
    candidate_attempt_id: str
    candidate_digest: str
    candidate_commit: str
    candidate_receipt_sha256: str
    gate: GateSpec
    certification_mode: CertificationMode
    execution_scope: EvidenceScope
    expected_world_size: int
    expected_rank: int | None
    expected_local_rank: int | None
    product_rank_body: bool
    mapping: ParallelMapping
    resource: ResourceClass
    rank_input_path: Path
    rank_report_directory: Path
    expected_product_identity: ExpectedProductIdentity | None

    def __post_init__(self) -> None:
        if self.manifest.role is not Role.GATE:
            raise RankSupervisorWorkerError("rank supervisor input must use the Gate role")
        if _DIGEST.fullmatch(self.input_digest) is None:
            raise RankSupervisorWorkerError("rank supervisor input digest must be SHA-256")
        if not self.candidate_attempt_id or _DIGEST.fullmatch(self.candidate_digest) is None:
            raise RankSupervisorWorkerError("rank supervisor candidate identity is invalid")
        if _COMMIT.fullmatch(self.candidate_commit) is None:
            raise RankSupervisorWorkerError("candidate commit must be a lowercase Git SHA-1")
        if _DIGEST.fullmatch(self.candidate_receipt_sha256) is None:
            raise RankSupervisorWorkerError("candidate receipt identity must be SHA-256")
        if self.execution_scope not in {
            EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
            EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
            EvidenceScope.REAL_MULTI_NODE_PRODUCT,
        }:
            raise RankSupervisorWorkerError("rank supervisor requires a multi-rank scope")
        expected_mode = {
            EvidenceScope.LOCAL_FOUR_GPU_PRODUCT: CertificationMode.LOCAL,
            EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER: CertificationMode.SYNTHETIC,
            EvidenceScope.REAL_MULTI_NODE_PRODUCT: CertificationMode.REAL,
        }[self.execution_scope]
        if self.certification_mode is not expected_mode:
            raise RankSupervisorWorkerError(
                "rank supervisor certification mode differs from its evidence scope"
            )
        if self.resource.name != "deterministic_gate":
            raise RankSupervisorWorkerError(
                "rank supervisor resource class must be deterministic_gate"
            )
        resource_values = (
            self.resource.nodes,
            self.resource.tasks_per_node,
            self.resource.gpus_per_node,
            self.resource.cpus_per_task,
            self.resource.memory_mib,
            self.resource.time_limit_seconds,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in resource_values
        ):
            raise RankSupervisorWorkerError(
                "rank supervisor resource values must be positive integers"
            )
        if self.resource.tasks_per_node != self.resource.gpus_per_node:
            raise RankSupervisorWorkerError("rank supervisor requires one task per allocated GPU")
        world_size = self.mapping.tensor_parallel_size * self.mapping.pipeline_parallel_size
        if self.resource.total_tasks != world_size:
            raise RankSupervisorWorkerError(
                "resource world size differs from the frozen parallel mapping"
            )
        if self.expected_world_size != world_size:
            raise RankSupervisorWorkerError(
                "execution contract world size differs from resource and mapping"
            )
        if self.expected_rank is not None or self.expected_local_rank is not None:
            raise RankSupervisorWorkerError(
                "multi-node execution cannot predeclare a concrete rank binding"
            )
        expected_product_body = self.execution_scope in {
            EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
            EvidenceScope.REAL_MULTI_NODE_PRODUCT,
        }
        if self.product_rank_body is not expected_product_body:
            raise RankSupervisorWorkerError(
                "execution product-body identity differs from its scope"
            )
        if (
            self.execution_scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER
            and self.gate.phase is not GatePhase.COLLECTIVE
        ):
            raise RankSupervisorWorkerError(
                "synthetic runner evidence is limited to an explicit COLLECTIVE canary"
            )
        if self.execution_scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
            if (
                self.resource.nodes,
                self.resource.tasks_per_node,
                self.resource.gpus_per_node,
            ) != (1, 4, 4):
                raise RankSupervisorWorkerError(
                    "local four-GPU product scope requires one node and four ranks/GPUs"
                )
        elif self.resource.nodes < 2:
            raise RankSupervisorWorkerError(
                "multi-node rank supervisor scope requires at least two nodes"
            )
        if self.execution_scope is EvidenceScope.REAL_MULTI_NODE_PRODUCT:
            if not isinstance(self.expected_product_identity, ExpectedProductIdentity):
                raise RankSupervisorWorkerError(
                    "real product scope requires expected execution identity"
                )
            if self.expected_product_identity.repository_commit != self.candidate_commit:
                raise RankSupervisorWorkerError(
                    "expected source commit differs from the frozen candidate"
                )
        elif self.expected_product_identity is not None:
            raise RankSupervisorWorkerError(
                "only real multi-node scope may carry product execution identity"
            )


AllocationProvider = Callable[[ResourceClass, Mapping[str, str]], RankAllocation]


def load_rank_supervisor_input(path: Path) -> RankSupervisorInput:
    """Load the standard immutable input and enforce the private runtime schema."""
    manifest, input_digest = load_input_manifest(path)
    if set(manifest.payload) != {"runtime", "context"}:
        raise RankSupervisorWorkerError(
            "rank supervisor manifest requires the generic runtime/context envelope"
        )
    runtime_value = manifest.payload["runtime"]
    if not isinstance(runtime_value, dict):
        raise RankSupervisorWorkerError("rank supervisor runtime must be an object")
    runtime = cast(dict[str, object], runtime_value)
    _exact_keys(runtime, _RUNTIME_KEYS, "rank supervisor runtime")
    if runtime["schema_version"] != RANK_SUPERVISOR_SCHEMA_VERSION:
        raise RankSupervisorWorkerError("unsupported rank supervisor schema version")
    if runtime["kind"] != "rank_supervisor":
        raise RankSupervisorWorkerError("worker input is not a rank supervisor contract")
    candidate = _object(runtime["candidate"], "candidate")
    _exact_keys(
        candidate,
        frozenset(
            {
                "attempt_id",
                "digest",
                "commit",
                "receipt_sha256",
            }
        ),
        "candidate",
    )
    execution = _object(runtime["execution"], "execution")
    _exact_keys(
        execution,
        frozenset(
            {
                "scope",
                "expected_world_size",
                "expected_rank",
                "expected_local_rank",
                "product_rank_body",
            }
        ),
        "execution",
    )
    resource = _parse_resource(_object(runtime["resource"], "resource"))
    rank_input_path = Path(_string(runtime, "rank_input_path"))
    rank_report_directory = Path(_string(runtime, "rank_report_directory"))
    output_dir = path.parent / OUTPUT_DIRECTORY
    if rank_input_path != output_dir / "rank-input.json":
        raise RankSupervisorWorkerError("rank input path differs from the fixed mailbox path")
    if rank_report_directory != output_dir / "rank-reports":
        raise RankSupervisorWorkerError("rank report path differs from the fixed mailbox path")
    return RankSupervisorInput(
        manifest=manifest,
        input_digest=input_digest,
        candidate_attempt_id=_string(candidate, "attempt_id"),
        candidate_digest=_string(candidate, "digest"),
        candidate_commit=_string(candidate, "commit"),
        candidate_receipt_sha256=_string(candidate, "receipt_sha256"),
        gate=_parse_gate(_object(runtime["gate"], "gate")),
        certification_mode=_parse_certification_mode(runtime),
        execution_scope=EvidenceScope(_string(execution, "scope")),
        expected_world_size=_integer(execution, "expected_world_size"),
        expected_rank=_optional_integer(execution, "expected_rank"),
        expected_local_rank=_optional_integer(execution, "expected_local_rank"),
        product_rank_body=_boolean(execution, "product_rank_body"),
        mapping=_parse_mapping(_object(runtime["mapping"], "mapping")),
        resource=resource,
        rank_input_path=rank_input_path,
        rank_report_directory=rank_report_directory,
        expected_product_identity=_parse_expected_identity(runtime["expected_product_identity"]),
    )


def execute_rank_supervisor(
    input_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    allocation_provider: AllocationProvider | None = None,
    supervisor: Callable[..., RankSupervisionResult] = supervise_rank_launch,
) -> None:
    """Execute one exact collective and atomically publish its ordinary result."""
    rank_input = load_rank_supervisor_input(input_path)
    runtime_environment = dict(os.environ if environment is None else environment)
    attempt_dir = input_path.parent / OUTPUT_DIRECTORY
    try:
        allocation = (allocation_provider or _slurm_allocation)(
            rank_input.resource, runtime_environment
        )
        plan = _build_plan(rank_input, allocation)
        published_input, _rank_input_digest = publish_rank_input(
            plan,
            cwd=Path(rank_input.manifest.worktree),
            report_directory=rank_input.rank_report_directory,
        )
        if published_input.expected_product_identity != rank_input.expected_product_identity:
            raise RankSupervisorWorkerError("published rank identity differs from supervisor input")
        supervision = supervisor(plan, cwd=Path(rank_input.manifest.worktree))
        receipt = _successful_receipt(rank_input, plan, supervision)
        status = WorkerResultStatus.SUCCEEDED
        summary = f"collective gate {rank_input.gate.gate_id!r} passed"
        diagnostic = _diagnostic_payload(rank_input, plan, supervision, None)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        receipt = GateReceipt(
            gate_id=rank_input.gate.gate_id,
            purpose=rank_input.gate.purpose,
            scope=EvidenceScope.CPU_STATIC,
            placements=(),
            product_rank_body=False,
            passed=False,
            accuracy=rank_input.gate.accuracy,
        )
        status = WorkerResultStatus.REJECTED
        summary = f"collective gate {rank_input.gate.gate_id!r} rejected: {error}"
        diagnostic = _diagnostic_payload(rank_input, None, None, error)
    evidence_path = attempt_dir / "rank-supervision.json"
    _write_evidence_once(evidence_path, diagnostic)
    evidence = (describe_evidence(attempt_dir, evidence_path.name),)
    result = WorkerResultManifest(
        run_id=rank_input.manifest.run_id,
        item_id=rank_input.manifest.item_id,
        attempt_id=rank_input.manifest.attempt_id,
        task_digest=rank_input.manifest.task_digest,
        generation=rank_input.manifest.generation,
        input_digest=rank_input.input_digest,
        status=status,
        summary=summary[:_MAX_DIAGNOSTIC_CHARS],
        evidence=evidence,
        reviewed_candidate_digest=rank_input.candidate_digest,
        payload=_gate_payload(rank_input, receipt, evidence),
    )
    publish_result(attempt_dir, result, input_path=input_path)


def _build_plan(
    supervisor_input: RankSupervisorInput, allocation: RankAllocation
) -> RankLaunchPlan:
    arguments = {
        "mapping": supervisor_input.mapping,
        "world_size": supervisor_input.resource.total_tasks,
        "allocation": allocation,
        "command": supervisor_input.gate.command,
        "rank_input_path": supervisor_input.rank_input_path,
        "cpus_per_rank": supervisor_input.resource.cpus_per_task,
    }
    if supervisor_input.execution_scope is EvidenceScope.REAL_MULTI_NODE_PRODUCT:
        identity = supervisor_input.expected_product_identity
        if identity is None:
            raise RankSupervisorWorkerError("real product plan lacks expected identity")
        return build_slurm_multi_node_product_plan(**arguments, expected_product_identity=identity)
    if supervisor_input.execution_scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
        return build_single_node_multi_rank_plan(**arguments)
    return build_synthetic_multi_node_plan(**arguments)


def _slurm_allocation(resource: ResourceClass, environment: Mapping[str, str]) -> RankAllocation:
    job_id = environment.get("SLURM_JOB_ID")
    node_list = environment.get("SLURM_JOB_NODELIST")
    if job_id is None or _JOB_ID.fullmatch(job_id) is None or not node_list:
        raise RankSupervisorWorkerError(
            "rank supervisor must run inside one exact Slurm allocation"
        )
    try:
        scontrol = resolve_trusted_scheduler_client("scontrol", environment)
    except RankSupervisorError as error:
        raise RankSupervisorWorkerError(str(error)) from error
    try:
        completed = subprocess.run(
            (str(scontrol), "show", "hostnames", node_list),
            env={},
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RankSupervisorWorkerError(f"could not inspect Slurm allocation: {error}") from error
    if completed.returncode != 0:
        raise RankSupervisorWorkerError("scontrol could not resolve the allocated node list")
    hostnames = tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())
    if len(hostnames) != resource.nodes or len(set(hostnames)) != resource.nodes:
        raise RankSupervisorWorkerError(
            "observed Slurm nodes differ from the frozen resource request"
        )
    gpu_ids = tuple(range(resource.gpus_per_node))
    return RankAllocation(
        nodes=tuple(AllocationNode(hostname, gpu_ids) for hostname in hostnames),
        slurm_job_id=job_id,
    )


def _successful_receipt(
    supervisor_input: RankSupervisorInput,
    plan: RankLaunchPlan,
    supervision: RankSupervisionResult,
) -> GateReceipt:
    placement = supervision.receipt
    expected_kind = {
        EvidenceScope.LOCAL_FOUR_GPU_PRODUCT: LaunchKind.SINGLE_NODE_MULTI_RANK,
        EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER: LaunchKind.SYNTHETIC_MULTI_NODE_RUNNER,
        EvidenceScope.REAL_MULTI_NODE_PRODUCT: LaunchKind.SLURM_MULTI_NODE_PRODUCT,
    }[supervisor_input.execution_scope]
    if (
        placement.plan_digest != plan.plan_digest
        or placement.kind is not expected_kind
        or placement.world_size != plan.world_size
        or placement.placements != plan.placements
        or placement.evidence_scope is not supervisor_input.execution_scope
        or placement.product_rank_body is not supervisor_input.product_rank_body
    ):
        raise RankSupervisorWorkerError("supervisor receipt differs from requested certification")
    placements = tuple(
        RankPlacement(binding.rank, binding.hostname, binding.local_rank)
        for binding in placement.placements
    )
    return GateReceipt(
        gate_id=supervisor_input.gate.gate_id,
        purpose=supervisor_input.gate.purpose,
        scope=supervisor_input.execution_scope,
        placements=placements,
        product_rank_body=(
            supervisor_input.execution_scope
            in {
                EvidenceScope.LOCAL_FOUR_GPU_PRODUCT,
                EvidenceScope.REAL_MULTI_NODE_PRODUCT,
            }
        ),
        passed=True,
        accuracy=supervisor_input.gate.accuracy,
    )


def _diagnostic_payload(
    supervisor_input: RankSupervisorInput,
    plan: RankLaunchPlan | None,
    supervision: RankSupervisionResult | None,
    error: Exception | None,
) -> dict[str, JsonValue]:
    identities: list[JsonValue] = []
    if supervision is not None:
        identities = [identity.to_dict() for identity in supervision.receipt.product_identities]
    return {
        "schema_version": 1,
        "gate_id": supervisor_input.gate.gate_id,
        "candidate_commit": supervisor_input.candidate_commit,
        "candidate_digest": supervisor_input.candidate_digest,
        "certification_mode": supervisor_input.certification_mode.value,
        "requested_scope": supervisor_input.execution_scope.value,
        "plan_digest": plan.plan_digest if plan is not None else None,
        "identity_provenance": {
            "controller_attested": [
                "repository_commit",
                "native_build_identity",
                "image_identity",
            ],
            "trusted_rank_observed": [
                "python_tensorrt_llm_path",
                "compute_capability",
                "cuda_device_uuid",
                "cuda_device_model",
            ],
            "product_reported": ["collective_backend", "transport"],
        },
        "product_identities": identities,
        "error": str(error)[:_MAX_DIAGNOSTIC_CHARS] if error is not None else None,
    }


def _gate_payload(
    supervisor_input: RankSupervisorInput,
    receipt: GateReceipt,
    evidence: tuple[EvidenceFile, ...],
) -> dict[str, JsonValue]:
    accuracy: JsonValue = None
    if receipt.accuracy is not None:
        accuracy = {
            "selector": receipt.accuracy.selector,
            "reference": receipt.accuracy.reference,
            "protocol": receipt.accuracy.protocol,
            "tolerance": receipt.accuracy.tolerance,
        }
    return {
        "schema_version": 1,
        "result_kind": "deterministic_gate",
        "certification_mode": supervisor_input.certification_mode.value,
        "candidate_attempt_id": supervisor_input.candidate_attempt_id,
        "candidate_digest": supervisor_input.candidate_digest,
        "receipt": {
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
        },
        "evidence": [
            {
                "path": item.path,
                "sha256": item.sha256,
                "size_bytes": item.size_bytes,
            }
            for item in evidence
        ],
    }


def _write_evidence_once(path: Path, payload: Mapping[str, JsonValue]) -> None:
    serialized = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as output:
        output.write(serialized)
        output.flush()
        os.fsync(output.fileno())
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _parse_gate(value: Mapping[str, object]) -> GateSpec:
    _exact_keys(
        value,
        frozenset({"gate_id", "phase", "purpose", "command", "hard_gate", "accuracy"}),
        "gate",
    )
    command = _object(value["command"], "gate command")
    _exact_keys(command, frozenset({"argv", "environment"}), "gate command")
    accuracy_value = value["accuracy"]
    accuracy = None
    if accuracy_value is not None:
        accuracy_object = _object(accuracy_value, "accuracy")
        _exact_keys(
            accuracy_object,
            frozenset({"selector", "reference", "protocol", "tolerance"}),
            "accuracy",
        )
        tolerance = accuracy_object["tolerance"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise RankSupervisorWorkerError("accuracy tolerance must be numeric")
        accuracy = AccuracyCriteria(
            selector=_string(accuracy_object, "selector"),
            reference=_string(accuracy_object, "reference"),
            protocol=_string(accuracy_object, "protocol"),
            tolerance=float(tolerance),
        )
    argv = command["argv"]
    environment = command["environment"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise RankSupervisorWorkerError("gate command argv must be a string list")
    if not isinstance(environment, dict) or not all(
        isinstance(name, str) and isinstance(item, str) for name, item in environment.items()
    ):
        raise RankSupervisorWorkerError("gate command environment must be a string object")
    hard_gate = value["hard_gate"]
    if not isinstance(hard_gate, bool):
        raise RankSupervisorWorkerError("gate hard_gate must be a boolean")
    return GateSpec(
        gate_id=_string(value, "gate_id"),
        phase=GatePhase(_integer(value, "phase")),
        purpose=GatePurpose(_string(value, "purpose")),
        command=GateCommand(tuple(argv), tuple(sorted(environment.items()))),
        hard_gate=hard_gate,
        accuracy=accuracy,
    )


def _parse_mapping(value: Mapping[str, object]) -> ParallelMapping:
    keys = frozenset(
        {
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "moe_expert_parallel_size",
            "moe_tensor_parallel_size",
            "attention_data_parallel_size",
        }
    )
    _exact_keys(value, keys, "mapping")
    return ParallelMapping(**{name: _integer(value, name) for name in keys})


def _parse_resource(value: Mapping[str, object]) -> ResourceClass:
    keys = frozenset(
        {
            "name",
            "nodes",
            "tasks_per_node",
            "gpus_per_node",
            "cpus_per_task",
            "memory_mib",
            "time_limit_seconds",
        }
    )
    _exact_keys(value, keys, "resource")
    return ResourceClass(
        name=_string(value, "name"),
        nodes=_integer(value, "nodes"),
        tasks_per_node=_integer(value, "tasks_per_node"),
        gpus_per_node=_integer(value, "gpus_per_node"),
        cpus_per_task=_integer(value, "cpus_per_task"),
        memory_mib=_integer(value, "memory_mib"),
        time_limit_seconds=_integer(value, "time_limit_seconds"),
    )


def _parse_expected_identity(value: object) -> ExpectedProductIdentity | None:
    if value is None:
        return None
    identity = _object(value, "expected product identity")
    keys = frozenset(
        {
            "repository_commit",
            "python_tensorrt_llm_path",
            "native_build_identity",
            "image_identity",
            "compute_capability",
            "collective_backend",
            "transport",
        }
    )
    _exact_keys(identity, keys, "expected product identity")
    return ExpectedProductIdentity(**{name: _string(identity, name) for name in keys})


def _parse_certification_mode(value: Mapping[str, object]) -> CertificationMode:
    try:
        return CertificationMode(_string(value, "certification_mode"))
    except ValueError as error:
        raise RankSupervisorWorkerError("certification_mode is invalid") from error


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RankSupervisorWorkerError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _exact_keys(value: Mapping[str, object], keys: frozenset[str], label: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        raise RankSupervisorWorkerError(
            f"{label} keys differ from schema: missing={missing!r}, unknown={unknown!r}"
        )


def _string(value: Mapping[str, object], name: str) -> str:
    item = value[name]
    if not isinstance(item, str) or not item or "\x00" in item or "\n" in item:
        raise RankSupervisorWorkerError(f"{name} must be a non-empty single-line string")
    return item


def _integer(value: Mapping[str, object], name: str) -> int:
    item = value[name]
    if isinstance(item, bool) or not isinstance(item, int):
        raise RankSupervisorWorkerError(f"{name} must be an integer")
    return item


def _optional_integer(value: Mapping[str, object], name: str) -> int | None:
    item = value[name]
    if item is None:
        return None
    return _integer(value, name)


def _boolean(value: Mapping[str, object], name: str) -> bool:
    item = value[name]
    if not isinstance(item, bool):
        raise RankSupervisorWorkerError(f"{name} must be a boolean")
    return item


__all__ = [
    "AllocationProvider",
    "RANK_SUPERVISOR_SCHEMA_VERSION",
    "RankSupervisorInput",
    "RankSupervisorWorkerError",
    "execute_rank_supervisor",
    "load_rank_supervisor_input",
]
