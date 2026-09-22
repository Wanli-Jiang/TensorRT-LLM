# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shell-free rank launch plans for ModelingV2 gate execution.

This module describes execution; it never starts a process.  The trusted
worker supervisor is responsible for executing the returned argv without a
shell and for collecting one :class:`ObservedRank` from every rank.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePath
from typing import Sequence

from ..task_schema import ParallelMapping
from .gates import (
    EvidenceScope,
    GateCommand,
    GatePolicyError,
    GatePurpose,
    GateReceipt,
    RankPlacement,
)

_HOSTNAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?\Z")
_SLURM_JOB_ID = re.compile(r"[1-9][0-9]*\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SENSITIVE_ENVIRONMENT = re.compile(
    r"(?:AUTH|COOKIE|CREDENTIAL|KEY|PASS|SECRET|TOKEN)", re.IGNORECASE
)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_COMPUTE_CAPABILITY = re.compile(r"[1-9][0-9]*\.[0-9]+\Z")
_IDENTITY_VALUE = re.compile(r"[^\x00\r\n]{1,512}\Z")
_RANK_WORKER_MODULE = "agent_flow.workflows.staircase.internal"
_PYTHON_EXECUTABLE = str(Path(sys.executable).absolute())
_PROHIBITED_PROGRAMS = frozenset(
    {
        "bash",
        "dash",
        "mpiexec",
        "mpirun",
        "sacct",
        "sbatch",
        "scancel",
        "sh",
        "squeue",
        "ssh",
        "srun",
        "zsh",
    }
)
_RESERVED_RANK_ENVIRONMENT = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "RANK",
        "SLURM_LOCALID",
        "SLURM_NODEID",
        "SLURM_PROCID",
        "WORLD_SIZE",
    }
)


class LaunchPolicyError(ValueError):
    """Raised when a rank launch would be ambiguous or overstate evidence."""


class LaunchKind(str, Enum):
    """Supported execution shapes with distinct certification boundaries."""

    SINGLE_PROCESS = "single_process"
    SINGLE_NODE_MULTI_RANK = "single_node_multi_rank"
    SYNTHETIC_MULTI_NODE_RUNNER = "synthetic_multi_node_runner"
    SLURM_MULTI_NODE_PRODUCT = "slurm_multi_node_product"


@dataclass(frozen=True, slots=True)
class ExpectedProductIdentity:
    """Controller-audited execution identity required from every product rank."""

    repository_commit: str
    python_tensorrt_llm_path: str
    native_build_identity: str
    image_identity: str
    compute_capability: str
    collective_backend: str
    transport: str

    def __post_init__(self) -> None:
        if _COMMIT.fullmatch(self.repository_commit) is None:
            raise LaunchPolicyError("repository_commit must be a lowercase Git SHA-1")
        python_path = Path(self.python_tensorrt_llm_path)
        if not python_path.is_absolute() or python_path != python_path.resolve(strict=False):
            raise LaunchPolicyError("python_tensorrt_llm_path must be canonical and absolute")
        for name in (
            "native_build_identity",
            "image_identity",
            "collective_backend",
            "transport",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _IDENTITY_VALUE.fullmatch(value) is None:
                raise LaunchPolicyError(f"{name} must be a bounded single-line identity")
        if _COMPUTE_CAPABILITY.fullmatch(self.compute_capability) is None:
            raise LaunchPolicyError("compute_capability must use '<major>.<minor>'")

    def to_dict(self) -> dict[str, str]:
        """Return the strict JSON representation used by rank inputs."""
        return {
            "repository_commit": self.repository_commit,
            "python_tensorrt_llm_path": self.python_tensorrt_llm_path,
            "native_build_identity": self.native_build_identity,
            "image_identity": self.image_identity,
            "compute_capability": self.compute_capability,
            "collective_backend": self.collective_backend,
            "transport": self.transport,
        }


@dataclass(frozen=True, slots=True)
class ObservedProductIdentity:
    """Independently observed product identity emitted by one completed rank."""

    repository_commit: str
    python_tensorrt_llm_path: str
    native_build_identity: str
    image_identity: str
    compute_capability: str
    collective_backend: str
    transport: str
    cuda_device_uuid: str
    cuda_device_model: str
    body_evidence_digest: str

    def __post_init__(self) -> None:
        _ = self.expected
        for name in ("cuda_device_uuid", "cuda_device_model"):
            value = getattr(self, name)
            if not isinstance(value, str) or _IDENTITY_VALUE.fullmatch(value) is None:
                raise LaunchPolicyError(f"{name} must be a bounded single-line identity")
        if _DIGEST.fullmatch(self.body_evidence_digest) is None:
            raise LaunchPolicyError("body_evidence_digest must be a lowercase SHA-256")

    @property
    def expected(self) -> ExpectedProductIdentity:
        """Project independently observed fields onto the controller expectation."""
        return ExpectedProductIdentity(
            repository_commit=self.repository_commit,
            python_tensorrt_llm_path=self.python_tensorrt_llm_path,
            native_build_identity=self.native_build_identity,
            image_identity=self.image_identity,
            compute_capability=self.compute_capability,
            collective_backend=self.collective_backend,
            transport=self.transport,
        )

    def to_dict(self) -> dict[str, str]:
        """Return the strict JSON representation used by rank reports."""
        return {
            **self.expected.to_dict(),
            "cuda_device_uuid": self.cuda_device_uuid,
            "cuda_device_model": self.cuda_device_model,
            "body_evidence_digest": self.body_evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class AllocationNode:
    """Exact GPUs made available on one named allocation node."""

    hostname: str
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hostname, str) or not _HOSTNAME.fullmatch(self.hostname):
            raise LaunchPolicyError(f"invalid exact node hostname {self.hostname!r}")
        if not isinstance(self.gpu_ids, tuple) or not self.gpu_ids:
            raise LaunchPolicyError("an allocation node must contain at least one GPU ID")
        for gpu_id in self.gpu_ids:
            if isinstance(gpu_id, bool) or not isinstance(gpu_id, int) or gpu_id < 0:
                raise LaunchPolicyError("GPU IDs must be non-negative integers")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise LaunchPolicyError("GPU IDs must be unique within an allocation node")


@dataclass(frozen=True, slots=True)
class RankAllocation:
    """Exact allocation from which rank placement is derived in block order."""

    nodes: tuple[AllocationNode, ...]
    slurm_job_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or not self.nodes:
            raise LaunchPolicyError("rank allocation must contain at least one node")
        if not all(isinstance(node, AllocationNode) for node in self.nodes):
            raise LaunchPolicyError("rank allocation nodes must use AllocationNode")
        hostnames = [node.hostname for node in self.nodes]
        if len(set(hostnames)) != len(hostnames):
            raise LaunchPolicyError("rank allocation contains duplicate node hostnames")
        if self.slurm_job_id is not None and (
            not isinstance(self.slurm_job_id, str) or not _SLURM_JOB_ID.fullmatch(self.slurm_job_id)
        ):
            raise LaunchPolicyError("slurm_job_id must be one exact positive numeric job ID")

    @property
    def world_size(self) -> int:
        """Return the exact number of allocated rank GPUs."""
        return sum(len(node.gpu_ids) for node in self.nodes)


@dataclass(frozen=True, slots=True)
class RankBinding:
    """Expected global rank, node-local rank and physical GPU placement."""

    rank: int
    hostname: str
    local_rank: int
    gpu_id: int

    def __post_init__(self) -> None:
        for name in ("rank", "local_rank", "gpu_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise LaunchPolicyError(f"{name} must be a non-negative integer")
        if not isinstance(self.hostname, str) or not _HOSTNAME.fullmatch(self.hostname):
            raise LaunchPolicyError(f"invalid exact node hostname {self.hostname!r}")


@dataclass(frozen=True, slots=True)
class ObservedRank:
    """Rank placement reported by a completed rank process."""

    rank: int
    hostname: str
    local_rank: int
    gpu_id: int
    product_identity: ObservedProductIdentity | None = None

    def __post_init__(self) -> None:
        RankBinding(self.rank, self.hostname, self.local_rank, self.gpu_id)
        if self.product_identity is not None and not isinstance(
            self.product_identity, ObservedProductIdentity
        ):
            raise LaunchPolicyError(
                "rank product_identity must use ObservedProductIdentity when present"
            )


@dataclass(frozen=True, slots=True)
class RankLaunchPlan:
    """Validated argv and expected placements for one collective invocation."""

    kind: LaunchKind
    mapping: ParallelMapping
    world_size: int
    allocation: RankAllocation
    rank_command: GateCommand
    rank_input_path: Path
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    placements: tuple[RankBinding, ...]
    product_rank_body: bool
    evidence_scope: EvidenceScope | None
    expected_product_identity: ExpectedProductIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LaunchKind):
            raise LaunchPolicyError("launch kind must use LaunchKind")
        _validate_mapping(self.mapping, self.world_size)
        if self.allocation.world_size != self.world_size:
            raise LaunchPolicyError("allocation GPU count must equal Mapping world size")
        if not isinstance(self.rank_command, GateCommand):
            raise LaunchPolicyError("rank command must use GateCommand")
        _validate_rank_input_path(self.rank_input_path)
        if not self.argv or not all(
            isinstance(argument, str) and argument for argument in self.argv
        ):
            raise LaunchPolicyError("launcher argv must contain non-empty strings")
        if any("\x00" in argument or "\n" in argument for argument in self.argv):
            raise LaunchPolicyError("launcher argv cannot contain NUL or newline characters")
        _validate_environment(self.environment)
        if self.placements != _derive_placements(self.allocation):
            raise LaunchPolicyError("rank placements must exactly match allocation block order")
        if not isinstance(self.product_rank_body, bool):
            raise LaunchPolicyError("product_rank_body must be a boolean")
        if self.evidence_scope is not None and not isinstance(self.evidence_scope, EvidenceScope):
            raise LaunchPolicyError("evidence_scope must use EvidenceScope when present")
        if self.kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT:
            if not isinstance(self.expected_product_identity, ExpectedProductIdentity):
                raise LaunchPolicyError(
                    "real multi-node product launch requires an expected product identity"
                )
        elif self.expected_product_identity is not None:
            raise LaunchPolicyError(
                "only a real multi-node product launch may carry expected product identity"
            )
        _validate_kind_boundary(
            kind=self.kind,
            world_size=self.world_size,
            placements=self.placements,
            product_rank_body=self.product_rank_body,
            evidence_scope=self.evidence_scope,
        )
        if self.kind is not LaunchKind.SINGLE_PROCESS:
            if self.allocation.slurm_job_id is None or self.argv[0] != "srun":
                raise LaunchPolicyError("every multi-rank plan must use its exact Slurm allocation")
        if self.argv[-len(_rank_worker_argv(self.rank_input_path)) :] != _rank_worker_argv(
            self.rank_input_path
        ):
            raise LaunchPolicyError("launcher must invoke only the fixed private rank worker")

    @property
    def plan_digest(self) -> str:
        """Return a stable digest covering command, mapping and placement."""
        payload = {
            "allocation": {
                "slurm_job_id": self.allocation.slurm_job_id,
                "nodes": [
                    {"hostname": node.hostname, "gpu_ids": list(node.gpu_ids)}
                    for node in self.allocation.nodes
                ],
            },
            "argv": list(self.argv),
            "environment": [list(pair) for pair in self.environment],
            "kind": self.kind.value,
            "mapping": {
                "attention_data_parallel_size": self.mapping.attention_data_parallel_size,
                "moe_expert_parallel_size": self.mapping.moe_expert_parallel_size,
                "moe_tensor_parallel_size": self.mapping.moe_tensor_parallel_size,
                "pipeline_parallel_size": self.mapping.pipeline_parallel_size,
                "tensor_parallel_size": self.mapping.tensor_parallel_size,
            },
            "product_rank_body": self.product_rank_body,
            "expected_product_identity": (
                self.expected_product_identity.to_dict()
                if self.expected_product_identity is not None
                else None
            ),
            "rank_command": {
                "argv": list(self.rank_command.argv),
                "environment": [list(pair) for pair in self.rank_command.environment],
            },
            "rank_input_path": str(self.rank_input_path),
            "world_size": self.world_size,
        }
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @property
    def rank_command_digest(self) -> str:
        """Return a stable digest for the exact body argv and environment."""
        payload = {
            "argv": list(self.rank_command.argv),
            "environment": [list(pair) for pair in self.rank_command.environment],
        }
        serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RankPlacementReceipt:
    """Auditable proof that every reported rank matched its launch plan."""

    schema_version: int
    plan_digest: str
    kind: LaunchKind
    world_size: int
    placements: tuple[RankBinding, ...]
    product_rank_body: bool
    evidence_scope: EvidenceScope | None
    product_identities: tuple[ObservedProductIdentity, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise LaunchPolicyError("unsupported rank placement receipt version")
        if not _DIGEST.fullmatch(self.plan_digest):
            raise LaunchPolicyError("plan_digest must be a lowercase SHA-256 digest")
        if not isinstance(self.kind, LaunchKind):
            raise LaunchPolicyError("receipt kind must use LaunchKind")
        if isinstance(self.world_size, bool) or not isinstance(self.world_size, int):
            raise LaunchPolicyError("receipt world_size must be an integer")
        if not isinstance(self.placements, tuple) or not all(
            isinstance(placement, RankBinding) for placement in self.placements
        ):
            raise LaunchPolicyError("receipt placements must use RankBinding")
        if len(self.placements) != self.world_size:
            raise LaunchPolicyError("receipt must contain exactly one placement per rank")
        if tuple(placement.rank for placement in self.placements) != tuple(range(self.world_size)):
            raise LaunchPolicyError("receipt placements must be in complete rank order")
        if not isinstance(self.product_rank_body, bool):
            raise LaunchPolicyError("receipt product_rank_body must be a boolean")
        if self.evidence_scope is not None and not isinstance(self.evidence_scope, EvidenceScope):
            raise LaunchPolicyError("receipt evidence_scope must use EvidenceScope when present")
        if self.kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT:
            if len(self.product_identities) != self.world_size:
                raise LaunchPolicyError(
                    "real multi-node receipt requires one product identity per rank"
                )
            if not all(
                isinstance(identity, ObservedProductIdentity)
                for identity in self.product_identities
            ):
                raise LaunchPolicyError(
                    "receipt product identities must use ObservedProductIdentity"
                )
        elif self.product_identities:
            raise LaunchPolicyError(
                "non-product-multi-node receipt cannot carry product identities"
            )
        _validate_kind_boundary(
            kind=self.kind,
            world_size=self.world_size,
            placements=self.placements,
            product_rank_body=self.product_rank_body,
            evidence_scope=self.evidence_scope,
        )

    def to_gate_receipt(self, *, gate_id: str, purpose: GatePurpose, passed: bool) -> GateReceipt:
        """Convert a placement proof into the narrower gate evidence schema.

        General one-node collectives have no current gate claim scope.  Their
        placement receipt remains useful, but cannot silently become local
        four-GPU evidence.
        """
        if self.evidence_scope is None:
            raise GatePolicyError(
                "this launch shape has no gate evidence scope; an explicit product scope is required"
            )
        return GateReceipt(
            gate_id=gate_id,
            purpose=purpose,
            scope=self.evidence_scope,
            placements=tuple(
                RankPlacement(
                    rank=placement.rank,
                    node=placement.hostname,
                    local_rank=placement.local_rank,
                )
                for placement in self.placements
            ),
            product_rank_body=self.product_rank_body,
            passed=passed,
        )


def build_single_process_plan(
    *,
    mapping: ParallelMapping,
    world_size: int,
    allocation: RankAllocation,
    command: GateCommand,
    rank_input_path: Path,
) -> RankLaunchPlan:
    """Build one directly executable product-rank plan with an empty inherited environment."""
    _validate_shape(mapping, world_size, allocation, expected_nodes=1, multi_rank=False)
    environment = _command_environment(command)
    gpu_id = allocation.nodes[0].gpu_ids[0]
    process_environment = tuple(
        sorted(
            {
                **dict(environment),
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "LOCAL_RANK": "0",
                "RANK": "0",
                "WORLD_SIZE": "1",
            }.items()
        )
    )
    argv = (
        "/usr/bin/env",
        "-i",
        *_environment_assignments(process_environment),
        *_rank_worker_argv(rank_input_path),
    )
    return RankLaunchPlan(
        kind=LaunchKind.SINGLE_PROCESS,
        mapping=mapping,
        world_size=world_size,
        allocation=allocation,
        rank_command=command,
        rank_input_path=rank_input_path,
        argv=argv,
        environment=process_environment,
        placements=_derive_placements(allocation),
        product_rank_body=True,
        evidence_scope=EvidenceScope.SINGLE_GPU_PRODUCT,
    )


def build_single_node_multi_rank_plan(
    *,
    mapping: ParallelMapping,
    world_size: int,
    allocation: RankAllocation,
    command: GateCommand,
    rank_input_path: Path,
    cpus_per_rank: int = 1,
) -> RankLaunchPlan:
    """Build an exact Slurm step for a single-node product collective."""
    _validate_shape(mapping, world_size, allocation, expected_nodes=1, multi_rank=True)
    evidence_scope = EvidenceScope.LOCAL_FOUR_GPU_PRODUCT if world_size == 4 else None
    return _build_slurm_plan(
        kind=LaunchKind.SINGLE_NODE_MULTI_RANK,
        mapping=mapping,
        world_size=world_size,
        allocation=allocation,
        command=command,
        rank_input_path=rank_input_path,
        cpus_per_rank=cpus_per_rank,
        product_rank_body=True,
        evidence_scope=evidence_scope,
    )


def build_synthetic_multi_node_plan(
    *,
    mapping: ParallelMapping,
    world_size: int,
    allocation: RankAllocation,
    command: GateCommand,
    rank_input_path: Path,
    cpus_per_rank: int = 1,
) -> RankLaunchPlan:
    """Build a multi-node runner canary that cannot certify product behavior."""
    _validate_shape(mapping, world_size, allocation, minimum_nodes=2, multi_rank=True)
    return _build_slurm_plan(
        kind=LaunchKind.SYNTHETIC_MULTI_NODE_RUNNER,
        mapping=mapping,
        world_size=world_size,
        allocation=allocation,
        command=command,
        rank_input_path=rank_input_path,
        cpus_per_rank=cpus_per_rank,
        product_rank_body=False,
        evidence_scope=EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
    )


def build_slurm_multi_node_product_plan(
    *,
    mapping: ParallelMapping,
    world_size: int,
    allocation: RankAllocation,
    command: GateCommand,
    rank_input_path: Path,
    expected_product_identity: ExpectedProductIdentity,
    cpus_per_rank: int = 1,
) -> RankLaunchPlan:
    """Build an explicit Slurm-aware multi-node ModelingV2 product collective."""
    _validate_shape(mapping, world_size, allocation, minimum_nodes=2, multi_rank=True)
    return _build_slurm_plan(
        kind=LaunchKind.SLURM_MULTI_NODE_PRODUCT,
        mapping=mapping,
        world_size=world_size,
        allocation=allocation,
        command=command,
        rank_input_path=rank_input_path,
        cpus_per_rank=cpus_per_rank,
        product_rank_body=True,
        evidence_scope=EvidenceScope.REAL_MULTI_NODE_PRODUCT,
        expected_product_identity=expected_product_identity,
    )


def make_rank_placement_receipt(
    plan: RankLaunchPlan, observations: Sequence[ObservedRank]
) -> RankPlacementReceipt:
    """Verify runtime placement reports and bind them to the immutable plan digest."""
    observed = tuple(observations)
    if not all(isinstance(observation, ObservedRank) for observation in observed):
        raise LaunchPolicyError("rank observations must use ObservedRank")
    ordered = tuple(sorted(observed, key=lambda observation: observation.rank))
    ranks = tuple(observation.rank for observation in ordered)
    if len(set(ranks)) != len(ranks):
        raise LaunchPolicyError("rank observations contain a duplicate rank")
    observed_bindings = tuple(
        RankBinding(
            rank=observation.rank,
            hostname=observation.hostname,
            local_rank=observation.local_rank,
            gpu_id=observation.gpu_id,
        )
        for observation in ordered
    )
    if observed_bindings != plan.placements:
        raise LaunchPolicyError("observed rank/node/local-rank/GPU placement differs from plan")
    product_identities = tuple(
        observation.product_identity
        for observation in ordered
        if observation.product_identity is not None
    )
    if plan.kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT:
        expected = plan.expected_product_identity
        if expected is None or len(product_identities) != plan.world_size:
            raise LaunchPolicyError(
                "real multi-node product evidence requires identity proof from every rank"
            )
        if any(identity.expected != expected for identity in product_identities):
            raise LaunchPolicyError(
                "observed product identity differs from the controller expectation"
            )
        if len({identity.cuda_device_uuid for identity in product_identities}) != plan.world_size:
            raise LaunchPolicyError(
                "real multi-node product evidence requires a distinct CUDA UUID per rank"
            )
        if len({identity.cuda_device_model for identity in product_identities}) != 1:
            raise LaunchPolicyError(
                "real multi-node product evidence requires one CUDA model across ranks"
            )
    elif product_identities:
        raise LaunchPolicyError(
            "product identity proof is accepted only for real multi-node product launches"
        )
    return RankPlacementReceipt(
        schema_version=1,
        plan_digest=plan.plan_digest,
        kind=plan.kind,
        world_size=plan.world_size,
        placements=observed_bindings,
        product_rank_body=plan.product_rank_body,
        evidence_scope=plan.evidence_scope,
        product_identities=product_identities,
    )


def _build_slurm_plan(
    *,
    kind: LaunchKind,
    mapping: ParallelMapping,
    world_size: int,
    allocation: RankAllocation,
    command: GateCommand,
    rank_input_path: Path,
    cpus_per_rank: int,
    product_rank_body: bool,
    evidence_scope: EvidenceScope | None,
    expected_product_identity: ExpectedProductIdentity | None = None,
) -> RankLaunchPlan:
    if allocation.slurm_job_id is None:
        raise LaunchPolicyError("a multi-rank launch requires an exact Slurm allocation job ID")
    if isinstance(cpus_per_rank, bool) or not isinstance(cpus_per_rank, int) or cpus_per_rank < 1:
        raise LaunchPolicyError("cpus_per_rank must be a positive integer")
    ranks_per_node = len(allocation.nodes[0].gpu_ids)
    if any(len(node.gpu_ids) != ranks_per_node for node in allocation.nodes):
        raise LaunchPolicyError("Slurm rank launch requires equal ranks per node")
    gpu_map = allocation.nodes[0].gpu_ids
    if any(node.gpu_ids != gpu_map for node in allocation.nodes[1:]):
        raise LaunchPolicyError(
            "exact Slurm GPU binding requires the same GPU ID map on every node"
        )
    environment = _command_environment(command)
    node_list = ",".join(node.hostname for node in allocation.nodes)
    argv = (
        "srun",
        f"--jobid={allocation.slurm_job_id}",
        "--exact",
        f"--nodes={len(allocation.nodes)}",
        f"--ntasks={world_size}",
        f"--ntasks-per-node={ranks_per_node}",
        "--gpus-per-task=1",
        f"--cpus-per-task={cpus_per_rank}",
        f"--nodelist={node_list}",
        "--distribution=block:block",
        f"--gpu-bind=map_gpu:{','.join(str(gpu_id) for gpu_id in gpu_map)}",
        "--kill-on-bad-exit=1",
        f"--export={','.join(_environment_assignments(environment))}",
        *_rank_worker_argv(rank_input_path),
    )
    return RankLaunchPlan(
        kind=kind,
        mapping=mapping,
        world_size=world_size,
        allocation=allocation,
        rank_command=command,
        rank_input_path=rank_input_path,
        argv=argv,
        environment=environment,
        placements=_derive_placements(allocation),
        product_rank_body=product_rank_body,
        evidence_scope=evidence_scope,
        expected_product_identity=expected_product_identity,
    )


def _validate_shape(
    mapping: ParallelMapping,
    world_size: int,
    allocation: RankAllocation,
    *,
    expected_nodes: int | None = None,
    minimum_nodes: int | None = None,
    multi_rank: bool,
) -> None:
    _validate_mapping(mapping, world_size)
    if allocation.world_size != world_size:
        raise LaunchPolicyError("allocation GPU count must equal Mapping world size")
    if expected_nodes is not None and len(allocation.nodes) != expected_nodes:
        raise LaunchPolicyError(f"launch requires exactly {expected_nodes} allocation node(s)")
    if minimum_nodes is not None and len(allocation.nodes) < minimum_nodes:
        raise LaunchPolicyError(f"launch requires at least {minimum_nodes} allocation nodes")
    if multi_rank and world_size < 2:
        raise LaunchPolicyError("multi-rank launch requires world_size greater than one")
    if not multi_rank and world_size != 1:
        raise LaunchPolicyError("single-process launch requires world_size equal to one")


def _validate_mapping(mapping: ParallelMapping, world_size: int) -> None:
    if not isinstance(mapping, ParallelMapping):
        raise LaunchPolicyError("mapping must use ParallelMapping")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise LaunchPolicyError("world_size must be a positive integer")
    values = (
        mapping.tensor_parallel_size,
        mapping.pipeline_parallel_size,
        mapping.moe_expert_parallel_size,
        mapping.moe_tensor_parallel_size,
        mapping.attention_data_parallel_size,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
        raise LaunchPolicyError("every Mapping axis must be a positive integer")
    expected_world_size = mapping.tensor_parallel_size * mapping.pipeline_parallel_size
    if world_size != expected_world_size:
        raise LaunchPolicyError(
            "world_size must equal Mapping tensor_parallel_size * pipeline_parallel_size"
        )
    moe_width = mapping.moe_expert_parallel_size * mapping.moe_tensor_parallel_size
    if moe_width not in {1, mapping.tensor_parallel_size}:
        raise LaunchPolicyError("Mapping MoE width must be one or tensor_parallel_size")
    if mapping.tensor_parallel_size % mapping.attention_data_parallel_size != 0:
        raise LaunchPolicyError(
            "Mapping attention_data_parallel_size must divide tensor_parallel_size"
        )


def _derive_placements(allocation: RankAllocation) -> tuple[RankBinding, ...]:
    placements: list[RankBinding] = []
    for node in allocation.nodes:
        for local_rank, gpu_id in enumerate(node.gpu_ids):
            placements.append(
                RankBinding(
                    rank=len(placements),
                    hostname=node.hostname,
                    local_rank=local_rank,
                    gpu_id=gpu_id,
                )
            )
    return tuple(placements)


def _validate_kind_boundary(
    *,
    kind: LaunchKind,
    world_size: int,
    placements: tuple[RankBinding, ...],
    product_rank_body: bool,
    evidence_scope: EvidenceScope | None,
) -> None:
    node_count = len({placement.hostname for placement in placements})
    expected: tuple[int | None, bool, EvidenceScope | None]
    if kind is LaunchKind.SINGLE_PROCESS:
        expected = (1, True, EvidenceScope.SINGLE_GPU_PRODUCT)
    elif kind is LaunchKind.SINGLE_NODE_MULTI_RANK:
        expected = (
            1,
            True,
            EvidenceScope.LOCAL_FOUR_GPU_PRODUCT if world_size == 4 else None,
        )
    elif kind is LaunchKind.SYNTHETIC_MULTI_NODE_RUNNER:
        expected = (None, False, EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER)
    elif kind is LaunchKind.SLURM_MULTI_NODE_PRODUCT:
        expected = (None, True, EvidenceScope.REAL_MULTI_NODE_PRODUCT)
    else:
        raise LaunchPolicyError(f"unsupported launch kind {kind!r}")
    expected_node_count, expected_product_body, expected_scope = expected
    if expected_node_count is not None and node_count != expected_node_count:
        raise LaunchPolicyError(f"{kind.value} has an incompatible node topology")
    if expected_node_count is None and node_count < 2:
        raise LaunchPolicyError(f"{kind.value} requires at least two nodes")
    if product_rank_body is not expected_product_body or evidence_scope is not expected_scope:
        raise LaunchPolicyError(
            f"{kind.value} has an incompatible product-body or evidence-scope boundary"
        )


def _command_environment(command: GateCommand) -> tuple[tuple[str, str], ...]:
    _validate_rank_command(command)
    environment = tuple(sorted(command.environment))
    _validate_environment(environment)
    reserved = sorted(set(dict(environment)) & _RESERVED_RANK_ENVIRONMENT)
    if reserved:
        raise LaunchPolicyError(f"rank environment is controller-owned: {reserved!r}")
    return environment


def _validate_rank_command(command: GateCommand) -> None:
    if not isinstance(command, GateCommand):
        raise LaunchPolicyError("rank command must use GateCommand")
    program = PurePath(command.argv[0]).name.lower()
    if program in _PROHIBITED_PROGRAMS:
        raise LaunchPolicyError(
            "rank command must be a program body, not a shell, SSH, or nested rank launcher"
        )


def _validate_environment(environment: tuple[tuple[str, str], ...]) -> None:
    if not isinstance(environment, tuple):
        raise LaunchPolicyError("launch environment must be an immutable tuple")
    names: set[str] = set()
    for pair in environment:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise LaunchPolicyError("launch environment entries must be name/value pairs")
        name, value = pair
        if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
            raise LaunchPolicyError(f"unsafe environment name {name!r}")
        if not isinstance(value, str) or any(character in value for character in "\x00\n\r,="):
            raise LaunchPolicyError(f"environment value for {name!r} cannot be exported safely")
        if _SENSITIVE_ENVIRONMENT.search(name):
            raise LaunchPolicyError(f"credential-like environment variable is prohibited: {name!r}")
        if name in names:
            raise LaunchPolicyError(f"duplicate environment variable {name!r}")
        names.add(name)
    if dict(environment).get("TRTLLM_MODELING_V2") != "require":
        raise LaunchPolicyError("every rank must set TRTLLM_MODELING_V2=require")


def _environment_assignments(environment: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    return tuple(f"{name}={value}" for name, value in environment)


def _rank_worker_argv(rank_input_path: Path) -> tuple[str, ...]:
    _validate_rank_input_path(rank_input_path)
    return (
        _PYTHON_EXECUTABLE,
        "-m",
        _RANK_WORKER_MODULE,
        "rank-worker",
        "--input",
        str(rank_input_path),
    )


def _validate_rank_input_path(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute():
        raise LaunchPolicyError("rank input path must be an absolute Path")
    if any(character in str(path) for character in "\x00\n"):
        raise LaunchPolicyError("rank input path must be a single-line path")
    if path != path.resolve(strict=False):
        raise LaunchPolicyError("rank input path must be canonical")


__all__ = [
    "AllocationNode",
    "ExpectedProductIdentity",
    "LaunchKind",
    "LaunchPolicyError",
    "ObservedProductIdentity",
    "ObservedRank",
    "RankAllocation",
    "RankBinding",
    "RankLaunchPlan",
    "RankPlacementReceipt",
    "build_single_node_multi_rank_plan",
    "build_single_process_plan",
    "build_slurm_multi_node_product_plan",
    "build_synthetic_multi_node_plan",
    "make_rank_placement_receipt",
]
