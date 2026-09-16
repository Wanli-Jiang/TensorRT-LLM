# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Controller adapter for trusted in-allocation collective gate execution."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

from ..common.artifacts import (
    INPUT_FILENAME,
    JsonValue,
    ResultExpectation,
    WorkerInputManifest,
    load_input_manifest,
    worker_output_directory,
    write_input_manifest,
)
from ..common.gates import EvidenceScope, GatePhase, GateSpec
from ..common.gitops import ControllerGitOps
from ..common.isolation import (
    IsolationPolicyError,
    WorkerIsolation,
    build_gate_worker_environment,
    build_worker_isolation,
)
from ..common.launchers import ExpectedProductIdentity
from ..common.policy import WorkItemProposal
from ..common.slurm import InternalCommand, InternalEntrypoint, Mount, ResourceRequest
from ..state import AttemptKind, AttemptRecord, Role, RunState, WorkItemStatus
from ..task_schema import CertificationMode, NormalizedTask, ParallelMapping, ResourceClass
from .attempts import AttemptExecution
from .domain import (
    DomainActionBlock,
    DomainActionKind,
    DomainActionPlan,
    DomainBlockCode,
    FrozenCandidate,
    GateExecutionContract,
    _action,
    _Blocked,
    _new_attempt,
    _validate_candidate,
    _validate_item_identity,
    _validate_non_smith_profile,
    _validate_resource_name,
)
from .execution import ExecutionAdapterError, MaterializableAction, MaterializedWorker
from .rank_gates import CollectiveCertification
from .results import CandidateReceipt


class CollectiveAdapterError(ExecutionAdapterError):
    """Raised when a persisted collective action cannot be reconstructed."""


def build_local_product_certifications(
    *,
    gates: Sequence[GateSpec],
    executions: Mapping[str, GateExecutionContract],
    resources: Mapping[str, ResourceClass],
    mapping: ParallelMapping,
) -> dict[str, CollectiveCertification]:
    """Derive only the safe default one-node/four-GPU certification mapping.

    Real multi-node product and synthetic-runner certification are never
    inferred here.  They require an explicit caller-owned mode (and, for real
    product evidence, an :class:`ExpectedProductIdentity`).
    """
    world_size = mapping.tensor_parallel_size * mapping.pipeline_parallel_size
    result: dict[str, CollectiveCertification] = {}
    for gate in gates:
        execution = executions.get(gate.gate_id)
        resource = resources.get(gate.gate_id)
        if execution is None or resource is None:
            raise CollectiveAdapterError(
                f"gate {gate.gate_id!r} lacks frozen execution/resource input"
            )
        if execution.scope in {
            EvidenceScope.REAL_MULTI_NODE_PRODUCT,
            EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
        }:
            raise CollectiveAdapterError(
                f"gate {gate.gate_id!r} requires explicit collective certification"
            )
        if execution.scope is not EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
            continue
        if (
            world_size != 4
            or execution.expected_world_size != 4
            or not execution.product_rank_body
            or (
                resource.nodes,
                resource.tasks_per_node,
                resource.gpus_per_node,
            )
            != (1, 4, 4)
        ):
            raise CollectiveAdapterError(
                f"gate {gate.gate_id!r} has a noncanonical local four-GPU topology"
            )
        result[gate.gate_id] = CollectiveCertification.LOCAL_PRODUCT
    return result


def build_task_collective_adapter(
    *, workspace: Path, task: NormalizedTask, git: ControllerGitOps
) -> TrustedCollectiveGateExecutionAdapter | None:
    """Construct the explicit SYNTHETIC/REAL collective adapter from a task.

    LOCAL execution retains the runtime's canonical four-GPU derivation. It
    never silently promotes into a Slurm synthetic or real-product claim.
    """
    mode = task.certification.mode
    if mode is CertificationMode.LOCAL:
        return None
    certification = (
        CollectiveCertification.REAL_PRODUCT
        if mode is CertificationMode.REAL
        else CollectiveCertification.SYNTHETIC_RUNNER
    )
    gate_ids = tuple(gate_class.gate_id for gate_class in task.execution.slurm.gate_classes)
    if not gate_ids:
        raise CollectiveAdapterError(
            f"{mode.value} certification requires explicit trusted gate classes"
        )
    certifications = {gate_id: certification for gate_id in gate_ids}
    identities: dict[str, ExpectedProductIdentity] = {}
    if mode is CertificationMode.REAL:
        configured = task.certification.expected_product_identity
        if configured is None:
            raise CollectiveAdapterError("REAL certification lacks expected product identity")
        identity = ExpectedProductIdentity(
            repository_commit=configured.repository_commit,
            python_tensorrt_llm_path=configured.python_tensorrt_llm_path,
            native_build_identity=configured.native_build_identity,
            image_identity=configured.image_identity,
            compute_capability=configured.compute_capability,
            collective_backend=configured.collective_backend,
            transport=configured.transport,
        )
        identities = {gate_id: identity for gate_id in gate_ids}
    return TrustedCollectiveGateExecutionAdapter(
        workspace=workspace,
        task=task,
        git=git,
        certifications=certifications,
        expected_product_identities=identities,
    )


class TrustedCollectiveGateExecutionAdapter:
    """Plan and execute collective gates through the fixed rank supervisor."""

    def __init__(
        self,
        *,
        workspace: Path,
        task: NormalizedTask,
        git: ControllerGitOps,
        certifications: Mapping[str, CollectiveCertification],
        expected_product_identities: Mapping[str, ExpectedProductIdentity] | None = None,
    ) -> None:
        self._workspace = workspace.expanduser().resolve(strict=True)
        self._task = task
        self._git = git
        self._certifications = dict(certifications)
        self._expected_product_identities = dict(expected_product_identities or {})
        for gate_id, certification in self._certifications.items():
            if not gate_id or not isinstance(certification, CollectiveCertification):
                raise CollectiveAdapterError("collective certifications must be typed by gate ID")
            identity = self._expected_product_identities.get(gate_id)
            if certification is CollectiveCertification.REAL_PRODUCT and identity is None:
                raise CollectiveAdapterError(
                    f"real collective gate {gate_id!r} lacks expected product identity"
                )
            if certification is not CollectiveCertification.REAL_PRODUCT and identity is not None:
                raise CollectiveAdapterError(
                    f"non-multi-node gate {gate_id!r} cannot carry product identity"
                )
        unknown = set(self._expected_product_identities) - set(self._certifications)
        if unknown:
            raise CollectiveAdapterError(
                f"product identities name unconfigured collective gates: {sorted(unknown)!r}"
            )

    def plan(
        self,
        state: RunState,
        proposal: WorkItemProposal,
        candidate: FrozenCandidate,
        candidate_receipt: CandidateReceipt,
        gate: GateSpec,
        execution: GateExecutionContract,
        *,
        resource: ResourceClass,
        mapping: ParallelMapping,
        workspace: Path,
    ) -> DomainActionPlan:
        """Reserve one deterministic gate action without filesystem mutation."""
        try:
            if workspace.expanduser().resolve() != self._workspace:
                raise _Blocked(
                    DomainBlockCode.GATE_CONTRACT,
                    "collective adapter workspace differs from runtime workspace",
                )
            item = _validate_item_identity(state, proposal)
            _validate_non_smith_profile(item, proposal)
            if item.status is not WorkItemStatus.CODING:
                raise _Blocked(
                    DomainBlockCode.INVALID_STATE,
                    "collective gate requires a WorkItem in CODING",
                )
            _validate_resource_name(resource, "deterministic_gate")
            _validate_candidate(
                item,
                proposal,
                candidate,
                require_frozen=item.candidate_attempt_id is not None,
            )
            self._validate_candidate_receipt(state, candidate, candidate_receipt)
            expected_identity = self._validate_collective_contract(
                gate, execution, resource, mapping, candidate
            )
            attempt = _new_attempt(
                state,
                item,
                Role.GATE,
                AttemptKind.DETERMINISTIC_GATE,
                resource_class=resource.name,
            )
            desired_item = item
            if item.candidate_attempt_id is None:
                desired_item = replace(
                    item,
                    candidate_attempt_id=candidate.coder_attempt_id,
                    candidate_digest=candidate.digest,
                )
            desired = state.replace_item(desired_item.add_attempt(attempt))
            runtime = self._runtime_payload(
                candidate,
                candidate_receipt,
                gate,
                execution,
                resource,
                mapping,
                attempt.sequence,
                expected_identity,
            )
            action = _action(
                desired,
                proposal,
                attempt,
                resource,
                workspace=self._workspace,
                base_commit=candidate.commit,
                payload={
                    "runtime": runtime,
                    "context": {
                        "action": DomainActionKind.DETERMINISTIC_GATE.value,
                        "gate_id": gate.gate_id,
                        "candidate_digest": candidate.digest,
                        "collective_scope": execution.scope.value,
                    },
                },
                allowed_paths=(),
                kind=DomainActionKind.DETERMINISTIC_GATE,
            )
            return DomainActionPlan(desired, action=action)
        except _Blocked as error:
            return DomainActionPlan(
                state,
                blocked=DomainActionBlock(error.code, proposal.item_id, str(error)),
            )
        except (KeyError, TypeError, ValueError) as error:
            return DomainActionPlan(
                state,
                blocked=DomainActionBlock(
                    DomainBlockCode.GATE_CONTRACT,
                    proposal.item_id,
                    str(error) or type(error).__name__,
                ),
            )

    def materialize(self, action: MaterializableAction) -> MaterializedWorker:
        """Create/adopt a metadata-free candidate snapshot and immutable input."""
        self._validate_action_paths(action)
        try:
            self._materialize_candidate_snapshot(action)
            action.attempt_dir.mkdir(parents=True, exist_ok=True)
            worker_output_directory(action.attempt_dir).mkdir(parents=True, exist_ok=True)
            manifest, isolation = self._prepared_manifest(action)
            input_path = action.attempt_dir / INPUT_FILENAME
            if input_path.exists():
                persisted, input_digest = load_input_manifest(input_path)
                if persisted != manifest:
                    raise CollectiveAdapterError(
                        "immutable rank-supervisor input differs from reconstructed action"
                    )
            else:
                input_digest = write_input_manifest(input_path, manifest)
            self._ensure_runtime_directories(action)
            return MaterializedWorker(input_digest, isolation)
        except (
            IsolationPolicyError,
            OSError,
            subprocess.SubprocessError,
            tarfile.TarError,
            ValueError,
        ) as error:
            if isinstance(error, CollectiveAdapterError):
                raise
            raise CollectiveAdapterError(
                f"collective materialization failed closed: {error}"
            ) from error

    def _materialize_candidate_snapshot(self, action: MaterializableAction) -> None:
        archive = self._candidate_archive(action.base_commit)
        archive_sha256 = hashlib.sha256(archive).hexdigest()
        marker = action.worktree.with_name(f"{action.worktree.name}.snapshot.json")
        expected_marker = {
            "schema_version": 1,
            "candidate_commit": action.base_commit,
            "archive_sha256": archive_sha256,
        }
        if action.worktree.exists() or marker.exists():
            if not action.worktree.is_dir() or action.worktree.is_symlink():
                raise CollectiveAdapterError("candidate snapshot is not a regular directory")
            try:
                persisted = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CollectiveAdapterError(
                    f"candidate snapshot marker is unreadable: {error}"
                ) from error
            if persisted != expected_marker or (action.worktree / ".git").exists():
                raise CollectiveAdapterError(
                    "existing candidate snapshot differs from persisted action"
                )
            self._validate_candidate_snapshot(action.worktree)
            return

        action.worktree.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{action.worktree.name}.", dir=action.worktree.parent)
        )
        try:
            self._extract_candidate_archive(archive, temporary)
            temporary.rename(action.worktree)
            serialized = json.dumps(expected_marker, sort_keys=True, separators=(",", ":"))
            marker.write_text(f"{serialized}\n", encoding="utf-8")
            marker.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            self._validate_candidate_snapshot(action.worktree)
        except (CollectiveAdapterError, OSError, tarfile.TarError, ValueError):
            if temporary.exists():
                shutil.rmtree(temporary)
            elif action.worktree.exists() and not marker.exists():
                shutil.rmtree(action.worktree)
            raise

    @staticmethod
    def _validate_candidate_snapshot(snapshot: Path) -> None:
        for path in (snapshot, *snapshot.rglob("*")):
            relative = path.relative_to(snapshot) if path != snapshot else Path()
            if ".git" in relative.parts:
                raise CollectiveAdapterError("candidate snapshot exposes Git metadata")
            if path.is_symlink():
                if not path.resolve(strict=False).is_relative_to(snapshot):
                    raise CollectiveAdapterError("candidate snapshot symlink escapes its root")
                continue
            if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                raise CollectiveAdapterError("candidate snapshot contains a writable entry")

    def _candidate_archive(self, commit: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise CollectiveAdapterError("candidate snapshot requires a full Git commit")
        completed = subprocess.run(
            (
                "git",
                "-C",
                str(self._git.repository),
                "archive",
                "--format=tar",
                commit,
            ),
            env={},
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise CollectiveAdapterError(f"could not archive candidate commit: {detail}")
        return completed.stdout

    @staticmethod
    def _extract_candidate_archive(archive: bytes, destination: Path) -> None:
        members: list[tuple[tarfile.TarInfo, Path]] = []
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            for member in source.getmembers():
                pure_path = PurePosixPath(member.name)
                if (
                    pure_path.is_absolute()
                    or not pure_path.parts
                    or any(part in {"", ".", ".."} for part in pure_path.parts)
                    or ".git" in pure_path.parts
                ):
                    raise CollectiveAdapterError("candidate archive contains an unsafe path")
                target = destination.joinpath(*pure_path.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    body = source.extractfile(member)
                    if body is None:
                        raise CollectiveAdapterError("candidate archive file has no body")
                    with target.open("xb") as output:
                        shutil.copyfileobj(body, output)
                elif member.issym():
                    link = PurePosixPath(member.linkname)
                    if link.is_absolute() or any(part == ".." for part in link.parts):
                        raise CollectiveAdapterError(
                            "candidate archive symlink escapes its snapshot"
                        )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(member.linkname)
                else:
                    raise CollectiveAdapterError(
                        "candidate archive contains an unsupported special entry"
                    )
                members.append((member, target))
        for member, target in reversed(members):
            if member.isdir():
                target.chmod(
                    stat.S_IRUSR
                    | stat.S_IXUSR
                    | stat.S_IRGRP
                    | stat.S_IXGRP
                    | stat.S_IROTH
                    | stat.S_IXOTH
                )
            elif member.isfile():
                mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
                if member.mode & 0o111:
                    mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
                target.chmod(mode)
        destination.chmod(
            stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
        )

    def execution(self, state: RunState, action: MaterializableAction) -> AttemptExecution:
        """Reconstruct the scheduler contract and fixed supervisor command."""
        self._validate_action_paths(action)
        input_path = action.attempt_dir / INPUT_FILENAME
        try:
            manifest, input_digest = load_input_manifest(input_path)
            expected, isolation = self._prepared_manifest(action)
            if manifest != expected:
                raise CollectiveAdapterError(
                    "persisted rank-supervisor input differs from reconstructed action"
                )
            if (
                manifest.run_id != state.run_id
                or manifest.task_digest != state.task_digest
                or manifest.generation != action.attempt.generation
            ):
                raise CollectiveAdapterError(
                    "rank-supervisor input differs from authoritative run identity"
                )
            resources = self._resources(action, isolation)
            command = InternalCommand(
                InternalEntrypoint.RANK_SUPERVISOR,
                input_bundle=isolation.container_path(input_path),
            )
        except (IsolationPolicyError, OSError, ValueError) as error:
            if isinstance(error, CollectiveAdapterError):
                raise
            raise CollectiveAdapterError(f"collective execution failed closed: {error}") from error
        token_digest = hashlib.sha256(
            (
                f"{state.run_id}\0{action.attempt.attempt_id}\0{state.task_digest}\0"
                f"{action.attempt.generation}\0{input_digest}"
            ).encode()
        ).hexdigest()[:40]
        return AttemptExecution(
            resources=resources,
            command=command,
            submission_token=f"worker-{token_digest}",
            attempt_dir=worker_output_directory(action.attempt_dir),
            expectation=ResultExpectation(
                state.run_id,
                action.attempt.item_id,
                action.attempt.attempt_id,
                state.task_digest,
                action.attempt.generation,
                input_digest,
            ),
            receipt_root=self._workspace / "receipts",
            quarantine_root=self._workspace / "quarantine",
            environment=tuple(isolation.environment.values),
        )

    def revoke(self, state: RunState, attempt: AttemptRecord) -> bool:
        """Collective deterministic gates never receive revocable agent credentials."""
        _ = state, attempt
        return False

    def _validate_candidate_receipt(
        self,
        state: RunState,
        candidate: FrozenCandidate,
        receipt: CandidateReceipt,
    ) -> None:
        if (
            receipt.run_id != state.run_id
            or receipt.item_id != candidate.item_id
            or receipt.attempt_id != candidate.coder_attempt_id
            or receipt.candidate_commit != candidate.commit
            or receipt.candidate_digest != candidate.digest
            or receipt.changed_paths != candidate.changed_paths
        ):
            raise _Blocked(
                DomainBlockCode.CANDIDATE_MISMATCH,
                "candidate receipt differs from the frozen candidate",
            )

    def _validate_collective_contract(
        self,
        gate: GateSpec,
        execution: GateExecutionContract,
        resource: ResourceClass,
        mapping: ParallelMapping,
        candidate: FrozenCandidate,
    ) -> ExpectedProductIdentity | None:
        certification = self._certifications.get(gate.gate_id)
        if certification is None:
            raise _Blocked(
                DomainBlockCode.GATE_CONTRACT,
                "collective GateSpec has no frozen certification mode",
            )
        expected_scope = (
            EvidenceScope.REAL_MULTI_NODE_PRODUCT
            if certification is CollectiveCertification.REAL_PRODUCT
            else (
                EvidenceScope.LOCAL_FOUR_GPU_PRODUCT
                if certification is CollectiveCertification.LOCAL_PRODUCT
                else EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER
            )
        )
        if execution.scope is not expected_scope:
            raise _Blocked(
                DomainBlockCode.GATE_CONTRACT,
                "collective execution scope differs from certification mode",
            )
        if (
            certification is CollectiveCertification.SYNTHETIC_RUNNER
            and gate.phase is not GatePhase.COLLECTIVE
        ):
            raise _Blocked(
                DomainBlockCode.GATE_CONTRACT,
                "synthetic runner certification is limited to a COLLECTIVE canary",
            )
        if mapping != self._task.target.mapping:
            raise _Blocked(
                DomainBlockCode.GATE_CONTRACT,
                "collective mapping differs from the normalized task",
            )
        world_size = mapping.tensor_parallel_size * mapping.pipeline_parallel_size
        topology_matches = (
            resource.total_tasks == world_size
            and resource.tasks_per_node == resource.gpus_per_node
            and execution.expected_world_size == world_size
            and execution.expected_rank is None
            and execution.expected_local_rank is None
        )
        if certification is CollectiveCertification.LOCAL_PRODUCT:
            topology_matches = topology_matches and (
                resource.nodes,
                resource.tasks_per_node,
                resource.gpus_per_node,
            ) == (1, 4, 4)
        else:
            topology_matches = topology_matches and resource.nodes >= 2
        if not topology_matches:
            raise _Blocked(
                DomainBlockCode.RESOURCE_CONTRACT,
                "collective resource, mapping, and placement topology differ",
            )
        identity = self._expected_product_identities.get(gate.gate_id)
        if identity is None:
            return None
        controller = self._task.execution.slurm.controller
        if (
            identity.native_build_identity != controller.build_identity
            or identity.image_identity != str(controller.image)
            or identity.python_tensorrt_llm_path
            != str(self._task.repository.root / "tensorrt_llm" / "__init__.py")
        ):
            raise _Blocked(
                DomainBlockCode.GATE_CONTRACT,
                "product Python/build/image identity differs from the normalized task",
            )
        expected_compute_capability = f"{self._task.target.sm // 10}.{self._task.target.sm % 10}"
        if identity.compute_capability != expected_compute_capability:
            raise _Blocked(
                DomainBlockCode.GATE_CONTRACT,
                "product compute capability differs from the normalized target",
            )
        return replace(identity, repository_commit=candidate.commit)

    def _runtime_payload(
        self,
        candidate: FrozenCandidate,
        receipt: CandidateReceipt,
        gate: GateSpec,
        execution: GateExecutionContract,
        resource: ResourceClass,
        mapping: ParallelMapping,
        sequence: int,
        identity: ExpectedProductIdentity | None,
    ) -> dict[str, JsonValue]:
        attempt_dir = self._workspace / "items" / candidate.item_id / "attempts" / f"{sequence:04d}"
        accuracy: JsonValue = None
        if gate.accuracy is not None:
            accuracy = {
                "selector": gate.accuracy.selector,
                "reference": gate.accuracy.reference,
                "protocol": gate.accuracy.protocol,
                "tolerance": gate.accuracy.tolerance,
            }
        return {
            "schema_version": 2,
            "kind": "rank_supervisor",
            "certification_mode": self._certifications[gate.gate_id].certification_mode.value,
            "candidate": {
                "attempt_id": candidate.coder_attempt_id,
                "digest": candidate.digest,
                "commit": candidate.commit,
                "receipt_sha256": receipt.receipt_sha256,
            },
            "gate": {
                "gate_id": gate.gate_id,
                "phase": int(gate.phase),
                "purpose": gate.purpose.value,
                "command": {
                    "argv": list(gate.command.argv),
                    "environment": dict(gate.command.environment),
                },
                "hard_gate": gate.hard_gate,
                "accuracy": accuracy,
            },
            "execution": {
                "scope": execution.scope.value,
                "expected_world_size": execution.expected_world_size,
                "expected_rank": execution.expected_rank,
                "expected_local_rank": execution.expected_local_rank,
                "product_rank_body": execution.product_rank_body,
            },
            "mapping": {
                "tensor_parallel_size": mapping.tensor_parallel_size,
                "pipeline_parallel_size": mapping.pipeline_parallel_size,
                "moe_expert_parallel_size": mapping.moe_expert_parallel_size,
                "moe_tensor_parallel_size": mapping.moe_tensor_parallel_size,
                "attention_data_parallel_size": mapping.attention_data_parallel_size,
            },
            "resource": {
                "name": resource.name,
                "nodes": resource.nodes,
                "tasks_per_node": resource.tasks_per_node,
                "gpus_per_node": resource.gpus_per_node,
                "cpus_per_task": resource.cpus_per_task,
                "memory_mib": resource.memory_mib,
                "time_limit_seconds": resource.time_limit_seconds,
            },
            "rank_input_path": str(worker_output_directory(attempt_dir) / "rank-input.json"),
            "rank_report_directory": str(worker_output_directory(attempt_dir) / "rank-reports"),
            "expected_product_identity": identity.to_dict() if identity is not None else None,
        }

    def _prepared_manifest(
        self, action: MaterializableAction
    ) -> tuple[WorkerInputManifest, WorkerIsolation]:
        if action.attempt.role is not Role.GATE:
            raise CollectiveAdapterError("collective adapter requires a Gate attempt")
        environment = build_gate_worker_environment({})
        controller = self._task.execution.slurm.controller
        isolation = build_worker_isolation(
            controller_mounts=tuple(
                Mount(mount.host_path, Path(mount.container_path), mount.read_only)
                for mount in controller.mounts
            ),
            workspace=self._workspace,
            repository=self._task.repository.root,
            mailbox=worker_output_directory(action.attempt_dir),
            input_bundle=action.attempt_dir / INPUT_FILENAME,
            checkpoint=self._task.reference.checkpoint,
            reference_sources=self._task.reference.additional_sources,
            worktree=action.worktree,
            worktree_writable=False,
            environment=environment,
        )
        runtime = action.manifest.payload.get("runtime")
        if not isinstance(runtime, dict):
            raise CollectiveAdapterError("collective action lacks a runtime object")
        translated_runtime = dict(runtime)
        translated_runtime["rank_input_path"] = str(
            isolation.container_path(
                worker_output_directory(action.attempt_dir) / "rank-input.json",
                require_exists=False,
            )
        )
        translated_runtime["rank_report_directory"] = str(
            isolation.container_path(
                worker_output_directory(action.attempt_dir) / "rank-reports",
                require_exists=False,
            )
        )
        identity = translated_runtime.get("expected_product_identity")
        if identity is not None:
            if not isinstance(identity, dict):
                raise CollectiveAdapterError("expected product identity must be an object")
            translated_identity = dict(identity)
            translated_identity["python_tensorrt_llm_path"] = str(
                isolation.container_path(action.worktree / "tensorrt_llm" / "__init__.py")
            )
            translated_runtime["expected_product_identity"] = translated_identity
        manifest = replace(
            action.manifest,
            worktree=str(isolation.container_path(action.worktree)),
            payload={
                "runtime": translated_runtime,
                "context": action.manifest.payload["context"],
            },
        )
        return manifest, isolation

    def _resources(
        self, action: MaterializableAction, isolation: WorkerIsolation
    ) -> ResourceRequest:
        controller = self._task.execution.slurm.controller
        log_root = self._workspace / "slurm" / "workers" / action.attempt.attempt_id
        if not log_root.is_dir() or log_root.is_symlink():
            raise CollectiveAdapterError("collective log directory was not materialized")
        resource = action.resource
        return ResourceRequest(
            label=action.attempt.attempt_id,
            account=controller.account,
            partition=controller.partition,
            qos=controller.qos,
            reservation=controller.reservation,
            time_limit=_slurm_time(resource.time_limit_seconds),
            output_path=log_root / "%j.out",
            error_path=log_root / "%j.err",
            nodes=resource.nodes,
            tasks_per_node=resource.tasks_per_node,
            cpus_per_task=resource.cpus_per_task,
            gpus_per_node=resource.gpus_per_node,
            memory_mb=resource.memory_mib,
            requeue=False,
            container_image=str(controller.image),
            mounts=isolation.mounts,
        )

    def _ensure_runtime_directories(self, action: MaterializableAction) -> None:
        for path in (
            self._workspace / "receipts",
            self._workspace / "quarantine",
            self._workspace / "slurm" / "workers" / action.attempt.attempt_id,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _validate_action_paths(self, action: MaterializableAction) -> None:
        expected_worktree = (
            self._workspace / "candidates" / action.attempt.item_id / action.attempt.attempt_id
        )
        expected_mailbox = (
            self._workspace
            / "items"
            / action.attempt.item_id
            / "attempts"
            / f"{action.attempt.sequence:04d}"
        )
        if action.worktree != expected_worktree or action.attempt_dir != expected_mailbox:
            raise CollectiveAdapterError(
                "collective action paths differ from persisted attempt identity"
            )
        if action.manifest.attempt_id != action.attempt.attempt_id:
            raise CollectiveAdapterError("collective manifest belongs to another attempt")


def _slurm_time(seconds: int) -> str:
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


__all__ = [
    "CollectiveAdapterError",
    "TrustedCollectiveGateExecutionAdapter",
    "build_local_product_certifications",
    "build_task_collective_adapter",
]
