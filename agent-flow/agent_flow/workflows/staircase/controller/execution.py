# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restart-safe materialization and execution of generic worker attempts.

The adapter is deliberately mechanical. Domain planners authorize an action;
after that state has been persisted this module creates/adopts its metadata-free
candidate overlay and immutable mailbox. Later calls reconstruct the Slurm
request from that exact mailbox rather than from ambient process state.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Protocol

from ..common.artifacts import (
    INPUT_FILENAME,
    ResultExpectation,
    WorkerInputManifest,
    load_input_manifest,
    worker_output_directory,
    write_input_manifest,
)
from ..common.credentials import (
    ALL_BACKEND_CREDENTIAL_NAMES,
    CREDENTIAL_REVOCATION_ROOT_NAME,
    CREDENTIAL_ROOT_NAME,
    CredentialBinding,
    CredentialBroker,
    CredentialDescriptor,
    CredentialError,
    CredentialRevocationCause,
    CredentialState,
    descriptor_from_public_dict,
    locate_credential_provision,
)
from ..common.gitops import ControllerGitOps, GitOpsError
from ..common.isolation import (
    IsolationPolicyError,
    WorkerIsolation,
    attach_agent_credential_provision,
    build_gate_worker_environment,
    build_worker_isolation,
    digest_metadata_free_tree,
)
from ..common.placement import HorizontalPlacementContract
from ..common.slurm import Mount, ResourceRequest
from ..state import AttemptRecord, AttemptStatus, DomainProfile, Role, RunState
from ..task_schema import NormalizedTask, ResourceClass
from .attempts import AttemptExecution
from .role_policy import ResolvedRolePolicy, RolePolicyError, resolve_role_policy


class ExecutionAdapterError(RuntimeError):
    """Raised when a persisted worker action cannot be reproduced exactly."""


class MaterializableAction(Protocol):
    """Structural action contract shared by Smith and domain planners."""

    attempt: AttemptRecord
    resource: ResourceClass
    worktree: Path
    branch: str
    base_commit: str
    attempt_dir: Path
    manifest: WorkerInputManifest


@dataclass(frozen=True, slots=True)
class MaterializedWorker:
    """Immutable worker input plus the exact isolated launch boundary."""

    input_digest: str
    isolation: WorkerIsolation


class GenericWorkerExecutionAdapter:
    """Materialize and reconstruct one generic worker Slurm contract."""

    def __init__(
        self,
        *,
        workspace: Path,
        task: NormalizedTask,
        git: ControllerGitOps,
        credential_broker: CredentialBroker | None = None,
        backend_environment: Mapping[str, str] | None = None,
    ) -> None:
        self._workspace = workspace.expanduser().resolve(strict=True)
        self._task = task
        self._git = git
        if credential_broker is None:
            raise ExecutionAdapterError(
                "worker execution requires an explicit CredentialBroker; pass "
                "NoCredentialBroker only for task policies with no allowed credential names"
            )
        self._credential_broker = credential_broker
        if backend_environment is not None:
            leaked_names = sorted(
                ALL_BACKEND_CREDENTIAL_NAMES.intersection(backend_environment.keys())
            )
            if leaked_names:
                raise ExecutionAdapterError(
                    "worker adapter cannot receive backend credential values; configure a "
                    f"trusted CredentialBroker handle instead: {leaked_names!r}"
                )
        self._backend_environment = {} if backend_environment is None else backend_environment

    def materialize(self, action: MaterializableAction) -> MaterializedWorker:
        """Create or adopt one exact metadata-free overlay and publish input once."""
        self._validate_action_paths(action)
        if action.attempt.status is not AttemptStatus.PREPARED:
            raise ExecutionAdapterError("only a PREPARED worker action may be materialized")
        try:
            policy = self._role_policy(action)
            with self._git.transaction() as transaction:
                if action.worktree.exists():
                    if not action.worktree.is_dir() or action.worktree.is_symlink():
                        raise ExecutionAdapterError("existing candidate overlay is invalid")
                else:
                    transaction.create_candidate_overlay(
                        action.worktree,
                        base_commit=action.base_commit,
                    )
                action.attempt_dir.mkdir(parents=True, exist_ok=True)
                worker_output_directory(action.attempt_dir).mkdir(parents=True, exist_ok=True)
                input_path = action.attempt_dir / INPUT_FILENAME
                if input_path.exists():
                    persisted, input_digest = load_input_manifest(input_path)
                    descriptor = self._persisted_credential_descriptor(action, persisted)
                    worker_manifest, isolation = self._prepared_manifest(
                        action,
                        policy=policy,
                        descriptor=descriptor,
                    )
                    if persisted != worker_manifest:
                        raise ExecutionAdapterError(
                            "immutable worker input differs from reconstructed action"
                        )
                else:
                    worker_manifest, _prepublication = self._prepared_manifest(
                        action,
                        policy=policy,
                        input_bundle_prepublication=True,
                    )
                    input_digest = write_input_manifest(input_path, worker_manifest)
                    descriptor = self._persisted_credential_descriptor(action, worker_manifest)
                    expected, isolation = self._prepared_manifest(
                        action,
                        policy=policy,
                        descriptor=descriptor,
                    )
                    if expected != worker_manifest:
                        raise ExecutionAdapterError(
                            "published worker input differs from strict reconstruction"
                        )
                self._ensure_runtime_directories(action)
                return MaterializedWorker(input_digest, isolation)
        except (CredentialError, GitOpsError, IsolationPolicyError, OSError, ValueError) as error:
            if isinstance(error, ExecutionAdapterError):
                raise
            raise ExecutionAdapterError(f"worker materialization failed closed: {error}") from error

    def execution(
        self,
        state: RunState,
        action: MaterializableAction,
    ) -> AttemptExecution:
        """Reconstruct the exact execution from persisted identity and input bytes."""
        self._validate_action_paths(action)
        self._validate_authoritative_attempt(state, action.attempt)
        input_path = action.attempt_dir / INPUT_FILENAME
        try:
            policy = self._role_policy(action)
            manifest, input_digest = load_input_manifest(input_path)
            descriptor = self._persisted_credential_descriptor(action, manifest)
            expected, isolation = self._prepared_manifest(
                action,
                policy=policy,
                descriptor=descriptor,
                require_live=action.attempt.status is AttemptStatus.PREPARED,
            )
            if manifest != expected:
                raise ExecutionAdapterError(
                    "persisted worker input differs from the reconstructed domain action"
                )
            if (
                manifest.run_id != state.run_id
                or manifest.task_digest != state.task_digest
                or manifest.generation != action.attempt.generation
            ):
                raise ExecutionAdapterError("worker input differs from authoritative run identity")
            if descriptor is not None:
                provision = (
                    self._credential_broker.inject_after_admission(
                        workspace=self._workspace,
                        descriptor=descriptor,
                    )
                    if action.attempt.status is AttemptStatus.PREPARED
                    else locate_credential_provision(
                        workspace=self._workspace,
                        descriptor=descriptor,
                    )
                )
                if provision.descriptor != descriptor:
                    raise ExecutionAdapterError(
                        "credential broker changed the admitted public handle"
                    )
                isolation = attach_agent_credential_provision(
                    replace(isolation, credential_descriptor=None),
                    provision,
                )
            resources = self._resources(action, isolation, policy)
            command = isolation.worker_command(input_path)
        except (CredentialError, IsolationPolicyError, OSError, ValueError) as error:
            if isinstance(error, ExecutionAdapterError):
                raise
            raise ExecutionAdapterError(f"worker execution failed closed: {error}") from error

        token_digest = hashlib.sha256(
            (
                f"{state.run_id}\0{action.attempt.attempt_id}\0{state.task_digest}\0"
                f"{action.attempt.generation}\0{input_digest}"
            ).encode("utf-8")
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
            credential_expires_at_epoch_seconds=(
                None
                if descriptor is None or descriptor.handle is None
                else descriptor.handle.expires_at_epoch_seconds
            ),
        )

    def revoke(self, state: RunState, attempt: AttemptRecord) -> bool:
        """Revoke one terminal agent credential bundle exactly and idempotently."""
        if not attempt.terminal:
            raise ExecutionAdapterError("credentials may be revoked only for a terminal attempt")
        self._validate_authoritative_attempt(state, attempt)
        attempt_dir = (
            self._workspace / "items" / attempt.item_id / "attempts" / f"{attempt.sequence:04d}"
        )
        try:
            input_path = attempt_dir / INPUT_FILENAME
            if not input_path.exists():
                return False
            manifest, _input_digest = load_input_manifest(input_path)
            runtime = manifest.payload.get("runtime")
            if not isinstance(runtime, dict):
                raise ExecutionAdapterError("worker input lacks a runtime object")
            if runtime.get("kind") != "agent":
                return False
            descriptor = descriptor_from_public_dict(runtime.get("credential_descriptor"))
            expected = self._credential_binding(action_manifest=manifest, attempt=attempt)
            if expected.run_id != state.run_id or expected.task_digest != state.task_digest:
                raise ExecutionAdapterError(
                    "credential binding differs from authoritative run identity"
                )
            if descriptor.binding != expected:
                raise ExecutionAdapterError(
                    "persisted credential descriptor differs from the exact terminal attempt"
                )
            self._validate_credential_descriptor(
                descriptor,
                expected,
                self._role_policy_for_attempt(attempt),
                require_live=False,
            )
            cause = self._credential_revocation_cause(attempt)
            receipt_path = (
                self._workspace
                / CREDENTIAL_ROOT_NAME
                / CREDENTIAL_REVOCATION_ROOT_NAME
                / expected.run_id
                / expected.item_id
                / f"{expected.attempt_id}.json"
            )
            existed = receipt_path.is_file() and not receipt_path.is_symlink()
            receipt = self._credential_broker.revoke(
                workspace=self._workspace,
                expected_binding=expected,
                descriptor=descriptor,
                cause=cause,
            )
            if (
                receipt.descriptor != descriptor
                or receipt.cause is not cause
                or receipt.receipt_path != receipt_path
            ):
                raise ExecutionAdapterError("credential broker returned a mismatched receipt")
            return not existed
        except (CredentialError, OSError, TypeError, ValueError) as error:
            if isinstance(error, ExecutionAdapterError):
                raise
            raise ExecutionAdapterError(f"credential revocation failed closed: {error}") from error

    def _prepared_manifest(
        self,
        action: MaterializableAction,
        *,
        policy: ResolvedRolePolicy | None,
        descriptor: CredentialDescriptor | None = None,
        require_live: bool = True,
        input_bundle_prepublication: bool = False,
    ) -> tuple[WorkerInputManifest, WorkerIsolation]:
        runtime = action.manifest.payload.get("runtime")
        if not isinstance(runtime, dict):
            raise ExecutionAdapterError("generic worker input lacks a runtime object")
        kind = runtime.get("kind")
        controller = self._task.execution.slurm.controller
        if kind == "agent":
            if policy is None:
                raise ExecutionAdapterError("agent worker lacks an exact task role policy")
            backend = runtime.get("backend_kind")
            if backend not in {"claude-code", "codex"}:
                raise ExecutionAdapterError("agent input has no supported backend identity")
            environment = policy.environment
            configured = self._task.execution.agent
            if (backend, runtime.get("model")) != (
                configured.backend_kind,
                configured.model,
            ):
                raise ExecutionAdapterError(
                    "worker backend/model differs from the frozen task agent identity"
                )
            binding = CredentialBinding(
                run_id=action.manifest.run_id,
                item_id=action.manifest.item_id,
                attempt_id=action.manifest.attempt_id,
                task_digest=action.manifest.task_digest,
                generation=action.manifest.generation,
                backend_kind=configured.backend_kind,
            )
            if descriptor is None:
                descriptor = self._credential_broker.describe(binding)
            self._validate_credential_descriptor(
                descriptor,
                binding,
                policy,
                require_live=require_live,
            )
            agent_runtime = dict(runtime)
            agent_runtime["credential_descriptor"] = descriptor.to_public_dict()
            agent_runtime["launch_policy"] = policy.launch_policy.to_public_dict()
            is_horizontal_smith_role = (
                action.attempt.role in {Role.CODER, Role.REVIEWER}
                and action.attempt.profile is DomainProfile.SMITH
            )
            smith = self._task.execution.slurm.smith
            agent_runtime["placement_contract"] = HorizontalPlacementContract(
                distinct_nodes_required=(
                    smith.distinct_nodes_required if is_horizontal_smith_role else False
                ),
                exclusive=(smith.exclusive if is_horizontal_smith_role else False),
            ).to_public_dict()
            if action.attempt.role in {Role.REVIEWER, Role.QA}:
                agent_runtime["candidate_overlay_digest"] = digest_metadata_free_tree(
                    action.worktree
                )
            else:
                agent_runtime["candidate_overlay_digest"] = None
            manifest = replace(
                action.manifest,
                payload={
                    "runtime": agent_runtime,
                    "context": action.manifest.payload["context"],
                },
            )
        elif kind == "gate":
            if policy is not None:
                raise ExecutionAdapterError("deterministic gate cannot carry an agent role policy")
            environment = build_gate_worker_environment({})
            manifest = action.manifest
        else:
            raise ExecutionAdapterError(f"unsupported generic worker runtime kind {kind!r}")
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
            worktree_writable=action.attempt.role is Role.CODER,
            environment=environment,
            input_bundle_prepublication=input_bundle_prepublication,
        )
        if kind == "agent":
            isolation = replace(isolation, credential_descriptor=descriptor)
        return (
            replace(
                manifest,
                worktree=str(isolation.container_path(action.worktree)),
            ),
            isolation,
        )

    def _persisted_credential_descriptor(
        self,
        action: MaterializableAction,
        manifest: WorkerInputManifest,
    ) -> CredentialDescriptor | None:
        """Recover the immutable public handle without asking the broker to mint one."""
        runtime = manifest.payload.get("runtime")
        if not isinstance(runtime, dict):
            raise ExecutionAdapterError("worker input lacks a runtime object")
        if runtime.get("kind") != "agent":
            return None
        descriptor = descriptor_from_public_dict(runtime.get("credential_descriptor"))
        if descriptor.binding != self._credential_binding(
            action_manifest=action.manifest,
            attempt=action.attempt,
        ):
            raise ExecutionAdapterError(
                "persisted credential descriptor differs from the exact worker attempt"
            )
        return descriptor

    def _credential_binding(
        self,
        *,
        action_manifest: WorkerInputManifest,
        attempt: AttemptRecord,
    ) -> CredentialBinding:
        if (
            action_manifest.item_id != attempt.item_id
            or action_manifest.attempt_id != attempt.attempt_id
            or action_manifest.generation != attempt.generation
        ):
            raise ExecutionAdapterError(
                "worker manifest differs from the exact persisted attempt identity"
            )
        return CredentialBinding(
            run_id=action_manifest.run_id,
            item_id=action_manifest.item_id,
            attempt_id=action_manifest.attempt_id,
            task_digest=action_manifest.task_digest,
            generation=action_manifest.generation,
            backend_kind=self._task.execution.agent.backend_kind,
        )

    def _role_policy(self, action: MaterializableAction) -> ResolvedRolePolicy | None:
        runtime = action.manifest.payload.get("runtime")
        if not isinstance(runtime, dict):
            raise ExecutionAdapterError("generic worker input lacks a runtime object")
        if runtime.get("kind") == "gate":
            if action.attempt.role is not Role.GATE:
                raise ExecutionAdapterError("gate runtime must belong to a deterministic Gate")
            return None
        if runtime.get("kind") != "agent":
            raise ExecutionAdapterError(
                f"unsupported generic worker runtime kind {runtime.get('kind')!r}"
            )
        policy = self._role_policy_for_attempt(action.attempt)
        if action.resource != policy.resource:
            raise ExecutionAdapterError(
                "worker action resource differs from the exact task role policy"
            )
        return policy

    def _role_policy_for_attempt(self, attempt: AttemptRecord) -> ResolvedRolePolicy:
        try:
            return resolve_role_policy(
                self._task,
                role=attempt.role,
                profile=attempt.profile,
                resource_class=attempt.resource_class,
                ambient_environment=self._backend_environment,
            )
        except RolePolicyError as error:
            raise ExecutionAdapterError(f"worker role policy failed closed: {error}") from error

    @staticmethod
    def _validate_credential_descriptor(
        descriptor: CredentialDescriptor,
        binding: CredentialBinding,
        policy: ResolvedRolePolicy,
        *,
        require_live: bool,
    ) -> None:
        if descriptor.binding != binding:
            raise ExecutionAdapterError(
                "credential broker handle is not bound to the exact worker attempt"
            )
        broker_policy = policy.credential_broker_policy
        if descriptor.credential_names != broker_policy.allowed_credential_names:
            raise ExecutionAdapterError("credential descriptor names differ from task role policy")
        if not broker_policy.allowed_credential_names:
            if descriptor.state is not CredentialState.NONE or descriptor.handle is not None:
                raise ExecutionAdapterError(
                    "preauthenticated role policy requires an explicit no-credential descriptor"
                )
            return
        if descriptor.state is not CredentialState.BUNDLE or descriptor.handle is None:
            raise ExecutionAdapterError("credential-bearing role policy requires a broker handle")
        if descriptor.handle.broker_id != broker_policy.broker_id:
            raise ExecutionAdapterError(
                "credential descriptor broker differs from task role policy"
            )
        now = int(time.time())
        expiry = descriptor.handle.expires_at_epoch_seconds
        if require_live and expiry <= now:
            raise ExecutionAdapterError("credential descriptor is already expired")
        if expiry > now + broker_policy.per_attempt_ttl_seconds:
            raise ExecutionAdapterError("credential descriptor expiry exceeds task role policy TTL")

    @staticmethod
    def _validate_authoritative_attempt(state: RunState, attempt: AttemptRecord) -> None:
        try:
            authoritative = next(
                candidate
                for candidate in state.item(attempt.item_id).attempts
                if candidate.attempt_id == attempt.attempt_id
            )
        except (KeyError, StopIteration) as error:
            raise ExecutionAdapterError(
                "worker attempt is absent from authoritative run state"
            ) from error
        if authoritative != attempt:
            raise ExecutionAdapterError("worker attempt differs from authoritative run state")

    @staticmethod
    def _credential_revocation_cause(
        attempt: AttemptRecord,
    ) -> CredentialRevocationCause:
        if attempt.status is AttemptStatus.CANCELLED:
            return CredentialRevocationCause.CANCELLED
        if attempt.status in {
            AttemptStatus.PREEMPTED,
            AttemptStatus.RETRYABLE_FAILED,
            AttemptStatus.LOST,
        }:
            return CredentialRevocationCause.REPLACED
        return CredentialRevocationCause.TERMINAL

    def _resources(
        self,
        action: MaterializableAction,
        isolation: WorkerIsolation,
        policy: ResolvedRolePolicy | None,
    ) -> ResourceRequest:
        controller = self._task.execution.slurm.controller
        log_root = self._workspace / "slurm" / "workers" / action.attempt.attempt_id
        if not log_root.is_dir() or log_root.is_symlink():
            raise ExecutionAdapterError("worker log directory was not materialized")
        resource = action.resource if policy is None else policy.resource
        is_horizontal_smith_role = (
            action.attempt.role in {Role.CODER, Role.REVIEWER}
            and action.attempt.profile is DomainProfile.SMITH
        )
        agent_policy_digest = None if policy is None else policy.launch_policy.digest
        container_image = str(controller.image) if policy is None else policy.launch_policy.image
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
            exclusive=(
                self._task.execution.slurm.smith.exclusive if is_horizontal_smith_role else False
            ),
            container_image=container_image,
            mounts=isolation.mounts,
            container_launch_mode=controller.container_launch_mode.value,
            agent_policy_digest=agent_policy_digest,
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
            raise ExecutionAdapterError(
                "worker action paths differ from persisted attempt identity"
            )
        if action.manifest.attempt_id != action.attempt.attempt_id:
            raise ExecutionAdapterError("worker action manifest belongs to another attempt")


def _slurm_time(seconds: int) -> str:
    days, remainder = divmod(seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    prefix = f"{days}-" if days else ""
    return f"{prefix}{hours:02d}:{minutes:02d}:{seconds:02d}"


__all__ = [
    "ExecutionAdapterError",
    "GenericWorkerExecutionAdapter",
    "MaterializableAction",
    "MaterializedWorker",
]
