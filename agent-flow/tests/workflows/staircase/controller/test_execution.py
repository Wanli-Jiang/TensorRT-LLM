# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the generic worker credential admission boundary."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from agent_flow.workflows.staircase.common.artifacts import (
    INPUT_FILENAME,
    WorkerInputManifest,
    load_input_manifest,
)
from agent_flow.workflows.staircase.common.credentials import (
    CREDENTIAL_MOUNT_PATH,
    CredentialBinding,
    CredentialDescriptor,
    CredentialHandle,
    CredentialProvision,
    CredentialRevocationCause,
    CredentialRevocationReceipt,
    CredentialState,
    NoCredentialBroker,
    descriptor_from_public_dict,
    locate_credential_provision,
    materialize_credential_provision,
    revoke_terminal_attempt_credentials,
)
from agent_flow.workflows.staircase.controller.execution import (
    ExecutionAdapterError,
    GenericWorkerExecutionAdapter,
)
from agent_flow.workflows.staircase.state import (
    AttemptKind,
    AttemptRecord,
    AttemptStatus,
    DomainProfile,
    JobReference,
    Role,
)
from agent_flow.workflows.staircase.task_schema import (
    ContainerLaunchMode,
    CredentialBrokerPolicy,
    NetworkEnforcement,
    ResourceClass,
    RoleEnvironmentPolicy,
    RoleNetworkPolicy,
    SlurmRole,
    SlurmRoleClass,
)

_TASK_DIGEST = "a" * 64
_SECRET = "credential-must-never-enter-input"


@dataclass(frozen=True, slots=True)
class _Action:
    attempt: AttemptRecord
    resource: ResourceClass
    worktree: Path
    branch: str
    base_commit: str
    attempt_dir: Path
    manifest: WorkerInputManifest


@dataclass(frozen=True, slots=True)
class _State:
    run_id: str
    task_digest: str
    attempt: AttemptRecord

    def item(self, item_id: str) -> SimpleNamespace:
        if item_id != self.attempt.item_id:
            raise KeyError(item_id)
        return SimpleNamespace(attempts=(self.attempt,))


class _FakeGit:
    @contextmanager
    def transaction(self) -> Iterator[_FakeGit]:
        yield self

    def create_candidate_overlay(self, path: Path, *, base_commit: str) -> Path:
        assert base_commit == "base-commit"
        path.mkdir(parents=True)
        (path / "model.py").write_text("MODEL = 'test'\n", encoding="utf-8")
        return path


class _CredentialBroker:
    def __init__(
        self,
        *,
        broker_id: str = "test-broker",
        expiry_offset_seconds: int = 3_600,
        no_credentials: bool = False,
    ) -> None:
        self._broker_id = broker_id
        self._expiry_offset_seconds = expiry_offset_seconds
        self._no_credentials = no_credentials
        self.described: list[CredentialBinding] = []
        self.injected: list[CredentialDescriptor] = []
        self.revoked: list[tuple[CredentialBinding, CredentialRevocationCause]] = []

    def describe(self, binding: CredentialBinding) -> CredentialDescriptor:
        self.described.append(binding)
        if self._no_credentials:
            return CredentialDescriptor.no_credentials(binding)
        return CredentialDescriptor.create(
            binding,
            CredentialState.BUNDLE,
            ("OPENAI_API_KEY",),
            handle=CredentialHandle(
                self._broker_id,
                f"handle-{binding.attempt_id}",
                int(time.time()) + self._expiry_offset_seconds,
            ),
        )

    def inject_after_admission(
        self,
        *,
        workspace: Path,
        descriptor: CredentialDescriptor,
    ) -> CredentialProvision:
        self.injected.append(descriptor)
        return materialize_credential_provision(
            workspace=workspace,
            descriptor=descriptor,
            credential_values={"OPENAI_API_KEY": _SECRET},
        )

    def revoke(
        self,
        *,
        workspace: Path,
        expected_binding: CredentialBinding,
        descriptor: CredentialDescriptor,
        cause: CredentialRevocationCause,
    ) -> CredentialRevocationReceipt:
        self.revoked.append((expected_binding, cause))
        return revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=expected_binding,
            provision=locate_credential_provision(
                workspace=workspace,
                descriptor=descriptor,
            ),
            cause=cause,
        )


def _adapter_fixture(
    tmp_path: Path,
    *,
    broker_id: str = "test-broker",
    expiry_offset_seconds: int = 3_600,
    no_credentials: bool = False,
) -> tuple[GenericWorkerExecutionAdapter, _CredentialBroker, _Action, _State]:
    repository = tmp_path / "repository"
    repository.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = tmp_path / "runtime.sqsh"
    image.write_bytes(b"image")
    agent_image = tmp_path / "agent-worker.sqsh"
    agent_image.write_bytes(b"agent image")
    controller = SimpleNamespace(
        environment=(),
        mounts=(
            SimpleNamespace(
                host_path=tmp_path,
                container_path="/mnt/test",
                read_only=False,
            ),
        ),
        account="coreai",
        partition="batch",
        qos=None,
        reservation=None,
        image=image,
        container_launch_mode=ContainerLaunchMode.IN_ALLOCATION_SRUN,
    )
    resource = ResourceClass("coder_analysis", 1, 1, 0, 2, 1024, 600)
    role_class = SlurmRoleClass(
        SlurmRole.ASSEMBLER_CODER,
        resource.name,
        RoleNetworkPolicy.BACKEND_API_ONLY,
        RoleEnvironmentPolicy((), (("PYTHONNOUSERSITE", "1"),)),
        CredentialBrokerPolicy("test-broker", ("OPENAI_API_KEY",), 7_200),
    )
    task = SimpleNamespace(
        repository=SimpleNamespace(root=repository),
        reference=SimpleNamespace(checkpoint=checkpoint, additional_sources=()),
        execution=SimpleNamespace(
            slurm=SimpleNamespace(
                controller=controller,
                agent_worker=SimpleNamespace(
                    image=agent_image,
                    build_identity="agent-worker-test",
                    scheduler_clients=False,
                    network_enforcement=NetworkEnforcement.CERTIFIED_WORKER_IMAGE,
                ),
                smith=SimpleNamespace(
                    resource_classes=(resource,),
                    distinct_nodes_required=False,
                    exclusive=False,
                ),
                role_classes=(role_class,),
            ),
            agent=SimpleNamespace(backend_kind="codex", model="test-model"),
        ),
    )
    attempt = AttemptRecord(
        attempt_id="item-a.0001.role",
        item_id="item-a",
        sequence=1,
        role=Role.CODER,
        kind=AttemptKind.ROLE,
        generation=1,
        profile=DomainProfile.ASSEMBLER,
        resource_class="coder_analysis",
    )
    worktree = workspace / "candidates" / attempt.item_id / attempt.attempt_id
    attempt_dir = workspace / "items" / attempt.item_id / "attempts" / "0001"
    manifest = WorkerInputManifest(
        run_id="run-a",
        item_id=attempt.item_id,
        attempt_id=attempt.attempt_id,
        task_digest=_TASK_DIGEST,
        generation=attempt.generation,
        role=attempt.role,
        profile=attempt.profile,
        worktree=str(worktree),
        payload={
            "runtime": {
                "kind": "agent",
                "backend_kind": "codex",
                "model": "test-model",
            },
            "context": {"goal": "test credential isolation"},
        },
    )
    action = _Action(
        attempt=attempt,
        resource=resource,
        worktree=worktree,
        branch="staircase/item-a",
        base_commit="base-commit",
        attempt_dir=attempt_dir,
        manifest=manifest,
    )
    broker = _CredentialBroker(
        broker_id=broker_id,
        expiry_offset_seconds=expiry_offset_seconds,
        no_credentials=no_credentials,
    )
    adapter = GenericWorkerExecutionAdapter(
        workspace=workspace,
        task=task,
        git=_FakeGit(),
        credential_broker=broker,
    )
    return adapter, broker, action, _State("run-a", _TASK_DIGEST, attempt)


def test_adapter_requires_an_explicit_credential_broker(tmp_path: Path) -> None:
    adapter, _broker, _action, _state = _adapter_fixture(tmp_path)

    with pytest.raises(ExecutionAdapterError, match="explicit CredentialBroker"):
        GenericWorkerExecutionAdapter(
            workspace=adapter._workspace,
            task=adapter._task,
            git=adapter._git,
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"broker_id": "other-broker"}, "broker differs"),
        ({"expiry_offset_seconds": 7_300}, "expiry exceeds"),
        ({"no_credentials": True}, "names differ"),
    ],
)
def test_descriptor_must_match_selected_role_broker_policy(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    message: str,
) -> None:
    adapter, _broker, action, _state = _adapter_fixture(tmp_path, **fixture_kwargs)

    with pytest.raises(ExecutionAdapterError, match=message):
        adapter.materialize(action)


def test_explicit_empty_role_policy_accepts_only_no_credential_broker(
    tmp_path: Path,
) -> None:
    adapter, _broker, action, _state = _adapter_fixture(tmp_path)
    role_class = adapter._task.execution.slurm.role_classes[0]
    adapter._task.execution.slurm.role_classes = (
        replace(
            role_class,
            credential_broker=CredentialBrokerPolicy("preauthenticated", (), 7_200),
        ),
    )
    preauthenticated = GenericWorkerExecutionAdapter(
        workspace=adapter._workspace,
        task=adapter._task,
        git=adapter._git,
        credential_broker=NoCredentialBroker(),
    )

    preauthenticated.materialize(action)
    manifest, _digest = load_input_manifest(action.attempt_dir / INPUT_FILENAME)
    runtime = manifest.payload["runtime"]
    assert isinstance(runtime, dict)
    descriptor = descriptor_from_public_dict(runtime["credential_descriptor"])
    assert descriptor.state is CredentialState.NONE
    assert descriptor.handle is None


def test_materialize_persists_only_public_handle_in_metadata_free_overlay(
    tmp_path: Path,
) -> None:
    adapter, broker, action, _state = _adapter_fixture(tmp_path)

    materialized = adapter.materialize(action)

    assert len(broker.described) == 1
    assert broker.injected == []
    assert materialized.isolation.credential_descriptor is not None
    assert not (action.worktree / ".git").exists()
    persisted, _digest = load_input_manifest(action.attempt_dir / INPUT_FILENAME)
    runtime = persisted.payload["runtime"]
    assert isinstance(runtime, dict)
    descriptor = descriptor_from_public_dict(runtime["credential_descriptor"])
    assert descriptor.binding == broker.described[0]
    assert _SECRET not in (action.attempt_dir / INPUT_FILENAME).read_text(encoding="utf-8")
    assert all(mount.target != CREDENTIAL_MOUNT_PATH for mount in materialized.isolation.mounts)

    replayed = adapter.materialize(action)

    assert replayed == materialized
    assert len(broker.described) == 1
    assert broker.injected == []


def test_action_resource_must_equal_selected_role_policy(tmp_path: Path) -> None:
    adapter, _broker, action, _state = _adapter_fixture(tmp_path)
    mismatched = replace(
        action,
        resource=replace(action.resource, cpus_per_task=action.resource.cpus_per_task + 1),
    )

    with pytest.raises(ExecutionAdapterError, match="resource differs"):
        adapter.materialize(mismatched)


def test_execution_injects_once_before_authoritative_submission_intent(tmp_path: Path) -> None:
    adapter, broker, action, prepared_state = _adapter_fixture(tmp_path)
    adapter.materialize(action)

    prepared = adapter.execution(prepared_state, action)

    assert len(broker.injected) == 1
    assert prepared.environment == (("PYTHONNOUSERSITE", "1"),)
    assert any(mount.target == CREDENTIAL_MOUNT_PATH for mount in prepared.resources.mounts)
    assert prepared.resources.container_image.endswith("agent-worker.sqsh")
    assert prepared.resources.agent_policy_digest is not None
    assert not prepared.resources.exclusive
    persisted, _digest = load_input_manifest(action.attempt_dir / INPUT_FILENAME)
    runtime = persisted.payload["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["launch_policy"]["image"].endswith("agent-worker.sqsh")
    assert runtime["placement_contract"] == {
        "distinct_nodes_required": False,
        "exclusive": False,
    }

    submitting_attempt = action.attempt.transition(
        AttemptStatus.SUBMITTING,
        submission_token=prepared.submission_token,
        scheduler_state="SUBMISSION_INTENT",
    )
    submitting_action = replace(action, attempt=submitting_attempt)
    admitted = adapter.execution(
        replace(prepared_state, attempt=submitting_attempt),
        submitting_action,
    )

    assert len(broker.injected) == 1
    credential_mounts = tuple(
        mount for mount in admitted.resources.mounts if mount.target == CREDENTIAL_MOUNT_PATH
    )
    assert len(credential_mounts) == 1
    assert credential_mounts[0].read_only


def test_execution_rejects_broker_descriptor_substitution(tmp_path: Path) -> None:
    adapter, broker, action, prepared_state = _adapter_fixture(tmp_path)
    adapter.materialize(action)
    original_inject = broker.inject_after_admission

    def inject_other_handle(
        *, workspace: Path, descriptor: CredentialDescriptor
    ) -> CredentialProvision:
        provision = original_inject(workspace=workspace, descriptor=descriptor)
        other = CredentialDescriptor.create(
            descriptor.binding,
            descriptor.state,
            descriptor.credential_names,
            handle=CredentialHandle("test-broker", "substituted-handle", 2_147_483_647),
        )
        return CredentialProvision(other, provision.bundle_path)

    broker.inject_after_admission = inject_other_handle  # type: ignore[method-assign]

    with pytest.raises(ExecutionAdapterError, match="changed the admitted public handle"):
        adapter.execution(prepared_state, action)


def test_expired_descriptor_does_not_block_post_submission_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, broker, action, prepared_state = _adapter_fixture(tmp_path)
    adapter.materialize(action)
    prepared = adapter.execution(prepared_state, action)
    assert prepared.credential_expires_at_epoch_seconds is not None
    monkeypatch.setattr(
        "agent_flow.workflows.staircase.controller.execution.time.time",
        lambda: prepared.credential_expires_at_epoch_seconds + 1,
    )

    attempt = action.attempt.transition(
        AttemptStatus.SUBMITTING,
        submission_token=prepared.submission_token,
        scheduler_state="SUBMISSION_INTENT",
    )
    attempts = [attempt]
    attempt = attempt.transition(
        AttemptStatus.SUBMITTED,
        job=JobReference("12345"),
        scheduler_state="PENDING",
    )
    attempts.append(attempt)
    attempt = attempt.transition(AttemptStatus.PENDING, scheduler_state="PENDING")
    attempts.append(attempt)
    attempt = attempt.transition(AttemptStatus.RUNNING, scheduler_state="RUNNING")
    attempts.append(attempt)
    attempt = attempt.transition(
        AttemptStatus.TERMINAL_OBSERVED,
        scheduler_state="COMPLETED",
    )
    attempt = attempt.transition(AttemptStatus.COLLECTING)
    attempts.append(attempt)

    for persisted in attempts:
        execution = adapter.execution(
            replace(prepared_state, attempt=persisted),
            replace(action, attempt=persisted),
        )
        assert execution.submission_token == prepared.submission_token
    assert len(broker.injected) == 1


def test_horizontal_smith_coder_requests_exclusive_agent_image(tmp_path: Path) -> None:
    adapter, _broker, action, state = _adapter_fixture(tmp_path)
    attempt = replace(action.attempt, profile=DomainProfile.SMITH)
    manifest = replace(action.manifest, profile=DomainProfile.SMITH)
    action = replace(action, attempt=attempt, manifest=manifest)
    state = replace(state, attempt=attempt)
    role_class = adapter._task.execution.slurm.role_classes[0]
    adapter._task.execution.slurm.role_classes = (replace(role_class, role=SlurmRole.SMITH_CODER),)
    adapter._task.execution.slurm.smith.distinct_nodes_required = True
    adapter._task.execution.slurm.smith.exclusive = True

    adapter.materialize(action)
    execution = adapter.execution(state, action)

    assert execution.resources.exclusive
    persisted, _digest = load_input_manifest(action.attempt_dir / INPUT_FILENAME)
    runtime = persisted.payload["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["placement_contract"] == {
        "distinct_nodes_required": True,
        "exclusive": True,
    }


@pytest.mark.parametrize(
    "kind",
    [AttemptKind.REVIEWER_ANALYSIS, AttemptKind.REVIEWER_RERUN],
)
def test_horizontal_smith_reviewer_stage_requests_exclusive_placement(
    tmp_path: Path,
    kind: AttemptKind,
) -> None:
    adapter, _broker, action, state = _adapter_fixture(tmp_path)
    attempt = replace(
        action.attempt,
        role=Role.REVIEWER,
        kind=kind,
        profile=DomainProfile.SMITH,
        review_of_attempt_id="coder-attempt",
        reviewed_candidate_digest="c" * 64,
    )
    manifest = replace(
        action.manifest,
        role=Role.REVIEWER,
        profile=DomainProfile.SMITH,
    )
    action = replace(action, attempt=attempt, manifest=manifest)
    state = replace(state, attempt=attempt)
    role_class = adapter._task.execution.slurm.role_classes[0]
    adapter._task.execution.slurm.role_classes = (replace(role_class, role=SlurmRole.REVIEWER),)
    adapter._task.execution.slurm.smith.distinct_nodes_required = True
    adapter._task.execution.slurm.smith.exclusive = True

    adapter.materialize(action)
    execution = adapter.execution(state, action)

    assert execution.resources.exclusive
    persisted, _digest = load_input_manifest(action.attempt_dir / INPUT_FILENAME)
    runtime = persisted.payload["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["placement_contract"] == {
        "distinct_nodes_required": True,
        "exclusive": True,
    }


def test_reviewer_overlay_digest_rejects_post_publication_mutation(tmp_path: Path) -> None:
    adapter, _broker, action, state = _adapter_fixture(tmp_path)
    attempt = replace(action.attempt, role=Role.REVIEWER)
    manifest = replace(action.manifest, role=Role.REVIEWER)
    action = replace(action, attempt=attempt, manifest=manifest)
    state = replace(state, attempt=attempt)
    role_class = adapter._task.execution.slurm.role_classes[0]
    adapter._task.execution.slurm.role_classes = (replace(role_class, role=SlurmRole.REVIEWER),)

    adapter.materialize(action)
    persisted, _digest = load_input_manifest(action.attempt_dir / INPUT_FILENAME)
    runtime = persisted.payload["runtime"]
    assert isinstance(runtime, dict)
    assert isinstance(runtime["candidate_overlay_digest"], str)
    (action.worktree / "model.py").write_text("MODEL = 'mutated'\n", encoding="utf-8")

    with pytest.raises(ExecutionAdapterError, match="immutable worker input differs"):
        adapter.materialize(action)
    with pytest.raises(ExecutionAdapterError, match="persisted worker input differs"):
        adapter.execution(state, action)


@pytest.mark.parametrize(
    ("status", "cause"),
    [
        (AttemptStatus.FAILED, CredentialRevocationCause.TERMINAL),
        (AttemptStatus.RETRYABLE_FAILED, CredentialRevocationCause.REPLACED),
        (AttemptStatus.CANCELLED, CredentialRevocationCause.CANCELLED),
    ],
)
def test_terminal_revocation_is_exact_typed_and_idempotent(
    tmp_path: Path,
    status: AttemptStatus,
    cause: CredentialRevocationCause,
) -> None:
    adapter, broker, action, prepared_state = _adapter_fixture(tmp_path)
    materialized = adapter.materialize(action)
    input_path = action.attempt_dir / INPUT_FILENAME
    input_before = input_path.read_bytes()
    output = action.attempt_dir / "output"
    (output / INPUT_FILENAME).write_text('{"forged":true}\n', encoding="utf-8")
    input_mount = next(
        mount for mount in materialized.isolation.mounts if mount.source == input_path
    )
    output_mount = next(mount for mount in materialized.isolation.mounts if mount.source == output)
    assert input_mount.read_only
    assert not output_mount.read_only
    assert input_path.read_bytes() == input_before
    prepared = adapter.execution(prepared_state, action)
    submitting_attempt = action.attempt.transition(
        AttemptStatus.SUBMITTING,
        submission_token=prepared.submission_token,
        scheduler_state="SUBMISSION_INTENT",
    )
    adapter.execution(
        replace(prepared_state, attempt=submitting_attempt),
        replace(action, attempt=submitting_attempt),
    )
    terminal_attempt = replace(
        submitting_attempt,
        status=status,
        terminal_reason="terminal for test",
    )
    terminal_state = replace(prepared_state, attempt=terminal_attempt)

    assert adapter.revoke(terminal_state, terminal_attempt)
    assert not adapter.revoke(terminal_state, terminal_attempt)
    assert input_path.read_bytes() == input_before
    assert broker.revoked == [
        (broker.described[0], cause),
        (broker.described[0], cause),
    ]


def test_revocation_rejects_mismatched_broker_receipt(tmp_path: Path) -> None:
    adapter, broker, action, prepared_state = _adapter_fixture(tmp_path)
    adapter.materialize(action)
    terminal_attempt = replace(
        action.attempt,
        status=AttemptStatus.FAILED,
        submission_token="worker-token",
        terminal_reason="terminal for test",
    )
    terminal_state = replace(prepared_state, attempt=terminal_attempt)

    def mismatched_revoke(
        *,
        workspace: Path,
        expected_binding: CredentialBinding,
        descriptor: CredentialDescriptor,
        cause: CredentialRevocationCause,
    ) -> CredentialRevocationReceipt:
        del cause
        return CredentialRevocationReceipt(
            descriptor,
            CredentialRevocationCause.REPLACED,
            workspace
            / "controller-secrets"
            / "credential-revocation-receipts"
            / expected_binding.run_id
            / expected_binding.item_id
            / f"{expected_binding.attempt_id}.json",
        )

    broker.revoke = mismatched_revoke  # type: ignore[method-assign]

    with pytest.raises(ExecutionAdapterError, match="mismatched receipt"):
        adapter.revoke(terminal_state, terminal_attempt)
