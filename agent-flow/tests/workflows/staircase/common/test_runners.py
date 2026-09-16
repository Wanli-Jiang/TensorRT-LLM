# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for isolated Staircase role-process contracts."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialDescriptor,
    CredentialHandle,
    CredentialState,
)
from agent_flow.workflows.staircase.common.launch_policy import AgentLaunchPolicy
from agent_flow.workflows.staircase.common.runners import (
    RoleProcessError,
    RoleProcessSpec,
    execute_role,
    validate_role_spec,
)
from agent_flow.workflows.staircase.common.slurm import AGENT_POLICY_DIGEST_ENVIRONMENT

LAUNCH_POLICY = AgentLaunchPolicy.create(
    image="/images/agent-worker.sqsh",
    build_identity="agent-worker-test",
)


@pytest.fixture(autouse=True)
def bound_launch_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(AGENT_POLICY_DIGEST_ENVIRONMENT, LAUNCH_POLICY.digest)


def _spec(tmp_path: Path, **updates: object) -> RoleProcessSpec:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    worktree = source / "worktree"
    worktree.mkdir(exist_ok=True)
    values: dict[str, object] = {
        "schema_version": 1,
        "run_id": "run-1",
        "task_digest": "a" * 64,
        "generation": 1,
        "item_id": "catalog-1",
        "attempt_id": "attempt-1",
        "prompt_id": "prompt-1",
        "role": "coder",
        "profile": "smith",
        "backend_kind": "codex",
        "model": "test-model",
        "source_root": str(source),
        "cwd": str(worktree),
        "result_path": str((tmp_path / "result.json").resolve()),
        "system_prompt": "system",
        "prompt": "one item",
        "launch_policy": LAUNCH_POLICY.to_public_dict(),
    }
    values.update(updates)
    if "credential_descriptor" not in updates:
        binding = CredentialBinding(
            run_id=values["run_id"],  # type: ignore[arg-type]
            item_id=values["item_id"],  # type: ignore[arg-type]
            attempt_id=values["attempt_id"],  # type: ignore[arg-type]
            task_digest=values["task_digest"],  # type: ignore[arg-type]
            generation=values["generation"],  # type: ignore[arg-type]
            backend_kind=values["backend_kind"],  # type: ignore[arg-type]
        )
        values["credential_descriptor"] = CredentialDescriptor.no_credentials(
            binding
        ).to_public_dict()
    return RoleProcessSpec(**values)


def test_validate_role_spec_canonicalizes_paths(tmp_path: Path) -> None:
    checked = validate_role_spec(_spec(tmp_path))
    assert Path(checked.source_root).is_absolute()
    assert Path(checked.cwd).is_absolute()


def test_role_cwd_cannot_escape_source_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(RoleProcessError, match="beneath source_root"):
        validate_role_spec(_spec(tmp_path, cwd=str(outside)))


@pytest.mark.parametrize("field", ["run_id", "item_id", "attempt_id", "prompt_id"])
def test_identifiers_reject_path_traversal(tmp_path: Path, field: str) -> None:
    spec = _spec(tmp_path)
    values = asdict(spec)
    values[field] = "../escape"
    with pytest.raises(RoleProcessError, match="safe non-empty identifier"):
        validate_role_spec(RoleProcessSpec(**values))


def test_result_path_must_be_absolute(tmp_path: Path) -> None:
    with pytest.raises(RoleProcessError, match="must be absolute"):
        validate_role_spec(_spec(tmp_path, result_path="result.json"))


def test_role_spec_rejects_credential_binding_mismatch(tmp_path: Path) -> None:
    descriptor = CredentialDescriptor.no_credentials(
        CredentialBinding("run-1", "other-item", "attempt-1", "a" * 64, 1, "codex")
    )
    with pytest.raises(RoleProcessError, match="identity/backend differs"):
        validate_role_spec(_spec(tmp_path, credential_descriptor=descriptor.to_public_dict()))


def test_execute_role_clears_ambient_credentials_and_restores_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class FakeLayer:
        def __init__(self, config: object) -> None:
            observed["config"] = config

        def __enter__(self) -> FakeLayer:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __call__(self, prompt: str) -> str:
            observed["prompt"] = prompt
            observed["openai"] = os.environ.get("OPENAI_API_KEY")
            observed["anthropic"] = os.environ.get("ANTHROPIC_API_KEY")
            return '{"schema_version":1,"outcome":"ok"}'

    monkeypatch.setattr("agent_flow.AgentLayer", FakeLayer)
    monkeypatch.setattr(
        "agent_flow.workflows.staircase.common.runners._scheduler_client_path",
        lambda _name: None,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-anthropic")
    spec = _spec(tmp_path)

    result = execute_role(spec)

    assert observed["openai"] is None
    assert observed["anthropic"] is None
    assert os.environ["OPENAI_API_KEY"] == "ambient-openai"
    assert os.environ["ANTHROPIC_API_KEY"] == "ambient-anthropic"
    assert result.generation == 1
    assert Path(spec.result_path).is_file()


def test_execute_role_scopes_bundle_and_redacts_response_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-planning-secret-never-persist"
    base = _spec(tmp_path)
    descriptor = CredentialDescriptor.create(
        CredentialBinding(
            base.run_id,
            base.item_id,
            base.attempt_id,
            base.task_digest,
            base.generation,
            base.backend_kind,
        ),
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "attempt-1", 2_147_483_647),
    )
    spec = _spec(tmp_path, credential_descriptor=descriptor.to_public_dict())
    monkeypatch.setattr(
        "agent_flow.workflows.staircase.common.runners.load_worker_credentials",
        lambda *_args, **_kwargs: {"OPENAI_API_KEY": secret},
    )

    class LeakingLayer:
        def __init__(self, _config: object) -> None:
            pass

        def __enter__(self) -> LeakingLayer:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __call__(self, _prompt: str) -> str:
            assert os.environ["OPENAI_API_KEY"] == secret
            return f"leaked={secret}"

    monkeypatch.setattr("agent_flow.AgentLayer", LeakingLayer)
    with pytest.raises(RoleProcessError) as caught:
        execute_role(spec)
    assert secret not in str(caught.value)
    assert not Path(spec.result_path).exists()


def test_execute_role_rejects_scheduler_client_before_agent_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "agent_flow.workflows.staircase.common.runners._scheduler_client_path",
        lambda name: "/usr/bin/sbatch" if name == "sbatch" else None,
    )

    with pytest.raises(RoleProcessError, match="scheduler clients"):
        execute_role(_spec(tmp_path))
    assert not (tmp_path / "result.json").exists()


def test_execute_role_redacts_backend_standard_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    secret = "sk-planning-stream-secret"
    base = _spec(tmp_path)
    descriptor = CredentialDescriptor.create(
        CredentialBinding(
            base.run_id,
            base.item_id,
            base.attempt_id,
            base.task_digest,
            base.generation,
            base.backend_kind,
        ),
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle("test-broker", "attempt-1", 2_147_483_647),
    )
    monkeypatch.setattr(
        "agent_flow.workflows.staircase.common.runners.load_worker_credentials",
        lambda *_args, **_kwargs: {"OPENAI_API_KEY": secret},
    )

    class PrintingLayer:
        def __init__(self, _config: object) -> None:
            pass

        def __enter__(self) -> PrintingLayer:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __call__(self, _prompt: str) -> str:
            os.write(1, f"backend stdout {secret}\n".encode())
            os.write(2, f"backend stderr {secret}\n".encode())
            return '{"schema_version":1,"outcome":"ok"}'

    monkeypatch.setattr("agent_flow.AgentLayer", PrintingLayer)
    monkeypatch.setattr(
        "agent_flow.workflows.staircase.common.runners._scheduler_client_path",
        lambda _name: None,
    )
    execute_role(_spec(tmp_path, credential_descriptor=descriptor.to_public_dict()))

    captured = capfd.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "backend stdout <redacted>" in captured.out
    assert "backend stderr <redacted>" in captured.err
