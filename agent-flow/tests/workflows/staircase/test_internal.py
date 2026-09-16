# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for private Staircase process wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_flow.workflows.staircase import internal
from agent_flow.workflows.staircase.common.credential_broker_client import (
    UnixCredentialBrokerClient,
)
from agent_flow.workflows.staircase.common.credentials import NoCredentialBroker
from agent_flow.workflows.staircase.task_schema import CredentialBrokerPolicy, SlurmRole


def _task(
    tmp_path: Path,
    policy: CredentialBrokerPolicy,
    *,
    socket_path: str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        repository=SimpleNamespace(root=tmp_path),
        execution=SimpleNamespace(
            slurm=SimpleNamespace(
                controller=SimpleNamespace(credential_broker_socket=socket_path),
                role_classes=tuple(
                    SimpleNamespace(credential_broker=policy) for _role in SlurmRole
                ),
            )
        ),
    )


def test_build_credential_broker_uses_explicit_no_credential_mode(
    tmp_path: Path,
) -> None:
    task = _task(
        tmp_path,
        CredentialBrokerPolicy("preauthenticated", (), 3_600),
        socket_path=None,
    )

    assert isinstance(
        internal._build_credential_broker(task, workspace=tmp_path),
        NoCredentialBroker,
    )


def test_controller_preflights_and_shares_one_broker_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_flow.workflows.staircase import task_schema, workflow
    from agent_flow.workflows.staircase.common import gitops
    from agent_flow.workflows.staircase.controller import collective, runtime

    socket_path = (tmp_path / "credential-broker.sock").as_posix()
    task = _task(
        tmp_path,
        CredentialBrokerPolicy(
            "site-codex-broker",
            ("OPENAI_API_KEY",),
            3_600,
        ),
        socket_path=socket_path,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preflighted: list[UnixCredentialBrokerClient] = []
    constructed: dict[str, object] = {}
    invoked: dict[str, object] = {}
    collective_adapter = object()
    git = object()

    monkeypatch.setattr(task_schema, "load_normalized_task", lambda _path: task)
    monkeypatch.setattr(
        UnixCredentialBrokerClient,
        "preflight",
        lambda broker: preflighted.append(broker),
    )
    monkeypatch.setattr(gitops, "ControllerGitOps", lambda *_args: git)
    monkeypatch.setattr(
        collective,
        "build_task_collective_adapter",
        lambda **_kwargs: collective_adapter,
    )

    def build_runtime_factory(**kwargs: object) -> object:
        constructed.update(kwargs)
        return SimpleNamespace(**kwargs)

    def run_controller(path: Path, **kwargs: object) -> None:
        invoked["workspace"] = path
        invoked.update(kwargs)

    monkeypatch.setattr(runtime, "ProductionRuntimeFactory", build_runtime_factory)
    monkeypatch.setattr(workflow, "run_controller", run_controller)

    internal.main(
        [
            "controller",
            "--workspace",
            str(workspace),
            "--generation",
            "7",
            "--owner-nonce",
            "owner-1",
        ]
    )

    assert len(preflighted) == 1
    broker = preflighted[0]
    assert broker.config.socket_path == Path(socket_path)
    assert broker.config.allowed_credential_names == ("OPENAI_API_KEY",)
    assert broker._handle_ledger_path == (
        workspace / "credential-broker-state" / "handle-bindings.jsonl"
    )
    assert constructed["credential_broker"] is broker
    assert invoked["credential_broker"] is broker
    assert invoked["runtime_factory"].credential_broker is broker
    assert invoked["workspace"] == workspace
    assert invoked["generation"] == 7
