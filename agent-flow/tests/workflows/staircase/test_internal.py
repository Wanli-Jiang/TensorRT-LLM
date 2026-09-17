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
    source_auth_json_path: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        repository=SimpleNamespace(root=tmp_path),
        execution=SimpleNamespace(
            slurm=SimpleNamespace(
                controller=SimpleNamespace(
                    credential_broker_socket=socket_path,
                    source_auth_json_path=source_auth_json_path,
                ),
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
    assert internal._build_managed_credential_broker_server(task, workspace=tmp_path) is None


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


@pytest.mark.parametrize("controller_fails", [False, True])
def test_controller_owns_managed_oauth_broker_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controller_fails: bool,
) -> None:
    from agent_flow.workflows.staircase import task_schema, workflow
    from agent_flow.workflows.staircase.common import (
        credential_broker_client,
        credential_broker_server,
        gitops,
    )
    from agent_flow.workflows.staircase.controller import collective, runtime

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "auth.json"
    source.write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")
    source.chmod(0o600)
    socket_path = (
        Path("/tmp") / f"staircase-credential-broker-test-{workspace.stat().st_ino}" / "broker.sock"
    )
    task = _task(
        tmp_path,
        CredentialBrokerPolicy(
            "controller-codex-oauth",
            ("CODEX_AUTH_JSON",),
            3_600,
        ),
        socket_path=socket_path.as_posix(),
        source_auth_json_path=source,
    )
    events: list[str] = []
    captured: dict[str, object] = {}

    class FakeServer:
        def __init__(self, config: object) -> None:
            captured["config"] = config
            self.config = config

        def start_in_background(self) -> None:
            events.append("start")

        def shutdown(self) -> None:
            events.append("shutdown")

        def wait(self, timeout: float) -> None:
            captured["wait_timeout"] = timeout
            events.append("wait")

    class FakeClientConfig(SimpleNamespace):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)

    monkeypatch.setattr(
        credential_broker_server,
        "CredentialBrokerServerConfig",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(credential_broker_server, "UnixCredentialBrokerServer", FakeServer)
    monkeypatch.setattr(credential_broker_client, "UnixCredentialBrokerConfig", FakeClientConfig)
    monkeypatch.setattr(task_schema, "load_normalized_task", lambda _path: task)
    monkeypatch.setattr(
        UnixCredentialBrokerClient,
        "preflight",
        lambda _broker: events.append("preflight"),
    )
    monkeypatch.setattr(gitops, "ControllerGitOps", lambda *_args: object())
    monkeypatch.setattr(
        collective,
        "build_task_collective_adapter",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(runtime, "ProductionRuntimeFactory", lambda **kwargs: kwargs)

    def run_controller(_path: Path, **_kwargs: object) -> None:
        events.append("run")
        if controller_fails:
            raise RuntimeError("controller failed")

    monkeypatch.setattr(workflow, "run_controller", run_controller)

    argv = [
        "controller",
        "--workspace",
        str(workspace),
        "--generation",
        "1",
        "--owner-nonce",
        "owner-1",
    ]
    if controller_fails:
        with pytest.raises(RuntimeError, match="controller failed"):
            internal.main(argv)
    else:
        internal.main(argv)

    config = captured["config"]
    assert config.socket_path == socket_path
    assert config.source_auth_json_path == source
    assert config.workspace_roots == (workspace,)
    assert config.broker_id == "controller-codex-oauth"
    assert config.allowed_credential_names == ("CODEX_AUTH_JSON",)
    assert events == ["start", "preflight", "run", "shutdown", "wait"]
    assert captured["wait_timeout"] == internal._BROKER_SHUTDOWN_TIMEOUT_SECONDS
    assert not socket_path.parent.exists()


def test_managed_oauth_broker_rejects_socket_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = tmp_path / "auth.json"
    source.write_text("{}", encoding="utf-8")
    source.chmod(0o600)
    task = _task(
        tmp_path,
        CredentialBrokerPolicy("controller-codex-oauth", ("CODEX_AUTH_JSON",), 3_600),
        socket_path=(tmp_path / "outside" / "broker.sock").as_posix(),
        source_auth_json_path=source,
    )

    with pytest.raises(RuntimeError, match="does not use an admitted location"):
        internal._build_managed_credential_broker_server(task, workspace=workspace)
