# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the same-user Staircase credential broker sidecar."""

from __future__ import annotations

import json
import os
import socket
import struct
import time
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.credential_broker_client import (
    CredentialBrokerClientError,
    UnixCredentialBrokerClient,
    UnixCredentialBrokerConfig,
)
from agent_flow.workflows.staircase.common.credential_broker_server import (
    CODEX_AUTH_JSON,
    CredentialBrokerServerConfig,
    CredentialBrokerServerError,
    UnixCredentialBrokerServer,
)
from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialRevocationCause,
    load_worker_credentials,
)

_LENGTH = struct.Struct("!I")
_AUTH_DOCUMENT = {
    "auth_mode": "chatgpt",
    "tokens": {
        "access_token": "oauth-access-secret",
        "refresh_token": "oauth-refresh-secret",
    },
}


def _private_directory(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path.resolve()


def _source_auth(path: Path) -> Path:
    path.write_text(json.dumps(_AUTH_DOCUMENT), encoding="utf-8")
    path.chmod(0o600)
    return path.resolve()


def _server_config(tmp_path: Path, workspace_root: Path) -> CredentialBrokerServerConfig:
    private = _private_directory(tmp_path / "private")
    return CredentialBrokerServerConfig(
        socket_path=private / "broker.sock",
        broker_id="site-codex-oauth",
        allowed_credential_names=(CODEX_AUTH_JSON,),
        source_auth_json_path=_source_auth(private / "auth.json"),
        workspace_roots=(workspace_root,),
    )


def _client(
    config: CredentialBrokerServerConfig, controller_workspace: Path
) -> UnixCredentialBrokerClient:
    return UnixCredentialBrokerClient(
        UnixCredentialBrokerConfig(
            socket_path=config.socket_path,
            broker_id=config.broker_id,
            allowed_credential_names=config.allowed_credential_names,
            attempt_ttl_seconds=600,
            expected_socket_owner_uid=os.geteuid(),
            expected_socket_group_gid=os.getegid(),
        ),
        controller_workspace=controller_workspace,
        handle_ledger_path=(
            controller_workspace / "credential-broker-state" / "handle-bindings.jsonl"
        ),
    )


def _binding(attempt_id: str = "attempt-1") -> CredentialBinding:
    return CredentialBinding(
        run_id="run-1",
        item_id="item-1",
        attempt_id=attempt_id,
        task_digest="a" * 64,
        generation=1,
        backend_kind="codex",
    )


def test_sidecar_describes_injects_and_revokes_exact_codex_oauth_bundle(
    tmp_path: Path,
) -> None:
    workspace_root = _private_directory(tmp_path / "workspaces")
    workspace = _private_directory(workspace_root / "run")
    config = _server_config(tmp_path, workspace_root)
    server = UnixCredentialBrokerServer(config)

    with server.running():
        assert config.socket_path.is_socket()
        assert config.socket_path.stat().st_mode & 0o777 == 0o600
        client = _client(config, workspace)
        client.preflight()
        binding = _binding()
        descriptor = client.describe(binding)
        public = json.dumps(descriptor.to_public_dict(), sort_keys=True)
        assert "oauth-access-secret" not in public
        assert descriptor.credential_names == (CODEX_AUTH_JSON,)

        provision = client.inject_after_admission(
            workspace=workspace,
            descriptor=descriptor,
        )
        assert provision.bundle_path is not None
        credentials = load_worker_credentials(
            descriptor,
            expected_binding=binding,
            bundle_path=provision.bundle_path,
        )
        assert json.loads(credentials[CODEX_AUTH_JSON]) == _AUTH_DOCUMENT

        first = client.revoke(
            workspace=workspace,
            expected_binding=binding,
            descriptor=descriptor,
            cause=CredentialRevocationCause.TERMINAL,
        )
        repeated = client.revoke(
            workspace=workspace,
            expected_binding=binding,
            descriptor=descriptor,
            cause=CredentialRevocationCause.TERMINAL,
        )
        assert repeated == first
        assert not provision.bundle_path.exists()
        with pytest.raises(CredentialBrokerClientError, match="denied"):
            client.inject_after_admission(workspace=workspace, descriptor=descriptor)

    assert not config.socket_path.exists()


def test_sidecar_denies_workspace_outside_allowlist(tmp_path: Path) -> None:
    workspace_root = _private_directory(tmp_path / "workspaces")
    allowed_workspace = _private_directory(workspace_root / "allowed")
    foreign_workspace = _private_directory(tmp_path / "foreign")
    config = _server_config(tmp_path, workspace_root)
    server = UnixCredentialBrokerServer(config)

    with server.running():
        client = _client(config, allowed_workspace)
        descriptor = client.describe(_binding())
        with pytest.raises(CredentialBrokerClientError, match="denied"):
            client.inject_after_admission(
                workspace=foreign_workspace,
                descriptor=descriptor,
            )

    assert not (foreign_workspace / "controller-secrets").exists()


def test_sidecar_bounds_issued_handles(tmp_path: Path) -> None:
    workspace_root = _private_directory(tmp_path / "workspaces")
    workspace = _private_directory(workspace_root / "run")
    config = _server_config(tmp_path, workspace_root)
    server = UnixCredentialBrokerServer(config, max_issued_handles=1)

    with server.running():
        client = _client(config, workspace)
        assert client.describe(_binding("attempt-1")).binding.attempt_id == "attempt-1"
        with pytest.raises(CredentialBrokerClientError, match="unavailable"):
            client.describe(_binding("attempt-2"))


def test_sidecar_rejects_noncanonical_and_trailing_frames(tmp_path: Path) -> None:
    workspace_root = _private_directory(tmp_path / "workspaces")
    config = _server_config(tmp_path, workspace_root)
    server = UnixCredentialBrokerServer(config)

    with server.running():
        for raw in (b'{"schema_version": 1}', b"{}x"):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(2.0)
                connection.connect(os.fspath(config.socket_path))
                declared = len(raw) - 1 if raw.endswith(b"x") else len(raw)
                connection.sendall(_LENGTH.pack(declared) + raw)
                try:
                    closed = connection.recv(1)
                except ConnectionResetError:
                    closed = b""
                assert closed == b""


@pytest.mark.parametrize("mode", (0o644, 0o660, 0o777))
def test_config_rejects_nonprivate_source(tmp_path: Path, mode: int) -> None:
    private = _private_directory(tmp_path / "private")
    source = _source_auth(private / "auth.json")
    source.chmod(mode)
    workspace_root = _private_directory(tmp_path / "workspaces")

    with pytest.raises(CredentialBrokerServerError, match="private"):
        CredentialBrokerServerConfig(
            socket_path=private / "broker.sock",
            broker_id="site-codex-oauth",
            allowed_credential_names=(CODEX_AUTH_JSON,),
            source_auth_json_path=source,
            workspace_roots=(workspace_root,),
        )


def test_source_replacement_after_start_fails_without_materialization(tmp_path: Path) -> None:
    workspace_root = _private_directory(tmp_path / "workspaces")
    workspace = _private_directory(workspace_root / "run")
    config = _server_config(tmp_path, workspace_root)
    server = UnixCredentialBrokerServer(config)

    with server.running():
        client = _client(config, workspace)
        descriptor = client.describe(_binding())
        config.source_auth_json_path.chmod(0o644)
        with pytest.raises(CredentialBrokerClientError, match="invalid_request"):
            client.inject_after_admission(workspace=workspace, descriptor=descriptor)

    assert not (workspace / "controller-secrets").exists()


def test_shutdown_is_bounded_and_removes_owned_socket(tmp_path: Path) -> None:
    workspace_root = _private_directory(tmp_path / "workspaces")
    config = _server_config(tmp_path, workspace_root)
    server = UnixCredentialBrokerServer(config)
    thread = server.start_in_background()

    started = time.monotonic()
    server.shutdown()
    server.wait(2.0)

    assert not thread.is_alive()
    assert time.monotonic() - started < 2.0
    assert not config.socket_path.exists()
