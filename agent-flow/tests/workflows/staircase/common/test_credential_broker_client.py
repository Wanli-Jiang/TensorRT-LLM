# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded Unix-domain-socket credential broker client."""

from __future__ import annotations

import json
import os
import stat
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent_flow.workflows.staircase.common import credential_broker_client as broker_module
from agent_flow.workflows.staircase.common.credential_broker_client import (
    CredentialBrokerClientError,
    UnixCredentialBrokerClient,
    UnixCredentialBrokerConfig,
)
from agent_flow.workflows.staircase.common.credentials import (
    CredentialBinding,
    CredentialDescriptor,
    CredentialError,
    CredentialHandle,
    CredentialRevocationCause,
    CredentialState,
    materialize_credential_provision,
)

_LENGTH = struct.Struct("!I")
_CLOCK = 1_800_000_000
_SECRET = "sk-test-never-cross-the-broker-protocol"


def _binding(*, attempt_id: str = "attempt-1") -> CredentialBinding:
    return CredentialBinding(
        run_id="run-1",
        item_id="item-1",
        attempt_id=attempt_id,
        task_digest="a" * 64,
        generation=1,
        backend_kind="codex",
    )


def _config(socket_path: Path) -> UnixCredentialBrokerConfig:
    return UnixCredentialBrokerConfig(
        socket_path=socket_path,
        broker_id="site-codex-broker",
        allowed_credential_names=("OPENAI_API_KEY",),
        attempt_ttl_seconds=3_600,
        expected_socket_owner_uid=os.geteuid(),
        expected_socket_group_gid=os.getegid(),
    )


def _descriptor(
    binding: CredentialBinding,
    *,
    broker_id: str = "site-codex-broker",
    expiry: int = _CLOCK + 3_600,
) -> CredentialDescriptor:
    return CredentialDescriptor.create(
        binding,
        CredentialState.BUNDLE,
        ("OPENAI_API_KEY",),
        handle=CredentialHandle(broker_id, f"handle-{binding.attempt_id}", expiry),
    )


def _canonical(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _response(
    request: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    status: str = "ok",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": request["operation"],
        "request_id": request["request_id"],
        "broker_id": request["broker_id"],
        "status": status,
        "payload": dict(payload),
    }


@dataclass(frozen=True)
class _WireReply:
    payload: bytes
    declared_length: int | None = None
    trailing: bytes = b""


class _FakeConnection:
    def __init__(
        self,
        transport: _FakeTransport,
        index: int,
    ) -> None:
        self._transport = transport
        self._index = index
        self._response = b""

    def __enter__(self) -> _FakeConnection:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def settimeout(self, _timeout: float) -> None:
        return None

    def connect(self, _path: str) -> None:
        return None

    def getsockopt(self, _level: int, _option: int, _size: int) -> bytes:
        return struct.pack(
            "3i",
            os.getpid(),
            self._transport.peer_uid,
            self._transport.peer_gid,
        )

    def sendall(self, framed: bytes) -> None:
        assert len(framed) >= _LENGTH.size
        (length,) = _LENGTH.unpack(framed[: _LENGTH.size])
        raw_request = framed[_LENGTH.size :]
        assert len(raw_request) == length
        request = json.loads(raw_request)
        assert _canonical(request) == raw_request
        typed_request = cast(dict[str, object], request)
        self._transport.requests.append(typed_request)
        produced = self._transport.responder(typed_request, self._index)
        reply = produced if isinstance(produced, _WireReply) else _WireReply(_canonical(produced))
        declared = len(reply.payload) if reply.declared_length is None else reply.declared_length
        self._response = _LENGTH.pack(declared) + reply.payload + reply.trailing

    def recv(self, size: int) -> bytes:
        chunk = self._response[:size]
        self._response = self._response[size:]
        return chunk


class _FakeTransport:
    def __init__(
        self,
        responder: Callable[[dict[str, object], int], dict[str, object] | _WireReply],
        *,
        peer_uid: int | None = None,
        peer_gid: int | None = None,
    ) -> None:
        self.responder = responder
        self.requests: list[dict[str, object]] = []
        self._count = 0
        self.peer_uid = os.geteuid() if peer_uid is None else peer_uid
        self.peer_gid = os.getegid() if peer_gid is None else peer_gid

    def socket(self, *_args: object, **_kwargs: object) -> _FakeConnection:
        connection = _FakeConnection(self, self._count)
        self._count += 1
        return connection


def _install_transport(
    monkeypatch: pytest.MonkeyPatch,
    responder: Callable[[dict[str, object], int], dict[str, object] | _WireReply],
) -> _FakeTransport:
    transport = _FakeTransport(responder)
    monkeypatch.setattr(broker_module.socket, "socket", transport.socket)
    socket_info = SimpleNamespace(st_dev=1, st_ino=2)
    monkeypatch.setattr(
        broker_module,
        "_validate_socket_path",
        lambda _path, **_kwargs: socket_info,
    )
    return transport


def _client(
    socket_path: Path,
    *,
    nonce_factory: Callable[[], str] | None = None,
    max_message_bytes: int = 65_536,
) -> UnixCredentialBrokerClient:
    return UnixCredentialBrokerClient(
        _config(socket_path),
        timeout_seconds=0.5,
        max_message_bytes=max_message_bytes,
        clock=lambda: float(_CLOCK),
        nonce_factory=nonce_factory,
        controller_workspace=socket_path.parent,
        handle_ledger_path=(
            socket_path.parent / "credential-broker-state" / "handle-bindings.jsonl"
        ),
    )


def test_describe_inject_and_revoke_are_public_exact_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    binding = _binding()
    issued: dict[str, CredentialDescriptor] = {}

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        payload = cast(dict[str, object], request["payload"])
        operation = request["operation"]
        if operation == "describe":
            assert set(payload) == {
                "binding",
                "allowed_credential_names",
                "expires_at_epoch_seconds",
            }
            assert payload["allowed_credential_names"] == ["OPENAI_API_KEY"]
            descriptor = _descriptor(binding, expiry=cast(int, payload["expires_at_epoch_seconds"]))
            issued["descriptor"] = descriptor
            return _response(
                request,
                {"credential_descriptor": descriptor.to_public_dict()},
            )
        descriptor = issued["descriptor"]
        if operation == "inject":
            assert payload["workspace"] == workspace.as_posix()
            provision = materialize_credential_provision(
                workspace=workspace,
                descriptor=descriptor,
                credential_values={"OPENAI_API_KEY": _SECRET},
            )
            assert provision.bundle_path is not None
            return _response(
                request,
                {
                    "credential_descriptor": descriptor.to_public_dict(),
                    "bundle_path": provision.bundle_path.as_posix(),
                },
            )
        assert operation == "revoke"
        assert payload["cause"] == CredentialRevocationCause.TERMINAL.value
        return _response(
            request,
            {
                "credential_descriptor": descriptor.to_public_dict(),
                "cause": CredentialRevocationCause.TERMINAL.value,
                "revoked": True,
            },
        )

    transport = _install_transport(monkeypatch, respond)
    client = _client((tmp_path / "broker.sock").resolve())
    descriptor = client.describe(binding)
    public = json.dumps(descriptor.to_public_dict(), sort_keys=True)
    assert _SECRET not in public
    provision = client.inject_after_admission(
        workspace=workspace,
        descriptor=descriptor,
    )
    assert provision.bundle_path is not None and provision.bundle_path.is_file()
    receipt = client.revoke(
        workspace=workspace,
        expected_binding=binding,
        descriptor=descriptor,
        cause=CredentialRevocationCause.TERMINAL,
    )

    wire = json.dumps(transport.requests, sort_keys=True)
    assert _SECRET not in wire
    assert [request["operation"] for request in transport.requests] == [
        "describe",
        "inject",
        "revoke",
    ]
    assert provision.bundle_path is not None and not provision.bundle_path.exists()
    assert receipt.receipt_path.is_file()
    assert receipt.cause is CredentialRevocationCause.TERMINAL


def test_preflight_connects_without_sending_protocol_or_secret_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _install_transport(
        monkeypatch,
        lambda _request, _index: pytest.fail("preflight must not send a request"),
    )

    _client((tmp_path / "broker.sock").resolve()).preflight()

    assert transport.requests == []


def test_preflight_rejects_socket_replacement_during_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeTransport(
        lambda _request, _index: pytest.fail("preflight must not send a request")
    )
    monkeypatch.setattr(broker_module.socket, "socket", transport.socket)
    socket_infos = iter(
        (
            SimpleNamespace(st_dev=1, st_ino=2),
            SimpleNamespace(st_dev=1, st_ino=3),
        )
    )
    monkeypatch.setattr(
        broker_module,
        "_validate_socket_path",
        lambda _path, **_kwargs: next(socket_infos),
    )

    with pytest.raises(CredentialBrokerClientError, match="changed during connection"):
        _client((tmp_path / "broker.sock").resolve()).preflight()


def test_preflight_rejects_foreign_peer_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = _FakeTransport(
        lambda _request, _index: pytest.fail("preflight must not send a request"),
        peer_uid=os.geteuid() + 1,
    )
    monkeypatch.setattr(broker_module.socket, "socket", transport.socket)
    socket_info = SimpleNamespace(st_dev=1, st_ino=2)
    monkeypatch.setattr(
        broker_module,
        "_validate_socket_path",
        lambda _path, **_kwargs: socket_info,
    )

    with pytest.raises(CredentialBrokerClientError, match="peer identity mismatch"):
        _client((tmp_path / "broker.sock").resolve()).preflight()


@pytest.mark.parametrize("mode", (0o666, 0o700, 0o644))
def test_socket_validation_rejects_modes_other_than_0600_or_0660(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
) -> None:
    socket_info = SimpleNamespace(
        st_mode=stat.S_IFSOCK | mode,
        st_uid=os.geteuid(),
        st_gid=os.getegid(),
    )
    monkeypatch.setattr(Path, "lstat", lambda _path: socket_info)

    with pytest.raises(CredentialBrokerClientError, match="exactly 0600 or 0660"):
        broker_module._validate_socket_path(
            tmp_path / "broker.sock",
            expected_owner_uid=os.geteuid(),
            expected_group_gid=os.getegid(),
        )


@pytest.mark.parametrize(
    "uid,gid,match",
    (
        (os.geteuid() + 1, os.getegid(), "owner"),
        (os.geteuid(), os.getegid() + 1, "group"),
    ),
)
def test_socket_validation_rejects_foreign_owner_or_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    uid: int,
    gid: int,
    match: str,
) -> None:
    socket_info = SimpleNamespace(
        st_mode=stat.S_IFSOCK | 0o660,
        st_uid=uid,
        st_gid=gid,
    )
    monkeypatch.setattr(Path, "lstat", lambda _path: socket_info)

    with pytest.raises(CredentialBrokerClientError, match=match):
        broker_module._validate_socket_path(
            tmp_path / "broker.sock",
            expected_owner_uid=os.geteuid(),
            expected_group_gid=os.getegid(),
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("binding", "binding mismatch"),
        ("broker", "broker identity mismatch"),
        ("names", "names mismatch"),
        ("expiry", "expiry mismatch"),
    ),
)
def test_describe_rejects_wrong_public_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    binding = _binding()

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        payload = cast(dict[str, object], request["payload"])
        target = _binding(attempt_id="other") if mutation == "binding" else binding
        expiry = cast(int, payload["expires_at_epoch_seconds"])
        if mutation == "names":
            descriptor = CredentialDescriptor.no_credentials(target)
        else:
            descriptor = _descriptor(
                target,
                broker_id=("other-broker" if mutation == "broker" else "site-codex-broker"),
                expiry=expiry + (1 if mutation == "expiry" else 0),
            )
        return _response(
            request,
            {"credential_descriptor": descriptor.to_public_dict()},
        )

    _install_transport(monkeypatch, respond)
    with pytest.raises(CredentialBrokerClientError, match=match):
        _client((tmp_path / "broker.sock").resolve()).describe(binding)


@pytest.mark.parametrize(
    "wire_kind,match",
    (
        ("extra-key", "keys invalid"),
        ("wrong-request", "replay/request mismatch"),
        ("wrong-broker", "response identity mismatch"),
        ("noncanonical", "not canonically encoded"),
        ("duplicate", "not strict UTF-8 JSON"),
        ("truncated", "truncated"),
        ("oversized", "outside the message limit"),
        ("trailing", "bytes after"),
    ),
)
def test_transport_rejects_malformed_replayed_or_unbounded_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wire_kind: str,
    match: str,
) -> None:
    binding = _binding()

    def respond(request: dict[str, object], _index: int) -> dict[str, object] | _WireReply:
        payload = cast(dict[str, object], request["payload"])
        descriptor = _descriptor(
            binding,
            expiry=cast(int, payload["expires_at_epoch_seconds"]),
        )
        value = _response(
            request,
            {"credential_descriptor": descriptor.to_public_dict()},
        )
        if wire_kind == "extra-key":
            value["unexpected"] = True
            return value
        if wire_kind == "wrong-request":
            value["request_id"] = "f" * 32
            return value
        if wire_kind == "wrong-broker":
            value["broker_id"] = "other-broker"
            return value
        if wire_kind == "noncanonical":
            return _WireReply(json.dumps(value, sort_keys=True).encode("utf-8"))
        canonical = _canonical(value)
        if wire_kind == "duplicate":
            duplicate = canonical.replace(
                b'"status":"ok"',
                b'"status":"ok","status":"ok"',
                1,
            )
            return _WireReply(duplicate)
        if wire_kind == "truncated":
            return _WireReply(canonical, declared_length=len(canonical) + 1)
        if wire_kind == "oversized":
            return _WireReply(b"x", declared_length=65_537)
        assert wire_kind == "trailing"
        return _WireReply(canonical, trailing=b"x")

    _install_transport(monkeypatch, respond)
    with pytest.raises(CredentialBrokerClientError, match=match):
        _client((tmp_path / "broker.sock").resolve()).describe(binding)


@pytest.mark.parametrize("invalid", ["wrong-path", "symlink", "permissions"])
def test_inject_rejects_wrong_path_symlink_and_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    descriptor = _descriptor(_binding())

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        provision = materialize_credential_provision(
            workspace=workspace,
            descriptor=descriptor,
            credential_values={"OPENAI_API_KEY": _SECRET},
        )
        assert provision.bundle_path is not None
        response_path = provision.bundle_path
        if invalid == "wrong-path":
            response_path = workspace / "other.json"
        elif invalid == "symlink":
            target = workspace / "target.json"
            target.write_text("opaque", encoding="utf-8")
            target.chmod(0o600)
            provision.bundle_path.unlink()
            provision.bundle_path.symlink_to(target)
        else:
            provision.bundle_path.chmod(0o644)
        return _response(
            request,
            {
                "credential_descriptor": descriptor.to_public_dict(),
                "bundle_path": response_path.as_posix(),
            },
        )

    _install_transport(monkeypatch, respond)
    with pytest.raises(CredentialBrokerClientError):
        _client((tmp_path / "broker.sock").resolve()).inject_after_admission(
            workspace=workspace,
            descriptor=descriptor,
        )


def test_revoke_requires_exact_ack_before_local_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    binding = _binding()
    descriptor = _descriptor(binding)
    provision = materialize_credential_provision(
        workspace=workspace,
        descriptor=descriptor,
        credential_values={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        return _response(
            request,
            {
                "credential_descriptor": descriptor.to_public_dict(),
                "cause": CredentialRevocationCause.TERMINAL.value,
                "revoked": False,
            },
        )

    _install_transport(monkeypatch, respond)
    with pytest.raises(CredentialBrokerClientError, match="did not acknowledge"):
        _client((tmp_path / "broker.sock").resolve()).revoke(
            workspace=workspace,
            expected_binding=binding,
            descriptor=descriptor,
            cause=CredentialRevocationCause.TERMINAL,
        )
    assert provision.bundle_path.is_file()


def test_revoke_never_reads_acknowledged_credential_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    binding = _binding()
    descriptor = _descriptor(binding)
    provision = materialize_credential_provision(
        workspace=workspace,
        descriptor=descriptor,
        credential_values={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None
    bundle_path = provision.bundle_path

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        return _response(
            request,
            {
                "credential_descriptor": descriptor.to_public_dict(),
                "cause": CredentialRevocationCause.TERMINAL.value,
                "revoked": True,
            },
        )

    _install_transport(monkeypatch, respond)
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == bundle_path:
            raise AssertionError("controller attempted to read credential bundle bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    receipt = _client((tmp_path / "broker.sock").resolve()).revoke(
        workspace=workspace,
        expected_binding=binding,
        descriptor=descriptor,
        cause=CredentialRevocationCause.TERMINAL,
    )

    assert not bundle_path.exists()
    assert receipt.receipt_path.is_file()


def test_revoke_accepts_broker_owned_cleanup_and_remains_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    binding = _binding()
    descriptor = _descriptor(binding)
    provision = materialize_credential_provision(
        workspace=workspace,
        descriptor=descriptor,
        credential_values={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None

    def respond(request: dict[str, object], index: int) -> dict[str, object]:
        if index == 0:
            provision.bundle_path.unlink()
        return _response(
            request,
            {
                "credential_descriptor": descriptor.to_public_dict(),
                "cause": CredentialRevocationCause.TERMINAL.value,
                "revoked": True,
            },
        )

    _install_transport(monkeypatch, respond)
    client = _client((tmp_path / "broker.sock").resolve())
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


def test_revoke_rejects_symlink_substitution_after_broker_ack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = (tmp_path / "workspace").resolve()
    workspace.mkdir()
    binding = _binding()
    descriptor = _descriptor(binding)
    provision = materialize_credential_provision(
        workspace=workspace,
        descriptor=descriptor,
        credential_values={"OPENAI_API_KEY": _SECRET},
    )
    assert provision.bundle_path is not None
    target = workspace / "foreign.json"
    target.write_text("do-not-delete", encoding="utf-8")
    provision.bundle_path.unlink()
    provision.bundle_path.symlink_to(target)

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        return _response(
            request,
            {
                "credential_descriptor": descriptor.to_public_dict(),
                "cause": CredentialRevocationCause.TERMINAL.value,
                "revoked": True,
            },
        )

    _install_transport(monkeypatch, respond)
    with pytest.raises(CredentialError, match="credential bundle"):
        _client((tmp_path / "broker.sock").resolve()).revoke(
            workspace=workspace,
            expected_binding=binding,
            descriptor=descriptor,
            cause=CredentialRevocationCause.TERMINAL,
        )

    assert target.read_text(encoding="utf-8") == "do-not-delete"


def test_unavailable_backend_mismatch_and_nonce_replay_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = _client(tmp_path / "missing.sock")
    with pytest.raises(CredentialBrokerClientError, match="unavailable"):
        unavailable.describe(_binding())

    mismatch = UnixCredentialBrokerClient(
        UnixCredentialBrokerConfig(
            socket_path=(tmp_path / "missing.sock").resolve(),
            broker_id="site-claude-broker",
            allowed_credential_names=("ANTHROPIC_API_KEY",),
            attempt_ttl_seconds=60,
            expected_socket_owner_uid=os.geteuid(),
            expected_socket_group_gid=os.getegid(),
        ),
        controller_workspace=tmp_path,
        handle_ledger_path=(tmp_path / "credential-broker-state" / "handle-bindings.jsonl"),
    )
    with pytest.raises(CredentialBrokerClientError, match="inconsistent"):
        mismatch.describe(_binding())

    binding = _binding()

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        payload = cast(dict[str, object], request["payload"])
        descriptor = _descriptor(
            binding,
            expiry=cast(int, payload["expires_at_epoch_seconds"]),
        )
        return _response(
            request,
            {"credential_descriptor": descriptor.to_public_dict()},
        )

    _install_transport(monkeypatch, respond)
    client = _client(
        (tmp_path / "broker.sock").resolve(),
        nonce_factory=lambda: "0" * 32,
    )
    assert client.describe(binding).binding == binding
    with pytest.raises(CredentialBrokerClientError, match="nonce replay"):
        client.describe(binding)


def test_describe_rejects_handle_reuse_across_attempt_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = (_binding(attempt_id="attempt-1"), _binding(attempt_id="attempt-2"))

    def respond(request: dict[str, object], index: int) -> dict[str, object]:
        payload = cast(dict[str, object], request["payload"])
        descriptor = CredentialDescriptor.create(
            bindings[index],
            CredentialState.BUNDLE,
            ("OPENAI_API_KEY",),
            handle=CredentialHandle(
                "site-codex-broker",
                "reused-handle",
                cast(int, payload["expires_at_epoch_seconds"]),
            ),
        )
        return _response(
            request,
            {"credential_descriptor": descriptor.to_public_dict()},
        )

    _install_transport(monkeypatch, respond)
    client = _client((tmp_path / "broker.sock").resolve())
    assert client.describe(bindings[0]).binding == bindings[0]
    with pytest.raises(CredentialBrokerClientError, match="across attempt bindings"):
        client.describe(bindings[1])


def test_handle_reuse_defense_survives_client_restart_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = {
        "attempt-1": _binding(attempt_id="attempt-1"),
        "attempt-2": _binding(attempt_id="attempt-2"),
    }

    def respond(request: dict[str, object], _index: int) -> dict[str, object]:
        payload = cast(dict[str, object], request["payload"])
        raw_binding = cast(dict[str, object], payload["binding"])
        binding = bindings[cast(str, raw_binding["attempt_id"])]
        descriptor = CredentialDescriptor.create(
            binding,
            CredentialState.BUNDLE,
            ("OPENAI_API_KEY",),
            handle=CredentialHandle(
                "site-codex-broker",
                "restart-reused-handle",
                cast(int, payload["expires_at_epoch_seconds"]),
            ),
        )
        return _response(
            request,
            {"credential_descriptor": descriptor.to_public_dict()},
        )

    _install_transport(monkeypatch, respond)
    socket_path = (tmp_path / "broker.sock").resolve()
    assert _client(socket_path).describe(bindings["attempt-1"]).binding == bindings["attempt-1"]
    assert _client(socket_path).describe(bindings["attempt-1"]).binding == bindings["attempt-1"]
    with pytest.raises(CredentialBrokerClientError, match="across attempt bindings"):
        _client(socket_path).describe(bindings["attempt-2"])

    ledger = tmp_path / "credential-broker-state" / "handle-bindings.jsonl"
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_config_rejects_relative_socket_unsorted_names_and_unbounded_ttl(
    tmp_path: Path,
) -> None:
    with pytest.raises(CredentialBrokerClientError, match="absolute"):
        UnixCredentialBrokerConfig(
            Path("broker.sock"),
            "site-broker",
            ("OPENAI_API_KEY",),
            60,
            os.geteuid(),
            os.getegid(),
        )
    with pytest.raises(CredentialBrokerClientError, match="sorted and unique"):
        UnixCredentialBrokerConfig(
            (tmp_path / "broker.sock").resolve(),
            "site-broker",
            ("OPENAI_API_KEY", "OPENAI_API_KEY"),
            60,
            os.geteuid(),
            os.getegid(),
        )
    with pytest.raises(CredentialBrokerClientError, match="attempt TTL"):
        UnixCredentialBrokerConfig(
            (tmp_path / "broker.sock").resolve(),
            "site-broker",
            ("OPENAI_API_KEY",),
            86_401,
            os.geteuid(),
            os.getegid(),
        )


def test_request_and_public_error_response_are_bounded_and_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_transport(
        monkeypatch,
        lambda request, _index: _response(
            request,
            {"code": "denied"},
            status="error",
        ),
    )
    with pytest.raises(CredentialBrokerClientError, match="broker rejected request: denied"):
        _client((tmp_path / "broker.sock").resolve()).describe(_binding())

    _install_transport(monkeypatch, lambda _request, _index: {})
    with pytest.raises(CredentialBrokerClientError, match="request exceeds"):
        _client(
            (tmp_path / "broker.sock").resolve(),
            max_message_bytes=32,
        ).describe(_binding())
