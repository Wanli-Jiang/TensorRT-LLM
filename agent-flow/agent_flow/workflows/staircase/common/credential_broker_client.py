# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded Unix-socket client for the trusted Staircase credential broker.

The protocol carries public binding, handle, policy, and receipt metadata only.
Credential values remain inside the broker-created private bundle and are never
read or returned by this transport client.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import secrets
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .credentials import (
    ALL_BACKEND_CREDENTIAL_NAMES,
    BACKEND_CREDENTIAL_ALLOWLIST,
    CREDENTIAL_ROOT_NAME,
    BackendKind,
    CredentialBinding,
    CredentialDescriptor,
    CredentialError,
    CredentialProvision,
    CredentialRevocationCause,
    CredentialRevocationReceipt,
    CredentialState,
    descriptor_from_public_dict,
    revoke_broker_acknowledged_credentials,
)

PROTOCOL_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_MAX_MESSAGE_BYTES = 65_536
MAX_TIMEOUT_SECONDS = 30.0
MAX_MESSAGE_BYTES = 1_048_576
MAX_ATTEMPT_TTL_SECONDS = 86_400
MAX_HANDLE_LEDGER_BYTES = 1_048_576
MAX_HANDLE_LEDGER_ENTRIES = 4_096
HANDLE_LEDGER_SCHEMA_VERSION = 1

_LENGTH = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_SAFE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")
_BROKER_ERROR_CODES = frozenset(
    {"conflict", "denied", "expired", "invalid_request", "not_found", "unavailable"}
)


class CredentialBrokerClientError(CredentialError):
    """Raised when broker transport or public response validation fails."""


@dataclass(frozen=True, slots=True)
class UnixCredentialBrokerConfig:
    """Public, secret-free policy for one trusted Unix-socket broker."""

    socket_path: Path
    broker_id: str
    allowed_credential_names: tuple[str, ...]
    attempt_ttl_seconds: int
    expected_socket_owner_uid: int
    expected_socket_group_gid: int

    def __post_init__(self) -> None:
        path = self.socket_path
        if not isinstance(path, Path) or not path.is_absolute() or "\x00" in str(path):
            raise CredentialBrokerClientError("broker socket path must be absolute and NUL-free")
        if len(os.fsencode(path)) > 103:
            raise CredentialBrokerClientError("broker socket path exceeds the Unix-socket limit")
        if not isinstance(self.broker_id, str) or _SAFE_ID.fullmatch(self.broker_id) is None:
            raise CredentialBrokerClientError("broker_id must be a safe public identifier")
        names = self.allowed_credential_names
        if not isinstance(names, tuple):
            raise CredentialBrokerClientError("allowed credential names must be an immutable tuple")
        if tuple(sorted(set(names))) != names:
            raise CredentialBrokerClientError("allowed credential names must be sorted and unique")
        if not set(names).issubset(ALL_BACKEND_CREDENTIAL_NAMES):
            raise CredentialBrokerClientError("broker policy names an unsupported credential")
        if (
            isinstance(self.attempt_ttl_seconds, bool)
            or not isinstance(self.attempt_ttl_seconds, int)
            or not 1 <= self.attempt_ttl_seconds <= MAX_ATTEMPT_TTL_SECONDS
        ):
            raise CredentialBrokerClientError(
                f"attempt TTL must be an integer in [1, {MAX_ATTEMPT_TTL_SECONDS}]"
            )
        for name in ("expected_socket_owner_uid", "expected_socket_group_gid"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise CredentialBrokerClientError(f"{name} must be a non-negative integer")


class UnixCredentialBrokerClient:
    """Fail-closed implementation of the public :class:`CredentialBroker` API."""

    def __init__(
        self,
        config: UnixCredentialBrokerConfig,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        clock: Callable[[], float] = time.time,
        nonce_factory: Callable[[], str] | None = None,
        controller_workspace: Path,
        handle_ledger_path: Path,
    ) -> None:
        if not isinstance(config, UnixCredentialBrokerConfig):
            raise CredentialBrokerClientError("credential broker client requires typed config")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not 0 < float(timeout_seconds) <= MAX_TIMEOUT_SECONDS
        ):
            raise CredentialBrokerClientError(
                f"broker timeout must be in (0, {MAX_TIMEOUT_SECONDS}] seconds"
            )
        if (
            isinstance(max_message_bytes, bool)
            or not isinstance(max_message_bytes, int)
            or not 1 <= max_message_bytes <= MAX_MESSAGE_BYTES
        ):
            raise CredentialBrokerClientError(
                f"broker message limit must be in [1, {MAX_MESSAGE_BYTES}] bytes"
            )
        if not callable(clock):
            raise CredentialBrokerClientError("broker clock must be callable")
        if nonce_factory is not None and not callable(nonce_factory):
            raise CredentialBrokerClientError("broker nonce factory must be callable")
        canonical_workspace = _canonical_workspace(controller_workspace)
        expected_ledger_path = (
            canonical_workspace / "credential-broker-state" / "handle-bindings.jsonl"
        )
        if handle_ledger_path != expected_ledger_path:
            raise CredentialBrokerClientError(
                "handle ledger path must be the fixed path inside the controller workspace"
            )
        _validate_handle_ledger_path(handle_ledger_path)
        self._config = config
        self._timeout_seconds = float(timeout_seconds)
        self._max_message_bytes = max_message_bytes
        self._clock = clock
        self._nonce_factory = nonce_factory or (lambda: secrets.token_hex(16))
        self._handle_ledger_path = handle_ledger_path
        self._nonce_lock = threading.Lock()
        self._used_request_ids: set[str] = set()

    @property
    def config(self) -> UnixCredentialBrokerConfig:
        """Return the immutable public broker configuration."""
        return self._config

    def preflight(self) -> None:
        """Prove the configured socket is reachable and identity-stable.

        The probe sends no protocol frame, credential name, handle, or value.
        It only validates the socket node before and after connecting so
        controller startup fails before any planning mutation when the trusted
        broker endpoint is missing, weakly permissioned, or replaced.
        """
        before = self._validate_socket_path()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout_seconds)
                connection.connect(os.fspath(self._config.socket_path))
                after = self._validate_socket_path()
                _validate_socket_identity(before, after)
                self._validate_peer_identity(connection)
        except CredentialBrokerClientError:
            raise
        except (OSError, TimeoutError) as error:
            raise CredentialBrokerClientError(
                f"credential broker is unavailable: {type(error).__name__}"
            ) from error

    def describe(self, binding: CredentialBinding) -> CredentialDescriptor:
        """Mint one exact, bounded, attempt-specific public handle."""
        self._validate_binding_policy(binding)
        now = int(self._clock())
        expires_at = now + self._config.attempt_ttl_seconds
        payload = self._exchange(
            "describe",
            {
                "binding": _binding_dict(binding),
                "allowed_credential_names": list(self._config.allowed_credential_names),
                "expires_at_epoch_seconds": expires_at,
            },
        )
        _exact_keys(payload, {"credential_descriptor"}, "describe response payload")
        descriptor = _parse_descriptor(payload["credential_descriptor"])
        self._validate_descriptor(
            descriptor,
            expected_binding=binding,
            expected_expiry=expires_at,
        )
        self._record_handle_binding(descriptor)
        return descriptor

    def inject_after_admission(
        self,
        *,
        workspace: Path,
        descriptor: CredentialDescriptor,
    ) -> CredentialProvision:
        """Ask the broker to materialize, then validate only path metadata."""
        canonical_workspace = _canonical_workspace(workspace)
        self._validate_descriptor(
            descriptor,
            expected_binding=descriptor.binding,
            expected_expiry=None,
        )
        payload = self._exchange(
            "inject",
            {
                "workspace": canonical_workspace.as_posix(),
                "credential_descriptor": descriptor.to_public_dict(),
            },
        )
        _exact_keys(
            payload,
            {"credential_descriptor", "bundle_path"},
            "inject response payload",
        )
        echoed = _parse_descriptor(payload["credential_descriptor"])
        if echoed != descriptor:
            raise CredentialBrokerClientError("inject response descriptor identity mismatch")
        expected_path = _bundle_path(canonical_workspace, descriptor.binding)
        raw_path = payload["bundle_path"]
        if descriptor.state is CredentialState.NONE:
            if raw_path is not None:
                raise CredentialBrokerClientError(
                    "no-credential inject response cannot name a bundle"
                )
            if expected_path.exists() or expected_path.is_symlink():
                raise CredentialBrokerClientError(
                    "no-credential inject unexpectedly materialized a bundle"
                )
            return CredentialProvision(descriptor, None)
        if not isinstance(raw_path, str) or Path(raw_path) != expected_path:
            raise CredentialBrokerClientError(
                "inject response bundle path differs from the exact attempt path"
            )
        _validate_private_bundle(expected_path, canonical_workspace)
        return CredentialProvision(descriptor, expected_path)

    def revoke(
        self,
        *,
        workspace: Path,
        expected_binding: CredentialBinding,
        descriptor: CredentialDescriptor,
        cause: CredentialRevocationCause,
    ) -> CredentialRevocationReceipt:
        """Revoke the exact remote handle, then publish bounded local cleanup proof."""
        canonical_workspace = _canonical_workspace(workspace)
        if not isinstance(cause, CredentialRevocationCause):
            raise CredentialBrokerClientError("revoke requires a typed cause")
        self._validate_descriptor(
            descriptor,
            expected_binding=expected_binding,
            expected_expiry=None,
            permit_expired=True,
        )
        payload = self._exchange(
            "revoke",
            {
                "expected_binding": _binding_dict(expected_binding),
                "credential_descriptor": descriptor.to_public_dict(),
                "cause": cause.value,
            },
        )
        _exact_keys(
            payload,
            {"credential_descriptor", "cause", "revoked"},
            "revoke response payload",
        )
        echoed = _parse_descriptor(payload["credential_descriptor"])
        if echoed != descriptor or payload["cause"] != cause.value:
            raise CredentialBrokerClientError("revoke response identity or cause mismatch")
        if payload["revoked"] is not True:
            raise CredentialBrokerClientError("broker did not acknowledge exact revocation")
        return revoke_broker_acknowledged_credentials(
            workspace=canonical_workspace,
            expected_binding=expected_binding,
            descriptor=descriptor,
            cause=cause,
        )

    def _validate_binding_policy(self, binding: CredentialBinding) -> None:
        if not isinstance(binding, CredentialBinding):
            raise CredentialBrokerClientError("broker request requires an exact binding")
        allowed = BACKEND_CREDENTIAL_ALLOWLIST[binding.backend_kind]
        if not set(self._config.allowed_credential_names).issubset(allowed):
            raise CredentialBrokerClientError(
                "broker credential names are inconsistent with binding backend"
            )

    def _validate_descriptor(
        self,
        descriptor: CredentialDescriptor,
        *,
        expected_binding: CredentialBinding,
        expected_expiry: int | None,
        permit_expired: bool = False,
    ) -> None:
        if not isinstance(descriptor, CredentialDescriptor):
            raise CredentialBrokerClientError("broker returned an untyped descriptor")
        self._validate_binding_policy(expected_binding)
        if descriptor.binding != expected_binding:
            raise CredentialBrokerClientError("credential descriptor binding mismatch")
        if descriptor.credential_names != self._config.allowed_credential_names:
            raise CredentialBrokerClientError("credential descriptor names mismatch")
        if not descriptor.credential_names:
            if descriptor.state is not CredentialState.NONE or descriptor.handle is not None:
                raise CredentialBrokerClientError(
                    "empty broker policy requires an explicit no-credential descriptor"
                )
            return
        if descriptor.state is not CredentialState.BUNDLE or descriptor.handle is None:
            raise CredentialBrokerClientError(
                "non-empty broker policy requires a bundle descriptor"
            )
        handle = descriptor.handle
        if handle.broker_id != self._config.broker_id:
            raise CredentialBrokerClientError("credential descriptor broker identity mismatch")
        if expected_expiry is not None and handle.expires_at_epoch_seconds != expected_expiry:
            raise CredentialBrokerClientError("credential descriptor expiry mismatch")
        now = int(self._clock())
        if not permit_expired and handle.expires_at_epoch_seconds <= now:
            raise CredentialBrokerClientError("credential descriptor has expired")
        if handle.expires_at_epoch_seconds > now + self._config.attempt_ttl_seconds:
            raise CredentialBrokerClientError("credential descriptor exceeds attempt TTL")

    def _exchange(self, operation: str, payload: Mapping[str, object]) -> dict[str, object]:
        request_id = self._next_request_id()
        request = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "operation": operation,
            "request_id": request_id,
            "broker_id": self._config.broker_id,
            "payload": dict(payload),
        }
        encoded = _canonical_json(request)
        if len(encoded) > self._max_message_bytes:
            raise CredentialBrokerClientError("broker request exceeds the message limit")
        response = self._round_trip(encoded)
        data = _parse_canonical_json(response, "broker response")
        _exact_keys(
            data,
            {
                "schema_version",
                "operation",
                "request_id",
                "broker_id",
                "status",
                "payload",
            },
            "broker response",
        )
        if data["schema_version"] != PROTOCOL_SCHEMA_VERSION:
            raise CredentialBrokerClientError("broker response schema version mismatch")
        if data["operation"] != operation:
            raise CredentialBrokerClientError("broker response operation mismatch")
        if data["request_id"] != request_id:
            raise CredentialBrokerClientError("broker response replay/request mismatch")
        if data["broker_id"] != self._config.broker_id:
            raise CredentialBrokerClientError("broker response identity mismatch")
        response_payload = _mapping(data["payload"], "broker response payload")
        if data["status"] == "error":
            _exact_keys(response_payload, {"code"}, "broker error payload")
            code = response_payload["code"]
            if not isinstance(code, str) or code not in _BROKER_ERROR_CODES:
                raise CredentialBrokerClientError("broker returned an invalid error code")
            raise CredentialBrokerClientError(f"broker rejected request: {code}")
        if data["status"] != "ok":
            raise CredentialBrokerClientError("broker response status must be 'ok' or 'error'")
        return response_payload

    def _next_request_id(self) -> str:
        request_id = self._nonce_factory()
        if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
            raise CredentialBrokerClientError(
                "broker nonce factory must return 32 lowercase hex characters"
            )
        with self._nonce_lock:
            if request_id in self._used_request_ids:
                raise CredentialBrokerClientError("broker request nonce replay detected")
            self._used_request_ids.add(request_id)
        return request_id

    def _record_handle_binding(self, descriptor: CredentialDescriptor) -> None:
        handle = descriptor.handle
        if handle is None:
            return
        try:
            _record_durable_handle_binding(
                self._handle_ledger_path,
                broker_id=handle.broker_id,
                handle_id=handle.handle_id,
                binding=descriptor.binding,
            )
        except CredentialBrokerClientError:
            raise
        except OSError as error:
            raise CredentialBrokerClientError(
                f"credential handle ledger is unavailable: {type(error).__name__}"
            ) from error

    def _round_trip(self, encoded: bytes) -> bytes:
        before = self._validate_socket_path()
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout_seconds)
                connection.connect(os.fspath(self._config.socket_path))
                after = self._validate_socket_path()
                _validate_socket_identity(before, after)
                self._validate_peer_identity(connection)
                connection.sendall(_LENGTH.pack(len(encoded)) + encoded)
                header = _receive_exact(connection, _LENGTH.size)
                (length,) = _LENGTH.unpack(header)
                if length < 1 or length > self._max_message_bytes:
                    raise CredentialBrokerClientError(
                        "broker response length is outside the message limit"
                    )
                response = _receive_exact(connection, length)
                trailing = connection.recv(1)
                if trailing:
                    raise CredentialBrokerClientError("broker sent bytes after the framed response")
                return response
        except CredentialBrokerClientError:
            raise
        except (OSError, TimeoutError) as error:
            raise CredentialBrokerClientError(
                f"credential broker is unavailable: {type(error).__name__}"
            ) from error

    def _validate_socket_path(self) -> os.stat_result:
        return _validate_socket_path(
            self._config.socket_path,
            expected_owner_uid=self._config.expected_socket_owner_uid,
            expected_group_gid=self._config.expected_socket_group_gid,
        )

    def _validate_peer_identity(self, connection: socket.socket) -> None:
        _validate_peer_identity(
            connection,
            expected_owner_uid=self._config.expected_socket_owner_uid,
            expected_group_gid=self._config.expected_socket_group_gid,
        )


def _binding_dict(binding: CredentialBinding) -> dict[str, object]:
    return {
        "run_id": binding.run_id,
        "item_id": binding.item_id,
        "attempt_id": binding.attempt_id,
        "task_digest": binding.task_digest,
        "generation": binding.generation,
        "backend_kind": binding.backend_kind,
    }


def _parse_descriptor(value: object) -> CredentialDescriptor:
    try:
        return descriptor_from_public_dict(value)
    except (CredentialError, TypeError, ValueError) as error:
        raise CredentialBrokerClientError(
            f"broker returned an invalid public descriptor: {error}"
        ) from error


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise CredentialBrokerClientError("broker request is not canonical JSON") from error


def _parse_canonical_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CredentialBrokerClientError) as error:
        raise CredentialBrokerClientError(f"{label} is not strict UTF-8 JSON") from error
    data = _mapping(value, label)
    if _canonical_json(data) != raw:
        raise CredentialBrokerClientError(f"{label} is not canonically encoded")
    return data


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CredentialBrokerClientError(f"{label} must be a string-keyed object")
    return cast(dict[str, object], value)


def _exact_keys(data: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(data) != expected:
        raise CredentialBrokerClientError(
            f"{label} keys invalid; missing={sorted(expected - set(data))}, "
            f"unknown={sorted(set(data) - expected)}"
        )


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CredentialBrokerClientError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise CredentialBrokerClientError(f"non-finite JSON constant {value!r}")


def _receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise CredentialBrokerClientError("broker response is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _validate_socket_path(
    path: Path,
    *,
    expected_owner_uid: int,
    expected_group_gid: int,
) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as error:
        raise CredentialBrokerClientError(
            f"credential broker is unavailable: {type(error).__name__}"
        ) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISSOCK(info.st_mode):
        raise CredentialBrokerClientError("broker socket path must be a Unix socket")
    mode = stat.S_IMODE(info.st_mode)
    if mode not in {0o600, 0o660}:
        raise CredentialBrokerClientError("broker socket mode must be exactly 0600 or 0660")
    if info.st_uid != expected_owner_uid:
        raise CredentialBrokerClientError("broker socket owner does not match configured owner")
    if info.st_gid != expected_group_gid:
        raise CredentialBrokerClientError("broker socket group does not match configured group")
    return info


def _validate_socket_identity(before: os.stat_result, after: os.stat_result) -> None:
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise CredentialBrokerClientError("broker socket changed during connection")


def _validate_peer_identity(
    connection: socket.socket,
    *,
    expected_owner_uid: int,
    expected_group_gid: int,
) -> None:
    """Authenticate a connected peer where the platform exposes credentials."""
    if not hasattr(socket, "SO_PEERCRED"):
        return
    try:
        raw = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            _PEER_CREDENTIALS.size,
        )
    except OSError as error:
        raise CredentialBrokerClientError(
            f"credential broker peer identity is unavailable: {type(error).__name__}"
        ) from error
    if not isinstance(raw, bytes) or len(raw) != _PEER_CREDENTIALS.size:
        raise CredentialBrokerClientError("credential broker peer identity is malformed")
    peer_pid, peer_uid, peer_gid = _PEER_CREDENTIALS.unpack(raw)
    if peer_pid < 1 or peer_uid != expected_owner_uid or peer_gid != expected_group_gid:
        raise CredentialBrokerClientError("credential broker peer identity mismatch")


def _validate_handle_ledger_path(path: Path) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or "\x00" in os.fspath(path)
        or path.name in {"", ".", ".."}
        or path.is_symlink()
    ):
        raise CredentialBrokerClientError("handle ledger path must be absolute and symlink-free")
    parent = path.parent
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    except OSError as error:
        raise CredentialBrokerClientError("handle ledger directory is unavailable") from error
    try:
        info = parent.lstat()
        canonical = parent.resolve(strict=True)
    except OSError as error:
        raise CredentialBrokerClientError("handle ledger directory is unavailable") from error
    if (
        canonical != parent
        or stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.geteuid()
    ):
        raise CredentialBrokerClientError(
            "handle ledger directory must be canonical, owned, non-symlink mode 0700"
        )


def _record_durable_handle_binding(
    path: Path,
    *,
    broker_id: str,
    handle_id: str,
    binding: CredentialBinding,
) -> None:
    _validate_handle_ledger_path(path)
    lock_path = path.with_name(f".{path.name}.lock")
    lock_descriptor = _open_private_file(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        records, size = _load_handle_ledger(path)
        identity = (broker_id, handle_id)
        previous = records.get(identity)
        if previous is not None:
            if previous != binding:
                raise CredentialBrokerClientError(
                    "broker replayed one credential handle across attempt bindings"
                )
            return
        if len(records) >= MAX_HANDLE_LEDGER_ENTRIES:
            raise CredentialBrokerClientError("credential handle ledger entry limit exceeded")
        record = {
            "schema_version": HANDLE_LEDGER_SCHEMA_VERSION,
            "broker_id": broker_id,
            "handle_id": handle_id,
            "binding": _binding_dict(binding),
        }
        encoded = _canonical_json(record) + b"\n"
        if size + len(encoded) > MAX_HANDLE_LEDGER_BYTES:
            raise CredentialBrokerClientError("credential handle ledger size limit exceeded")
        ledger_descriptor = _open_private_file(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        )
        try:
            _write_all(ledger_descriptor, encoded)
            os.fsync(ledger_descriptor)
        finally:
            os.close(ledger_descriptor)
        _fsync_directory(path.parent)
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def _load_handle_ledger(
    path: Path,
) -> tuple[dict[tuple[str, str], CredentialBinding], int]:
    try:
        descriptor = _open_private_file(path, os.O_RDONLY, create=False)
    except FileNotFoundError:
        return {}, 0
    try:
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read(MAX_HANDLE_LEDGER_BYTES + 1)
    except OSError as error:
        raise CredentialBrokerClientError("credential handle ledger is unreadable") from error
    if len(raw) > MAX_HANDLE_LEDGER_BYTES:
        raise CredentialBrokerClientError("credential handle ledger size limit exceeded")
    if raw and not raw.endswith(b"\n"):
        raise CredentialBrokerClientError("credential handle ledger has a truncated record")
    records: dict[tuple[str, str], CredentialBinding] = {}
    lines = raw.splitlines()
    if len(lines) > MAX_HANDLE_LEDGER_ENTRIES:
        raise CredentialBrokerClientError("credential handle ledger entry limit exceeded")
    for line in lines:
        data = _parse_canonical_json(line, "credential handle ledger record")
        _exact_keys(
            data,
            {"schema_version", "broker_id", "handle_id", "binding"},
            "credential handle ledger record",
        )
        if data["schema_version"] != HANDLE_LEDGER_SCHEMA_VERSION:
            raise CredentialBrokerClientError("credential handle ledger schema mismatch")
        broker_id = data["broker_id"]
        handle_id = data["handle_id"]
        if (
            not isinstance(broker_id, str)
            or _SAFE_ID.fullmatch(broker_id) is None
            or not isinstance(handle_id, str)
            or _SAFE_ID.fullmatch(handle_id) is None
        ):
            raise CredentialBrokerClientError("credential handle ledger identity is invalid")
        binding = _parse_binding(data["binding"])
        identity = (broker_id, handle_id)
        previous = records.get(identity)
        if previous is not None and previous != binding:
            raise CredentialBrokerClientError(
                "credential handle ledger contains conflicting attempt bindings"
            )
        records[identity] = binding
    return records, len(raw)


def _parse_binding(value: object) -> CredentialBinding:
    data = _mapping(value, "credential handle ledger binding")
    expected = {"run_id", "item_id", "attempt_id", "task_digest", "generation", "backend_kind"}
    _exact_keys(data, expected, "credential handle ledger binding")
    try:
        return CredentialBinding(
            run_id=cast(str, data["run_id"]),
            item_id=cast(str, data["item_id"]),
            attempt_id=cast(str, data["attempt_id"]),
            task_digest=cast(str, data["task_digest"]),
            generation=cast(int, data["generation"]),
            backend_kind=cast(BackendKind, data["backend_kind"]),
        )
    except (CredentialError, TypeError, ValueError) as error:
        raise CredentialBrokerClientError("credential handle ledger binding is invalid") from error


def _open_private_file(path: Path, flags: int, *, create: bool = True) -> int:
    open_flags = flags
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    if create:
        open_flags |= os.O_CREAT
    descriptor = os.open(path, open_flags, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
        ):
            raise CredentialBrokerClientError(
                "handle ledger files must be owned regular mode 0600 files with one link"
            )
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    while view:
        written = os.write(descriptor, view)
        if written < 1:
            raise CredentialBrokerClientError("credential handle ledger append failed")
        view = view[written:]


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_workspace(workspace: Path) -> Path:
    if not isinstance(workspace, Path) or not workspace.is_absolute() or workspace.is_symlink():
        raise CredentialBrokerClientError(
            "credential workspace must be an absolute regular directory"
        )
    try:
        canonical = workspace.resolve(strict=True)
    except OSError as error:
        raise CredentialBrokerClientError("credential workspace is unavailable") from error
    if canonical != workspace or not canonical.is_dir():
        raise CredentialBrokerClientError("credential workspace must already be canonical")
    return canonical


def _bundle_path(workspace: Path, binding: CredentialBinding) -> Path:
    return (
        workspace
        / CREDENTIAL_ROOT_NAME
        / "agent-credentials"
        / binding.run_id
        / binding.item_id
        / f"{binding.attempt_id}.json"
    )


def _validate_private_bundle(path: Path, workspace: Path) -> None:
    root = workspace / CREDENTIAL_ROOT_NAME
    if path.parent == root.parent or root not in path.parents:
        raise CredentialBrokerClientError("credential bundle escaped the private root")
    current = path.parent
    while True:
        try:
            info = current.lstat()
        except OSError as error:
            raise CredentialBrokerClientError(
                "credential bundle directory is unavailable"
            ) from error
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o700
            or info.st_uid != os.geteuid()
        ):
            raise CredentialBrokerClientError(
                "credential bundle directories must be owned, non-symlink mode 0700"
            )
        if current == root:
            break
        current = current.parent
    try:
        before = path.lstat()
    except OSError as error:
        raise CredentialBrokerClientError("credential bundle is unavailable") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
    ):
        raise CredentialBrokerClientError(
            "credential bundle must be owned, regular, non-symlink mode 0600 with one link"
        )
    after = path.lstat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise CredentialBrokerClientError("credential bundle changed during validation")


__all__ = [
    "CredentialBrokerClientError",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_ATTEMPT_TTL_SECONDS",
    "MAX_MESSAGE_BYTES",
    "MAX_TIMEOUT_SECONDS",
    "PROTOCOL_SCHEMA_VERSION",
    "UnixCredentialBrokerClient",
    "UnixCredentialBrokerConfig",
]
