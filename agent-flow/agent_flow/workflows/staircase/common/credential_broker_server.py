# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-user Unix-socket credential broker for Staircase Codex workers.

The broker is intended to run as a sidecar in the controller allocation.  It
reads one private Codex OAuth file and materializes it only into the exact
attempt bundle admitted by the controller.  Neither protocol responses nor
exceptions contain credential material.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import select
import signal
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import cast

from .credential_broker_client import (
    DEFAULT_MAX_MESSAGE_BYTES,
    MAX_ATTEMPT_TTL_SECONDS,
    MAX_MESSAGE_BYTES,
    PROTOCOL_SCHEMA_VERSION,
)
from .credentials import (
    CODEX_AUTH_JSON_CREDENTIAL,
    MAX_CODEX_AUTH_JSON_BYTES,
    CredentialBinding,
    CredentialDescriptor,
    CredentialError,
    CredentialHandle,
    CredentialRevocationCause,
    CredentialState,
    descriptor_from_public_dict,
    materialize_credential_provision,
)

CODEX_AUTH_JSON = CODEX_AUTH_JSON_CREDENTIAL
DEFAULT_CONNECTION_TIMEOUT_SECONDS = 2.0
DEFAULT_ACCEPT_TIMEOUT_SECONDS = 0.25
MAX_CONNECTION_TIMEOUT_SECONDS = 30.0
MAX_ISSUED_HANDLES = 4_096
MAX_SOURCE_AUTH_BYTES = MAX_CODEX_AUTH_JSON_BYTES

_LENGTH = struct.Struct("!I")
_PEER_CREDENTIALS = struct.Struct("3i")
_SAFE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_REQUEST_ID = re.compile(r"[0-9a-f]{32}\Z")


class CredentialBrokerServerError(CredentialError):
    """Raised when the trusted broker cannot safely serve a request."""


@dataclass(frozen=True, slots=True)
class CredentialBrokerServerConfig:
    """Private sidecar configuration for one same-user broker endpoint."""

    socket_path: Path
    broker_id: str
    allowed_credential_names: tuple[str, ...]
    source_auth_json_path: Path
    workspace_roots: tuple[Path, ...]

    def __post_init__(self) -> None:
        _validate_socket_configuration(self.socket_path)
        if not isinstance(self.broker_id, str) or _SAFE_ID.fullmatch(self.broker_id) is None:
            raise CredentialBrokerServerError("broker_id must be a safe public identifier")
        if self.allowed_credential_names != (CODEX_AUTH_JSON,):
            raise CredentialBrokerServerError(
                "Codex OAuth broker must allow exactly CODEX_AUTH_JSON"
            )
        _validate_private_source(self.source_auth_json_path)
        if not isinstance(self.workspace_roots, tuple) or not self.workspace_roots:
            raise CredentialBrokerServerError("workspace root allowlist must be non-empty")
        if tuple(sorted(set(self.workspace_roots))) != self.workspace_roots:
            raise CredentialBrokerServerError("workspace roots must be sorted and unique")
        for root in self.workspace_roots:
            _canonical_directory(root, "workspace root")


@dataclass(slots=True)
class _IssuedHandle:
    descriptor: CredentialDescriptor
    workspace: Path | None = None
    revocation_cause: CredentialRevocationCause | None = None


class _RequestRejected(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class UnixCredentialBrokerServer:
    """Bounded, sequential same-UID implementation of protocol schema v1."""

    def __init__(
        self,
        config: CredentialBrokerServerConfig,
        *,
        connection_timeout_seconds: float = DEFAULT_CONNECTION_TIMEOUT_SECONDS,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        max_issued_handles: int = MAX_ISSUED_HANDLES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(config, CredentialBrokerServerConfig):
            raise CredentialBrokerServerError("credential broker server requires typed config")
        if (
            isinstance(connection_timeout_seconds, bool)
            or not isinstance(connection_timeout_seconds, (int, float))
            or not 0 < float(connection_timeout_seconds) <= MAX_CONNECTION_TIMEOUT_SECONDS
        ):
            raise CredentialBrokerServerError("connection timeout is outside the safe range")
        if (
            isinstance(max_message_bytes, bool)
            or not isinstance(max_message_bytes, int)
            or not 1 <= max_message_bytes <= MAX_MESSAGE_BYTES
        ):
            raise CredentialBrokerServerError("message limit is outside the safe range")
        if (
            isinstance(max_issued_handles, bool)
            or not isinstance(max_issued_handles, int)
            or not 1 <= max_issued_handles <= MAX_ISSUED_HANDLES
        ):
            raise CredentialBrokerServerError("issued handle limit is outside the safe range")
        if not callable(clock):
            raise CredentialBrokerServerError("broker clock must be callable")
        self._config = config
        self._connection_timeout_seconds = float(connection_timeout_seconds)
        self._max_message_bytes = max_message_bytes
        self._max_issued_handles = max_issued_handles
        self._clock = clock
        self._issued: dict[str, _IssuedHandle] = {}
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def config(self) -> CredentialBrokerServerConfig:
        """Return the immutable server configuration."""
        return self._config

    def start_in_background(self) -> threading.Thread:
        """Bind synchronously, then serve in one daemon sidecar thread."""
        self._open_listener()
        with self._lock:
            if self._thread is not None:
                raise CredentialBrokerServerError("credential broker is already running")
            thread = threading.Thread(
                target=self.serve_forever,
                kwargs={"install_signal_handlers": False},
                name=f"credential-broker-{self._config.broker_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return thread

    def serve_forever(self, *, install_signal_handlers: bool = True) -> None:
        """Serve until shutdown, optionally handling SIGINT and SIGTERM."""
        self._open_listener()
        previous_handlers: dict[int, signal.Handlers] = {}
        if install_signal_handlers:
            if threading.current_thread() is not threading.main_thread():
                raise CredentialBrokerServerError(
                    "signal handlers can only be installed by the main thread"
                )
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle_signal)
        try:
            self._accept_connections()
        finally:
            if install_signal_handlers:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
            self._close_listener()

    def shutdown(self) -> None:
        """Stop accepting requests and remove only this server's socket node."""
        self._stop.set()
        with self._lock:
            listener = self._listener
        if listener is not None:
            try:
                listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                listener.close()
            except OSError:
                pass

    def wait(self, timeout: float | None = None) -> None:
        """Wait for a background sidecar thread to finish."""
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    @contextmanager
    def running(self) -> Iterator[UnixCredentialBrokerServer]:
        """Run a background sidecar for the duration of a context."""
        self.start_in_background()
        try:
            yield self
        finally:
            self.shutdown()
            self.wait(DEFAULT_CONNECTION_TIMEOUT_SECONDS)

    def _open_listener(self) -> None:
        with self._lock:
            if self._listener is not None:
                return
            if self._stop.is_set():
                raise CredentialBrokerServerError("stopped credential broker cannot restart")
            _validate_socket_configuration(self._config.socket_path)
            if self._config.socket_path.exists() or self._config.socket_path.is_symlink():
                raise CredentialBrokerServerError("broker socket path already exists")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                listener.bind(os.fspath(self._config.socket_path))
                os.chmod(self._config.socket_path, 0o600, follow_symlinks=False)
                info = self._config.socket_path.lstat()
                _validate_bound_socket(info)
                listener.listen(16)
                listener.settimeout(DEFAULT_ACCEPT_TIMEOUT_SECONDS)
            except Exception:
                listener.close()
                _unlink_socket_if_owned(self._config.socket_path, None)
                raise
            self._socket_identity = (info.st_dev, info.st_ino)
            self._listener = listener

    def _close_listener(self) -> None:
        with self._lock:
            listener = self._listener
            self._listener = None
            identity = self._socket_identity
            self._socket_identity = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        _unlink_socket_if_owned(self._config.socket_path, identity)

    def _accept_connections(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                listener = self._listener
            if listener is None:
                return
            try:
                connection, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            with connection:
                connection.settimeout(self._connection_timeout_seconds)
                try:
                    self._serve_connection(connection)
                except (CredentialBrokerServerError, OSError, TimeoutError):
                    continue

    def _serve_connection(self, connection: socket.socket) -> None:
        _validate_same_uid_peer(connection)
        header = _receive_exact(connection, _LENGTH.size, allow_empty=True)
        if not header:
            return
        (length,) = _LENGTH.unpack(header)
        if length < 1 or length > self._max_message_bytes:
            raise CredentialBrokerServerError("request length is outside the message limit")
        raw = _receive_exact(connection, length)
        _reject_trailing_bytes(connection)
        request = _parse_canonical_json(raw, "broker request")
        response = self._respond(request)
        encoded = _canonical_json(response)
        if len(encoded) > self._max_message_bytes:
            raise CredentialBrokerServerError("response exceeds the message limit")
        connection.sendall(_LENGTH.pack(len(encoded)) + encoded)

    def _respond(self, request: dict[str, object]) -> dict[str, object]:
        _exact_keys(
            request,
            {"schema_version", "operation", "request_id", "broker_id", "payload"},
            "broker request",
        )
        if request["schema_version"] != PROTOCOL_SCHEMA_VERSION:
            raise CredentialBrokerServerError("request schema version mismatch")
        operation = request["operation"]
        request_id = request["request_id"]
        broker_id = request["broker_id"]
        if operation not in {"describe", "inject", "revoke"}:
            raise CredentialBrokerServerError("unsupported broker operation")
        if not isinstance(request_id, str) or _REQUEST_ID.fullmatch(request_id) is None:
            raise CredentialBrokerServerError("request_id must be 32 lowercase hex characters")
        if broker_id != self._config.broker_id:
            raise CredentialBrokerServerError("request broker identity mismatch")
        payload = _mapping(request["payload"], "broker request payload")
        try:
            response_payload = self._dispatch(cast(str, operation), payload)
        except _RequestRejected as error:
            status = "error"
            response_payload = {"code": error.code}
        else:
            status = "ok"
        return {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "operation": operation,
            "request_id": request_id,
            "broker_id": self._config.broker_id,
            "status": status,
            "payload": response_payload,
        }

    def _dispatch(self, operation: str, payload: dict[str, object]) -> dict[str, object]:
        try:
            if operation == "describe":
                return self._describe(payload)
            if operation == "inject":
                return self._inject(payload)
            return self._revoke(payload)
        except _RequestRejected:
            raise
        except (CredentialError, OSError, TypeError, ValueError):
            raise _RequestRejected("invalid_request") from None

    def _describe(self, payload: dict[str, object]) -> dict[str, object]:
        _exact_keys(
            payload,
            {"binding", "allowed_credential_names", "expires_at_epoch_seconds"},
            "describe payload",
        )
        binding = _parse_binding(payload["binding"])
        if binding.backend_kind != "codex":
            raise _RequestRejected("denied")
        names = payload["allowed_credential_names"]
        if names != list(self._config.allowed_credential_names):
            raise _RequestRejected("denied")
        expires_at = _positive_integer(payload["expires_at_epoch_seconds"], "handle expiry")
        now = int(self._clock())
        if expires_at <= now:
            raise _RequestRejected("expired")
        if expires_at > now + MAX_ATTEMPT_TTL_SECONDS:
            raise _RequestRejected("denied")
        handle_id = _exact_handle_id(
            broker_id=self._config.broker_id,
            binding=binding,
            credential_names=self._config.allowed_credential_names,
            expires_at_epoch_seconds=expires_at,
        )
        descriptor = CredentialDescriptor.create(
            binding,
            CredentialState.BUNDLE,
            self._config.allowed_credential_names,
            handle=CredentialHandle(self._config.broker_id, handle_id, expires_at),
        )
        issued = self._issued.get(handle_id)
        if issued is not None:
            if issued.descriptor != descriptor:
                raise _RequestRejected("conflict")
        else:
            if len(self._issued) >= self._max_issued_handles:
                raise _RequestRejected("unavailable")
            self._issued[handle_id] = _IssuedHandle(descriptor)
        return {"credential_descriptor": descriptor.to_public_dict()}

    def _inject(self, payload: dict[str, object]) -> dict[str, object]:
        _exact_keys(payload, {"workspace", "credential_descriptor"}, "inject payload")
        descriptor = _parse_descriptor(payload["credential_descriptor"])
        issued = self._validate_issued_descriptor(descriptor, permit_expired=False)
        if issued.revocation_cause is not None:
            raise _RequestRejected("denied")
        raw_workspace = payload["workspace"]
        if not isinstance(raw_workspace, str):
            raise _RequestRejected("invalid_request")
        workspace = _canonical_directory(Path(raw_workspace), "workspace")
        if not any(
            root == workspace or root in workspace.parents for root in self._config.workspace_roots
        ):
            raise _RequestRejected("denied")
        if issued.workspace is not None and issued.workspace != workspace:
            raise _RequestRejected("conflict")
        source_json = _read_private_source(self._config.source_auth_json_path)
        provision = materialize_credential_provision(
            workspace=workspace,
            descriptor=descriptor,
            credential_values={CODEX_AUTH_JSON: source_json},
        )
        if provision.bundle_path is None:
            raise _RequestRejected("unavailable")
        issued.workspace = workspace
        return {
            "credential_descriptor": descriptor.to_public_dict(),
            "bundle_path": provision.bundle_path.as_posix(),
        }

    def _revoke(self, payload: dict[str, object]) -> dict[str, object]:
        _exact_keys(
            payload,
            {"expected_binding", "credential_descriptor", "cause"},
            "revoke payload",
        )
        binding = _parse_binding(payload["expected_binding"])
        descriptor = _parse_descriptor(payload["credential_descriptor"])
        if descriptor.binding != binding:
            raise _RequestRejected("conflict")
        issued = self._validate_issued_descriptor(descriptor, permit_expired=True)
        try:
            cause = CredentialRevocationCause(payload["cause"])
        except (TypeError, ValueError):
            raise _RequestRejected("invalid_request") from None
        if issued.revocation_cause is not None and issued.revocation_cause is not cause:
            raise _RequestRejected("conflict")
        issued.revocation_cause = cause
        return {
            "credential_descriptor": descriptor.to_public_dict(),
            "cause": cause.value,
            "revoked": True,
        }

    def _validate_issued_descriptor(
        self,
        descriptor: CredentialDescriptor,
        *,
        permit_expired: bool,
    ) -> _IssuedHandle:
        if descriptor.credential_names != self._config.allowed_credential_names:
            raise _RequestRejected("denied")
        handle = descriptor.handle
        if descriptor.state is not CredentialState.BUNDLE or handle is None:
            raise _RequestRejected("denied")
        if handle.broker_id != self._config.broker_id:
            raise _RequestRejected("denied")
        expected_handle_id = _exact_handle_id(
            broker_id=self._config.broker_id,
            binding=descriptor.binding,
            credential_names=descriptor.credential_names,
            expires_at_epoch_seconds=handle.expires_at_epoch_seconds,
        )
        if handle.handle_id != expected_handle_id:
            raise _RequestRejected("conflict")
        if not permit_expired and handle.expires_at_epoch_seconds <= int(self._clock()):
            raise _RequestRejected("expired")
        issued = self._issued.get(handle.handle_id)
        if issued is None:
            raise _RequestRejected("not_found")
        if issued.descriptor != descriptor:
            raise _RequestRejected("conflict")
        return issued

    def _handle_signal(self, _signum: int, _frame: FrameType | None) -> None:
        self.shutdown()


def _validate_socket_configuration(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or "\x00" in os.fspath(path):
        raise CredentialBrokerServerError("broker socket path must be absolute and NUL-free")
    if len(os.fsencode(path)) > 103:
        raise CredentialBrokerServerError("broker socket path exceeds the Unix-socket limit")
    _validate_private_directory(path.parent, "broker socket parent")


def _validate_bound_socket(info: os.stat_result) -> None:
    if (
        not stat.S_ISSOCK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_uid != os.geteuid()
        or info.st_gid != os.getegid()
    ):
        raise CredentialBrokerServerError("bound broker socket identity is unsafe")


def _validate_private_source(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise CredentialBrokerServerError("source auth JSON must be absolute and symlink-free")
    try:
        canonical = path.resolve(strict=True)
        info = path.lstat()
    except OSError as error:
        raise CredentialBrokerServerError("source auth JSON is unavailable") from error
    if (
        canonical != path
        or not stat.S_ISREG(info.st_mode)
        or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
    ):
        raise CredentialBrokerServerError(
            "source auth JSON must be canonical, private, owned, and singly linked"
        )


def _read_private_source(path: Path) -> str:
    _validate_private_source(path)
    before = path.lstat()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise CredentialBrokerServerError("source auth JSON changed while opening")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(MAX_SOURCE_AUTH_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > MAX_SOURCE_AUTH_BYTES:
        raise CredentialBrokerServerError("source auth JSON exceeds the size limit")
    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CredentialBrokerServerError) as error:
        raise CredentialBrokerServerError("source auth JSON is not strict UTF-8 JSON") from error
    if not isinstance(value, dict) or not value:
        raise CredentialBrokerServerError("source auth JSON must be a non-empty object")
    return decoded


def _validate_private_directory(path: Path, label: str) -> Path:
    canonical = _canonical_directory(path, label)
    info = path.lstat()
    if stat.S_IMODE(info.st_mode) != 0o700 or info.st_uid != os.geteuid():
        raise CredentialBrokerServerError(f"{label} must be owned and mode 0700")
    return canonical


def _canonical_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise CredentialBrokerServerError(f"{label} must be absolute and symlink-free")
    try:
        canonical = path.resolve(strict=True)
        info = path.lstat()
    except OSError as error:
        raise CredentialBrokerServerError(f"{label} is unavailable") from error
    if canonical != path or not stat.S_ISDIR(info.st_mode):
        raise CredentialBrokerServerError(f"{label} must be an existing canonical directory")
    return canonical


def _validate_same_uid_peer(connection: socket.socket) -> None:
    if not hasattr(socket, "SO_PEERCRED"):
        raise CredentialBrokerServerError("same-UID peer credentials are unavailable")
    raw = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEER_CREDENTIALS.size)
    if not isinstance(raw, bytes) or len(raw) != _PEER_CREDENTIALS.size:
        raise CredentialBrokerServerError("peer credentials are malformed")
    peer_pid, peer_uid, _peer_gid = _PEER_CREDENTIALS.unpack(raw)
    if peer_pid < 1 or peer_uid != os.geteuid():
        raise CredentialBrokerServerError("peer UID does not match broker UID")


def _receive_exact(
    connection: socket.socket,
    size: int,
    *,
    allow_empty: bool = False,
) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            if allow_empty and not chunks:
                return b""
            raise CredentialBrokerServerError("broker request is truncated")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _reject_trailing_bytes(connection: socket.socket) -> None:
    readable, _, _ = select.select((connection,), (), (), 0.0)
    if not readable:
        return
    trailing = connection.recv(1, socket.MSG_PEEK)
    if trailing:
        raise CredentialBrokerServerError("broker request has trailing bytes")


def _parse_binding(value: object) -> CredentialBinding:
    data = _mapping(value, "credential binding")
    _exact_keys(
        data,
        {"run_id", "item_id", "attempt_id", "task_digest", "generation", "backend_kind"},
        "credential binding",
    )
    try:
        return CredentialBinding(
            run_id=_string(data["run_id"], "run_id"),
            item_id=_string(data["item_id"], "item_id"),
            attempt_id=_string(data["attempt_id"], "attempt_id"),
            task_digest=_string(data["task_digest"], "task_digest"),
            generation=_positive_integer(data["generation"], "generation"),
            backend_kind=cast(str, data["backend_kind"]),
        )
    except CredentialError as error:
        raise CredentialBrokerServerError("credential binding is invalid") from error


def _parse_descriptor(value: object) -> CredentialDescriptor:
    try:
        return descriptor_from_public_dict(value)
    except (CredentialError, TypeError, ValueError) as error:
        raise CredentialBrokerServerError("credential descriptor is invalid") from error


def _exact_handle_id(
    *,
    broker_id: str,
    binding: CredentialBinding,
    credential_names: Sequence[str],
    expires_at_epoch_seconds: int,
) -> str:
    payload = {
        "broker_id": broker_id,
        "binding": {
            "run_id": binding.run_id,
            "item_id": binding.item_id,
            "attempt_id": binding.attempt_id,
            "task_digest": binding.task_digest,
            "generation": binding.generation,
            "backend_kind": binding.backend_kind,
        },
        "credential_names": list(credential_names),
        "expires_at_epoch_seconds": expires_at_epoch_seconds,
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"attempt-{digest[:40]}"


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
        raise CredentialBrokerServerError("broker message is not canonical JSON") from error


def _parse_canonical_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CredentialBrokerServerError) as error:
        raise CredentialBrokerServerError(f"{label} is not strict UTF-8 JSON") from error
    data = _mapping(value, label)
    if _canonical_json(data) != raw:
        raise CredentialBrokerServerError(f"{label} is not canonically encoded")
    return data


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CredentialBrokerServerError(f"{label} must be a string-keyed object")
    return cast(dict[str, object], value)


def _exact_keys(data: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(data) != expected:
        raise CredentialBrokerServerError(f"{label} keys are invalid")


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CredentialBrokerServerError(f"{label} must be a positive integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CredentialBrokerServerError(f"{label} must be a string")
    return value


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CredentialBrokerServerError(f"duplicate JSON key in {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise CredentialBrokerServerError(f"non-finite JSON constant {value!r}")


def _unlink_socket_if_owned(path: Path, identity: tuple[int, int] | None) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if identity is not None and (info.st_dev, info.st_ino) != identity:
        return
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
        return
    path.unlink()


__all__ = [
    "CODEX_AUTH_JSON",
    "CredentialBrokerServerConfig",
    "CredentialBrokerServerError",
    "UnixCredentialBrokerServer",
]
