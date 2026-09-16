# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Handle-only credentials for fresh Staircase agent workers.

The deterministic controller handles only a public, attempt-bound descriptor.
A trusted launcher or broker materializes the corresponding private bundle
after the attempt has been admitted and revokes it through the same narrow
interface.  Credential values never enter controller arguments, state,
manifests, evidence, or lifecycle receipts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    BinaryIO,
    Callable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    cast,
    runtime_checkable,
)

from .slurm import Mount

BackendKind = Literal["claude-code", "codex"]

CREDENTIAL_SCHEMA_VERSION = 2
CREDENTIAL_REVOCATION_SCHEMA_VERSION = 2
CREDENTIAL_ROOT_NAME = "controller-secrets"
CREDENTIAL_REVOCATION_ROOT_NAME = "credential-revocation-receipts"
CREDENTIAL_MOUNT_PATH = Path("/run/staircase/agent-credentials.json")
MAX_CREDENTIAL_BUNDLE_BYTES = 65_536
MAX_RUN_CREDENTIAL_REVOCATIONS = 256

_SAFE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SECRET_SCAN_BLOCK_BYTES = 65_536
_SECRET_SCAN_PATTERN_BYTES = 8_192
_SECRET_SHAPED_BYTES = re.compile(
    rb"(?:\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,4096}|"
    rb"(?:sk|nvapi|gh[oprsu])[-_][A-Za-z0-9_-]{8,4096}|"
    rb"-----BEGIN [A-Z ]{0,64}PRIVATE KEY-----|"
    rb"[A-Za-z][A-Za-z0-9+.-]{1,31}://[^/@\s]{1,1024}:[^/@\s]{1,1024}@|"
    rb"\b(?:[A-Za-z0-9_]*(?:AUTH|COOKIE|CREDENTIAL|KEY|PASS(?:WORD)?|SECRET|TOKEN)"
    rb"[A-Za-z0-9_]*)\s*[:=]\s*['\"]?[A-Za-z0-9._~+/=-]{12,4096})",
    re.IGNORECASE,
)

# Expanding this list changes the trusted secret boundary and requires review.
BACKEND_CREDENTIAL_ALLOWLIST: Mapping[BackendKind, frozenset[str]] = {
    "claude-code": frozenset(
        {
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
        }
    ),
    "codex": frozenset({"OPENAI_API_KEY"}),
}
ALL_BACKEND_CREDENTIAL_NAMES = frozenset().union(*BACKEND_CREDENTIAL_ALLOWLIST.values())


class CredentialError(ValueError):
    """Raised when a credential descriptor or private bundle is unsafe."""


class CredentialState(str, Enum):
    """Public statement of whether a worker receives an environment bundle."""

    NONE = "none"
    BUNDLE = "bundle"


class CredentialRevocationCause(str, Enum):
    """Lifecycle boundary that requires an exact-attempt revocation."""

    TERMINAL = "terminal"
    CANCELLED = "cancelled"
    REPLACED = "replaced"
    GENERATION_FENCE = "generation_fence"
    ORPHAN_EXPIRED = "orphan_expired"


@dataclass(frozen=True, slots=True)
class CredentialHandle:
    """Opaque public broker handle; it contains no credential material."""

    broker_id: str
    handle_id: str
    expires_at_epoch_seconds: int

    def __post_init__(self) -> None:
        for name in ("broker_id", "handle_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise CredentialError(f"credential {name} must be a safe identifier")
        if (
            isinstance(self.expires_at_epoch_seconds, bool)
            or not isinstance(self.expires_at_epoch_seconds, int)
            or self.expires_at_epoch_seconds < 1
        ):
            raise CredentialError("credential handle expiry must be a positive integer")

    def to_public_dict(self) -> dict[str, str | int]:
        """Return a strict secret-free serialized handle."""
        return {
            "broker_id": self.broker_id,
            "handle_id": self.handle_id,
            "expires_at_epoch_seconds": self.expires_at_epoch_seconds,
        }


@dataclass(frozen=True, slots=True)
class CredentialBinding:
    """Public immutable identity to which a credential bundle is bound."""

    run_id: str
    item_id: str
    attempt_id: str
    task_digest: str
    generation: int
    backend_kind: BackendKind

    def __post_init__(self) -> None:
        for name in ("run_id", "item_id", "attempt_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
                raise CredentialError(f"{name} must be a safe identifier")
        if not isinstance(self.task_digest, str) or _DIGEST.fullmatch(self.task_digest) is None:
            raise CredentialError("task_digest must be a lowercase SHA-256 digest")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise CredentialError("generation must be a positive integer")
        if self.backend_kind not in BACKEND_CREDENTIAL_ALLOWLIST:
            raise CredentialError(f"unsupported credential backend {self.backend_kind!r}")


@dataclass(frozen=True, slots=True)
class CredentialDescriptor:
    """Secret-free descriptor safe for manifests, state, and reports."""

    binding: CredentialBinding
    state: CredentialState
    credential_names: tuple[str, ...]
    handle: CredentialHandle | None
    descriptor_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.binding, CredentialBinding):
            raise CredentialError("credential descriptor requires a binding")
        if not isinstance(self.state, CredentialState):
            raise CredentialError("credential descriptor requires a typed state")
        if not isinstance(self.credential_names, tuple):
            raise CredentialError("credential_names must be an immutable tuple")
        if tuple(sorted(set(self.credential_names))) != self.credential_names:
            raise CredentialError("credential_names must be sorted and unique")
        allowed = BACKEND_CREDENTIAL_ALLOWLIST[self.binding.backend_kind]
        if not set(self.credential_names).issubset(allowed):
            raise CredentialError("credential descriptor contains a non-allowlisted name")
        if self.state is CredentialState.NONE and self.credential_names:
            raise CredentialError("no-credential descriptor cannot name credentials")
        if self.state is CredentialState.NONE and self.handle is not None:
            raise CredentialError("no-credential descriptor cannot carry a broker handle")
        if self.state is CredentialState.BUNDLE and not self.credential_names:
            raise CredentialError("bundle descriptor must name at least one credential")
        if self.state is CredentialState.BUNDLE and not isinstance(self.handle, CredentialHandle):
            raise CredentialError("bundle descriptor requires an opaque broker handle")
        if not isinstance(self.descriptor_digest, str) or not _DIGEST.fullmatch(
            self.descriptor_digest
        ):
            raise CredentialError("descriptor_digest must be a lowercase SHA-256 digest")
        if self.descriptor_digest != _descriptor_digest(
            self.binding, self.state, self.credential_names, self.handle
        ):
            raise CredentialError("credential descriptor digest mismatch")

    @classmethod
    def create(
        cls,
        binding: CredentialBinding,
        state: CredentialState,
        credential_names: Sequence[str],
        *,
        handle: CredentialHandle | None = None,
    ) -> CredentialDescriptor:
        """Create a canonical descriptor without observing credential values."""
        names = tuple(sorted(credential_names))
        return cls(
            binding,
            state,
            names,
            handle,
            _descriptor_digest(binding, state, names, handle),
        )

    @classmethod
    def no_credentials(cls, binding: CredentialBinding) -> CredentialDescriptor:
        """Create an explicit preauthenticated/no-environment-credential state."""
        return cls.create(binding, CredentialState.NONE, ())

    def to_public_dict(self) -> dict[str, object]:
        """Return the only credential metadata permitted in durable artifacts."""
        return {
            "schema_version": CREDENTIAL_SCHEMA_VERSION,
            "run_id": self.binding.run_id,
            "item_id": self.binding.item_id,
            "attempt_id": self.binding.attempt_id,
            "task_digest": self.binding.task_digest,
            "generation": self.binding.generation,
            "backend_kind": self.binding.backend_kind,
            "state": self.state.value,
            "credential_names": list(self.credential_names),
            "credential_handle": (None if self.handle is None else self.handle.to_public_dict()),
            "descriptor_digest": self.descriptor_digest,
        }


@runtime_checkable
class CredentialBroker(Protocol):
    """Secret-free controller view of a trusted credential launcher/broker."""

    def describe(self, binding: CredentialBinding) -> CredentialDescriptor:
        """Mint or recover one public, exact-attempt broker handle."""

    def inject_after_admission(
        self,
        *,
        workspace: Path,
        descriptor: CredentialDescriptor,
    ) -> CredentialProvision:
        """Materialize the handle without returning credential values."""

    def revoke(
        self,
        *,
        workspace: Path,
        expected_binding: CredentialBinding,
        descriptor: CredentialDescriptor,
        cause: CredentialRevocationCause,
    ) -> CredentialRevocationReceipt:
        """Revoke one exact handle idempotently without returning secrets."""


@dataclass(frozen=True, slots=True)
class NoCredentialBroker:
    """Explicit preauthenticated deployment with no environment credential."""

    def describe(self, binding: CredentialBinding) -> CredentialDescriptor:
        """Return a binding-specific no-credential descriptor."""
        return CredentialDescriptor.no_credentials(binding)

    def inject_after_admission(
        self,
        *,
        workspace: Path,
        descriptor: CredentialDescriptor,
    ) -> CredentialProvision:
        """Validate that admission did not unexpectedly acquire a credential."""
        _canonical_private_workspace(workspace)
        if descriptor.state is not CredentialState.NONE:
            raise CredentialError("no-credential broker cannot inject a bundle")
        return CredentialProvision(descriptor, None)

    def revoke(
        self,
        *,
        workspace: Path,
        expected_binding: CredentialBinding,
        descriptor: CredentialDescriptor,
        cause: CredentialRevocationCause,
    ) -> CredentialRevocationReceipt:
        """Publish the same exact idempotent receipt used by credential brokers."""
        if not isinstance(cause, CredentialRevocationCause):
            raise CredentialError("credential revocation requires a typed cause")
        return revoke_terminal_attempt_credentials(
            workspace=workspace,
            expected_binding=expected_binding,
            provision=CredentialProvision(descriptor, None),
            cause=cause,
        )


@dataclass(frozen=True, slots=True)
class CredentialProvision:
    """Trusted-launcher result: mount source plus secret-free descriptor."""

    descriptor: CredentialDescriptor
    bundle_path: Path | None

    def __post_init__(self) -> None:
        if self.descriptor.state is CredentialState.NONE:
            if self.bundle_path is not None:
                raise CredentialError("no-credential provision cannot have a bundle path")
            return
        if self.bundle_path is None or not self.bundle_path.is_absolute():
            raise CredentialError("bundle provision requires an absolute bundle path")

    @property
    def mounts(self) -> tuple[Mount, ...]:
        """Return zero mounts or exactly one fixed-path read-only secret mount."""
        if self.bundle_path is None:
            return ()
        return (Mount(self.bundle_path, CREDENTIAL_MOUNT_PATH, True),)


@dataclass(frozen=True, slots=True)
class CredentialRevocationReceipt:
    """Secret-free evidence that one exact credential provision was revoked."""

    descriptor: CredentialDescriptor
    cause: CredentialRevocationCause
    receipt_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, CredentialDescriptor):
            raise CredentialError("credential revocation receipt requires a descriptor")
        if not isinstance(self.cause, CredentialRevocationCause):
            raise CredentialError("credential revocation receipt requires a typed cause")
        if not isinstance(self.receipt_path, Path) or not self.receipt_path.is_absolute():
            raise CredentialError("credential revocation receipt requires an absolute path")

    def to_public_dict(self) -> dict[str, object]:
        """Return the immutable receipt content without its controller path."""
        return _revocation_receipt_payload(self.descriptor, self.cause)


def descriptor_from_public_dict(value: object) -> CredentialDescriptor:
    """Strictly parse a secret-free descriptor from a worker manifest."""
    data = _object(value, "credential descriptor")
    expected = {
        "schema_version",
        "run_id",
        "item_id",
        "attempt_id",
        "task_digest",
        "generation",
        "backend_kind",
        "state",
        "credential_names",
        "credential_handle",
        "descriptor_digest",
    }
    _exact_keys(data, expected, "credential descriptor")
    if data["schema_version"] != CREDENTIAL_SCHEMA_VERSION:
        raise CredentialError("unsupported credential descriptor schema")
    names = data["credential_names"]
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise CredentialError("credential_names must be a string list")
    try:
        backend_kind = cast(BackendKind, data["backend_kind"])
        binding = CredentialBinding(
            run_id=_string(data["run_id"], "run_id"),
            item_id=_string(data["item_id"], "item_id"),
            attempt_id=_string(data["attempt_id"], "attempt_id"),
            task_digest=_string(data["task_digest"], "task_digest"),
            generation=cast(int, data["generation"]),
            backend_kind=backend_kind,
        )
        state = CredentialState(data["state"])
        handle = _credential_handle_from_public_dict(data["credential_handle"])
    except (TypeError, ValueError) as error:
        raise CredentialError("credential descriptor has invalid typed fields") from error
    return CredentialDescriptor(
        binding=binding,
        state=state,
        credential_names=tuple(names),
        handle=handle,
        descriptor_digest=_string(data["descriptor_digest"], "descriptor_digest"),
    )


def validate_unique_credential_handles(
    descriptors: Sequence[CredentialDescriptor],
) -> None:
    """Reject reuse of one broker handle across exact attempt bindings."""
    seen: dict[tuple[str, str], CredentialBinding] = {}
    for descriptor in descriptors:
        if not isinstance(descriptor, CredentialDescriptor):
            raise CredentialError("credential handle uniqueness requires descriptors")
        handle = descriptor.handle
        if handle is None:
            continue
        identity = (handle.broker_id, handle.handle_id)
        previous = seen.get(identity)
        if previous is not None and previous != descriptor.binding:
            raise CredentialError("credential broker reused one handle across attempt bindings")
        seen[identity] = descriptor.binding


def materialize_credential_provision(
    *,
    workspace: Path,
    descriptor: CredentialDescriptor,
    credential_values: Mapping[str, str],
) -> CredentialProvision:
    """Materialize one public handle inside the trusted launcher boundary.

    This function is intentionally not a controller API: ``credential_values``
    is secret-bearing.  A production launcher/broker calls it (or an equivalent
    remote implementation) and returns only :class:`CredentialProvision` to
    controller code.
    """
    canonical_workspace = _canonical_private_workspace(workspace)
    if descriptor_from_public_dict(descriptor.to_public_dict()) != descriptor:
        raise CredentialError("credential materialization requires a canonical descriptor")
    bundle_path = _bundle_path(canonical_workspace, descriptor.binding)
    if descriptor.state is CredentialState.NONE:
        if credential_values:
            raise CredentialError("no-credential descriptor cannot materialize values")
        if bundle_path.exists() or bundle_path.is_symlink():
            raise CredentialError("unexpected bundle for no-credential descriptor")
        return CredentialProvision(descriptor, None)
    _ensure_handle_unexpired(cast(CredentialHandle, descriptor.handle))

    credentials = {
        name: _credential_value(value, name) for name, value in credential_values.items()
    }
    if tuple(sorted(credentials)) != descriptor.credential_names:
        raise CredentialError("credential material does not match the public handle names")
    _ensure_private_directory(bundle_path.parent)
    payload = {**descriptor.to_public_dict(), "credentials": credentials}
    try:
        _write_exclusive_bundle(bundle_path, payload)
    except FileExistsError:
        loaded = load_worker_credentials(
            descriptor,
            expected_binding=descriptor.binding,
            bundle_path=bundle_path,
        )
        if loaded != credentials:
            raise CredentialError("existing credential bundle does not match broker material")
    return CredentialProvision(descriptor, bundle_path)


def prepare_credential_provision(
    *,
    workspace: Path,
    binding: CredentialBinding,
    ambient_environment: Mapping[str, str],
) -> CredentialProvision:
    """Legacy trusted-launcher adapter for environment-backed credentials.

    Deterministic controller code must use :class:`CredentialBroker` instead.
    This compatibility adapter remains only for trusted launcher migration and
    tests; it makes its legacy provenance explicit in the public handle.
    """
    canonical_workspace = _canonical_private_workspace(workspace)
    credentials = _select_credentials(binding.backend_kind, ambient_environment)
    bundle_path = _bundle_path(canonical_workspace, binding)
    if not credentials:
        if bundle_path.exists() or bundle_path.is_symlink():
            raise CredentialError("credential availability changed for an existing attempt")
        return CredentialProvision(CredentialDescriptor.no_credentials(binding), None)

    descriptor = CredentialDescriptor.create(
        binding,
        CredentialState.BUNDLE,
        tuple(credentials),
        handle=CredentialHandle(
            "legacy-environment",
            _legacy_handle_id(binding),
            2_147_483_647,
        ),
    )
    try:
        return materialize_credential_provision(
            workspace=canonical_workspace,
            descriptor=descriptor,
            credential_values=credentials,
        )
    except CredentialError as error:
        if "does not match broker material" in str(error):
            raise CredentialError(
                "existing credential bundle does not match controller input"
            ) from None
        raise


def locate_credential_provision(
    *,
    workspace: Path,
    descriptor: CredentialDescriptor,
) -> CredentialProvision:
    """Reconstruct a provision location from durable public metadata only.

    This legacy filesystem-broker helper never creates or adopts a bundle.
    General controller code asks its broker to revoke the public handle and
    does not reconstruct or inspect a private bundle path.
    """
    canonical_workspace = _canonical_private_workspace(workspace)
    if not isinstance(descriptor, CredentialDescriptor):
        raise CredentialError("credential provision locator requires a descriptor")
    if descriptor_from_public_dict(descriptor.to_public_dict()) != descriptor:
        raise CredentialError("credential provision locator requires a canonical descriptor")
    bundle_path = (
        None
        if descriptor.state is CredentialState.NONE
        else _bundle_path(canonical_workspace, descriptor.binding)
    )
    return CredentialProvision(descriptor, bundle_path)


def revoke_terminal_attempt_credentials(
    *,
    workspace: Path,
    expected_binding: CredentialBinding,
    provision: CredentialProvision,
    cause: CredentialRevocationCause = CredentialRevocationCause.TERMINAL,
) -> CredentialRevocationReceipt:
    """Revoke one exact terminal-attempt provision without directory deletion.

    The caller must first reconcile the exact attempt as terminal and ingest its
    result.  A durable, secret-free receipt is published before the validated
    bundle is unlinked, making a completed call safely idempotent.
    """
    canonical_workspace = _canonical_private_workspace(workspace)
    bundle_info = _validate_revocation_target(
        canonical_workspace,
        expected_binding,
        provision,
        cause,
    )
    receipt_path = _revocation_receipt_path(canonical_workspace, expected_binding)

    if provision.descriptor.state is CredentialState.NONE:
        receipt = _ensure_revocation_receipt(receipt_path, provision.descriptor, cause)
        return CredentialRevocationReceipt(receipt, cause, receipt_path)

    if bundle_info is None:
        receipt = _load_revocation_receipt(receipt_path, provision.descriptor, cause)
        if receipt is None:
            raise CredentialError("credential revocation receipt disappeared during validation")
        return CredentialRevocationReceipt(receipt, cause, receipt_path)

    receipt = _ensure_revocation_receipt(receipt_path, provision.descriptor, cause)
    revalidated_info = _validate_revocation_target(
        canonical_workspace,
        expected_binding,
        provision,
        cause,
    )
    if revalidated_info is not None:
        _unlink_exact_bundle(cast(Path, provision.bundle_path), revalidated_info)
    return CredentialRevocationReceipt(receipt, cause, receipt_path)


def revoke_broker_acknowledged_credentials(
    *,
    workspace: Path,
    expected_binding: CredentialBinding,
    descriptor: CredentialDescriptor,
    cause: CredentialRevocationCause,
) -> CredentialRevocationReceipt:
    """Publish local cleanup evidence after an exact broker revocation.

    Unlike the trusted-launcher compatibility helper above, this controller-side
    path never opens or parses credential bundle contents.  The broker's exact
    descriptor acknowledgement is the authority for revocation; local cleanup
    is restricted to validating the canonical path and file metadata before an
    inode-stable unlink.
    """
    canonical_workspace = _canonical_private_workspace(workspace)
    provision = locate_credential_provision(
        workspace=canonical_workspace,
        descriptor=descriptor,
    )
    bundle_info = _validate_revocation_target_metadata(
        canonical_workspace,
        expected_binding,
        provision,
        cause,
    )
    receipt_path = _revocation_receipt_path(canonical_workspace, expected_binding)
    receipt = _ensure_revocation_receipt(receipt_path, descriptor, cause)
    if bundle_info is not None:
        revalidated_info = _validate_revocation_target_metadata(
            canonical_workspace,
            expected_binding,
            provision,
            cause,
        )
        if revalidated_info is not None:
            if (revalidated_info.st_dev, revalidated_info.st_ino) != (
                bundle_info.st_dev,
                bundle_info.st_ino,
            ):
                raise CredentialError("credential bundle changed during broker cleanup")
            _unlink_exact_bundle(cast(Path, provision.bundle_path), bundle_info)
    return CredentialRevocationReceipt(receipt, cause, receipt_path)


def revoke_run_credential_provisions(
    *,
    workspace: Path,
    run_id: str,
    provisions: Sequence[CredentialProvision],
) -> tuple[CredentialRevocationReceipt, ...]:
    """Revoke a bounded, explicit set of provisions for one terminal run.

    This helper never discovers files and never removes directories.  Every
    supplied target is prevalidated before the first bundle is unlinked.
    """
    if not isinstance(run_id, str) or _SAFE_ID.fullmatch(run_id) is None:
        raise CredentialError("run_id must be a safe identifier")
    if not isinstance(provisions, (list, tuple)):
        raise CredentialError("run credential provisions must be an explicit sequence")
    if len(provisions) > MAX_RUN_CREDENTIAL_REVOCATIONS:
        raise CredentialError("run credential revocation exceeds the bounded target limit")

    canonical_workspace = _canonical_private_workspace(workspace)
    seen: set[CredentialBinding] = set()
    for provision in provisions:
        if not isinstance(provision, CredentialProvision):
            raise CredentialError("run credential revocation target must be a provision")
        binding = provision.descriptor.binding
        if binding.run_id != run_id:
            raise CredentialError("credential revocation target belongs to another run")
        if binding in seen:
            raise CredentialError("duplicate credential revocation target")
        seen.add(binding)
        _validate_revocation_target(
            canonical_workspace,
            binding,
            provision,
            CredentialRevocationCause.TERMINAL,
        )

    return tuple(
        revoke_terminal_attempt_credentials(
            workspace=canonical_workspace,
            expected_binding=provision.descriptor.binding,
            provision=provision,
        )
        for provision in provisions
    )


def revoke_generation_fenced_provisions(
    *,
    broker: CredentialBroker,
    workspace: Path,
    current_generation: int,
    descriptors: Sequence[CredentialDescriptor],
    adopted_bindings: Sequence[CredentialBinding],
) -> tuple[CredentialRevocationReceipt, ...]:
    """Revoke every exact old-generation provision not explicitly adopted.

    The successor supplies the bounded descriptor set reconstructed from
    immutable attempt manifests and a separately reconciled set of exact live
    adoptions. All inputs are validated before the first broker mutation; the
    controller never resolves the private bundle behind a descriptor.
    """
    if (
        isinstance(current_generation, bool)
        or not isinstance(current_generation, int)
        or current_generation < 2
    ):
        raise CredentialError("current generation must be an integer greater than one")
    if not isinstance(descriptors, (list, tuple)) or not isinstance(
        adopted_bindings, (list, tuple)
    ):
        raise CredentialError("generation fencing requires explicit bounded sequences")
    if len(descriptors) > MAX_RUN_CREDENTIAL_REVOCATIONS:
        raise CredentialError("generation-fenced revocation exceeds the bounded target limit")

    by_binding: dict[CredentialBinding, CredentialDescriptor] = {}
    validate_unique_credential_handles(descriptors)
    for descriptor in descriptors:
        if not isinstance(descriptor, CredentialDescriptor):
            raise CredentialError("generation-fenced target must be a credential descriptor")
        binding = descriptor.binding
        if binding.generation >= current_generation:
            raise CredentialError("generation-fenced target is not from an older generation")
        if binding in by_binding:
            raise CredentialError("duplicate generation-fenced credential target")
        by_binding[binding] = descriptor

    adopted: set[CredentialBinding] = set()
    for binding in adopted_bindings:
        if not isinstance(binding, CredentialBinding):
            raise CredentialError("adopted credential identity must be an exact binding")
        if binding not in by_binding:
            raise CredentialError("adopted credential binding has no supplied provision")
        if binding in adopted:
            raise CredentialError("duplicate adopted credential binding")
        adopted.add(binding)

    canonical_workspace = _canonical_private_workspace(workspace)
    return tuple(
        broker.revoke(
            workspace=canonical_workspace,
            expected_binding=binding,
            descriptor=descriptor,
            cause=CredentialRevocationCause.GENERATION_FENCE,
        )
        for binding, descriptor in by_binding.items()
        if binding not in adopted
    )


def load_worker_credentials(
    descriptor: CredentialDescriptor,
    *,
    expected_binding: CredentialBinding,
    bundle_path: Path = CREDENTIAL_MOUNT_PATH,
) -> dict[str, str]:
    """Validate and load one mounted bundle for exactly one agent attempt."""
    if descriptor.binding != expected_binding:
        raise CredentialError("credential descriptor identity/backend mismatch")
    if descriptor.state is CredentialState.NONE:
        if bundle_path.exists() or bundle_path.is_symlink():
            raise CredentialError("unexpected credential bundle for no-credential attempt")
        return {}
    _ensure_handle_unexpired(cast(CredentialHandle, descriptor.handle))
    _validate_bundle_file(bundle_path)
    raw = bundle_path.read_bytes()
    if len(raw) > MAX_CREDENTIAL_BUNDLE_BYTES:
        raise CredentialError("credential bundle exceeds the size limit")
    try:
        value = json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError("credential bundle is not strict UTF-8 JSON") from error
    data = _object(value, "credential bundle")
    expected = set(descriptor.to_public_dict()) | {"credentials"}
    _exact_keys(data, expected, "credential bundle")
    public = {key: entry for key, entry in data.items() if key != "credentials"}
    if descriptor_from_public_dict(public) != descriptor:
        raise CredentialError("credential bundle descriptor mismatch")
    credential_data = _object(data["credentials"], "credentials")
    if tuple(sorted(credential_data)) != descriptor.credential_names:
        raise CredentialError("credential bundle names do not match its descriptor")
    credentials: dict[str, str] = {}
    for name, value in credential_data.items():
        credentials[name] = _credential_value(value, name)
    return credentials


@contextmanager
def agent_credential_environment(
    backend_kind: BackendKind,
    credentials: Mapping[str, str],
) -> Iterator[None]:
    """Expose credentials only during one fresh agent invocation and restore env."""
    allowed = BACKEND_CREDENTIAL_ALLOWLIST.get(backend_kind)
    if allowed is None or not set(credentials).issubset(allowed):
        raise CredentialError("agent credential scope received non-allowlisted names")
    checked = {name: _credential_value(value, name) for name, value in credentials.items()}
    absent = object()
    previous: dict[str, object] = {
        name: os.environ.get(name, absent) for name in ALL_BACKEND_CREDENTIAL_NAMES
    }
    try:
        for name in ALL_BACKEND_CREDENTIAL_NAMES:
            os.environ.pop(name, None)
        os.environ.update(checked)
        yield
    finally:
        for name in ALL_BACKEND_CREDENTIAL_NAMES:
            os.environ.pop(name, None)
        for name, value in previous.items():
            if value is not absent:
                os.environ[name] = cast(str, value)


def ensure_no_credential_values(
    value: str | bytes,
    credentials: Mapping[str, str],
    *,
    label: str,
) -> None:
    """Reject response or evidence bytes containing an exact credential value."""
    haystack = value.encode("utf-8") if isinstance(value, str) else value
    leaked = [name for name, secret in credentials.items() if secret.encode("utf-8") in haystack]
    if leaked:
        raise CredentialError(f"{label} contains credential values for {sorted(leaked)!r}")


def ensure_file_has_no_credential_values(
    path: Path,
    credentials: Mapping[str, str],
    *,
    label: str,
) -> None:
    """Stream-scan a file for exact credential values without hashing them."""
    with path.open("rb") as stream:
        scan_stream_for_secret_material(
            stream,
            credentials,
            label=label,
            detect_secret_shapes=False,
        )


def scan_stream_for_secret_material(
    stream: BinaryIO,
    credentials: Mapping[str, str],
    *,
    label: str,
    detect_secret_shapes: bool = True,
    consume: Callable[[bytes], None] | None = None,
) -> int:
    """Scan arbitrary bytes without disclosing matches and optionally consume each chunk."""
    needles = {
        name: _credential_value(value, name).encode("utf-8") for name, value in credentials.items()
    }
    overlap = (
        max(
            [_SECRET_SCAN_PATTERN_BYTES if detect_secret_shapes else 1]
            + [len(value) for value in needles.values()]
        )
        - 1
    )
    suffix = b""
    size = 0
    while chunk := stream.read(_SECRET_SCAN_BLOCK_BYTES):
        if not isinstance(chunk, bytes):
            raise CredentialError(f"{label} stream must be binary")
        size += len(chunk)
        block = suffix + chunk
        leaked = [name for name, secret in needles.items() if secret in block]
        if leaked:
            raise CredentialError(f"{label} contains credential values for {sorted(leaked)!r}")
        if detect_secret_shapes and _SECRET_SHAPED_BYTES.search(block) is not None:
            raise CredentialError(f"{label} contains secret-shaped material")
        if consume is not None:
            consume(chunk)
        suffix = block[-overlap:] if overlap > 0 else b""
    return size


def redact_credential_values(text: str, credentials: Mapping[str, str]) -> str:
    """Redact exact credential values from an error before durable publication."""
    redacted = text
    for secret in sorted(credentials.values(), key=len, reverse=True):
        redacted = redacted.replace(secret, "<redacted>")
    return redacted


@contextmanager
def redact_durable_standard_streams(credentials: Mapping[str, str]) -> Iterator[None]:
    """Capture and redact process stdout/stderr during one backend invocation.

    Slurm persists file descriptors 1 and 2 directly.  Replacing only
    ``sys.stdout`` would not cover a backend subprocess, so this boundary uses
    descriptor duplication and replays the complete captured streams only
    after exact credential values have been removed.
    """
    checked = {name: _credential_value(value, name) for name, value in credentials.items()}
    if not checked:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    try:
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            os.dup2(stdout.fileno(), 1)
            os.dup2(stderr.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved[0], 1)
                os.dup2(saved[1], 2)
                _replay_redacted_stream(stdout, saved[0], checked)
                _replay_redacted_stream(stderr, saved[1], checked)
    finally:
        os.close(saved[0])
        os.close(saved[1])


def _descriptor_digest(
    binding: CredentialBinding,
    state: CredentialState,
    credential_names: Sequence[str],
    handle: CredentialHandle | None,
) -> str:
    payload = {
        "schema_version": CREDENTIAL_SCHEMA_VERSION,
        "run_id": binding.run_id,
        "item_id": binding.item_id,
        "attempt_id": binding.attempt_id,
        "task_digest": binding.task_digest,
        "generation": binding.generation,
        "backend_kind": binding.backend_kind,
        "state": state.value,
        "credential_names": list(credential_names),
        "credential_handle": None if handle is None else handle.to_public_dict(),
    }
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _credential_handle_from_public_dict(value: object) -> CredentialHandle | None:
    if value is None:
        return None
    data = _object(value, "credential handle")
    _exact_keys(
        data,
        {"broker_id", "handle_id", "expires_at_epoch_seconds"},
        "credential handle",
    )
    return CredentialHandle(
        broker_id=_string(data["broker_id"], "broker_id"),
        handle_id=_string(data["handle_id"], "handle_id"),
        expires_at_epoch_seconds=cast(int, data["expires_at_epoch_seconds"]),
    )


def _ensure_handle_unexpired(handle: CredentialHandle) -> None:
    if handle.expires_at_epoch_seconds <= int(time.time()):
        raise CredentialError("credential handle has expired")


def _legacy_handle_id(binding: CredentialBinding) -> str:
    serialized = json.dumps(
        {
            "run_id": binding.run_id,
            "item_id": binding.item_id,
            "attempt_id": binding.attempt_id,
            "task_digest": binding.task_digest,
            "generation": binding.generation,
            "backend_kind": binding.backend_kind,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"attempt-{hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:32]}"


def _replay_redacted_stream(
    stream: BinaryIO,
    destination: int,
    credentials: Mapping[str, str],
) -> None:
    stream.seek(0)
    content = stream.read()
    if not isinstance(content, bytes):
        raise CredentialError("captured durable stream is not binary")
    for secret in sorted(credentials.values(), key=len, reverse=True):
        content = content.replace(secret.encode("utf-8"), b"<redacted>")
    view = memoryview(content)
    while view:
        written = os.write(destination, view)
        view = view[written:]


def _select_credentials(
    backend_kind: BackendKind,
    ambient_environment: Mapping[str, str],
) -> dict[str, str]:
    allowed = BACKEND_CREDENTIAL_ALLOWLIST[backend_kind]
    selected: dict[str, str] = {}
    for name in sorted(allowed):
        if name in ambient_environment:
            selected[name] = _credential_value(ambient_environment[name], name)
    return selected


def _bundle_path(workspace: Path, binding: CredentialBinding) -> Path:
    return (
        workspace
        / CREDENTIAL_ROOT_NAME
        / "agent-credentials"
        / binding.run_id
        / binding.item_id
        / f"{binding.attempt_id}.json"
    )


def _revocation_receipt_path(workspace: Path, binding: CredentialBinding) -> Path:
    return (
        workspace
        / CREDENTIAL_ROOT_NAME
        / CREDENTIAL_REVOCATION_ROOT_NAME
        / binding.run_id
        / binding.item_id
        / f"{binding.attempt_id}.json"
    )


def _validate_revocation_target(
    workspace: Path,
    expected_binding: CredentialBinding,
    provision: CredentialProvision,
    cause: CredentialRevocationCause,
) -> os.stat_result | None:
    if not isinstance(expected_binding, CredentialBinding):
        raise CredentialError("credential revocation requires an exact binding")
    if not isinstance(provision, CredentialProvision):
        raise CredentialError("credential revocation requires a provision")
    if not isinstance(cause, CredentialRevocationCause):
        raise CredentialError("credential revocation requires a typed cause")
    if provision.descriptor.binding != expected_binding:
        raise CredentialError("credential revocation binding/descriptor mismatch")
    if descriptor_from_public_dict(provision.descriptor.to_public_dict()) != provision.descriptor:
        raise CredentialError("credential revocation descriptor is not canonical")

    expected_path = _bundle_path(workspace, expected_binding)
    if provision.descriptor.state is CredentialState.NONE:
        if provision.bundle_path is not None:
            raise CredentialError("no-credential revocation cannot name a bundle")
        if expected_path.exists() or expected_path.is_symlink():
            raise CredentialError("unexpected credential bundle for no-credential revocation")
        return None

    if provision.bundle_path != expected_path:
        raise CredentialError("credential revocation bundle path is not canonical for its binding")
    _validate_private_tree(expected_path.parent, workspace / CREDENTIAL_ROOT_NAME)
    try:
        bundle_info = expected_path.lstat()
    except FileNotFoundError:
        receipt_path = _revocation_receipt_path(workspace, expected_binding)
        if _load_revocation_receipt(receipt_path, provision.descriptor, cause) is None:
            raise CredentialError(
                "credential bundle is missing without an exact revocation receipt"
            )
        return None
    load_worker_credentials(
        provision.descriptor,
        expected_binding=expected_binding,
        bundle_path=expected_path,
    )
    return bundle_info


def _validate_revocation_target_metadata(
    workspace: Path,
    expected_binding: CredentialBinding,
    provision: CredentialProvision,
    cause: CredentialRevocationCause,
) -> os.stat_result | None:
    """Validate a broker-acknowledged cleanup target without reading it."""
    if not isinstance(expected_binding, CredentialBinding):
        raise CredentialError("credential revocation requires an exact binding")
    if not isinstance(provision, CredentialProvision):
        raise CredentialError("credential revocation requires a provision")
    if not isinstance(cause, CredentialRevocationCause):
        raise CredentialError("credential revocation requires a typed cause")
    if provision.descriptor.binding != expected_binding:
        raise CredentialError("credential revocation binding/descriptor mismatch")
    if descriptor_from_public_dict(provision.descriptor.to_public_dict()) != provision.descriptor:
        raise CredentialError("credential revocation descriptor is not canonical")

    expected_path = _bundle_path(workspace, expected_binding)
    if provision.descriptor.state is CredentialState.NONE:
        if provision.bundle_path is not None:
            raise CredentialError("no-credential revocation cannot name a bundle")
        if expected_path.exists() or expected_path.is_symlink():
            raise CredentialError("unexpected credential bundle for no-credential revocation")
        return None

    if provision.bundle_path != expected_path:
        raise CredentialError("credential revocation bundle path is not canonical for its binding")
    _validate_private_tree(expected_path.parent, workspace / CREDENTIAL_ROOT_NAME)
    try:
        bundle_info = expected_path.lstat()
    except FileNotFoundError:
        return None
    _validate_bundle_file(expected_path)
    return bundle_info


def _revocation_receipt_payload(
    descriptor: CredentialDescriptor,
    cause: CredentialRevocationCause,
) -> dict[str, object]:
    return {
        "schema_version": CREDENTIAL_REVOCATION_SCHEMA_VERSION,
        "receipt_type": "credential_bundle_revocation",
        "cause": cause.value,
        "credential_descriptor": descriptor.to_public_dict(),
    }


def _ensure_revocation_receipt(
    path: Path,
    descriptor: CredentialDescriptor,
    cause: CredentialRevocationCause,
) -> CredentialDescriptor:
    existing = _load_revocation_receipt(path, descriptor, cause)
    if existing is not None:
        _remove_recovered_receipt_stage(path, descriptor, cause)
        return existing

    _ensure_private_directory(path.parent)
    stage = _revocation_receipt_stage_path(path, descriptor)
    if stage.exists() or stage.is_symlink():
        _validate_revocation_receipt_file(
            stage,
            descriptor,
            cause,
            allowed_link_counts=(1,),
        )
    else:
        _write_exclusive_json(
            stage,
            _revocation_receipt_payload(descriptor, cause),
            mode=0o400,
        )
    try:
        os.link(stage, path, follow_symlinks=False)
    except FileExistsError:
        _validate_revocation_receipt_file(
            path,
            descriptor,
            cause,
            allowed_link_counts=(1, 2),
        )
    _remove_recovered_receipt_stage(path, descriptor, cause)
    _validate_revocation_receipt_file(
        path,
        descriptor,
        cause,
        allowed_link_counts=(1,),
    )
    _fsync_directory(path.parent)
    return descriptor


def _load_revocation_receipt(
    path: Path,
    descriptor: CredentialDescriptor,
    cause: CredentialRevocationCause,
) -> CredentialDescriptor | None:
    if not path.exists() and not path.is_symlink():
        return None
    _validate_private_tree(path.parent, path.parents[3])
    stage = _revocation_receipt_stage_path(path, descriptor)
    if stage.exists() or stage.is_symlink():
        _validate_revocation_receipt_file(
            path,
            descriptor,
            cause,
            allowed_link_counts=(2,),
        )
        _remove_recovered_receipt_stage(path, descriptor, cause)
    else:
        _validate_revocation_receipt_file(
            path,
            descriptor,
            cause,
            allowed_link_counts=(1,),
        )
    return descriptor


def _validate_revocation_receipt_file(
    path: Path,
    descriptor: CredentialDescriptor,
    cause: CredentialRevocationCause,
    *,
    allowed_link_counts: tuple[int, ...],
) -> None:
    _validate_private_file(
        path,
        mode=0o400,
        allowed_link_counts=allowed_link_counts,
        label="credential revocation receipt",
    )
    value = _read_strict_json(path, "credential revocation receipt")
    data = _object(value, "credential revocation receipt")
    expected = {"schema_version", "receipt_type", "cause", "credential_descriptor"}
    _exact_keys(data, expected, "credential revocation receipt")
    if data["schema_version"] != CREDENTIAL_REVOCATION_SCHEMA_VERSION:
        raise CredentialError("unsupported credential revocation receipt schema")
    if data["receipt_type"] != "credential_bundle_revocation":
        raise CredentialError("credential revocation receipt type mismatch")
    if data["cause"] != cause.value:
        raise CredentialError("credential revocation receipt cause mismatch")
    if descriptor_from_public_dict(data["credential_descriptor"]) != descriptor:
        raise CredentialError("credential revocation receipt descriptor mismatch")


def _revocation_receipt_stage_path(
    path: Path,
    descriptor: CredentialDescriptor,
) -> Path:
    return path.with_name(f".{path.name}.{descriptor.descriptor_digest}.pending")


def _remove_recovered_receipt_stage(
    path: Path,
    descriptor: CredentialDescriptor,
    cause: CredentialRevocationCause,
) -> None:
    stage = _revocation_receipt_stage_path(path, descriptor)
    if not stage.exists() and not stage.is_symlink():
        return
    _validate_revocation_receipt_file(
        stage,
        descriptor,
        cause,
        allowed_link_counts=(2,),
    )
    receipt_info = path.lstat()
    stage_info = stage.lstat()
    if (receipt_info.st_dev, receipt_info.st_ino) != (stage_info.st_dev, stage_info.st_ino):
        raise CredentialError("credential revocation receipt has an unexpected staged file")
    stage.unlink()
    _fsync_directory(path.parent)


def _write_exclusive_json(path: Path, payload: Mapping[str, object], *, mode: int) -> None:
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    file_descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _read_strict_json(path: Path, label: str) -> object:
    raw = path.read_bytes()
    if len(raw) > MAX_CREDENTIAL_BUNDLE_BYTES:
        raise CredentialError(f"{label} exceeds the size limit")
    try:
        return json.loads(raw, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError(f"{label} is not strict UTF-8 JSON") from error


def _unlink_exact_bundle(path: Path, expected_info: os.stat_result) -> None:
    current = path.lstat()
    if (current.st_dev, current.st_ino) != (expected_info.st_dev, expected_info.st_ino):
        raise CredentialError("credential bundle changed during revocation")
    _validate_bundle_file(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    parent_descriptor = os.open(path.parent, flags)
    try:
        os.unlink(path.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _validate_private_tree(path: Path, root: Path) -> None:
    if not root.is_absolute() or path == root.parent or root not in path.parents:
        raise CredentialError("credential path is outside the controller-owned root")
    current = path
    while True:
        _validate_private_directory(current)
        if current == root:
            return
        current = current.parent


def _validate_private_file(
    path: Path,
    *,
    mode: int,
    allowed_link_counts: tuple[int, ...],
    label: str,
) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise CredentialError(f"{label} must be an absolute regular file")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise CredentialError(f"{label} must be a regular file")
    if stat.S_IMODE(info.st_mode) != mode:
        raise CredentialError(f"{label} must have mode {mode:04o}")
    if info.st_uid != os.geteuid():
        raise CredentialError(f"{label} must be owned by the controller user")
    if info.st_nlink not in allowed_link_counts:
        raise CredentialError(f"{label} has an invalid hard-link count")


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


def _canonical_private_workspace(workspace: Path) -> Path:
    if not isinstance(workspace, Path) or not workspace.is_absolute() or workspace.is_symlink():
        raise CredentialError("credential workspace must be an absolute regular directory")
    canonical = workspace.resolve(strict=True)
    if canonical != workspace or not canonical.is_dir():
        raise CredentialError("credential workspace must already be canonical")
    return canonical


def _ensure_private_directory(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        if current.is_symlink():
            raise CredentialError("credential directory cannot be a symlink")
        missing.append(current)
        current = current.parent
    if current.is_symlink() or not current.is_dir():
        raise CredentialError("credential directory parent must be a regular directory")
    for directory in reversed(missing):
        directory.mkdir(mode=0o700)
    current = path
    while current.name != CREDENTIAL_ROOT_NAME:
        _validate_private_directory(current)
        current = current.parent
    _validate_private_directory(current)


def _validate_private_directory(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise CredentialError("credential path component must be a regular directory")
    if stat.S_IMODE(info.st_mode) != 0o700:
        raise CredentialError("credential directories must have mode 0700")
    if info.st_uid != os.geteuid():
        raise CredentialError("credential directories must be owned by the controller user")


def _write_exclusive_bundle(path: Path, payload: Mapping[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"), sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    _validate_bundle_file(path)


def _validate_bundle_file(path: Path) -> None:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise CredentialError("credential bundle must be an absolute regular file")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise CredentialError("credential bundle must be a regular file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise CredentialError("credential bundle must have mode 0600")
    if info.st_uid != os.geteuid():
        raise CredentialError("credential bundle must be owned by the worker user")
    if info.st_nlink != 1:
        raise CredentialError("credential bundle must have exactly one hard link")


def _credential_value(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise CredentialError(f"credential {name!r} must be a non-empty string without NUL")
    return value


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CredentialError(f"{name} must be a string-keyed object")
    return value


def _exact_keys(data: Mapping[str, object], expected: set[str], name: str) -> None:
    unknown = sorted(set(data) - expected)
    missing = sorted(expected - set(data))
    if unknown or missing:
        raise CredentialError(f"{name} keys invalid; missing={missing}, unknown={unknown}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise CredentialError(f"{name} must be a string")
    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CredentialError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


__all__ = [
    "ALL_BACKEND_CREDENTIAL_NAMES",
    "BACKEND_CREDENTIAL_ALLOWLIST",
    "BackendKind",
    "CREDENTIAL_MOUNT_PATH",
    "CREDENTIAL_REVOCATION_ROOT_NAME",
    "CREDENTIAL_ROOT_NAME",
    "MAX_RUN_CREDENTIAL_REVOCATIONS",
    "CredentialBroker",
    "CredentialBinding",
    "CredentialDescriptor",
    "CredentialError",
    "CredentialHandle",
    "CredentialProvision",
    "CredentialRevocationCause",
    "CredentialRevocationReceipt",
    "CredentialState",
    "NoCredentialBroker",
    "agent_credential_environment",
    "descriptor_from_public_dict",
    "ensure_file_has_no_credential_values",
    "ensure_no_credential_values",
    "load_worker_credentials",
    "locate_credential_provision",
    "materialize_credential_provision",
    "prepare_credential_provision",
    "redact_durable_standard_streams",
    "redact_credential_values",
    "revoke_broker_acknowledged_credentials",
    "revoke_generation_fenced_provisions",
    "revoke_run_credential_provisions",
    "revoke_terminal_attempt_credentials",
    "scan_stream_for_secret_material",
    "validate_unique_credential_handles",
]
