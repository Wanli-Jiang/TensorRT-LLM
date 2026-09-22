# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated role-process contracts used by Staircase worker jobs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from .credentials import BackendKind as CredentialBackendKind
from .credentials import (
    CredentialBinding,
    agent_credential_runtime,
    descriptor_from_public_dict,
    ensure_no_credential_values,
    load_worker_credentials,
    redact_credential_values,
)
from .launch_policy import AgentLaunchPolicyError, agent_launch_policy_from_public_dict
from .slurm import AGENT_POLICY_DIGEST_ENVIRONMENT

RoleName = Literal["plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"]
DomainProfile = Literal["planner", "smith", "assembler", "tuner", "qa"]
BackendKind = Literal["claude-code", "codex"]

_ROLES = {"plan_drafter", "plan_reviewer", "coder", "reviewer", "qa"}
_PROFILES = {"planner", "smith", "assembler", "tuner", "qa"}
_BACKENDS = {"claude-code", "codex"}
_SCHEDULER_CLIENTS = ("sacct", "sbatch", "scancel", "scontrol", "squeue", "srun")
_SCHEDULER_AUTH_ENVIRONMENT = frozenset(
    {
        "KRB5CCNAME",
        "MUNGE_SOCKET",
        "MUNGE_SOCKET_PATH",
        "SLURM_JWT",
        "SSH_AUTH_SOCK",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RoleProcessError(RuntimeError):
    """Raised when a role process receives or emits an invalid contract."""


@dataclass(frozen=True)
class RoleProcessSpec:
    """Immutable input for exactly one fresh role process."""

    schema_version: int
    run_id: str
    task_digest: str
    generation: int
    item_id: str
    attempt_id: str
    prompt_id: str
    role: RoleName
    profile: DomainProfile
    backend_kind: BackendKind
    model: str
    source_root: str
    cwd: str
    result_path: str
    system_prompt: str
    prompt: str
    credential_descriptor: dict[str, object]
    launch_policy: dict[str, object]


@dataclass(frozen=True)
class RoleProcessResult:
    """Atomic result produced by one role process."""

    schema_version: int
    run_id: str
    task_digest: str
    generation: int
    item_id: str
    attempt_id: str
    prompt_id: str
    role: RoleName
    profile: DomainProfile
    response: str
    response_digest: str


def _require_safe_id(name: str, value: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise RoleProcessError(f"{name} must be a safe non-empty identifier")


def _canonical_child(root: str, child: str, *, name: str) -> str:
    root_path = Path(root).resolve(strict=True)
    child_path = Path(child).resolve(strict=True)
    try:
        child_path.relative_to(root_path)
    except ValueError as exc:
        raise RoleProcessError(f"{name} must resolve beneath source_root") from exc
    if not child_path.is_dir():
        raise RoleProcessError(f"{name} must be an existing directory")
    return str(child_path)


def validate_role_spec(spec: RoleProcessSpec) -> RoleProcessSpec:
    """Validate and canonicalize an immutable worker input."""
    if spec.schema_version != 1:
        raise RoleProcessError(f"unsupported role spec version {spec.schema_version}")
    for name in ("run_id", "item_id", "attempt_id", "prompt_id"):
        _require_safe_id(name, getattr(spec, name))
    if not re.fullmatch(r"[0-9a-f]{64}", spec.task_digest):
        raise RoleProcessError("task_digest must be a lowercase SHA-256 digest")
    if (
        isinstance(spec.generation, bool)
        or not isinstance(spec.generation, int)
        or spec.generation < 1
    ):
        raise RoleProcessError("generation must be a positive integer")
    if spec.role not in _ROLES:
        raise RoleProcessError(f"unsupported role {spec.role!r}")
    if spec.profile not in _PROFILES:
        raise RoleProcessError(f"unsupported domain profile {spec.profile!r}")
    if spec.backend_kind not in _BACKENDS:
        raise RoleProcessError(f"unsupported backend {spec.backend_kind!r}")
    if not spec.model.strip() or not spec.system_prompt.strip() or not spec.prompt.strip():
        raise RoleProcessError("model, system_prompt, and prompt must be non-empty")
    binding = _credential_binding(spec)
    try:
        descriptor = descriptor_from_public_dict(spec.credential_descriptor)
    except ValueError as error:
        raise RoleProcessError(f"invalid public credential descriptor: {error}") from error
    if descriptor.binding != binding:
        raise RoleProcessError("credential descriptor identity/backend differs from role spec")
    try:
        launch_policy = agent_launch_policy_from_public_dict(spec.launch_policy)
    except AgentLaunchPolicyError as error:
        raise RoleProcessError(f"invalid role launch policy: {error}") from error
    canonical_root = str(Path(spec.source_root).resolve(strict=True))
    canonical_cwd = _canonical_child(canonical_root, spec.cwd, name="cwd")
    result_path = Path(spec.result_path)
    if not result_path.is_absolute():
        raise RoleProcessError("result_path must be absolute")
    return RoleProcessSpec(
        **{
            **asdict(spec),
            "source_root": canonical_root,
            "cwd": canonical_cwd,
            "result_path": str(result_path.resolve(strict=False)),
            "credential_descriptor": descriptor.to_public_dict(),
            "launch_policy": launch_policy.to_public_dict(),
        }
    )


def load_role_spec(path: Path) -> RoleProcessSpec:
    """Load and strictly validate one JSON role input."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RoleProcessError("role input must be a JSON object")
    expected = {field.name for field in RoleProcessSpec.__dataclass_fields__.values()}
    unknown = sorted(set(data) - expected)
    missing = sorted(expected - set(data))
    if unknown or missing:
        raise RoleProcessError(f"role input keys invalid; missing={missing}, unknown={unknown}")
    return validate_role_spec(RoleProcessSpec(**data))


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def execute_role(spec: RoleProcessSpec) -> RoleProcessResult:
    """Execute one role in a fresh stateless AgentLayer and atomically publish its response."""
    checked = validate_role_spec(spec)
    from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig, SessionConfig

    credentials: dict[str, str] = {}
    try:
        validate_role_process_negative_preflight()
        launch_policy = agent_launch_policy_from_public_dict(checked.launch_policy)
        if os.environ.get(AGENT_POLICY_DIGEST_ENVIRONMENT) != launch_policy.digest:
            raise RoleProcessError(
                "role launch policy is not bound by the trusted scheduler environment"
            )
        descriptor = descriptor_from_public_dict(checked.credential_descriptor)
        credentials = load_worker_credentials(
            descriptor,
            expected_binding=_credential_binding(checked),
        )
        with agent_credential_runtime(
            checked.backend_kind,
            credentials,
        ) as protected_credentials:
            credentials = protected_credentials
            with AgentLayer(
                AgentLayerConfig(
                    name=f"staircase-{checked.role}-{checked.item_id}",
                    system_prompt=checked.system_prompt,
                    backend=BackendConfig(
                        kind=checked.backend_kind,
                        model=checked.model,
                        cwd=Path(checked.cwd),
                    ),
                    session=SessionConfig(mode="stateless"),
                    human_input_enabled=False,
                )
            ) as layer:
                response = layer(checked.prompt)
        if not isinstance(response, str):
            raise RoleProcessError("role response must be text")
        ensure_no_credential_values(response, credentials, label="role response")
    except (OSError, ValueError, RuntimeError) as error:
        detail = redact_credential_values(str(error), credentials)
        raise RoleProcessError(f"role execution failed closed: {detail}") from None
    result = RoleProcessResult(
        schema_version=1,
        run_id=checked.run_id,
        task_digest=checked.task_digest,
        generation=checked.generation,
        item_id=checked.item_id,
        attempt_id=checked.attempt_id,
        prompt_id=checked.prompt_id,
        role=checked.role,
        profile=checked.profile,
        response=response,
        response_digest=hashlib.sha256(response.encode("utf-8")).hexdigest(),
    )
    _atomic_write_json(Path(checked.result_path), asdict(result))
    return result


def validate_role_process_negative_preflight(
    environment: dict[str, str] | None = None,
) -> None:
    """Reject scheduler clients and auth material before an agent starts."""
    observed_environment = os.environ if environment is None else environment
    contaminated = sorted(_SCHEDULER_AUTH_ENVIRONMENT.intersection(observed_environment))
    if contaminated:
        raise RoleProcessError(
            f"role negative preflight found scheduler authentication material: {contaminated!r}"
        )
    clients = tuple(name for name in _SCHEDULER_CLIENTS if _scheduler_client_path(name) is not None)
    if clients:
        raise RoleProcessError(
            f"role negative preflight found scheduler clients in PATH: {list(clients)!r}"
        )


def _scheduler_client_path(name: str) -> str | None:
    """Resolve a prohibited scheduler client through an injectable test seam."""
    return shutil.which(name)


def _credential_binding(spec: RoleProcessSpec) -> CredentialBinding:
    return CredentialBinding(
        run_id=spec.run_id,
        item_id=spec.item_id,
        attempt_id=spec.attempt_id,
        task_digest=spec.task_digest,
        generation=spec.generation,
        backend_kind=cast(CredentialBackendKind, spec.backend_kind),
    )
