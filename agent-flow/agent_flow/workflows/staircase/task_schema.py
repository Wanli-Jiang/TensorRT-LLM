# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Strict task loading for the Slurm-first Staircase workflow.

The YAML document is an input format, not durable state. This module rejects
unknown fields, validates all filesystem and resource identities, and returns
an immutable normalized value. The SHA-256 digest is computed from that
normalized value and is therefore stable across YAML formatting and mapping
order changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal, TypeAlias, cast

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from .common.credentials import BACKEND_CREDENTIAL_ALLOWLIST
from .common.gates import GatePolicyError, validate_pytest_selector

TASK_SCHEMA_VERSION = 2

_OBSOLETE_DATA_PLANE_TOKENS = (
    "_torch/staircase",
    "TRTLLM_STAIRCASE",
)
_RESOURCE_CLASS_NAMES = (
    "coder_analysis",
    "exploratory_probe",
    "deterministic_gate",
    "reviewer_analysis",
    "reviewer_rerun",
)
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ARCHITECTURE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SLURM_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENVIRONMENT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_COMPUTE_CAPABILITY_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+$")
_IDENTITY_RE = re.compile(r"^[^\x00\r\n]{1,512}$")
_MEMORY_RE = re.compile(r"^([1-9][0-9]*)(MiB|GiB|M|G)$")
_TIME_RE = re.compile(r"^(?:(\d+)-)?(\d{1,3}):(\d{2}):(\d{2})$")
_SCHEDULER_ENVIRONMENT_RE = re.compile(r"^(?:SLURM|SBATCH|SRUN|PMI|PMIX)(?:_|$)")
_CREDENTIAL_ENVIRONMENT_RE = re.compile(
    r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_?KEY|(?:^|_)AUTH(?:_|$)|COOKIE)",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE_RE = re.compile(
    r"(?:bearer\s+[A-Za-z0-9._~+/-]{8,}|"
    r"basic\s+[A-Za-z0-9+/=]{8,}|"
    r"(?:sk|nvapi|gh[oprsu])[-_][A-Za-z0-9_-]{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)
_URL_USERINFO_RE = re.compile(
    r"\b[A-Za-z][A-Za-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@",
    re.IGNORECASE,
)
_NESTED_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_]*"
    r"(?:API_?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|PRIVATE_?KEY|AUTH|COOKIE)"
    r"[A-Za-z0-9_]*)\s*=",
    re.IGNORECASE,
)
_OPAQUE_TOKEN_RE = re.compile(r"[A-Za-z0-9._~+/-]{24,}\Z")
_PATH_ENVIRONMENT_NAMES = frozenset(
    {
        "CUDA_HOME",
        "HF_HOME",
        "LLM_MODELS_ROOT",
        "TRANSFORMERS_CACHE",
        "XDG_CACHE_HOME",
    }
)
_ROLE_CLASS_NAMES = (
    "plan_drafter",
    "plan_reviewer",
    "smith_coder",
    "assembler_coder",
    "tuner_coder",
    "reviewer",
    "qa",
)
_MAX_ROLE_ENVIRONMENT_VALUE_BYTES = 4_096
_MAX_ROLE_CREDENTIAL_TTL_SECONDS = 86_400

_DirtyPolicy: TypeAlias = Literal["reject", "allow"]
_ExecutionMode: TypeAlias = Literal["slurm", "local"]
_AgentBackendKind: TypeAlias = Literal["codex", "claude-code"]
_DispatchMode: TypeAlias = Literal["nested_submission"]
_DeliveryMode: TypeAlias = Literal["diff_only", "signed_commits"]


class TaskSchemaError(ValueError):
    """Raised when a Staircase task has one or more schema violations."""


class CertificationMode(str, Enum):
    """Explicit evidence boundary frozen into a Staircase task."""

    LOCAL = "LOCAL"
    SYNTHETIC = "SYNTHETIC"
    REAL = "REAL"


class SlurmRole(str, Enum):
    """Every fresh LLM invocation admitted by the Slurm control plane."""

    PLAN_DRAFTER = "plan_drafter"
    PLAN_REVIEWER = "plan_reviewer"
    SMITH_CODER = "smith_coder"
    ASSEMBLER_CODER = "assembler_coder"
    TUNER_CODER = "tuner_coder"
    REVIEWER = "reviewer"
    QA = "qa"


class RoleNetworkPolicy(str, Enum):
    """Closed egress policy understood by the trusted worker launcher."""

    BACKEND_API_ONLY = "backend_api_only"


class ContainerLaunchMode(str, Enum):
    """Site-certified boundary used to enter a Pyxis container."""

    IN_ALLOCATION_SRUN = "in_allocation_srun"


class NetworkEnforcement(str, Enum):
    """Trusted mechanism that enforces an agent role's egress policy."""

    CERTIFIED_WORKER_IMAGE = "certified_worker_image"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


@dataclass(frozen=True, slots=True)
class RepositoryConfig:
    """Normalized repository identity and shared workspace policy."""

    root: Path
    base_commit: str
    dirty_policy: _DirtyPolicy
    workspace_root: Path


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    """Normalized checkpoint and independent-reference identity."""

    checkpoint: Path
    provenance: str
    architecture: str
    additional_sources: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ParallelMapping:
    """Explicit TensorRT-LLM parallel axes."""

    tensor_parallel_size: int
    pipeline_parallel_size: int
    moe_expert_parallel_size: int
    moe_tensor_parallel_size: int
    attention_data_parallel_size: int


@dataclass(frozen=True, slots=True)
class TargetConfig:
    """Normalized ModelingV2 target identity."""

    family: str
    checkpoint_id: str
    sm: int
    world_size: int
    mapping: ParallelMapping
    features: tuple[str, ...]
    expected_route: str
    synthetic_target: bool


@dataclass(frozen=True, slots=True)
class AccuracyGate:
    """Accuracy reference and comparison protocol."""

    selector: str
    reference: str
    protocol: str
    tolerance: float


@dataclass(frozen=True, slots=True)
class GatesConfig:
    """Required stage and product gates."""

    accuracy: AccuracyGate
    feature_signal_tests: tuple[str, ...]
    boot_tests: tuple[str, ...]
    component_tests: tuple[str, ...]
    collective_tests: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExpectedProductIdentityConfig:
    """Controller-audited product identity required for REAL evidence."""

    repository_commit: str
    python_tensorrt_llm_path: str
    native_build_identity: str
    image_identity: str
    compute_capability: str
    collective_backend: str
    transport: str


@dataclass(frozen=True, slots=True)
class CertificationConfig:
    """Task-level certification mode and optional REAL product identity."""

    mode: CertificationMode
    expected_product_identity: ExpectedProductIdentityConfig | None


@dataclass(frozen=True, slots=True)
class MountConfig:
    """One controller-visible host-to-container mount."""

    host_path: Path
    container_path: str
    read_only: bool


@dataclass(frozen=True, slots=True)
class ControllerConfig:
    """Normalized Slurm controller request and execution identity."""

    account: str
    partition: str
    qos: str | None
    reservation: str | None
    time_limit_seconds: int
    cpus_per_task: int
    memory_mib: int
    image: Path
    mounts: tuple[MountConfig, ...]
    environment: tuple[tuple[str, str], ...]
    build_identity: str
    dispatch_mode: _DispatchMode
    requeue: bool
    advance_signal_lead_seconds: int
    heartbeat_timeout_seconds: int
    lease_timeout_seconds: int
    orphan_grace_seconds: int
    credential_broker_socket: str | None = None
    scheduler_clients: bool = True
    container_launch_mode: ContainerLaunchMode = ContainerLaunchMode.IN_ALLOCATION_SRUN


@dataclass(frozen=True, slots=True)
class AgentWorkerConfig:
    """Scheduler-client-free image and its trusted network-policy boundary."""

    image: Path
    build_identity: str
    scheduler_clients: bool
    network_enforcement: NetworkEnforcement


@dataclass(frozen=True, slots=True)
class ResourceClass:
    """One permitted immutable worker resource request."""

    name: str
    nodes: int
    tasks_per_node: int
    gpus_per_node: int
    cpus_per_task: int
    memory_mib: int
    time_limit_seconds: int

    @property
    def total_tasks(self) -> int:
        """Return the number of ranks in this resource class."""
        return self.nodes * self.tasks_per_node

    @property
    def total_gpus(self) -> int:
        """Return the aggregate GPU request in this resource class."""
        return self.nodes * self.gpus_per_node


@dataclass(frozen=True, slots=True)
class GateClass:
    """Bind one deterministic GateSpec identity to a trusted resource class."""

    gate_id: str
    resource_class: str


@dataclass(frozen=True, slots=True)
class RoleEnvironmentPolicy:
    """Non-secret ambient names and fixed values admitted for one role."""

    allowlist: tuple[str, ...]
    values: tuple[tuple[str, str], ...]

    def value(self, name: str) -> str:
        """Return one fixed environment value, raising for unknown names."""
        for candidate, value in self.values:
            if candidate == name:
                return value
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class CredentialBrokerPolicy:
    """Secret-free per-attempt credential acquisition policy."""

    broker_id: str
    allowed_credential_names: tuple[str, ...]
    per_attempt_ttl_seconds: int


@dataclass(frozen=True, slots=True)
class SlurmRoleClass:
    """Least-privilege scheduler, network, environment, and broker binding."""

    role: SlurmRole
    resource_class: str
    network_policy: RoleNetworkPolicy
    environment: RoleEnvironmentPolicy
    credential_broker: CredentialBrokerPolicy


@dataclass(frozen=True, slots=True)
class OverrideBounds:
    """Per-item upper bounds for controller-approved escalation."""

    max_nodes: int
    max_tasks_per_node: int
    max_gpus_per_node: int
    max_cpus_per_task: int
    max_memory_mib: int
    max_time_limit_seconds: int


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded infrastructure retry counts."""

    preempted: int
    node_failure: int


@dataclass(frozen=True, slots=True)
class SmithConfig:
    """Parallel Smith resource envelope."""

    max_parallel_items: int
    max_nodes_total: int
    max_gpus_total: int
    resource_classes: tuple[ResourceClass, ...]
    per_item_override_bounds: OverrideBounds
    retry_policy: RetryPolicy
    distinct_nodes_required: bool = False
    exclusive: bool = False

    def resource_class(self, name: str) -> ResourceClass:
        """Return a named class from the closed resource-class vocabulary.

        Args:
            name: Resource-class name.

        Returns:
            The matching immutable resource class.

        Raises:
            KeyError: If ``name`` is not configured.
        """
        for resource in self.resource_classes:
            if resource.name == name:
                return resource
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class SlurmConfig:
    """Slurm controller and Smith configuration."""

    controller: ControllerConfig
    smith: SmithConfig
    gate_classes: tuple[GateClass, ...] = ()
    role_classes: tuple[SlurmRoleClass, ...] = ()
    agent_worker: AgentWorkerConfig | None = None

    def gate_class(self, gate_id: str) -> GateClass:
        """Return the frozen trusted class for one deterministic gate."""
        for gate_class in self.gate_classes:
            if gate_class.gate_id == gate_id:
                return gate_class
        raise KeyError(gate_id)

    def role_class(self, role: SlurmRole | str) -> SlurmRoleClass:
        """Return the frozen least-privilege policy for one agent role."""
        try:
            normalized = SlurmRole(role)
        except ValueError as error:
            raise KeyError(role) from error
        for role_class in self.role_classes:
            if role_class.role is normalized:
                return role_class
        raise KeyError(normalized.value)


@dataclass(frozen=True, slots=True)
class AgentExecutionConfig:
    """Reproducible agent backend and model identity."""

    backend_kind: _AgentBackendKind
    model: str


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Production execution backend configuration."""

    mode: _ExecutionMode
    slurm: SlurmConfig
    agent: AgentExecutionConfig


@dataclass(frozen=True, slots=True)
class DeliveryConfig:
    """Review-ready delivery policy."""

    mode: _DeliveryMode
    branch: str | None
    merge: bool
    push: bool


@dataclass(frozen=True, slots=True)
class NormalizedTask:
    """Immutable validated task shared by controller, prompts, and workers."""

    schema_version: int
    repository: RepositoryConfig
    reference: ReferenceConfig
    target: TargetConfig
    gates: GatesConfig
    certification: CertificationConfig
    execution: ExecutionConfig
    delivery: DeliveryConfig
    digest: str


def _type_name(value: object) -> str:
    return type(value).__name__


def _mapping(
    value: object,
    path: str,
    allowed: frozenset[str],
    required: frozenset[str],
    errors: list[str],
) -> dict[str, object]:
    if not isinstance(value, dict):
        errors.append(f"'{path}' must be a mapping, got {_type_name(value)}")
        return {}
    result: dict[str, object] = {}
    for key, entry in value.items():
        if not isinstance(key, str):
            errors.append(f"'{path}' has a non-string key {key!r}")
            continue
        result[key] = entry
        if key not in allowed:
            errors.append(f"unknown field '{path}.{key}'")
    for key in sorted(required - result.keys()):
        errors.append(f"missing required field '{path}.{key}'")
    return result


def _required_mapping(
    parent: dict[str, object],
    key: str,
    parent_path: str,
    allowed: frozenset[str],
    required: frozenset[str],
    errors: list[str],
) -> dict[str, object]:
    path = f"{parent_path}.{key}" if parent_path else key
    if key not in parent:
        return {}
    return _mapping(parent[key], path, allowed, required, errors)


def _nonempty_string(value: object, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"'{path}' must be a non-empty string, got {value!r}")
        return ""
    return value.strip()


def _optional_string(value: object, path: str, errors: list[str]) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, path, errors)


def _single_line_string(value: object, path: str, errors: list[str]) -> str:
    result = _nonempty_string(value, path, errors)
    if isinstance(value, str) and any(
        character in value
        for character in (
            "\n",
            "\r",
            "\v",
            "\f",
            "\x1c",
            "\x1d",
            "\x1e",
            "\x85",
            "\u2028",
            "\u2029",
        )
    ):
        errors.append(f"'{path}' must be a single-line string")
    return result


def _strict_bool(value: object, path: str, errors: list[str]) -> bool:
    if not isinstance(value, bool):
        errors.append(f"'{path}' must be a boolean, got {value!r}")
        return False
    return value


def _positive_int(value: object, path: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        errors.append(f"'{path}' must be an integer >= 1, got {value!r}")
        return 1
    return value


def _nonnegative_int(value: object, path: str, errors: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"'{path}' must be an integer >= 0, got {value!r}")
        return 0
    return value


def _nonnegative_float(value: object, path: str, errors: list[str]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"'{path}' must be a finite number >= 0, got {value!r}")
        return 0.0
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        errors.append(f"'{path}' must be a finite number >= 0, got {value!r}")
        return 0.0
    return normalized


def _slug(value: object, path: str, errors: list[str]) -> str:
    result = _nonempty_string(value, path, errors)
    if result and _SLUG_RE.fullmatch(result) is None:
        errors.append(
            f"'{path}' must be a lowercase safe slug containing only letters, digits, '_' or '-', "
            f"got {result!r}"
        )
    return result


def _slurm_name(value: object, path: str, errors: list[str]) -> str:
    result = _nonempty_string(value, path, errors)
    if result and _SLURM_NAME_RE.fullmatch(result) is None:
        errors.append(f"'{path}' contains unsafe Slurm-name characters: {result!r}")
    return result


def _existing_path(
    value: object,
    path: str,
    expected_kind: Literal["file", "directory", "either"],
    errors: list[str],
) -> Path:
    text = _nonempty_string(value, path, errors)
    if not text:
        return Path("/")
    candidate = Path(text).expanduser()
    if ".." in candidate.parts:
        errors.append(f"'{path}' must not contain '..' path traversal: {text!r}")
    if not candidate.is_absolute():
        errors.append(f"'{path}' must be an absolute path: {text!r}")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        errors.append(f"'{path}' does not resolve to an existing path: {text!r} ({exc})")
        return candidate.absolute()
    if expected_kind == "file" and not resolved.is_file():
        errors.append(f"'{path}' must point to a file: {resolved}")
    elif expected_kind == "directory" and not resolved.is_dir():
        errors.append(f"'{path}' must point to a directory: {resolved}")
    return resolved


def _string_list(
    value: object,
    path: str,
    errors: list[str],
    *,
    allow_empty: bool,
    slug_entries: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        errors.append(f"'{path}' must be a list, got {_type_name(value)}")
        return ()
    if not allow_empty and not value:
        errors.append(f"'{path}' must not be empty")
    result: list[str] = []
    for index, entry in enumerate(value):
        entry_path = f"{path}[{index}]"
        item = (
            _slug(entry, entry_path, errors)
            if slug_entries
            else _nonempty_string(entry, entry_path, errors)
        )
        if item:
            result.append(item)
    if len(set(result)) != len(result):
        errors.append(f"'{path}' must not contain duplicate entries")
    return tuple(result)


def _pytest_selector_list(
    value: object,
    path: str,
    errors: list[str],
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    selectors = _string_list(value, path, errors, allow_empty=allow_empty)
    validated: list[str] = []
    for index, selector in enumerate(selectors):
        try:
            validated.append(validate_pytest_selector(selector))
        except GatePolicyError as error:
            errors.append(f"'{path}[{index}]' {error}")
    return tuple(validated)


def _time_limit(value: object, path: str, errors: list[str]) -> int:
    text = _nonempty_string(value, path, errors)
    match = _TIME_RE.fullmatch(text)
    if match is None:
        errors.append(f"'{path}' must use Slurm time format '[D-]HH:MM:SS', got {value!r}")
        return 1
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        errors.append(f"'{path}' has minutes or seconds outside [0, 59]: {text!r}")
        return 1
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if total < 1:
        errors.append(f"'{path}' must be greater than zero")
        return 1
    return total


def _memory_mib(value: object, path: str, errors: list[str]) -> int:
    text = _nonempty_string(value, path, errors)
    match = _MEMORY_RE.fullmatch(text)
    if match is None:
        errors.append(f"'{path}' must use positive MiB/GiB/M/G units, got {value!r}")
        return 1
    amount = int(match.group(1))
    return amount * 1024 if match.group(2) in {"GiB", "G"} else amount


def _parse_repository(data: dict[str, object], errors: list[str]) -> RepositoryConfig:
    root = _existing_path(data.get("root"), "repository.root", "directory", errors)
    base_commit = _nonempty_string(data.get("base_commit"), "repository.base_commit", errors)
    if base_commit and _COMMIT_RE.fullmatch(base_commit) is None:
        errors.append("'repository.base_commit' must be a full lowercase 40-character Git SHA")
    dirty_policy_value = _nonempty_string(
        data.get("dirty_policy"), "repository.dirty_policy", errors
    )
    if dirty_policy_value not in {"reject", "allow"}:
        errors.append("'repository.dirty_policy' must be one of ['reject', 'allow']")
    dirty_policy = cast(_DirtyPolicy, dirty_policy_value or "reject")
    workspace_root = _existing_path(
        data.get("workspace_root"), "repository.workspace_root", "directory", errors
    )
    if root.exists():
        if not (root / ".git").exists():
            errors.append(f"'repository.root' is not a Git worktree: {root}")
        modeling_v2 = root / "tensorrt_llm" / "_torch" / "modeling_v2"
        if not modeling_v2.is_dir():
            errors.append(f"'repository.root' does not contain ModelingV2 at {modeling_v2}")
    return RepositoryConfig(root, base_commit, dirty_policy, workspace_root)


def _parse_reference(data: dict[str, object], errors: list[str]) -> ReferenceConfig:
    checkpoint = _existing_path(data.get("checkpoint"), "reference.checkpoint", "directory", errors)
    provenance = _nonempty_string(data.get("provenance"), "reference.provenance", errors)
    architecture = _nonempty_string(data.get("architecture"), "reference.architecture", errors)
    if architecture and _ARCHITECTURE_RE.fullmatch(architecture) is None:
        errors.append(
            f"'reference.architecture' is not a public architecture identifier: {architecture!r}"
        )

    additional_raw = data.get("additional_sources", [])
    additional_sources: list[Path] = []
    if not isinstance(additional_raw, list):
        errors.append(
            f"'reference.additional_sources' must be a list, got {_type_name(additional_raw)}"
        )
    else:
        for index, entry in enumerate(additional_raw):
            additional_sources.append(
                _existing_path(entry, f"reference.additional_sources[{index}]", "either", errors)
            )

    config_path = checkpoint / "config.json"
    if checkpoint.is_dir() and not config_path.is_file():
        errors.append(
            f"'reference.checkpoint' is missing required Hugging Face config: {config_path}"
        )
    elif config_path.is_file():
        try:
            config_value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"failed to parse checkpoint config '{config_path}': {exc}")
        else:
            architectures = (
                config_value.get("architectures") if isinstance(config_value, dict) else None
            )
            first = architectures[0] if isinstance(architectures, list) and architectures else None
            if not isinstance(first, str):
                errors.append(f"checkpoint config '{config_path}' has no public architectures[0]")
            elif architecture and first != architecture:
                errors.append(
                    f"'reference.architecture' {architecture!r} does not match checkpoint "
                    f"architectures[0] {first!r}"
                )
    return ReferenceConfig(checkpoint, provenance, architecture, tuple(additional_sources))


def _parse_mapping(data: dict[str, object], errors: list[str]) -> ParallelMapping:
    return ParallelMapping(
        tensor_parallel_size=_positive_int(
            data.get("tensor_parallel_size"), "target.mapping.tensor_parallel_size", errors
        ),
        pipeline_parallel_size=_positive_int(
            data.get("pipeline_parallel_size"), "target.mapping.pipeline_parallel_size", errors
        ),
        moe_expert_parallel_size=_positive_int(
            data.get("moe_expert_parallel_size"),
            "target.mapping.moe_expert_parallel_size",
            errors,
        ),
        moe_tensor_parallel_size=_positive_int(
            data.get("moe_tensor_parallel_size"),
            "target.mapping.moe_tensor_parallel_size",
            errors,
        ),
        attention_data_parallel_size=_positive_int(
            data.get("attention_data_parallel_size"),
            "target.mapping.attention_data_parallel_size",
            errors,
        ),
    )


def _relative_modeling_v2_route(
    value: object, family: str, repository_root: Path, errors: list[str]
) -> str:
    path = "target.expected_route"
    text = _nonempty_string(value, path, errors)
    pure_path = PurePosixPath(text)
    if pure_path.is_absolute() or ".." in pure_path.parts or "." in pure_path.parts:
        errors.append(f"'{path}' must be a normalized relative path without traversal: {text!r}")
    expected_prefix = PurePosixPath("tensorrt_llm/_torch/modeling_v2/models")
    if family:
        expected_prefix /= family
    try:
        pure_path.relative_to(expected_prefix)
    except ValueError:
        errors.append(f"'{path}' must be inside '{expected_prefix}', got {text!r}")
    resolved = (repository_root / pure_path).resolve(strict=False)
    try:
        resolved.relative_to(repository_root)
    except ValueError:
        errors.append(f"'{path}' escapes the canonical repository root through a symlink: {text!r}")
    if pure_path.name in {"smoke.py", "TARGET.md"} or "configs" in pure_path.parts:
        errors.append(f"'{path}' selects an obsolete target-local product surface: {text!r}")
    return pure_path.as_posix()


def _parse_target(
    data: dict[str, object], repository: RepositoryConfig, errors: list[str]
) -> TargetConfig:
    family = _slug(data.get("family"), "target.family", errors)
    checkpoint_id = _slug(data.get("checkpoint_id"), "target.checkpoint_id", errors)
    sm = _positive_int(data.get("sm"), "target.sm", errors)
    world_size = _positive_int(data.get("world_size"), "target.world_size", errors)
    mapping_data = _required_mapping(
        data,
        "mapping",
        "target",
        frozenset(
            {
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "moe_expert_parallel_size",
                "moe_tensor_parallel_size",
                "attention_data_parallel_size",
            }
        ),
        frozenset(
            {
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "moe_expert_parallel_size",
                "moe_tensor_parallel_size",
                "attention_data_parallel_size",
            }
        ),
        errors,
    )
    mapping = _parse_mapping(mapping_data, errors)
    features = _string_list(
        data.get("features"), "target.features", errors, allow_empty=True, slug_entries=True
    )
    expected_route = _relative_modeling_v2_route(
        data.get("expected_route"), family, repository.root, errors
    )
    synthetic_target = _strict_bool(data.get("synthetic_target"), "target.synthetic_target", errors)

    expected_world_size = mapping.tensor_parallel_size * mapping.pipeline_parallel_size
    if world_size != expected_world_size:
        errors.append(
            f"'target.world_size' ({world_size}) must equal tensor_parallel_size * "
            f"pipeline_parallel_size ({expected_world_size})"
        )
    moe_width = mapping.moe_expert_parallel_size * mapping.moe_tensor_parallel_size
    if moe_width not in {1, mapping.tensor_parallel_size}:
        errors.append(
            "'target.mapping' MoE EP * MoE TP must be 1 for a non-partitioned MoE axis or "
            f"equal tensor_parallel_size ({mapping.tensor_parallel_size}), got {moe_width}"
        )
    if mapping.tensor_parallel_size % mapping.attention_data_parallel_size != 0:
        errors.append(
            "'target.mapping.attention_data_parallel_size' must divide tensor_parallel_size"
        )
    return TargetConfig(
        family,
        checkpoint_id,
        sm,
        world_size,
        mapping,
        features,
        expected_route,
        synthetic_target,
    )


def _parse_gates(data: dict[str, object], errors: list[str]) -> GatesConfig:
    accuracy_data = _required_mapping(
        data,
        "accuracy",
        "gates",
        frozenset({"selector", "reference", "protocol", "tolerance"}),
        frozenset({"selector", "reference", "protocol", "tolerance"}),
        errors,
    )
    accuracy_selector = _nonempty_string(
        accuracy_data.get("selector"), "gates.accuracy.selector", errors
    )
    if accuracy_selector:
        try:
            accuracy_selector = validate_pytest_selector(accuracy_selector)
        except GatePolicyError as error:
            errors.append(f"'gates.accuracy.selector' {error}")
    accuracy = AccuracyGate(
        selector=accuracy_selector,
        reference=_nonempty_string(
            accuracy_data.get("reference"), "gates.accuracy.reference", errors
        ),
        protocol=_nonempty_string(accuracy_data.get("protocol"), "gates.accuracy.protocol", errors),
        tolerance=_nonnegative_float(
            accuracy_data.get("tolerance"), "gates.accuracy.tolerance", errors
        ),
    )
    return GatesConfig(
        accuracy=accuracy,
        feature_signal_tests=_pytest_selector_list(
            data.get("feature_signal_tests"),
            "gates.feature_signal_tests",
            errors,
            allow_empty=True,
        ),
        boot_tests=_pytest_selector_list(
            data.get("boot_tests"), "gates.boot_tests", errors, allow_empty=False
        ),
        component_tests=_pytest_selector_list(
            data.get("component_tests"),
            "gates.component_tests",
            errors,
            allow_empty=False,
        ),
        collective_tests=_pytest_selector_list(
            data.get("collective_tests"),
            "gates.collective_tests",
            errors,
            allow_empty=True,
        ),
    )


def _parse_expected_product_identity(
    value: object,
    repository: RepositoryConfig,
    target: TargetConfig,
    execution: ExecutionConfig,
    errors: list[str],
) -> ExpectedProductIdentityConfig | None:
    path = "certification.expected_product_identity"
    if value is None:
        return None
    data = _mapping(
        value,
        path,
        frozenset(
            {
                "repository_commit",
                "python_tensorrt_llm_path",
                "native_build_identity",
                "image_identity",
                "compute_capability",
                "collective_backend",
                "transport",
            }
        ),
        frozenset(
            {
                "repository_commit",
                "python_tensorrt_llm_path",
                "native_build_identity",
                "image_identity",
                "compute_capability",
                "collective_backend",
                "transport",
            }
        ),
        errors,
    )
    repository_commit = _nonempty_string(
        data.get("repository_commit"), f"{path}.repository_commit", errors
    )
    if repository_commit and _COMMIT_RE.fullmatch(repository_commit) is None:
        errors.append(f"'{path}.repository_commit' must be a full lowercase Git SHA")
    python_path = _existing_path(
        data.get("python_tensorrt_llm_path"),
        f"{path}.python_tensorrt_llm_path",
        "file",
        errors,
    )
    values = {
        name: _single_line_string(data.get(name), f"{path}.{name}", errors)
        for name in (
            "native_build_identity",
            "image_identity",
            "collective_backend",
            "transport",
        )
    }
    for name, identity in values.items():
        if identity and _IDENTITY_RE.fullmatch(identity) is None:
            errors.append(f"'{path}.{name}' must be a bounded single-line identity")
    compute_capability = _nonempty_string(
        data.get("compute_capability"), f"{path}.compute_capability", errors
    )
    if compute_capability and _COMPUTE_CAPABILITY_RE.fullmatch(compute_capability) is None:
        errors.append(f"'{path}.compute_capability' must use '<major>.<minor>'")
    controller = execution.slurm.controller
    if repository_commit and repository_commit != repository.base_commit:
        errors.append(
            f"'{path}.repository_commit' must equal repository.base_commit before candidate freeze"
        )
    expected_python_path = repository.root / "tensorrt_llm" / "__init__.py"
    if python_path != expected_python_path:
        errors.append(
            f"'{path}.python_tensorrt_llm_path' must name the audited repository package "
            f"at {expected_python_path}"
        )
    if values["native_build_identity"] != controller.build_identity:
        errors.append(
            f"'{path}.native_build_identity' must equal execution.slurm.controller.build_identity"
        )
    if values["image_identity"] != str(controller.image):
        errors.append(f"'{path}.image_identity' must equal execution.slurm.controller.image")
    expected_compute_capability = f"{target.sm // 10}.{target.sm % 10}"
    if compute_capability != expected_compute_capability:
        errors.append(
            f"'{path}.compute_capability' must match target.sm as {expected_compute_capability!r}"
        )
    return ExpectedProductIdentityConfig(
        repository_commit=repository_commit,
        python_tensorrt_llm_path=str(python_path),
        native_build_identity=values["native_build_identity"],
        image_identity=values["image_identity"],
        compute_capability=compute_capability,
        collective_backend=values["collective_backend"],
        transport=values["transport"],
    )


def _parse_certification(
    data: dict[str, object],
    repository: RepositoryConfig,
    target: TargetConfig,
    execution: ExecutionConfig,
    errors: list[str],
) -> CertificationConfig:
    mode_value = _nonempty_string(data.get("mode"), "certification.mode", errors)
    try:
        mode = CertificationMode(mode_value)
    except ValueError:
        errors.append("'certification.mode' must be one of ['LOCAL', 'REAL', 'SYNTHETIC']")
        mode = CertificationMode.LOCAL
    identity = _parse_expected_product_identity(
        data.get("expected_product_identity"), repository, target, execution, errors
    )
    if mode is CertificationMode.REAL and identity is None:
        errors.append("'certification.expected_product_identity' is required for REAL mode")
    if mode is not CertificationMode.REAL and identity is not None:
        errors.append("'certification.expected_product_identity' is permitted only for REAL mode")
    return CertificationConfig(mode, identity)


def _parse_mount(value: object, index: int, errors: list[str]) -> MountConfig:
    path = f"execution.slurm.controller.mounts[{index}]"
    data = _mapping(
        value,
        path,
        frozenset({"host_path", "container_path", "read_only"}),
        frozenset({"host_path", "container_path", "read_only"}),
        errors,
    )
    host_path = _existing_path(data.get("host_path"), f"{path}.host_path", "directory", errors)
    container_path = _nonempty_string(data.get("container_path"), f"{path}.container_path", errors)
    container = PurePosixPath(container_path)
    if not container.is_absolute() or ".." in container.parts:
        errors.append(f"'{path}.container_path' must be an absolute path without traversal")
    read_only = _strict_bool(data.get("read_only"), f"{path}.read_only", errors)
    return MountConfig(host_path, container.as_posix(), read_only)


def _parse_environment(value: object, errors: list[str]) -> tuple[tuple[str, str], ...]:
    path = "execution.slurm.controller.environment"
    if not isinstance(value, dict):
        errors.append(f"'{path}' must be a string-to-string mapping, got {_type_name(value)}")
        return ()
    normalized: list[tuple[str, str]] = []
    for key, entry in value.items():
        if not isinstance(key, str) or _ENVIRONMENT_NAME_RE.fullmatch(key) is None:
            errors.append(f"'{path}' has an invalid environment-variable name: {key!r}")
            continue
        if _CREDENTIAL_ENVIRONMENT_RE.search(key):
            errors.append(
                f"'{path}.{key}' must not persist a credential-like controller environment name"
            )
        if not isinstance(entry, str):
            errors.append(f"'{path}.{key}' must be a string, got {_type_name(entry)}")
            continue
        _validate_environment_value(
            entry,
            f"{path}.{key}",
            "controller",
            errors,
            name=key,
        )
        if any(character in entry for character in ("\x00", "\n", "\r")):
            errors.append(f"'{path}.{key}' must be a single-line value")
        normalized.append((key, entry))
    return tuple(sorted(normalized))


def _parse_controller(data: dict[str, object], errors: list[str]) -> ControllerConfig:
    account = _slurm_name(data.get("account"), "execution.slurm.controller.account", errors)
    partition = _slurm_name(data.get("partition"), "execution.slurm.controller.partition", errors)
    qos = _optional_string(data.get("qos"), "execution.slurm.controller.qos", errors)
    if qos is not None and _SLURM_NAME_RE.fullmatch(qos) is None:
        errors.append("'execution.slurm.controller.qos' contains unsafe characters")
    reservation = _optional_string(
        data.get("reservation"), "execution.slurm.controller.reservation", errors
    )
    if reservation is not None and _SLURM_NAME_RE.fullmatch(reservation) is None:
        errors.append("'execution.slurm.controller.reservation' contains unsafe characters")
    mounts_raw = data.get("mounts")
    mounts: list[MountConfig] = []
    if not isinstance(mounts_raw, list):
        errors.append(
            "'execution.slurm.controller.mounts' must be a non-empty list of mount mappings"
        )
    else:
        if not mounts_raw:
            errors.append("'execution.slurm.controller.mounts' must not be empty")
        for index, entry in enumerate(mounts_raw):
            mounts.append(_parse_mount(entry, index, errors))

    dispatch_value = _nonempty_string(
        data.get("dispatch_mode"), "execution.slurm.controller.dispatch_mode", errors
    )
    if dispatch_value != "nested_submission":
        errors.append(
            "'execution.slurm.controller.dispatch_mode' must be 'nested_submission'; "
            "dispatcher and pool adapters are not admitted"
        )
    heartbeat = _positive_int(
        data.get("heartbeat_timeout_seconds"),
        "execution.slurm.controller.heartbeat_timeout_seconds",
        errors,
    )
    lease = _positive_int(
        data.get("lease_timeout_seconds"),
        "execution.slurm.controller.lease_timeout_seconds",
        errors,
    )
    if lease <= heartbeat:
        errors.append(
            "'execution.slurm.controller.lease_timeout_seconds' must be greater than "
            "heartbeat_timeout_seconds"
        )
    credential_broker_socket: str | None = None
    raw_broker_socket = data.get("credential_broker_socket")
    if raw_broker_socket is not None:
        socket_path = _single_line_string(
            raw_broker_socket,
            "execution.slurm.controller.credential_broker_socket",
            errors,
        )
        pure_socket_path = PurePosixPath(socket_path)
        if (
            not pure_socket_path.is_absolute()
            or ".." in pure_socket_path.parts
            or pure_socket_path.as_posix() != socket_path
        ):
            errors.append(
                "'execution.slurm.controller.credential_broker_socket' must be a "
                "normalized absolute container path without traversal"
            )
        else:
            credential_broker_socket = pure_socket_path.as_posix()
    scheduler_clients = _strict_bool(
        data.get("scheduler_clients"),
        "execution.slurm.controller.scheduler_clients",
        errors,
    )
    if not scheduler_clients:
        errors.append(
            "'execution.slurm.controller.scheduler_clients' must be true for nested submission"
        )
    launch_mode_value = _nonempty_string(
        data.get("container_launch_mode"),
        "execution.slurm.controller.container_launch_mode",
        errors,
    )
    try:
        launch_mode = ContainerLaunchMode(launch_mode_value)
    except ValueError:
        errors.append(
            "'execution.slurm.controller.container_launch_mode' must be "
            f"{ContainerLaunchMode.IN_ALLOCATION_SRUN.value!r}"
        )
        launch_mode = ContainerLaunchMode.IN_ALLOCATION_SRUN
    return ControllerConfig(
        account=account,
        partition=partition,
        qos=qos,
        reservation=reservation,
        time_limit_seconds=_time_limit(
            data.get("time_limit"), "execution.slurm.controller.time_limit", errors
        ),
        cpus_per_task=_positive_int(
            data.get("cpus_per_task"), "execution.slurm.controller.cpus_per_task", errors
        ),
        memory_mib=_memory_mib(data.get("memory"), "execution.slurm.controller.memory", errors),
        image=_existing_path(data.get("image"), "execution.slurm.controller.image", "file", errors),
        scheduler_clients=scheduler_clients,
        container_launch_mode=launch_mode,
        mounts=tuple(mounts),
        environment=_parse_environment(data.get("environment"), errors),
        build_identity=_nonempty_string(
            data.get("build_identity"), "execution.slurm.controller.build_identity", errors
        ),
        dispatch_mode=cast(_DispatchMode, dispatch_value or "nested_submission"),
        requeue=_strict_bool(data.get("requeue"), "execution.slurm.controller.requeue", errors),
        advance_signal_lead_seconds=_nonnegative_int(
            data.get("advance_signal_lead_seconds"),
            "execution.slurm.controller.advance_signal_lead_seconds",
            errors,
        ),
        heartbeat_timeout_seconds=heartbeat,
        lease_timeout_seconds=lease,
        orphan_grace_seconds=_positive_int(
            data.get("orphan_grace_seconds"),
            "execution.slurm.controller.orphan_grace_seconds",
            errors,
        ),
        credential_broker_socket=credential_broker_socket,
    )


def _parse_agent_worker(data: dict[str, object], errors: list[str]) -> AgentWorkerConfig:
    path = "execution.slurm.agent_worker"
    scheduler_clients = _strict_bool(
        data.get("scheduler_clients"), f"{path}.scheduler_clients", errors
    )
    if scheduler_clients:
        errors.append(f"'{path}.scheduler_clients' must be false")
    enforcement_value = _nonempty_string(
        data.get("network_enforcement"), f"{path}.network_enforcement", errors
    )
    try:
        enforcement = NetworkEnforcement(enforcement_value)
    except ValueError:
        errors.append(
            f"'{path}.network_enforcement' must be "
            f"{NetworkEnforcement.CERTIFIED_WORKER_IMAGE.value!r}"
        )
        enforcement = NetworkEnforcement.CERTIFIED_WORKER_IMAGE
    return AgentWorkerConfig(
        image=_existing_path(data.get("image"), f"{path}.image", "file", errors),
        build_identity=_single_line_string(
            data.get("build_identity"), f"{path}.build_identity", errors
        ),
        scheduler_clients=scheduler_clients,
        network_enforcement=enforcement,
    )


def _parse_resource_class(name: str, value: object, errors: list[str]) -> ResourceClass:
    path = f"execution.slurm.smith.resource_classes.{name}"
    data = _mapping(
        value,
        path,
        frozenset(
            {
                "nodes",
                "tasks_per_node",
                "gpus_per_node",
                "cpus_per_task",
                "memory",
                "time_limit",
            }
        ),
        frozenset(
            {
                "nodes",
                "tasks_per_node",
                "gpus_per_node",
                "cpus_per_task",
                "memory",
                "time_limit",
            }
        ),
        errors,
    )
    return ResourceClass(
        name=name,
        nodes=_positive_int(data.get("nodes"), f"{path}.nodes", errors),
        tasks_per_node=_positive_int(data.get("tasks_per_node"), f"{path}.tasks_per_node", errors),
        gpus_per_node=_nonnegative_int(data.get("gpus_per_node"), f"{path}.gpus_per_node", errors),
        cpus_per_task=_positive_int(data.get("cpus_per_task"), f"{path}.cpus_per_task", errors),
        memory_mib=_memory_mib(data.get("memory"), f"{path}.memory", errors),
        time_limit_seconds=_time_limit(data.get("time_limit"), f"{path}.time_limit", errors),
    )


def _parse_override_bounds(data: dict[str, object], errors: list[str]) -> OverrideBounds:
    path = "execution.slurm.smith.per_item_override_bounds"
    return OverrideBounds(
        max_nodes=_positive_int(data.get("max_nodes"), f"{path}.max_nodes", errors),
        max_tasks_per_node=_positive_int(
            data.get("max_tasks_per_node"), f"{path}.max_tasks_per_node", errors
        ),
        max_gpus_per_node=_nonnegative_int(
            data.get("max_gpus_per_node"), f"{path}.max_gpus_per_node", errors
        ),
        max_cpus_per_task=_positive_int(
            data.get("max_cpus_per_task"), f"{path}.max_cpus_per_task", errors
        ),
        max_memory_mib=_memory_mib(data.get("max_memory"), f"{path}.max_memory", errors),
        max_time_limit_seconds=_time_limit(
            data.get("max_time_limit"), f"{path}.max_time_limit", errors
        ),
    )


def _parse_smith(data: dict[str, object], target: TargetConfig, errors: list[str]) -> SmithConfig:
    max_parallel_items = _positive_int(
        data.get("max_parallel_items"),
        "execution.slurm.smith.max_parallel_items",
        errors,
    )
    max_nodes_total = _positive_int(
        data.get("max_nodes_total"), "execution.slurm.smith.max_nodes_total", errors
    )
    max_gpus_total = _positive_int(
        data.get("max_gpus_total"), "execution.slurm.smith.max_gpus_total", errors
    )
    resource_data = _required_mapping(
        data,
        "resource_classes",
        "execution.slurm.smith",
        frozenset(_RESOURCE_CLASS_NAMES),
        frozenset(_RESOURCE_CLASS_NAMES),
        errors,
    )
    resources = tuple(
        _parse_resource_class(name, resource_data.get(name), errors)
        for name in _RESOURCE_CLASS_NAMES
    )
    bounds_data = _required_mapping(
        data,
        "per_item_override_bounds",
        "execution.slurm.smith",
        frozenset(
            {
                "max_nodes",
                "max_tasks_per_node",
                "max_gpus_per_node",
                "max_cpus_per_task",
                "max_memory",
                "max_time_limit",
            }
        ),
        frozenset(
            {
                "max_nodes",
                "max_tasks_per_node",
                "max_gpus_per_node",
                "max_cpus_per_task",
                "max_memory",
                "max_time_limit",
            }
        ),
        errors,
    )
    bounds = _parse_override_bounds(bounds_data, errors)
    retry_data = _required_mapping(
        data,
        "retry_policy",
        "execution.slurm.smith",
        frozenset({"preempted", "node_failure"}),
        frozenset({"preempted", "node_failure"}),
        errors,
    )
    retry_policy = RetryPolicy(
        preempted=_nonnegative_int(
            retry_data.get("preempted"),
            "execution.slurm.smith.retry_policy.preempted",
            errors,
        ),
        node_failure=_nonnegative_int(
            retry_data.get("node_failure"),
            "execution.slurm.smith.retry_policy.node_failure",
            errors,
        ),
    )
    distinct_nodes_required = _strict_bool(
        data.get("distinct_nodes_required"),
        "execution.slurm.smith.distinct_nodes_required",
        errors,
    )
    exclusive = _strict_bool(
        data.get("exclusive"),
        "execution.slurm.smith.exclusive",
        errors,
    )
    if distinct_nodes_required and not exclusive:
        errors.append(
            "'execution.slurm.smith.exclusive' must be true when distinct_nodes_required is true"
        )

    if max_parallel_items > max_nodes_total:
        errors.append("'execution.slurm.smith.max_parallel_items' must not exceed max_nodes_total")
    if bounds.max_nodes > max_nodes_total:
        errors.append(
            "'execution.slurm.smith.per_item_override_bounds.max_nodes' exceeds max_nodes_total"
        )
    if bounds.max_nodes * bounds.max_gpus_per_node > max_gpus_total:
        errors.append(
            "'execution.slurm.smith.per_item_override_bounds' GPU request exceeds max_gpus_total"
        )
    for resource in resources:
        prefix = f"execution.slurm.smith.resource_classes.{resource.name}"
        if resource.nodes > max_nodes_total:
            errors.append(f"'{prefix}.nodes' exceeds max_nodes_total")
        if resource.total_gpus > max_gpus_total:
            errors.append(f"'{prefix}' aggregate GPUs exceed max_gpus_total")
        if resource.nodes > bounds.max_nodes:
            errors.append(f"'{prefix}.nodes' exceeds per-item override bounds")
        if resource.tasks_per_node > bounds.max_tasks_per_node:
            errors.append(f"'{prefix}.tasks_per_node' exceeds per-item override bounds")
        if resource.gpus_per_node > bounds.max_gpus_per_node:
            errors.append(f"'{prefix}.gpus_per_node' exceeds per-item override bounds")
        if resource.cpus_per_task > bounds.max_cpus_per_task:
            errors.append(f"'{prefix}.cpus_per_task' exceeds per-item override bounds")
        if resource.memory_mib > bounds.max_memory_mib:
            errors.append(f"'{prefix}.memory' exceeds per-item override bounds")
        if resource.time_limit_seconds > bounds.max_time_limit_seconds:
            errors.append(f"'{prefix}.time_limit' exceeds per-item override bounds")

    resource_by_name = {resource.name: resource for resource in resources}
    role_class_names = set(_RESOURCE_CLASS_NAMES) - {"deterministic_gate"}
    for name in sorted(role_class_names):
        resource = resource_by_name[name]
        if resource.nodes != 1 or resource.tasks_per_node != 1:
            errors.append(
                f"'execution.slurm.smith.resource_classes.{name}' must be a "
                "single-node, single-task role class"
            )
    resource = resource_by_name["deterministic_gate"]
    name = resource.name
    if resource.total_tasks != target.world_size or resource.total_gpus != target.world_size:
        errors.append(
            f"'execution.slurm.smith.resource_classes.{name}' must request exactly "
            f"target.world_size={target.world_size} tasks and GPUs, got "
            f"tasks={resource.total_tasks}, GPUs={resource.total_gpus}"
        )
    if resource.tasks_per_node > resource.gpus_per_node:
        errors.append(
            f"'execution.slurm.smith.resource_classes.{name}' cannot place more ranks "
            "than GPUs per node"
        )
    return SmithConfig(
        max_parallel_items,
        max_nodes_total,
        max_gpus_total,
        resources,
        bounds,
        retry_policy,
        distinct_nodes_required,
        exclusive,
    )


def _parse_gate_classes(value: object, errors: list[str]) -> tuple[GateClass, ...]:
    path = "execution.slurm.gate_classes"
    if not isinstance(value, dict):
        errors.append(f"'{path}' must be a mapping, got {_type_name(value)}")
        return ()
    result: list[GateClass] = []
    for gate_id_value, gate_value in value.items():
        if not isinstance(gate_id_value, str):
            errors.append(f"'{path}' has a non-string gate ID {gate_id_value!r}")
            continue
        gate_id = _slug(gate_id_value, f"{path}.{gate_id_value}", errors)
        data = _mapping(
            gate_value,
            f"{path}.{gate_id_value}",
            frozenset({"resource_class"}),
            frozenset({"resource_class"}),
            errors,
        )
        resource_class = _nonempty_string(
            data.get("resource_class"), f"{path}.{gate_id_value}.resource_class", errors
        )
        if resource_class != "deterministic_gate":
            errors.append(f"'{path}.{gate_id_value}.resource_class' must be 'deterministic_gate'")
        if gate_id:
            result.append(GateClass(gate_id, resource_class))
    if len({entry.gate_id for entry in result}) != len(result):
        errors.append(f"'{path}' contains duplicate normalized gate IDs")
    return tuple(sorted(result, key=lambda entry: entry.gate_id))


def _validate_role_environment_name(name: object, path: str, errors: list[str]) -> str:
    if not isinstance(name, str) or _ENVIRONMENT_NAME_RE.fullmatch(name) is None:
        errors.append(f"'{path}' must be a valid environment-variable name, got {name!r}")
        return ""
    if _SCHEDULER_ENVIRONMENT_RE.search(name):
        errors.append(f"'{path}' must not admit scheduler-owned environment {name!r}")
    if _CREDENTIAL_ENVIRONMENT_RE.search(name):
        errors.append(f"'{path}' must not admit credential-like environment {name!r}")
    return name


def _validate_environment_value(
    value: str,
    path: str,
    owner: Literal["controller", "role"],
    errors: list[str],
    *,
    name: str,
) -> None:
    """Reject secret-shaped durable values without including the value in diagnostics."""
    invalid_path = False
    if name in _PATH_ENVIRONMENT_NAMES:
        pure_path = PurePosixPath(value)
        invalid_path = (
            not pure_path.is_absolute()
            or ".." in pure_path.parts
            or "=" in value
            or _URL_USERINFO_RE.search(value) is not None
        )
        if invalid_path:
            errors.append(f"'{path}' must be an absolute non-secret path value")
    if (
        _CREDENTIAL_VALUE_RE.search(value)
        or _URL_USERINFO_RE.search(value)
        or _NESTED_CREDENTIAL_ASSIGNMENT_RE.search(value)
        or (not invalid_path and _looks_like_opaque_token(value))
    ):
        errors.append(f"'{path}' must not persist a credential-like {owner} environment value")


def _looks_like_opaque_token(value: str) -> bool:
    """Identify bounded high-entropy ASCII tokens while preserving ordinary paths/text."""
    if (
        _OPAQUE_TOKEN_RE.fullmatch(value) is None
        or value.startswith(("/", "./", "../"))
        or len(value) > _MAX_ROLE_ENVIRONMENT_VALUE_BYTES
    ):
        return False
    counts = Counter(value)
    entropy = -sum(
        (count / len(value)) * math.log2(count / len(value)) for count in counts.values()
    )
    character_classes = sum(
        (
            any(character.islower() for character in value),
            any(character.isupper() for character in value),
            any(character.isdigit() for character in value),
            any(character in "._~+/-" for character in value),
        )
    )
    return entropy >= 3.5 and character_classes >= 3


def _parse_role_environment(
    value: object,
    path: str,
    errors: list[str],
) -> RoleEnvironmentPolicy:
    data = _mapping(
        value,
        path,
        frozenset({"allowlist", "values"}),
        frozenset({"allowlist", "values"}),
        errors,
    )
    raw_allowlist = data.get("allowlist")
    allowlist: list[str] = []
    if not isinstance(raw_allowlist, list):
        errors.append(f"'{path}.allowlist' must be a list, got {_type_name(raw_allowlist)}")
    else:
        for index, name in enumerate(raw_allowlist):
            normalized = _validate_role_environment_name(name, f"{path}.allowlist[{index}]", errors)
            if normalized:
                allowlist.append(normalized)
        if len(set(allowlist)) != len(allowlist):
            errors.append(f"'{path}.allowlist' must not contain duplicate names")

    raw_values = data.get("values")
    values: list[tuple[str, str]] = []
    if not isinstance(raw_values, dict):
        errors.append(f"'{path}.values' must be a string-to-string mapping")
    else:
        for raw_name, raw_value in raw_values.items():
            name = _validate_role_environment_name(raw_name, f"{path}.values.{raw_name}", errors)
            if not isinstance(raw_value, str):
                errors.append(
                    f"'{path}.values.{raw_name}' must be a string, got {_type_name(raw_value)}"
                )
                continue
            try:
                encoded = raw_value.encode("utf-8")
            except UnicodeEncodeError:
                errors.append(f"'{path}.values.{raw_name}' must be valid UTF-8")
                continue
            if any(character in raw_value for character in ("\x00", "\n", "\r")):
                errors.append(f"'{path}.values.{raw_name}' must be a single-line value")
            if len(encoded) > _MAX_ROLE_ENVIRONMENT_VALUE_BYTES:
                errors.append(
                    f"'{path}.values.{raw_name}' exceeds "
                    f"{_MAX_ROLE_ENVIRONMENT_VALUE_BYTES} UTF-8 bytes"
                )
            _validate_environment_value(
                raw_value,
                f"{path}.values.{raw_name}",
                "role",
                errors,
                name=name,
            )
            if name:
                values.append((name, raw_value))
    overlap = set(allowlist).intersection(name for name, _value in values)
    if overlap:
        errors.append(
            f"'{path}' names must appear in either allowlist or values, not both: "
            f"{sorted(overlap)!r}"
        )
    return RoleEnvironmentPolicy(tuple(sorted(allowlist)), tuple(sorted(values)))


def _parse_credential_broker_policy(
    value: object,
    path: str,
    backend_kind: _AgentBackendKind,
    errors: list[str],
) -> CredentialBrokerPolicy:
    data = _mapping(
        value,
        path,
        frozenset({"broker_id", "allowed_credential_names", "per_attempt_ttl_seconds"}),
        frozenset({"broker_id", "allowed_credential_names", "per_attempt_ttl_seconds"}),
        errors,
    )
    broker_id = _slurm_name(data.get("broker_id"), f"{path}.broker_id", errors)
    if len(broker_id) > 128:
        errors.append(f"'{path}.broker_id' must not exceed 128 characters")
    if broker_id == "legacy-environment":
        errors.append(f"'{path}.broker_id' must identify a per-attempt credential broker")
    raw_names = data.get("allowed_credential_names")
    names: list[str] = []
    if not isinstance(raw_names, list):
        errors.append(
            f"'{path}.allowed_credential_names' must be a list, got {_type_name(raw_names)}"
        )
    else:
        for index, raw_name in enumerate(raw_names):
            if not isinstance(raw_name, str) or _ENVIRONMENT_NAME_RE.fullmatch(raw_name) is None:
                errors.append(
                    f"'{path}.allowed_credential_names[{index}]' must be a valid "
                    f"credential name, got {raw_name!r}"
                )
                continue
            names.append(raw_name)
        if len(set(names)) != len(names):
            errors.append(f"'{path}.allowed_credential_names' must not contain duplicates")
    names = sorted(names)
    allowed = BACKEND_CREDENTIAL_ALLOWLIST.get(backend_kind, frozenset())
    unsupported = sorted(set(names) - allowed)
    if unsupported:
        errors.append(
            f"'{path}.allowed_credential_names' is inconsistent with backend "
            f"{backend_kind!r}: {unsupported!r}"
        )
    ttl = _positive_int(
        data.get("per_attempt_ttl_seconds"),
        f"{path}.per_attempt_ttl_seconds",
        errors,
    )
    if ttl > _MAX_ROLE_CREDENTIAL_TTL_SECONDS:
        errors.append(
            f"'{path}.per_attempt_ttl_seconds' must not exceed {_MAX_ROLE_CREDENTIAL_TTL_SECONDS}"
        )
    return CredentialBrokerPolicy(broker_id, tuple(names), ttl)


def _parse_role_classes(
    value: object,
    smith: SmithConfig,
    backend_kind: _AgentBackendKind,
    errors: list[str],
) -> tuple[SlurmRoleClass, ...]:
    path = "execution.slurm.role_classes"
    data = _mapping(
        value,
        path,
        frozenset(_ROLE_CLASS_NAMES),
        frozenset(_ROLE_CLASS_NAMES),
        errors,
    )
    resource_by_name = {resource.name: resource for resource in smith.resource_classes}
    result: list[SlurmRoleClass] = []
    for role_name in _ROLE_CLASS_NAMES:
        role_path = f"{path}.{role_name}"
        role_data = _mapping(
            data.get(role_name),
            role_path,
            frozenset(
                {
                    "resource_class",
                    "network_policy",
                    "environment",
                    "credential_broker",
                }
            ),
            frozenset(
                {
                    "resource_class",
                    "network_policy",
                    "environment",
                    "credential_broker",
                }
            ),
            errors,
        )
        resource_name = _nonempty_string(
            role_data.get("resource_class"), f"{role_path}.resource_class", errors
        )
        resource = resource_by_name.get(resource_name)
        if resource is None:
            errors.append(
                f"'{role_path}.resource_class' must name an existing Smith resource class"
            )
        elif resource.nodes != 1 or resource.tasks_per_node != 1:
            errors.append(
                f"'{role_path}.resource_class' must bind a single-node, single-task resource class"
            )
        network_value = _nonempty_string(
            role_data.get("network_policy"), f"{role_path}.network_policy", errors
        )
        try:
            network_policy = RoleNetworkPolicy(network_value)
        except ValueError:
            errors.append(
                f"'{role_path}.network_policy' must be {RoleNetworkPolicy.BACKEND_API_ONLY.value!r}"
            )
            network_policy = RoleNetworkPolicy.BACKEND_API_ONLY
        environment = _parse_role_environment(
            role_data.get("environment"), f"{role_path}.environment", errors
        )
        broker = _parse_credential_broker_policy(
            role_data.get("credential_broker"),
            f"{role_path}.credential_broker",
            backend_kind,
            errors,
        )
        result.append(
            SlurmRoleClass(
                SlurmRole(role_name),
                resource_name,
                network_policy,
                environment,
                broker,
            )
        )
    return tuple(result)


def _validate_credential_broker_deployment(
    controller: ControllerConfig,
    role_classes: tuple[SlurmRoleClass, ...],
    errors: list[str],
) -> None:
    """Validate one closed broker mode shared by every role invocation."""
    policies = tuple(role_class.credential_broker for role_class in role_classes)
    if not policies:
        return
    credentialed = any(policy.allowed_credential_names for policy in policies)
    socket_path = controller.credential_broker_socket
    if credentialed:
        if socket_path is None:
            errors.append(
                "'execution.slurm.controller.credential_broker_socket' is required "
                "when any role requests credentials"
            )
        first = policies[0]
        if any(policy != first for policy in policies[1:]):
            errors.append(
                "all execution.slurm.role_classes credential_broker policies must be "
                "byte-identical after normalization when credentials are requested"
            )
        if first.broker_id == "preauthenticated":
            errors.append("credentialed role policies must not use broker_id 'preauthenticated'")
        return

    if socket_path is not None:
        errors.append(
            "'execution.slurm.controller.credential_broker_socket' must be omitted "
            "when all roles are preauthenticated"
        )
    for role_class in role_classes:
        policy = role_class.credential_broker
        if policy.broker_id != "preauthenticated":
            errors.append(
                f"'execution.slurm.role_classes.{role_class.role.value}.credential_broker."
                "broker_id' must be 'preauthenticated' when no credentials are requested"
            )


def _parse_execution(
    data: dict[str, object], target: TargetConfig, errors: list[str]
) -> ExecutionConfig:
    mode_value = _nonempty_string(data.get("mode"), "execution.mode", errors)
    if mode_value != "slurm":
        errors.append("'execution.mode' must be 'slurm'; local execution is a CLI test override")
    agent_data = _required_mapping(
        data,
        "agent",
        "execution",
        frozenset({"backend_kind", "model"}),
        frozenset({"backend_kind", "model"}),
        errors,
    )
    slurm_data = _required_mapping(
        data,
        "slurm",
        "execution",
        frozenset({"controller", "agent_worker", "smith", "gate_classes", "role_classes"}),
        frozenset({"controller", "agent_worker", "smith", "gate_classes", "role_classes"}),
        errors,
    )
    controller_data = _required_mapping(
        slurm_data,
        "controller",
        "execution.slurm",
        frozenset(
            {
                "account",
                "partition",
                "qos",
                "reservation",
                "time_limit",
                "cpus_per_task",
                "memory",
                "image",
                "scheduler_clients",
                "container_launch_mode",
                "mounts",
                "environment",
                "build_identity",
                "dispatch_mode",
                "requeue",
                "advance_signal_lead_seconds",
                "heartbeat_timeout_seconds",
                "lease_timeout_seconds",
                "orphan_grace_seconds",
                "credential_broker_socket",
            }
        ),
        frozenset(
            {
                "account",
                "partition",
                "time_limit",
                "cpus_per_task",
                "memory",
                "image",
                "scheduler_clients",
                "container_launch_mode",
                "mounts",
                "environment",
                "build_identity",
                "dispatch_mode",
                "requeue",
                "advance_signal_lead_seconds",
                "heartbeat_timeout_seconds",
                "lease_timeout_seconds",
                "orphan_grace_seconds",
            }
        ),
        errors,
    )
    agent_worker_data = _required_mapping(
        slurm_data,
        "agent_worker",
        "execution.slurm",
        frozenset({"image", "build_identity", "scheduler_clients", "network_enforcement"}),
        frozenset({"image", "build_identity", "scheduler_clients", "network_enforcement"}),
        errors,
    )
    smith_data = _required_mapping(
        slurm_data,
        "smith",
        "execution.slurm",
        frozenset(
            {
                "max_parallel_items",
                "max_nodes_total",
                "max_gpus_total",
                "resource_classes",
                "per_item_override_bounds",
                "retry_policy",
                "distinct_nodes_required",
                "exclusive",
            }
        ),
        frozenset(
            {
                "max_parallel_items",
                "max_nodes_total",
                "max_gpus_total",
                "resource_classes",
                "per_item_override_bounds",
                "retry_policy",
                "distinct_nodes_required",
                "exclusive",
            }
        ),
        errors,
    )
    agent = _parse_agent_execution(agent_data, errors)
    controller = _parse_controller(controller_data, errors)
    agent_worker = _parse_agent_worker(agent_worker_data, errors)
    if controller.image == agent_worker.image:
        errors.append(
            "'execution.slurm.agent_worker.image' must differ from the scheduler-capable "
            "controller/gate image"
        )
    if controller.build_identity == agent_worker.build_identity:
        errors.append(
            "'execution.slurm.agent_worker.build_identity' must differ from the "
            "scheduler-capable controller/gate build identity"
        )
    smith = _parse_smith(smith_data, target, errors)
    role_classes = _parse_role_classes(
        slurm_data.get("role_classes"), smith, agent.backend_kind, errors
    )
    _validate_credential_broker_deployment(controller, role_classes, errors)
    return ExecutionConfig(
        cast(_ExecutionMode, mode_value or "slurm"),
        SlurmConfig(
            controller=controller,
            smith=smith,
            gate_classes=_parse_gate_classes(slurm_data.get("gate_classes"), errors),
            role_classes=role_classes,
            agent_worker=agent_worker,
        ),
        agent,
    )


def _parse_agent_execution(data: dict[str, object], errors: list[str]) -> AgentExecutionConfig:
    backend_value = _nonempty_string(
        data.get("backend_kind"), "execution.agent.backend_kind", errors
    )
    if backend_value not in {"codex", "claude-code"}:
        errors.append("'execution.agent.backend_kind' must be one of ['claude-code', 'codex']")
    model = _single_line_string(data.get("model"), "execution.agent.model", errors)
    return AgentExecutionConfig(cast(_AgentBackendKind, backend_value or "codex"), model)


def _validate_shared_mounts(
    repository: RepositoryConfig,
    reference: ReferenceConfig,
    execution: ExecutionConfig,
    errors: list[str],
) -> None:
    writable_required = (repository.root, repository.workspace_root)
    visible_required = writable_required + (reference.checkpoint,) + reference.additional_sources
    mounts = execution.slurm.controller.mounts
    for index, mount in enumerate(mounts):
        if mount.container_path != mount.host_path.as_posix():
            errors.append(
                "'execution.slurm.controller.mounts["
                f"{index}].container_path' must equal its canonical host_path "
                "for the identity-mount controller contract"
            )
    for required_path in visible_required:
        covering = [
            mount
            for mount in mounts
            if required_path == mount.host_path or required_path.is_relative_to(mount.host_path)
        ]
        if not covering:
            errors.append(
                f"path '{required_path}' is not visible through any execution.slurm.controller mount"
            )
            continue
        if required_path in writable_required and all(mount.read_only for mount in covering):
            errors.append(f"path '{required_path}' is only covered by read-only controller mounts")


def _validate_gate_classes(
    certification: CertificationConfig,
    gates: GatesConfig,
    target: TargetConfig,
    execution: ExecutionConfig,
    errors: list[str],
) -> None:
    actual = {entry.gate_id for entry in execution.slurm.gate_classes}
    if certification.mode is CertificationMode.SYNTHETIC:
        expected = {"collective"}
        if not gates.collective_tests:
            errors.append(
                "SYNTHETIC certification requires at least one gates.collective_tests selector"
            )
    elif certification.mode is CertificationMode.REAL and target.world_size > 1:
        expected = {"boot", "accuracy"}
        if gates.collective_tests:
            expected.add("collective")
        if gates.feature_signal_tests:
            expected.add("feature-signal")
    else:
        # LOCAL is an explicit CLI override over a complete production task;
        # retained Slurm gate bindings are inert but remain part of the digest.
        return
    if actual != expected:
        errors.append(
            "'execution.slurm.gate_classes' must exactly bind the trusted multi-rank "
            f"gates for {certification.mode.value}: expected={sorted(expected)!r}, "
            f"observed={sorted(actual)!r}"
        )


def _parse_delivery(data: dict[str, object], errors: list[str]) -> DeliveryConfig:
    mode_value = _nonempty_string(data.get("mode"), "delivery.mode", errors)
    if mode_value not in {"diff_only", "signed_commits"}:
        errors.append("'delivery.mode' must be one of ['diff_only', 'signed_commits']")
    branch = _optional_string(data.get("branch"), "delivery.branch", errors)
    if branch is not None:
        pure_branch = PurePosixPath(branch)
        if pure_branch.is_absolute() or ".." in pure_branch.parts or branch.endswith("/"):
            errors.append("'delivery.branch' is not a safe relative Git branch name")
    if mode_value == "signed_commits" and branch is None:
        errors.append("'delivery.branch' is required when delivery.mode is 'signed_commits'")
    if mode_value == "diff_only" and branch is not None:
        errors.append("'delivery.branch' must be omitted when delivery.mode is 'diff_only'")
    merge = _strict_bool(data.get("merge"), "delivery.merge", errors)
    push = _strict_bool(data.get("push"), "delivery.push", errors)
    if merge:
        errors.append("'delivery.merge' must be false; merge requires separate authorization")
    if push:
        errors.append("'delivery.push' must be false; push requires separate authorization")
    return DeliveryConfig(cast(_DeliveryMode, mode_value or "diff_only"), branch, merge, push)


def _scan_obsolete_tokens(
    value: object, path: str, errors: list[str], seen: set[int] | None = None
) -> None:
    if seen is None:
        seen = set()
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
    if isinstance(value, str):
        for token in _OBSOLETE_DATA_PLANE_TOKENS:
            if token in value:
                errors.append(f"'{path}' contains obsolete data-plane token {token!r}")
        return
    if isinstance(value, list):
        for index, entry in enumerate(value):
            _scan_obsolete_tokens(entry, f"{path}[{index}]", errors, seen)
        return
    if isinstance(value, dict):
        for key, entry in value.items():
            _scan_obsolete_tokens(entry, f"{path}.{key}", errors, seen)


def _task_payload(task: NormalizedTask) -> dict[str, object]:
    payload = asdict(task)
    payload.pop("digest", None)

    def normalize(value: object) -> object:
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, tuple):
            return [normalize(entry) for entry in value]
        if isinstance(value, list):
            return [normalize(entry) for entry in value]
        if isinstance(value, dict):
            return {str(key): normalize(entry) for key, entry in value.items()}
        return value

    return cast(dict[str, object], normalize(payload))


def normalized_task_dict(task: NormalizedTask, *, include_digest: bool = True) -> dict[str, object]:
    """Return a JSON-safe copy of a normalized task.

    Args:
        task: Validated immutable task.
        include_digest: Whether to include the derived digest field.

    Returns:
        A newly allocated dictionary containing canonical normalized values.
    """
    payload = _task_payload(task)
    if include_digest:
        payload["digest"] = task.digest
    return payload


def normalized_task_json(task: NormalizedTask, *, include_digest: bool = True) -> str:
    """Serialize a normalized task deterministically as compact JSON.

    Args:
        task: Validated immutable task.
        include_digest: Whether to include the derived digest field.

    Returns:
        Canonical JSON with sorted keys and no non-finite numbers.
    """
    return json.dumps(
        normalized_task_dict(task, include_digest=include_digest),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _format_time_limit(seconds: int) -> str:
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    time = f"{hours:02d}:{minutes:02d}:{final_seconds:02d}"
    return f"{days}-{time}" if days else time


def _source_task_dict(task: NormalizedTask) -> dict[str, object]:
    controller = task.execution.slurm.controller
    agent_worker = task.execution.slurm.agent_worker
    if agent_worker is None:
        raise TaskSchemaError("normalized production task is missing agent_worker")
    smith = task.execution.slurm.smith
    delivery: dict[str, object] = {
        "mode": task.delivery.mode,
        "merge": task.delivery.merge,
        "push": task.delivery.push,
    }
    if task.delivery.branch is not None:
        delivery["branch"] = task.delivery.branch
    return {
        "schema_version": task.schema_version,
        "repository": {
            "root": task.repository.root.as_posix(),
            "base_commit": task.repository.base_commit,
            "dirty_policy": task.repository.dirty_policy,
            "workspace_root": task.repository.workspace_root.as_posix(),
        },
        "reference": {
            "checkpoint": task.reference.checkpoint.as_posix(),
            "provenance": task.reference.provenance,
            "architecture": task.reference.architecture,
            "additional_sources": [path.as_posix() for path in task.reference.additional_sources],
        },
        "target": {
            "family": task.target.family,
            "checkpoint_id": task.target.checkpoint_id,
            "sm": task.target.sm,
            "world_size": task.target.world_size,
            "mapping": asdict(task.target.mapping),
            "features": list(task.target.features),
            "expected_route": task.target.expected_route,
            "synthetic_target": task.target.synthetic_target,
        },
        "gates": {
            "accuracy": asdict(task.gates.accuracy),
            "feature_signal_tests": list(task.gates.feature_signal_tests),
            "boot_tests": list(task.gates.boot_tests),
            "component_tests": list(task.gates.component_tests),
            "collective_tests": list(task.gates.collective_tests),
        },
        "certification": {
            "mode": task.certification.mode.value,
            "expected_product_identity": (
                asdict(task.certification.expected_product_identity)
                if task.certification.expected_product_identity is not None
                else None
            ),
        },
        "execution": {
            # A persisted local effective mode remains a CLI-only override.
            "mode": "slurm",
            "agent": {
                "backend_kind": task.execution.agent.backend_kind,
                "model": task.execution.agent.model,
            },
            "slurm": {
                "role_classes": {
                    role_class.role.value: {
                        "resource_class": role_class.resource_class,
                        "network_policy": role_class.network_policy.value,
                        "environment": {
                            "allowlist": list(role_class.environment.allowlist),
                            "values": dict(role_class.environment.values),
                        },
                        "credential_broker": {
                            "broker_id": role_class.credential_broker.broker_id,
                            "allowed_credential_names": list(
                                role_class.credential_broker.allowed_credential_names
                            ),
                            "per_attempt_ttl_seconds": (
                                role_class.credential_broker.per_attempt_ttl_seconds
                            ),
                        },
                    }
                    for role_class in task.execution.slurm.role_classes
                },
                "gate_classes": {
                    gate_class.gate_id: {
                        "resource_class": gate_class.resource_class,
                    }
                    for gate_class in task.execution.slurm.gate_classes
                },
                "controller": {
                    "account": controller.account,
                    "partition": controller.partition,
                    "qos": controller.qos,
                    "reservation": controller.reservation,
                    "time_limit": _format_time_limit(controller.time_limit_seconds),
                    "cpus_per_task": controller.cpus_per_task,
                    "memory": f"{controller.memory_mib}MiB",
                    "image": controller.image.as_posix(),
                    "scheduler_clients": controller.scheduler_clients,
                    "container_launch_mode": controller.container_launch_mode.value,
                    "mounts": [
                        {
                            "host_path": mount.host_path.as_posix(),
                            "container_path": mount.container_path,
                            "read_only": mount.read_only,
                        }
                        for mount in controller.mounts
                    ],
                    "environment": dict(controller.environment),
                    "build_identity": controller.build_identity,
                    "dispatch_mode": controller.dispatch_mode,
                    "requeue": controller.requeue,
                    "advance_signal_lead_seconds": controller.advance_signal_lead_seconds,
                    "heartbeat_timeout_seconds": controller.heartbeat_timeout_seconds,
                    "lease_timeout_seconds": controller.lease_timeout_seconds,
                    "orphan_grace_seconds": controller.orphan_grace_seconds,
                    "credential_broker_socket": controller.credential_broker_socket,
                },
                "agent_worker": {
                    "image": agent_worker.image.as_posix(),
                    "build_identity": agent_worker.build_identity,
                    "scheduler_clients": agent_worker.scheduler_clients,
                    "network_enforcement": agent_worker.network_enforcement.value,
                },
                "smith": {
                    "max_parallel_items": smith.max_parallel_items,
                    "max_nodes_total": smith.max_nodes_total,
                    "max_gpus_total": smith.max_gpus_total,
                    "distinct_nodes_required": smith.distinct_nodes_required,
                    "exclusive": smith.exclusive,
                    "resource_classes": {
                        resource.name: {
                            "nodes": resource.nodes,
                            "tasks_per_node": resource.tasks_per_node,
                            "gpus_per_node": resource.gpus_per_node,
                            "cpus_per_task": resource.cpus_per_task,
                            "memory": f"{resource.memory_mib}MiB",
                            "time_limit": _format_time_limit(resource.time_limit_seconds),
                        }
                        for resource in smith.resource_classes
                    },
                    "per_item_override_bounds": {
                        "max_nodes": smith.per_item_override_bounds.max_nodes,
                        "max_tasks_per_node": smith.per_item_override_bounds.max_tasks_per_node,
                        "max_gpus_per_node": smith.per_item_override_bounds.max_gpus_per_node,
                        "max_cpus_per_task": smith.per_item_override_bounds.max_cpus_per_task,
                        "max_memory": f"{smith.per_item_override_bounds.max_memory_mib}MiB",
                        "max_time_limit": _format_time_limit(
                            smith.per_item_override_bounds.max_time_limit_seconds
                        ),
                    },
                    "retry_policy": asdict(smith.retry_policy),
                },
            },
        },
        "delivery": delivery,
    }


def write_normalized_task(path: str | Path, task: NormalizedTask) -> None:
    """Atomically persist a canonical task snapshot for controller resume.

    Args:
        path: Destination JSON path. Its parent directory must already exist.
        task: Validated immutable task.

    Raises:
        FileNotFoundError: If the destination parent does not exist.
        OSError: If the snapshot cannot be written or atomically replaced.
    """
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"normalized task parent does not exist: {destination.parent}")
    envelope: dict[str, object] = {
        "format_version": 1,
        "digest": task.digest,
        "execution_override": task.execution.mode,
        "task": _source_task_dict(task),
    }
    text = json.dumps(
        envelope,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(f"{text}\n", encoding="utf-8")
    temporary.replace(destination)


def load_normalized_task(path: str | Path) -> NormalizedTask:
    """Load a persisted canonical task and fail closed on drift or corruption.

    Args:
        path: JSON snapshot written by :func:`write_normalized_task`.

    Returns:
        The fully reconstructed frozen task model.

    Raises:
        TaskSchemaError: If the envelope, task, or claimed digest is invalid.
    """
    snapshot_path = Path(path)
    if not snapshot_path.is_file():
        raise TaskSchemaError(f"normalized task file not found: {snapshot_path}")
    try:
        raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskSchemaError(f"{snapshot_path} is not valid readable JSON: {exc}") from exc
    errors: list[str] = []
    envelope = _mapping(
        raw,
        "normalized_task",
        frozenset({"format_version", "digest", "execution_override", "task"}),
        frozenset({"format_version", "digest", "execution_override", "task"}),
        errors,
    )
    format_version = _positive_int(
        envelope.get("format_version"), "normalized_task.format_version", errors
    )
    if format_version != 1:
        errors.append(f"'normalized_task.format_version' must be 1, got {format_version!r}")
    claimed_digest = _nonempty_string(envelope.get("digest"), "normalized_task.digest", errors)
    if re.fullmatch(r"[0-9a-f]{64}", claimed_digest) is None:
        errors.append("'normalized_task.digest' must be a lowercase SHA-256 hex digest")
    override_value = _nonempty_string(
        envelope.get("execution_override"), "normalized_task.execution_override", errors
    )
    if override_value not in {"slurm", "local"}:
        errors.append("'normalized_task.execution_override' must be 'slurm' or 'local'")
    if errors:
        bullets = "\n".join(f"  - {error}" for error in errors)
        raise TaskSchemaError(f"{snapshot_path} failed normalized-task validation:\n{bullets}")
    task = _normalize_task_value(
        envelope["task"],
        str(snapshot_path),
        execution_override=cast(Literal["slurm", "local"], override_value),
    )
    if task.digest != claimed_digest:
        raise TaskSchemaError(
            f"{snapshot_path} normalized task digest mismatch: claimed {claimed_digest}, "
            f"computed {task.digest}"
        )
    return task


def _normalize_task_value(
    raw: object,
    source: str,
    *,
    execution_override: Literal["slurm", "local"] | None = None,
) -> NormalizedTask:
    """Strictly validate and normalize one already parsed task value.

    Args:
        raw: Parsed YAML-compatible value.
        source: Human-readable input identity used in errors.
        execution_override: Optional CLI-selected execution mode. ``local``
            is reserved for CPU/fake-runner tests; it does not remove the
            requirement for a complete production Slurm resource policy.

    Returns:
        A deeply immutable normalized task.

    Raises:
        TaskSchemaError: If any schema validation fails. All
            independently discoverable schema errors are reported together.
    """
    if execution_override not in {None, "slurm", "local"}:
        raise TaskSchemaError(
            f"{source} has unsupported execution override {execution_override!r}; "
            "expected 'slurm' or test-only 'local'"
        )
    errors: list[str] = []
    root = _mapping(
        raw,
        "task",
        frozenset(
            {
                "schema_version",
                "repository",
                "reference",
                "target",
                "gates",
                "certification",
                "execution",
                "delivery",
            }
        ),
        frozenset(
            {
                "schema_version",
                "repository",
                "reference",
                "target",
                "gates",
                "certification",
                "execution",
                "delivery",
            }
        ),
        errors,
    )
    _scan_obsolete_tokens(raw, "task", errors)
    schema_version = _positive_int(root.get("schema_version"), "schema_version", errors)
    if schema_version != TASK_SCHEMA_VERSION:
        errors.append(f"'schema_version' must be {TASK_SCHEMA_VERSION}, got {schema_version!r}")

    repository_data = _required_mapping(
        root,
        "repository",
        "",
        frozenset({"root", "base_commit", "dirty_policy", "workspace_root"}),
        frozenset({"root", "base_commit", "dirty_policy", "workspace_root"}),
        errors,
    )
    reference_data = _required_mapping(
        root,
        "reference",
        "",
        frozenset({"checkpoint", "provenance", "architecture", "additional_sources"}),
        frozenset({"checkpoint", "provenance", "architecture"}),
        errors,
    )
    target_data = _required_mapping(
        root,
        "target",
        "",
        frozenset(
            {
                "family",
                "checkpoint_id",
                "sm",
                "world_size",
                "mapping",
                "features",
                "expected_route",
                "synthetic_target",
            }
        ),
        frozenset(
            {
                "family",
                "checkpoint_id",
                "sm",
                "world_size",
                "mapping",
                "features",
                "expected_route",
                "synthetic_target",
            }
        ),
        errors,
    )
    gates_data = _required_mapping(
        root,
        "gates",
        "",
        frozenset(
            {
                "accuracy",
                "feature_signal_tests",
                "boot_tests",
                "component_tests",
                "collective_tests",
            }
        ),
        frozenset(
            {
                "accuracy",
                "feature_signal_tests",
                "boot_tests",
                "component_tests",
                "collective_tests",
            }
        ),
        errors,
    )
    certification_data = _required_mapping(
        root,
        "certification",
        "",
        frozenset({"mode", "expected_product_identity"}),
        frozenset({"mode"}),
        errors,
    )
    execution_data = _required_mapping(
        root,
        "execution",
        "",
        frozenset({"mode", "agent", "slurm"}),
        frozenset({"mode", "agent", "slurm"}),
        errors,
    )
    delivery_data = _required_mapping(
        root,
        "delivery",
        "",
        frozenset({"mode", "branch", "merge", "push"}),
        frozenset({"mode", "merge", "push"}),
        errors,
    )

    repository = _parse_repository(repository_data, errors)
    reference = _parse_reference(reference_data, errors)
    target = _parse_target(target_data, repository, errors)
    gates = _parse_gates(gates_data, errors)
    if target.features and not gates.feature_signal_tests:
        errors.append(
            "'gates.feature_signal_tests' must include an executable acceptance test when "
            "target.features is non-empty"
        )
    execution = _parse_execution(execution_data, target, errors)
    certification = _parse_certification(certification_data, repository, target, execution, errors)
    _validate_gate_classes(certification, gates, target, execution, errors)
    if certification.mode is CertificationMode.LOCAL and execution_override != "local":
        errors.append(
            "'certification.mode: LOCAL' requires the explicit CLI local execution override"
        )
    if execution_override is not None:
        execution = replace(execution, mode=execution_override)
        if execution_override == "local":
            certification = CertificationConfig(CertificationMode.LOCAL, None)
    delivery = _parse_delivery(delivery_data, errors)
    _validate_shared_mounts(repository, reference, execution, errors)

    if errors:
        bullets = "\n".join(f"  - {error}" for error in errors)
        raise TaskSchemaError(f"{source} failed Staircase task validation:\n{bullets}")

    without_digest = NormalizedTask(
        schema_version,
        repository,
        reference,
        target,
        gates,
        certification,
        execution,
        delivery,
        digest="",
    )
    digest = hashlib.sha256(
        normalized_task_json(without_digest, include_digest=False).encode()
    ).hexdigest()
    return NormalizedTask(
        schema_version,
        repository,
        reference,
        target,
        gates,
        certification,
        execution,
        delivery,
        digest,
    )


def load_and_normalize_task(
    path: str | Path, *, execution_override: Literal["slurm", "local"] | None = None
) -> NormalizedTask:
    """Load, strictly validate, normalize, and digest a Staircase task.

    Args:
        path: YAML task file.
        execution_override: Optional CLI-selected execution mode. ``local``
            is reserved for CPU/fake-runner tests; it does not remove the
            requirement for a complete production Slurm resource policy.

    Returns:
        A deeply immutable normalized task.

    Raises:
        TaskSchemaError: If YAML parsing or any schema validation fails. All
            independently discoverable schema errors are reported together.
    """
    task_path = Path(path)
    if not task_path.is_file():
        raise TaskSchemaError(f"task file not found: {task_path}")
    try:
        raw = yaml.load(task_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, TypeError, yaml.YAMLError) as exc:
        raise TaskSchemaError(f"{task_path} is not valid readable YAML: {exc}") from exc
    return _normalize_task_value(raw, str(task_path), execution_override=execution_override)


def load_and_validate_task_yaml(path: str | Path) -> NormalizedTask:
    """Compatibility alias for loading a task without a CLI mode override.

    Args:
        path: YAML task file.

    Returns:
        A deeply immutable normalized task.

    Raises:
        TaskSchemaError: If YAML parsing or schema validation fails.
    """
    return load_and_normalize_task(path)


__all__ = [
    "AccuracyGate",
    "AgentExecutionConfig",
    "AgentWorkerConfig",
    "CertificationConfig",
    "CertificationMode",
    "ControllerConfig",
    "ContainerLaunchMode",
    "CredentialBrokerPolicy",
    "DeliveryConfig",
    "ExecutionConfig",
    "ExpectedProductIdentityConfig",
    "GateClass",
    "GatesConfig",
    "MountConfig",
    "NetworkEnforcement",
    "NormalizedTask",
    "OverrideBounds",
    "ParallelMapping",
    "ReferenceConfig",
    "RepositoryConfig",
    "ResourceClass",
    "RetryPolicy",
    "RoleEnvironmentPolicy",
    "RoleNetworkPolicy",
    "SlurmConfig",
    "SlurmRole",
    "SlurmRoleClass",
    "SmithConfig",
    "TASK_SCHEMA_VERSION",
    "TargetConfig",
    "TaskSchemaError",
    "load_and_normalize_task",
    "load_normalized_task",
    "load_and_validate_task_yaml",
    "normalized_task_dict",
    "normalized_task_json",
    "write_normalized_task",
]
