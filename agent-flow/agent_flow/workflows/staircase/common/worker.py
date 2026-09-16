# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Isolated execution bridge for generic Staircase worker manifests."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence, cast

from agent_flow.workflows.staircase.prompts import build_staircase_prompts
from agent_flow.workflows.staircase.prompts.profiles import DomainProfile as PromptProfile
from agent_flow.workflows.staircase.state import DomainProfile, Role

from .artifacts import (
    INPUT_FILENAME,
    OUTPUT_DIRECTORY,
    JsonValue,
    WorkerInputManifest,
    WorkerResultManifest,
    WorkerResultStatus,
    describe_evidence,
    load_input_manifest,
    publish_result,
)
from .credentials import BackendKind as CredentialBackendKind
from .credentials import (
    CredentialBinding,
    CredentialDescriptor,
    agent_credential_environment,
    descriptor_from_public_dict,
    ensure_file_has_no_credential_values,
    ensure_no_credential_values,
    load_worker_credentials,
    redact_credential_values,
    redact_durable_standard_streams,
)
from .gates import AccuracyCriteria, EvidenceScope, GatePurpose, GateReceipt, RankPlacement
from .isolation import build_gate_worker_environment, digest_metadata_free_tree
from .launch_policy import (
    AgentLaunchPolicy,
    AgentLaunchPolicyError,
    agent_launch_policy_from_public_dict,
)
from .placement import HorizontalPlacementContract, PlacementError, publish_worker_placement
from .slurm import AGENT_POLICY_DIGEST_ENVIRONMENT

WORKER_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_GATE_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_SAFE_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_NODE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_NUMERIC_GPU_ID = re.compile(r"^[0-9]+$")
_TRUSTED_SINGLE_GPU_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "SLURM_LOCALID",
    "SLURMD_NODENAME",
    "SLURM_NTASKS",
    "SLURM_PROCID",
)
_CPU_STATIC_LAUNCHER_ENVIRONMENT = {
    "SLURM_LOCALID": "0",
    "SLURM_NTASKS": "1",
    "SLURM_PROCID": "0",
    "SLURM_STEP_NUM_TASKS": "1",
    "SLURM_TASKS_PER_NODE": "1",
}
_CPU_FORBIDDEN_PLACEMENT_ENVIRONMENT = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "NVIDIA_VISIBLE_DEVICES",
        "RANK",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_NODE",
        "SLURM_GPUS_PER_TASK",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "WORLD_SIZE",
    }
)
_COMMAND_FORBIDDEN_PLACEMENT_ENVIRONMENT = _CPU_FORBIDDEN_PLACEMENT_ENVIRONMENT.union(
    _CPU_STATIC_LAUNCHER_ENVIRONMENT,
    {"HOSTNAME", "SLURMD_NODENAME"},
)
_AGENT_ROLES = {Role.CODER, Role.REVIEWER, Role.QA}
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
_CERTIFICATION_MODE_BY_SCOPE = {
    EvidenceScope.CPU_STATIC: "LOCAL",
    EvidenceScope.SINGLE_GPU_PRODUCT: "LOCAL",
    EvidenceScope.LOCAL_FOUR_GPU_PRODUCT: "LOCAL",
    EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER: "SYNTHETIC",
    EvidenceScope.REAL_MULTI_NODE_PRODUCT: "REAL",
}
_ROLE_OUTCOMES = {
    Role.CODER: "coder_result",
    Role.REVIEWER: "reviewer_result",
    Role.QA: "qa_result",
}
_AGENT_STATUSES = {
    Role.CODER: {
        WorkerResultStatus.SUCCEEDED,
        WorkerResultStatus.FAILED,
        WorkerResultStatus.RETRYABLE_FAILED,
        WorkerResultStatus.BLOCKED,
        WorkerResultStatus.RESOURCE_ESCALATION,
    },
    Role.REVIEWER: {
        WorkerResultStatus.SUCCEEDED,
        WorkerResultStatus.FAILED,
        WorkerResultStatus.RETRYABLE_FAILED,
        WorkerResultStatus.BLOCKED,
        WorkerResultStatus.REJECTED,
        WorkerResultStatus.RESOURCE_ESCALATION,
    },
    Role.QA: {
        WorkerResultStatus.SUCCEEDED,
        WorkerResultStatus.FAILED,
        WorkerResultStatus.RETRYABLE_FAILED,
        WorkerResultStatus.BLOCKED,
        WorkerResultStatus.REJECTED,
        WorkerResultStatus.RESOURCE_ESCALATION,
    },
}


class WorkerRuntimeError(RuntimeError):
    """Raised when a generic worker input or response violates its contract."""


@dataclass(frozen=True, slots=True)
class AgentRuntimeSpec:
    """Controller-authorized inputs for one fresh role agent."""

    backend_kind: str
    model: str
    prompt_id: str
    prompt: str
    evidence_paths: tuple[str, ...]
    candidate_digest: str | None
    credential_descriptor: CredentialDescriptor
    launch_policy: AgentLaunchPolicy
    candidate_overlay_digest: str | None
    placement_contract: HorizontalPlacementContract


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """Strict lifecycle-neutral JSON returned by a role agent."""

    status: WorkerResultStatus
    summary: str
    evidence_paths: tuple[str, ...]
    payload: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class GateCommandSpec:
    """One controller-authorized shell-free deterministic command."""

    command_id: str
    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]
    cwd: str


@dataclass(frozen=True, slots=True)
class GateRuntimeSpec:
    """Exactly one deterministic command and typed receipt contract."""

    command: GateCommandSpec
    candidate_attempt_id: str
    candidate_digest: str | None
    receipt_contract: GateReceiptExpectation


@dataclass(frozen=True, slots=True)
class GateReceiptExpectation:
    """Placement-free gate result shape known before Slurm allocation."""

    gate_id: str
    purpose: GatePurpose
    scope: EvidenceScope
    certification_mode: str
    expected_world_size: int
    expected_rank: int | None
    expected_local_rank: int | None
    product_rank_body: bool
    accuracy: AccuracyCriteria | None = None

    def __post_init__(self) -> None:
        if _SAFE_GATE_ID.fullmatch(self.gate_id) is None:
            raise WorkerRuntimeError("gate receipt contract has an unsafe gate_id")
        if not isinstance(self.purpose, GatePurpose) or not isinstance(self.scope, EvidenceScope):
            raise WorkerRuntimeError("gate receipt purpose and scope must use typed values")
        expected_certification = _CERTIFICATION_MODE_BY_SCOPE[self.scope]
        if self.certification_mode != expected_certification:
            raise WorkerRuntimeError(
                f"{self.scope.value} requires certification_mode {expected_certification!r}"
            )
        if self.scope is EvidenceScope.CPU_STATIC:
            expected = (0, None, None, False)
        elif self.scope is EvidenceScope.SINGLE_GPU_PRODUCT:
            expected = (1, 0, 0, True)
        else:
            raise WorkerRuntimeError(
                "generic gate worker supports only CPU_STATIC or SINGLE_GPU_PRODUCT evidence"
            )
        observed = (
            self.expected_world_size,
            self.expected_rank,
            self.expected_local_rank,
            self.product_rank_body,
        )
        if observed != expected:
            raise WorkerRuntimeError(
                f"{self.scope.value} requires expected execution shape {expected!r}"
            )
        if (self.purpose is GatePurpose.ACCURACY) != (self.accuracy is not None):
            raise WorkerRuntimeError("only an accuracy gate may carry accuracy criteria")


def is_worker_input_manifest(path: Path) -> bool:
    """Return whether ``path`` declares the generic manifest envelope.

    A JSON object containing any envelope discriminator is treated as generic
    input even when malformed. This prevents a damaged worker manifest from
    being reinterpreted as the legacy planning-role contract.

    Args:
        path: Candidate JSON input.

    Returns:
        Whether the object uses a manifest-envelope discriminator.
    """
    value = _load_json_object(path)
    return any(key in value for key in ("kind", "digest", "payload"))


def execute_worker(input_path: Path) -> WorkerResultManifest:
    """Execute one generic worker manifest and publish exactly one typed result.

    Args:
        input_path: Immutable ``input.json`` in an attempt mailbox.

    Returns:
        The result published with ``COMPLETE`` written last.
    """
    attempt_dir, manifest, input_digest = _load_worker_input(input_path)
    try:
        worktree = _canonical_worktree(manifest.worktree)
        runtime = _runtime_object(manifest.payload)
        kind = _required_string(runtime, "kind")
        if kind == "agent":
            if manifest.role not in _AGENT_ROLES:
                raise WorkerRuntimeError(
                    f"agent runtime does not support role {manifest.role.value!r}"
                )
            result = _execute_agent(manifest, input_digest, attempt_dir, worktree, runtime)
        elif kind == "gate":
            if manifest.role is not Role.GATE:
                raise WorkerRuntimeError("gate runtime requires the deterministic gate role")
            result = _execute_gate(manifest, input_digest, attempt_dir, worktree, runtime)
        else:
            raise WorkerRuntimeError(f"unsupported worker runtime kind {kind!r}")
    except (OSError, ValueError, RuntimeError) as error:
        result = _failed_result(manifest, input_digest, f"worker failed closed: {error}")
    publish_result(attempt_dir, result, input_path=input_path)
    return result


def _load_worker_input(path: Path) -> tuple[Path, WorkerInputManifest, str]:
    if path.name != INPUT_FILENAME:
        raise WorkerRuntimeError(f"generic worker input must be named {INPUT_FILENAME!r}")
    if not path.is_absolute():
        raise WorkerRuntimeError("generic worker input path must be absolute")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise WorkerRuntimeError("attempt mailbox must be a regular directory")
    canonical_path = path.resolve(strict=True)
    if canonical_path != path:
        raise WorkerRuntimeError("generic worker input path must be canonical")
    output_dir = canonical_path.parent / OUTPUT_DIRECTORY
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise WorkerRuntimeError("worker output mailbox must be a regular directory")
    canonical_output = output_dir.resolve(strict=True)
    if canonical_output != output_dir:
        raise WorkerRuntimeError("worker output mailbox must be canonical")
    manifest, input_digest = load_input_manifest(canonical_path)
    return canonical_output, manifest, input_digest


def _canonical_worktree(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise WorkerRuntimeError("worker worktree must be an absolute regular directory")
    canonical = path.resolve(strict=True)
    if canonical != path:
        raise WorkerRuntimeError("worker worktree must be canonical")
    return canonical


def _runtime_object(payload: Mapping[str, JsonValue]) -> dict[str, object]:
    _exact_keys(cast(dict[str, object], payload), {"runtime", "context"}, "worker payload")
    runtime = _required_object(payload["runtime"], "runtime")
    _required_object(payload["context"], "context")
    return runtime


def _execute_agent(
    manifest: WorkerInputManifest,
    input_digest: str,
    attempt_dir: Path,
    worktree: Path,
    runtime: dict[str, object],
) -> WorkerResultManifest:
    credentials: dict[str, str] = {}
    try:
        validate_role_worker_negative_preflight()
        spec = _parse_agent_runtime(runtime, manifest)
        _validate_agent_launch_policy(spec, worktree, os.environ)
        publish_worker_placement(
            attempt_dir,
            manifest,
            spec.placement_contract,
            environment=os.environ,
        )
        if manifest.role is Role.CODER and spec.candidate_digest is not None:
            raise WorkerRuntimeError("Coder input cannot predeclare a candidate digest")
        if manifest.role in {Role.REVIEWER, Role.QA} and spec.candidate_digest is None:
            raise WorkerRuntimeError(f"{manifest.role.value} input requires a candidate digest")
        system_prompt = _agent_system_prompt(manifest.role, manifest.profile)
        schema_instruction = _agent_schema_instruction(manifest.role, manifest.profile)
        prompt = _agent_prompt(manifest, spec, manifest.payload["context"], schema_instruction)
        binding = _credential_binding(manifest, spec.backend_kind)
        credentials = load_worker_credentials(
            spec.credential_descriptor,
            expected_binding=binding,
        )
        with agent_credential_environment(spec.backend_kind, credentials):
            with redact_durable_standard_streams(credentials):
                response = _invoke_agent(
                    backend_kind=spec.backend_kind,
                    model=spec.model,
                    cwd=worktree,
                    layer_name=f"staircase-{manifest.role.value}-{manifest.item_id}",
                    system_prompt=f"{system_prompt}\n\n{schema_instruction}",
                    prompt=prompt,
                )
        ensure_no_credential_values(response, credentials, label="agent response")
        outcome = _parse_agent_outcome(response, manifest.role, spec.evidence_paths)
        _validate_agent_payload(manifest, outcome.status, outcome.payload)
        candidate_overlay_digest = None
        if manifest.role is Role.CODER and outcome.status is WorkerResultStatus.SUCCEEDED:
            candidate_overlay_digest = digest_metadata_free_tree(
                worktree,
                secret_scan_paths=_coder_secret_scan_paths(outcome.payload),
                credential_values=credentials,
            )
        evidence = _collect_agent_evidence(
            worktree,
            attempt_dir,
            outcome.evidence_paths,
            credentials,
        )
        payload = dict(outcome.payload)
        if (
            manifest.role is Role.CODER
            and manifest.profile is DomainProfile.TUNER
            and payload.get("result_kind") == "tuner_measurement"
        ):
            payload["evidence"] = [
                {
                    "path": entry.path,
                    "sha256": entry.sha256,
                    "size_bytes": entry.size_bytes,
                }
                for entry in (describe_evidence(attempt_dir, path) for path in evidence)
            ]
        reviewed_digest = (
            spec.candidate_digest if manifest.role in {Role.REVIEWER, Role.QA} else None
        )
        return _result(
            manifest,
            input_digest,
            attempt_dir,
            status=outcome.status,
            summary=outcome.summary,
            evidence_paths=evidence,
            candidate_digest=candidate_overlay_digest,
            reviewed_candidate_digest=reviewed_digest,
            payload=payload,
        )
    except (OSError, ValueError, RuntimeError) as error:
        detail = redact_credential_values(str(error), credentials)
        return _failed_result(manifest, input_digest, f"agent worker failed closed: {detail}")


def validate_role_worker_negative_preflight(
    environment: Mapping[str, str] | None = None,
) -> None:
    """Fail closed if an agent role can authenticate to or invoke Slurm.

    Placement variables injected by ``slurmd`` are not credentials and remain
    available to deterministic gate code.  Agent roles, however, must run in a
    worker image/PATH without scheduler clients and without any scheduler-auth
    material.  This check happens before the backend credential is mounted into
    the process environment.
    """
    observed_environment = os.environ if environment is None else environment
    contaminated = sorted(_SCHEDULER_AUTH_ENVIRONMENT.intersection(observed_environment))
    if contaminated:
        raise WorkerRuntimeError(
            "agent role negative preflight found scheduler authentication material: "
            f"{contaminated!r}"
        )
    clients = tuple(
        (name, path)
        for name in _SCHEDULER_CLIENTS
        if (path := _scheduler_client_path(name)) is not None
    )
    if clients:
        names = [name for name, _path in clients]
        raise WorkerRuntimeError(
            f"agent role negative preflight found scheduler clients in PATH: {names!r}"
        )


def _scheduler_client_path(name: str) -> str | None:
    """Resolve one prohibited scheduler client for an injectable unit-test seam."""
    return shutil.which(name)


def _parse_agent_runtime(
    runtime: dict[str, object], manifest: WorkerInputManifest
) -> AgentRuntimeSpec:
    required = {
        "schema_version",
        "kind",
        "backend_kind",
        "model",
        "prompt_id",
        "prompt",
        "evidence_paths",
        "candidate_digest",
    }
    required.update({"launch_policy", "candidate_overlay_digest", "placement_contract"})
    optional = {"credential_descriptor"}
    unknown = sorted(set(runtime) - required - optional)
    missing = sorted(required - set(runtime))
    if unknown or missing:
        raise WorkerRuntimeError(
            f"agent runtime keys invalid; missing={missing}, unknown={unknown}"
        )
    _schema_version(runtime)
    if runtime["kind"] != "agent":
        raise WorkerRuntimeError("agent runtime kind must be 'agent'")
    backend_kind = _required_string(runtime, "backend_kind")
    if backend_kind not in _BACKENDS:
        raise WorkerRuntimeError(f"unsupported agent backend {backend_kind!r}")
    model = _required_string(runtime, "model")
    prompt_id = _safe_id(runtime, "prompt_id")
    prompt = _required_string(runtime, "prompt")
    evidence_paths = _relative_paths(runtime["evidence_paths"], "evidence_paths")
    candidate_digest = _optional_digest(runtime["candidate_digest"], "candidate_digest")
    binding = _credential_binding(manifest, backend_kind)
    raw_descriptor = runtime.get("credential_descriptor")
    descriptor = (
        CredentialDescriptor.no_credentials(binding)
        if raw_descriptor is None
        else descriptor_from_public_dict(raw_descriptor)
    )
    try:
        launch_policy = agent_launch_policy_from_public_dict(runtime["launch_policy"])
    except AgentLaunchPolicyError as error:
        raise WorkerRuntimeError(f"invalid agent launch policy: {error}") from error
    overlay_digest = _optional_digest(
        runtime["candidate_overlay_digest"], "candidate_overlay_digest"
    )
    if manifest.role in {Role.REVIEWER, Role.QA} and overlay_digest is None:
        raise WorkerRuntimeError("Reviewer and QA require an immutable candidate overlay digest")
    if manifest.role is Role.CODER and overlay_digest is not None:
        raise WorkerRuntimeError("Coder input cannot freeze a mutable candidate overlay")
    try:
        placement_contract = HorizontalPlacementContract.from_public_dict(
            runtime["placement_contract"]
        )
    except PlacementError as error:
        raise WorkerRuntimeError(f"invalid horizontal placement contract: {error}") from error
    if placement_contract.distinct_nodes_required and not (
        manifest.role in {Role.CODER, Role.REVIEWER} and manifest.profile is DomainProfile.SMITH
    ):
        raise WorkerRuntimeError(
            "distinct-node placement is admitted only for horizontal Smith Coder/Reviewer jobs"
        )
    return AgentRuntimeSpec(
        backend_kind,
        model,
        prompt_id,
        prompt,
        evidence_paths,
        candidate_digest,
        descriptor,
        launch_policy,
        overlay_digest,
        placement_contract,
    )


def _validate_agent_launch_policy(
    spec: AgentRuntimeSpec,
    worktree: Path,
    environment: Mapping[str, str],
) -> None:
    observed_digest = environment.get(AGENT_POLICY_DIGEST_ENVIRONMENT)
    if observed_digest != spec.launch_policy.digest:
        raise WorkerRuntimeError(
            "agent launch policy is not bound by the trusted scheduler environment"
        )
    if spec.candidate_overlay_digest is not None:
        observed_overlay = digest_metadata_free_tree(worktree)
        if observed_overlay != spec.candidate_overlay_digest:
            raise WorkerRuntimeError("immutable candidate overlay digest mismatch")


def _credential_binding(manifest: WorkerInputManifest, backend_kind: str) -> CredentialBinding:
    return CredentialBinding(
        run_id=manifest.run_id,
        item_id=manifest.item_id,
        attempt_id=manifest.attempt_id,
        task_digest=manifest.task_digest,
        generation=manifest.generation,
        backend_kind=cast(CredentialBackendKind, backend_kind),
    )


def _agent_system_prompt(role: Role, profile: DomainProfile | None) -> str:
    if role in {Role.CODER, Role.REVIEWER}:
        if profile is None:
            raise WorkerRuntimeError(f"{role.value} requires a domain profile")
        prompt_profile = PromptProfile(profile.value)
        prompts = build_staircase_prompts(prompt_profile)
    else:
        prompts = build_staircase_prompts()
    prompt_by_role = {
        Role.CODER: prompts.coder,
        Role.REVIEWER: prompts.reviewer,
        Role.QA: prompts.qa,
    }
    return prompt_by_role[role]


def _agent_schema_instruction(role: Role, profile: DomainProfile | None) -> str:
    statuses = ", ".join(sorted(status.value for status in _AGENT_STATUSES[role]))
    instruction = (
        "Return exactly one JSON object and no Markdown. Required keys are: "
        "schema_version (integer 1), outcome (exact string "
        f"'{_ROLE_OUTCOMES[role]}'), status (one of: {statuses}), summary "
        "(non-empty string), evidence_paths (a unique list containing only "
        "controller-authorized POSIX-relative regular files), and payload (a JSON object). "
        "Do not add keys and do not encode lifecycle decisions in prose."
    )
    if role is Role.CODER and profile is DomainProfile.TUNER:
        instruction += (
            " For a successful Tuner measurement, payload must contain exactly "
            "schema_version=1, result_kind='tuner_measurement', baseline, and candidate. "
            "Each arm must contain exactly identity, arm_digest, curve, and gates. Identity "
            "contains checkpoint, target, route, workload, topology, build, protocol, and "
            "hardware; curve contains finite samples and uncertainty; every gate contains "
            "name, kind, passed, and evidence_digest. Do not include a keep/reject or "
            "promotion decision; the controller computes it."
        )
    return instruction


def _agent_prompt(
    manifest: WorkerInputManifest,
    spec: AgentRuntimeSpec,
    context: JsonValue,
    schema_instruction: str,
) -> str:
    controller_context: dict[str, JsonValue] = {
        "run_id": manifest.run_id,
        "item_id": manifest.item_id,
        "attempt_id": manifest.attempt_id,
        "task_digest": manifest.task_digest,
        "generation": manifest.generation,
        "role": manifest.role.value,
        "profile": manifest.profile.value if manifest.profile is not None else None,
        "allowed_paths": list(manifest.allowed_paths),
        "authorized_evidence_paths": list(spec.evidence_paths),
        "candidate_digest": spec.candidate_digest,
        "task": context,
    }
    context_json = json.dumps(
        controller_context,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{spec.prompt}\n\nController context (JSON):\n{context_json}\n\n{schema_instruction}"


def _invoke_agent(
    *,
    backend_kind: str,
    model: str,
    cwd: Path,
    layer_name: str,
    system_prompt: str,
    prompt: str,
) -> str:
    from agent_flow import AgentLayer, AgentLayerConfig, BackendConfig, SessionConfig
    from agent_flow.config import BackendKind

    with AgentLayer(
        AgentLayerConfig(
            name=layer_name,
            system_prompt=system_prompt,
            backend=BackendConfig(
                kind=cast(BackendKind, backend_kind),
                model=model,
                cwd=cwd,
            ),
            session=SessionConfig(mode="stateless"),
            human_input_enabled=False,
        )
    ) as layer:
        response = layer(prompt)
    if not isinstance(response, str):
        raise WorkerRuntimeError("agent response must be text containing one JSON object")
    return response


def _parse_agent_outcome(
    response: str,
    role: Role,
    authorized_evidence: Sequence[str],
) -> AgentOutcome:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as error:
        raise WorkerRuntimeError("agent response is not exactly one JSON object") from error
    data = _required_object(value, "agent response")
    _exact_keys(
        data,
        {"schema_version", "outcome", "status", "summary", "evidence_paths", "payload"},
        "agent response",
    )
    _schema_version(data)
    if data["outcome"] != _ROLE_OUTCOMES[role]:
        raise WorkerRuntimeError(f"agent response outcome must be {_ROLE_OUTCOMES[role]!r}")
    try:
        status = WorkerResultStatus(_required_string(data, "status"))
    except ValueError as error:
        raise WorkerRuntimeError("agent response has an unsupported status") from error
    if status not in _AGENT_STATUSES[role]:
        raise WorkerRuntimeError(f"status {status.value!r} is not valid for role {role.value!r}")
    evidence_paths = _relative_paths(data["evidence_paths"], "evidence_paths")
    unauthorized = sorted(set(evidence_paths) - set(authorized_evidence))
    if unauthorized:
        raise WorkerRuntimeError(f"agent declared unauthorized evidence paths: {unauthorized}")
    if status is WorkerResultStatus.SUCCEEDED and evidence_paths != tuple(authorized_evidence):
        raise WorkerRuntimeError(
            "a successful agent result must declare every authorized evidence path in order"
        )
    payload = _json_object(data["payload"], "agent payload")
    return AgentOutcome(
        status=status,
        summary=_required_string(data, "summary"),
        evidence_paths=evidence_paths,
        payload=payload,
    )


def _validate_agent_payload(
    manifest: WorkerInputManifest,
    status: WorkerResultStatus,
    payload: Mapping[str, JsonValue],
) -> None:
    role = manifest.role
    result_kind = payload.get("result_kind")
    if not isinstance(result_kind, str):
        raise WorkerRuntimeError("agent payload requires a string result_kind")
    if status is WorkerResultStatus.RESOURCE_ESCALATION:
        allowed = {"resource_escalation"}
    elif (
        role is Role.CODER
        and manifest.profile is DomainProfile.TUNER
        and status is WorkerResultStatus.SUCCEEDED
    ):
        allowed = {"tuner_measurement"}
    else:
        allowed = {
            Role.CODER: {"coder"},
            Role.REVIEWER: {"reviewer_analysis", "reviewer_rerun"},
            Role.QA: {"qa"},
        }[role]
    if result_kind not in allowed:
        raise WorkerRuntimeError(
            f"agent payload result_kind {result_kind!r} is invalid for {role.value!r}"
        )
    if result_kind == "tuner_measurement":
        _exact_keys(
            cast(dict[str, object], payload),
            {"schema_version", "result_kind", "baseline", "candidate"},
            "Tuner agent payload",
        )
        _schema_version(cast(dict[str, object], payload))


def _coder_secret_scan_paths(payload: Mapping[str, JsonValue]) -> tuple[str, ...]:
    """Return the Coder-declared files whose bytes can enter a candidate."""
    if payload.get("result_kind") == "tuner_measurement":
        return ()
    values = _required_list(payload.get("changed_paths"), "Coder changed_paths")
    paths = tuple(
        _relative_path(_plain_string(value, "Coder changed path"), "Coder changed path")
        for value in values
    )
    if tuple(sorted(set(paths))) != paths:
        raise WorkerRuntimeError("Coder changed_paths must be sorted and unique")
    return paths


def _collect_agent_evidence(
    worktree: Path,
    attempt_dir: Path,
    relative_paths: Sequence[str],
    credentials: Mapping[str, str],
) -> tuple[str, ...]:
    collected: list[str] = []
    for relative_path in relative_paths:
        source = _regular_member(worktree, relative_path, "agent evidence")
        ensure_file_has_no_credential_values(
            source,
            credentials,
            label=f"agent evidence {relative_path!r}",
        )
        destination_relative = f"evidence/agent/{relative_path}"
        destination = _evidence_destination(attempt_dir, destination_relative)
        _copy_exclusive(source, destination)
        collected.append(destination_relative)
    return tuple(collected)


def _execute_gate(
    manifest: WorkerInputManifest,
    input_digest: str,
    attempt_dir: Path,
    worktree: Path,
    runtime: dict[str, object],
) -> WorkerResultManifest:
    try:
        spec = _parse_gate_runtime(runtime)
        if spec.candidate_digest is None:
            raise WorkerRuntimeError("gate input requires a frozen candidate digest")
        _validate_gate_candidate_context(
            manifest.payload["context"],
            spec.candidate_attempt_id,
            spec.candidate_digest,
        )
        receipt, trusted_environment = _resolve_gate_receipt(
            spec.receipt_contract,
            os.environ,
            socket.gethostname(),
        )
    except (ValueError, RuntimeError) as error:
        return _failed_result(manifest, input_digest, f"gate worker failed closed: {error}")

    evidence_relative = f"evidence/gates/{spec.command.command_id}.json"
    try:
        cwd = _gate_cwd(worktree, spec.command.cwd)
        record = _run_gate_command(spec.command, cwd, trusted_environment)
        destination = _evidence_destination(attempt_dir, evidence_relative)
        _write_exclusive_json(destination, record)
    except (OSError, ValueError, RuntimeError) as error:
        return _failed_result(manifest, input_digest, f"gate worker failed closed: {error}")

    passed = record["returncode"] == 0
    status = WorkerResultStatus.SUCCEEDED if passed else WorkerResultStatus.REJECTED
    receipt_payload = _gate_receipt_payload(
        receipt,
        certification_mode=spec.receipt_contract.certification_mode,
        passed=passed,
    )
    evidence = describe_evidence(attempt_dir, evidence_relative)
    return _result(
        manifest,
        input_digest,
        attempt_dir,
        status=status,
        summary=(
            "deterministic gate command passed" if passed else "deterministic gate command failed"
        ),
        evidence_paths=(evidence_relative,),
        candidate_digest=None,
        reviewed_candidate_digest=spec.candidate_digest,
        payload={
            "schema_version": WORKER_SCHEMA_VERSION,
            "result_kind": "deterministic_gate",
            "candidate_attempt_id": spec.candidate_attempt_id,
            "candidate_digest": spec.candidate_digest,
            "receipt": receipt_payload,
            "evidence": [
                {
                    "path": evidence.path,
                    "sha256": evidence.sha256,
                    "size_bytes": evidence.size_bytes,
                }
            ],
        },
    )


def _parse_gate_runtime(runtime: dict[str, object]) -> GateRuntimeSpec:
    _exact_keys(
        runtime,
        {
            "schema_version",
            "kind",
            "command",
            "candidate_attempt_id",
            "candidate_digest",
            "receipt",
        },
        "gate runtime",
    )
    _schema_version(runtime)
    if runtime["kind"] != "gate":
        raise WorkerRuntimeError("gate runtime kind must be 'gate'")
    command = _parse_gate_command(runtime["command"])
    receipt_contract = _parse_gate_receipt_contract(runtime["receipt"])
    if command.command_id != receipt_contract.gate_id:
        raise WorkerRuntimeError("gate command_id must match receipt gate_id")
    return GateRuntimeSpec(
        command=command,
        candidate_attempt_id=_safe_id(runtime, "candidate_attempt_id"),
        candidate_digest=_optional_digest(runtime["candidate_digest"], "candidate_digest"),
        receipt_contract=receipt_contract,
    )


def _parse_gate_command(value: object) -> GateCommandSpec:
    data = _required_object(value, "gate command")
    _exact_keys(data, {"command_id", "argv", "environment", "cwd"}, "gate command")
    command_id = _safe_id(data, "command_id")
    raw_argv = _required_list(data["argv"], "argv")
    if not raw_argv:
        raise WorkerRuntimeError("gate argv cannot be empty")
    argv = tuple(_plain_string(argument, "argv argument") for argument in raw_argv)
    environment_data = _required_object(data["environment"], "environment")
    environment: list[tuple[str, str]] = []
    for name in sorted(environment_data):
        if not _SAFE_ENVIRONMENT_NAME.fullmatch(name):
            raise WorkerRuntimeError(f"unsafe gate environment name {name!r}")
        if name in _COMMAND_FORBIDDEN_PLACEMENT_ENVIRONMENT:
            raise WorkerRuntimeError(
                f"gate command cannot supply scheduler placement environment {name!r}"
            )
        environment.append((name, _plain_string(environment_data[name], f"environment {name}")))
    build_gate_worker_environment(dict(environment))
    cwd = _relative_path(_required_string(data, "cwd"), "cwd", allow_dot=True)
    return GateCommandSpec(command_id, argv, tuple(environment), cwd)


def _parse_gate_receipt_contract(value: object) -> GateReceiptExpectation:
    data = _required_object(value, "gate receipt contract")
    _exact_keys(
        data,
        {
            "gate_id",
            "purpose",
            "scope",
            "certification_mode",
            "expected_world_size",
            "expected_rank",
            "expected_local_rank",
            "product_rank_body",
            "accuracy",
        },
        "gate receipt contract",
    )
    accuracy = _parse_accuracy_criteria(data["accuracy"])
    try:
        return GateReceiptExpectation(
            gate_id=_required_string(data, "gate_id"),
            purpose=GatePurpose(_required_string(data, "purpose")),
            scope=EvidenceScope(_required_string(data, "scope")),
            certification_mode=_required_string(data, "certification_mode"),
            expected_world_size=_required_int(data, "expected_world_size"),
            expected_rank=_optional_non_negative_int(data["expected_rank"], "expected_rank"),
            expected_local_rank=_optional_non_negative_int(
                data["expected_local_rank"], "expected_local_rank"
            ),
            product_rank_body=_required_bool(data, "product_rank_body"),
            accuracy=accuracy,
        )
    except ValueError as error:
        raise WorkerRuntimeError(f"invalid gate receipt contract: {error}") from error


def _parse_accuracy_criteria(value: object) -> AccuracyCriteria | None:
    if value is None:
        return None
    data = _required_object(value, "gate accuracy")
    _exact_keys(
        data,
        {"selector", "reference", "protocol", "tolerance"},
        "gate accuracy",
    )
    tolerance = data["tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise WorkerRuntimeError("gate accuracy tolerance must be numeric")
    try:
        return AccuracyCriteria(
            selector=_required_string(data, "selector"),
            reference=_required_string(data, "reference"),
            protocol=_required_string(data, "protocol"),
            tolerance=float(tolerance),
        )
    except ValueError as error:
        raise WorkerRuntimeError(f"invalid gate accuracy: {error}") from error


def _validate_gate_candidate_context(
    context: JsonValue,
    candidate_attempt_id: str,
    candidate_digest: str,
) -> None:
    data = _required_object(context, "gate context")
    candidate = _required_object(data.get("candidate"), "gate candidate context")
    if _safe_id(candidate, "coder_attempt_id") != candidate_attempt_id:
        raise WorkerRuntimeError("gate context candidate attempt differs from runtime")
    if _optional_digest(candidate.get("digest"), "gate candidate digest") != candidate_digest:
        raise WorkerRuntimeError("gate context candidate digest differs from runtime")


def _resolve_gate_receipt(
    contract: GateReceiptExpectation,
    environment: Mapping[str, str],
    hostname: str,
) -> tuple[GateReceipt, tuple[tuple[str, str], ...]]:
    if contract.scope is EvidenceScope.CPU_STATIC:
        _validate_cpu_static_environment(environment)
        return (
            GateReceipt(
                gate_id=contract.gate_id,
                purpose=contract.purpose,
                scope=contract.scope,
                placements=(),
                product_rank_body=contract.product_rank_body,
                passed=False,
                accuracy=contract.accuracy,
            ),
            (),
        )

    world_size = _runtime_integer(environment, "SLURM_NTASKS")
    rank = _runtime_integer(environment, "SLURM_PROCID")
    local_rank = _runtime_integer(environment, "SLURM_LOCALID")
    observed_shape = (world_size, rank, local_rank)
    expected_shape = (
        contract.expected_world_size,
        contract.expected_rank,
        contract.expected_local_rank,
    )
    if observed_shape != expected_shape:
        raise WorkerRuntimeError(
            f"Slurm runtime shape {observed_shape!r} differs from expected {expected_shape!r}"
        )
    node = environment.get("SLURMD_NODENAME")
    if node is None or _SAFE_NODE_NAME.fullmatch(node) is None:
        raise WorkerRuntimeError("runtime SLURMD_NODENAME must be a safe node name")
    if node != hostname:
        raise WorkerRuntimeError("runtime SLURMD_NODENAME differs from the worker hostname")
    visible_gpu = environment.get("CUDA_VISIBLE_DEVICES")
    if visible_gpu is None or _NUMERIC_GPU_ID.fullmatch(visible_gpu) is None:
        raise WorkerRuntimeError("runtime must expose exactly one numeric CUDA device")
    receipt = GateReceipt(
        gate_id=contract.gate_id,
        purpose=contract.purpose,
        scope=contract.scope,
        placements=(RankPlacement(rank=rank, node=node, local_rank=local_rank),),
        product_rank_body=contract.product_rank_body,
        passed=False,
        accuracy=contract.accuracy,
    )
    trusted_environment = tuple(
        (name, environment[name]) for name in _TRUSTED_SINGLE_GPU_ENVIRONMENT
    )
    return receipt, trusted_environment


def _validate_cpu_static_environment(environment: Mapping[str, str]) -> None:
    contaminated = sorted(_CPU_FORBIDDEN_PLACEMENT_ENVIRONMENT.intersection(environment))
    if contaminated:
        raise WorkerRuntimeError(
            f"CPU-static gate received GPU/distributed placement environment: {contaminated!r}"
        )
    observed = {
        name: environment[name] for name in _CPU_STATIC_LAUNCHER_ENVIRONMENT if name in environment
    }
    if not observed:
        return
    if observed != _CPU_STATIC_LAUNCHER_ENVIRONMENT:
        raise WorkerRuntimeError(
            "CPU-static gate requires the exact single-task Slurm launcher environment"
        )


def _runtime_integer(environment: Mapping[str, str], name: str) -> int:
    value = environment.get(name)
    if value is None or not value.isdigit():
        raise WorkerRuntimeError(f"runtime {name} must be a non-negative integer")
    return int(value)


def _gate_receipt_payload(
    receipt: GateReceipt,
    *,
    certification_mode: str,
    passed: bool,
) -> dict[str, JsonValue]:
    accuracy: dict[str, JsonValue] | None = None
    if receipt.accuracy is not None:
        accuracy = {
            "selector": receipt.accuracy.selector,
            "reference": receipt.accuracy.reference,
            "protocol": receipt.accuracy.protocol,
            "tolerance": receipt.accuracy.tolerance,
        }
    return {
        "gate_id": receipt.gate_id,
        "purpose": receipt.purpose.value,
        "scope": receipt.scope.value,
        "certification_mode": certification_mode,
        "placements": [
            {
                "rank": placement.rank,
                "node": placement.node,
                "local_rank": placement.local_rank,
            }
            for placement in receipt.placements
        ],
        "product_rank_body": receipt.product_rank_body,
        "passed": passed,
        "accuracy": accuracy,
    }


def _gate_cwd(worktree: Path, relative_path: str) -> Path:
    if relative_path == ".":
        return worktree
    candidate = worktree.joinpath(*PurePosixPath(relative_path).parts)
    if candidate.is_symlink() or not candidate.is_dir():
        raise WorkerRuntimeError(f"gate cwd is not a regular directory: {relative_path!r}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(worktree)
    except ValueError as error:
        raise WorkerRuntimeError("gate cwd resolves outside the immutable worktree") from error
    return resolved


def _run_gate_command(
    command: GateCommandSpec,
    cwd: Path,
    trusted_environment: Sequence[tuple[str, str]],
) -> dict[str, JsonValue]:
    environment = dict(command.environment)
    for name, value in trusted_environment:
        if name in environment:
            raise WorkerRuntimeError(f"trusted placement environment collision for {name!r}")
        environment[name] = value
    try:
        completed = subprocess.run(
            command.argv,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
        )
    except OSError as error:
        return {
            "schema_version": WORKER_SCHEMA_VERSION,
            "command_id": command.command_id,
            "argv": list(command.argv),
            "cwd": command.cwd,
            "environment_names": sorted(environment),
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "execution_error": str(error),
        }
    return {
        "schema_version": WORKER_SCHEMA_VERSION,
        "command_id": command.command_id,
        "argv": list(command.argv),
        "cwd": command.cwd,
        "environment_names": sorted(environment),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "execution_error": None,
    }


def _result(
    manifest: WorkerInputManifest,
    input_digest: str,
    attempt_dir: Path,
    *,
    status: WorkerResultStatus,
    summary: str,
    evidence_paths: Sequence[str],
    candidate_digest: str | None,
    reviewed_candidate_digest: str | None,
    payload: dict[str, JsonValue],
) -> WorkerResultManifest:
    evidence = tuple(describe_evidence(attempt_dir, path) for path in evidence_paths)
    return WorkerResultManifest(
        run_id=manifest.run_id,
        item_id=manifest.item_id,
        attempt_id=manifest.attempt_id,
        task_digest=manifest.task_digest,
        generation=manifest.generation,
        input_digest=input_digest,
        status=status,
        summary=summary,
        evidence=evidence,
        candidate_digest=candidate_digest,
        reviewed_candidate_digest=reviewed_candidate_digest,
        payload=payload,
    )


def _failed_result(
    manifest: WorkerInputManifest,
    input_digest: str,
    summary: str,
) -> WorkerResultManifest:
    return WorkerResultManifest(
        run_id=manifest.run_id,
        item_id=manifest.item_id,
        attempt_id=manifest.attempt_id,
        task_digest=manifest.task_digest,
        generation=manifest.generation,
        input_digest=input_digest,
        status=WorkerResultStatus.FAILED,
        summary=summary,
        payload={"failure_kind": "worker_contract"},
    )


def _regular_member(root: Path, relative_path: str, name: str) -> Path:
    normalized = _relative_path(relative_path, name)
    candidate = root.joinpath(*PurePosixPath(normalized).parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise WorkerRuntimeError(f"{name} is not a regular file: {relative_path!r}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise WorkerRuntimeError(f"{name} resolves outside the worktree") from error
    return resolved


def _evidence_destination(attempt_dir: Path, relative_path: str) -> Path:
    normalized = _relative_path(relative_path, "evidence destination")
    parts = PurePosixPath(normalized).parts
    parent = attempt_dir
    for part in parts[:-1]:
        parent /= part
        if parent.exists():
            if parent.is_symlink() or not parent.is_dir():
                raise WorkerRuntimeError("evidence parent must be a regular directory")
        else:
            parent.mkdir(mode=0o700)
    destination = parent / parts[-1]
    if destination.exists() or destination.is_symlink():
        raise WorkerRuntimeError(f"immutable evidence already exists: {relative_path!r}")
    return destination


def _copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
    except OSError:
        try:
            destination.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_exclusive_json(path: Path, payload: Mapping[str, JsonValue]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, allow_nan=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _load_json_object(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise WorkerRuntimeError(f"worker input is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise WorkerRuntimeError(f"invalid JSON worker input: {path}") from error
    return _required_object(value, "worker input")


def _schema_version(data: Mapping[str, object]) -> None:
    if data["schema_version"] != WORKER_SCHEMA_VERSION:
        raise WorkerRuntimeError(f"schema_version must be {WORKER_SCHEMA_VERSION}")


def _exact_keys(data: Mapping[str, object], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(data))
    unknown = sorted(set(data) - expected)
    if missing or unknown:
        raise WorkerRuntimeError(f"{name} keys invalid; missing={missing}, unknown={unknown}")


def _required_object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise WorkerRuntimeError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _json_object(value: object, name: str) -> dict[str, JsonValue]:
    data = _required_object(value, name)
    try:
        json.dumps(data, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise WorkerRuntimeError(f"{name} must contain only finite JSON values") from error
    return cast(dict[str, JsonValue], data)


def _required_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise WorkerRuntimeError(f"{name} must be a list")
    return value


def _plain_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise WorkerRuntimeError(f"{name} must be a non-empty single-line string")
    return value


def _required_string(data: Mapping[str, object], name: str) -> str:
    value = data[name]
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise WorkerRuntimeError(f"{name} must be a non-empty string")
    return value


def _required_bool(data: Mapping[str, object], name: str) -> bool:
    value = data[name]
    if not isinstance(value, bool):
        raise WorkerRuntimeError(f"{name} must be a boolean")
    return value


def _required_int(data: Mapping[str, object], name: str) -> int:
    value = data[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerRuntimeError(f"{name} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkerRuntimeError(f"{name} must be null or a non-negative integer")
    return value


def _safe_id(data: Mapping[str, object], name: str) -> str:
    value = _required_string(data, name)
    if not _SAFE_ID.fullmatch(value):
        raise WorkerRuntimeError(f"{name} must be a safe identifier")
    return value


def _optional_digest(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise WorkerRuntimeError(f"{name} must be null or a lowercase SHA-256 digest")
    return value


def _relative_paths(value: object, name: str) -> tuple[str, ...]:
    paths = tuple(
        _relative_path(_plain_string(entry, f"{name} entry"), name)
        for entry in _required_list(value, name)
    )
    if len(set(paths)) != len(paths):
        raise WorkerRuntimeError(f"{name} cannot contain duplicates")
    return paths


def _relative_path(value: str, name: str, *, allow_dot: bool = False) -> str:
    path = PurePosixPath(value)
    if allow_dot and value == ".":
        return value
    if (
        value in {"", ".", ".."}
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise WorkerRuntimeError(f"{name} must be a canonical POSIX-relative path")
    return value


__all__ = [
    "AgentOutcome",
    "AgentRuntimeSpec",
    "GateCommandSpec",
    "GateReceiptExpectation",
    "GateRuntimeSpec",
    "WorkerRuntimeError",
    "execute_worker",
    "is_worker_input_manifest",
]
