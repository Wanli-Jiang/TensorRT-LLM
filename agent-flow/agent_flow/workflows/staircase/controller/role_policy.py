# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure task-policy resolution for one Staircase agent attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..common.credentials import BACKEND_CREDENTIAL_ALLOWLIST
from ..common.isolation import (
    IsolationPolicyError,
    WorkerEnvironment,
    build_gate_worker_environment,
)
from ..common.launch_policy import AgentLaunchPolicy, AgentLaunchPolicyError
from ..state import DomainProfile, Role
from ..task_schema import (
    CredentialBrokerPolicy,
    NormalizedTask,
    ResourceClass,
    RoleNetworkPolicy,
    SlurmRole,
    SlurmRoleClass,
)


class RolePolicyError(ValueError):
    """Raised when an attempt cannot bind to one exact task role policy."""


@dataclass(frozen=True, slots=True)
class ResolvedRolePolicy:
    """Secret-free task policy ready for a later worker-launch adapter."""

    slurm_role: SlurmRole
    role_class: SlurmRoleClass
    resource: ResourceClass
    environment: WorkerEnvironment
    credential_broker_policy: CredentialBrokerPolicy
    launch_policy: AgentLaunchPolicy


_ROLE_BINDINGS: dict[tuple[Role, DomainProfile | None], SlurmRole] = {
    (Role.PLAN_DRAFTER, None): SlurmRole.PLAN_DRAFTER,
    (Role.PLAN_REVIEWER, None): SlurmRole.PLAN_REVIEWER,
    (Role.CODER, DomainProfile.SMITH): SlurmRole.SMITH_CODER,
    (Role.CODER, DomainProfile.ASSEMBLER): SlurmRole.ASSEMBLER_CODER,
    (Role.CODER, DomainProfile.TUNER): SlurmRole.TUNER_CODER,
    (Role.REVIEWER, DomainProfile.SMITH): SlurmRole.REVIEWER,
    (Role.REVIEWER, DomainProfile.ASSEMBLER): SlurmRole.REVIEWER,
    (Role.REVIEWER, DomainProfile.TUNER): SlurmRole.REVIEWER,
    (Role.QA, DomainProfile.SMITH): SlurmRole.QA,
    (Role.QA, DomainProfile.ASSEMBLER): SlurmRole.QA,
    (Role.QA, DomainProfile.TUNER): SlurmRole.QA,
}


def resolve_role_policy(
    task: NormalizedTask,
    *,
    role: Role,
    profile: DomainProfile | None,
    resource_class: str | None,
    ambient_environment: Mapping[str, str],
) -> ResolvedRolePolicy:
    """Resolve one exact agent role without reading process-global state.

    ``ambient_environment`` is an explicit caller-owned mapping. Only names in
    the selected role class's non-secret allowlist are accessed; all other
    ambient entries are ignored. Credentials remain represented solely by the
    public broker policy frozen into the task.
    """
    slurm_role = _resolve_slurm_role(role, profile)
    role_class = _exact_role_class(task, slurm_role)
    if not isinstance(resource_class, str) or not resource_class:
        raise RolePolicyError(f"{slurm_role.value} attempt is missing its resource class")
    if resource_class != role_class.resource_class:
        raise RolePolicyError(
            f"{slurm_role.value} attempt resource {resource_class!r} differs from task binding "
            f"{role_class.resource_class!r}"
        )
    resource = _exact_resource(task, role_class)
    environment = _resolve_environment(role_class, ambient_environment)
    credential_policy = _validate_credential_policy(task, role_class)
    worker = task.execution.slurm.agent_worker
    if worker is None:
        raise RolePolicyError("task role policy requires an explicit agent worker image")
    try:
        launch_policy = AgentLaunchPolicy.create(
            image=str(worker.image),
            build_identity=worker.build_identity,
            network_policy=role_class.network_policy.value,
            network_enforcement=worker.network_enforcement.value,
        )
    except AgentLaunchPolicyError as error:
        raise RolePolicyError(f"unsafe agent launch policy: {error}") from error
    return ResolvedRolePolicy(
        slurm_role,
        role_class,
        resource,
        environment,
        credential_policy,
        launch_policy,
    )


def _resolve_slurm_role(role: Role, profile: DomainProfile | None) -> SlurmRole:
    if not isinstance(role, Role):
        raise RolePolicyError("role policy requires a typed Role")
    if profile is not None and not isinstance(profile, DomainProfile):
        raise RolePolicyError("role policy profile must use DomainProfile or None")
    try:
        return _ROLE_BINDINGS[(role, profile)]
    except KeyError:
        profile_name = None if profile is None else profile.value
        raise RolePolicyError(
            f"role/profile combination is not an agent role class: {role.value}/{profile_name}"
        ) from None


def _exact_role_class(task: NormalizedTask, slurm_role: SlurmRole) -> SlurmRoleClass:
    matches = tuple(
        policy for policy in task.execution.slurm.role_classes if policy.role is slurm_role
    )
    if len(matches) != 1:
        raise RolePolicyError(
            f"task must define exactly one {slurm_role.value!r} role class; found {len(matches)}"
        )
    role_class = matches[0]
    if role_class.network_policy is not RoleNetworkPolicy.BACKEND_API_ONLY:
        raise RolePolicyError(
            f"{slurm_role.value} role class must use backend_api_only network policy"
        )
    return role_class


def _exact_resource(task: NormalizedTask, role_class: SlurmRoleClass) -> ResourceClass:
    matches = tuple(
        resource
        for resource in task.execution.slurm.smith.resource_classes
        if resource.name == role_class.resource_class
    )
    if len(matches) != 1:
        raise RolePolicyError(
            f"task role {role_class.role.value!r} must resolve exactly one resource "
            f"{role_class.resource_class!r}; found {len(matches)}"
        )
    resource = matches[0]
    if resource.nodes != 1 or resource.tasks_per_node != 1:
        raise RolePolicyError(
            f"task role {role_class.role.value!r} must bind a single-node, single-task resource"
        )
    return resource


def _resolve_environment(
    role_class: SlurmRoleClass,
    ambient_environment: Mapping[str, str],
) -> WorkerEnvironment:
    policy = role_class.environment
    if len(policy.allowlist) != len(set(policy.allowlist)):
        raise RolePolicyError("role environment allowlist contains duplicate names")
    fixed_names = tuple(name for name, _value in policy.values)
    if fixed_names != tuple(sorted(set(fixed_names))):
        raise RolePolicyError("fixed role environment must have sorted unique names")
    if set(policy.allowlist).intersection(fixed_names):
        raise RolePolicyError("role environment names cannot be both fixed and ambient")

    try:
        fixed = build_gate_worker_environment(dict(policy.values))
        selected = dict(fixed.values)
        for name in policy.allowlist:
            build_gate_worker_environment({name: "validation"})
            try:
                selected[name] = ambient_environment[name]
            except KeyError:
                raise RolePolicyError(
                    f"explicit ambient environment is missing allowlisted name {name!r}"
                ) from None
        return build_gate_worker_environment(selected)
    except (IsolationPolicyError, TypeError, ValueError) as error:
        if isinstance(error, RolePolicyError):
            raise
        raise RolePolicyError(f"unsafe role environment policy: {error}") from error


def _validate_credential_policy(
    task: NormalizedTask,
    role_class: SlurmRoleClass,
) -> CredentialBrokerPolicy:
    policy = role_class.credential_broker
    if not isinstance(policy.broker_id, str) or not policy.broker_id:
        raise RolePolicyError("credential broker policy is missing its public broker id")
    if (
        isinstance(policy.per_attempt_ttl_seconds, bool)
        or not isinstance(policy.per_attempt_ttl_seconds, int)
        or policy.per_attempt_ttl_seconds < 1
    ):
        raise RolePolicyError("credential broker policy requires a positive per-attempt TTL")
    names = policy.allowed_credential_names
    if tuple(sorted(set(names))) != names:
        raise RolePolicyError("credential broker names must be sorted and unique")
    allowed = BACKEND_CREDENTIAL_ALLOWLIST.get(task.execution.agent.backend_kind)
    if allowed is None or not set(names).issubset(allowed):
        raise RolePolicyError(
            "credential broker policy differs from the task agent backend credential boundary"
        )
    return policy


__all__ = ["ResolvedRolePolicy", "RolePolicyError", "resolve_role_policy"]
