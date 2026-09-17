# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.controller.role_policy import (
    RolePolicyError,
    resolve_role_policy,
)
from agent_flow.workflows.staircase.state import DomainProfile, Role
from agent_flow.workflows.staircase.task_schema import (
    AccuracyGate,
    AgentExecutionConfig,
    AgentWorkerConfig,
    CertificationConfig,
    CertificationMode,
    ControllerConfig,
    CredentialBrokerPolicy,
    DeliveryConfig,
    ExecutionConfig,
    GatesConfig,
    NetworkEnforcement,
    NormalizedTask,
    OverrideBounds,
    ParallelMapping,
    ReferenceConfig,
    RepositoryConfig,
    ResourceClass,
    RetryPolicy,
    RoleEnvironmentPolicy,
    RoleNetworkPolicy,
    SlurmConfig,
    SlurmRole,
    SlurmRoleClass,
    SmithConfig,
    TargetConfig,
)

_RESOURCE_BY_ROLE = {
    SlurmRole.PLAN_DRAFTER: "coder_analysis",
    SlurmRole.PLAN_REVIEWER: "reviewer_analysis",
    SlurmRole.SMITH_CODER: "coder_analysis",
    SlurmRole.ASSEMBLER_CODER: "coder_analysis",
    SlurmRole.TUNER_CODER: "exploratory_probe",
    SlurmRole.REVIEWER: "reviewer_analysis",
    SlurmRole.QA: "reviewer_rerun",
}


class _ObservedAmbient(Mapping[str, str]):
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values
        self.accessed: list[str] = []

    def __getitem__(self, name: str) -> str:
        self.accessed.append(name)
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("role policy must not enumerate ambient environment")

    def __len__(self) -> int:
        raise AssertionError("role policy must not inspect ambient environment size")


def _task() -> NormalizedTask:
    resources = tuple(
        ResourceClass(name, 1, 1, 0, 2, 1_024, 600)
        for name in (
            "coder_analysis",
            "exploratory_probe",
            "deterministic_gate",
            "reviewer_analysis",
            "reviewer_rerun",
        )
    )
    role_classes = tuple(
        SlurmRoleClass(
            role,
            resource,
            RoleNetworkPolicy.BACKEND_API_ONLY,
            RoleEnvironmentPolicy((), (("PYTHONNOUSERSITE", "1"),)),
            CredentialBrokerPolicy("site-codex-broker", ("OPENAI_API_KEY",), 7_200),
        )
        for role, resource in _RESOURCE_BY_ROLE.items()
    )
    slurm = SlurmConfig(
        ControllerConfig(
            "coreai",
            "batch",
            None,
            None,
            3_600,
            2,
            2_048,
            Path("/containers/trtllm.sqsh"),
            (),
            (),
            "test-build",
            "nested_submission",
            False,
            60,
            120,
            300,
            300,
        ),
        SmithConfig(
            2,
            2,
            0,
            resources,
            OverrideBounds(2, 1, 1, 8, 8_192, 3_600),
            RetryPolicy(1, 1),
        ),
        role_classes=role_classes,
        agent_worker=AgentWorkerConfig(
            Path("/containers/agent-worker.sqsh"),
            "agent-worker-test",
            False,
            NetworkEnforcement.CERTIFIED_WORKER_IMAGE,
        ),
    )
    return NormalizedTask(
        2,
        RepositoryConfig(Path("/repo"), "a" * 40, "reject", Path("/workspaces")),
        ReferenceConfig(Path("/checkpoint"), "test", "TestForCausalLM", ()),
        TargetConfig(
            "test",
            "test-checkpoint",
            100,
            1,
            ParallelMapping(1, 1, 1, 1, 1),
            (),
            "tensorrt_llm/_torch/modeling_v2/models/test",
            True,
        ),
        GatesConfig(AccuracyGate("tests/test.py", "reference", "exact", 0.0), (), (), (), ()),
        CertificationConfig(CertificationMode.LOCAL, None),
        ExecutionConfig("slurm", slurm, AgentExecutionConfig("codex", "test-model")),
        DeliveryConfig("diff_only", None, False, False),
        "b" * 64,
    )


def _replace_role_class(
    task: NormalizedTask,
    slurm_role: SlurmRole,
    **changes: object,
) -> NormalizedTask:
    role_classes = tuple(
        replace(policy, **changes) if policy.role is slurm_role else policy
        for policy in task.execution.slurm.role_classes
    )
    slurm = replace(task.execution.slurm, role_classes=role_classes)
    return replace(task, execution=replace(task.execution, slurm=slurm))


@pytest.mark.parametrize(
    ("role", "profile", "slurm_role"),
    [
        (Role.PLAN_DRAFTER, None, SlurmRole.PLAN_DRAFTER),
        (Role.PLAN_REVIEWER, None, SlurmRole.PLAN_REVIEWER),
        (Role.CODER, DomainProfile.SMITH, SlurmRole.SMITH_CODER),
        (Role.CODER, DomainProfile.ASSEMBLER, SlurmRole.ASSEMBLER_CODER),
        (Role.CODER, DomainProfile.TUNER, SlurmRole.TUNER_CODER),
        (Role.REVIEWER, DomainProfile.SMITH, SlurmRole.REVIEWER),
        (Role.REVIEWER, DomainProfile.ASSEMBLER, SlurmRole.REVIEWER),
        (Role.REVIEWER, DomainProfile.TUNER, SlurmRole.REVIEWER),
        (Role.QA, DomainProfile.SMITH, SlurmRole.QA),
        (Role.QA, DomainProfile.ASSEMBLER, SlurmRole.QA),
        (Role.QA, DomainProfile.TUNER, SlurmRole.QA),
    ],
)
def test_exact_role_profile_mapping_and_public_policy(
    role: Role,
    profile: DomainProfile | None,
    slurm_role: SlurmRole,
) -> None:
    task = _task()
    expected_policy = task.execution.slurm.role_class(slurm_role)

    resolved = resolve_role_policy(
        task,
        role=role,
        profile=profile,
        resource_class=expected_policy.resource_class,
        ambient_environment={},
    )

    assert resolved.slurm_role is slurm_role
    assert resolved.role_class is expected_policy
    assert resolved.resource.name == expected_policy.resource_class
    assert resolved.resource.nodes == resolved.resource.tasks_per_node == 1
    assert resolved.environment.mapping == {"PYTHONNOUSERSITE": "1"}
    assert resolved.credential_broker_policy is expected_policy.credential_broker


@pytest.mark.parametrize(
    ("role", "profile"),
    [
        (Role.PLAN_DRAFTER, DomainProfile.SMITH),
        (Role.PLAN_REVIEWER, DomainProfile.ASSEMBLER),
        (Role.CODER, None),
        (Role.REVIEWER, None),
        (Role.QA, None),
        (Role.GATE, DomainProfile.SMITH),
    ],
)
def test_role_profile_confusion_is_rejected(
    role: Role,
    profile: DomainProfile | None,
) -> None:
    with pytest.raises(RolePolicyError, match="role/profile combination"):
        resolve_role_policy(
            _task(),
            role=role,
            profile=profile,
            resource_class="coder_analysis",
            ambient_environment={},
        )


def test_fixed_and_explicit_ambient_environment_are_the_only_exports() -> None:
    task = _replace_role_class(
        _task(),
        SlurmRole.SMITH_CODER,
        environment=RoleEnvironmentPolicy(
            ("HTTP_PROXY",),
            (("PYTHONNOUSERSITE", "1"),),
        ),
    )
    ambient = _ObservedAmbient(
        {
            "HTTP_PROXY": "http://proxy.example:8080",
            "OPENAI_API_KEY": "must-not-be-read",
            "SLURM_JOB_ID": "must-not-be-read",
        }
    )

    resolved = resolve_role_policy(
        task,
        role=Role.CODER,
        profile=DomainProfile.SMITH,
        resource_class="coder_analysis",
        ambient_environment=ambient,
    )

    assert resolved.environment.mapping == {
        "HTTP_PROXY": "http://proxy.example:8080",
        "PYTHONNOUSERSITE": "1",
    }
    assert ambient.accessed == ["HTTP_PROXY"]


def test_missing_allowlisted_ambient_name_is_rejected() -> None:
    task = _replace_role_class(
        _task(),
        SlurmRole.SMITH_CODER,
        environment=RoleEnvironmentPolicy(("HTTP_PROXY",), ()),
    )
    with pytest.raises(RolePolicyError, match="missing allowlisted name 'HTTP_PROXY'"):
        resolve_role_policy(
            task,
            role=Role.CODER,
            profile=DomainProfile.SMITH,
            resource_class="coder_analysis",
            ambient_environment={},
        )


@pytest.mark.parametrize("name", ["SLURM_JOB_ID", "OPENAI_API_KEY"])
@pytest.mark.parametrize("fixed", [False, True])
def test_scheduler_and_credential_like_environment_names_are_rejected(
    name: str,
    fixed: bool,
) -> None:
    environment = (
        RoleEnvironmentPolicy((), ((name, "value"),))
        if fixed
        else RoleEnvironmentPolicy((name,), ())
    )
    task = _replace_role_class(_task(), SlurmRole.SMITH_CODER, environment=environment)
    with pytest.raises(RolePolicyError, match="unsafe role environment policy"):
        resolve_role_policy(
            task,
            role=Role.CODER,
            profile=DomainProfile.SMITH,
            resource_class="coder_analysis",
            ambient_environment={name: "value"},
        )


def test_missing_duplicate_and_mismatched_role_bindings_are_rejected() -> None:
    task = _task()
    slurm = task.execution.slurm
    missing = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(
                slurm,
                role_classes=tuple(
                    policy
                    for policy in slurm.role_classes
                    if policy.role is not SlurmRole.SMITH_CODER
                ),
            ),
        ),
    )
    with pytest.raises(RolePolicyError, match="exactly one 'smith_coder'.*found 0"):
        resolve_role_policy(
            missing,
            role=Role.CODER,
            profile=DomainProfile.SMITH,
            resource_class="coder_analysis",
            ambient_environment={},
        )

    smith_policy = slurm.role_class(SlurmRole.SMITH_CODER)
    duplicate = replace(
        task,
        execution=replace(
            task.execution,
            slurm=replace(slurm, role_classes=(*slurm.role_classes, smith_policy)),
        ),
    )
    with pytest.raises(RolePolicyError, match="exactly one 'smith_coder'.*found 2"):
        resolve_role_policy(
            duplicate,
            role=Role.CODER,
            profile=DomainProfile.SMITH,
            resource_class="coder_analysis",
            ambient_environment={},
        )

    with pytest.raises(RolePolicyError, match="differs from task binding"):
        resolve_role_policy(
            task,
            role=Role.CODER,
            profile=DomainProfile.SMITH,
            resource_class="reviewer_analysis",
            ambient_environment={},
        )


@pytest.mark.parametrize("resource_mode", ["missing", "duplicate", "multi-node"])
def test_bound_resource_must_be_unique_single_node_and_single_task(resource_mode: str) -> None:
    task = _task()
    smith = task.execution.slurm.smith
    resources = smith.resource_classes
    coder = next(resource for resource in resources if resource.name == "coder_analysis")
    if resource_mode == "missing":
        changed = tuple(resource for resource in resources if resource is not coder)
    elif resource_mode == "duplicate":
        changed = (*resources, coder)
    else:
        changed = tuple(
            replace(resource, nodes=2) if resource is coder else resource for resource in resources
        )
    slurm = replace(task.execution.slurm, smith=replace(smith, resource_classes=changed))
    task = replace(task, execution=replace(task.execution, slurm=slurm))

    with pytest.raises(RolePolicyError, match="resolve exactly one|single-node"):
        resolve_role_policy(
            task,
            role=Role.CODER,
            profile=DomainProfile.SMITH,
            resource_class="coder_analysis",
            ambient_environment={},
        )


def test_credential_policy_must_match_task_backend_without_reading_secret_values() -> None:
    broker = CredentialBrokerPolicy("site-claude-broker", ("ANTHROPIC_API_KEY",), 7_200)
    task = _replace_role_class(_task(), SlurmRole.SMITH_CODER, credential_broker=broker)
    ambient = _ObservedAmbient({"ANTHROPIC_API_KEY": "must-not-be-read"})

    with pytest.raises(RolePolicyError, match="task agent backend credential boundary"):
        resolve_role_policy(
            task,
            role=Role.CODER,
            profile=DomainProfile.SMITH,
            resource_class="coder_analysis",
            ambient_environment=ambient,
        )
    assert ambient.accessed == []


def test_typed_role_and_profile_are_required() -> None:
    with pytest.raises(RolePolicyError, match="typed Role"):
        resolve_role_policy(
            _task(),
            role="coder",  # type: ignore[arg-type]
            profile=DomainProfile.SMITH,
            resource_class="coder_analysis",
            ambient_environment={},
        )
    with pytest.raises(RolePolicyError, match="DomainProfile or None"):
        resolve_role_policy(
            _task(),
            role=Role.CODER,
            profile="smith",  # type: ignore[arg-type]
            resource_class="coder_analysis",
            ambient_environment={},
        )
