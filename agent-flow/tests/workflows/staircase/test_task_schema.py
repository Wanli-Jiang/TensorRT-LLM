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

from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase.task_schema import (
    AgentExecutionConfig,
    CertificationMode,
    NormalizedTask,
    RoleNetworkPolicy,
    SlurmRole,
    TaskSchemaError,
    load_and_normalize_task,
    load_normalized_task,
    normalized_task_dict,
    normalized_task_json,
    write_normalized_task,
)


def _resource_class(
    *,
    nodes: int = 1,
    tasks: int = 1,
    gpus: int = 0,
    gpu_allocation_padding: bool | None = None,
    partition: str | None = None,
    qos: str | None = None,
) -> dict[str, object]:
    resource: dict[str, object] = {
        "nodes": nodes,
        "tasks_per_node": tasks,
        "gpus_per_node": gpus,
        "cpus_per_task": 4,
        "memory": "8G",
        "time_limit": "00:20:00",
    }
    if gpu_allocation_padding is not None:
        resource["gpu_allocation_padding"] = gpu_allocation_padding
    if partition is not None:
        resource["partition"] = partition
    if qos is not None:
        resource["qos"] = qos
    return resource


def _role_classes() -> dict[str, object]:
    resources = {
        "plan_drafter": "coder_analysis",
        "plan_reviewer": "reviewer_analysis",
        "smith_coder": "coder_analysis",
        "assembler_coder": "coder_analysis",
        "tuner_coder": "exploratory_probe",
        "reviewer": "reviewer_analysis",
        "qa": "reviewer_rerun",
    }
    return {
        role: {
            "resource_class": resource,
            "network_policy": "backend_api_only",
            "environment": {
                "allowlist": [],
                "values": {"PYTHONNOUSERSITE": "1"},
            },
            "credential_broker": {
                "broker_id": "site-codex-broker",
                "allowed_credential_names": ["OPENAI_API_KEY"],
                "per_attempt_ttl_seconds": 7_200,
            },
        }
        for role, resource in resources.items()
    }


def _valid_task(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    (repository / "tensorrt_llm" / "_torch" / "modeling_v2" / "models").mkdir(parents=True)
    python_package = repository / "tensorrt_llm" / "__init__.py"
    python_package.write_text("# package\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"architectures": ["ExampleForCausalLM"]}), encoding="utf-8"
    )
    additional = tmp_path / "independent_reference.py"
    additional.write_text("# reference\n", encoding="utf-8")
    image = tmp_path / "trtllm.sqsh"
    image.touch()
    agent_image = tmp_path / "agent-worker.sqsh"
    agent_image.touch()
    return {
        "schema_version": 2,
        "repository": {
            "root": str(repository),
            "base_commit": "a" * 40,
            "dirty_policy": "reject",
            "workspace_root": str(workspace),
        },
        "reference": {
            "checkpoint": str(checkpoint),
            "provenance": "example/model@aabbccdd",
            "architecture": "ExampleForCausalLM",
            "additional_sources": [str(additional)],
        },
        "target": {
            "family": "example_moe",
            "checkpoint_id": "example_4gpu",
            "sm": 100,
            "world_size": 4,
            "mapping": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "moe_expert_parallel_size": 4,
                "moe_tensor_parallel_size": 1,
                "attention_data_parallel_size": 2,
            },
            "features": ["cuda_graph"],
            "expected_route": "tensorrt_llm/_torch/modeling_v2/models/example_moe",
            "synthetic_target": False,
        },
        "gates": {
            "accuracy": {
                "selector": "tests/integration/defs/accuracy/test_modeling_v2_gpt_oss.py::test_accuracy",
                "reference": "independent reference",
                "protocol": "greedy-10",
                "tolerance": 0.0,
            },
            "feature_signal_tests": [
                "tests/integration/defs/accuracy/test_modeling_v2_gpt_oss.py::test_feature_signal"
            ],
            "boot_tests": [
                "tests/integration/defs/accuracy/test_modeling_v2_gpt_oss.py::test_boot"
            ],
            "component_tests": ["tests/unittest/_torch/modeling_v2/test_target.py::test_component"],
            "collective_tests": [],
        },
        "certification": {
            "mode": "REAL",
            "expected_product_identity": {
                "repository_commit": "a" * 40,
                "python_tensorrt_llm_path": str(python_package),
                "native_build_identity": "local-dev-abcdef",
                "image_identity": str(image),
                "compute_capability": "10.0",
                "collective_backend": "nccl",
                "transport": "ib",
            },
        },
        "execution": {
            "mode": "slurm",
            "agent": {"backend_kind": "codex", "model": "gpt-5-codex"},
            "slurm": {
                "agent_worker": {
                    "image": str(agent_image),
                    "build_identity": "agent-worker-abcdef",
                    "scheduler_clients": False,
                    "network_enforcement": "certified_worker_image",
                },
                "role_classes": _role_classes(),
                "gate_classes": {
                    "accuracy": {"resource_class": "deterministic_gate"},
                    "boot": {"resource_class": "deterministic_gate"},
                    "feature-signal": {"resource_class": "deterministic_gate"},
                },
                "controller": {
                    "account": "coreai",
                    "partition": "batch",
                    "qos": None,
                    "reservation": None,
                    "time_limit": "01:00:00",
                    "cpus_per_task": 8,
                    "memory": "16G",
                    "image": str(image),
                    "scheduler_clients": True,
                    "container_launch_mode": "in_allocation_srun",
                    "mounts": [
                        {
                            "host_path": str(tmp_path),
                            "container_path": str(tmp_path),
                            "read_only": False,
                        }
                    ],
                    "environment": {"PYTHONNOUSERSITE": "1"},
                    "build_identity": "local-dev-abcdef",
                    "dispatch_mode": "nested_submission",
                    "requeue": True,
                    "advance_signal_lead_seconds": 60,
                    "heartbeat_timeout_seconds": 30,
                    "lease_timeout_seconds": 90,
                    "orphan_grace_seconds": 120,
                    "credential_broker_socket": str(tmp_path / "credential-broker.sock"),
                },
                "smith": {
                    "max_parallel_items": 4,
                    "max_nodes_total": 8,
                    "max_gpus_total": 16,
                    "distinct_nodes_required": True,
                    "exclusive": True,
                    "resource_classes": {
                        "coder_analysis": _resource_class(),
                        "exploratory_probe": _resource_class(gpus=1),
                        "deterministic_gate": _resource_class(nodes=2, tasks=2, gpus=2),
                        "reviewer_analysis": _resource_class(),
                        "reviewer_rerun": _resource_class(gpus=1),
                    },
                    "per_item_override_bounds": {
                        "max_nodes": 2,
                        "max_tasks_per_node": 2,
                        "max_gpus_per_node": 2,
                        "max_cpus_per_task": 8,
                        "max_memory": "16G",
                        "max_time_limit": "01:00:00",
                    },
                    "retry_policy": {"preempted": 2, "node_failure": 1},
                },
            },
        },
        "delivery": {"mode": "diff_only", "merge": False, "push": False},
    }


def _add_native_artifact_bundle(task: dict[str, object], tmp_path: Path) -> None:
    manifest = tmp_path / "native-manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    certification = task["certification"]
    assert isinstance(certification, dict)
    certification["native_artifact_bundle"] = {
        "package_root": str(tmp_path / "repo" / "tensorrt_llm"),
        "manifest": str(manifest),
        "manifest_sha256": "b" * 64,
    }


def _write_task(tmp_path: Path, task: dict[str, object], name: str = "task.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    return path


def _nested(task: dict[str, object], *keys: str) -> dict[str, object]:
    current = task
    for key in keys:
        value = current[key]
        assert isinstance(value, dict)
        current = value
    return current


def _enable_managed_codex_oauth(task: dict[str, object], tmp_path: Path) -> Path:
    source = tmp_path.parent / f"{tmp_path.name}-codex-auth.json"
    source.write_text('{"auth_mode":"chatgpt"}', encoding="utf-8")
    source.chmod(0o600)
    controller = _nested(task, "execution", "slurm", "controller")
    runtime_name = tmp_path.name.replace("_", "-")
    controller["credential_broker_socket"] = (
        f"/tmp/staircase-credential-broker-{runtime_name}/broker.sock"
    )
    controller["source_auth_json_path"] = str(source)
    mounts = controller["mounts"]
    assert isinstance(mounts, list)
    mounts.append(
        {
            "host_path": str(source),
            "container_path": str(source),
            "read_only": True,
        }
    )
    roles = _nested(task, "execution", "slurm", "role_classes")
    for role in roles.values():
        assert isinstance(role, dict)
        broker = role["credential_broker"]
        assert isinstance(broker, dict)
        broker["allowed_credential_names"] = ["CODEX_AUTH_JSON"]
    return source


def test_normalizes_immutable_task_and_stable_digest(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    first = load_and_normalize_task(_write_task(tmp_path, task, "first.yaml"))

    reordered = {key: task[key] for key in reversed(task)}
    second = load_and_normalize_task(_write_task(tmp_path, reordered, "second.yaml"))

    assert isinstance(first, NormalizedTask)
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert first.repository.root.is_absolute()
    assert first.certification.mode is CertificationMode.REAL
    assert first.execution.agent == AgentExecutionConfig("codex", "gpt-5-codex")
    assert first.execution.slurm.controller.memory_mib == 16 * 1024
    assert first.execution.slurm.controller.scheduler_clients
    assert first.execution.slurm.agent_worker is not None
    assert not first.execution.slurm.agent_worker.scheduler_clients
    assert first.execution.slurm.smith.distinct_nodes_required
    assert first.execution.slurm.smith.exclusive
    assert first.execution.slurm.smith.resource_class("deterministic_gate").total_gpus == 4
    assert first.execution.slurm.smith.resource_class("reviewer_analysis").partition is None
    assert first.execution.slurm.smith.resource_class("reviewer_analysis").qos is None
    assert json.loads(normalized_task_json(first))["digest"] == first.digest
    assert normalized_task_dict(first)["digest"] == first.digest
    with pytest.raises(FrozenInstanceError):
        setattr(first, "schema_version", 2)


def test_native_artifact_bundle_is_normalized_and_changes_task_digest(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    without_bundle = load_and_normalize_task(_write_task(tmp_path, task, "without.yaml"))
    _add_native_artifact_bundle(task, tmp_path)
    certification = _nested(task, "certification")
    certification["mode"] = "LOCAL"
    certification["expected_product_identity"] = None
    target = _nested(task, "target")
    target["world_size"] = 1
    mapping = _nested(task, "target", "mapping")
    for name in mapping:
        mapping[name] = 1
    resource_classes = _nested(task, "execution", "slurm", "smith", "resource_classes")
    resource_classes["deterministic_gate"] = _resource_class(nodes=1, tasks=1, gpus=1)

    with_bundle = load_and_normalize_task(_write_task(tmp_path, task, "with.yaml"))

    assert with_bundle.certification.native_artifact_bundle is not None
    assert with_bundle.certification.mode is CertificationMode.LOCAL
    assert with_bundle.certification.expected_product_identity is None
    assert with_bundle.execution.mode == "slurm"
    assert with_bundle.certification.native_artifact_bundle.manifest_sha256 == "b" * 64
    assert with_bundle.digest != without_bundle.digest
    serialized = normalized_task_dict(with_bundle)
    certification = serialized["certification"]
    assert isinstance(certification, dict)
    native_bundle = certification["native_artifact_bundle"]
    assert isinstance(native_bundle, dict)
    assert native_bundle["manifest_sha256"] == "b" * 64


def test_native_artifact_bundle_is_prohibited_for_synthetic_mode(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    _add_native_artifact_bundle(task, tmp_path)
    certification = _nested(task, "certification")
    certification["mode"] = "SYNTHETIC"
    certification["expected_product_identity"] = None
    gates = _nested(task, "gates")
    gates["collective_tests"] = [
        "tests/integration/defs/accuracy/test_modeling_v2_gpt_oss.py::test_collective"
    ]
    _nested(task, "execution", "slurm")["gate_classes"] = {
        "collective": {"resource_class": "deterministic_gate"}
    }

    with pytest.raises(TaskSchemaError, match="prohibited for SYNTHETIC"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_native_artifact_bundle_fails_closed_for_collective_gates(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    _add_native_artifact_bundle(task, tmp_path)
    gates = _nested(task, "gates")
    gates["collective_tests"] = [
        "tests/integration/defs/accuracy/test_modeling_v2_gpt_oss.py::test_collective"
    ]
    gate_classes = _nested(task, "execution", "slurm", "gate_classes")
    gate_classes["collective"] = {"resource_class": "deterministic_gate"}

    with pytest.raises(TaskSchemaError, match="generic single-process gates"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_controller_and_agent_images_have_disjoint_capability_contracts(
    tmp_path: Path,
) -> None:
    task = _valid_task(tmp_path)
    controller = _nested(task, "execution", "slurm", "controller")
    worker = _nested(task, "execution", "slurm", "agent_worker")
    worker["image"] = controller["image"]
    with pytest.raises(TaskSchemaError, match="must differ"):
        load_and_normalize_task(_write_task(tmp_path, task))

    task = _valid_task(tmp_path / "scheduler-client")
    _nested(task, "execution", "slurm", "agent_worker")["scheduler_clients"] = True
    with pytest.raises(TaskSchemaError, match="scheduler_clients.*must be false"):
        load_and_normalize_task(_write_task(tmp_path / "scheduler-client", task))

    task = _valid_task(tmp_path / "same-build")
    controller = _nested(task, "execution", "slurm", "controller")
    _nested(task, "execution", "slurm", "agent_worker")["build_identity"] = controller[
        "build_identity"
    ]
    with pytest.raises(TaskSchemaError, match="build_identity.*must differ"):
        load_and_normalize_task(_write_task(tmp_path / "same-build", task))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OPENAI_API_KEY", "not-even-a-real-secret"),
        ("SITE_HEADER", "Bearer abcdefghijklmnop"),
    ],
)
def test_controller_environment_rejects_credential_like_names_and_values(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    task = _valid_task(tmp_path)
    _nested(task, "execution", "slurm", "controller")["environment"] = {name: value}
    with pytest.raises(TaskSchemaError, match="credential-like controller environment"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_controller_rejects_wrapped_credential_assignment_without_echo(tmp_path: Path) -> None:
    secret = "X=OPENAI_API_KEY=opaqueCredentialMaterial123456789ABC"
    task = _valid_task(tmp_path)
    _nested(task, "execution", "slurm", "controller")["environment"] = {"PYTHONNOUSERSITE": secret}

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "credential-like controller environment value" in message
    assert secret not in message


@pytest.mark.parametrize(
    "role",
    [
        "plan_drafter",
        "plan_reviewer",
        "smith_coder",
        "assembler_coder",
        "tuner_coder",
        "reviewer",
        "qa",
    ],
)
@pytest.mark.parametrize(
    "secret",
    [
        "BeArEr\u2003AbCdEfGhIjKlMnOp",
        "SK-AbCdEfGhIjKlMnOp",
        "-----begin private key-----",
    ],
)
def test_every_role_rejects_secret_shaped_fixed_environment_values_without_echo(
    tmp_path: Path,
    role: str,
    secret: str,
) -> None:
    task = _valid_task(tmp_path)
    environment = _nested(task, "execution", "slurm", "role_classes", role, "environment")
    environment["values"] = {"PYTHONNOUSERSITE": secret}

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "credential-like role environment value" in message
    assert secret not in message


@pytest.mark.parametrize(
    "role",
    [
        "plan_drafter",
        "plan_reviewer",
        "smith_coder",
        "assembler_coder",
        "tuner_coder",
        "reviewer",
        "qa",
    ],
)
def test_every_role_rejects_wrapped_credential_assignment_without_echo(
    tmp_path: Path,
    role: str,
) -> None:
    secret = "X=OPENAI_API_KEY=opaqueCredentialMaterial123456789ABC"
    task = _valid_task(tmp_path)
    environment = _nested(task, "execution", "slurm", "role_classes", role, "environment")
    environment["values"] = {"PYTHONNOUSERSITE": secret}

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "credential-like role environment value" in message
    assert secret not in message


@pytest.mark.parametrize("value", ["/shared/cache", "实验-α", "X=MODE=offline"])
def test_controller_and_roles_allow_benign_environment_values(
    tmp_path: Path,
    value: str,
) -> None:
    task = _valid_task(tmp_path)
    _nested(task, "execution", "slurm", "controller")["environment"] = {"TLLM_LOG_LEVEL": value}
    role_classes = _nested(task, "execution", "slurm", "role_classes")
    for role_config in role_classes.values():
        assert isinstance(role_config, dict)
        environment = role_config["environment"]
        assert isinstance(environment, dict)
        environment["values"] = {"TLLM_LOG_LEVEL": value}

    normalized = load_and_normalize_task(_write_task(tmp_path, task))

    assert dict(normalized.execution.slurm.controller.environment)["TLLM_LOG_LEVEL"] == value
    assert all(
        dict(role_class.environment.values)["TLLM_LOG_LEVEL"] == value
        for role_class in normalized.execution.slurm.role_classes
    )


def test_role_fixed_environment_allows_unicode_nonsecret_value_and_round_trips(
    tmp_path: Path,
) -> None:
    task = _valid_task(tmp_path)
    environment = _nested(
        task,
        "execution",
        "slurm",
        "role_classes",
        "plan_drafter",
        "environment",
    )
    environment["values"] = {"SITE_LABEL": "实验-α"}
    normalized = load_and_normalize_task(_write_task(tmp_path, task))
    normalized_path = tmp_path / "normalized-task.json"

    write_normalized_task(normalized_path, normalized)
    restored = load_normalized_task(normalized_path)

    assert restored == normalized
    assert restored.execution.slurm.role_class(SlurmRole.PLAN_DRAFTER).environment.values == (
        ("SITE_LABEL", "实验-α"),
    )


@pytest.mark.parametrize(
    "secret",
    [
        "Basic dXNlcjpwYXNzd29yZA==",
        "https://user:password@example.test/cache",
        "prefix OPENAI_API_KEY=opaque-value",
        "A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
    ],
)
def test_controller_rejects_extended_secret_shapes_under_allowlisted_path_name(
    tmp_path: Path,
    secret: str,
) -> None:
    task = _valid_task(tmp_path)
    _nested(task, "execution", "slurm", "controller")["environment"] = {"HF_HOME": secret}

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "environment value" in message or "path value" in message
    assert secret not in message


@pytest.mark.parametrize(
    "role",
    [
        "plan_drafter",
        "plan_reviewer",
        "smith_coder",
        "assembler_coder",
        "tuner_coder",
        "reviewer",
        "qa",
    ],
)
@pytest.mark.parametrize(
    "secret",
    [
        "Basic dXNlcjpwYXNzd29yZA==",
        "https://user:password@example.test/cache",
        "OPENAI_API_KEY=opaque-value",
        "A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
    ],
)
def test_every_role_rejects_extended_secret_shapes_under_allowlisted_path_name(
    tmp_path: Path,
    role: str,
    secret: str,
) -> None:
    task = _valid_task(tmp_path)
    environment = _nested(task, "execution", "slurm", "role_classes", role, "environment")
    environment["values"] = {"LLM_MODELS_ROOT": secret}

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "environment value" in message or "path value" in message
    assert secret not in message


def test_allowlisted_path_environment_preserves_absolute_unicode_path(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    _nested(task, "execution", "slurm", "controller")["environment"] = {"HF_HOME": "/共享/缓存-α"}
    role_environment = _nested(
        task,
        "execution",
        "slurm",
        "role_classes",
        "qa",
        "environment",
    )
    role_environment["values"] = {"LLM_MODELS_ROOT": "/共享/模型-β"}

    normalized = load_and_normalize_task(_write_task(tmp_path, task))

    assert dict(normalized.execution.slurm.controller.environment)["HF_HOME"] == "/共享/缓存-α"
    assert (
        dict(normalized.execution.slurm.role_class(SlurmRole.QA).environment.values)[
            "LLM_MODELS_ROOT"
        ]
        == "/共享/模型-β"
    )


def test_distinct_node_smith_requires_exclusive_allocations(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    _nested(task, "execution", "slurm", "smith")["exclusive"] = False
    with pytest.raises(TaskSchemaError, match="exclusive.*must be true"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_local_override_is_explicit_and_changes_effective_digest(tmp_path: Path) -> None:
    task_path = _write_task(tmp_path, _valid_task(tmp_path))
    slurm = load_and_normalize_task(task_path)
    local = load_and_normalize_task(task_path, execution_override="local")

    assert slurm.execution.mode == "slurm"
    assert local.execution.mode == "local"
    assert local.execution.agent == slurm.execution.agent
    assert local.execution.slurm == slurm.execution.slurm
    assert local.digest != slurm.digest


def test_role_classes_are_normalized_public_and_have_typed_lookup(tmp_path: Path) -> None:
    task = load_and_normalize_task(_write_task(tmp_path, _valid_task(tmp_path)))

    assert tuple(role_class.role for role_class in task.execution.slurm.role_classes) == tuple(
        SlurmRole(role) for role in _role_classes()
    )
    drafter = task.execution.slurm.role_class(SlurmRole.PLAN_DRAFTER)
    assert drafter.resource_class == "coder_analysis"
    assert drafter.network_policy is RoleNetworkPolicy.BACKEND_API_ONLY
    assert drafter.environment.allowlist == ()
    assert drafter.environment.value("PYTHONNOUSERSITE") == "1"
    assert drafter.credential_broker.broker_id == "site-codex-broker"
    assert drafter.credential_broker.allowed_credential_names == ("OPENAI_API_KEY",)
    assert drafter.credential_broker.per_attempt_ttl_seconds == 7_200
    assert task.execution.slurm.role_class("qa").role is SlurmRole.QA
    with pytest.raises(KeyError):
        task.execution.slurm.role_class("unknown")

    payload = json.loads(normalized_task_json(task))
    role_payload = payload["execution"]["slurm"]["role_classes"]
    assert len(role_payload) == len(_role_classes())
    assert role_payload[0]["credential_broker"] == {
        "allowed_credential_names": ["OPENAI_API_KEY"],
        "broker_id": "site-codex-broker",
        "per_attempt_ttl_seconds": 7_200,
    }
    assert "handle_id" not in json.dumps(role_payload)


def test_credentialed_roles_require_one_socket_and_identical_broker_policy(
    tmp_path: Path,
) -> None:
    missing_socket = _valid_task(tmp_path / "missing-socket")
    controller = _nested(missing_socket, "execution", "slurm", "controller")
    del controller["credential_broker_socket"]
    with pytest.raises(TaskSchemaError, match="credential_broker_socket.*required"):
        load_and_normalize_task(_write_task(tmp_path / "missing-socket", missing_socket))

    divergent = _valid_task(tmp_path / "divergent")
    reviewer = _nested(
        divergent,
        "execution",
        "slurm",
        "role_classes",
        "plan_reviewer",
        "credential_broker",
    )
    reviewer["per_attempt_ttl_seconds"] = 60
    with pytest.raises(TaskSchemaError, match="byte-identical after normalization"):
        load_and_normalize_task(_write_task(tmp_path / "divergent", divergent))


def test_codex_oauth_policy_is_supported_but_mutually_exclusive_with_api_key(
    tmp_path: Path,
) -> None:
    oauth = _valid_task(tmp_path / "oauth")
    source = _enable_managed_codex_oauth(oauth, tmp_path / "oauth")

    normalized = load_and_normalize_task(_write_task(tmp_path / "oauth", oauth))
    assert all(
        role.credential_broker.allowed_credential_names == ("CODEX_AUTH_JSON",)
        for role in normalized.execution.slurm.role_classes
    )
    assert normalized.execution.slurm.controller.source_auth_json_path == source
    payload = normalized_task_dict(normalized)
    assert payload["execution"]["slurm"]["controller"]["source_auth_json_path"] == str(source)
    snapshot = tmp_path / "oauth" / "task.normalized.json"
    write_normalized_task(snapshot, normalized)
    assert load_normalized_task(snapshot) == normalized

    conflicting = _valid_task(tmp_path / "conflicting")
    conflicting_roles = _nested(conflicting, "execution", "slurm", "role_classes")
    for role in conflicting_roles.values():
        role["credential_broker"]["allowed_credential_names"] = [
            "CODEX_AUTH_JSON",
            "OPENAI_API_KEY",
        ]
    with pytest.raises(TaskSchemaError, match="must select exactly one"):
        load_and_normalize_task(_write_task(tmp_path / "conflicting", conflicting))


def test_managed_codex_oauth_requires_private_source_and_exact_read_only_mount(
    tmp_path: Path,
) -> None:
    missing = _valid_task(tmp_path / "missing")
    roles = _nested(missing, "execution", "slurm", "role_classes")
    for role in roles.values():
        role["credential_broker"]["allowed_credential_names"] = ["CODEX_AUTH_JSON"]
    with pytest.raises(TaskSchemaError, match="source_auth_json_path.*required"):
        load_and_normalize_task(_write_task(tmp_path / "missing", missing))

    writable = _valid_task(tmp_path / "writable")
    _enable_managed_codex_oauth(writable, tmp_path / "writable")
    mounts = _nested(writable, "execution", "slurm", "controller")["mounts"]
    assert isinstance(mounts, list)
    mounts[-1]["read_only"] = False
    with pytest.raises(TaskSchemaError, match="exactly one identity-mapped read-only"):
        load_and_normalize_task(_write_task(tmp_path / "writable", writable))

    broad_only = _valid_task(tmp_path / "broad-only")
    _enable_managed_codex_oauth(broad_only, tmp_path / "broad-only")
    broad_mounts = _nested(broad_only, "execution", "slurm", "controller")["mounts"]
    assert isinstance(broad_mounts, list)
    broad_mounts.pop()
    with pytest.raises(TaskSchemaError, match="exactly one identity-mapped read-only"):
        load_and_normalize_task(_write_task(tmp_path / "broad-only", broad_only))


def test_managed_codex_oauth_rejects_unsafe_source_and_socket_location(tmp_path: Path) -> None:
    unsafe = _valid_task(tmp_path / "unsafe")
    source = _enable_managed_codex_oauth(unsafe, tmp_path / "unsafe")
    source.chmod(0o644)
    controller = _nested(unsafe, "execution", "slurm", "controller")
    controller["credential_broker_socket"] = str(tmp_path / "outside" / "broker.sock")

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path / "unsafe", unsafe))

    message = str(exc_info.value)
    assert "private mode 0400/0600 regular file" in message
    assert "must be beneath repository.workspace_root or use the private" in message


def test_preauthenticated_roles_require_empty_names_and_no_socket(tmp_path: Path) -> None:
    task_value = _valid_task(tmp_path / "valid")
    controller = _nested(task_value, "execution", "slurm", "controller")
    del controller["credential_broker_socket"]
    roles = _nested(task_value, "execution", "slurm", "role_classes")
    for role in roles.values():
        assert isinstance(role, dict)
        broker = role["credential_broker"]
        assert isinstance(broker, dict)
        broker["broker_id"] = "preauthenticated"
        broker["allowed_credential_names"] = []

    task = load_and_normalize_task(_write_task(tmp_path / "valid", task_value))
    assert task.execution.slurm.controller.credential_broker_socket is None
    assert all(
        role.credential_broker.broker_id == "preauthenticated"
        and role.credential_broker.allowed_credential_names == ()
        for role in task.execution.slurm.role_classes
    )

    snapshot = tmp_path / "valid" / "task.normalized.json"
    write_normalized_task(snapshot, task)
    assert load_normalized_task(snapshot) == task

    with_socket = copy.deepcopy(task_value)
    _nested(with_socket, "execution", "slurm", "controller")["credential_broker_socket"] = (
        "/run/staircase/credential-broker.sock"
    )
    with pytest.raises(TaskSchemaError, match="must be omitted"):
        load_and_normalize_task(_write_task(tmp_path / "valid", with_socket, "with-socket.yaml"))

    wrong_id = copy.deepcopy(task_value)
    wrong_roles = _nested(wrong_id, "execution", "slurm", "role_classes")
    wrong_roles["qa"]["credential_broker"]["broker_id"] = "site-codex-broker"
    with pytest.raises(TaskSchemaError, match="must be 'preauthenticated'"):
        load_and_normalize_task(_write_task(tmp_path / "valid", wrong_id, "wrong-id.yaml"))


@pytest.mark.parametrize(
    "socket_path",
    ("relative/broker.sock", "/run/staircase/../broker.sock", "/run//broker.sock"),
)
def test_credential_broker_socket_is_a_normalized_absolute_container_path(
    tmp_path: Path,
    socket_path: str,
) -> None:
    task = _valid_task(tmp_path)
    _nested(task, "execution", "slurm", "controller")["credential_broker_socket"] = socket_path
    with pytest.raises(TaskSchemaError, match="normalized absolute container path"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_v2_role_classes_require_exact_role_vocabulary(tmp_path: Path) -> None:
    missing_map = _valid_task(tmp_path / "missing-map")
    del _nested(missing_map, "execution", "slurm")["role_classes"]
    with pytest.raises(
        TaskSchemaError,
        match="missing required field 'execution.slurm.role_classes'",
    ):
        load_and_normalize_task(_write_task(tmp_path / "missing-map", missing_map))
    with pytest.raises(
        TaskSchemaError,
        match="missing required field 'execution.slurm.role_classes'",
    ):
        load_and_normalize_task(
            _write_task(tmp_path / "missing-map", missing_map, "local.yaml"),
            execution_override="local",
        )

    missing_role = _valid_task(tmp_path / "missing-role")
    del _nested(missing_role, "execution", "slurm", "role_classes")["qa"]
    with pytest.raises(
        TaskSchemaError,
        match="missing required field 'execution.slurm.role_classes.qa'",
    ):
        load_and_normalize_task(_write_task(tmp_path / "missing-role", missing_role))

    unknown_role = _valid_task(tmp_path / "unknown-role")
    roles = _nested(unknown_role, "execution", "slurm", "role_classes")
    roles["arbitrary_agent"] = copy.deepcopy(roles["qa"])
    with pytest.raises(
        TaskSchemaError,
        match="unknown field 'execution.slurm.role_classes.arbitrary_agent'",
    ):
        load_and_normalize_task(_write_task(tmp_path / "unknown-role", unknown_role))


def test_role_classes_reject_duplicate_yaml_role_key(tmp_path: Path) -> None:
    text = yaml.safe_dump(_valid_task(tmp_path), sort_keys=False)
    needle = "      plan_drafter:\n"
    assert needle in text
    duplicate = text.replace(needle, "      plan_drafter: {}\n" + needle, 1)
    task_path = tmp_path / "duplicate-role.yaml"
    task_path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(TaskSchemaError, match="duplicate key 'plan_drafter'"):
        load_and_normalize_task(task_path)


def test_role_classes_reject_unknown_or_multi_rank_resources(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    roles = _nested(task, "execution", "slurm", "role_classes")
    plan_drafter = roles["plan_drafter"]
    qa = roles["qa"]
    assert isinstance(plan_drafter, dict)
    assert isinstance(qa, dict)
    plan_drafter["resource_class"] = "deterministic_gate"
    qa["resource_class"] = "not_configured"

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "plan_drafter.resource_class' must bind a single-node, single-task" in message
    assert "qa.resource_class' must name an existing Smith resource class" in message


def test_role_classes_reject_open_network_secret_environment_and_broker_handles(
    tmp_path: Path,
) -> None:
    task = _valid_task(tmp_path)
    roles = _nested(task, "execution", "slurm", "role_classes")
    drafter = roles["plan_drafter"]
    reviewer = roles["reviewer"]
    assert isinstance(drafter, dict)
    assert isinstance(reviewer, dict)
    drafter["network_policy"] = "unrestricted"
    environment = drafter["environment"]
    assert isinstance(environment, dict)
    environment["allowlist"] = ["SLURM_JOB_ID", "OPENAI_API_KEY", "SAFE_NAME"]
    environment["values"] = {
        "SAFE_NAME": "duplicate-source",
        "ANTHROPIC_AUTH_TOKEN": "secret",
    }
    broker = drafter["credential_broker"]
    assert isinstance(broker, dict)
    broker["allowed_credential_names"] = ["ANTHROPIC_API_KEY"]
    broker["per_attempt_ttl_seconds"] = 86_401
    broker["handle_id"] = "reusable-handle"
    broker["credential_value"] = "secret"
    reviewer_broker = reviewer["credential_broker"]
    assert isinstance(reviewer_broker, dict)
    reviewer_broker["broker_id"] = "legacy-environment"

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "network_policy' must be 'backend_api_only'" in message
    assert "must not admit scheduler-owned environment 'SLURM_JOB_ID'" in message
    assert "must not admit credential-like environment 'OPENAI_API_KEY'" in message
    assert "must not admit credential-like environment 'ANTHROPIC_AUTH_TOKEN'" in message
    assert "names must appear in either allowlist or values" in message
    assert (
        "unknown field 'execution.slurm.role_classes.plan_drafter.credential_broker.handle_id'"
        in message
    )
    assert (
        "unknown field "
        "'execution.slurm.role_classes.plan_drafter.credential_broker.credential_value'"
    ) in message
    assert "is inconsistent with backend 'codex'" in message
    assert "per_attempt_ttl_seconds' must not exceed 86400" in message
    assert "broker_id' must identify a per-attempt credential broker" in message


def test_agent_identity_is_required_and_rejects_security_or_backend_option_fields(
    tmp_path: Path,
) -> None:
    missing = _valid_task(tmp_path / "missing")
    del _nested(missing, "execution")["agent"]
    with pytest.raises(TaskSchemaError, match="missing required field 'execution.agent'"):
        load_and_normalize_task(_write_task(tmp_path / "missing", missing))

    unknown = _valid_task(tmp_path / "unknown")
    agent = _nested(unknown, "execution", "agent")
    agent.update(
        {
            "api_key": "not-accepted",
            "credential_path": "/tmp/not-accepted",
            "command": "codex --dangerous-option",
            "backend_options": {"arbitrary": True},
        }
    )
    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path / "unknown", unknown))

    message = str(exc_info.value)
    for field in ("api_key", "credential_path", "command", "backend_options"):
        assert f"unknown field 'execution.agent.{field}'" in message


def test_agent_identity_batches_invalid_backend_and_multiline_model(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    agent = _nested(task, "execution", "agent")
    agent["backend_kind"] = "arbitrary-backend"
    agent["model"] = "model\n--shell-option"

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "'execution.agent.backend_kind' must be one of ['claude-code', 'codex']" in message
    assert "'execution.agent.model' must be a single-line string" in message


@pytest.mark.parametrize("model", ["", "   ", "model\rsecond-line"])
def test_agent_model_must_be_nonempty_and_single_line(tmp_path: Path, model: str) -> None:
    task = _valid_task(tmp_path)
    _nested(task, "execution", "agent")["model"] = model

    with pytest.raises(TaskSchemaError, match="execution.agent.model"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_agent_identity_never_reads_ambient_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STAIRCASE_AGENT_BACKEND", "claude-code")
    monkeypatch.setenv("STAIRCASE_AGENT_MODEL", "ambient-model")

    task = load_and_normalize_task(_write_task(tmp_path, _valid_task(tmp_path)))

    assert task.execution.agent == AgentExecutionConfig("codex", "gpt-5-codex")


def test_batches_unknown_keys_bool_numbers_and_nonfinite_tolerance(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    task["unknown_top"] = True
    target = _nested(task, "target")
    target["surprise"] = "field"
    target["sm"] = True
    accuracy = _nested(task, "gates", "accuracy")
    accuracy["tolerance"] = float("inf")
    smith = _nested(task, "execution", "slurm", "smith")
    smith["max_parallel_items"] = True

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "unknown field 'task.unknown_top'" in message
    assert "unknown field 'target.surprise'" in message
    assert "'target.sm' must be an integer >= 1" in message
    assert "'gates.accuracy.tolerance' must be a finite number >= 0" in message
    assert "'execution.slurm.smith.max_parallel_items' must be an integer >= 1" in message


def test_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    text = yaml.safe_dump(task, sort_keys=False)
    duplicate = text.replace("schema_version: 2\n", "schema_version: 2\nschema_version: 2\n", 1)
    task_path = tmp_path / "duplicate.yaml"
    task_path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(TaskSchemaError, match="duplicate key 'schema_version'"):
        load_and_normalize_task(task_path)


def test_rejects_unsafe_slugs_traversal_wrong_path_kind_and_symlink_escape(
    tmp_path: Path,
) -> None:
    task = _valid_task(tmp_path)
    repository = _nested(task, "repository")
    repository["workspace_root"] = str(tmp_path / "checkpoint" / ".." / "workspace")
    target = _nested(task, "target")
    target["family"] = "../../escape"
    target["checkpoint_id"] = "bad/name"
    target["expected_route"] = "../outside"
    reference = _nested(task, "reference")
    reference["checkpoint"] = str(tmp_path / "trtllm.sqsh")

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "must not contain '..' path traversal" in message
    assert "target.family" in message
    assert "target.checkpoint_id" in message
    assert "normalized relative path without traversal" in message
    assert "reference.checkpoint' must point to a directory" in message

    clean_task = _valid_task(tmp_path / "symlink_case")
    clean_repository = Path(_nested(clean_task, "repository")["root"])
    outside = tmp_path / "outside"
    outside.mkdir()
    family_link = (
        clean_repository / "tensorrt_llm" / "_torch" / "modeling_v2" / "models" / "example_moe"
    )
    family_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(TaskSchemaError, match="escapes the canonical repository root"):
        load_and_normalize_task(_write_task(tmp_path / "symlink_case", clean_task, "symlink.yaml"))


def test_controller_mounts_require_canonical_identity_paths(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    mount = _nested(task, "execution", "slurm", "controller")["mounts"][0]
    mount["container_path"] = "/container/shared"

    with pytest.raises(TaskSchemaError, match="identity-mount controller contract"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_overlapping_identity_mounts_round_trip_without_path_translation(
    tmp_path: Path,
) -> None:
    task_value = _valid_task(tmp_path)
    mounts = _nested(task_value, "execution", "slurm", "controller")["mounts"]
    repository = Path(_nested(task_value, "repository")["root"]).resolve()
    mounts.append(
        {
            "host_path": str(repository),
            "container_path": repository.as_posix(),
            "read_only": False,
        }
    )

    task = load_and_normalize_task(_write_task(tmp_path, task_value))
    assert all(
        mount.host_path.as_posix() == mount.container_path
        for mount in task.execution.slurm.controller.mounts
    )
    snapshot = tmp_path / "task.normalized.json"
    write_normalized_task(snapshot, task)
    assert load_normalized_task(snapshot) == task


def test_rejects_topology_resource_cap_and_override_mismatches(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    target = _nested(task, "target")
    target["world_size"] = 8
    mapping = _nested(task, "target", "mapping")
    mapping["moe_expert_parallel_size"] = 3
    mapping["attention_data_parallel_size"] = 3
    smith = _nested(task, "execution", "slurm", "smith")
    smith["max_nodes_total"] = 1
    smith["max_gpus_total"] = 2
    bounds = _nested(task, "execution", "slurm", "smith", "per_item_override_bounds")
    bounds["max_nodes"] = 3

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "must equal tensor_parallel_size * pipeline_parallel_size" in message
    assert "MoE EP * MoE TP" in message
    assert "must divide tensor_parallel_size" in message
    assert "max_parallel_items' must not exceed max_nodes_total" in message
    assert "max_nodes' exceeds max_nodes_total" in message
    assert "aggregate GPUs exceed max_gpus_total" in message
    assert "must request exactly target.world_size=8 tasks and GPUs" in message


def test_single_rank_gate_allows_explicit_site_gpu_allocation_padding(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    target = _nested(task, "target")
    target["world_size"] = 1
    mapping = _nested(task, "target", "mapping")
    mapping.update(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "attention_data_parallel_size": 1,
        }
    )
    resources = _nested(task, "execution", "slurm", "smith", "resource_classes")
    resources["deterministic_gate"] = _resource_class(
        tasks=1,
        gpus=4,
        gpu_allocation_padding=True,
        partition="batch",
        qos="normal",
    )
    resources["coder_analysis"]["partition"] = "cpu"
    resources["coder_analysis"]["qos"] = "cpu-short"
    bounds = _nested(task, "execution", "slurm", "smith", "per_item_override_bounds")
    bounds["max_gpus_per_node"] = 4

    normalized = load_and_normalize_task(_write_task(tmp_path, task))

    gate = normalized.execution.slurm.smith.resource_class("deterministic_gate")
    assert gate.total_tasks == 1
    assert gate.total_gpus == 4
    assert gate.gpu_allocation_padding is True
    assert (gate.partition, gate.qos) == ("batch", "normal")
    coder = normalized.execution.slurm.smith.resource_class("coder_analysis")
    assert (coder.partition, coder.qos) == ("cpu", "cpu-short")
    resources_json = normalized_task_dict(normalized)["execution"]["slurm"]["smith"][
        "resource_classes"
    ]
    gate_json = next(
        resource for resource in resources_json if resource["name"] == "deterministic_gate"
    )
    assert gate_json["gpu_allocation_padding"] is True
    assert (gate_json["partition"], gate_json["qos"]) == ("batch", "normal")


def test_single_rank_gate_rejects_silent_gpu_overallocation(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    target = _nested(task, "target")
    target["world_size"] = 1
    mapping = _nested(task, "target", "mapping")
    mapping.update(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "attention_data_parallel_size": 1,
        }
    )
    resources = _nested(task, "execution", "slurm", "smith", "resource_classes")
    resources["deterministic_gate"] = _resource_class(tasks=1, gpus=4)
    bounds = _nested(task, "execution", "slurm", "smith", "per_item_override_bounds")
    bounds["max_gpus_per_node"] = 4

    with pytest.raises(TaskSchemaError, match="must request exactly target.world_size=1"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_gpu_allocation_padding_is_narrow_and_nonvacuous(tmp_path: Path) -> None:
    non_gate = _valid_task(tmp_path / "non-gate")
    resources = _nested(
        non_gate,
        "execution",
        "slurm",
        "smith",
        "resource_classes",
    )
    resources["exploratory_probe"]["gpu_allocation_padding"] = True
    with pytest.raises(TaskSchemaError, match="only supported for deterministic_gate"):
        load_and_normalize_task(_write_task(tmp_path / "non-gate", non_gate))

    vacuous = _valid_task(tmp_path / "vacuous")
    resources = _nested(
        vacuous,
        "execution",
        "slurm",
        "smith",
        "resource_classes",
    )
    resources["deterministic_gate"]["gpu_allocation_padding"] = True
    with pytest.raises(TaskSchemaError, match="only supported for a one-rank target"):
        load_and_normalize_task(_write_task(tmp_path / "vacuous", vacuous))

    role_bound = _valid_task(tmp_path / "role-bound")
    target = _nested(role_bound, "target")
    target["world_size"] = 1
    mapping = _nested(role_bound, "target", "mapping")
    mapping.update(
        {
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "moe_expert_parallel_size": 1,
            "moe_tensor_parallel_size": 1,
            "attention_data_parallel_size": 1,
        }
    )
    resources = _nested(
        role_bound,
        "execution",
        "slurm",
        "smith",
        "resource_classes",
    )
    resources["deterministic_gate"] = _resource_class(
        tasks=1,
        gpus=4,
        gpu_allocation_padding=True,
    )
    bounds = _nested(
        role_bound,
        "execution",
        "slurm",
        "smith",
        "per_item_override_bounds",
    )
    bounds["max_gpus_per_node"] = 4
    roles = _nested(role_bound, "execution", "slurm", "role_classes")
    roles["qa"]["resource_class"] = "deterministic_gate"
    with pytest.raises(TaskSchemaError, match="cannot bind a GPU allocation-padding resource"):
        load_and_normalize_task(_write_task(tmp_path / "role-bound", role_bound))


def test_resource_scheduler_overrides_reject_unsafe_names(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    resources = _nested(task, "execution", "slurm", "smith", "resource_classes")
    resources["coder_analysis"]["partition"] = "cpu;sbatch"
    resources["deterministic_gate"]["qos"] = "normal,*"

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "coder_analysis.partition' contains unsafe Slurm-name characters" in message
    assert "deterministic_gate.qos' contains unsafe Slurm-name characters" in message


def test_rejects_missing_controller_and_resource_class_fields(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    controller = _nested(task, "execution", "slurm", "controller")
    del controller["account"]
    resources = _nested(task, "execution", "slurm", "smith", "resource_classes")
    del resources["reviewer_rerun"]
    gate = _nested(
        task,
        "execution",
        "slurm",
        "smith",
        "resource_classes",
        "deterministic_gate",
    )
    del gate["memory"]

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "missing required field 'execution.slurm.controller.account'" in message
    assert (
        "missing required field 'execution.slurm.smith.resource_classes.reviewer_rerun'" in message
    )
    assert (
        "missing required field 'execution.slurm.smith.resource_classes.deterministic_gate.memory'"
    ) in message


@pytest.mark.parametrize("token", ["_torch/staircase", "TRTLLM_STAIRCASE"])
def test_rejects_obsolete_data_plane_tokens(tmp_path: Path, token: str) -> None:
    task = _valid_task(tmp_path)
    accuracy = _nested(task, "gates", "accuracy")
    accuracy["protocol"] = f"invoke {token}"

    with pytest.raises(TaskSchemaError, match="obsolete data-plane token"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_rejects_checkpoint_architecture_and_shared_mount_mismatches(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    reference = _nested(task, "reference")
    reference["architecture"] = "OtherForCausalLM"
    mounts = _nested(task, "execution", "slurm", "controller")["mounts"]
    assert isinstance(mounts, list)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    mounts[0]["host_path"] = str(unrelated)

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "does not match checkpoint architectures[0]" in message
    assert "is not visible through any" in message


def test_rejects_unsafe_delivery_and_non_slurm_task_mode(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    execution = _nested(task, "execution")
    execution["mode"] = "local"
    delivery = _nested(task, "delivery")
    delivery["mode"] = "signed_commits"
    delivery["branch"] = "../main"
    delivery["merge"] = True
    delivery["push"] = True

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "local execution is a CLI test override" in message
    assert "not a safe relative Git branch name" in message
    assert "merge requires separate authorization" in message
    assert "push requires separate authorization" in message


def test_rejects_unimplemented_dispatch_modes_and_multi_task_role_classes(
    tmp_path: Path,
) -> None:
    task = _valid_task(tmp_path)
    controller = _nested(task, "execution", "slurm", "controller")
    controller["dispatch_mode"] = "preallocated_pool"
    reviewer = _nested(
        task,
        "execution",
        "slurm",
        "smith",
        "resource_classes",
        "reviewer_rerun",
    )
    reviewer["nodes"] = 2
    reviewer["tasks_per_node"] = 2

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "must be 'nested_submission'" in message
    assert "reviewer_rerun' must be a single-node, single-task role class" in message


def test_certification_identity_and_gate_classes_are_explicit(tmp_path: Path) -> None:
    missing_identity = _valid_task(tmp_path / "missing")
    _nested(missing_identity, "certification")["expected_product_identity"] = None
    with pytest.raises(TaskSchemaError, match="required for REAL mode"):
        load_and_normalize_task(_write_task(tmp_path / "missing", missing_identity))

    synthetic = _valid_task(tmp_path / "synthetic")
    certification = _nested(synthetic, "certification")
    certification["mode"] = "SYNTHETIC"
    certification["expected_product_identity"] = None
    gates = _nested(synthetic, "gates")
    gates["collective_tests"] = [
        "tests/unittest/_torch/modeling_v2/test_target.py::test_collective"
    ]
    _nested(synthetic, "execution", "slurm")["gate_classes"] = {
        "collective": {"resource_class": "deterministic_gate"}
    }

    normalized = load_and_normalize_task(_write_task(tmp_path / "synthetic", synthetic))
    assert normalized.certification.mode is CertificationMode.SYNTHETIC
    assert tuple(entry.gate_id for entry in normalized.execution.slurm.gate_classes) == (
        "collective",
    )

    local = _valid_task(tmp_path / "local")
    local_certification = _nested(local, "certification")
    local_certification["mode"] = "LOCAL"
    local_certification["expected_product_identity"] = None
    local_path = _write_task(tmp_path / "local", local)
    with pytest.raises(TaskSchemaError, match="explicit CLI local execution override"):
        load_and_normalize_task(local_path)
    assert (
        load_and_normalize_task(local_path, execution_override="local").certification.mode
        is CertificationMode.LOCAL
    )


def test_negative_and_boolean_resource_values_are_rejected(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    probe = _nested(
        task,
        "execution",
        "slurm",
        "smith",
        "resource_classes",
        "exploratory_probe",
    )
    probe["nodes"] = True
    probe["gpus_per_node"] = -1
    retry = _nested(task, "execution", "slurm", "smith", "retry_policy")
    retry["preempted"] = True

    with pytest.raises(TaskSchemaError) as exc_info:
        load_and_normalize_task(_write_task(tmp_path, task))

    message = str(exc_info.value)
    assert "exploratory_probe.nodes' must be an integer >= 1" in message
    assert "exploratory_probe.gpus_per_node' must be an integer >= 0" in message
    assert "retry_policy.preempted' must be an integer >= 0" in message


def test_requested_features_require_acceptance_signal(tmp_path: Path) -> None:
    task = _valid_task(tmp_path)
    gates = _nested(task, "gates")
    gates["feature_signal_tests"] = []

    with pytest.raises(TaskSchemaError, match="must include an executable acceptance test"):
        load_and_normalize_task(_write_task(tmp_path, task))


@pytest.mark.parametrize(
    "field,value",
    (
        ("boot_tests", "pytest -q tests/test_boot.py"),
        ("component_tests", "component output matches"),
        ("collective_tests", "-k collective"),
        ("feature_signal_tests", "../tests/test_feature.py"),
    ),
)
def test_gate_fields_accept_selectors_not_commands_or_prose(
    tmp_path: Path, field: str, value: str
) -> None:
    task = _valid_task(tmp_path)
    gates = _nested(task, "gates")
    gates[field] = [value]

    with pytest.raises(TaskSchemaError, match="pytest selector"):
        load_and_normalize_task(_write_task(tmp_path, task))


def test_digest_changes_with_normalized_semantics(tmp_path: Path) -> None:
    original = _valid_task(tmp_path)
    changed = copy.deepcopy(original)
    _nested(changed, "gates", "accuracy")["tolerance"] = 0.5

    first = load_and_normalize_task(_write_task(tmp_path, original, "original.yaml"))
    second = load_and_normalize_task(_write_task(tmp_path, changed, "changed.yaml"))

    assert first.digest != second.digest


@pytest.mark.parametrize(
    "field,value",
    (("backend_kind", "claude-code"), ("model", "claude-opus-4-1")),
)
def test_digest_changes_with_agent_execution_identity(
    tmp_path: Path, field: str, value: str
) -> None:
    original = _valid_task(tmp_path)
    changed = copy.deepcopy(original)
    _nested(changed, "execution", "agent")[field] = value
    if field == "backend_kind":
        role_classes = _nested(changed, "execution", "slurm", "role_classes")
        for raw_policy in role_classes.values():
            assert isinstance(raw_policy, dict)
            broker = raw_policy["credential_broker"]
            assert isinstance(broker, dict)
            broker["broker_id"] = "site-claude-broker"
            broker["allowed_credential_names"] = ["ANTHROPIC_API_KEY"]

    first = load_and_normalize_task(_write_task(tmp_path, original, "original.yaml"))
    second = load_and_normalize_task(_write_task(tmp_path, changed, "changed.yaml"))

    assert first.digest != second.digest


def test_normalized_task_round_trip_reconstructs_frozen_models(tmp_path: Path) -> None:
    original = load_and_normalize_task(
        _write_task(tmp_path, _valid_task(tmp_path)), execution_override="local"
    )
    snapshot = tmp_path / "task.normalized.json"

    write_normalized_task(snapshot, original)
    restored = load_normalized_task(snapshot)

    assert restored == original
    assert restored.execution.mode == "local"
    assert normalized_task_json(restored) == normalized_task_json(original)


@pytest.mark.parametrize("mutation", ["digest", "unknown", "semantic", "agent"])
def test_normalized_task_rejects_digest_unknown_and_semantic_drift(
    tmp_path: Path, mutation: str
) -> None:
    original = load_and_normalize_task(_write_task(tmp_path, _valid_task(tmp_path)))
    snapshot = tmp_path / "task.normalized.json"
    write_normalized_task(snapshot, original)
    value = json.loads(snapshot.read_text(encoding="utf-8"))
    if mutation == "digest":
        value["digest"] = "0" * 64
    elif mutation == "unknown":
        value["task"]["target"]["unknown"] = True
    elif mutation == "agent":
        value["task"]["execution"]["agent"]["model"] = "mutated-after-persist"
    else:
        value["task"]["target"]["world_size"] = -1
        # Even recomputing the claimed digest cannot turn an invalid task into
        # a valid resume snapshot; schema validation runs before comparison.
        value["digest"] = "1" * 64
    snapshot.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(TaskSchemaError):
        load_normalized_task(snapshot)


@pytest.mark.parametrize(
    "relative_path",
    ("onboarding/task.example.yaml", "tuning/task.example.yaml"),
)
def test_packaged_examples_declare_only_non_secret_agent_identity_fields(
    relative_path: str,
) -> None:
    package_root = Path(__file__).parents[3] / "agent_flow" / "workflows" / "staircase"
    value = yaml.safe_load((package_root / relative_path).read_text(encoding="utf-8"))

    agent = value["execution"]["agent"]
    assert value["schema_version"] == 2
    assert value["certification"]["mode"] == "REAL"
    assert set(value["execution"]["slurm"]["gate_classes"]) == {
        "accuracy",
        "boot",
        "collective",
        "feature-signal",
    }
    role_classes = value["execution"]["slurm"]["role_classes"]
    controller = value["execution"]["slurm"]["controller"]
    assert controller["credential_broker_socket"] == (
        "/tmp/staircase-credential-broker-example/broker.sock"
    )
    assert controller["source_auth_json_path"] == "/run/secrets/codex-auth.json"
    assert {
        "host_path": "/run/secrets/codex-auth.json",
        "container_path": "/run/secrets/codex-auth.json",
        "read_only": True,
    } in controller["mounts"]
    assert set(role_classes) == set(_role_classes())
    for role_policy in role_classes.values():
        assert set(role_policy) == {
            "resource_class",
            "network_policy",
            "environment",
            "credential_broker",
        }
        assert role_policy["network_policy"] == "backend_api_only"
        assert role_policy["environment"] == {
            "allowlist": [],
            "values": {"PYTHONNOUSERSITE": "1"},
        }
        assert role_policy["credential_broker"] == {
            "broker_id": "site-codex-broker",
            "allowed_credential_names": ["CODEX_AUTH_JSON"],
            "per_attempt_ttl_seconds": 14_400,
        }
    assert set(agent) == {"backend_kind", "model"}
    assert agent == {"backend_kind": "codex", "model": "gpt-5-codex"}
