# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Staircase worker launch isolation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent_flow.workflows.staircase.common.credentials import (
    CREDENTIAL_MOUNT_PATH,
    CredentialBinding,
    CredentialDescriptor,
    prepare_credential_provision,
)
from agent_flow.workflows.staircase.common.isolation import (
    ContainerPathTranslator,
    IsolationPolicyError,
    attach_agent_credential_provision,
    build_agent_worker_environment,
    build_gate_worker_environment,
    build_worker_isolation,
    preserve_controller_mounts,
)
from agent_flow.workflows.staircase.common.launch_policy import AgentLaunchPolicy
from agent_flow.workflows.staircase.common.native_artifacts import (
    NATIVE_ARTIFACT_MANIFEST_PATH,
    NativeArtifactEntry,
    NativeArtifactManifest,
    ValidatedNativeArtifactBundle,
)
from agent_flow.workflows.staircase.common.runners import RoleProcessSpec
from agent_flow.workflows.staircase.common.slurm import Mount


def _layout(tmp_path: Path) -> dict[str, Path]:
    host = tmp_path / "host"
    repository = host / "repository"
    workspace = host / "runs" / "run-1"
    worktree = workspace / "candidates" / "entry-a" / "coder-1"
    attempt = workspace / "items" / "entry-a" / "attempts" / "0001"
    mailbox = attempt / "output"
    input_bundle = attempt / "role-input.json"
    checkpoint = host / "models" / "checkpoint"
    reference = host / "references" / "reference.py"
    for directory in (repository, worktree, mailbox, checkpoint, reference.parent):
        directory.mkdir(parents=True, exist_ok=True)
    reference.write_text("REFERENCE = True\n", encoding="utf-8")
    input_bundle.write_text("{}\n", encoding="utf-8")
    return {
        "host": host,
        "repository": repository,
        "workspace": workspace,
        "worktree": worktree,
        "mailbox": mailbox,
        "input": input_bundle,
        "checkpoint": checkpoint,
        "reference": reference,
    }


def test_worker_mounts_are_exact_and_controller_mount_is_preserved_separately(
    tmp_path: Path,
) -> None:
    paths = _layout(tmp_path)
    controller = (Mount(paths["host"], Path("/container/shared"), False),)
    isolation = build_worker_isolation(
        controller_mounts=controller,
        workspace=paths["workspace"],
        repository=paths["repository"],
        worktree=paths["worktree"],
        worktree_writable=True,
        mailbox=paths["mailbox"],
        input_bundle=paths["input"],
        checkpoint=paths["checkpoint"],
        reference_sources=(paths["reference"],),
    )

    assert preserve_controller_mounts(controller) == controller
    assert all(mount.source != paths["host"] for mount in isolation.mounts)
    assert [(mount.source, mount.read_only) for mount in isolation.mounts] == [
        (paths["worktree"], False),
        (paths["mailbox"], False),
        (paths["input"], True),
        (paths["checkpoint"], True),
        (paths["reference"], True),
    ]
    assert isolation.container_path(paths["worktree"]) == Path(
        "/container/shared/runs/run-1/candidates/entry-a/coder-1"
    )
    assert isolation.container_path(paths["mailbox"]) == Path(
        "/container/shared/runs/run-1/items/entry-a/attempts/0001/output"
    )


def test_nested_mount_translation_uses_most_specific_mapping(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    translator = ContainerPathTranslator.from_mounts(
        (
            Mount(paths["host"], Path("/container/shared")),
            Mount(paths["repository"], Path("/src/trtllm")),
        )
    )

    assert translator.to_container(paths["repository"]) == Path("/src/trtllm")
    assert translator.to_container(paths["checkpoint"]) == Path(
        "/container/shared/models/checkpoint"
    )
    assert translator.to_host(Path("/src/trtllm")) == paths["repository"]


def test_gate_native_artifacts_shadow_candidate_with_nested_read_only_mounts(
    tmp_path: Path,
) -> None:
    paths = _layout(tmp_path)
    package = paths["repository"] / "tensorrt_llm"
    libraries = package / "libs"
    libraries.mkdir(parents=True)
    extension = package / "bindings.test.so"
    extension.write_bytes(b"extension")
    manifest_path = paths["host"] / "native-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    bundle = ValidatedNativeArtifactBundle(
        package,
        manifest_path,
        NativeArtifactManifest(
            "build-id",
            "a" * 40,
            ".test.so",
            (NativeArtifactEntry("bindings.test.so", 9, "b" * 64),),
            "c" * 64,
        ),
    )

    isolation = build_worker_isolation(
        controller_mounts=(Mount(paths["host"], Path("/container/shared"), False),),
        workspace=paths["workspace"],
        repository=paths["repository"],
        worktree=paths["worktree"],
        mailbox=paths["mailbox"],
        input_bundle=paths["input"],
        checkpoint=paths["checkpoint"],
        native_artifact_bundle=bundle,
    )

    nested = {mount.target: mount for mount in isolation.mounts}
    candidate_package = Path("/container/shared/runs/run-1/candidates/entry-a/coder-1/tensorrt_llm")
    assert nested[candidate_package / "libs"].source == libraries
    assert nested[candidate_package / "bindings.test.so"].source == extension
    assert nested[NATIVE_ARTIFACT_MANIFEST_PATH].source == manifest_path
    assert all(
        mount.read_only
        for target, mount in nested.items()
        if target == NATIVE_ARTIFACT_MANIFEST_PATH or target.is_relative_to(candidate_package)
    )


@pytest.mark.parametrize(
    "mounts,match",
    [
        (
            lambda root: (
                Mount(root, Path("/one")),
                Mount(root, Path("/two")),
            ),
            "duplicate controller source",
        ),
        (
            lambda root: (
                Mount(root, Path("/same")),
                Mount(root / "repository", Path("/same")),
            ),
            "duplicate controller target",
        ),
    ],
)
def test_ambiguous_controller_mount_table_is_rejected(
    tmp_path: Path,
    mounts,  # type: ignore[no-untyped-def]
    match: str,
) -> None:
    paths = _layout(tmp_path)
    with pytest.raises(IsolationPolicyError, match=match):
        ContainerPathTranslator.from_mounts(mounts(paths["host"]))


def test_uncovered_path_and_broad_writable_paths_fail_closed(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    controller = (Mount(paths["host"], Path("/container/shared"), False),)
    outside = tmp_path / "outside"
    outside.mkdir()
    translator = ContainerPathTranslator.from_mounts(controller)
    with pytest.raises(IsolationPolicyError, match="not covered"):
        translator.to_container(outside)

    with pytest.raises(IsolationPolicyError, match="one exact candidates"):
        build_worker_isolation(
            controller_mounts=controller,
            workspace=paths["workspace"],
            repository=paths["repository"],
            worktree=paths["workspace"] / "candidates",
            worktree_writable=True,
            mailbox=paths["mailbox"],
            input_bundle=paths["input"],
            checkpoint=paths["checkpoint"],
        )
    with pytest.raises(IsolationPolicyError, match="one exact workspace/items"):
        build_worker_isolation(
            controller_mounts=controller,
            workspace=paths["workspace"],
            repository=paths["repository"],
            mailbox=paths["workspace"] / "items" / "entry-a" / "attempts",
            input_bundle=paths["input"],
            checkpoint=paths["checkpoint"],
        )


def test_candidate_overlay_cannot_expose_git_metadata(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    (paths["worktree"] / ".git").write_text("gitdir: /shared/admin\n", encoding="utf-8")

    with pytest.raises(IsolationPolicyError, match="Git metadata"):
        build_worker_isolation(
            controller_mounts=(Mount(paths["host"], Path("/container/shared"), False),),
            workspace=paths["workspace"],
            repository=paths["repository"],
            worktree=paths["worktree"],
            worktree_writable=True,
            mailbox=paths["mailbox"],
            input_bundle=paths["input"],
            checkpoint=paths["checkpoint"],
        )

    (paths["worktree"] / ".git").unlink()
    (paths["worktree"] / "future-git").symlink_to(".git")
    with pytest.raises(IsolationPolicyError, match="would expose Git metadata"):
        build_worker_isolation(
            controller_mounts=(Mount(paths["host"], Path("/container/shared"), False),),
            workspace=paths["workspace"],
            repository=paths["repository"],
            worktree=paths["worktree"],
            worktree_writable=True,
            mailbox=paths["mailbox"],
            input_bundle=paths["input"],
            checkpoint=paths["checkpoint"],
        )


def test_reference_source_cannot_expose_controller_workspace(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    controller = (Mount(paths["host"], Path("/container/shared"), False),)

    with pytest.raises(IsolationPolicyError, match="cannot overlap controller state"):
        build_worker_isolation(
            controller_mounts=controller,
            workspace=paths["workspace"],
            repository=paths["repository"],
            mailbox=paths["mailbox"],
            input_bundle=paths["input"],
            checkpoint=paths["host"],
        )

    controller_state = paths["workspace"] / "state.json"
    controller_state.write_text("{}\n", encoding="utf-8")
    with pytest.raises(IsolationPolicyError, match="cannot overlap controller state"):
        build_worker_isolation(
            controller_mounts=controller,
            workspace=paths["workspace"],
            repository=paths["repository"],
            mailbox=paths["mailbox"],
            input_bundle=paths["input"],
            checkpoint=paths["checkpoint"],
            reference_sources=(controller_state,),
        )


def test_internal_command_and_role_spec_use_container_visible_paths(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    role_input = paths["input"]
    isolation = build_worker_isolation(
        controller_mounts=(Mount(paths["host"], Path("/container/shared"), False),),
        workspace=paths["workspace"],
        repository=paths["repository"],
        mailbox=paths["mailbox"],
        input_bundle=paths["input"],
        checkpoint=paths["checkpoint"],
    )
    host_spec = RoleProcessSpec(
        schema_version=1,
        run_id="run-1",
        task_digest="a" * 64,
        generation=1,
        item_id="planning",
        attempt_id="draft-1",
        prompt_id="prompt-1",
        role="plan_drafter",
        profile="planner",
        backend_kind="codex",
        model="test-model",
        source_root=str(paths["repository"]),
        cwd=str(paths["repository"]),
        result_path=str(paths["mailbox"] / "role-result.json"),
        system_prompt="system",
        prompt="prompt",
        credential_descriptor=CredentialDescriptor.no_credentials(
            CredentialBinding("run-1", "planning", "draft-1", "a" * 64, 1, "codex")
        ).to_public_dict(),
        launch_policy=AgentLaunchPolicy.create(
            image="/images/agent-worker.sqsh",
            build_identity="agent-worker-test",
        ).to_public_dict(),
    )

    command = isolation.worker_command(role_input)
    worker_spec = isolation.to_worker_role_spec(host_spec)
    metadata_masks = [mount for mount in isolation.mounts if mount.target.name == ".git"]
    assert len(metadata_masks) == 1
    assert metadata_masks[0].read_only
    assert metadata_masks[0].source != paths["repository"] / ".git"
    assert not any(metadata_masks[0].source.iterdir())
    assert command.input_bundle == Path(
        "/container/shared/runs/run-1/items/entry-a/attempts/0001/role-input.json"
    )
    assert worker_spec.source_root == "/container/shared/repository"
    assert worker_spec.cwd == "/container/shared/repository"
    assert worker_spec.result_path.endswith("/items/entry-a/attempts/0001/output/role-result.json")
    assert isolation.to_host_role_spec(worker_spec) == host_spec

    escaped = replace(
        worker_spec,
        result_path="/container/shared/runs/run-1/state.json",
    )
    with pytest.raises(IsolationPolicyError, match="inside its mailbox"):
        isolation.to_host_role_spec(escaped)


def test_worker_command_rejects_forged_input_inside_writable_mailbox(
    tmp_path: Path,
) -> None:
    paths = _layout(tmp_path)
    isolation = build_worker_isolation(
        controller_mounts=(Mount(paths["host"], Path("/container/shared"), False),),
        workspace=paths["workspace"],
        repository=paths["repository"],
        mailbox=paths["mailbox"],
        input_bundle=paths["input"],
        checkpoint=paths["checkpoint"],
    )
    forged = paths["mailbox"] / "role-input.json"
    forged.write_text("{}\n", encoding="utf-8")

    with pytest.raises(IsolationPolicyError, match="immutable input mount"):
        isolation.worker_command(forged)

    input_mount = next(mount for mount in isolation.mounts if mount.source == paths["input"])
    mailbox_mount = next(mount for mount in isolation.mounts if mount.source == paths["mailbox"])
    assert input_mount.read_only
    assert not mailbox_mount.read_only
    assert input_mount.source.parent == mailbox_mount.source.parent
    assert not input_mount.source.is_relative_to(mailbox_mount.source)


def test_agent_environment_never_exports_ambient_backend_credentials() -> None:
    environment = build_agent_worker_environment(
        backend_kind="codex",
        task_environment={"PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1"},
        ambient_environment={
            "OPENAI_API_KEY": "openai-secret",
            "ANTHROPIC_API_KEY": "wrong-backend-secret",
            "SLURM_JOB_ID": "123",
            "AWS_SECRET_ACCESS_KEY": "ambient-secret",
            "PATH": "/ambient/bin",
        },
    )

    assert environment.mapping == {
        "HF_HUB_OFFLINE": "1",
        "PYTHONNOUSERSITE": "1",
    }
    assert environment.credential_names == frozenset()
    assert "SLURM_JOB_ID" not in environment.mapping
    assert "ANTHROPIC_API_KEY" not in environment.mapping
    assert "OPENAI_API_KEY" not in environment.mapping


@pytest.mark.parametrize(
    "task_environment,match",
    [
        ({"SLURM_JOB_ID": "123"}, "controller-only"),
        ({"OPENAI_API_KEY": "secret"}, "credential-like task"),
        ({"PATH": "/bin"}, "not worker-allowlisted"),
    ],
)
def test_agent_task_environment_rejects_unauthorized_values(
    task_environment: dict[str, str], match: str
) -> None:
    with pytest.raises(IsolationPolicyError, match=match):
        build_agent_worker_environment(
            backend_kind="codex",
            task_environment=task_environment,
            ambient_environment={},
        )


def test_deterministic_gate_environment_is_exact_and_credential_free() -> None:
    environment = build_gate_worker_environment(
        {"TRTLLM_MODELING_V2": "require", "PYTHONNOUSERSITE": "1"}
    )
    assert environment.values == (
        ("PYTHONNOUSERSITE", "1"),
        ("TRTLLM_MODELING_V2", "require"),
    )
    assert environment.credential_names == frozenset()

    with pytest.raises(IsolationPolicyError, match="cannot receive credentials"):
        build_gate_worker_environment({"OPENAI_API_KEY": "secret"})


def test_agent_credential_attachment_adds_only_fixed_read_only_file(tmp_path: Path) -> None:
    paths = _layout(tmp_path)
    isolation = build_worker_isolation(
        controller_mounts=(Mount(paths["host"], Path("/container/shared"), False),),
        workspace=paths["workspace"],
        repository=paths["repository"],
        mailbox=paths["mailbox"],
        input_bundle=paths["input"],
        checkpoint=paths["checkpoint"],
    )
    provision = prepare_credential_provision(
        workspace=paths["workspace"],
        binding=CredentialBinding(
            "run-1",
            "entry-a",
            "attempt-1",
            "a" * 64,
            1,
            "codex",
        ),
        ambient_environment={"OPENAI_API_KEY": "secret-value"},
    )

    attached = attach_agent_credential_provision(isolation, provision)
    assert attached.mounts[:-1] == isolation.mounts
    assert attached.mounts[-1].target == CREDENTIAL_MOUNT_PATH
    assert attached.mounts[-1].read_only
    assert attached.mounts[-1].source.is_file()
    assert attached.credential_descriptor == provision.descriptor
    assert "secret-value" not in json.dumps(
        attached.credential_descriptor.to_public_dict(), sort_keys=True
    )
