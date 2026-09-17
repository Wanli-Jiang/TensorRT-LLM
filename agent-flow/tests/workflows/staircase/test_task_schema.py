# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lightweight Staircase task contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.staircase.task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    load_and_validate_task_yaml,
    target_relpath,
    world_size,
)


def _task_data(tmp_path: Path) -> dict[str, object]:
    reference = tmp_path / "modeling_example.py"
    reference.write_text("# reference\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    repository = tmp_path / "TensorRT-LLM"
    repository.mkdir()
    return {
        "mode": "onboard",
        "reference_code_path": str(reference),
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": str(repository),
        "target": {
            "family": "deepseek_v3",
            "checkpoint": "r1_0528_nvfp4",
            "gpu_arch": "sm_103",
            "parallel": "dep4",
        },
        "accuracy_anchor": {
            "benchmark": "mmlu",
            "score": 95.07,
            "tolerance": 5.0,
            "source": "reference.yaml @ deadbeef",
        },
        "completion_criteria": ["accuracy passes"],
        "implements_tips": ["reuse the local-dev build"],
    }


def _write_task(path: Path, data: object) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_valid_task_preserves_extensions_and_resolves_target(tmp_path: Path) -> None:
    data = _task_data(tmp_path)
    data["site_note"] = {"owner": "modeling"}
    path = _write_task(tmp_path / "task.yaml", data)

    loaded = load_and_validate_task_yaml(path)

    assert loaded["site_note"] == {"owner": "modeling"}
    assert not has_slurm_environment(loaded)
    assert world_size(loaded) == 4
    assert target_relpath(loaded) == (
        "tensorrt_llm/_torch/modeling_v2/models/deepseek_v3/targets/r1_0528_nvfp4/sm_103/dep4"
    )


def test_optional_lists_are_normalized(tmp_path: Path) -> None:
    data = _task_data(tmp_path)
    data.pop("completion_criteria")
    data["implements_tips"] = None

    loaded = load_and_validate_task_yaml(_write_task(tmp_path / "task.yaml", data))

    assert loaded["completion_criteria"] == []
    assert loaded["implements_tips"] == []


def test_slurm_and_parallel_smith_are_accepted(tmp_path: Path) -> None:
    data = _task_data(tmp_path)
    data["slurm-environment"] = {
        "slurm_account": "coreai_comparch_trtllm",
        "slurm_partition": "batch",
        "slurm_qos": "normal",
        "docker_image": "/shared/trtllm.sqsh",
    }
    data["smith"] = {
        "max_parallel_jobs": 4,
        "max_nodes_per_job": 2,
        "max_total_nodes": 8,
    }

    loaded = load_and_validate_task_yaml(_write_task(tmp_path / "task.yaml", data))

    assert has_slurm_environment(loaded)
    assert loaded["smith"]["max_parallel_jobs"] == 4


def test_validation_batches_independent_errors(tmp_path: Path) -> None:
    data = _task_data(tmp_path)
    data["mode"] = "production"
    data["reference_code_path"] = str(tmp_path / "missing.py")
    data["target"] = {
        "family": "../escape",
        "checkpoint": "Bad-Checkpoint",
        "gpu_arch": "hopper",
        "parallel": "single",
    }
    data["accuracy_anchor"] = {
        "benchmark": "",
        "score": True,
        "tolerance": -1,
    }
    data["completion_criteria"] = "not-a-list"
    data["slurm-environment"] = {"slurm_partition": ""}
    data["smith"] = {
        "max_parallel_jobs": False,
        "max_nodes_per_job": 4,
        "max_total_nodes": 2,
    }

    with pytest.raises(TaskSchemaError) as error:
        load_and_validate_task_yaml(_write_task(tmp_path / "task.yaml", data))

    message = str(error.value)
    for expected in (
        "'mode' must be either 'onboard' or 'tune'",
        "points to a non-existent path",
        "target.family",
        "target.checkpoint",
        "target.gpu_arch",
        "target.parallel",
        "accuracy_anchor.score",
        "accuracy_anchor.tolerance",
        "accuracy_anchor.source",
        "completion_criteria",
        "slurm-environment.docker_image",
        "smith.max_parallel_jobs",
        "cannot exceed",
    ):
        assert expected in message


def test_missing_file_and_non_mapping_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(TaskSchemaError, match="task file not found"):
        load_and_validate_task_yaml(tmp_path / "missing.yaml")

    path = _write_task(tmp_path / "task.yaml", ["not", "a", "mapping"])
    with pytest.raises(TaskSchemaError, match="top level"):
        load_and_validate_task_yaml(path)
