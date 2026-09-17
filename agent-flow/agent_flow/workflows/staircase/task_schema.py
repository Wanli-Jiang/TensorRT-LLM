# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation for a Staircase task consumed by the AgentTeam workflow."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_PATH_FIELDS = (
    "reference_code_path",
    "checkpoint_path",
    "trtllm_repo_path",
)
OPTIONAL_LIST_FIELDS = ("completion_criteria", "implements_tips")
TARGET_FIELD = "target"
TARGET_REQUIRED_FIELDS = ("family", "checkpoint", "gpu_arch", "parallel")
ACCURACY_ANCHOR_FIELD = "accuracy_anchor"
ACCURACY_ANCHOR_REQUIRED_FIELDS = ("benchmark", "score", "tolerance", "source")
SLURM_ENVIRONMENT_FIELD = "slurm-environment"
SLURM_REQUIRED_FIELDS = ("slurm_partition", "docker_image")
SLURM_OPTIONAL_STRING_FIELDS = ("slurm_account", "slurm_qos")
SMITH_FIELD = "smith"
SMITH_INTEGER_FIELDS = (
    "max_parallel_jobs",
    "max_nodes_per_job",
    "max_total_nodes",
)

_MODE_VALUES = frozenset({"onboard", "tune"})
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_GPU_ARCH_RE = re.compile(r"^sm_[0-9]{2,3}$")
_PARALLEL_RE = re.compile(r"^(?:tp|tep|dep)([1-9][0-9]*)$")


class TaskSchemaError(ValueError):
    """Raised when a Staircase task is not usable by the wrapper."""


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_target(value: object, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"'{TARGET_FIELD}' must be a mapping, got {type(value).__name__}")
        return
    for field in TARGET_REQUIRED_FIELDS:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            errors.append(f"'{TARGET_FIELD}.{field}' must be a non-empty string")

    for field in ("family", "checkpoint"):
        item = value.get(field)
        if isinstance(item, str) and not _SLUG_RE.fullmatch(item):
            errors.append(
                f"'{TARGET_FIELD}.{field}' must contain lowercase letters, digits, and underscores"
            )
    gpu_arch = value.get("gpu_arch")
    if isinstance(gpu_arch, str) and not _GPU_ARCH_RE.fullmatch(gpu_arch):
        errors.append(f"'{TARGET_FIELD}.gpu_arch' must look like 'sm_100' or 'sm_103'")
    parallel = value.get("parallel")
    if isinstance(parallel, str) and not _PARALLEL_RE.fullmatch(parallel):
        errors.append(f"'{TARGET_FIELD}.parallel' must be tp<N>, tep<N>, or dep<N>")


def _validate_accuracy_anchor(value: object, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"'{ACCURACY_ANCHOR_FIELD}' must be a mapping, got {type(value).__name__}")
        return
    for field in ACCURACY_ANCHOR_REQUIRED_FIELDS:
        if field not in value:
            errors.append(f"'{ACCURACY_ANCHOR_FIELD}.{field}' is required")
    for field in ("benchmark", "source"):
        item = value.get(field)
        if field in value and (not isinstance(item, str) or not item.strip()):
            errors.append(f"'{ACCURACY_ANCHOR_FIELD}.{field}' must be a non-empty string")
    for field in ("score", "tolerance"):
        item = value.get(field)
        if field in value and (
            not _is_number(item) or not math.isfinite(float(item)) or float(item) < 0
        ):
            errors.append(f"'{ACCURACY_ANCHOR_FIELD}.{field}' must be a finite non-negative number")


def _validate_slurm_environment(value: object, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"'{SLURM_ENVIRONMENT_FIELD}' must be a mapping, got {type(value).__name__}")
        return
    for field in SLURM_REQUIRED_FIELDS:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            errors.append(f"'{SLURM_ENVIRONMENT_FIELD}.{field}' must be a non-empty string")
    for field in SLURM_OPTIONAL_STRING_FIELDS:
        item = value.get(field)
        if field in value and (not isinstance(item, str) or not item.strip()):
            errors.append(f"'{SLURM_ENVIRONMENT_FIELD}.{field}' must be a non-empty string")


def _validate_smith(value: object, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"'{SMITH_FIELD}' must be a mapping, got {type(value).__name__}")
        return
    for field in SMITH_INTEGER_FIELDS:
        item = value.get(field)
        if field in value and (not isinstance(item, int) or isinstance(item, bool) or item < 1):
            errors.append(f"'{SMITH_FIELD}.{field}' must be an integer >= 1")
    max_nodes_per_job = value.get("max_nodes_per_job")
    max_total_nodes = value.get("max_total_nodes")
    if (
        isinstance(max_nodes_per_job, int)
        and not isinstance(max_nodes_per_job, bool)
        and isinstance(max_total_nodes, int)
        and not isinstance(max_total_nodes, bool)
        and max_nodes_per_job > max_total_nodes
    ):
        errors.append("'smith.max_nodes_per_job' cannot exceed 'smith.max_total_nodes'")


def load_and_validate_task_yaml(path: str | Path) -> dict[str, Any]:
    """Load one task and batch all validation errors into one exception."""
    task_path = Path(path)
    if not task_path.is_file():
        raise TaskSchemaError(f"task file not found: {task_path}")
    try:
        data = yaml.safe_load(task_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskSchemaError(f"{task_path} is not valid YAML: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TaskSchemaError(
            f"{task_path} must be a YAML mapping at the top level, got {type(data).__name__}"
        )

    errors: list[str] = []
    mode = data.get("mode", "onboard")
    if mode not in _MODE_VALUES:
        errors.append("'mode' must be either 'onboard' or 'tune'")

    for field in REQUIRED_PATH_FIELDS:
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'{field}' must be a non-empty string")
        elif not Path(value).exists():
            errors.append(f"'{field}' points to a non-existent path: {value}")

    if TARGET_FIELD not in data:
        errors.append(f"missing required field '{TARGET_FIELD}'")
    else:
        _validate_target(data[TARGET_FIELD], errors)

    if ACCURACY_ANCHOR_FIELD in data:
        _validate_accuracy_anchor(data[ACCURACY_ANCHOR_FIELD], errors)

    for field in OPTIONAL_LIST_FIELDS:
        value = data.get(field)
        if value is None:
            data[field] = []
        elif not isinstance(value, list):
            errors.append(f"'{field}' must be a list of strings")
        else:
            for index, item in enumerate(value):
                if not isinstance(item, str):
                    errors.append(f"'{field}[{index}]' must be a string")

    if SLURM_ENVIRONMENT_FIELD in data:
        _validate_slurm_environment(data[SLURM_ENVIRONMENT_FIELD], errors)
    if SMITH_FIELD in data:
        _validate_smith(data[SMITH_FIELD], errors)

    if errors:
        bullet = "\n  - "
        raise TaskSchemaError(
            f"{task_path} failed Staircase schema validation:{bullet}{bullet.join(errors)}"
        )
    return data


def has_slurm_environment(data: Mapping[str, Any]) -> bool:
    """Return whether this task asks agents to submit Slurm work."""
    return SLURM_ENVIRONMENT_FIELD in data


def world_size(data: Mapping[str, Any]) -> int:
    """Return the target world size encoded by its parallel path segment."""
    parallel = data[TARGET_FIELD]["parallel"]
    match = _PARALLEL_RE.fullmatch(parallel)
    if match is None:
        raise TaskSchemaError(f"unparsable target parallel segment: {parallel!r}")
    return int(match.group(1))


def target_relpath(data: Mapping[str, Any]) -> str:
    """Return the target directory relative to the TensorRT-LLM checkout."""
    target = data[TARGET_FIELD]
    return (
        "tensorrt_llm/_torch/modeling_v2/models/"
        f"{target['family']}/targets/{target['checkpoint']}/"
        f"{target['gpu_arch']}/{target['parallel']}"
    )


__all__ = [
    "ACCURACY_ANCHOR_FIELD",
    "OPTIONAL_LIST_FIELDS",
    "REQUIRED_PATH_FIELDS",
    "SLURM_ENVIRONMENT_FIELD",
    "SMITH_FIELD",
    "TARGET_FIELD",
    "TaskSchemaError",
    "has_slurm_environment",
    "load_and_validate_task_yaml",
    "target_relpath",
    "world_size",
]
