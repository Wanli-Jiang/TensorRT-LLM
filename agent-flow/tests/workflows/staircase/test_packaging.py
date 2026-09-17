# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging checks for the Staircase wrapper and assets."""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_console_entry_point_and_package_data_are_declared() -> None:
    metadata = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"]["staircase"] == (
        "agent_flow.workflows.staircase.cli:main"
    )
    assets = metadata["tool"]["setuptools"]["package-data"]["agent_flow.workflows.staircase"]
    assert "*.md" in assets
    assert "*.yaml" in assets
    assert "references/*.md" in assets


def test_packaged_source_assets_exist() -> None:
    package = _PROJECT_ROOT / "agent_flow" / "workflows" / "staircase"

    for relative in (
        "README.md",
        "task.example.yaml",
        "task.slurm.example.yaml",
        "references/modeling_v2_contract.md",
    ):
        assert (package / relative).is_file()


def test_package_import_defers_cli_and_prompts() -> None:
    for name in (
        "agent_flow.workflows.staircase",
        "agent_flow.workflows.staircase.cli",
        "agent_flow.workflows.staircase.prompts",
    ):
        sys.modules.pop(name, None)

    module = importlib.import_module("agent_flow.workflows.staircase")

    assert "agent_flow.workflows.staircase.cli" not in sys.modules
    assert "agent_flow.workflows.staircase.prompts" not in sys.modules
    assert "main" in module.__all__
