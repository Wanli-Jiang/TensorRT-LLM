# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wheel and console-entrypoint smoke tests for the Staircase package."""

from __future__ import annotations

import configparser
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from zipfile import ZipFile

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE_PREFIX = "agent_flow/workflows/staircase/"
_STAIRCASE_ROOT = _PROJECT_ROOT / _PACKAGE_PREFIX
_EXPECTED_ASSETS = {
    f"{_PACKAGE_PREFIX}README.md",
    f"{_PACKAGE_PREFIX}onboarding/README.md",
    f"{_PACKAGE_PREFIX}onboarding/task.example.yaml",
    f"{_PACKAGE_PREFIX}tuning/README.md",
    f"{_PACKAGE_PREFIX}tuning/task.example.yaml",
    f"{_PACKAGE_PREFIX}references/modeling_v2_contract.md",
}
_EXPECTED_MODULES = {
    f"{_PACKAGE_PREFIX}{path.relative_to(_STAIRCASE_ROOT).as_posix()}"
    for path in _STAIRCASE_ROOT.rglob("*.py")
}
_BLOCKED_HELP_IMPORTS = (
    "agent_flow.workflows.staircase.common.slurm",
    "anyio",
    "claude_agent_sdk",
    "cuda",
    "cupy",
    "pyslurm",
    "tensorrt",
    "tensorrt_llm",
    "torch",
)


@pytest.fixture(scope="module")
def staircase_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a non-editable wheel from an isolated project copy."""
    root = tmp_path_factory.mktemp("staircase-wheel")
    project = root / "agent-flow"
    shutil.copytree(
        _PROJECT_ROOT,
        project,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.egg-info",
            ".pytest_cache",
            "build",
            "dist",
        ),
    )
    wheel_dir = root / "wheelhouse"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(project),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = tuple(wheel_dir.glob("agent_flow-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def test_pyproject_declares_entrypoint_and_all_runtime_assets() -> None:
    data = tomllib.loads((_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert data["project"]["scripts"]["staircase"] == ("agent_flow.workflows.staircase.cli:main")
    package_data = set(data["tool"]["setuptools"]["package-data"]["agent_flow.workflows.staircase"])
    assert package_data == {
        "*.md",
        "onboarding/*.yaml",
        "onboarding/*.md",
        "tuning/*.yaml",
        "tuning/*.md",
        "references/*.md",
    }


def test_noneditable_wheel_contains_staircase_assets_and_entrypoint(
    staircase_wheel: Path,
) -> None:
    with ZipFile(staircase_wheel) as archive:
        names = set(archive.namelist())
        assert _EXPECTED_MODULES <= names
        assert _EXPECTED_ASSETS <= names
        entrypoint_files = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        assert len(entrypoint_files) == 1
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entrypoint_files[0]).decode("utf-8"))

    assert parser["console_scripts"]["staircase"] == ("agent_flow.workflows.staircase.cli:main")


def test_wheel_console_entrypoint_help_needs_no_live_service(
    staircase_wheel: Path,
    tmp_path: Path,
) -> None:
    environment_dir = tmp_path / "venv"
    create_environment = subprocess.run(
        [sys.executable, "-m", "venv", str(environment_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert create_environment.returncode == 0, create_environment.stdout + create_environment.stderr

    python = environment_dir / "bin" / "python"
    isolated_environment = os.environ.copy()
    isolated_environment.pop("PYTHONPATH", None)
    isolated_environment["PYTHONNOUSERSITE"] = "1"
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(staircase_wheel),
        ],
        cwd=tmp_path,
        env=isolated_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr

    import_guard = tmp_path / "import-guard"
    import_guard.mkdir()
    (import_guard / "sitecustomize.py").write_text(
        """\
import importlib.abc
import sys

BLOCKED = {blocked!r}


class BlockedLiveDependencyFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(f"{{name}}.") for name in BLOCKED):
            raise ImportError(f"staircase --help imported blocked dependency: {{fullname}}")
        return None


sys.meta_path.insert(0, BlockedLiveDependencyFinder())
""".format(blocked=_BLOCKED_HELP_IMPORTS),
        encoding="utf-8",
    )
    executable = environment_dir / "bin" / "staircase"
    assert executable.is_file(), install.stdout + install.stderr
    environment = isolated_environment.copy()
    environment["PYTHONPATH"] = str(import_guard)
    result = subprocess.run(
        [str(executable), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "{onboard,tune,status,cancel,respond}" in result.stdout
    assert "Slurm-first Staircase control plane" in result.stdout
