# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the thin Staircase CLI wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.agent_team.state import STATE_FILENAME
from agent_flow.workflows.staircase import cli


def _write_task(path: Path, *, slurm: bool = False) -> Path:
    root = path.parent
    reference = root / f"{path.stem}-reference.py"
    reference.write_text("# reference\n", encoding="utf-8")
    checkpoint = root / f"{path.stem}-checkpoint"
    checkpoint.mkdir(exist_ok=True)
    repository = root / f"{path.stem}-repo"
    repository.mkdir(exist_ok=True)
    data: dict[str, object] = {
        "reference_code_path": str(reference),
        "checkpoint_path": str(checkpoint),
        "trtllm_repo_path": str(repository),
        "target": {
            "family": "gpt_oss",
            "checkpoint": "gpt_oss_120b",
            "gpu_arch": "sm_103",
            "parallel": "tp1",
        },
    }
    if slurm:
        data["slurm-environment"] = {
            "slurm_partition": "batch",
            "docker_image": "/shared/trtllm.sqsh",
        }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_cli_forwards_to_agent_team_with_staircase_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task = _write_task(tmp_path / "task.yaml")
    workspace = tmp_path / "workspace"
    observed: dict[str, object] = {}

    def fake_team_main(argv: list[str], *, prompts: object) -> None:
        observed["argv"] = argv
        observed["prompts"] = prompts

    monkeypatch.setattr(cli, "_team_main", fake_team_main)
    cli.main(["--task", str(task), "--workspace", str(workspace)])

    argv = observed["argv"]
    assert isinstance(argv, list)
    assert "--replan-on-qa" in argv
    reset_index = argv.index("--coder-context-reset-interval")
    assert argv[reset_index + 1] == "5"
    assert "Login-node orchestration" not in observed["prompts"].coder
    output = capsys.readouterr().out
    assert "mode: onboard" in output
    assert "world size: 1 GPU(s)" in output


def test_cli_preserves_explicit_context_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _write_task(tmp_path / "task.yaml")
    observed: dict[str, list[str]] = {}

    def fake_team_main(argv: list[str], *, prompts: object) -> None:
        del prompts
        observed["argv"] = argv

    monkeypatch.setattr(cli, "_team_main", fake_team_main)
    cli.main(
        [
            "--task",
            str(task),
            "--workspace",
            str(tmp_path / "workspace"),
            "--coder-context-reset-interval=9",
        ]
    )

    argv = observed["argv"]
    assert "--coder-context-reset-interval=9" in argv
    assert "--coder-context-reset-interval" not in argv


def test_cli_builds_slurm_prompts_from_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _write_task(tmp_path / "task.yaml", slurm=True)
    observed: dict[str, object] = {}

    def fake_team_main(argv: list[str], *, prompts: object) -> None:
        del argv
        observed["prompts"] = prompts

    monkeypatch.setattr(cli, "_team_main", fake_team_main)
    cli.main(["--task", str(task), "--workspace", str(tmp_path / "workspace")])

    assert "## Shared `test_command.md`" in observed["prompts"].coder


def test_resume_uses_stored_task_to_select_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supplied_task = _write_task(tmp_path / "supplied.yaml", slurm=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_task(workspace / "task.yaml", slurm=True)
    (workspace / STATE_FILENAME).write_text("{}\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_team_main(argv: list[str], *, prompts: object) -> None:
        del argv
        observed["prompts"] = prompts

    monkeypatch.setattr(cli, "_team_main", fake_team_main)
    cli.main(["--task", str(supplied_task), "--workspace", str(workspace)])

    assert "## Shared `test_command.md`" in observed["prompts"].coder


def test_clean_uses_supplied_task_instead_of_stored_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supplied_task = _write_task(tmp_path / "supplied.yaml", slurm=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_task(workspace / "task.yaml", slurm=True)
    (workspace / STATE_FILENAME).write_text("{}\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def fake_team_main(argv: list[str], *, prompts: object) -> None:
        del argv
        observed["prompts"] = prompts

    monkeypatch.setattr(cli, "_team_main", fake_team_main)
    cli.main(["--task", str(supplied_task), "--workspace", str(workspace), "--clean"])

    assert "## Shared `test_command.md`" not in observed["prompts"].coder


def test_invalid_task_fails_before_agent_team_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("target: {}\n", encoding="utf-8")
    called = False

    def fake_team_main(argv: list[str], *, prompts: object) -> None:
        del argv, prompts
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "_team_main", fake_team_main)
    with pytest.raises(SystemExit) as error:
        cli.main(["--task", str(invalid), "--workspace", str(tmp_path / "workspace")])

    assert error.value.code == 2
    assert not called
