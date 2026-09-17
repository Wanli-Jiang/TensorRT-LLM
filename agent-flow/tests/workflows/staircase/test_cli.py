# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public Staircase CLI surface."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from agent_flow.workflows.staircase import cli


def test_start_defaults_to_slurm() -> None:
    args = cli._parse_args(["onboard", "--task", "task.yaml", "--workspace", "workspace"])
    assert args.execution == "slurm"
    assert args.command == "onboard"
    assert args.preflight_facts is None


def test_start_accepts_explicit_observed_preflight_facts() -> None:
    args = cli._parse_args(
        [
            "onboard",
            "--task",
            "task.yaml",
            "--workspace",
            "workspace",
            "--preflight-facts",
            "observed.json",
        ]
    )
    assert args.preflight_facts == Path("observed.json")


@pytest.mark.parametrize("command", ["onboard", "tune"])
def test_start_accepts_explicit_local_mode(command: str) -> None:
    args = cli._parse_args(
        [command, "--task", "task.yaml", "--workspace", "workspace", "--execution", "local"]
    )
    assert args.execution == "local"


def test_status_surface_is_read_only_shape() -> None:
    args = cli._parse_args(["status", "--workspace", "workspace", "--json"])
    assert args.command == "status"
    assert args.as_json is True
    assert not hasattr(args, "task")


def test_respond_requires_response_file() -> None:
    with pytest.raises(SystemExit):
        cli._parse_args(["respond", "--workspace", "workspace", "--request-id", "request-1"])


def test_help_does_not_import_runtime_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "agent_flow.layers",
        "agent_flow.backends.claude_code",
        "agent_flow.backends.codex",
        "agent_flow.workflows.staircase.workflow",
        "agent_flow.workflows.staircase.common.slurm",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
    assert "agent_flow.workflows.staircase.workflow" not in sys.modules
    assert "agent_flow.workflows.staircase.common.slurm" not in sys.modules


def test_invalid_task_never_imports_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, "agent_flow.workflows.staircase.workflow", raising=False)
    task = tmp_path / "task.yaml"
    task.write_text("unknown: true\n", encoding="utf-8")

    result = cli._run_start(
        cli._parse_args(
            ["onboard", "--task", str(task), "--workspace", str(tmp_path / "workspace")]
        )
    )

    assert result == 2
    assert "agent_flow.workflows.staircase.workflow" not in sys.modules


def test_public_runtime_error_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli, "_run_status", lambda _args: (_ for _ in ()).throw(RuntimeError("bad"))
    )
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["status", "--workspace", "/missing"])
    assert exc_info.value.code == 2
    assert capsys.readouterr().err == "error: bad\n"


def test_package_main_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "agent_flow.workflows.staircase.cli", raising=False)
    module = importlib.reload(importlib.import_module("agent_flow.workflows.staircase"))
    assert callable(module.main)
    assert "agent_flow.workflows.staircase.cli" not in sys.modules
