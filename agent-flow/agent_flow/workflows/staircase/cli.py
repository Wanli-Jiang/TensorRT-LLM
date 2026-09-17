# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Console entry point for the Staircase AgentTeam specialization."""

from __future__ import annotations

import sys
from pathlib import Path

from agent_flow.workflows.agent_team.cli import _parse_args
from agent_flow.workflows.agent_team.cli import main as _team_main
from agent_flow.workflows.agent_team.state import STATE_FILENAME

from .prompts import build_staircase_prompts
from .task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    load_and_validate_task_yaml,
    target_relpath,
    world_size,
)

_CODER_RESET_FLAG = "--coder-context-reset-interval"
_REPLAN_FLAG = "--replan-on-qa"
STAIRCASE_CODER_CONTEXT_RESET_INTERVAL = 5


def _has_option(argv: list[str], option: str) -> bool:
    """Return whether ``argv`` contains ``option`` in split or equals form."""
    return any(argument == option or argument.startswith(f"{option}=") for argument in argv)


def _with_staircase_defaults(argv: list[str] | None) -> list[str]:
    """Apply Staircase defaults without overriding explicit AgentTeam flags."""
    effective = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(effective, _CODER_RESET_FLAG):
        effective.extend([_CODER_RESET_FLAG, str(STAIRCASE_CODER_CONTEXT_RESET_INTERVAL)])
    if not _has_option(effective, _REPLAN_FLAG):
        effective.append(_REPLAN_FLAG)
    return effective


def _effective_task_path(*, task: Path, workspace: Path, clean: bool) -> Path:
    """Use the workspace task when resuming an AgentTeam checkpoint."""
    stored_task = workspace / "task.yaml"
    state_path = workspace / STATE_FILENAME
    if not clean and state_path.is_file() and stored_task.is_file():
        return stored_task
    return task


def main(argv: list[str] | None = None) -> None:
    """Validate the Staircase task, then run the existing AgentTeam loop."""
    effective_argv = _with_staircase_defaults(argv)
    args = _parse_args(effective_argv)
    task_path = _effective_task_path(
        task=args.task,
        workspace=args.workspace,
        clean=args.clean,
    )
    try:
        task_data = load_and_validate_task_yaml(task_path)
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(f"[staircase] mode: {task_data.get('mode', 'onboard')}")
    print(f"[staircase] target: {target_relpath(task_data)}")
    print(f"[staircase] world size: {world_size(task_data)} GPU(s)")

    prompts = build_staircase_prompts(
        include_slurm_environment=has_slurm_environment(task_data),
    )
    _team_main(effective_argv, prompts=prompts)


if __name__ == "__main__":
    main()
