# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt bundle for the Staircase AgentTeam specialization."""

from __future__ import annotations

from agent_flow.workflows.agent_team.prompts import DEFAULT_PROMPTS, PromptBundle

from . import coder_extra, plan_drafter_extra, plan_reviewer_extra, qa_extra, reviewer_extra
from ._common import SLURM_LOGIN_NODE_EXECUTION, TEST_COMMAND_CACHE


def build_staircase_prompts(*, include_slurm_environment: bool = False) -> PromptBundle:
    """Build a task-scoped Staircase prompt bundle."""
    prompts = DEFAULT_PROMPTS.with_extensions(
        plan_drafter=plan_drafter_extra.SYSTEM_PROMPT_EXTENSION,
        plan_reviewer=plan_reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        coder=coder_extra.SYSTEM_PROMPT_EXTENSION,
        reviewer=reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        qa=qa_extra.SYSTEM_PROMPT_EXTENSION,
    )
    if not include_slurm_environment:
        return prompts
    build_phase = "\n\n".join((SLURM_LOGIN_NODE_EXECUTION, TEST_COMMAND_CACHE))
    return prompts.with_extensions(
        plan_drafter=SLURM_LOGIN_NODE_EXECUTION,
        plan_reviewer=SLURM_LOGIN_NODE_EXECUTION,
        coder=build_phase,
        reviewer=build_phase,
        qa=build_phase,
    )


STAIRCASE_PROMPTS = build_staircase_prompts()

__all__ = ["STAIRCASE_PROMPTS", "build_staircase_prompts"]
