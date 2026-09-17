# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Staircase's AgentTeam prompt extensions."""

from __future__ import annotations

from agent_flow.workflows.agent_team.prompts import DEFAULT_PROMPTS
from agent_flow.workflows.staircase.prompts import build_staircase_prompts

_ROLES = ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa")


def test_every_prompt_extends_the_agent_team_role() -> None:
    prompts = build_staircase_prompts()

    for role in _ROLES:
        prompt = getattr(prompts, role)
        base = getattr(DEFAULT_PROMPTS, role)
        assert prompt.startswith(base.rstrip())
        assert "## Staircase and ModelingV2 boundary" in prompt
        assert "built-in TensorRT-LLM" in prompt
        assert "TRTLLM_MODELING_V2=require" in prompt


def test_role_mapping_is_explicit_and_honest() -> None:
    prompts = build_staircase_prompts()

    for role in _ROLES:
        prompt = getattr(prompts, role)
        assert "[Smith]" in prompt
        assert "[Assembler]" in prompt
        assert "[Tuner]" in prompt
        assert "not extra orchestrator processes" in prompt

    assert "one catalog entry at a time" in prompts.coder
    assert "independently rerun" in prompts.reviewer
    assert "hard gates" in prompts.qa


def test_local_prompts_do_not_claim_slurm_mode() -> None:
    prompts = build_staircase_prompts()

    for role in _ROLES:
        assert "## Login-node orchestration and Slurm execution" not in getattr(prompts, role)
    assert "## Shared `test_command.md`" not in prompts.coder


def test_slurm_prompts_add_execution_and_command_cache_by_role() -> None:
    prompts = build_staircase_prompts(include_slurm_environment=True)

    for role in _ROLES:
        prompt = getattr(prompts, role)
        assert "## Login-node orchestration and Slurm execution" in prompt
        assert "Do not install packages or rebuild TensorRT-LLM on the login node" in prompt

    for role in ("coder", "reviewer", "qa"):
        prompt = getattr(prompts, role)
        assert "## Shared `test_command.md`" in prompt
        assert "parallel Slurm work coordinated by one Smith-mode Coder" in prompt

    for role in ("plan_drafter", "plan_reviewer"):
        assert "## Shared `test_command.md`" not in getattr(prompts, role)


def test_extensions_do_not_use_the_removed_controller_outcome_protocol() -> None:
    prompts = build_staircase_prompts(include_slurm_environment=True)
    removed_outcomes = (
        "RESOURCE_ESCALATION",
        "CANDIDATE",
        "DRAFTED",
        "result manifest",
        "trusted runner",
    )

    for role in _ROLES:
        prompt = getattr(prompts, role)
        for outcome in removed_outcomes:
            assert outcome not in prompt
