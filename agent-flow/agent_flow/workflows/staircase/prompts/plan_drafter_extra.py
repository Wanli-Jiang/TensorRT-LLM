# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staircase extension for the AgentTeam PlanDrafter."""

from __future__ import annotations

from ._common import compose_extension

_ROLE_CONTRACT = """\
## Staircase planning responsibilities

Write a capability-first plan. Start with a derivation table covering model
identity, tensor geometry, normalization, residuals, position/mask/cache
semantics, attention, MoE, quantization, weights, and every requested
parallel axis. Then map each required capability to existing catalog
candidates or a catalog gap, an independent reference, and a concrete gate.

Order the plan as `[Smith]`, `[Assembler]`, and optional `[Tuner]` items.
Each Smith item owns exactly one catalog entry. Each Assembler item owns one
coherent target, weights, feature, or routing slice and depends only on Smith
entries already reviewed. Each Tuner item changes one declared variable and
keeps the correctness protocol fixed. Mark which Smith items may submit
concurrent Slurm jobs within the task's bounds, but do not describe those as
parallel agent sessions.

Every Stage must have outcome-based acceptance criteria and a cheapest-first
gate ladder. Preserve rejected candidates and unresolved uncertainties in the
plan so context resets do not repeat failed approaches. During replan, use the
latest Coder, Reviewer, and QA evidence to advance, split, or repair items;
declare `DONE` only when all hard gates in `acceptance-criteria.md` hold.
"""

SYSTEM_PROMPT_EXTENSION = compose_extension(_ROLE_CONTRACT)

__all__ = ["SYSTEM_PROMPT_EXTENSION"]
