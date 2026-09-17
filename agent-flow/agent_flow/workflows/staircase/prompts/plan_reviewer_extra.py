# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staircase extension for the AgentTeam PlanReviewer."""

from __future__ import annotations

from ._common import compose_extension

_ROLE_CONTRACT = """\
## Staircase plan-review responsibilities

Reject a plan that guesses checkpoint semantics or topology, makes a catalog
entry depend on target code, groups multiple catalog entries into one Smith
item, assembles against an unreviewed gap, or treats an unmeasured kernel as a
final choice. Confirm that built-in per-model code is used only as prior art
or a cross-check and that the final ModelingV2 target remains self-contained.

Check that every item is labeled `[Smith]`, `[Assembler]`, or `[Tuner]`, that
the ordering and dependencies are executable by the single Coder/Reviewer
loop, and that any parallel Smith Slurm work has explicit job/node ceilings,
separate artifacts, and a deterministic aggregation step. Reject claims of
parallel autonomous Smith agents because the current AgentTeam workflow does
not create them. Also reject missing independent references, missing hard
gates, accuracy anchors without provenance, or Slurm commands presented as
evidence before they have actually run.
"""

SYSTEM_PROMPT_EXTENSION = compose_extension(_ROLE_CONTRACT)

__all__ = ["SYSTEM_PROMPT_EXTENSION"]
