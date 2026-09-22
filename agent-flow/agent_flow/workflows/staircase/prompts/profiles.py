# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed domain-profile extensions for Staircase Coder and Reviewer roles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DomainProfile(str, Enum):
    """Domain specialization selected from a typed WorkItem kind."""

    SMITH = "smith"
    ASSEMBLER = "assembler"
    TUNER = "tuner"


@dataclass(frozen=True)
class DomainPromptExtension:
    """Prompt additions for the implementation and review sides of a profile."""

    coder: str
    reviewer: str


_SMITH = DomainPromptExtension(
    coder="""\
## Domain profile: Smith

Handle exactly one catalog entry. For verification, prove the requested
surface against an independent oracle on the named GPU architecture. For
onboarding, keep the entry's contract `.md`, thin wrapper `.py`, and collected
GPU test coherent; return a typed catalog-index delta instead of editing the
shared index. A receipt records an observed run for these exact files and
surface, never an inference. Do not inspect or edit another entry, recursively
launch Smith work, or assemble a target. For a collective or other vertical
multi-node probe, request a bounded resource escalation; only the trusted
supervisor may run across the assigned nodes.""",
    reviewer="""\
## Domain profile: Smith

Review only the named catalog entry. Check one-invocation wrapper scope,
contract completeness, argument/shape/dtype/error surface, independent
expected values, GPU-architecture match, receipt freshness, and the exact
entry-specific test. Independently rerun the decisive cell or matrix. Approve
an index delta only as semantic data for controller fan-in; never edit the
shared index or review a sibling entry in the same verdict.""",
)

_ASSEMBLER = DomainPromptExtension(
    coder="""\
## Domain profile: Assembler

Implement exactly the assigned target core, module, feature, weight, or routing
item. By default use only controller-declared, already integrated catalog
dependencies and keep the target self-contained and flat. If the immutable
task or approved plan selects task-scoped delegated built-in reuse, implement
only a narrow adapter to its exact named mature model and weight mapper; do not
widen that dependency or turn it into fallback. Otherwise, return a typed catalog-gap
request at an uncertified surface. Preserve checkpoint, GPU, structural feature,
and parallel-topology identity; keep workload/tuning knobs out of routing.""",
    reviewer="""\
## Domain profile: Assembler

Review the assigned assembly item and its integrated dependency hashes. Audit
route explainability, weight mapping, topology, and feature hard paths. For a
self-contained item, audit flat-forward catalog coverage and reject direct
tensor computation. For task-scoped delegated built-in reuse, verify explicit
task/plan authorization and the exact named model/mapper boundary; reject
undeclared, unbounded, sibling-target, or fallback imports. Always reject uncertified
dependencies, target-owned checkpoint configuration, or workload/tuning routing.
Rerun the smallest parity or routing evidence that can falsify the item.""",
)

_TUNER = DomainPromptExtension(
    coder="""\
## Domain profile: Tuner

Test one explicit performance hypothesis and change one declared variable.
Keep baseline and candidate checkpoint, target, workload, topology, build,
correctness gates, and measurement protocol matched; prefer paired same-session
evidence where session variance matters. Do not combine code and configuration
changes, change routing identity for a tuning knob, or promote your own result.
Return measured deltas, uncertainty, correctness/acceptance results, and a
bounded keep/reject recommendation.""",
    reviewer="""\
## Domain profile: Tuner

Independently check that the hypothesis changed one variable and that baseline
and candidate are comparable. Reproduce the decisive measurement, verify
correctness and feature-acceptance gates first, and reject confounded or
fallback runs. A faster incorrect candidate fails; a correct slowdown is an
honest negative result. Return evidence to the controller, which alone decides
promotion.""",
)

_EXTENSIONS = {
    DomainProfile.SMITH: _SMITH,
    DomainProfile.ASSEMBLER: _ASSEMBLER,
    DomainProfile.TUNER: _TUNER,
}


def get_domain_prompt_extension(profile: DomainProfile) -> DomainPromptExtension:
    """Return the Coder and Reviewer prompt extensions for ``profile``.

    Args:
        profile: Controller-selected domain specialization.

    Returns:
        The immutable role extensions for the selected profile.
    """
    return _EXTENSIONS[profile]


__all__ = [
    "DomainProfile",
    "DomainPromptExtension",
    "get_domain_prompt_extension",
]
