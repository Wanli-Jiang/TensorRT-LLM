# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport-neutral role prompts for the Staircase workflow."""

from __future__ import annotations

from agent_flow.workflows.agent_team.prompts import PromptBundle

from .coder import SYSTEM_PROMPT as CODER_SYSTEM_PROMPT
from .plan_drafter import SYSTEM_PROMPT as PLAN_DRAFTER_SYSTEM_PROMPT
from .plan_reviewer import SYSTEM_PROMPT as PLAN_REVIEWER_SYSTEM_PROMPT
from .profiles import DomainProfile, DomainPromptExtension, get_domain_prompt_extension
from .qa import SYSTEM_PROMPT as QA_SYSTEM_PROMPT
from .reviewer import SYSTEM_PROMPT as REVIEWER_SYSTEM_PROMPT

STAIRCASE_PROMPTS = PromptBundle(
    plan_drafter=PLAN_DRAFTER_SYSTEM_PROMPT,
    plan_reviewer=PLAN_REVIEWER_SYSTEM_PROMPT,
    coder=CODER_SYSTEM_PROMPT,
    reviewer=REVIEWER_SYSTEM_PROMPT,
    qa=QA_SYSTEM_PROMPT,
)
"""Base prompts shared by onboarding and tuning before domain specialization."""

DEFAULT_PROMPTS = STAIRCASE_PROMPTS
"""Default prompt bundle used by Staircase role workers."""


def build_staircase_prompts(
    domain_profile: DomainProfile | None = None,
) -> PromptBundle:
    """Build the five-role bundle for an optional typed domain profile.

    Domain profiles specialize only implementation and independent review.
    Planning and Stage-level QA keep one invariant contract for every WorkItem.

    Args:
        domain_profile: Smith, Assembler, or Tuner specialization selected by
            the controller from the typed WorkItem kind. ``None`` returns the
            base bundle.

    Returns:
        An immutable transport-neutral prompt bundle.
    """
    if domain_profile is None:
        return STAIRCASE_PROMPTS
    extension = get_domain_prompt_extension(domain_profile)
    return STAIRCASE_PROMPTS.with_extensions(
        coder=extension.coder,
        reviewer=extension.reviewer,
    )


__all__ = [
    "CODER_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "DomainProfile",
    "DomainPromptExtension",
    "PLAN_DRAFTER_SYSTEM_PROMPT",
    "PLAN_REVIEWER_SYSTEM_PROMPT",
    "PromptBundle",
    "QA_SYSTEM_PROMPT",
    "REVIEWER_SYSTEM_PROMPT",
    "STAIRCASE_PROMPTS",
    "build_staircase_prompts",
    "get_domain_prompt_extension",
]
