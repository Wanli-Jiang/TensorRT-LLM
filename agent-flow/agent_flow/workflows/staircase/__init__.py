# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelingV2 bring-up specialization built on the AgentTeam workflow."""

from __future__ import annotations

from typing import Any

__all__ = ["STAIRCASE_PROMPTS", "build_staircase_prompts", "main"]


def __getattr__(name: str) -> Any:
    """Load the CLI and prompt bundle only when requested."""
    if name == "main":
        from .cli import main

        return main
    if name in {"STAIRCASE_PROMPTS", "build_staircase_prompts"}:
        from .prompts import STAIRCASE_PROMPTS, build_staircase_prompts

        return {
            "STAIRCASE_PROMPTS": STAIRCASE_PROMPTS,
            "build_staircase_prompts": build_staircase_prompts,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
