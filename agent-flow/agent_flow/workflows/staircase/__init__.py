# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Slurm-first Staircase control plane for ModelingV2 bring-up and tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .task_schema import NormalizedTask


def main(argv: list[str] | None = None) -> None:
    """Run the Staircase CLI without importing runtime dependencies eagerly."""
    from .cli import main as cli_main

    cli_main(argv)


__all__ = ["main"]
