#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged disaggregated benchmark power-log processor."""

from __future__ import annotations

import runpy

PROCESSOR = (
    "/lustre/fsw/portfolios/coreai/users/williamj/nemotron-ultra-benchmarking/"
    "disagg/bin/harness/trtllm-disagg-benchmark/process_power_logs.py"
)

runpy.run_path(PROCESSOR, run_name="__main__")
