#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the unchanged disaggregated benchmark environment helper."""

from __future__ import annotations

import runpy

GET_ENV = (
    "/lustre/fsw/portfolios/coreai/users/williamj/nemotron-ultra-benchmarking/"
    "disagg/bin/harness/trtllm-disagg-benchmark/get_env.py"
)

runpy.run_path(GET_ENV, run_name="__main__")
