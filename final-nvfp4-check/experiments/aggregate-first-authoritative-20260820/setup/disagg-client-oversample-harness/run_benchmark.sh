#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly OVERLAY=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/setup/disagg-client-oversample
export PYTHONPATH="${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
exec bash /lustre/fsw/portfolios/coreai/users/williamj/nemotron-ultra-benchmarking/disagg/bin/harness/trtllm-disagg-benchmark/run_benchmark.sh "$@"
