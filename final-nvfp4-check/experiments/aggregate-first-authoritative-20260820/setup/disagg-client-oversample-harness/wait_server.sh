#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail
exec bash /lustre/fsw/portfolios/coreai/users/williamj/nemotron-ultra-benchmarking/disagg/bin/harness/trtllm-disagg-benchmark/wait_server.sh "$@"
