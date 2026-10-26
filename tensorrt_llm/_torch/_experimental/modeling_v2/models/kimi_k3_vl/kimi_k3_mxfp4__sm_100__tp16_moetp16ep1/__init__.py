# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 (MXFP4) / sm_100 / tp16 attention, routed experts moe_tp 16 x moe_ep 1."""

from tensorrt_llm._torch._experimental.modeling_v2._native import load_target_ops

# This build carries the target, or routing gets a NoTarget here; its native ops are
# the family's shared ones (load_family_ops).
load_target_ops(__name__)
