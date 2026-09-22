# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight-loading delegate for the Qwen3.8-27B-NVFP4 ModelingV2 target."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5VLModel

WEIGHT_MAPPER_CLASS = Qwen3_5MoeHfWeightMapper


def load(
    model: Qwen3_5VLModel,
    weights: dict[str, torch.Tensor],
    weight_mapper: BaseWeightMapper,
    allow_partial_loading: bool = False,
) -> None:
    """Reuse the production dense-Qwen VLM loader and mapper contract."""
    if not isinstance(weight_mapper, WEIGHT_MAPPER_CLASS):
        raise TypeError(
            "Qwen3.8-27B-NVFP4 ModelingV2 requires "
            f"{WEIGHT_MAPPER_CLASS.__name__}, got {type(weight_mapper).__name__}"
        )
    Qwen3_5VLModel.load_weights(
        model,
        weights,
        weight_mapper,
        allow_partial_loading=allow_partial_loading,
    )
