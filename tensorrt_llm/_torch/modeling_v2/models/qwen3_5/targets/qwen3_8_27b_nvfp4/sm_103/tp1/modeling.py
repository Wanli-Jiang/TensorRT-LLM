# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target for Qwen3.8-27B-NVFP4 on one GB300.

This first target intentionally reuses the production Qwen3.5 dense VLM
implementation.  The ModelingV2 routing boundary still provides an exact,
auditable deployment identity while the mature implementation owns the
hybrid GDN/GQA forward, text-only VLM construction, mixed-precision modules,
and cache lifecycle.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5VLModel
from tensorrt_llm._torch.models.modeling_utils import register_auto_model

from . import weights as _weights

TARGET_CHECKPOINT = "qwen3_8_27b_nvfp4"
TARGET_GPU_ARCH = "sm_103"
TARGET_PARALLEL = "tp1"
DELEGATED_MODEL_CLASS = Qwen3_5VLModel

# Qualified names use ``namespace::op``. Unqualified declarations in older
# targets continue to mean ``trtllm::op`` in the target-contract test.
REQUIRED_TRTLLM_OPS = (
    "tensorrt_llm::static_quantize_e4m3_per_tensor",
    "trtllm::cublas_scaled_mm",
    "trtllm::nvfp4_gemm",
    "trtllm::causal_conv1d_fwd",
    "trtllm::causal_conv1d_update",
    "trtllm::rms_norm_gated_token_major",
)


@register_auto_model("ModelingV2Qwen3827BNvfp4Sm103Tp1")
class ModelingV2Qwen3827BNvfp4Sm103Tp1(Qwen3_5VLModel):
    """Synthetic target class for the certified text-only deployment."""

    def __init__(self, model_config: ModelConfig, *args: object, **kwargs: object) -> None:
        mapping = model_config.mapping
        if not model_config.disable_mm_encoder:
            raise ValueError("Qwen3.8-27B-NVFP4 ModelingV2 requires disable_mm_encoder=True")
        if (
            mapping.world_size != 1
            or mapping.tp_size != 1
            or mapping.pp_size != 1
            or mapping.cp_size != 1
            or mapping.enable_attention_dp
        ):
            raise ValueError("Qwen3.8-27B-NVFP4 ModelingV2 supports only tp1/pp1/cp1")
        super().__init__(model_config, *args, **kwargs)

    def load_weights(
        self,
        weights: dict[str, torch.Tensor],
        weight_mapper: BaseWeightMapper,
        allow_partial_loading: bool = False,
    ) -> None:
        """Load through the production Qwen3.5 mixed-precision mapper."""
        _weights.load(self, weights, weight_mapper, allow_partial_loading)


@register_auto_model("ModelingV2Qwen3827BNvfp4TextSm103Tp1")
class ModelingV2Qwen3827BNvfp4TextSm103Tp1(Qwen3_5ForCausalLM):
    """Exact inner decoder selected by the delegated composite target."""
