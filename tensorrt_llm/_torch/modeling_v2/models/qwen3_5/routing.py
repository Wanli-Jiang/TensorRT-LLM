# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route the supported dense Qwen3.5/Qwen3.8 checkpoint to its target.

The checkpoint advertises the multimodal outer architecture even when it is
deployed text-only.  Routing therefore checks both the nested text shape and
``disable_mm_encoder``.  The mixed-precision manifest is part of the identity:
another checkpoint with the same geometry but a different FP8/NVFP4 layout
must not silently enter a target validated for these exact operands.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Optional

from ..._router_index import NULL_TRACE, ModelingV2Context, Trace

_SM = (10, 3)
_LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 16


def _expected_quantized_layers() -> frozenset[str]:
    names = {"lm_head"}
    for layer_idx, layer_type in enumerate(_LAYER_TYPES):
        prefix = f"model.language_model.layers.{layer_idx}"
        names.update(
            f"{prefix}.mlp.{projection}" for projection in ("gate_proj", "up_proj", "down_proj")
        )
        if layer_type == "linear_attention":
            names.update(
                f"{prefix}.linear_attn.{projection}"
                for projection in ("in_proj_qkv", "in_proj_z", "out_proj")
            )
        else:
            names.update(
                f"{prefix}.self_attn.{projection}"
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            )
    return frozenset(names)


_EXPECTED_QUANTIZED_LAYERS = _expected_quantized_layers()

# Nested text-config fingerprint -> checkpoint identity.
_OUTER_CHECKPOINTS = {
    (
        "qwen3_5",
        False,
        "qwen3_5_text",
        64,
        5120,
        17408,
        248320,
        24,
        4,
        256,
        16,
        48,
        128,
        128,
        4,
        4,
        True,
        False,
    ): "qwen3_8_27b_nvfp4",
}

_INNER_CHECKPOINTS = {
    (
        "qwen3_5_text",
        64,
        5120,
        17408,
        248320,
        24,
        4,
        256,
        16,
        48,
        128,
        128,
        4,
        4,
        True,
        False,
    ): "qwen3_8_27b_nvfp4",
}

_TARGETS = {
    ("qwen3_8_27b_nvfp4", "outer", "tp1"): "ModelingV2Qwen3827BNvfp4Sm103Tp1",
    ("qwen3_8_27b_nvfp4", "inner", "tp1"): ("ModelingV2Qwen3827BNvfp4TextSm103Tp1"),
}

TARGET_MODULES = {
    "ModelingV2Qwen3827BNvfp4Sm103Tp1": (
        "models.qwen3_5.targets.qwen3_8_27b_nvfp4.sm_103.tp1.modeling"
    ),
    "ModelingV2Qwen3827BNvfp4TextSm103Tp1": (
        "models.qwen3_5.targets.qwen3_8_27b_nvfp4.sm_103.tp1.modeling"
    ),
}


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _algorithm_name(value: object) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _shape(config: object) -> tuple[object, ...]:
    text = _field(config, "text_config")
    return (
        _field(config, "model_type"),
        _field(config, "language_model_only", False),
        _field(text, "model_type"),
        _field(text, "num_hidden_layers"),
        _field(text, "hidden_size"),
        _field(text, "intermediate_size"),
        _field(text, "vocab_size"),
        _field(text, "num_attention_heads"),
        _field(text, "num_key_value_heads"),
        _field(text, "head_dim"),
        _field(text, "linear_num_key_heads"),
        _field(text, "linear_num_value_heads"),
        _field(text, "linear_key_head_dim"),
        _field(text, "linear_value_head_dim"),
        _field(text, "linear_conv_kernel_dim"),
        _field(text, "full_attention_interval"),
        _field(text, "attn_output_gate"),
        _field(text, "tie_word_embeddings", _field(config, "tie_word_embeddings")),
    )


def _inner_shape(config: object) -> tuple[object, ...]:
    return (
        _field(config, "model_type"),
        _field(config, "num_hidden_layers"),
        _field(config, "hidden_size"),
        _field(config, "intermediate_size"),
        _field(config, "vocab_size"),
        _field(config, "num_attention_heads"),
        _field(config, "num_key_value_heads"),
        _field(config, "head_dim"),
        _field(config, "linear_num_key_heads"),
        _field(config, "linear_num_value_heads"),
        _field(config, "linear_key_head_dim"),
        _field(config, "linear_value_head_dim"),
        _field(config, "linear_conv_kernel_dim"),
        _field(config, "full_attention_interval"),
        _field(config, "attn_output_gate"),
        _field(config, "tie_word_embeddings"),
    )


def _rope_identity(config: object, *, inner: bool) -> tuple[object, ...]:
    text = config if inner else _field(config, "text_config")
    rope = _field(text, "rope_parameters") or _field(text, "rope_scaling") or {}
    return (
        _field(text, "partial_rotary_factor", _field(rope, "partial_rotary_factor")),
        _field(rope, "rope_theta", _field(text, "rope_theta")),
        tuple(_field(rope, "mrope_section", ()) or ()),
        _field(rope, "mrope_interleaved", False),
    )


def _quantization_identity(ctx: ModelingV2Context) -> tuple[object, ...]:
    layer_configs = ctx.quant_config_dict or {}
    algorithms = Counter(
        _algorithm_name(_field(layer_config, "quant_algo"))
        for layer_config in layer_configs.values()
    )
    nvfp4_group_sizes = {
        _field(layer_config, "group_size")
        for layer_config in layer_configs.values()
        if _algorithm_name(_field(layer_config, "quant_algo")) == "NVFP4"
    }
    return (
        _algorithm_name(_field(ctx.quant_config, "quant_algo")),
        len(layer_configs),
        algorithms["FP8"],
        algorithms["NVFP4"],
        tuple(sorted(nvfp4_group_sizes)),
        frozenset(layer_configs) == _EXPECTED_QUANTIZED_LAYERS,
    )


def _parallel(mapping: object) -> Optional[str]:
    if (
        mapping.world_size == 1
        and mapping.tp_size == 1
        and mapping.pp_size == 1
        and mapping.cp_size == 1
        and not mapping.enable_attention_dp
    ):
        return "tp1"
    return None


def route(ctx: ModelingV2Context, trace: Trace = NULL_TRACE) -> Optional[str]:
    config = ctx.pretrained_config

    if not trace.check("sm", ctx.sm, ctx.sm == _SM):
        return None

    architecture = tuple(_field(config, "architectures", ()) or ())
    inner = architecture[:1] == ("Qwen3_5ForCausalLM",)
    shape = _inner_shape(config) if inner else _shape(config)
    checkpoints = _INNER_CHECKPOINTS if inner else _OUTER_CHECKPOINTS
    checkpoint = trace.resolve("shape", shape, checkpoints.get(shape))
    if checkpoint is None:
        return None

    layer_config = config if inner else _field(config, "text_config")
    layer_types = tuple(_field(layer_config, "layer_types", ()) or ())
    if not trace.check("layers", f"{len(layer_types)} layers", layer_types == _LAYER_TYPES):
        return None

    rope = _rope_identity(config, inner=inner)
    expected_rope = (0.25, 10_000_000, (11, 11, 10), True)
    if not trace.check("rope", rope, rope == expected_rope):
        return None

    quantization = _quantization_identity(ctx)
    expected_quantization = ("MIXED_PRECISION", 401, 208, 193, (16,), True)
    if not trace.check("quant", quantization, quantization == expected_quantization):
        return None

    if not trace.check("text_only", ctx.disable_mm_encoder, ctx.disable_mm_encoder):
        return None

    parallel = trace.resolve("parallel", f"ws={ctx.mapping.world_size}", _parallel(ctx.mapping))
    if parallel is None:
        return None

    level = "inner" if inner else "outer"
    return _TARGETS.get((checkpoint, level, parallel))
