# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What ``modeling_v2_resolve`` actually does, driven by synthetic configs.

No checkpoint and no weights: routing reads config *shape*, the mapping and
the SM version, all of which can be stated directly. The SM version is
monkeypatched so these run on any device -- the point here is the decision
logic, not the kernels.

The three things worth proving:

* ``off`` changes nothing. This is the whole safety argument for putting the
  hook in ``_resolve_class`` at all.
* a matching configuration reaches a target class, and that class is
  *external* -- so it wins the registry slot rather than losing it to the
  built-in provider, which the lazy zoo may import afterwards.
* ``require`` raises on a near-miss and says which criterion missed.
"""

from __future__ import annotations

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2._router_index import (
    MODELING_V2_ENV,
    ModelingV2Mode,
    modeling_v2_resolve,
)
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm._torch.models.modeling_utils import (
    _is_builtin_model_class,
    get_registered_model_class,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

_SM103 = (10, 3)


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: _SM103)


def _gpt_oss_config(**overrides):
    """The shape gpt-oss-120b's own config.json declares."""
    fields = dict(
        architectures=["GptOssForCausalLM"],
        model_type="gpt_oss",
        num_hidden_layers=36,
        hidden_size=2880,
        num_local_experts=128,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _r1_config(**overrides):
    """The shape DeepSeek-R1-0528-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["DeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        num_hidden_layers=61,
        hidden_size=7168,
        n_routed_experts=256,
        q_lora_rank=1536,
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


_QWEN_LAYER_TYPES = [
    layer_type
    for _ in range(16)
    for layer_type in ("linear_attention", "linear_attention", "linear_attention", "full_attention")
]


def _qwen_config(*, architecture="QwenImageBenchForConditionalGeneration", **text_overrides):
    """The published Qwen3.8-27B dense nested configuration."""
    text_fields = dict(
        model_type="qwen3_5_text",
        num_hidden_layers=64,
        hidden_size=5120,
        intermediate_size=17408,
        vocab_size=248320,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        attn_output_gate=True,
        tie_word_embeddings=False,
        partial_rotary_factor=0.25,
        rope_parameters=dict(
            rope_theta=10_000_000,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        ),
        layer_types=list(_QWEN_LAYER_TYPES),
    )
    text_fields.update(text_overrides)
    return PretrainedConfig(
        architectures=[architecture],
        model_type="qwen3_5",
        language_model_only=False,
        tie_word_embeddings=False,
        text_config=PretrainedConfig(**text_fields),
    )


def _qwen_quant_config_dict():
    configs = {}
    for layer_idx, layer_type in enumerate(_QWEN_LAYER_TYPES):
        prefix = f"model.language_model.layers.{layer_idx}"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            configs[f"{prefix}.mlp.{projection}"] = QuantConfig(
                quant_algo=QuantAlgo.NVFP4, group_size=16
            )
        projections = (
            ("in_proj_qkv", "in_proj_z", "out_proj")
            if layer_type == "linear_attention"
            else ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        module = "linear_attn" if layer_type == "linear_attention" else "self_attn"
        for projection in projections:
            configs[f"{prefix}.{module}.{projection}"] = QuantConfig(quant_algo=QuantAlgo.FP8)
    configs["lm_head"] = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    return configs


def _qwen_model_config(
    pretrained_config=None,
    *,
    disable_mm_encoder=True,
    quant_config_dict=None,
    **mapping_kwargs,
):
    return ModelConfig(
        pretrained_config=pretrained_config or _qwen_config(),
        mapping=Mapping(**mapping_kwargs) if mapping_kwargs else Mapping(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict=(
            _qwen_quant_config_dict() if quant_config_dict is None else quant_config_dict
        ),
        disable_mm_encoder=disable_mm_encoder,
    )


def _qwen_inner_model_config():
    outer = _qwen_config()
    inner = outer.text_config
    inner.architectures = ["Qwen3_5ForCausalLM"]
    return ModelConfig(
        pretrained_config=inner,
        mapping=Mapping(),
        quant_config=QuantConfig(quant_algo=QuantAlgo.MIXED_PRECISION),
        quant_config_dict=_qwen_quant_config_dict(),
        disable_mm_encoder=True,
    )


@pytest.fixture(autouse=True)
def _mode(monkeypatch, request):
    """Default every case to 'auto'; a case that wants another mode sets it.

    The switch is an environment variable, so the tests set one too -- that is
    the surface under test.
    """
    monkeypatch.setenv(MODELING_V2_ENV, "auto")


def _set_mode(monkeypatch, mode):
    if mode is None:
        monkeypatch.delenv(MODELING_V2_ENV, raising=False)
    else:
        monkeypatch.setenv(MODELING_V2_ENV, mode)


def _model_config(pretrained_config, **mapping_kwargs):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(pretrained_config=pretrained_config, mapping=mapping)


_DEP4 = dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1, enable_attention_dp=True)


@pytest.mark.parametrize("unset", [True, False], ids=["env-unset", "env-off"])
def test_off_resolves_nothing(monkeypatch, unset):
    """The default must be indistinguishable from modeling_v2 not existing.

    Unset and an explicit "off" have to behave identically: the common case is
    that nobody has heard of this package.
    """
    _set_mode(monkeypatch, None if unset else "off")
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config) is None


def test_off_still_reaches_the_builtin_implementation(monkeypatch):
    _set_mode(monkeypatch, "off")
    config = _model_config(_gpt_oss_config())
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved is not None
    assert resolved.__module__ == "tensorrt_llm._torch.models.modeling_gpt_oss"


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_gpt_oss_tp1_matches(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    config = _model_config(_gpt_oss_config())
    assert modeling_v2_resolve(config) == "ModelingV2GptOss120bSm103Tp1"


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_r1_dep4_matches(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    config = _model_config(_r1_config(), **_DEP4)
    assert modeling_v2_resolve(config) == "ModelingV2DeepseekR10528Nvfp4Sm103Dep4"


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_qwen3_8_27b_nvfp4_tp1_matches(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    assert modeling_v2_resolve(_qwen_model_config()) == "ModelingV2Qwen3827BNvfp4Sm103Tp1"


def test_qwen3_8_resolving_registers_external_target():
    name = modeling_v2_resolve(_qwen_model_config())
    cls = get_registered_model_class(name)
    assert cls is not None
    assert cls.__name__ == name
    assert cls.__module__.endswith(
        "modeling_v2.models.qwen3_5.targets.qwen3_8_27b_nvfp4.sm_103.tp1.modeling"
    )
    assert not _is_builtin_model_class(cls)


def test_qwen3_8_resolve_class_rewrites_architecture_end_to_end():
    resolved = AutoModelForCausalLM._resolve_class(_qwen_model_config())
    assert resolved.__name__ == "ModelingV2Qwen3827BNvfp4Sm103Tp1"


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_qwen3_8_delegated_inner_decoder_has_exact_target(monkeypatch, mode):
    _set_mode(monkeypatch, mode)
    name = modeling_v2_resolve(_qwen_inner_model_config())
    assert name == "ModelingV2Qwen3827BNvfp4TextSm103Tp1"
    cls = get_registered_model_class(name)
    assert cls is not None
    assert cls.__name__ == name
    assert not _is_builtin_model_class(cls)


@pytest.mark.parametrize(
    ("near_miss", "criterion"),
    [
        ("shape", "shape"),
        ("layers", "layers"),
        ("quant", "quant"),
        ("text_only", "text_only"),
        ("parallel", "parallel"),
    ],
)
def test_qwen3_8_near_misses_do_not_match(monkeypatch, near_miss, criterion):
    if near_miss == "shape":
        config = _qwen_model_config(_qwen_config(hidden_size=4096))
    elif near_miss == "layers":
        layer_types = list(_QWEN_LAYER_TYPES)
        layer_types[0] = "full_attention"
        config = _qwen_model_config(_qwen_config(layer_types=layer_types))
    elif near_miss == "quant":
        quant_config_dict = _qwen_quant_config_dict()
        quant_config_dict.pop("lm_head")
        config = _qwen_model_config(quant_config_dict=quant_config_dict)
    elif near_miss == "text_only":
        config = _qwen_model_config(disable_mm_encoder=False)
    else:
        config = _qwen_model_config(world_size=2, tp_size=2)

    assert modeling_v2_resolve(config) is None
    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match=criterion):
        modeling_v2_resolve(config)


def test_resolving_registers_the_target_class():
    """The synthetic name is a key; the import behind it is what fills it."""
    config = _model_config(_gpt_oss_config())
    name = modeling_v2_resolve(config)
    cls = get_registered_model_class(name)
    assert cls is not None, f"{name} resolved to no class"
    assert cls.__name__ == name
    assert cls.__module__.endswith(
        "modeling_v2.models.gpt_oss.targets.gpt_oss_120b.sm_103.tp1.modeling"
    )


def test_the_target_registration_counts_as_external():
    """External registrations always win their slot; built-ins only fill
    empty ones. Living beside the zoo rather than inside it is what buys
    this, and a move into _torch/models/ would silently reverse it."""
    config = _model_config(_gpt_oss_config())
    cls = get_registered_model_class(modeling_v2_resolve(config))
    assert not _is_builtin_model_class(cls)


def test_resolve_class_rewrites_the_architecture_end_to_end():
    config = _model_config(_gpt_oss_config())
    resolved = AutoModelForCausalLM._resolve_class(config)
    assert resolved.__name__ == "ModelingV2GptOss120bSm103Tp1"


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, missed",
    [
        # a GptOss checkpoint of another size
        (dict(num_hidden_layers=24, num_local_experts=32), {}, "shape"),
        # the right checkpoint, a topology no target implements
        (dict(), dict(world_size=2, tp_size=2), "parallel"),
    ],
)
def test_gpt_oss_near_misses_do_not_match(monkeypatch, config_kwargs, mapping_kwargs, missed):
    config = _model_config(_gpt_oss_config(**config_kwargs), **mapping_kwargs)
    assert modeling_v2_resolve(config) is None

    _set_mode(monkeypatch, "require")
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config)


def test_r1_without_attention_dp_does_not_match():
    """dep4 and tep4 differ in where the attention weights are split, so
    attention DP is an identity criterion rather than a knob."""
    mapping_kwargs = dict(_DEP4, enable_attention_dp=False)
    config = _model_config(_r1_config(), **mapping_kwargs)
    assert modeling_v2_resolve(config) is None


def test_require_names_the_criterion_that_missed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    _set_mode(monkeypatch, "require")
    config = _model_config(_gpt_oss_config())
    with pytest.raises(ValueError) as excinfo:
        modeling_v2_resolve(config)
    message = str(excinfo.value)
    assert "sm" in message and "(10, 0)" in message
    assert "no match" in message


def test_an_unrouted_architecture_is_not_an_error_under_auto():
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    assert modeling_v2_resolve(config) is None


def test_an_unrouted_architecture_raises_under_require(monkeypatch):
    _set_mode(monkeypatch, "require")
    config = _model_config(PretrainedConfig(architectures=["LlamaForCausalLM"]))
    with pytest.raises(ValueError, match="LlamaForCausalLM"):
        modeling_v2_resolve(config)


@pytest.mark.parametrize(
    "raw,expected",
    [(None, "off"), ("off", "off"), ("AUTO", "auto"), (" require ", "require"), ("", "off")],
)
def test_the_env_var_is_read_leniently(monkeypatch, raw, expected):
    """``None`` is the unset case, and it is the one that must never drift.

    Everything in the accuracy suite rests on modeling_v2 being opt-in: unset has
    to read as off on the code path the engine actually takes.
    """
    if raw is None:
        monkeypatch.delenv(MODELING_V2_ENV, raising=False)
    else:
        monkeypatch.setenv(MODELING_V2_ENV, raw)
    assert ModelingV2Mode.from_env().value == expected


def test_an_unknown_mode_raises_rather_than_falling_back(monkeypatch):
    """A typo must not read as "off".

    That would hand back the built-in implementation while the caller believed
    they had asked for a target -- the exact mis-attribution the require mode
    exists to prevent.
    """
    # "yes" rather than a misspelling: it is what someone reaching for a
    # boolean would write, and it is the reading that must not be invented.
    monkeypatch.setenv(MODELING_V2_ENV, "yes")
    with pytest.raises(ValueError, match="not a modeling_v2 mode"):
        ModelingV2Mode.from_env()
