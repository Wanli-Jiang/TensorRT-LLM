# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused contracts for the delegated Qwen3.8-27B-NVFP4 target."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.modeling_v2._router_index import ModelingV2Context
from tensorrt_llm._torch.modeling_v2.explain import build_parser
from tensorrt_llm._torch.modeling_v2.models.qwen3_5.targets.qwen3_8_27b_nvfp4.sm_103.tp1 import (
    modeling,
    weights,
)
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5MoeHfWeightMapper
from tensorrt_llm._torch.models.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5VLModel
from tensorrt_llm._torch.pyexecutor import model_loader as model_loader_mod


def test_target_is_the_modeling_v2_identity_for_the_production_dense_model():
    assert issubclass(modeling.ModelingV2Qwen3827BNvfp4Sm103Tp1, Qwen3_5VLModel)
    assert issubclass(
        modeling.ModelingV2Qwen3827BNvfp4TextSm103Tp1,
        Qwen3_5ForCausalLM,
    )
    assert modeling.DELEGATED_MODEL_CLASSES == (Qwen3_5VLModel, Qwen3_5ForCausalLM)
    assert modeling.TARGET_CHECKPOINT == "qwen3_8_27b_nvfp4"
    assert modeling.TARGET_GPU_ARCH == "sm_103"
    assert modeling.TARGET_PARALLEL == "tp1"


def test_explain_can_replay_the_text_only_dimension():
    args = build_parser().parse_args(
        ["--model", "/checkpoint", "--sm", "10.3", "--disable-mm-encoder"]
    )
    assert args.disable_mm_encoder is True


def test_context_carries_route_critical_deployment_and_quantization_state():
    quant_config_dict = {"lm_head": object()}
    config = SimpleNamespace(
        pretrained_config=object(),
        mapping=object(),
        quant_config=object(),
        quant_config_dict=quant_config_dict,
        spec_config=None,
        disable_mm_encoder=True,
    )
    context = ModelingV2Context.from_model_config(config, sm=(10, 3))
    assert context.quant_config_dict is quant_config_dict
    assert context.disable_mm_encoder is True


def test_direct_target_construction_rejects_multimodal_deployment():
    config = SimpleNamespace(
        disable_mm_encoder=False,
        mapping=SimpleNamespace(
            world_size=1,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            enable_attention_dp=False,
        ),
    )
    with pytest.raises(ValueError, match="disable_mm_encoder=True"):
        modeling.ModelingV2Qwen3827BNvfp4Sm103Tp1.__init__(object(), config)


def test_target_load_weights_uses_the_target_weight_module():
    mapper = Qwen3_5MoeHfWeightMapper()
    target = object()
    tensor = object()
    with patch.object(weights, "load") as delegated_load:
        modeling.ModelingV2Qwen3827BNvfp4Sm103Tp1.load_weights(
            target, {"tensor": tensor}, mapper, allow_partial_loading=True
        )
    delegated_load.assert_called_once_with(target, {"tensor": tensor}, mapper, True)


def test_weight_module_reuses_the_production_loader_and_mapper():
    mapper = Qwen3_5MoeHfWeightMapper()
    target = object()
    checkpoint_weights = {"tensor": object()}
    with patch.object(Qwen3_5VLModel, "load_weights") as production_load:
        weights.load(target, checkpoint_weights, mapper, allow_partial_loading=True)
    production_load.assert_called_once_with(
        target,
        checkpoint_weights,
        mapper,
        allow_partial_loading=True,
    )


def test_weight_module_rejects_an_unrelated_mapper():
    with pytest.raises(TypeError, match="Qwen3_5MoeHfWeightMapper"):
        weights.load(object(), {}, object())


def test_provisional_model_config_preserves_text_only_routing(
    monkeypatch: pytest.MonkeyPatch,
):
    """The early model-class resolution must see the runtime text-only flag."""
    checkpoint_loader = MagicMock()
    config = MagicMock()
    config.pretrained_config.architectures = ["QwenImageBenchForConditionalGeneration"]
    checkpoint_loader.load_config.return_value = config

    class _ResolvedModel:
        @classmethod
        def get_model_defaults(cls, _llm_args):
            return {}

    monkeypatch.setattr(
        model_loader_mod.AutoModelForCausalLM,
        "_resolve_class",
        staticmethod(lambda _config: _ResolvedModel),
    )
    monkeypatch.setattr(
        model_loader_mod,
        "_resolve_transceiver_runtime_auto",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_loader_mod,
        "_resolve_kv_cache_manager_v2_auto",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_loader_mod,
        "_validate_and_adjust_mamba_snapshot_config",
        lambda *_args, **_kwargs: None,
    )

    llm_args = SimpleNamespace(
        trust_remote_code=False,
        mm_encoder_only=False,
        disable_mm_encoder=True,
        parallel_config=None,
        speculative_config=None,
        kv_cache_config=SimpleNamespace(use_kv_cache_manager_v2=False),
    )
    model_loader_mod.ModelLoader.load_config_and_apply_defaults(
        "/checkpoint", llm_args, checkpoint_loader
    )

    checkpoint_loader.load_config.assert_called_once_with(
        "/checkpoint",
        trust_remote_code=False,
        mm_encoder_only=False,
        disable_mm_encoder=True,
    )
