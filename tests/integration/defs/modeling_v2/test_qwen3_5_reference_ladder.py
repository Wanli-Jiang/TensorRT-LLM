# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The reference ladder every Qwen3.8-27B-NVFP4 parity claim is judged against.

Two rungs, cheapest first:

``test_source_reference_matches_hf_modules``
    A four-layer random-weight model drives stock ``transformers`` ``Qwen3_5``
    modules and :mod:`_qwen3_5_source_reference` from the same inputs and the
    same weights. Small shapes, but every property that can be silently
    transposed is preserved -- per-head query/output-gate interleaving, partial
    rotary over a subset of the head, unequal interleaved mRoPE sections,
    grouped-query attention, and the 3x value-to-key head expansion in the
    Gated DeltaNet. Seconds, and it catches the layout mistakes that would
    otherwise cost a 27 B load per attempt.

``test_hf_reference_ladder_real_checkpoint``
    The real checkpoint: content-hash the published files, assert the declared
    manifest inventory, build native ``Qwen3_5ForCausalLM``, replay HF's own
    hooked hidden states through the source equations at representative Gated
    DeltaNet and full-attention layers in prefill and in a cached decode step,
    check the head, emit the golden token fixture with stock ``generate()``,
    re-hash, and then re-read every artifact from disk and check it means what
    the report says it means.

The order matters. Later criteria (`source_activation_replay`,
`source_logit_replay`, `generation_parity`) compare the target against this
tier, so a bug here would be inherited by every one of them and would show up
as agreement.

Both tests require CUDA and fail rather than skip: a skipped GPU test reads as
green, and a CPU run of this ladder proves nothing about sm_103.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

from . import _qwen3_5_source_reference as ref
from ._qwen3_5_checkpoint import (
    EXPECTED_CHECKPOINT_FILE_COUNT,
    CheckpointReader,
    assert_manifest_inventory,
    checkpoint_digest,
    checkpoint_path,
    compare_digests,
)
from ._qwen3_5_evidence import (
    artifact_digests,
    artifacts_dir,
    evidence_dir,
    json_block,
    provenance,
    run_dir,
    write_report,
)
from ._qwen3_5_hf_reference import (
    REFERENCE_LAYERS,
    REFERENCE_PROMPTS,
    ActivationCapture,
    GenerationRecord,
    LoadReport,
    build_hf_text_model,
    capture_layers,
    greedy_generate,
    load_generation_fixture,
    load_tokenizer,
    module_parameters,
    save_generation_fixture,
)

if TYPE_CHECKING:
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

#: Hooked-layer bars the plan fixes for every replay tier.
COSINE_BAR = 0.99
MEAN_ABS_BAR = 0.10

#: Module-vs-module bars for the small random config. Same math in a different
#: order, so the residual is bf16 rounding -- and one bf16 ULP at magnitude 12
#: is 0.0625, which is why the bar is relative with an absolute floor for
#: near-zero tensors rather than absolute alone.
ATOL = 2e-2
RTOL = 1e-2

#: Golden fixture length. Long enough that recurrent drift shows up, short
#: enough to stay a few minutes.
GENERATED_TOKENS = 32

#: The artifacts this tier must leave behind. Later tiers assert against them,
#: so each is re-read from disk and checked before the report cites its hash.
ARTIFACT_ACTIVATIONS = "hf_activations.pt"
ARTIFACT_FIXTURE = "hf_generation_fixture.json"
ARTIFACT_STEP_LOGITS = "hf_step_logits.pt"
ARTIFACT_CHECKPOINT_HASHES = "checkpoint_hashes.json"
REQUIRED_ARTIFACTS = (
    ARTIFACT_ACTIVATIONS,
    ARTIFACT_FIXTURE,
    ARTIFACT_STEP_LOGITS,
    ARTIFACT_CHECKPOINT_HASHES,
)

#: The hidden-state sites every hooked layer must have captured.
CAPTURE_SITES = ("layer_in", "mixer_in", "mixer_out", "layer_out")

#: Captured for every hooked layer, not only the full-attention ones:
#: ``Qwen3_5TextModel.forward`` computes one ``position_embeddings`` pair and
#: hands it to all 64 decoder layers, and the linear-attention layers ignore it.
ROTARY_SITES = ("cos", "sin")

#: The task pins sm_103 (GB300) at tp=pp=world_size=1 with one visible device.
#: Asserted rather than merely reported: a green run on a B200, or on a node
#: where four devices are visible and a later tier silently spreads across them,
#: would otherwise be indistinguishable from the run the criteria ask for.
EXPECTED_CAPABILITY = (10, 3)
EXPECTED_VISIBLE_DEVICES = 1

#: A metric row, as :func:`_qwen3_5_source_reference.compare` returns it plus
#: the two fields :func:`_record` adds.
Metrics = dict[str, object]


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail(
            "the Qwen3.5 reference ladder is real-checkpoint GPU evidence; "
            "a CPU run is not evidence and must not be reported as one"
        )
    visible = torch.cuda.device_count()
    assert visible == EXPECTED_VISIBLE_DEVICES, (
        f"{visible} CUDA devices are visible, expected {EXPECTED_VISIBLE_DEVICES}: "
        f"this target is tp1 and must run with CUDA_VISIBLE_DEVICES pinned to one "
        f"GPU (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
    )
    capability = torch.cuda.get_device_capability(0)
    assert capability == EXPECTED_CAPABILITY, (
        f"device 0 is {torch.cuda.get_device_name(0)} with capability {capability}, "
        f"expected sm_{EXPECTED_CAPABILITY[0]}{EXPECTED_CAPABILITY[1]} -- this "
        f"evidence is only claimed for sm_103"
    )
    return torch.device("cuda")


@contextlib.contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    """Build HF modules in bf16 without leaking the default into other tests."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _record(
    results: list[Metrics],
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    passed: Callable[[Metrics], bool],
) -> Metrics:
    metrics: Metrics = dict(ref.compare(actual, expected))
    metrics["name"] = name
    metrics["pass"] = bool(metrics["finite"] and passed(metrics))
    results.append(metrics)
    print(
        f"  {name:<46} cos={metrics['cosine']:.6f} max_abs={metrics['max_abs']:.4e} "
        f"rel={metrics['rel_max_abs']:.4e} mean_abs={metrics['mean_abs']:.4e} "
        f"{'PASS' if metrics['pass'] else 'FAIL'}",
        flush=True,
    )
    return metrics


def _metrics_table(results: list[Metrics]) -> list[str]:
    lines = [
        "",
        "| comparison | cosine | max_abs | rel_max_abs | mean_abs | pass |",
        "|---|---|---|---|---|---|",
    ]
    for item in results:
        lines.append(
            f"| {item['name']} | {item['cosine']:.6f} | {item['max_abs']:.4e} | "
            f"{item['rel_max_abs']:.4e} | {item['mean_abs']:.4e} | "
            f"{'yes' if item['pass'] else 'NO'} |"
        )
    return lines


def _assert_all_passed(results: list[Metrics]) -> None:
    failures = [item["name"] for item in results if not item["pass"]]
    assert not failures, f"{len(failures)}/{len(results)} comparisons failed: {failures}"


# ----------------------------------------------------------------------
# Rung 1: source equations vs HF's own modules, small random config
# ----------------------------------------------------------------------


def _small_config() -> Qwen3_5TextConfig:
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        hidden_size=256,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=128,
        rms_norm_eps=1e-6,
        attn_output_gate=True,
        output_gate_type="swish",
        partial_rotary_factor=0.25,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )


def test_source_reference_matches_hf_modules(cuda_device: torch.device) -> None:
    """The pure-torch source equations reproduce stock HF module outputs."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5DecoderLayer,
        Qwen3_5RMSNorm,
        Qwen3_5RMSNormGated,
        Qwen3_5TextRotaryEmbedding,
    )

    started = time.time()
    torch.manual_seed(0)
    config = _small_config()
    results: list[Metrics] = []
    batch, seq = 2, 12

    def close(metrics: Metrics) -> bool:
        return metrics["max_abs"] <= ATOL or metrics["rel_max_abs"] <= RTOL

    with _default_dtype(torch.bfloat16), torch.no_grad():
        print("===== norms =====", flush=True)
        norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(cuda_device)
        norm.weight.normal_(0, 0.2)
        hidden = torch.randn(
            batch, seq, config.hidden_size, device=cuda_device, dtype=torch.bfloat16
        )
        _record(
            results,
            "rms_norm_delta",
            ref.rms_norm_delta(hidden, norm.weight, config.rms_norm_eps),
            norm(hidden),
            close,
        )

        gated = Qwen3_5RMSNormGated(config.linear_value_head_dim, eps=config.rms_norm_eps).to(
            cuda_device
        )
        gated.weight.normal_(1.0, 0.2)
        value = torch.randn(
            batch * seq, config.linear_value_head_dim, device=cuda_device, dtype=torch.bfloat16
        )
        gate = torch.randn_like(value)
        _record(
            results,
            "rms_norm_gated",
            ref.rms_norm_gated(value, gated.weight, gate, config.rms_norm_eps),
            gated(value, gate),
            close,
        )

        print("===== mRoPE (unequal T/H/W rows) =====", flush=True)
        rotary = Qwen3_5TextRotaryEmbedding(config).to(cuda_device)
        rope_args = (
            config.head_dim,
            config.rope_parameters["partial_rotary_factor"],
            config.rope_parameters["rope_theta"],
            config.rope_parameters["mrope_section"],
        )
        positions = torch.stack(
            [
                torch.arange(seq, device=cuda_device).expand(batch, seq),
                torch.arange(seq, device=cuda_device).flip(0).expand(batch, seq),
                (torch.arange(seq, device=cuda_device) * 2).expand(batch, seq),
            ]
        )
        hf_cos, hf_sin = rotary(hidden, positions)
        cos, sin = ref.rope_cos_sin(positions, *rope_args)
        _record(results, "rope cos (3 distinct rows)", cos, hf_cos, close)
        _record(results, "rope sin (3 distinct rows)", sin, hf_sin, close)

        # A flattened mRoPE would agree with the three-row form; the sections
        # are only load bearing if the two disagree.
        flat = torch.arange(seq, device=cuda_device).expand(batch, seq)
        flat_cos, _ = ref.rope_cos_sin(flat, *rope_args)
        assert not torch.allclose(flat_cos.float(), cos.float(), atol=1e-3), (
            "flat and three-row mRoPE positions produced the same cos: the "
            "interleaved sections are not routing, so this check is blind"
        )

        print("===== full attention layer =====", flush=True)
        layer_full = Qwen3_5DecoderLayer(config, 3).to(cuda_device)
        for param in layer_full.parameters():
            param.normal_(0, 0.1)
        params = {name: param.detach() for name, param in layer_full.named_parameters()}
        mask = (
            torch.full((seq, seq), float("-inf"), device=cuda_device, dtype=torch.bfloat16)
            .triu(1)[None, None]
            .expand(batch, 1, seq, seq)
        )
        flat_cos, flat_sin = rotary(hidden, flat)
        geometry = {
            "num_heads": config.num_attention_heads,
            "num_kv_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
        }

        mixer_in = ref.rms_norm_delta(hidden, params["input_layernorm.weight"], config.rms_norm_eps)
        hf_attn_out, _ = layer_full.self_attn(
            hidden_states=mixer_in,
            position_embeddings=(flat_cos, flat_sin),
            attention_mask=mask,
        )
        hf_layer_out = layer_full(
            hidden, position_embeddings=(flat_cos, flat_sin), attention_mask=mask
        )
        mixed, _ = ref.full_attention_layer(
            mixer_in, params, flat_cos, flat_sin, eps=config.rms_norm_eps, **geometry
        )
        _record(results, "full attention mixer", mixed, hf_attn_out, close)
        layer_out, _ = ref.decoder_layer(
            hidden,
            params,
            "full_attention",
            cos=flat_cos,
            sin=flat_sin,
            eps=config.rms_norm_eps,
            **geometry,
        )
        _record(results, "full attention decoder layer", layer_out, hf_layer_out, close)

        print("===== gated delta net layer =====", flush=True)
        layer_linear = Qwen3_5DecoderLayer(config, 0).to(cuda_device)
        for param in layer_linear.parameters():
            param.normal_(0, 0.1)
        layer_linear.linear_attn.A_log.uniform_(0.1, 2.0).log_()
        layer_linear.linear_attn.dt_bias.uniform_(-1.0, 1.0)
        params = {name: param.detach() for name, param in layer_linear.named_parameters()}
        gdn_geometry = {
            "num_k_heads": config.linear_num_key_heads,
            "num_v_heads": config.linear_num_value_heads,
            "head_k_dim": config.linear_key_head_dim,
            "head_v_dim": config.linear_value_head_dim,
        }

        mixer_in = ref.rms_norm_delta(hidden, params["input_layernorm.weight"], config.rms_norm_eps)
        hf_gdn_out = layer_linear.linear_attn(hidden_states=mixer_in)
        # A linear-attention layer ignores position_embeddings, but the decoder
        # layer's signature still requires them.
        hf_layer_out = layer_linear(hidden, position_embeddings=(flat_cos, flat_sin))
        mixed, conv_state, recurrent_state = ref.gated_delta_net_layer(
            mixer_in, params, eps=config.rms_norm_eps, **gdn_geometry
        )
        _record(results, "gated delta net mixer", mixed, hf_gdn_out, close)
        layer_out, _ = ref.decoder_layer(
            hidden, params, "linear_attention", eps=config.rms_norm_eps, **gdn_geometry
        )
        _record(results, "gated delta net decoder layer", layer_out, hf_layer_out, close)
        print(
            f"  conv state {tuple(conv_state.shape)}, "
            f"recurrent state {tuple(recurrent_state.shape)}",
            flush=True,
        )

        print("===== mlp =====", flush=True)
        mlp_params = {f"mlp.{k}": v.detach() for k, v in layer_full.mlp.named_parameters()}
        _record(
            results,
            "swiglu mlp",
            ref.swiglu_mlp(hidden, mlp_params),
            layer_full.mlp(hidden),
            close,
        )

    elapsed = time.time() - started
    meta = provenance(
        tier="source equations vs stock HF modules, 4-layer random config",
        attention_backend="n/a: HF eager reference tier",
        cache_manager="n/a: HF reference tier",
        cuda_graph=False,
        overlap_scheduler=False,
        resolved_target="n/a: no TensorRT-LLM target is constructed in this tier",
    )
    lines = (
        json_block("Provenance", meta)
        + [
            "",
            "## Configuration under test",
            "",
            "```json",
            json.dumps(
                {
                    "hidden_size": config.hidden_size,
                    "num_attention_heads": config.num_attention_heads,
                    "num_key_value_heads": config.num_key_value_heads,
                    "head_dim": config.head_dim,
                    "partial_rotary_factor": config.rope_parameters["partial_rotary_factor"],
                    "mrope_section": config.rope_parameters["mrope_section"],
                    "mrope_interleaved": config.rope_parameters["mrope_interleaved"],
                    "linear_num_key_heads": config.linear_num_key_heads,
                    "linear_num_value_heads": config.linear_num_value_heads,
                    "linear_key_head_dim": config.linear_key_head_dim,
                    "linear_value_head_dim": config.linear_value_head_dim,
                    "linear_conv_kernel_dim": config.linear_conv_kernel_dim,
                },
                indent=2,
            ),
            "```",
            "",
            "## Source equations vs HF modules",
            "",
            f"Bars: max abs err <= {ATOL} or rel max abs err <= {RTOL}, all metrics finite.",
        ]
        + _metrics_table(results)
        + [
            "",
            "Flat vs three-row mRoPE positions disagree, so the interleaved "
            "sections are demonstrably routing.",
        ]
    )
    report = write_report(
        evidence_dir() / "source_reference_selftest.md",
        "Stage 1 / Goal 1.1 -- source-equation reference vs stock HF modules",
        lines,
        elapsed,
    )
    print(f"\n===== {len(results)} comparisons, report {report} =====", flush=True)
    _assert_all_passed(results)


# ----------------------------------------------------------------------
# Rung 2: the real checkpoint
# ----------------------------------------------------------------------


def _hf_path_facts(model: torch.nn.Module) -> dict[str, str]:
    """Which implementation HF actually selected for each moving part.

    Recorded because a later reference run that silently picks a different
    recurrent or norm kernel is a different reference, and the metrics alone
    would not say so.
    """
    gdn = model.model.layers[0].linear_attn
    return {
        "attn_implementation": model.config._attn_implementation,
        "gdn_chunk_fn": getattr(
            gdn.chunk_gated_delta_rule, "__name__", type(gdn.chunk_gated_delta_rule).__name__
        ),
        "gdn_recurrent_fn": getattr(
            gdn.recurrent_gated_delta_rule,
            "__name__",
            type(gdn.recurrent_gated_delta_rule).__name__,
        ),
        "gdn_conv_fn": getattr(gdn.causal_conv1d_fn, "__name__", str(gdn.causal_conv1d_fn)),
        "gdn_conv_update_fn": getattr(
            gdn.causal_conv1d_update, "__name__", type(gdn.causal_conv1d_update).__name__
        ),
        "gdn_norm_class": type(gdn.norm).__name__,
        "attention_class": type(model.model.layers[3].self_attn).__name__,
        "norm_class": type(model.model.norm).__name__,
    }


def _replay_layer(
    model: torch.nn.Module,
    capture: ActivationCapture,
    layer_index: int,
    tag: str,
    text_config: Qwen3_5TextConfig,
    results: list[Metrics],
    state: dict[str, object],
    passed: Callable[[Metrics], bool],
) -> None:
    """Drive the source reference with HF's own hidden states for one layer."""
    prefix = f"{tag}.layer{layer_index}"
    kind = capture.meta[f"{prefix}.kind"]
    device = model.device
    params = module_parameters(model, layer_index)
    mixer_in = capture.tensors[f"{prefix}.mixer_in"].to(device, torch.bfloat16)
    mixer_out = capture.tensors[f"{prefix}.mixer_out"].to(device, torch.bfloat16)
    layer_in = capture.tensors[f"{prefix}.layer_in"].to(device, torch.bfloat16)
    layer_out = capture.tensors[f"{prefix}.layer_out"].to(device, torch.bfloat16)

    # Decode replays continue from the state the *prefill* replay produced, so
    # the cache the reference carries is its own rather than HF's -- a decode
    # that only matches when handed HF's cache has not been checked at all.
    carried = f"prefill.layer{layer_index}"

    if kind == "full_attention":
        cos = capture.tensors[f"{prefix}.cos"].to(device, torch.bfloat16)
        sin = capture.tensors[f"{prefix}.sin"].to(device, torch.bfloat16)
        past = state.get(f"{carried}.kv") if tag == "decode" else None
        geometry = {
            "num_heads": text_config.num_attention_heads,
            "num_kv_heads": text_config.num_key_value_heads,
            "head_dim": text_config.head_dim,
        }
        mixed, kv = ref.full_attention_layer(
            mixer_in,
            params,
            cos,
            sin,
            eps=text_config.rms_norm_eps,
            past_key_value=past,
            **geometry,
        )
        if tag == "prefill":
            state[f"{carried}.kv"] = kv
        full, _ = ref.decoder_layer(
            layer_in,
            params,
            "full_attention",
            cos=cos,
            sin=sin,
            eps=text_config.rms_norm_eps,
            past_key_value=past,
            **geometry,
        )
    else:
        conv_state = state.get(f"{carried}.conv") if tag == "decode" else None
        recurrent_state = state.get(f"{carried}.recurrent") if tag == "decode" else None
        geometry = {
            "num_k_heads": text_config.linear_num_key_heads,
            "num_v_heads": text_config.linear_num_value_heads,
            "head_k_dim": text_config.linear_key_head_dim,
            "head_v_dim": text_config.linear_value_head_dim,
        }
        mixed, new_conv, new_recurrent = ref.gated_delta_net_layer(
            mixer_in,
            params,
            eps=text_config.rms_norm_eps,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            **geometry,
        )
        if tag == "prefill":
            state[f"{carried}.conv"] = new_conv
            state[f"{carried}.recurrent"] = new_recurrent
        full, _ = ref.decoder_layer(
            layer_in,
            params,
            "linear_attention",
            eps=text_config.rms_norm_eps,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            **geometry,
        )

    _record(results, f"{tag} L{layer_index} {kind} mixer", mixed, mixer_out, passed)
    _record(results, f"{tag} L{layer_index} {kind} layer", full, layer_out, passed)


#: Characters that mean the reference text is damaged rather than merely odd.
_CORRUPTION = ("\x00", "�")


def _assert_fixture_usable(
    records: Sequence[GenerationRecord], step_logits: dict[str, torch.Tensor]
) -> None:
    """The golden fixture has to be usable by the tiers that will assert on it.

    `generation_parity` compares the target token-for-token against these ids
    for at least 32 steps. A reference that stopped early, emitted an empty
    continuation, or carries non-finite step logits would make that comparison
    pass or fail for reasons that have nothing to do with the target, so the
    defects are caught here, where they are still attributable.

    Run twice: once on the in-memory results and once on what was actually
    written to disk, because it is the written bytes a later tier will load.
    """
    assert len(records) >= 5, f"expected at least 5 fixed prompts, got {len(records)}"
    for record in records:
        assert len(record.generated_token_ids) == GENERATED_TOKENS, (
            f"{record.prompt!r}: generated {len(record.generated_token_ids)} tokens, "
            f"expected {GENERATED_TOKENS} -- a short reference silently shortens "
            f"every parity comparison built on it"
        )
        assert record.text.strip(), f"{record.prompt!r}: empty continuation"
        damaged = [marker for marker in _CORRUPTION if marker in record.text]
        assert not damaged, f"{record.prompt!r}: corrupted reference text {damaged}"

        logits = step_logits[record.prompt]
        assert logits.shape[0] == GENERATED_TOKENS, (
            f"{record.prompt!r}: {logits.shape[0]} step-logit rows for {GENERATED_TOKENS} tokens"
        )
        assert torch.isfinite(logits).all(), f"{record.prompt!r}: non-finite step logits"
        argmax = logits.argmax(dim=-1).tolist()
        assert argmax == record.generated_token_ids, (
            f"{record.prompt!r}: greedy argmax of the recorded step logits does not "
            f"reproduce the recorded token ids -- the two artifacts disagree, so one "
            f"of them is not what generate() actually did"
        )


def _verify_artifacts(
    artifacts: Path,
    capture: ActivationCapture,
    records: Sequence[GenerationRecord],
    text_config: Qwen3_5TextConfig,
) -> list[str]:
    """Re-read every artifact from disk and check it means what it claims.

    Writing a file and hashing it proves only that bytes landed. Later tiers
    will *load* these four and assert against them, so the failure worth
    catching here is the one where the load succeeds and the contents are
    subtly not what the report says: a capture missing the decode rows, a
    fixture whose step logits no longer agree with its token ids, a witness
    that hashed 18 files instead of 19. Each is re-read exactly the way its
    consumer will read it.
    """
    checks: list[str] = []
    hidden = text_config.hidden_size
    rotary = int(text_config.head_dim * text_config.partial_rotary_factor)
    prompt_length = len(capture.meta["prompt_token_ids"])

    reloaded = ActivationCapture.load(str(artifacts / ARTIFACT_ACTIVATIONS))
    expected_keys = {"prefill.logits", "decode.logits"}
    for tag, length in (("prefill", prompt_length), ("decode", 1)):
        for index in REFERENCE_LAYERS:
            prefix = f"{tag}.layer{index}"
            kind = reloaded.meta[f"{prefix}.kind"]
            assert kind == text_config.layer_types[index], (
                f"{prefix}: captured a {kind} layer but the config declares "
                f"{text_config.layer_types[index]} -- the hybrid pattern the replay "
                f"claims to cover is not the pattern it captured"
            )
            for site in CAPTURE_SITES + ROTARY_SITES:
                key = f"{prefix}.{site}"
                expected_keys.add(key)
                assert key in reloaded.tensors, f"{key} missing from the reloaded capture"
                tensor = reloaded.tensors[key]
                assert tensor.dtype is torch.float32, f"{key}: {tensor.dtype}, expected float32"
                assert torch.isfinite(tensor).all(), f"{key}: non-finite values"
                width = rotary if site in ROTARY_SITES else hidden
                assert tuple(tensor.shape) == (1, length, width), (
                    f"{key}: shape {tuple(tensor.shape)}, expected {(1, length, width)}"
                )
    for key in ("prefill.logits", "decode.logits"):
        logits = reloaded.tensors[key]
        assert tuple(logits.shape) == (text_config.vocab_size,), (
            f"{key}: shape {tuple(logits.shape)}, expected {(text_config.vocab_size,)}"
        )
        assert torch.isfinite(logits).all(), f"{key}: non-finite values"
    assert set(reloaded.tensors) == expected_keys, (
        f"reloaded capture has {sorted(set(reloaded.tensors) ^ expected_keys)} "
        f"that the replay contract does not account for"
    )
    for field in ("prompt", "prompt_token_ids", "prefill_argmax", "decode_argmax"):
        assert reloaded.meta[field] == capture.meta[field], f"meta {field} did not round-trip"
    checks.append(
        f"- `{ARTIFACT_ACTIVATIONS}`: {len(reloaded.tensors)} tensors reloaded; every "
        f"hidden-state site float32, finite, `(1, {prompt_length} | 1, {hidden})`, with "
        f"`cos`/`sin` `(1, *, {rotary})` on every hooked layer and final logits "
        f"`({text_config.vocab_size},)`; captured layer kinds match "
        f"`config.layer_types`; prompt/argmax metadata round-tripped."
    )

    fixture_meta, reloaded_records = load_generation_fixture(str(artifacts / ARTIFACT_FIXTURE))
    assert len(reloaded_records) == len(REFERENCE_PROMPTS), (
        f"fixture has {len(reloaded_records)} records for "
        f"{len(REFERENCE_PROMPTS)} reference prompts"
    )
    for original, reloaded_record in zip(records, reloaded_records, strict=True):
        assert reloaded_record.as_dict() == original.as_dict(), (
            f"{original.prompt!r}: the written fixture differs from what generate() returned"
        )
    assert fixture_meta["source"] == "stock transformers Qwen3_5ForCausalLM.generate()"

    step_logits = torch.load(
        str(artifacts / ARTIFACT_STEP_LOGITS), map_location="cpu", weights_only=True
    )
    assert set(step_logits) == set(REFERENCE_PROMPTS), (
        f"step logits cover {sorted(set(step_logits) ^ set(REFERENCE_PROMPTS))} "
        f"differently from the reference prompts"
    )
    for prompt, logits in step_logits.items():
        assert logits.dtype is torch.float32, f"{prompt!r}: step logits are {logits.dtype}"
        assert tuple(logits.shape) == (GENERATED_TOKENS, text_config.vocab_size), (
            f"{prompt!r}: step logits {tuple(logits.shape)}, expected "
            f"{(GENERATED_TOKENS, text_config.vocab_size)}"
        )
    # Cross-checks the two reloaded artifacts against each other: the written
    # step logits must still greedily reproduce the written token ids.
    _assert_fixture_usable(reloaded_records, step_logits)
    checks.append(
        f"- `{ARTIFACT_FIXTURE}` + `{ARTIFACT_STEP_LOGITS}`: {len(reloaded_records)} records "
        f"reloaded, each exactly {GENERATED_TOKENS} tokens with non-empty uncorrupted text "
        f"and byte-identical to what `generate()` returned; step logits float32, finite, "
        f"`({GENERATED_TOKENS}, {text_config.vocab_size})` per prompt, and their greedy "
        f"argmax reproduces the recorded token ids."
    )

    witness = json.loads((artifacts / ARTIFACT_CHECKPOINT_HASHES).read_text())
    for phase in ("before", "after"):
        assert len(witness[phase]) == EXPECTED_CHECKPOINT_FILE_COUNT, (
            f"{phase} witness hashed {len(witness[phase])} files, expected "
            f"{EXPECTED_CHECKPOINT_FILE_COUNT} -- a witness that covers fewer files is "
            f"not weaker by a little, it stops covering whichever file was dropped"
        )
    assert witness["before"] == witness["after"], "reloaded witness disagrees with itself"
    assert witness["changes"] == [], f"reloaded witness records changes: {witness['changes']}"
    checks.append(
        f"- `{ARTIFACT_CHECKPOINT_HASHES}`: {EXPECTED_CHECKPOINT_FILE_COUNT} before and "
        f"{EXPECTED_CHECKPOINT_FILE_COUNT} after hashes reloaded, identical, `changes: []`."
    )

    for line in checks:
        print(f"  {line[2:]}", flush=True)
    return checks


def test_hf_reference_ladder_real_checkpoint(cuda_device: torch.device) -> None:
    """Native HF on the published checkpoint, replayed by the source equations."""
    started = time.time()
    artifacts = artifacts_dir()

    print("===== checkpoint content hashes (before) =====", flush=True)
    hashed_at = time.time()
    digest_before = checkpoint_digest()
    print(
        f"  hashed {len(digest_before)} files in {time.time() - hashed_at:.1f}s",
        flush=True,
    )
    for name in sorted(digest_before):
        print(f"  {name:<40} {digest_before[name]}", flush=True)
    assert len(digest_before) == EXPECTED_CHECKPOINT_FILE_COUNT, (
        f"the read-only witness covers {len(digest_before)} files, expected "
        f"{EXPECTED_CHECKPOINT_FILE_COUNT}"
    )

    print("\n===== checkpoint manifest =====", flush=True)
    reader = CheckpointReader()
    counts = reader.namespace_counts()
    algo_counts = reader.quant_algo_counts()
    print(f"  namespaces        {counts}", flush=True)
    print(f"  declared algos    {algo_counts}", flush=True)
    print(f"  total tensors     {len(reader.weight_map)}", flush=True)
    # Asserted, not merely printed: the text total is what the load accounting
    # is checked against, vision/MTP are the text-only exclusion budget, and the
    # FP8/NVFP4 split is the MIXED_PRECISION intent the target must preserve.
    inventory = assert_manifest_inventory(reader)
    print(f"  inventory asserted {inventory['asserted']}", flush=True)

    print("\n===== building native HF text model (dequantized load) =====", flush=True)
    built_at = time.time()
    model, load_report, reader = build_hf_text_model(reader=reader)
    print(f"  built in {time.time() - built_at:.1f}s", flush=True)
    print(f"  parameters written {len(load_report.parameters_written)}", flush=True)
    print(f"  keys consumed      {len(load_report.keys_consumed)}", flush=True)
    print(f"  keys excluded      {load_report.keys_excluded}", flush=True)
    print(f"  dequant algos      {load_report.algo_counts}", flush=True)
    facts = _hf_path_facts(model)
    for key, value in facts.items():
        print(f"  {key:<22} {value}", flush=True)

    text_config = model.config
    tokenizer = load_tokenizer()

    print("\n===== prefill + decode capture with HF hooks =====", flush=True)
    meta = provenance(
        tier="native HF Qwen3_5ForCausalLM on the real checkpoint",
        attention_backend="n/a: HF eager reference tier",
        cache_manager="n/a: HF DynamicCache reference tier",
        cuda_graph=False,
        overlap_scheduler=False,
        resolved_target="n/a: no TensorRT-LLM target is constructed in this tier",
        checkpoint=checkpoint_path(),
    )
    capture = ActivationCapture()
    capture.meta.update({"provenance": meta, "hf_path": facts})
    prompt = REFERENCE_PROMPTS[0]
    capture.meta["prompt"] = prompt
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    capture.meta["prompt_token_ids"] = encoded["input_ids"][0].tolist()

    with torch.no_grad():
        with capture_layers(model, capture, REFERENCE_LAYERS, tag="prefill"):
            prefill = model(**encoded, use_cache=True)
        capture.add("prefill.logits", prefill.logits[0, -1])
        next_token = prefill.logits[0, -1].argmax().view(1, 1)
        capture.meta["prefill_argmax"] = int(next_token.item())
        with capture_layers(model, capture, REFERENCE_LAYERS, tag="decode"):
            decode = model(
                input_ids=next_token,
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
        capture.add("decode.logits", decode.logits[0, -1])
        capture.meta["decode_argmax"] = int(decode.logits[0, -1].argmax().item())
    print(
        f"  prompt              {prompt!r} ({len(capture.meta['prompt_token_ids'])} tokens)",
        flush=True,
    )
    print(
        f"  prefill argmax      {capture.meta['prefill_argmax']} "
        f"{tokenizer.decode([capture.meta['prefill_argmax']])!r}",
        flush=True,
    )
    print(
        f"  decode argmax       {capture.meta['decode_argmax']} "
        f"{tokenizer.decode([capture.meta['decode_argmax']])!r}",
        flush=True,
    )
    capture.save(str(artifacts / ARTIFACT_ACTIVATIONS))

    def passed(metrics: Metrics) -> bool:
        return metrics["cosine"] >= COSINE_BAR and metrics["mean_abs"] <= MEAN_ABS_BAR

    print("\n===== pure-PyTorch source reference vs HF hooks =====", flush=True)
    results: list[Metrics] = []
    state: dict[str, object] = {}
    for tag in ("prefill", "decode"):
        for layer_index in REFERENCE_LAYERS:
            _replay_layer(model, capture, layer_index, tag, text_config, results, state, passed)

    print("\n===== final norm + head =====", flush=True)
    with torch.no_grad():
        last_hidden = capture.tensors["prefill.layer63.layer_out"].to(model.device, torch.bfloat16)
        logits = ref.language_head(
            last_hidden,
            model.model.norm.weight.detach(),
            model.lm_head.weight.detach(),
            eps=text_config.rms_norm_eps,
        )[0, -1]
    _record(
        results,
        "prefill head logits",
        logits,
        capture.tensors["prefill.logits"].to(model.device),
        passed,
    )
    reference_argmax = int(logits.argmax().item())
    print(
        f"  reference argmax    {reference_argmax} (HF {capture.meta['prefill_argmax']})",
        flush=True,
    )
    argmax_match = reference_argmax == capture.meta["prefill_argmax"]

    print("\n===== native generate() golden fixture =====", flush=True)
    generated_at = time.time()
    step_logits: dict[str, torch.Tensor] = {}
    records = greedy_generate(
        model,
        tokenizer,
        prompts=REFERENCE_PROMPTS,
        max_new_tokens=GENERATED_TOKENS,
        logits_sink=step_logits.__setitem__,
    )
    print(f"  generated in {time.time() - generated_at:.1f}s", flush=True)
    for record in records:
        print(f"  {record.prompt!r}\n      -> {record.text!r}", flush=True)
    _assert_fixture_usable(records, step_logits)
    save_generation_fixture(
        records,
        str(artifacts / ARTIFACT_FIXTURE),
        {
            **meta,
            "hf_path": facts,
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": GENERATED_TOKENS,
                "min_new_tokens": GENERATED_TOKENS,
            },
            "source": "stock transformers Qwen3_5ForCausalLM.generate()",
        },
    )
    # fp32, matching what generate() produced: the reloaded artifact then has
    # the same argmax as the tensor this run asserted on, so the disk check is
    # an equality rather than a tolerance.
    torch.save(step_logits, str(artifacts / ARTIFACT_STEP_LOGITS))

    print("\n===== checkpoint content hashes (after) =====", flush=True)
    hashed_at = time.time()
    digest_after = checkpoint_digest()
    changes = compare_digests(digest_before, digest_after)
    print(
        f"  rehashed {len(digest_after)} files in {time.time() - hashed_at:.1f}s",
        flush=True,
    )
    print(f"  checkpoint unchanged: {not changes}", flush=True)
    for name, change in changes:
        print(f"  CHANGED {name}: {change}", flush=True)
    (artifacts / ARTIFACT_CHECKPOINT_HASHES).write_text(
        json.dumps(
            {"before": digest_before, "after": digest_after, "changes": changes},
            indent=2,
        )
    )

    print("\n===== artifact verification (re-read from disk) =====", flush=True)
    artifact_checks = _verify_artifacts(artifacts, capture, records, text_config)

    elapsed = time.time() - started
    report = _write_ladder_report(
        meta,
        facts,
        load_report,
        inventory,
        results,
        records,
        digest_before,
        changes,
        argmax_match,
        artifact_checks,
        elapsed,
    )
    print(f"\n===== {len(results)} comparisons, report {report} =====", flush=True)

    assert not changes, f"the run modified checkpoint-owned files: {changes}"
    assert argmax_match, (
        f"source reference greedy argmax {reference_argmax} != HF {capture.meta['prefill_argmax']}"
    )
    _assert_all_passed(results)


def _write_ladder_report(
    meta: dict[str, object],
    facts: dict[str, str],
    load_report: LoadReport,
    inventory: dict[str, object],
    results: list[Metrics],
    records: Sequence[GenerationRecord],
    digest: dict[str, str],
    changes: list[tuple[str, str]],
    argmax_match: bool,
    artifact_checks: list[str],
    elapsed: float,
) -> Path:
    lines = [
        "Native `transformers` Qwen3.5 on the real Qwen3.8-27B-NVFP4 checkpoint,",
        "stock `generate()` for the golden fixture, and the pure-PyTorch source",
        "reference checked against HF's own hooked activations.",
        "",
        f"Run directory: `{run_dir()}`",
    ]
    lines += json_block("Provenance", meta)
    lines += json_block("HF implementation actually selected", facts)
    lines += [
        "",
        "## Checkpoint",
        "",
        f"- namespaces: `{inventory['namespaces']}`",
        f"- declared quantization: `{inventory['quantization']}`",
        f"- total tensors: {inventory['total_tensors']}",
        f"- parameters written: {len(load_report.parameters_written)}",
        f"- active keys consumed: {len(load_report.keys_consumed)}",
        f"- excluded namespaces: `{load_report.keys_excluded}`",
        f"- dequantized per algorithm: `{load_report.algo_counts}`",
        "",
        "The manifest inventory is asserted against the published declaration, not",
        f"merely reported: `{inventory['asserted']}`. Any namespace or quantization",
        "drift fails the run before the model is built.",
        "",
        "### Read-only witness (BLAKE2b-256 over whole file contents)",
        "",
        f"Hashed before and after the run; changed files: `{changes}`.",
        f"Checkpoint files unchanged by this run: **{not changes}**",
        "",
        "| file | blake2b256 (before == after) |",
        "|---|---|",
    ]
    for name in sorted(digest):
        lines.append(f"| `{name}` | `{digest[name].split(':', 1)[1]}` |")
    lines += [
        "",
        "## Source reference vs HF hooks",
        "",
        f"Bars: cosine >= {COSINE_BAR}, mean abs err <= {MEAN_ABS_BAR}, all metrics finite.",
    ]
    lines += _metrics_table(results)
    lines += [
        "",
        f"Reference greedy argmax equals HF's: **{argmax_match}**",
        "",
        "## Golden generation fixture (stock `generate()`, greedy)",
        "",
    ]
    for record in records:
        lines.append(f"- `{record.prompt}`")
        lines.append(f"  - ids: `{record.generated_token_ids}`")
        lines.append(f"  - text: `{record.text}`")
    artifacts = artifacts_dir()
    digests = artifact_digests([artifacts / name for name in REQUIRED_ARTIFACTS])
    lines += [
        "",
        "## Artifacts",
        "",
        "Written under this run's own directory and hashed here, so the report "
        "and the bytes it describes cannot drift apart. Every one of them is "
        "re-read from disk and semantically checked before being cited:",
        "",
    ]
    lines += artifact_checks
    lines += [
        "",
        "| artifact | blake2b256 |",
        "|---|---|",
    ]
    for path, digest_value in digests.items():
        lines.append(f"| `{path}` | `{digest_value.split(':', 1)[1]}` |")
    return write_report(
        evidence_dir() / "hf_reference_ladder.md",
        "Stage 1 / Goal 1.1 -- HF reference ladder on the real checkpoint",
        lines,
        elapsed,
    )
