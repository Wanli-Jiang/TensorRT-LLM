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
"""Qwen3.5 source equations in plain PyTorch, independent of ``transformers``.

The HF modules are the semantic source of truth, but they are not a usable
*probe*: they carry their own cache objects, mask construction and module
wiring, so a target-versus-HF mismatch localizes to a whole layer at best. This
module restates the same equations -- read out of
``transformers/models/qwen3_5/modeling_qwen3_5.py`` and rewritten here -- as
free functions over explicit tensors, so a single transform can be driven with
whatever input a failure suggests.

That independence is the point: this tier is worth something only because it
was written from the source equations rather than by calling them. It is
checked against stock HF modules and then against hooked HF activations on the
real checkpoint before it is used to judge anything
(``test_qwen3_5_reference_ladder.py``), which is what keeps it from quietly
reproducing the target's bug.

Precision follows the source rather than being uniformly upcast: projections in
bf16 as ``nn.Linear`` runs them, and normalization, softmax, the decay gate and
the delta-rule recurrence in fp32 where the source forces fp32. A reference
that upcast everything would disagree with HF by more than the target does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------


def rms_norm_delta(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``Qwen3_5RMSNorm``: RMS normalize in fp32, scale by ``1 + weight``.

    The stored weight is a *delta*: a zero row means unit scale. Dropping the
    ``+1`` leaves a model that still produces finite plausible-looking noise,
    which is why every norm-bearing comparison here goes through this one
    function.
    """
    dtype = x.dtype
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (normed * (1.0 + weight.float())).to(dtype)


def rms_norm_gated(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """``Qwen3_5RMSNormGated``: the Gated DeltaNet output norm.

    Deliberately *not* ``rms_norm_delta``: this norm multiplies by ``weight``
    itself, with no ``+1``, then by ``silu(gate)`` computed in fp32. The two
    norms differing is a property of the source, and using either for the other
    is a plausible, finite, wrong answer.
    """
    dtype = x.dtype
    x32 = x.float()
    normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    scaled = weight * normed.to(dtype)
    return (scaled * F.silu(gate.float())).to(dtype)


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 normalization: ``eps`` is added to the sum of squares."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


# ----------------------------------------------------------------------
# Positional encoding
# ----------------------------------------------------------------------


def rope_inv_freq(
    head_dim: int,
    partial_rotary_factor: float,
    theta: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    rotary_dim = int(head_dim * partial_rotary_factor)
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(
        device=device, dtype=torch.float32
    )
    return 1.0 / (theta ** (exponents / rotary_dim))


def interleave_mrope(freqs: torch.Tensor, mrope_section: Sequence[int]) -> torch.Tensor:
    """Fold the (T, H, W) frequency rows into one interleaved row.

    ``freqs`` is ``(3, batch, seq, rotary_dim // 2)``. T keeps positions
    ``0, 3, 6, ...``, H takes ``1, 4, ...`` and W takes ``2, 5, ...``, so the
    sections are interleaved rather than concatenated. For text the three rows
    are equal and this is numerically an identity -- which is exactly why a
    flattened implementation passes every text test and fails the moment the
    rows differ.
    """
    out = freqs[0].clone()
    for axis, offset in enumerate((1, 2), start=1):
        length = mrope_section[axis] * 3
        index = slice(offset, length, 3)
        out[..., index] = freqs[axis, ..., index]
    return out


def rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    partial_rotary_factor: float,
    theta: float,
    mrope_section: Sequence[int],
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ``(cos, sin)`` of shape ``(batch, seq, rotary_dim)``.

    ``position_ids`` is ``(3, batch, seq)`` -- the (T, H, W) rows -- or
    ``(batch, seq)``, which is broadcast to three equal rows the way the source
    does for text.
    """
    if position_ids.ndim == 2:
        position_ids = position_ids[None].expand(3, *position_ids.shape)
    device = position_ids.device
    inv_freq = rope_inv_freq(head_dim, partial_rotary_factor, theta, device=device)
    inv_freq_expanded = inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
    positions = position_ids[:, :, None, :].float()
    freqs = (inv_freq_expanded @ positions).transpose(2, 3)
    freqs = interleave_mrope(freqs, mrope_section)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_partial_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the leading ``rotary_dim`` channels and pass the rest through.

    ``rotary_dim`` is read off ``cos`` (64 of 256 here), so a full-width
    rotation cannot be produced by accident.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_out = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_out = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_out, k_out


# ----------------------------------------------------------------------
# Full attention
# ----------------------------------------------------------------------


def split_query_and_gate(
    projected: torch.Tensor, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``q_proj``'s output into query and output gate, per head.

    The projection emits ``head_dim * 2`` per head and the source views it as
    ``(..., num_heads, head_dim * 2)`` before chunking, so query and gate
    alternate *within* each head. Splitting the flat 12288 rows in half instead
    gives a well-shaped tensor of the wrong 24 heads, which is why this is its
    own function with its own sentinel test.
    """
    shaped = projected.view(*projected.shape[:-1], num_heads, head_dim * 2)
    query, gate = torch.chunk(shaped, 2, dim=-1)
    return query, gate.reshape(*projected.shape[:-1], num_heads * head_dim)


def repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden
    batch, heads, seq, dim = hidden.shape
    expanded = hidden[:, :, None, :, :].expand(batch, heads, repeats, seq, dim)
    return expanded.reshape(batch, heads * repeats, seq, dim)


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager GQA with an fp32 softmax, matching ``eager_attention_forward``."""
    groups = query.shape[1] // key.shape[1]
    key = repeat_kv(key, groups)
    value = repeat_kv(value, groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if mask is None:
        q_len, k_len = query.shape[-2], key.shape[-2]
        offset = k_len - q_len
        causal = (
            torch.arange(q_len, device=query.device)[:, None] + offset
            < torch.arange(k_len, device=query.device)[None, :]
        )
        scores = scores.masked_fill(causal, float("-inf"))
    else:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


def full_attention_layer(
    hidden: torch.Tensor,
    params: dict[str, torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_heads: int = 24,
    num_kv_heads: int = 4,
    head_dim: int = 256,
    eps: float = 1e-6,
    past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """One full-attention mixer: projections, norms, RoPE, GQA, gate, output.

    ``params`` holds the layer's dequantized bf16 weights under their HF names
    (``self_attn.q_proj.weight`` and friends). Returns the mixer output and the
    updated ``(key, value)``, so a decode step can be driven from a prefill's
    own cache rather than a synthesized one.
    """
    batch, seq, _ = hidden.shape
    query, gate = split_query_and_gate(
        F.linear(hidden, params["self_attn.q_proj.weight"]), num_heads, head_dim
    )
    key = F.linear(hidden, params["self_attn.k_proj.weight"]).view(
        batch, seq, num_kv_heads, head_dim
    )
    value = F.linear(hidden, params["self_attn.v_proj.weight"]).view(
        batch, seq, num_kv_heads, head_dim
    )

    query = rms_norm_delta(query, params["self_attn.q_norm.weight"], eps).transpose(1, 2)
    key = rms_norm_delta(key, params["self_attn.k_norm.weight"], eps).transpose(1, 2)
    value = value.transpose(1, 2)

    query, key = apply_partial_rope(query, key, cos, sin)

    if past_key_value is not None:
        key = torch.cat([past_key_value[0], key], dim=2)
        value = torch.cat([past_key_value[1], value], dim=2)

    attn = causal_attention(query, key, value, scaling=head_dim**-0.5)
    attn = attn.reshape(batch, seq, num_heads * head_dim)
    attn = attn * torch.sigmoid(gate)
    return F.linear(attn, params["self_attn.o_proj.weight"]), (key, value)


# ----------------------------------------------------------------------
# Gated DeltaNet
# ----------------------------------------------------------------------


def gated_delta_rule_recurrent(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The delta rule, one token at a time, in fp32.

    ``query``/``key`` are ``(batch, seq, heads, k_dim)`` already L2-normalized,
    ``value`` is ``(batch, seq, heads, v_dim)``, ``g`` and ``beta`` are
    ``(batch, seq, heads)``. Returns ``(output, final_state)`` with state
    ``(batch, heads, k_dim, v_dim)``.

    Written as the plain recurrence rather than the chunked form the source
    runs: the chunked kernel and this agreeing is a real check, and this is the
    version whose every term maps to one line of the update rule.
    """
    query, key, value, g, beta = (
        t.transpose(1, 2).contiguous().float() for t in (query, key, value, g, beta)
    )
    batch, heads, seq, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (1.0 / math.sqrt(k_dim))

    state = (
        torch.zeros(batch, heads, k_dim, v_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.float().clone()
    )
    out = torch.zeros(batch, heads, seq, v_dim, device=value.device, dtype=value.dtype)
    for step in range(seq):
        q_t = query[:, :, step]
        k_t = key[:, :, step]
        v_t = value[:, :, step]
        decay = g[:, :, step].exp()[..., None, None]
        beta_t = beta[:, :, step][..., None]

        state = state * decay
        recalled = (state * k_t[..., None]).sum(dim=-2)
        delta = (v_t - recalled) * beta_t
        state = state + k_t[..., None] * delta[..., None, :]
        out[:, :, step] = (state * q_t[..., None]).sum(dim=-2)
    return out.transpose(1, 2).contiguous(), state


def causal_conv1d_reference(
    x: torch.Tensor, weight: torch.Tensor, state: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution with SiLU, plus the window it leaves.

    ``x`` is ``(batch, channels, seq)`` and ``weight`` ``(channels, kernel)``.
    The returned state is the trailing ``kernel`` columns of the padded input,
    which is the window layout the source's decode update expects.
    """
    kernel = weight.shape[-1]
    channels = x.shape[1]
    if state is None:
        padded = F.pad(x, (kernel - 1, 0))
    else:
        padded = torch.cat([state[..., -(kernel - 1) :], x], dim=-1)
    out = F.conv1d(padded, weight.unsqueeze(1), None, groups=channels)
    new_state = F.pad(padded, (kernel - padded.shape[-1], 0))
    return F.silu(out), new_state


def gated_delta_net_layer(
    hidden: torch.Tensor,
    params: dict[str, torch.Tensor],
    num_k_heads: int = 16,
    num_v_heads: int = 48,
    head_k_dim: int = 128,
    head_v_dim: int = 128,
    eps: float = 1e-6,
    conv_state: torch.Tensor | None = None,
    recurrent_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One Gated DeltaNet mixer, returning output plus both cache states."""
    batch, seq, _ = hidden.shape
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim

    mixed = F.linear(hidden, params["linear_attn.in_proj_qkv.weight"]).transpose(1, 2)
    z = F.linear(hidden, params["linear_attn.in_proj_z.weight"]).reshape(
        batch, seq, num_v_heads, head_v_dim
    )
    b = F.linear(hidden, params["linear_attn.in_proj_b.weight"])
    a = F.linear(hidden, params["linear_attn.in_proj_a.weight"])

    mixed, new_conv_state = causal_conv1d_reference(
        mixed, params["linear_attn.conv1d.weight"].squeeze(1), conv_state
    )
    mixed = mixed.transpose(1, 2)
    query, key, value = torch.split(mixed, [key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch, seq, num_k_heads, head_k_dim)
    key = key.reshape(batch, seq, num_k_heads, head_k_dim)
    value = value.reshape(batch, seq, num_v_heads, head_v_dim)

    beta = b.sigmoid()
    g = -params["linear_attn.A_log"].float().exp() * F.softplus(
        a.float() + params["linear_attn.dt_bias"].float()
    )

    repeats = num_v_heads // num_k_heads
    if repeats > 1:
        query = query.repeat_interleave(repeats, dim=2)
        key = key.repeat_interleave(repeats, dim=2)

    core, new_recurrent_state = gated_delta_rule_recurrent(
        l2norm(query.float()),
        l2norm(key.float()),
        value,
        g,
        beta,
        initial_state=recurrent_state,
    )
    core = core.to(hidden.dtype).reshape(-1, head_v_dim)
    gated = rms_norm_gated(core, params["linear_attn.norm.weight"], z.reshape(-1, head_v_dim), eps)
    gated = gated.reshape(batch, seq, value_dim)
    return (
        F.linear(gated, params["linear_attn.out_proj.weight"]),
        new_conv_state,
        new_recurrent_state,
    )


# ----------------------------------------------------------------------
# Dense MLP, decoder layer, head
# ----------------------------------------------------------------------


def swiglu_mlp(hidden: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """``down(silu(gate(x)) * up(x))`` -- dense in every layer, no router."""
    gate = F.linear(hidden, params["mlp.gate_proj.weight"])
    up = F.linear(hidden, params["mlp.up_proj.weight"])
    return F.linear(F.silu(gate) * up, params["mlp.down_proj.weight"])


def decoder_layer(
    hidden: torch.Tensor,
    params: dict[str, torch.Tensor],
    layer_type: str,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    eps: float = 1e-6,
    **mixer_kwargs: object,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Pre-norm residual around the layer's mixer, then around the MLP.

    The second element is the mixer's own carried state: ``(key, value)`` for
    full attention, ``(conv_state, recurrent_state)`` for a Gated DeltaNet
    layer, so a decode step can continue from a prefill's own cache.
    """
    residual = hidden
    normed = rms_norm_delta(hidden, params["input_layernorm.weight"], eps)
    if layer_type == "full_attention":
        mixed, extra = full_attention_layer(normed, params, cos, sin, eps=eps, **mixer_kwargs)
    elif layer_type == "linear_attention":
        mixed, conv_state, recurrent_state = gated_delta_net_layer(
            normed, params, eps=eps, **mixer_kwargs
        )
        extra = (conv_state, recurrent_state)
    else:
        raise ValueError(f"unknown layer type {layer_type!r}")
    hidden = residual + mixed

    residual = hidden
    normed = rms_norm_delta(hidden, params["post_attention_layernorm.weight"], eps)
    return residual + swiglu_mlp(normed, params), extra


def language_head(
    hidden: torch.Tensor, norm_weight: torch.Tensor, head_weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return F.linear(rms_norm_delta(hidden, norm_weight, eps), head_weight)


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """The metric set every parity report in this onboarding quotes.

    All are reported together and all must be finite: cosine alone hides a
    scale error, ``max_abs`` alone hides a systematic small bias, and a NaN in
    either operand makes both look fine in isolation.

    ``rel_max_abs`` normalizes ``max_abs`` by the reference's own magnitude,
    because an absolute bar is not scale free and bf16 is coarse: one ULP at
    magnitude 12 *is* 0.0625, so an atol tight enough to be meaningful for a
    normalized hidden state reads as a failure on an unnormalized one.
    """
    a = actual.detach().float().flatten()
    b = expected.detach().float().flatten()
    diff = (a - b).abs()
    scale = b.abs().max().item()
    max_abs = diff.max().item()
    return {
        "max_abs": max_abs,
        "mean_abs": diff.mean().item(),
        "rel_max_abs": max_abs / scale if scale > 0 else 0.0,
        "cosine": F.cosine_similarity(a[None], b[None]).item(),
        "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
    }
