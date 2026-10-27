# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family library: it loads, carries both targets, and defines and fakes its ops.

Importing a target package loads ``libth_modeling_v2_kimi_k3_vl.so`` through
``_native.load_target_ops``, the way routing does. The ops' numerics are certified
by their entries' own tests (``norm/``, ``comm/`` and ``kernels/kimi_k3/``); this is
the family's smoke test, which CBTS selects for any change to the family.
"""

import importlib

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tensorrt_llm._torch._experimental.modeling_v2 import _native
from tensorrt_llm._torch._experimental.modeling_v2._native import load_family_ops, namespace_of

_FAMILY = "kimi_k3_vl"
_PACKAGE = f"tensorrt_llm._torch._experimental.modeling_v2.models.{_FAMILY}"
_TARGETS = ("kimi_k3_mxfp4__sm_100__tp16_moetp4ep4", "kimi_k3_mxfp4__sm_100__tp16_moetp16ep1")
_OPS = (
    "k3_attn_res_rmsnorm_fwd",
    "k3_attn_res_add_rmsnorm_fwd",
    "k3_mnnvl_fusion_allreduce",
    "mnnvl_allgather_split",
    "mnnvl_allreduce_attn_res",
)

assert torch.cuda.is_available(), "the Kimi K3 family library needs a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] != 10, reason="built for sm_100f only"
)


@pytest.mark.parametrize("target", _TARGETS)
def test_each_target_loads_the_family_library(target):
    importlib.import_module(f"{_PACKAGE}.{target}")
    assert set(_TARGETS) <= _native._load_family_library(_FAMILY)


def test_the_library_defines_every_shared_op():
    load_family_ops(_PACKAGE)
    prefix = f"{namespace_of(_PACKAGE)}::"
    defined = {
        s.name[len(prefix) :] for s in torch._C._jit_get_all_schemas() if s.name.startswith(prefix)
    }
    assert set(_OPS) <= defined, f"missing from the library: {sorted(set(_OPS) - defined)}"


def test_the_attn_res_op_runs_and_its_fake_describes_it():
    ops = load_family_ops(_PACKAGE)
    tokens, candidates, hidden = 2, 3, 7168
    g = torch.Generator(device="cuda").manual_seed(0)

    def bf16(*shape):
        return torch.randn(*shape, generator=g, device="cuda").bfloat16()

    args = (
        bf16(tokens, 1, hidden),
        bf16(candidates, tokens, 1, hidden),
        bf16(hidden),
        bf16(hidden),
        bf16(hidden),
        1e-6,
        1e-6,
    )
    out = ops.k3_attn_res_rmsnorm_fwd(*args)
    assert out.shape == args[0].shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()

    with FakeTensorMode() as mode:
        fake = ops.k3_attn_res_rmsnorm_fwd(
            *(mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args)
        )
    assert fake.shape == out.shape and fake.dtype == out.dtype
