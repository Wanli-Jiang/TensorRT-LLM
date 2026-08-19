<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replay, wide-head, low-M/Split-K, and >4k technical clarification

Updated: `2026-08-19` (`America/Los_Angeles`)

This note corrects three misleading shortcuts in the original delivery language without
changing any retained result:

1. A commit being present in PR #17537 does not prove that it is causally required for
   >4k peak throughput.
2. DeepGEMM MoE and FlashInfer low-M direct/Split-K are independent paths and must not
   be named as one "DeepGEMM Split-K" component.
3. The older cross-allocation `3784.483 -> 4029.055` comparison does not prove that static
   T=1 GDN specialization caused a 6.462% serving gain. A later same-allocation A/B/A did
   not isolate a positive E2E effect beyond trajectory noise.

## Causal classification

| Item | Evidence-backed classification |
|---|---|
| FlashInfer T=1 GDN specialization/cache fix | Bit-exact and faster at one exact GB300 kernel shape; latest same-allocation production A/B/A found no resolved E2E gain; optional for >4k |
| GDN cached replay from #16464 plus V2 support from #16768 | Proven state-lifecycle/capacity repair for the tested V2 stack |
| M48 at C1536/ADP32 | Proven same-allocation residency threshold |
| Absolute KV cap 479232 | Strong allocator/OOM causal evidence; not a strict same-allocation performance A/B |
| No-chunk prefill | Same-allocation directional prefill/ramp evidence; not an independently proven code root cause |
| FlashInfer low-M BF16 auto | Matched +1.470% benefit, but not required for >4k |
| `d8d10ab` wide-head replay tuning | Exact PR provenance; peak-HT benefit unproven |

## Corrected T=1 GDN production attribution

The exact-shape graph-replay microbenchmark remains valid: dynamic measured about 55.288 us and
static about 51.186 us, a 4.102 us or 7.420% local kernel reduction with bit-exact output/state.
However, this model executes a much larger serving critical path. An intentionally conservative
Amdahl bound places the local saving below about 0.5% of an MTP cycle.

Job `505210` restarted the full server three times on the same eight GB300 nodes:

| Arm | GDN | C1536 total tok/s/GPU |
|---|---|---:|
| A | dynamic | 4049.545 |
| B | static | 4064.498 |
| A2 | dynamic | 4096.202 |

Static is inside the dynamic bracket and 0.206% below the dynamic mean. All three arms passed the
same exact request and formal-window audits. Consequently, the static patch is valid local
hardening but is not a high-throughput acceptance gate. The stable >4k mechanism is cached replay
capacity plus M48 residency, no-chunk population ramp, and explicit KV/transient headroom.

## Replay source lineage

- PR #16464 / `ee241d25f4`: introduced functional GDN MTP cached replay,
  compact replay histories, bookkeeping, and the base commit mechanism.
- PR #16768 / `57f2781e4e`: improved low-batch tiling and bookkeeping, enabled
  eligible replay by default, and completed V2 cache-manager/state-layout and
  all-layer commit support used by this delivery.
- PR #17537 / `d8d10ab354`: added a ratio-8 wide-value-head follow-up, a ratio-8
  commit-pipeline adjustment, and fused replay work-item preparation for at most 256
  decode requests.

The source-verification script checks all three commits because this bundle pins exact
PR #17537 ancestry. That is a provenance gate. The minimal tested replay mechanism is
#16464 plus #16768 for the V2 configuration; no isolated A/B proves that `d8d10ab` is
required to cross 4k.

## Wide-head trigger scope

The ratio-8 main replay mapping requires:

```text
T=4, BF16 state, K=V=128, history<=16, HV=8*H, N*HV<=512
```

For Oakhaven `H=16,HV=128`, this reduces to `N<=4`. C1536/ADP32 starts with 48
requests/rank, so the main wide-head mapping does not apply to the steady-state
high-throughput population. A microbenchmark using T=1, N>=8, ordinary replay only,
or no metadata preparation should therefore not be expected to show its intended
benefit.

The commit could still affect small-N tails, periodic all-layer commit, or metadata
preparation. Those possibilities remain unproven because no retained run reverts only
`d8d10ab` while holding the image, allocation, M48, KV cap, no-chunk setting, and
FlashInfer fix constant.

## What cached replay actually proves

Legacy M104/T4 verification resource geometry:

```text
69 layers * 104 * 4 * 128 * 128 * 128 * 2 bytes
= 112.125 GiB/GPU intermediate SSM state
+ 3.285 GiB convolution scratch
= 115.410 GiB/GPU
```

The retained failure requests 112.12 GiB, matching the formula. Cached replay retains
compact `old_u`, `old_k`, and `old_G` histories and reduces audited verification
scratch to 11.388 GiB/GPU, saving 104.022 GiB/GPU or 90.133%.

This proves a capacity/state-lifecycle repair. It does not prove that replay makes the
same M36 kernel latency faster.

## Why M48 changes total throughput

```text
C1536 / ADP32 = 48 requests/rank
M36: 36*32=1152 resident, 384 requests queued for a second wave
M48: 48*32=1536 resident, no initial second wave
```

Job 491712 same-allocation A/B:

| Metric | M36 | M48 | Change |
|---|---:|---:|---:|
| Total tok/s/GPU | 3971.592 | 4120.374 | +3.746% |
| Makespan | 334.149 s | 322.083 s | -3.611% |
| Median TTFT | 27.539 s | 7.978 s | -71.032% |
| Median TPOT | 74.296 ms | 95.857 ms | 29.021% worse |

TPOT gets worse while fixed-work throughput improves. That signature proves a
residency/population-ramp effect rather than faster per-user decode.

## DeepGEMM and FlashInfer low-M are different paths

```text
FP8 routed experts
  -> TensorRT-LLM DeepGEMM MoE backend
  -> triton_fused_gather_finalize

eligible BF16 Linear with M<=32
  -> TRTLLM low-M dispatcher
  -> FlashInfer mm_bf16 backend="cute-dsl"
  -> internal direct/Split-K heuristic
```

`TRTLLM_LOW_M_GEMM_BACKEND=auto` does not enable a DeepGEMM Split-K mode. It makes
the FlashInfer path eligible and permits measured crossover shapes to fall back to the
normal BF16 Linear implementation. Logs record that a shape is routed to the heuristic;
they do not record whether FlashInfer ultimately selected direct or Split-K for every
shape.

Split-K divides GEMM's K reduction into independent partial products and reduces them
afterward. It can improve occupancy when M is very small, at the cost of reduction and
possible workspace overhead.

Job 489183 compares `F.linear` against FlashInfer `mm_bf16`, not DeepGEMM. Across 72
GB300 CUDA-graph shapes, FlashInfer is faster in 63; representative existing/FlashInfer
ratios are 1.3308 at N=256, 1.1997 at N=4096, and 1.0718 at N=8192. Maximum BF16
absolute error is 0.03125.

At matched KV=0.85, low-M auto measures 3974.603 tok/s/GPU versus 3917.025 with
low-M off, a +1.470% benefit. However, low-M-off M104 runs reach 4114.062 and
4110.993 tok/s/GPU. The route is beneficial but is not required for >4k.

## Correct interpretation of the 830--838 MiB OOM

The retained traceback is:

```text
DeepGemmFusedMoE.run_moe
  -> triton_fused_gather_finalize
  -> torch.empty((num_rows, unpadded_hidden_size), dtype=h3.dtype)
  -> CUDA OOM requesting 830 or 838 MiB
```

The failed allocation is the final DeepGEMM MoE output tensor. FlashInfer low-M auto,
CUDA graph private pools, communication workspace, and other resources may be live at
the same time and contribute to peak usage, but the traceback does not identify the
830--838 MiB tensor as Split-K partial workspace.

Use these terms:

| Avoid | Use instead |
|---|---|
| DeepGEMM Split-K | Name DeepGEMM MoE and FlashInfer low-M separately |
| Split-K OOM | DeepGEMM gather/finalize output OOM, unless traceback proves otherwise |
| Split-K on | FlashInfer direct/Split-K available, unless tactic evidence exists |
| wide-head required | Present in exact PR source; peak-HT benefit unproven |

## KV cap and transient headroom

`free_gpu_memory_fraction` is allocator feedback, not a fixed non-KV reserve. In the
M36 uncapped control, replay/state savings are converted into an oversized KV pool:

| Resource | fraction=0.88 only | fraction=0.88 + cap 479232 |
|---|---:|---:|
| Manager quota | 62.918 GiB | 34.189 GiB |
| Attention KV tokens | 1,075,436 | 479,232 |
| Rank-0 PyTorch allowance | 207.735 GiB | 236.454 GiB |
| DeepGEMM gather/finalize | 830 MiB request OOM | formal run succeeds |

The robust MTP3 selected point uses `479232 = 52*9216` tokens/rank. It remains four
average sequences above the 48-request residency threshold while preventing KV from
reclaiming approximately 28.7 GiB of transient headroom.

## Final selected-point settings

```yaml
TP/EP/ADP/PP: 32/32/32/1
MoE: TensorRT-LLM DeepGEMM, Static544
workload: ISL8192, OSL1024, RR1.0, request_rate=inf, C1536
MTP: draft=3, forced accepted drafts=2.3, total accept length=3.3
max_batch_size: 48
max_num_tokens: 8512
enable_chunked_prefill: false
GDN replay: true
KV fraction: 0.88
KV max_tokens: 479232
TRTLLM_LOW_M_GEMM_BACKEND: auto
TRTLLM_DEEPGEMM_MOE_MAX_NUM_TOKENS: 65536
TRTLLM_MOE_A2A_WORKSPACE_MB: 2304
```

The causal requirements and optional optimizations are not interchangeable:

- proven causal/enabling: replay capacity, resident capacity at least C/ADP, and explicit
  KV/transient budget;
- strong directional: no-chunk prefill for the 8192-token prompt;
- beneficial but optional: FlashInfer low-M auto;
- bit-exact local hardening but optional for >4k: FlashInfer static T=1 specialization;
- exact PR provenance with unproven peak-HT benefit: `d8d10ab` wide-head tuning.

## Remaining experiments

To establish the wide-head contribution, run one image/allocation with only
`d8d10ab` reverted/restored and measure ordinary replay at N=1/2/4/8/16/32/48,
periodic commit, metadata preparation at num_decodes=16/40/48/256/257, and final
C1536 end-to-end throughput. Keep all other source, caches, M48, KV cap, no-chunk, and
FlashInfer identities unchanged.
