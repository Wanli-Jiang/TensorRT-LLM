<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MTP3 maximum-throughput final-fix root-cause analysis

> **Superseded causal attribution (2026-08-19):** The retained run data and
> replay/residency/KV-cap conclusions remain valid, but the older claim that the
> FlashInfer static T=1 change caused a 6.462% serving gain is not accepted. That pair
> used different allocations. Same-allocation production A/B/A later measured
> dynamic/static/dynamic at 4049.545/4064.498/4096.202 tok/s/GPU, so static T=1 is
> optional for >4k. Use the repository-level
> `GDN_T1_AND_HIGH_THROUGHPUT_4000TPS_ROOT_CAUSE_REAUDIT_ZH_20260818.md` and this
> bundle's `TECHNICAL_CLARIFICATION.md` for the corrected interpretation.

Generated: `2026-08-16T21:22:22-07:00` (`America/Los_Angeles`)

Evidence cutoff: `2026-08-16T21:20:26-07:00` (Job 491712 completion)

Terminology clarification (`2026-08-17`): the phrase "DeepGEMM Split-K" in the
original report conflated two independent paths. DeepGEMM is the FP8 MoE backend;
`TRTLLM_LOW_M_GEMM_BACKEND=auto` routes eligible BF16 `M<=32` Linear operations to
FlashInfer `mm_bf16`, whose internal heuristic selects direct or Split-K. The observed
830--838 MiB failed allocation is the DeepGEMM MoE `triton_fused_gather_finalize`
output tensor, not a demonstrated FlashInfer Split-K workspace allocation. Also,
`d8d10ab` wide-value-head replay tuning is present in the tested PR #17537 source but
has no isolated peak-throughput A/B and is not classified as a required >4k fix. See
the delivery README and the canonical clarification report for the complete correction.

## Executive conclusion

The recovered `>4000 total tok/s/GPU` result is not one mysterious tuning win. It is the
result of two real software defects interacting with the serving geometry, followed by
two necessary resource/scheduling choices:

1. **FlashInfer GDN T=1 compile-cache specialization was under-specified.** The old path compiled
   from a `B=1` batch-dynamic template and omitted the concrete batch and tensor layout
   from its cache identity. At `B=1,T=1`, singleton dimensions make the dynamic layout
   descriptor ambiguous; a CUDA graph can then retain a cubin compiled without the
   intended concrete specialization. Keying and compiling T=1 by exact batch plus
   shape/stride removes a real
   per-decode-cycle penalty. Exact-shape microbenchmarks measure about 8% lower GDN
   latency, and the isolated end-to-end control improves by 6.462%. Functional output
   and recurrent state are bit-exact in the exact-shape control.
2. **Legacy MTP GDN state verification scaled with the full speculative-state tensor.**
   At `M=104,T=4`, the intermediate SSM state alone is 112.125 GiB/GPU. Cached replay
   retains prefix-invariant causal update vectors, normalized keys, and cumulative
   log-decay, then solves only the at-most-eight new cached updates. This reduces the
   audited verification scratch from 115.410 to 11.388 GiB/GPU and makes the resident
   batch needed by `C=1536, ADP=32` feasible.
3. **The throughput step after replay is primarily a residency/ramp effect, not faster
   per-user decode.** `1536/32 = 48` requests arrive per ADP rank. `M=36` can initially
   retain only 1152 globally, leaving 384 requests for another scheduler wave; `M>=48`
   can retain all 1536. The final M104 cohort has much lower TTFT and a much higher
   aggregate output ramp even though median TPOT is worse. Because input tokens are
   88.889% of the reported total-token metric, eliminating the second-wave makespan is
   worth more than a small decode-latency change.
4. **No-chunk prefill and an absolute KV cap are enabling choices.** No-chunk avoids at
   least eight 1024-token chunk scheduling/collective boundaries for an 8192-token
   prompt. The initially selected KV fraction of 0.88 works at M104, but a new negative
   control proves that a fraction is allocator feedback, not a portable headroom
   guarantee: uncapped M36 at the same 0.88 over-provisions attention KV and OOMs while
   DeepGEMM MoE gather/finalize allocates its output tensor. The robust setting is
   `max_tokens=479232` (52 average sequences per rank) in addition to 0.88. FlashInfer
   low-M auto, with direct/Split-K selected internally, is beneficial when that headroom
   exists; it is not the regression source and is not required to exceed 4k.

Therefore the root cause is **not** a node restart, metrics traffic, JIT contamination,
or the FlashInfer low-M route. The stable gain comes from fixing the T=1 GDN
specialization defect and the
MTP GDN state-lifecycle/capacity defect; no-chunk prefill and an absolute KV bound expose
the fixed stack's attainable throughput.

## What was fixed at the kernel level

### T=1 GDN cache identity

FlashInfer commit `baad0dca27d165341d188b895f3ab161e8098344` changes the T=1 path in
`flashinfer/gdn_kernels/gdn_decode_bf16_state.py` as follows:

- adds concrete `B` to the Python kernel cache key;
- adds every caller-visible tensor shape and stride to the T=1 cache key;
- builds concrete descriptors for T=1 tensors instead of deriving all runtime layouts
  from a singleton `B=1` template;
- passes `static_batch_size` into the CuTe compile signature so CuTe's internal cache
  also distinguishes CUDA-graph batch buckets;
- preserves the upstream dynamic-batch behavior for actual MTP calls with `T>=2`.

This matters because size-one dimensions do not uniquely identify which logical
dimension owns a runtime stride. A Python cache-key change alone would be insufficient
if CuTe's internal compile signature still reused a cubin; conversely, an exact CuTe
signature without layout-aware Python identity would still collide packed and
non-compact views. The fix closes both cache layers.

The regression test exercises a graph ladder (`B=24,16,4,2,1`) with packed/non-compact
QKV, padded state, and guard rows. That test is a correctness guard. The measured speed
effect is established separately:

| Control | Dynamic/old | Static/fixed | Change | Correctness |
|---|---:|---:|---:|---|
| GB300 exact shape, `B36,T1,H16,HV128,K128,V128`, CUDA graph, 50 warmup + 500 replay | 0.0553032 ms | 0.0512040 ms | +8.0055% speed | output/state bit-exact |
| Same shape, eager | — | — | +8.4460% speed | output/state bit-exact |
| Immutable fixed-vs-fixed rerun | ratio 1.000086 | reference | effectively identical | output/state bit-exact |
| End-to-end C1536, only GDN overlay changed | 3784.4831 | 4029.0545 tok/s/GPU | **+6.4625%** | complete, exact 8192/1024, zero errors |

The model has 69 GDN layers out of 92. T=1 is executed on the ordinary decode portion
of every speculative cycle, so a repeatable single-kernel penalty is amplified across
layers and decode iterations. The microbenchmark direction and the isolated end-to-end
direction agree, which is the strongest evidence that this portion is a real software
root cause.

### Cached replay changes state-computation complexity

The legacy verification path materializes an intermediate SSM state with the relevant
production dimensions:

```text
69 layers * M104 * T4 * K128 * V128 * HV128 * 2 bytes
= 120,393,302,016 bytes
= 112.125 GiB/GPU
```

Adding convolution scratch gives 115.410 GiB/GPU. Job 488785 fails while trying to
allocate 112.12 GiB, matching the formula rather than suggesting a random cluster
failure.

The cached replay implementation stores `old_u`, `old_k`, and `old_G`: the
prefix-invariant update vectors, normalized keys, and cumulative log-decay from the
checkpoint. For every verify call it solves only the `T` new updates with forward
substitution (`T<=8`), and the V2 all-layer commit advances checkpoint state in one
partitioned path. PR #16464 commit
`ee241d25f43973ad52495119d6536176b91c0aec` introduced the functional replay
mechanism. PR #16768 commit `57f2781e4e9f679cfa429400b64e447fbefa253e`
improved low-batch execution, default enablement, and V2 cache-manager/all-layer commit
support used by the tested stack. The later PR #17537 commit `d8d10ab` tunes ratio-8
wide heads and metadata preparation; it is exact-source provenance, not a proven
peak-throughput dependency. The audited M104 footprint is:

| Verification state | GiB/GPU |
|---|---:|
| Legacy intermediate SSM | 112.125 |
| Legacy convolution scratch | 3.285 |
| **Legacy total** | **115.410** |
| Replay history buffers + common convolution scratch | **11.388** |
| **Saved** | **104.022 (90.133%)** |

This is a state-lifecycle and capacity repair. The current evidence does **not** claim
that replay itself makes the same M36 batch's kernel faster. Its proven causal role is
that it makes M48/M104 viable, allowing the scheduler to cross the resident-batch
threshold.

## Why the maximum-throughput metric rises

The formal point always processes:

```text
4608 requests * (8192 input + 1024 output)
= 42,467,328 tokens

total tok/s/GPU = 42,467,328 / duration / 32
```

At `C=1536, ADP=32`, every rank initially receives 48 requests:

| Capacity | Global resident slots | Initially queued | Geometry |
|---|---:|---:|---|
| M36 | 1152 | 384 (12/rank) | requires another scheduler wave |
| M48 | 1536 | 0 | exactly admits the initial population |
| M104 | 3328 | 0 | admits the initial population with margin |

Before the final same-allocation threshold control, the repeat-group comparison already
showed the signature of aggregate-ramp improvement:

| Repeat-group mean | Static GDN M36 | Replay M104 | M104 change |
|---|---:|---:|---:|
| Total tok/s/GPU | 4013.968 | 4112.527 | **+2.455%** |
| Makespan | 330.626 s | 322.698 s | **-2.398%** |
| Median TTFT | 32.988 s | 7.789 s | **-76.390%** |
| Median TPOT | 69.903 ms | 95.979 ms | **+37.303% (worse)** |
| Peak aggregate output | 39,349 | 57,386 tok/s | **+45.837%** |

The result cannot be explained as faster per-request decode: TPOT goes in the opposite
direction. It is explained by admitting and prefilling the whole initial population,
raising aggregate decode occupancy earlier and reducing the fixed-work makespan.

### Direct same-allocation M36 versus M48 threshold control

Job 491712 ran both arms sequentially on the same eight-node segment. Both formal
results pass the full-result and corrected-window audits:

| Formal C1536 metric | M36, 1152 resident slots | M48, 1536 resident slots | M48 change |
|---|---:|---:|---:|
| Total tok/s/GPU | 3971.592 | **4120.374** | **+3.746%** |
| Fixed-work makespan | 334.149 s | 322.083 s | **-3.611%** |
| Median TTFT | 27.539 s | 7.978 s | **-71.032%** |
| Median TPOT | 74.296 ms | 95.857 ms | **+29.021% (worse)** |
| Peak aggregate output | 51,864 tok/s | 57,370 tok/s | **+10.616%** |
| Total manager quota | 34.189 GiB | 37.694 GiB | +3.504 GiB fixed state |
| Rank-0 PyTorch allowance | 236.454 GiB | 232.925 GiB | -3.529 GiB |

Each arm completed 4608/4608 requests with zero errors, nonempty generated text, exact
ISL 8192 and OSL 1024. Throughput above is recomputed from the same 42,467,328 tokens.
Neither formal window contains `/metrics`, loading heartbeat, JIT/tuning, graph capture,
OOM, or another fatal runtime signature.

The two recipes differ semantically only in isolated cache directory names and
`max_batch_size`/CUDA-graph maximum batch size (`36` versus `48`). Crucially, both use
the same `free_gpu_memory_fraction=0.88` and the same absolute
`max_tokens=479232`, equal to 52 average 9216-token sequences per rank. They use the
same allocation, image (SHA256
`08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59`), model,
TP32/EP32/ADP32, replay V2, no-chunk T8512, Static544,
DeepGEMM 65536, workspace 2304 MiB, KV=0.88, low-M auto, accept length 3.3,
concurrency 1536, one 4608-request warmup population, and one 4608-request measured
population (each three times the concurrency).

This is the missing direct proof of the population-residency mechanism. M48 improves
total throughput and makespan while making median TPOT substantially worse. A per-cycle
kernel speedup cannot produce that signature; removing the 384-request second wave can.
M48 also crosses 4k in the direct control, whereas headroom-safe M36 remains below it.
The extra 3.504 GiB quota and 3.529 GiB reduction in PyTorch allowance are expected
fixed-state costs of twelve additional resident slots/rank, not an attention-KV capacity
difference: both arms retain the same 479232-token cap.

## Why no-chunk helps but is not the root fix

Job 488694 is an in-allocation A/B/A-style control. Apart from isolated cache names, the
directional change is `max_num_tokens=1024` with chunking versus `8512` with chunking
disabled:

| Arm | Total tok/s/GPU | Makespan | Median TTFT | Median TPOT | Peak aggregate output |
|---|---:|---:|---:|---:|---:|
| Chunked | 3552.008 | 249.081 s | 35.308 s | 73.439 ms | 39,632 |
| No-chunk B1 | 3889.608 | 227.462 s | 26.446 s | 74.427 ms | 47,903 |
| No-chunk B2 | 3701.809 | 239.001 s | 29.560 s | 77.433 ms | 48,025 |

Both no-chunk arms increase total throughput and aggregate output ramp while slightly
worsening TPOT. For an 8192-token prompt, a 1024-token budget requires at least eight
chunk scheduling and collective boundaries. T8512 admits the context in one scheduling
unit and brings requests into decode sooner. However, M36 no-chunk did not stably cross
4k, so no-chunk is a directional enabler rather than the underlying code fix.

## Why a KV fraction alone is not a valid headroom control

The measured rank-0 KV quotas and calculated average-9216-token capacities are:

| KV fraction | Device quota | Approx. sequences/rank | Outcome |
|---|---:|---:|---|
| 0.85 | 49.620 GiB | 47.91 | below the 48/rank residency threshold |
| **0.88** | **51.371 GiB** | **52.06** | enough residency plus transient headroom |
| 0.90 | 52.547 GiB | 54.85 | DeepGEMM MoE gather/finalize output OOM while low-M auto is enabled |

At M104 and KV=0.90, `triton_fused_gather_finalize` requests an 838 MiB allocation after
model, KV, communication workspace, frontend context, PyTorch reserved/unallocated
memory, and CUDA-graph private pools are live. Lowering to 0.88 reduces KV quota by 1.176 GiB and
increases the rank-0 PyTorch allowance by 1.219 GiB while staying above 48 sequences.
KV=0.85 avoids the OOM but drops below the desired capacity cliff and reaches only
3974.603 tok/s/GPU in the tested run.

That table does **not** mean 0.88 is universally safe. A direct M36 negative control
(Job 491622) uses the final replay stack and the same 0.88 fraction, but leaves
`max_tokens` uncapped. Because M36 retains less recurrent state than M104, the KV
manager consumes the newly available memory:

| Matched M36 case | Uncapped fraction | Absolute cap | Effect of cap |
|---|---:|---:|---:|
| Total manager device quota | 62.918 GiB | 34.189 GiB | **-28.728 GiB** |
| Attention-KV tokens | 1,075,436 derived | 479,232 configured | -55.44% |
| Approx. 9216-token sequences/rank | 116.69 derived | 52.00 exact | -64.69 |
| Rank-0 PyTorch allowance | 207.735 GiB | 236.454 GiB | **+28.719 GiB** |
| DeepGEMM gather/finalize output | requests 830 MiB with only 236.94 MiB free; OOM | completes formal run at 3971.592 tok/s/GPU | headroom restored |

This negative result is deterministic allocator feedback, not a failed throughput
sample. It reached server ready, stopped the loading heartbeat before KV/JIT, entered
benchmark warmup, and failed precisely in `triton_fused_gather_finalize`. It shows why
“same fraction” is not “same non-KV headroom”: any state-memory optimization can be
silently converted into a larger KV pool. The corrected M36/M48 control therefore caps
both arms at `479232 = 52 * 9216` KV tokens. That remains four sequences/rank above the
C1536 residency requirement while holding the attention-KV allocation constant. The
manager's logged total quota also includes batch-dependent fixed/context state and the
`max_util_for_resume=0.95` reserve. Applying the exact affine inverse used by
`_get_max_tokens_from_quota()` to the matched capped quota gives the uncapped
116.69-sequence value. The uncapped attention pool is more than triple the scheduler's
36-request capacity;
`max_batch_size` does not itself prevent this KV over-provisioning.

The FlashInfer low-M route is exonerated by two independent checks:

- Matched KV=0.85 end-to-end: low-M auto 3974.603 versus low-M off 3917.025 tok/s/GPU,
  a **+1.470%** advantage; median TPOT improves 3.167%.
- BF16 Linear CUDA-graph microbenchmark: 72 shapes, 100 replays by seven trials, PDL
  enabled. This compares `F.linear` with FlashInfer `mm_bf16`; it is not a DeepGEMM
  microbenchmark. FlashInfer/low-M is preferred for 63/72 shapes; median
  existing/FlashInfer ratios include 1.3308 at N=256, 1.1997 at N=4096, and 1.0718 at
  N=8192. Maximum BF16 absolute error is 0.03125.

This is a joint serving-memory budgeting issue. The traceback identifies the failed
830--838 MiB allocation as DeepGEMM MoE gather/finalize output; it does not identify it
as FlashInfer Split-K scratch. Reserve non-KV headroom for that output, graph pools,
communication workspace, possible low-M temporaries, and fragmentation. Set both a
capacity floor and an absolute KV ceiling; a free-memory fraction by itself satisfies
neither invariant across batch/state configurations.

## Stability, confounder elimination, and limitations

The repaired M104 software/capacity family was reproduced across four allocations. The
first two use low-M off with KV=0.90; the latter two use the selected low-M auto with
KV=0.88. They are evidence that the repaired GDN/replay/no-chunk mechanism is stable,
not four identical recipe repeats:

| Job | Low-M | KV | Total tok/s/GPU |
|---:|---|---:|---:|
| 488899 | off | 0.90 | 4114.062 |
| 488993 | off | 0.90 | 4110.993 |
| 489192 | auto | 0.88 | 4112.350 |
| 489281 | auto | 0.88 | 4065.823 |
| **Family mean** | — | — | **4100.807** |

The four-run family CV is 0.493%; range/mean is 1.176%; every run is above 4k. The two
low-M-off repeats have a 0.037% CV; the two exact low-M-auto/KV=0.88 repeats have a
0.569% CV and 1.138% range/mean. This excludes “one lucky node restart” as a sufficient
explanation while keeping configuration differences explicit.

The following contamination paths were also excluded:

- no `/metrics` request and response-level performance metrics disabled;
- loading heartbeat stopped at the 70–79% model-loading boundary, before KV allocation,
  JIT/tuning, CUDA-graph capture, warmup, and formal measurement;
- all JIT and graph work completed before three explicit warmups and the formal window;
- every accepted JSON has `completed == num_prompts`, zero nonempty errors, nonempty
  generated text, exact ISL 8192, exact OSL 1024, and throughput recomputed from fixed
  work;
- raw-result hashes are retained in the machine-readable audit; direct-control recipe,
  runner, auditor, and image hashes are retained with the final controlled job.

An earlier `max_num_tokens` propagation bug is also real but is not this throughput root
cause. Forwarding the value removes false T8512-versus-default-8192 warnings and repairs
request-geometry/lifecycle reporting; the corrected control did not independently
recover 4k.

Generated-token trajectories are not stable enough to treat an isolated ~1% result as
causal: one audited pair has only 13.997% index-aligned exact output. That changes expert
routing and scheduler timing. Consequently, this report relies on same-allocation
controls, exact-shape microbenchmarks, hard memory formulas/OOM signatures, and repeated
end-to-end results; it does not attribute every final percentage additively.

## Root-cause decision tree for future failures

1. **Validate output and workload invariants first.** Reject incomplete, errored, empty,
   or wrong-length results before looking at throughput.
2. **Separate per-cycle kernel regression from population-ramp regression.** If TPOT and
   exact-shape kernels regress together, inspect specialization/cache identity. If TTFT,
   peak aggregate output, and makespan move while TPOT does not, inspect resident slots,
   prefill scheduling, and request waves.
3. **Compute capacity before running.** Record `concurrency/ADP`, `max_batch_size`, global
   resident slots, KV sequence capacity, and queued initial requests. Do not compare arms
   on opposite sides of a residency cliff without labeling that fact.
4. **Budget the largest transient jointly and cap KV absolutely.** Include KV, PyTorch allocated and reserved,
   CUDA graph private pools, frontend CUDA context, communication workspace, replay
   scratch, and measured GEMM transient. “Free memory” from one allocator is not the
   complete budget. A fixed KV fraction can consume every byte saved by another fix;
   record the final quota and enforce `max_tokens` when comparing memory geometries.
5. **Cold-start every changed compilation identity, then warm it out.** Use isolated
   FlashInfer/CUDA/Triton/TorchInductor caches, finish JIT and graph capture, run explicit
   warmups, and exclude all of them from the formal interval.
6. **Never query `/metrics` during the formal interval.** Also stop loading heartbeat
   before KV/JIT/graphs/warmup.
7. **Require a causal hierarchy.** Prefer same-allocation single-variable A/B, then exact
   microbench plus bit-exactness, then independent allocation repeats. Treat historical
   cross-allocation comparisons as contextual only.

## Evidence and reproducibility

- Machine-readable recomputation:
  `final-rerecheck/audits/ht4000/final-fix-causal-decomposition-20260816.json`
- Audit generator:
  `final-rerecheck/scripts/analyze_final_fix_root_cause.py`
- Direct residency A/B recipes with absolute KV cap:
  `final-rerecheck/recipes/perf/final-fix-residency-kvcap-ab-20260816/`
- Direct residency A/B output with absolute KV cap:
  `final-rerecheck/outputs/final-fix-residency-kvcap-ab-20260816/`
- Uncapped-fraction negative control (Job 491622):
  `final-rerecheck/outputs/final-fix-residency-ab-20260816/m36-a/491622/`
- GDN exact-shape microbenchmark:
  `final-rerecheck/outputs/ht4000/gdn-t1-exact-shape-microbench/467068/`
- Immutable GDN repeat:
  `final-rerecheck/outputs/ht4000/gdn-t1-immutable-exact-shape/467826/`
- Replay memory audit:
  `final-rerecheck/audits/ht4000/gdn-replay-resource-audit.json`
- FlashInfer low-M BF16 direct/Split-K-heuristic microbenchmark:
  `final-rerecheck/outputs/microbench/low-m-graph/job-489183/`
- Prior full-curve report:
  `final-rerecheck/reports/mtp3-over4k-full-curve-and-root-cause-20260816.md`

All percentage changes are local comparisons between their named controls. They must
not be summed, because the fixes change interacting bottlenecks at different layers.
