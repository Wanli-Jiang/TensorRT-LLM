<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# MTP3 >4k qualification, full curves, and root-cause analysis

Updated: 2026-08-16 PDT

Terminology corrected: 2026-08-17 PDT. DeepGEMM is the FP8 MoE backend. The
separate low-M path is FlashInfer `mm_bf16` for eligible BF16 Linear shapes;
FlashInfer internally selects direct or Split-K. The 830--838 MiB OOM cited below
occurred while DeepGEMM MoE gather/finalize allocated its output, not at a proven
FlashInfer Split-K workspace allocation.

## Conclusion

The current stack can sustain more than 4000 total tokens/s/GPU for the
requested FP8 high-throughput workload.  Four independent C=1536 allocations
measured `4114.062`, `4110.993`, `4112.350`, and `4065.823` tokens/s/GPU.  The
mean is `4100.807`, every run is above 4000, the population CV is `0.493%`, and
the full range is `1.176%` of the mean.

The gain is not one opaque tuning effect.  Three deterministic mechanisms
compose the stable result:

1. A FlashInfer GDN T=1 descriptor/cache-key repair removes a real kernel
   specialization bug.
2. GDN cached replay removes 104.022 GiB/GPU of MTP verify scratch at M104,
   allowing every one of the 48 C=1536 requests per ADP rank to remain
   resident instead of entering a second scheduler wave.
3. KV fraction 0.88 reserves enough PyTorch transient headroom for the DeepGEMM
   MoE output, CUDA graph pools, communication workspace, and the separately enabled
   FlashInfer low-M path while preserving a theoretical 52.06 average
   9216-token sequences per rank.  This is a workaround for a combined peak-memory
   budgeting gap, not a faster kernel choice by itself.

Those changes eliminate deterministic bad specialization, M36 slot-turnover
sensitivity, and a peak-memory cliff.  This is why the benefit repeats across
different domains and even in GPU-colocated allocations.

## Frozen workload and serving configuration

- Model: `oakhaven-max-final-fp8_vv3`
- TP32 / EP32 / ADP32, one rank per GPU
- FP8, TRT-LLM MoE backend, DeepGEMM Static544 placement, RR=1.0
- ISL=8192, OSL=1024, infinite request rate, exact token lengths
- MTP3: three draft tokens, forced value 2.3, which means average accept
  length 3.3 because the metric includes the target token
- MTP3 serving: V2 hybrid manager, M104, no chunking, max tokens 8512,
  GDN cached replay on, FlashInfer low-M backend auto with an internal direct/Split-K
  heuristic, DeepGEMM row budget
  65536, MoE A2A workspace 2304 MiB, KV fraction 0.88
- no-MTP serving: V2 hybrid manager, M104, no chunking, max tokens 8448,
  DeepGEMM row budget 65536, MoE A2A workspace 2304 MiB, KV fraction 0.92
- Image:
  `trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh`

The loading heartbeat stopped at 70--78% of safetensors loading.  Kernel JIT,
CUDA graph capture, benchmark warmup, and every formal timing window occur
afterward.  Formal windows contain no `/metrics` requests, heartbeat activity,
JIT/tuning, graph capture, fatal signatures, or serving probes.

## Stable C=1536 reproduction

| Job | FlashInfer low-M backend | KV fraction | Domain | Total tok/s/GPU | Audit |
|---:|---|---:|---|---:|---|
| 488899 | off | 0.90 | d031 | 4114.062 | pass |
| 488993 | off | 0.90 | d010 | 4110.993 | pass |
| 489192 | auto | 0.88 | d178 | 4112.350 | pass |
| 489281 | auto | 0.88 | d138 | 4065.823 | pass |

The first three runs alone have a `0.0305%` population CV and a `0.0746%`
range.  Including the final full-curve run gives the more conservative
four-allocation statistics reported above.  Jobs 489192 and 489281 were
reported as GPU-colocated by the post-hoc topology audit, so clean-domain
placement is not a hidden requirement for crossing 4k.

## Full curve result

The canonical CSV, JSON, and SVG contain 24 no-MTP points and 21 MTP3 points.
Every included point was rebuilt directly from its raw SA-Bench JSON and then
cross-checked against the request-invariant and corrected formal-window
audits.

| Mode | C=1 | C=512 | C=1024 | C=1280 | C=1536 | Observed peak |
|---|---:|---:|---:|---:|---:|---:|
| no-MTP | 10.496 | 2289.363 | 3135.358 | 3268.752 | 3500.194 | 4055.616 at C=3264 |
| MTP3 | 27.094 | 3187.123 | 3825.015 | 3980.583 | 4065.823 | 4074.901 at C=1696 |

MTP3 stays above 4000 at C=1408, 1536, 1568, and 1600, falls to 3936.828 at
C=1632, and measures 4011.161 / 4074.901 at C=1664 / 1696.  The
cross-allocation wobble does not change the conclusion: C=1536--1696 is a
saturation plateau.  From C=1536 to C=1696, throughput rises only 0.223%, while
median TTFT rises 81.90% (7.916 to 14.399 seconds) and p99 TTFT rises 41.03%
(81.239 to 114.568 seconds).  no-MTP continues scaling much farther and reaches
4055.616 at C=3264; C=3328 is the exact M104 capacity edge and fails with CUDA
OOM.

At the same C=1536, MTP3 is 16.160% faster than no-MTP.  MTP3 first crosses
4k at C=1408, while no-MTP needs C=3072, so MTP3 reaches essentially the same
throughput with less than half the request concurrency.

Artifacts:

- `results/full-curves-20260816/nomtp-mtp3-full-curve.csv`
- `results/full-curves-20260816/nomtp-mtp3-full-curve.json`
- `results/full-curves-20260816/nomtp-mtp3-full-curve.svg`
- `results/full-curves-20260816/manifest.json`

## Why the fixes produce a stable gain

### 1. FlashInfer T=1 GDN descriptor/cache-key bug

FlashInfer 0.6.16/0.6.17 formed a batch-dynamic T=1 specialization from a
B=1 template and omitted the concrete batch/layout identity from the compile
cache.  Singleton dimensions make some serving layouts ambiguous; CUDA graph
capture then repeatedly reuses the selected slow or incorrectly described
cubin.  The local fix `baad0dca27d165341d188b895f3ab161e8098344` keys the
specialization by concrete batch and layout and has a guard-row regression
test for the B=24/16/4/2/1 serving ladder.

At the exact B=36 model shape, current dynamic descriptors take 0.0553032 ms
under CUDA graph replay and the repaired static descriptors take 0.0512040 ms,
an 8.0055% reduction with bit-exact output and recurrent state.  A GDN-only
end-to-end isolation at C=1536 improves 3784.483 to 4029.055 tokens/s/GPU,
or 6.462%.  Since 69 of 92 layers are GDN layers and MTP3 repeatedly executes
T=1 draft/decode work, this small kernel cost is amplified across the decode
loop.  Warming all variants before formal timing excludes its larger initial
compile cost.

This is a real correctness/performance bug fix, not node tuning: the faulty
key is deterministic process state, the exact-shape microbenchmark is
bit-exact and graph-replayed, and the end-to-end arm changes only the GDN
module.

Evidence: `audits/ht4000/flashinfer-gdn-version-audit.md` and
`outputs/ht4000/gdn-t1-exact-shape-microbench/467068`.

### 2. GDN replay is a state-lifecycle and capacity repair

Without replay, M104 verification attempts to allocate a 112.125 GiB/GPU
intermediate SSM tensor plus 3.285 GiB of convolution scratch.  The cached
replay path retains compact causal-update histories instead.  Its total
verify scratch is 11.388 GiB/GPU, saving 104.022 GiB/GPU, or 90.133% of the
legacy total.  Runtime controls confirm that legacy M104 fails on the
112.12-GiB allocation while replay M104 succeeds repeatedly.

At C=1536, ADP32 supplies 48 requests to each rank.  M36 necessarily creates a
second wave whose launch and completion order depends on generated tokens,
expert routing, and slot reuse.  M104 keeps all 48 resident, so these trajectory
differences no longer amplify into large throughput variance.  Only 645 of
4608 generated texts (13.997%) align exactly between two successful V2 runs,
yet throughput remains stable.  Stability therefore does not require an
identical autoregressive trajectory.

Relevant TensorRT-LLM changes are
`ee241d25f43973ad52495119d6536176b91c0aec` and
`57f2781e4e9f679cfa429400b64e447fbefa253e`.  Resource evidence is in
`audits/ht4000/gdn-replay-resource-audit.json`.

### 3. KV=0.88 fixes serving headroom, not low-M GEMM speed

With replay M104, KV=0.90 and FlashInfer low-M auto fail during serving warmup in
DeepGEMM `triton_fused_gather_finalize`: its final output tensor attempts an 838 MiB allocation
with only 1.21 GiB free.  The rank-0 process already uses 274.66/276.62 GiB,
PyTorch is allowed 216.88 GiB, 212.02 GiB is allocated, 694 MiB is in graph
private pools, 4.05 GiB is reserved but unallocated, and the frontend CUDA
context consumes another 750 MiB.  These independently budgeted consumers
meet at a deterministic peak-memory cliff.

KV=0.88 gives PyTorch about 1.219 GiB more allowance than KV=0.90, enough for
the 838-MiB transient, while retaining an estimated 52.06 average 9216-token
sequences per rank.  KV=0.85 is safe but holds only 47.91 such sequences and
falls into the capacity cliff at C=1536, measuring 3974.603 tokens/s/GPU.
The 0.90 frontier later deadlocks with 54 active requests after 1921 successes,
matching the predicted exhausted-KV failure mode.

The FlashInfer low-M route is not the regression. At the same KV=0.85 control setting,
auto measures 3974.603 while low-M off measures 3917.025, a directional +1.47%
advantage. A 72-shape CUDA-graph microbenchmark compares `F.linear` with FlashInfer
`mm_bf16`; it is not a DeepGEMM microbenchmark. The environment setting makes the
direct/Split-K heuristic available but does not prove which tactic each shape selected.
The required fix is joint resource budgeting/headroom. Low-M auto is beneficial in the
matched control but is not required for >4k: two low-M-off M104 runs reached 4114.062
and 4110.993 tok/s/GPU.

Evidence:

- `outputs/mtp3-v2-m104-replay-splitk-20260816/c1536/488992`
- `outputs/mtp3-v2-m104-replay-splitk-headroom-20260816`
- `outputs/mtp3-v2-m104-replay-frontier-20260816/high/489009`
- `outputs/microbench/low-m-graph/job-489183/results.json`

### 4. max_num_tokens propagation is a separate real bug

An earlier hybrid-manager construction path ignored the requested
`max_num_tokens`.  Job 464824 therefore emitted 13,997 false T=8512 versus
default-8192 warnings.  Forwarding the argument removes those warnings, but
the corrected control did not recover 4k.  This is a genuine request-geometry
and lifecycle fix, but it is not the causal explanation for the final MTP3
throughput.

## Accuracy and measurement integrity

Natural MTP3 serving, without forced acceptance, passes the deterministic
GSM8K smoke test at 8/8 exact and 8/8 parseable with zero invalid responses.
The prior full campaign also remains clean:

| Mode | GSM8K | HMMT25 | GPQA |
|---|---:|---:|---:|
| no-MTP | 1291/1319 (97.877%) | 30/30 | 179/198 (90.404%) |
| MTP3 | 1293/1319 (98.029%) | 30/30 | 181/198 (91.414%) |

The natural MTP smoke artifact is
`outputs/accuracy-smoke/mtp3-v2-m104-replay/job-489092/semantic-smoke/audit.json`.

The result generator refuses points with incomplete requests, non-exact
8192/1024 lengths, request errors, empty text, throughput disagreement, failed
formal windows, forbidden formal events, or `/metrics` requests.  Topology
colocation is retained as evidence rather than used as an automatic exclusion,
because successful and regressed runs occur in overlapping topology classes.

## Operational recommendation

For this workload, use the frozen MTP3 configuration above and C=1536 as the
default high-throughput operating point.  C=1696 adds only 0.223% throughput
but has 81.90% higher median TTFT and 41.03% higher p99 TTFT.  Keep C=1408 as a
small headroom option and do not select a concurrency solely from the
theoretical KV capacity edge.  Preserve three independent guards in future
qualification:

1. exact-shape graph-replayed GDN descriptor/cache-key regression test;
2. explicit combined KV, graph-pool, frontend-context, and largest-transient
   memory budget with a safety margin;
3. at least three audited end-to-end repeats across different allocations,
   with JIT and loading heartbeat outside formal windows and no `/metrics`
   traffic during measurement.
