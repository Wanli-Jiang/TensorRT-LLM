<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Oakhaven Max NVFP4 aggregate-first reproduction lock

Timestamp: 2026-08-20 America/Los_Angeles

## Scope and ordering

1. Run aggregate low latency for no-MTP and MTP3, without Static EPLB.
2. Run aggregate high throughput for no-MTP and MTP3, both no-EPLB and
   Static EPLB.
3. Accept aggregate results only after request-integrity, Slurm, heartbeat,
   formal-window, no-`/metrics`, and TRTLLM5-parity audits pass.
4. Only then run the best TRTLLM5 no-EPLB disaggregated E2E high-throughput
   topology for no-MTP and MTP3.

The latest accepted TRTLLM5 no-EPLB disaggregated anchors are frozen as:

- no-MTP: job `500677`, `3 x CTX8-ADP8/EP8 + 1 x GEN16-ADP16/EP16`,
  C1536, 40 deployed GPUs, 24 rounds, `5321.710736` total TPS/GPU.
- MTP3: job `524759`, `4 x CTX8-ADP8/EP8 + 1 x GEN16-ADP16/EP16`,
  C768, 48 deployed GPUs, 12 rounds, `5296.636805` total TPS/GPU. C640 is
  the practical knee, but C768 is the accepted maximum-total-throughput point
  requested for the single best-configuration reproduction.

Their current-image recipes are exact source-config derivatives. Controlled
changes are limited to the immutable current image with no source overlay,
experiment-local launcher/cache and heartbeat plumbing, and disabling all
TensorRT-LLM performance/iteration/request metrics and iteration logs. Native
disaggregate configs are expanded by the same reference `submit.py` used by
TRTLLM5; they are not srtctl-schema aggregate recipes.

After the authoritative V1 aggregate runs pass, run a separate KV-cache
manager V2 diagnostic. V2 is never substituted for the TRTLLM5 baseline:

- LL: no-EPLB no-MTP and MTP3 across the full C1-C512 ladder.
- HT: no-EPLB no-MTP at C2048/C2304 and MTP3 at
  C640/C768/C1024.
- Record both the same-fraction behavior and the V1-reported effective cache
  capacity. For a manager-only A/B, V2 must set `avg_seq_len=9216` for this
  hybrid model and cap `max_tokens` to the effective V1 capacity; otherwise a
  capacity difference is a confounder and the result is labeled accordingly.

The capacities frozen from the rank-synchronized V1 initialization logs are:

- LL no-MTP: `6,361,024` tokens.
- LL MTP3: `5,220,000` tokens.
- HT no-MTP: `1,295,328` tokens.
- HT MTP3: `1,606,784` tokens.

LL MTP3 additionally reports a `9,990,848`-token draft-model pool. That is not
the serving capacity: request admission is constrained by the smaller
`5,220,000`-token target-model pool. The V2 A/B therefore matches the smaller
pool. The auditor ignores the small profiling/warmup pools and, when target
and draft pools both exist, selects the smaller formal pool.

The canceled jobs and incomplete outputs under
`../full-matrix-agg-disagg-ll-ht-20260820` are diagnostic-only and ineligible
for the final comparison.

## Frozen deployment identity

- Model: `/lustre/fsw/portfolios/coreai/users/williamj/models/oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean`
- Model `config.json` SHA-256:
  `3b0153fad68686977da18f663455c4164499d0cf752d0767d19e2b46090b34e9`
- Current image:
  `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh`
- Image SHA-256:
  `08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59`
- Image size: `35,313,352,704` bytes
- No TensorRT-LLM source overlay is allowed.
- Workload: random exact 8192 input tokens, exact 1024 output tokens,
  `random_range_ratio=1.0`, request rate unlimited.
- Serving KV data type: FP8. Checkpoint precision: NVFP4.
- KV-cache manager: V1, matching the accepted TRTLLM5 references.
- Block reuse: disabled.
- TensorRT-LLM performance/iteration/request metrics are disabled. No client,
  monitor, readiness probe, or script may request `/metrics`.

## Low-latency lock

The authoritative TRTLLM5 references are aggregate jobs 422737 (no-MTP) and
422739 (MTP3). Both use TP8/EP1, attention DP disabled, TRTLLM MoE, two GB300
nodes/eight GPUs, max batch size 128, no chunked prefill, and the concurrency
ladder `1,2,4,8,16,32,64,128,256,512`.

- no-MTP: `max_num_tokens=8448`, KV fraction 0.74, GDN replay off.
- MTP3: `max_num_tokens=8704`, KV fraction 0.70, GDN replay on,
  `max_draft_len=3`, forced accepted draft tokens 2.3 (total acceptance length
  3.3).
- Benchmark count matches TRTLLM5: five formal requests and one warmup request
  per concurrency unit; HTTP connection reuse is disabled.

## High-throughput lock

TRTLLM5 shows that the accepted backend is CUTEDSL, not DeepGEMM. All four
aggregate arms use ADP16/EP16 (`tensor_parallel_size=16`,
`moe_expert_parallel_size=16`, attention DP enabled), four GB300 nodes/16
GPUs, NVLINK_ONE_SIDED MoE communication, 256-token A2A dispatch/combine
blocks, a 2304 MiB A2A workspace, and CUTEDSL MoE.

### no-MTP

- Reference no-EPLB job 431407: max batch size 128, max tokens 8448, no
  chunked prefill, KV fraction 0.92. Its accepted peak is 4940.091 total
  tokens/s/GPU at C2048; C2304 declines to 4876.871.
- Reference Static528 job 432389: identical serving configuration plus the
  calibrated 528-slot map. It reaches 5476.811 total tokens/s/GPU at C2048.
- Final ladder:
  `1,2,4,8,16,32,64,128,256,512,768,1024,1280,1536,1792,2048,2304`.

### MTP3

- Reference no-EPLB jobs 430411 and 431024: max batch size 32, max tokens
  1024, chunked prefill enabled, MoE max tokens 8192, KV fraction 0.92, GDN
  replay on, forced accepted draft tokens 2.3. The accepted peak is 2961.497
  total tokens/s/GPU at C640; the C768-C1024 tail is flat/noisy then declines.
- Reference Static528 job 431623: identical serving configuration plus the
  calibrated 528-slot map. It reaches 2617.220 total tokens/s/GPU at C640,
  i.e. Static is a regression for this workload and must not be assumed to
  help.
- Final ladder:
  `1,2,4,8,16,32,64,128,256,320,384,448,512,640,768,896,1024`.

All high-throughput arms use three formal requests and one warmup request per
concurrency unit and reuse HTTP connections, matching TRTLLM5.

## Static-map lock

- no-MTP Static528 SHA-256:
  `68ef80e1b996cc900f29c859221155bf6a3b7e7c1e5c83faf476d1c12b2f88be`
  (92 layers, 528 slots/layer, 512 experts, EP16, 33 slots/rank).
- MTP3 Static528 SHA-256:
  `cbc575a8d0bf221a76d3a04552a123d27964b788dee233be4eced6b6a71b30dc`
  (93 layers including the draft layer, 528 slots/layer, 512 experts, EP16,
  33 slots/rank).

The local copies are byte-identical to both the TRTLLM5 originals and the
published oakhaven-kernels copies.

## Acceptance gates

- Slurm allocation is `COMPLETED` with exit code `0:0`, correct GPU count,
  and all nodes in one NVL72 domain. Submissions request a Slurm segment equal
  to the node count in addition to `--switches=1@10:00`; a cross-domain
  allocation is invalidated before serving starts.
- Runtime config is byte-identical to the submitted recipe; image, model,
  maps, source-overlay absence, topology, backend, workload, and metrics flags
  match this lock.
- Every result has the planned/completed request count, exact input/output
  length arrays, no request errors, and correct throughput arithmetic divided
  by all serving GPUs.
- The loading heartbeat stops automatically before KV-cache sizing, CUDA graph
  capture, correctness/warmup, or formal measurement.
- No startup/JIT/autotuning event overlaps a formal measurement window.
- No `/metrics` request appears in any log.
- TRTLLM5 parity is evaluated at its reference concurrency and across the
  saturation shape; a discrepancy is investigated and rerun rather than
  silently accepted.

The full no-MTP HT sweep is estimated at roughly two hours including per-point
warmups. Slurm does not permit increasing the QoS/time limit after a job has
started, so two additional reference-style tail jobs cover
`1536,1792,2048,2304` for no-EPLB and Static528. They are part of the same
matrix, not new parameter exploration. If the full jobs also finish, the
duplicate tail is used to quantify cross-domain run-to-run variance.
