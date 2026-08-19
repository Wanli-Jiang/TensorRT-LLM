<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Acceptance checklist

Declare these gates before looking at target-cluster performance.

## Source and artifact gates

- [ ] TensorRT-LLM contains PR #17537 head `9a6889b2a2`.
- [ ] Functional/V2 replay commits `ee241d25f4` and `57f2781e4e` are ancestors.
- [ ] `d8d10ab354` is an ancestor when exact PR #17537 provenance is required; this gate does
      not classify its wide-head tuning as necessary for >4k.
- [ ] Delivery branch contains propagation commit `f572594361`; an older checkout has the
      equivalent compatibility patch applied and its focused unit tests pass.
- [ ] FlashInfer source/package identity is recorded. Dynamic GDN is accepted for >4k reproduction;
      if the optional static patch is used, its installed module hash is `4982b5...0a8fe`.
- [ ] Wheel is non-editable and image/sqsh runs without host source or `PYTHONPATH`.
- [ ] Docker and sqsh identities, sizes, and SHA-256 values are retained.

## Kernel/correctness gates

- [ ] Exact production-shape GDN test covers B=36,T=1,H=16,HV=128,K=V=128.
- [ ] Descending CUDA-graph ladder covers B=24/16/4/2/1, packed views, and padded state pool.
- [ ] Output/state agree with the trusted path; guard rows remain unchanged.
- [ ] Warm CUDA-graph latency is measured after JIT and graph capture.
- [ ] Semantic serving smoke has nonempty, parseable outputs and zero request errors.

Retained kernel reference: static `0.0512040 ms` versus dynamic `0.0553032 ms` at one exact GB300
shape. Treat this as platform/shape evidence, not a cross-cluster acceptance threshold. The latest
production A/B/A did not isolate a positive static-GDN E2E effect, and dynamic GDN itself exceeded
4k twice. A missing microbenchmark speedup does not fail the high-throughput reproduction.

## Formal request gates

For every accepted point:

- [ ] `completed == num_prompts`.
- [ ] error list is empty and every generated text is nonempty.
- [ ] every actual input length is exactly 8192.
- [ ] every actual output length is exactly 1024.
- [ ] total tokens and `total tok/s/GPU` are recomputed from raw duration and 32 GPUs.
- [ ] metric is input+output throughput, not output-only throughput.
- [ ] formal interval contains no `/metrics`, heartbeat, health probe, JIT/tuning, graph capture,
      OOM, traceback, CUDA/NCCL/MPI fatal signature, or silent fallback.

## Configuration gates

- [ ] TP32/EP32/ADP32/PP1, DeepGEMM, Static544, FP8 checkpoint/KV are unchanged.
- [ ] RR=1.0, request rate `inf`, ISL/OSL 8192/1024 are explicit.
- [ ] MTP3 uses draft=3, forced accepted drafts=2.3, total accept length=3.3.
- [ ] MTP3 uses M48, T8512, no chunking, replay on, FlashInfer low-M BF16 auto, KV 0.88
      + cap 479232.
- [ ] no-MTP uses M104, T8448, no chunking, replay off, FlashInfer low-M BF16 auto,
      KV 0.92.
- [ ] MTP3 and no-MTP use their respective placement maps.
- [ ] warmup population is `1 × concurrency`; formal population is `3 × concurrency`.

## Performance/stability decision

- [ ] MTP3 C1536 reaches at least 4000 total tok/s/GPU.
- [ ] MTP3 is repeated in three independent allocations; report all values, mean, median, range,
      and population CV. Target CV is at most 1.5%.
- [ ] no-MTP C3264 reaches at least 4000 total tok/s/GPU in the selected-point qualification.
- [ ] C3328 no-MTP remains labeled a rejected capacity boundary; do not publish it as a point.

If throughput is below threshold, first classify the signature:

| Signature | First checks |
|---|---|
| TPOT and exact-shape kernel both regress | GDN source/hash, cache key, cubin identity, tactic |
| TTFT/makespan regress while TPOT is flat/better | resident slots, request waves, chunked prefill |
| DeepGEMM gather/finalize output OOM after ready | KV over-provisioning, graph pools, communication/workspace, fragmentation |
| First run only is slow | JIT/tuning/graph capture leaked into formal interval |
| About 1% cross-allocation movement | routing trajectory and ordinary cluster noise |
| GDN microbenchmark is flat but E2E exceeds 4k | acceptable; static T=1 is not a >4k prerequisite |

Do not label an allocation as "Split-K OOM" unless the traceback identifies a Split-K
workspace or reduction allocation. The retained 830--838 MiB failures are DeepGEMM MoE
gather/finalize output allocations. `auto` means FlashInfer direct/Split-K is available;
it does not prove which tactic each shape selected.

Do not attribute a failure to node restart/cluster quality until same-allocation A/B, exact-shape
microbenchmarks, identity checks, memory geometry, and request invariants have been exhausted.
