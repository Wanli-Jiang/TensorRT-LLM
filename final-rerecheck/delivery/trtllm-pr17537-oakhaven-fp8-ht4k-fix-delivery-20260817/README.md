<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Oakhaven FP8 no-MTP / MTP3 high-throughput >4k delivery

Generated: `2026-08-17`; corrected and branch-integrated: `2026-08-19`
(`America/Los_Angeles`)

This is a source/configuration delivery for reproducing the audited Oakhaven-Max FP8
high-throughput result on another GB300 cluster. The preferred source is the delivery branch
`user/williamj/oakhaven-fp8-ht4k-repro-20260819` in
`https://github.com/Wanli-Jiang/TensorRT-LLM`. It contains the tested PR-stack base
`9a6889b2a2aba6f6e44483999dd972bc157c297b` and the hybrid cache-manager
`max_num_tokens` propagation commit `f572594361`.

The archive does **not** contain model weights, a TensorRT-LLM wheel, a Docker archive, or a `.sqsh`.
Build those on the target cluster so that CUDA, Python ABI, architecture, driver, and Pyxis/Enroot
requirements match that cluster. The accepted artifact must be a normal, non-editable wheel/image;
a source overlay is only acceptable for diagnosis.

Start with the repository-level `OAKHAVEN_FP8_HT4K_REPRODUCTION.md`. It is the authoritative
end-to-end procedure; this directory provides portable recipes, Static544 maps, auditors, evidence,
compatibility patches, and a ready-to-use Codex prompt.

## Source contract and compatibility patches

PR #17537 already contains the tested GDN replay source lineage. PR #16464 commit
`ee241d25f4` introduced functional GDN MTP replay; PR #16768 commit `57f2781e4e`
improved low-batch execution, default enablement, and V2 manager/all-layer commit support.
The later wide-value-head tuning commit `d8d10ab354` is also present. Do not cherry-pick
any of them again.

The first two commits form the replay mechanism/V2 stack used by this delivery. The
presence check for `d8d10ab354` verifies exact PR #17537 source provenance only: there is
no isolated A/B proving that wide-head follow-up is required for >4k peak throughput.
For Oakhaven `H=16,HV=128`, its ratio-8 main replay tuning applies only at `N<=4`, not
the C1536 steady-state population near 48 requests/rank. See `TECHNICAL_CLARIFICATION.md`.

The delivery branch already contains the TensorRT-LLM runtime change. Two compatibility/reference
patches remain in this directory:

1. `patches/flashinfer-gdn-t1-static-cache-vs-0.6.16-0.6.17.patch`
   - Optional version-targeted FlashInfer correctness/specialization hardening.
   - Restores concrete T=1 batch/layout identity in both Python and CuTe compile caches.
   - Applies only when the source module SHA-256 is
     `61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61`.
   - The resulting module SHA-256 must be
     `4982b5a9d20d9b18588020ab3e938238c9692ffab6265c3533f4b7cf8309a8fe`.
   - The latest same-allocation production A/B/A measured dynamic GDN at 4049.545 and
     4096.202 tok/s/GPU and static GDN at 4064.498. Static was inside the dynamic bracket and
     0.206% below the dynamic mean. Therefore this patch is not a prerequisite for >4k.
2. `patches/trtllm-pr17537-max-num-tokens-propagation.patch`
   - Configuration/lifecycle correctness fix.
   - Propagates `max_num_tokens` through all hybrid Mamba/GDN cache-manager construction paths,
     including mixed-manager and PP ranks without local Mamba layers.
   - Already committed as `f572594361` on the delivery branch. Apply it only to a compatible
     older checkout that still stops at `9a6889b2a2`.
   - It is not independently credited with the >4k gain, but prevents the configured token
     geometry from being silently replaced by the default.

`patches/flashinfer-baad0dca-full-commit-reference.patch` is the original FlashInfer commit with its
GPU regression test. It is included for review/provenance. The version-targeted patch above is the
one to apply to the audited 0.6.16/0.6.17 GDN source identity.

## Why throughput recovered

The stable recovery is primarily a state-capacity, residency, population-ramp, and allocator-headroom
result. The mechanisms are:

1. GDN cached replay:
   - legacy verify scratch: `115.410 GiB/GPU`;
   - replay scratch: `11.388 GiB/GPU`;
   - saving: `104.022 GiB/GPU` (`90.133%`).
2. Request residency:
   - C1536 / ADP32 = 48 requests/rank;
   - M36 forces 384 requests into a second wave;
   - same-allocation M36/M48 result: `3971.592 -> 4120.374 tok/s/GPU` (`+3.746%`).
3. No-chunk 8192-token prefill:
   - avoids forcing each prompt through at least eight 1024-token scheduling/collective stages;
   - same-allocation directional controls improved total throughput by 4.217% to 9.504%, while
     TPOT did not improve, matching a faster population-ramp signature.
4. Stable KV/transient budgeting:
   - MTP3 uses KV fraction `0.88` plus absolute `max_tokens=479232`;
   - this represents 52 average 9216-token sequences/rank, above the required 48;
   - it preserves headroom for DeepGEMM MoE output, CUDA graph pools, communication
     workspace, FlashInfer low-M temporaries, and other serving transients;
   - the observed 830–838 MiB failed allocation was the DeepGEMM MoE
     `triton_fused_gather_finalize` output tensor, not proven Split-K workspace.

FlashInfer T=1 static specialization is a separate local kernel change. It is bit-exact and reduces
one exact-shape GB300 graph-replayed kernel by about 4.1 microseconds, but the latest same-allocation
production A/B/A did not resolve a positive E2E effect beyond normal trajectory noise. Do not use
the older cross-allocation `3784.483 -> 4029.055` pair as a single-variable causal comparison.

`TRTLLM_LOW_M_GEMM_BACKEND=auto` enables a separate FlashInfer BF16 `mm_bf16` path
whose internal heuristic selects direct or Split-K for eligible `M<=32` Linear shapes.
At matched memory settings, this low-M `auto` route was 1.470% faster than `off`, but
low-M-off M104 runs also exceeded 4k. It is beneficial, not required for >4k.

## Audited measurement contract

The >4k numbers mean total input-plus-output token throughput divided by 32 GPUs:

| Field | Value |
|---|---|
| Hardware | 32 × GB300, 8 nodes × 4 GPUs |
| Parallelism | TP32 / EP32 / ADP32 / PP1 |
| MoE | TensorRT-LLM DeepGEMM, Static544 EPLB |
| Checkpoint | `oakhaven-max-final-fp8_vv3`, FP8 only |
| KV/state | FP8 attention KV, BF16 Mamba/GDN state |
| Workload | exact ISL=8192, OSL=1024, RR=1.0, request rate=`inf` |
| no-MTP point | C3264, observed `4055.616 total tok/s/GPU` |
| MTP3 point | C1536, four allocations `4114.062/4110.993/4112.350/4065.823` |
| MTP3 mean | `4100.807 total tok/s/GPU`, population CV `0.493%` |
| MTP acceptance | draft=3, forced accepted drafts=2.3, total accept length=3.3 |

This delivery does not claim the same number for RR=0.8, a different ISL/OSL distribution, another
GPU generation, another checkpoint, NVFP4, output-only TPS, or a different GPU denominator.

## Quick start

1. Checkout the delivery branch and verify this bundle:

   ```bash
   git clone --recursive https://github.com/Wanli-Jiang/TensorRT-LLM.git
   cd TensorRT-LLM
   git fetch origin user/williamj/oakhaven-fp8-ht4k-repro-20260819
   git switch --detach origin/user/williamj/oakhaven-fp8-ht4k-repro-20260819
   DELIVERY_ROOT="$PWD/final-rerecheck/delivery/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817"
   "$DELIVERY_ROOT/scripts/verify_bundle.sh"
   ```

2. Do not reapply the TensorRT-LLM patch on the delivery branch. If an older compatible checkout is
   required, use the version-guarded patch after reviewing the diff. Apply the FlashInfer patch only
   when its optional specialization/correctness behavior is explicitly desired and the guarded
   source hash matches.

3. Build a normal TensorRT-LLM wheel and FlashInfer package, then build a
   self-contained Docker image and `.sqsh`. Do not use `pip install -e`, `.pth`, a host `PYTHONPATH`,
   or source mounts for the accepted artifact. See `BUILD_AND_PORTING.md`.

4. Render the target-cluster recipes outside the srt-slurm repository:

   ```bash
   python3 scripts/render_recipe.py \
     configs/mtp3-c1536-m48-kvcap479232.yaml.in \
     rendered/mtp3-c1536.yaml \
     --model-path /shared/models/oakhaven-max-final-fp8_vv3 \
     --container-image /shared/images/trtllm-pr17537-gdnfix.sqsh \
     --delivery-root "$PWD" \
     --slurm-account ACCOUNT \
     --slurm-partition PARTITION \
     --slurm-qos QOS \
     --gpu-type gb300
   ```

5. Run artifact independence, correctness smoke, MTP3 C1536, and no-MTP C3264 in that order.
   Follow `ACCEPTANCE_CHECKLIST.md`. Do not request `/metrics` during warmup or formal measurement.

## Important operating details

- MTP3 C1536 uses `max_batch_size=48`, no chunked prefill, `max_num_tokens=8512`, KV fraction 0.88,
  and absolute KV `max_tokens=479232`.
- no-MTP C3264 uses `max_batch_size=104`, no chunked prefill, `max_num_tokens=8448`, and KV fraction
  0.92. C3328 is a rejected capacity boundary that OOMed.
- Keep `TRTLLM_LOW_M_GEMM_BACKEND=auto` for the measured optional FlashInfer BF16
  direct/Split-K heuristic, and reserve the configured 2304 MiB MoE A2A workspace.
- MTP3 enables `TRTLLM_USE_GDN_REPLAY=1`; no-MTP explicitly sets it to `0`.
- The no-MTP and MTP3 Static544 maps are different and must not be exchanged.
- First-start T=1 compilation increases because batches/layouts are now correctly specialized.
  Finish JIT/tuning and CUDA graph capture before formal timing.
- Use one warmup population (`num_warmup_mult=1`) and three formal populations
  (`num_prompts_mult=3`). Do not change these multipliers when comparing with the retained result.
- Loading heartbeat may run only while loading. Stop it before KV initialization, tuning, graph
  capture, correctness, warmup, and measurement.
- Never call `/metrics` in a formal job. Keep response-level performance metrics disabled.

## Bundle layout

```text
patches/      source patches and original FlashInfer commit reference
configs/      portable MTP3/no-MTP recipes and exact Static544 maps
scripts/      patch, identity, recipe-render, microbenchmark, and audit helpers
evidence/     full root-cause reports, causal JSON, and 45-point FP8 curve
reference/    fixed FlashInfer GDN module for byte-level identity comparison
TECHNICAL_CLARIFICATION.md  replay lineage, wide-head scope, low-M/DeepGEMM boundaries
PROMPT.md     ready-to-use handoff prompt for the target-cluster agent
VALIDATION_20260819.md  template, schema, preflight, dry-run, and repository-guard checks
```

Use `SHA256SUMS` and `MANIFEST.json` as the transfer identity. Verify the copied archive again on the
target cluster before applying anything.
