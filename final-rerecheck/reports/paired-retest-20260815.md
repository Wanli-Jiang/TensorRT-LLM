<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# FP8 high-throughput paired retest, 2026-08-15

## Conclusion

The current stack reproduces the no-MTP RR=1.0 result and exceeds 4000 total
tokens/s/GPU in the formal sweep. Job `472461` reaches `4055.616` total
tokens/s/GPU at C3264. All four points are within `+0.142%` to `+0.214%` of
the earlier accepted current-stack sweep `431988`.

MTP3 is also healthy when the allocation is healthy. Cross-chassis job
`472831` on d038 reaches `3860.044` total tokens/s/GPU at C2304, `+0.191%`
versus accepted current-stack retest `436026`. Its three-point arithmetic
mean is `1.828%` below `436026`, with the largest point delta at C1920
(`-4.481%`), but it remains in the established healthy MTP3 band and is not
the approximately 2.8k failure regime.

The paired d105 allocation gives a useful failure discriminator. no-MTP job
`472461` is normal on the exact d105-T01..T08 nodes, while MTP3 job `472459`
on the same nodes is only `2770.814`--`2792.892` tokens/s/GPU. Moving the
otherwise identical MTP3 arm to d038 recovers `+34.271%`, `+32.434%`, and
`+39.311%` at C1536, C1920, and C2304. Therefore the d105 observation is not
a stack-wide compute regression, an RR=1.0 regression, a bad output artifact,
or a metrics/JIT contamination artifact. It is an allocation-sensitive
failure that disproportionately affects the MTP small-token/high-iteration
path. The available logs do not identify a unique lower-level hardware or
fabric root cause.

## Fixed benchmark contract

- Model: `oakhaven-max-final-fp8_vv3`, FP8.
- Image: `trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh`.
- Image SHA256:
  `08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59`.
- Runtime: TensorRT-LLM `1.3.0rc25`, FlashInfer
  `0.6.17.dev20260806`; packaged installation only, no host source overlay.
- Topology: 32 GB300 GPUs, TP32 / EP32 / ADP32 / PP1.
- MoE: DeepGEMM and Static544 expert placement.
- Workload: exact ISL 8192, exact OSL 1024, RR=1.0, unlimited formal request
  rate, one warmup wave followed by three formal waves.
- MTP3: `max_draft_len=3`, forced accepted draft tokens `2.3`, effective
  accept length approximately `3.3`, GDN replay off, FlashInfer low-M BF16
  direct/Split-K-heuristic backend
  off.
- Observer controls: `return_perf_metrics=false`, iteration perf/request
  stats off, iteration log off, and no request to `/metrics`.

The no-MTP Static544 map has 92 layers and SHA256
`c4e7c0d57311629e32999c7b4f1f43144328134f15059c7f01ba7b03d7ddd68f`.
The MTP3 map has 93 layers and SHA256
`ce92f6484658759845cc104e8338c15b2a62fe95719e04b2b02ae5880f71e50a`.

## Accepted no-MTP retest

- Slurm job: `472461`, `COMPLETED`, exit `0:0`.
- Nodes: `nvl72d105-T[01-08]`.
- Elapsed: `01:27:48`.

| C | Requests | Total tok/s/GPU | Aggregate total tok/s | Duration | Median TTFT | Median TPOT | Delta vs 431988 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2944 | 8832/8832 | 3971.108 | 127075.453 | 640.531 s | 6202.983 ms | 200.771 ms | +0.162% |
| 3072 | 9216/9216 | 4010.963 | 128350.802 | 661.738 s | 4783.205 ms | 208.011 ms | +0.214% |
| 3136 | 9408/9408 | 3996.110 | 127875.523 | 678.035 s | 4776.092 ms | 213.841 ms | +0.142% |
| 3264 | 9792/9792 | 4055.616 | 129779.713 | 695.356 s | 9551.214 ms | 213.860 ms | +0.148% |

The four-point mean is `4008.449` tokens/s/GPU, `+0.166%` versus job
`431988`. The formal C3264 point, not only its warmup, exceeds 4000.

## Accepted MTP3 retest

- Slurm job: `472831`, `COMPLETED`, exit `0:0`.
- Nodes: `nvl72d038-T[01-08]`.
- Elapsed: `00:52:01`.

| C | Requests | Total tok/s/GPU | Aggregate total tok/s | Duration | Median TTFT | Median TPOT | Delta vs 436026 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536 | 4608/4608 | 3750.030 | 120000.975 | 353.892 s | 35412.462 ms | 74.717 ms | -1.188% |
| 1920 | 5760/5760 | 3674.565 | 117586.092 | 451.449 s | 64421.824 ms | 75.941 ms | -4.481% |
| 2304 | 6912/6912 | 3860.044 | 123521.393 | 515.708 s | 89929.048 ms | 74.077 ms | +0.191% |

The three-point mean is `3761.546` tokens/s/GPU. At C2304 it also matches
TRTLLM6 production repeat `422798` (`3838.946`) within `+0.550%`.

## Same-d105 MTP3 failure control

Job `472459` used the same d105-T01..T08 node set as the accepted no-MTP arm.
It completed cleanly but reproduced the allocation-sensitive MTP failure:

| C | d105 job 472459 | d038 job 472831 | Recovery on d038 |
|---:|---:|---:|---:|
| 1536 | 2792.892 | 3750.030 | +34.271% |
| 1920 | 2774.645 | 3674.565 | +32.434% |
| 2304 | 2770.814 | 3860.044 | +39.311% |

The formal d105 loss was already present in its C1536 warmup, so it did not
begin at a later formal point. JIT/autotune and CUDA graph setup completed
before serving on both placements. The same benign UCX rail-0 bind warning is
present in the slow d105 and healthy d038 logs, so that warning is not a useful
failure discriminator. GPU pre/post snapshots also show the same P0 state,
1400 W power limit, 2070 MHz maximum clock, and zero residual GPU memory.

This result mirrors the earlier recurrence: MTP3 job `431989` on d096 ran at
`2810`--`2856` tokens/s/GPU, while the same effective contract in job `436026`
on d140 recovered to `3795`--`3853`. A practical acceptance gate should
therefore reject an MTP3 allocation when the one-wave C1536 warmup is far
below the approximately 110k aggregate healthy band, then re-run the same
immutable arm on a different chassis before assigning a code regression.

## Integrity and contamination audit

All accepted formal JSONs and the d105 failure-control JSONs passed the same
strict checks:

- exactly `3 * concurrency` successful requests;
- every input length exactly 8192 and every output length exactly 1024;
- zero request errors and zero empty generated texts;
- token totals reconcile;
- zero fatal runtime signatures;
- zero `/metrics` requests.

For both accepted jobs, the loading-only heartbeat stopped automatically
during safetensors loading at approximately 70%, before KV-cache setup,
FlashInfer/JIT work, CUDA graph capture, warmup, and formal measurement. The
heartbeat supervisor was not alive during measurement. Both jobs produced
clean GPU post snapshots. The srt-slurm repository guard passed after both
runs: its HEAD and complete inherited worktree status are unchanged.

## Artifact map

- no-MTP recipe:
  `final-rerecheck/recipes/perf/paired-retest-20260815/nomtp-static544-rr10.yaml`
- MTP3 recipe:
  `final-rerecheck/recipes/perf/paired-retest-20260815/mtp3-static544-rr10.yaml`
- no-MTP result and full audit:
  `final-rerecheck/outputs/paired-retest-20260815/nomtp/472461`
- healthy d038 MTP3 result and full audit:
  `final-rerecheck/outputs/paired-retest-20260815/mtp3/472831`
- d105 MTP3 failure-control result and full audit:
  `final-rerecheck/outputs/paired-retest-20260815/mtp3/472459`
- allocation-level provenance, image identity, GPU snapshots, and hashes:
  `final-rerecheck/outputs/paired-retest-20260815/job-472461-nomtp` and
  `final-rerecheck/outputs/paired-retest-20260815/job-472831-mtp3`
- Slurm scripts:
  `final-rerecheck/scripts/run_retest_nomtp_20260815_d105.sbatch` and
  `final-rerecheck/scripts/run_retest_mtp3_20260815_d038.sbatch`

No benchmark job from this retest remains active.
