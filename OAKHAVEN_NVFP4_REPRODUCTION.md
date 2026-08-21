<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Oakhaven-Max NVFP4 完整结果与跨集群复现入口

更新时间：2026-08-20 18:45:59 PDT（America/Los_Angeles）

本文是 Oakhaven-Max NVFP4 验证的主入口，同时面向开发者和 Codex agent。它冻结实验身份、
完整数据、配置、关键日志、失败边界和跨集群复现流程。历史证据目录保持只读；新的复现必须写入
当前 TensorRT-LLM checkout 下的全新时间戳实验目录。

## 0. 给 Codex 的执行合同

当用户要求“读取本文并启动 NVFP4 复现”时，按以下合同执行：

1. 完整阅读仓库 `AGENTS.md`、本文、本文链接的最终分析和目标集群 srt-slurm 合同。
2. 使用 `run-trtllm-srt-slurm-benchmarks`；提交前用 `check-slurm-resources` 确认当前 account、
   partition、QoS 和 GB300 容量。只有目标集群存在合法 idle-GPU reaper 风险时，才使用
   `run-slurm-loading-heartbeat`，并保证它在 KV 初始化、JIT、graph capture、warmup 和正式测量前停止。
3. 不修改 TensorRT-LLM、FlashInfer 或 srt-slurm 源码来“追平”结果。当前 branch 已包含所需
   TensorRT-LLM 修复；若出现差异，先归因 image、模型、拓扑、recipe、cache/JIT 和集群。
4. 不覆盖本目录历史 outputs，不直接运行其中硬编码原集群路径的 submit 脚本。复制 recipe、map、
   runner 和 auditor 到 `<TRTLLM_REPO>/experiments/oakhaven-nvfp4-<timestamp>`，只修改部署路径与
   Slurm 字段，保留 engine 和 measurement contract。
5. 正式性能 job 的整个 server 生命周期禁止请求 JSON `/metrics`。保持
   `return_perf_metrics=false`、iteration/request stats=false、`print_iter_log=false`。
6. 先做 image/source identity 和一个非测量 correctness smoke，再跑 headline points；通过后才扩展
   full curves、KV V2 或 disaggregate。每个 job 监控到 Slurm terminal state。
7. 只接受 planned requests 全部完成、ISL/OSL 精确、errors=0、GPU denominator 正确、warmup/JIT
   不污染正式窗口且 `/metrics` hits=0 的数据。失败或污染结果保留，但不得进入性能曲线。
8. 结果与本文比较时区分：同 image 精确复现、从本 branch 重建的同源码复现，以及仅同配置复现。
   不允许把三者混写成 bitwise parity。

如果用户只是要求阅读或审计本文，不要提交 Slurm job。只有用户明确要求启动/复现时才分配资源。

## 1. 结论摘要

这里的 `total tok/s/GPU` 始终是：

```text
(全部 input tokens + 全部 output tokens) / formal duration / 全部 serving GPUs
```

它不是 output-only TPS，也不是单用户 decode TPS。

| 场景 | 推荐配置/点 | 可信结果 |
|---|---|---:|
| Aggregate LL no-MTP | TP8/EP1、TRTLLM MoE、C1-C512 | 峰值 2411.78 total tok/s/GPU |
| Aggregate LL MTP3 | TP8/EP1、TRTLLM MoE、C1-C512 | 峰值 3073.54；C512 为 3071.50 |
| Aggregate HT no-MTP baseline | ADP16/EP16、CUTEDSL、Static528、budget8448 | C2048 为 5506.88 |
| Aggregate HT no-MTP maximum | Static528、chunk、budget16896、C2048 | 6096.89；独立复测 6062.45 |
| Aggregate HT MTP3 baseline | ADP16/EP16、CUTEDSL、budget1024/8192 | 约 2.8k 平台 |
| Aggregate HT MTP3 maximum | Static528、chunk、budget33792、C512-C1024 | 稳定约 5.7k-5.9k；最高观察值 6036.81 |
| Disaggregate no-MTP | 3×CTX8 + 1×GEN16、40 GPU、C1536 | 5341.00，较 TRTLLM5 +0.363% |
| Disaggregate MTP3 | 4×CTX8 + 1×GEN16、48 GPU、C768 | 5298.25，较 TRTLLM5 +0.030% |

最重要的因果结论：**no-chunk 假设被否定。** 最大吞吐来自保留 chunked prefill，并提高同步的
scheduler/MoE token budget，再与正确的 Static528 placement 组合。MTP3 在相同 budget8448 下
关闭 chunking，C384-C1024 全部回退 5.50%-9.11%。

## 2. Branch、源码与 image 身份

| 项目 | 冻结值 |
|---|---|
| 分享 fork | `https://github.com/Wanli-Jiang/TensorRT-LLM` |
| 分享 branch | `user/williamj/oakhaven-fp8-ht4k-repro-20260819` |
| 验证时 branch HEAD | `e49b205b0423ec611a5b1062423f9598674ba0e8` |
| 性能栈基线 | `9a6889b2a2aba6f6e44483999dd972bc157c297b` |
| Hybrid KV token-limit 修复 | `f572594361`，已在分享 branch 中 |
| TensorRT-LLM version | `1.3.0rc25` |
| 历史运行模式 | image-contained、普通非 editable wheel、无 source overlay |
| srt-slurm HEAD | `922f005de4674cd51cfdc6f6b361ad07b1893014`，本轮未修改 |

历史 NVFP4 job 实际使用：

```text
/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/containers/
trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh
SHA256 08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59
size   35,313,352,704 bytes
```

该 image 的 TensorRT-LLM runtime 是 `9a6889b2a2` 加一份三文件 worktree patch；该 patch 后来
正式提交为 `f572594361`。当前 branch 已包含它，且从 `f572594361` 到验证 HEAD 的后续变化只有
文档/交付产物，没有新的运行时代码。NVFP4 campaign 期间没有新增 NVFP4 专用源码 patch。

image 还包含固定的 FlashInfer GDN T=1 payload。它不是 NVFP4 campaign 期间新增的改动，也没有
NVFP4 matched A/B 证明它是吞吐前提。要求 bitwise 环境时使用原 image；从当前 branch 重建时，
必须记录新的 FlashInfer、wheel、native library 和 image hash，不能宣称与历史 image bitwise 相同。

完整 source 说明见：

- [SOURCE_PROVENANCE.md](final-rerecheck/delivery/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817/SOURCE_PROVENANCE.md)
- [NVFP4 reference lock](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/PLAN_AND_REFERENCE_LOCK.md)

## 3. 模型、硬件与测量合同

| 字段 | 值 |
|---|---|
| Checkpoint | `oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean` |
| 原集群模型路径 | `/lustre/fsw/portfolios/coreai/users/williamj/models/oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean` |
| `config.json` SHA-256 | `3b0153fad68686977da18f663455c4164499d0cf752d0767d19e2b46090b34e9` |
| Checkpoint precision | NVFP4 |
| Attention KV cache | FP8 |
| Mamba/GDN state | BF16 |
| GPU | GB300，4 GPU/node，要求单 NVL72 domain |
| Workload | random，精确 ISL8192、OSL1024、RR=1.0、request rate=`inf` |
| Aggregate LL sampling | 1×C warmup + 5×C formal，HTTP connection reuse=false |
| Aggregate HT sampling | 1×C warmup + 3×C formal，HTTP connection reuse=true |
| Block reuse | disabled |
| Metrics | response perf、iteration/request stats、iteration log 全关；禁止 `/metrics` |

跨集群结果只有在 GPU generation、节点/NVL domain、模型 hash、ISL/OSL/RR、采样倍数、MTP
acceptance、GPU denominator 和关键 engine 配置一致时，才能与本文直接比较。

## 4. 冻结配置

### 4.1 Aggregate low latency

| 字段 | no-MTP | MTP3 |
|---|---:|---:|
| Nodes / GPUs | 2 / 8 | 2 / 8 |
| TP / EP / PP | 8 / 1 / 1 | 8 / 1 / 1 |
| Attention DP | off | off |
| MoE backend | TRTLLM | TRTLLM |
| `max_batch_size` | 128 | 128 |
| scheduler/MoE `max_num_tokens` | 8448 | 8704 |
| `max_seq_len` | 9472 | 9472 |
| chunked prefill | off | off |
| KV manager | V1 | V1 |
| KV fraction | 0.74 | 0.70 |
| GDN replay | off | on |
| MTP | disabled | draft=3，forced accepted drafts=2.3 |
| Concurrency | 1,2,4,8,16,32,64,128,256,512 | 相同 |

权威 recipes：

- [LL no-MTP](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/recipes/aggregate-ll-nomtp-tp8-trtllm.yaml)，SHA-256 `85dd89dfa8113c187b804bb6eceaa8712e20585ad2c24a923395a0f31b9de7cd`。
- [LL MTP3](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/recipes/aggregate-ll-mtp3-tp8-trtllm.yaml)，SHA-256 `274438bf3eb49d09165f85732fc847631339ec9a48e5755edde7d1c4503914da`。

### 4.2 Aggregate high-throughput 公共设置

```text
nodes/GPUs                    4 / 16
TP/EP/ADP/PP                  16/16/16/1
attention DP                  enabled
MoE backend                   CUTEDSL
MoE communication             NVLINK_ONE_SIDED
dispatch/combine block        256 / 256
MoE A2A workspace             2304 MiB
KV manager / dtype            V1 / FP8
KV fraction                   0.92
block reuse                   false
attention_dp balance          enabled, batching_wait_iters=0, timeout_iters=60
NVFP4 GEMM allowed backends   cutlass,cublaslt,cutedsl,cuda_core
```

原始 TRTLLM5-parity baseline：

| 字段 | no-MTP | MTP3 |
|---|---:|---:|
| `max_batch_size` | 128 | 32 |
| scheduler `max_num_tokens` | 8448 | 1024 |
| MoE `max_num_tokens` | 8448 | 8192 |
| chunked prefill | off | on |
| GDN replay | off | on |
| MTP | disabled | draft=3，forced drafts=2.3 |
| Placement arms | no-EPLB / Static528 | no-EPLB / Static528 |

最终最大吞吐配置：

| 字段 | no-MTP | MTP3 |
|---|---:|---:|
| `max_batch_size` | 128 | 32 |
| scheduler/MoE `max_num_tokens` | 16896 / 16896 | 33792 / 33792 |
| chunked prefill | on | on |
| Placement | Static528 | Static528 |
| GDN replay | off | on |
| 推荐 concurrency | 2048 | C512-C1024；关注 tail 时 C512-C640 |

最大吞吐 recipes：

- [no-MTP budget16896 curve](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/recipes/nomtp-budget-confirmation/chunk-budget16896-curve.yaml)，SHA-256 `7c01291e47eaf44b968bc249ef709ba9aba26f821b42a9ccd1f23a5dcfa808c7`。
- [MTP3 budget33792 curve](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/recipes/mtp3-budget33792-curve/static528-budget33792-curve.yaml)，SHA-256 `67d0f3262b3903143b8cc9d22138bc10196a7ecc6e4c1a97b178f2129462c290`。

Static maps：

| Mode | 文件 | SHA-256 |
|---|---|---|
| no-MTP | `nomtp-static528.yaml` | `68ef80e1b996cc900f29c859221155bf6a3b7e7c1e5c83faf476d1c12b2f88be` |
| MTP3 | `mtp3-static528.yaml` | `cbc575a8d0bf221a76d3a04552a123d27964b788dee233be4eced6b6a71b30dc` |

两份 map 也已跟踪在 `oakhaven-kernels` commit `60c84765ed54ab51aff25b5f21e339f03c870b64`
的 `config/trtllm/agg/oakhaven-max/nvfp4-dev/eplb/` 下，且与本实验副本 byte-identical。若分享 branch
不携带历史实验目录，目标集群应从该 commit 获取 map，并以上述 SHA-256 作为最终身份。

### 4.3 Disaggregate E2E

| 字段 | no-MTP | MTP3 |
|---|---|---|
| Topology | 3×CTX8-ADP8/EP8 + 1×GEN16-ADP16/EP16 | 4×CTX8-ADP8/EP8 + 1×GEN16-ADP16/EP16 |
| Total GPUs | 40 | 48 |
| Concurrency / rounds | 1536 / 24 | 768 / 12 |
| EPLB | no-EPLB | no-EPLB |
| MoE | CUTEDSL | CUTEDSL |
| KV transfer | NIXL Python runtime | NIXL Python runtime |
| MTP | disabled | draft=3，forced drafts=2.3 |

Recipes：

- [disagg no-MTP](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/recipes/disaggregate/best-noeplb-nomtp-client-oversample-retry2.yaml)，SHA-256 `fbb712a4957c5a02a080f257b583aa6c96ef6ca95145aa2e03c500f4355657fe`。
- [disagg MTP3](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/recipes/disaggregate/best-noeplb-mtp3-client-oversample-retry2.yaml)，SHA-256 `60d8313207e1156b4250049b754e1b641209dfdef99f7a192494c1cec0ef5c9b`。

Disaggregate 使用 4096-record 固定数据集，SHA-256：
`3ce073633c4a6aca0d50653d52383a3ff798245f32106913864c36e13c454b01`。

## 5. 完整数据点

以下表格为便于阅读保留两位小数；机器可读 JSON 保留完整精度：

- [aggregate V1 full audit](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/aggregate-v1-campaign-audit.json)
- [KV manager V2 audit](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/aggregate-kv-manager-v2-diagnostics.json)
- [maximum-throughput summary](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/nvfp4-final-aggregate-disagg-and-nochunk-max-summary.json)

### 5.1 Low latency V1 full curves

| C | no-MTP total TPS/GPU | no-MTP TTFT ms | no-MTP TPOT ms | MTP3 total TPS/GPU | MTP3 TTFT ms | MTP3 TPOT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 147.49 | 301.46 | 7.31 | 318.11 | 310.45 | 3.23 |
| 2 | 269.06 | 504.48 | 7.83 | 533.53 | 519.59 | 3.70 |
| 4 | 450.22 | 851.68 | 9.17 | 847.86 | 798.63 | 4.49 |
| 8 | 702.27 | 1260.97 | 11.55 | 1242.63 | 828.12 | 6.37 |
| 16 | 1083.68 | 1338.08 | 15.25 | 1718.53 | 1111.00 | 9.41 |
| 32 | 1507.85 | 1331.80 | 22.54 | 2217.75 | 1394.27 | 14.74 |
| 64 | 1982.38 | 1090.43 | 34.96 | 2611.12 | 1425.86 | 25.55 |
| 128 | 2403.22 | 1135.58 | 58.46 | 3073.54 | 1497.11 | 44.55 |
| 256 | 2411.65 | 61488.38 | 58.99 | 3010.95 | 48672.01 | 45.80 |
| 512 | 2411.78 | 183498.89 | 58.96 | 3071.50 | 144140.91 | 45.68 |

LL 的 C256/C512 total throughput 已饱和，但排队使 TTFT 急剧增加；它们不是低 tail-latency 推荐点。

### 5.2 Aggregate HT 原始 no-EPLB / Static528 curves

| C | no-MTP no-EPLB | no-MTP Static528 | MTP3 no-EPLB | MTP3 Static528 |
|---:|---:|---:|---:|---:|
| 1 | 22.90 | 22.06 | 54.14 | 52.61 |
| 2 | 42.52 | 41.92 | 103.01 | 102.25 |
| 4 | 82.75 | 81.00 | 204.50 | 195.66 |
| 8 | 162.81 | 160.41 | 395.39 | 386.76 |
| 16 | 319.23 | 309.23 | 758.80 | 760.34 |
| 32 | 541.27 | 536.18 | 1070.49 | 1041.17 |
| 64 | 993.43 | 939.13 | 1592.58 | 1580.74 |
| 128 | 1587.84 | 1543.43 | 2143.12 | 2134.98 |
| 256 | 2564.96 | 2424.65 | 2688.31 | 2590.01 |
| 320 | — | — | 2724.75 | 2693.27 |
| 384 | — | — | 2781.19 | 2814.02 |
| 448 | — | — | 2790.41 | 2772.06 |
| 512 | 3495.06 | 3304.80 | 2839.55 | 2878.78 |
| 640 | — | — | 2848.60 | 2852.49 |
| 768 | 4017.60 | 3733.95 | 2847.69 | 2840.31 |
| 896 | — | — | 2796.18 | 2833.15 |
| 1024 | 4163.35 | 4009.90 | 2875.02 | 2828.86 |
| 1280 | 4451.87 | 4216.39 | — | — |
| 1536 | 4802.82 | 5303.12 | — | — |
| 1792 | 4933.50 | 5479.86 | — | — |
| 2048 | 4984.08 | 5506.88 | — | — |
| 2304 | 4912.90 | 5404.79 | — | — |

这些是原始 baseline configs，不是最终 maximum configs。no-MTP Static528 在高并发改善明显；
MTP3 在旧的 scheduler budget1024 下 Static528 基本中性，不能用它否定较大 budget 下的收益。

### 5.3 no-MTP 最大吞吐搜索

C2048 matched screen：

| Job | Chunk | Scheduler/MoE budget | total TPS/GPU |
|---:|:---:|---:|---:|
| 527441 | off | 8448 | 5486.55 |
| 527442 | on | 8448 | 5494.49 |
| 527443 | on | 16896 | **6096.89** |

独立复测 Job 527727：

| C | 完成请求 | total TPS/GPU | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|
| 1536 | 4608 | 5777.31 | 12579.09 | 135.50 |
| 1792 | 5376 | 5970.15 | 12662.27 | 155.28 |
| 2048 | 6144 | **6062.45** | 14797.99 | 174.86 |
| 2304 | 6912 | 5905.02 | 23126.82 | 182.46 |

Job 527728 在 C2048 把 budget 加到 33792，得到 6030.29，比独立 budget16896 复测低 0.53%。
因此 no-MTP 推荐 16896；C2304 的下降确认 C2048 是可信饱和点。

### 5.4 MTP3 no-chunk 因果 A/B

| C | chunk+budget1024 | chunk+budget8448 | no-chunk+budget8448 | 关闭 chunk 的变化 |
|---:|---:|---:|---:|---:|
| 384 | 2757.64 | 4623.25 | 4368.99 | -5.50% |
| 512 | 2827.43 | 4910.95 | 4463.78 | -9.11% |
| 640 | 2897.29 | 4938.49 | 4499.30 | -8.89% |
| 768 | 2849.78 | 4915.28 | 4621.37 | -5.98% |
| 896 | 2885.98 | 4877.59 | 4564.54 | -6.42% |
| 1024 | 2788.49 | 4886.61 | 4459.48 | -8.74% |

结论：第一阶收益来自 scheduler token capacity，而不是关闭 chunked prefill。

### 5.5 MTP3 maximum search

C640 同日 screen：

| Job | Placement | Budget | total TPS/GPU |
|---:|---|---:|---:|
| 527208 | no-EPLB | 8448 | 4735.09 |
| 527209 | no-EPLB | 16896 | 4935.07 |
| 527210 | no-EPLB | 33792 | 5113.23 |
| 527211 | Static528 | 8448 | 5456.98 |

Static528 + budget33792 独立完整曲线 Job 527804：

| C | 完成请求 | total TPS/GPU | TTFT ms | TPOT ms |
|---:|---:|---:|---:|---:|
| 384 | 1152 | 5501.20 | 12321.46 | 25.33 |
| 512 | 1536 | 5793.83 | 11239.72 | 35.14 |
| 640 | 1920 | 5677.05 | 16207.41 | 43.12 |
| 768 | 2304 | 5525.36 | 29350.86 | 42.02 |
| 896 | 2688 | 5627.11 | 45557.81 | 43.64 |
| 1024 | 3072 | **5874.82** | 55261.86 | 42.31 |

C640 同配方独立 repeats：

| Job | total TPS/GPU |
|---:|---:|
| 527689 | **6036.81** |
| 527804 | 5677.05 |
| 528007 | 5820.72 |

三次最小/中位/均值/最大为 5677.05 / 5820.72 / 5844.86 / 6036.81，max/min 跨度 6.34%。
对外应表述为“稳定约 5.7k-5.9k，最高观察 6036.81”，不能承诺稳定 6k floor。

Job 527805 的 budget67584 在 warmup 只完成 114/640 请求，随后
`moeComputeRouteDevice / moe_load_balance_routing` 报 CUDA unspecified launch failure，
Slurm `FAILED/87:0`，没有正式结果。它是 invalid configuration boundary，不是性能回退点。

### 5.6 KV cache manager V1/V2 A/B

LL 表中为 V2 相对 V1 total TPS/GPU 变化：

| C | no-MTP V2 vs V1 | MTP3 V2 vs V1 |
|---:|---:|---:|
| 1 | -0.16% | -0.42% |
| 2 | +0.69% | -0.52% |
| 4 | +1.59% | +0.48% |
| 8 | +2.77% | -3.48% |
| 16 | -0.30% | +0.11% |
| 32 | +1.05% | +0.82% |
| 64 | -1.28% | +4.87% |
| 128 | -0.09% | +0.54% |
| 256 | -0.25% | +3.47% |
| 512 | -0.19% | +1.77% |

HT no-EPLB：

| Mode | C | V1 TPS/GPU | V2 TPS/GPU | 变化 | 判定 |
|---|---:|---:|---:|---:|---|
| no-MTP | 2048 | 4984.08 | 4961.64 | -0.45% | 中性，但 V2 capacity 较小，存在 confounder |
| no-MTP | 2304 | 4912.90 | 4904.60 | -0.17% | 中性，但两者均越过 full residency |
| MTP3 | 640 | 2848.60 | 2554.84 | -10.31% | 明确回退 |
| MTP3 | 768 | 2847.69 | 2813.63 | -1.20% | 近中性 |
| MTP3 | 1024 | 2875.02 | 2746.06 | -4.49% | 回退 |

因此最大吞吐结果保持 V1。V2 会把 scheduler policy 强制为 `MAX_UTILIZATION`，而且可用 token
capacity 与 V1 不完全相同；没有匹配 capacity 和 residency 时，不能把差异简单归因于 manager 实现。

### 5.7 Disaggregate E2E

| Mode | Job | C | 完成请求 | total TPS/GPU | TTFT ms | TPOT ms | Mean accept length | TRTLLM5 差异 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no-MTP | 526474 | 1536 | 36864 | 5341.00 | 14234.44 | 49.25 | 1.000 | +0.363% |
| MTP3 | 526475 | 768 | 9216 | 5298.25 | 9529.27 | 16.37 | 3.287 | +0.030% |

两项均通过完整性、单 NVL72 domain、heartbeat 和无 `/metrics` 审计。Aggregate 与 disaggregate
GPU denominator 和拓扑不同，绝对 TPS/GPU 不能直接代表部署优劣。

## 6. 关键日志和证据

历史实验根目录：

```text
final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
```

每个 aggregate accepted job 优先检查：

```text
<job>/slurm-job.txt                         Slurm state/resources
<job>/nodes.txt                             allocation/domain
<job>/image-sha256.expected.txt             image identity
<job>/recipe.lock.yaml                      冻结 recipe
<job>/logs/trtllm_config_agg.yaml           实际 server config
<job>/logs/fingerprint_agg_w0.json          TensorRT-LLM runtime fingerprint
<job>/heartbeat.final-status.txt            loading heartbeat boundary
<job>/audit-formal-window.json               formal-window integrity
<job>/logs/benchmark-rollup.json             curve rollup
<job>/logs/sa-bench_isl_8192_osl_1024/*.json raw data points
<job>/logs/*_agg_w0.out                      server/JIT/kernel log
```

关键 job roots：

| 目的 | Job | Repo-relative output root |
|---|---:|---|
| LL no-MTP | 525297 | `outputs/ll-nomtp-v1/525297` |
| LL MTP3 | 525298 | `outputs/ll-mtp3-v1/525298` |
| baseline HT no-MTP no-EPLB tail | 525325 | `outputs/ht-nomtp-noeplb-tail-v1/525325` |
| baseline HT no-MTP Static528 tail | 525326 | `outputs/ht-nomtp-static528-tail-v1/525326` |
| baseline HT MTP3 no-EPLB | 525301 | `outputs/ht-mtp3-noeplb-v1/525301` |
| baseline HT MTP3 Static528 | 525302 | `outputs/ht-mtp3-static528-v1/525302` |
| final no-MTP curve | 527727 | `outputs/nomtp-budget-confirm-chunk-budget16896-curve-v1/527727` |
| final MTP3 curve | 527804 | `outputs/mtp3-record-static528-budget33792-curve-v1/527804` |
| MTP3 C640 repeat | 528007 | `outputs/mtp3-static33792-c640-repeat2-v1/528007` |
| invalid budget67584 | 527805 | `outputs/mtp3-record-static528-budget67584-c640-v1/527805` |

上述路径都相对于历史实验根目录。汇总审计：

- [最终中文分析](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/NVFP4_FINAL_AGGREGATE_DISAGG_AND_NOCHUNK_MAX_THROUGHPUT_ANALYSIS_20260820.md)
- [no-chunk causal A/B](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/mtp3-nochunk-causal-ab.json)
- [disaggregate audit](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/disaggregate-best-client-oversample-retry2-audit.json)
- 完整 raw evidence 包中的 scheduler control-flow：
  `final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/evidence/image-scheduler-chunk-control-flow.txt`。

Disaggregate 关键文件：

```text
outputs/disaggregate/nomtp-best-noeplb-client-oversample-retry2/
  slurm-job.txt, nodes.txt, loading-heartbeat/final-status.txt
  ctx_config.yaml, gen_config.yaml, concurrency_1536/result.json
  start_logs/3_output_CTX_*.log, start_logs/3_output_GEN_0.log, 6_bench.log

outputs/disaggregate/mtp3-best-noeplb-client-oversample-retry2/
  slurm-job.txt, nodes.txt, loading-heartbeat/final-status.txt
  ctx_config.yaml, gen_config.yaml, concurrency_768/result.json
  start_logs/3_output_CTX_*.log, start_logs/3_output_GEN_0.log, 6_bench.log
```

## 7. 跨集群复现方法

### 7.1 Checkout 和身份检查

```bash
git clone --recursive https://github.com/Wanli-Jiang/TensorRT-LLM.git
cd TensorRT-LLM
git fetch origin user/williamj/oakhaven-fp8-ht4k-repro-20260819
git switch --detach origin/user/williamj/oakhaven-fp8-ht4k-repro-20260819

git rev-parse HEAD
git merge-base --is-ancestor 9a6889b2a2aba6f6e44483999dd972bc157c297b HEAD
git merge-base --is-ancestor f572594361 HEAD
git status --short
```

如果 fork 使用其他 remote 名称，替换 `origin`。文档发布产生的后续 docs-only commit 可以使 HEAD
晚于 `e49b205b04`；必须证明上述两个 runtime commits 是 ancestor，并记录最终 HEAD/status。

### 7.2 绑定目标集群路径

```bash
export TRTLLM_REPO="$PWD"
export REFERENCE_ROOT="$TRTLLM_REPO/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820"
export EXPERIMENT_ROOT="$TRTLLM_REPO/experiments/oakhaven-nvfp4-$(date +%Y%m%d-%H%M%S)"
export MODEL_PATH=/shared/models/oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean
export CONTAINER_IMAGE=/shared/images/trtllm-oakhaven-nvfp4.sqsh
export SRT_REPO=/path/to/srt-slurm
export SRTCTL="$SRT_REPO/.venv/bin/srtctl"
export OAKHAVEN_KERNELS_ROOT=/path/to/oakhaven-kernels
export SLURM_ACCOUNT=ACCOUNT
export SLURM_PARTITION=batch
export SLURM_QOS=short

mkdir -p "$EXPERIMENT_ROOT"/{recipes,maps,scripts,setup,outputs,reports,guards,submissions}
```

不得把实验文件写入 `$SRT_REPO`。先用 benchmark skill 的 `srt_repo_guard.py` 对 srt-slurm 保存
完整 HEAD/status baseline，并确认 `$EXPERIMENT_ROOT` 位于 TensorRT-LLM repo 内、srt-slurm 外。

校验固定输入：

```bash
sha256sum "$MODEL_PATH/config.json"
sha256sum "$CONTAINER_IMAGE"
```

若 image hash 不等于历史值，记录为 rebuilt-image 或 different-image reproduction。不能为了得到相同
version string 使用 host `PYTHONPATH`、editable install 或 source overlay；C++/CUDA 差异必须进入 wheel/image。

### 7.3 复制并重绑定 recipe

只复制执行所需文件，不复制 2.8 GiB 历史 outputs：

```bash
cp -a "$REFERENCE_ROOT/recipes/." "$EXPERIMENT_ROOT/recipes/"
cp -a "$REFERENCE_ROOT/scripts/." "$EXPERIMENT_ROOT/scripts/"
cp -a "$REFERENCE_ROOT/setup/." "$EXPERIMENT_ROOT/setup/"
cp "$OAKHAVEN_KERNELS_ROOT/config/trtllm/agg/oakhaven-max/nvfp4-dev/eplb/nomtp-static528.yaml" \
  "$EXPERIMENT_ROOT/maps/"
cp "$OAKHAVEN_KERNELS_ROOT/config/trtllm/agg/oakhaven-max/nvfp4-dev/eplb/mtp3-static528.yaml" \
  "$EXPERIMENT_ROOT/maps/"
sha256sum "$EXPERIMENT_ROOT/maps/"*-static528.yaml
```

Codex 必须用 `apply_patch` 修改新目录中的副本；不要修改历史 reference。至少重绑定：

- model path、container path 和 identity image；
- Slurm account/partition/QoS、GPU type、节点/GPU layout；
- Static map 的 `extra_mount` host path；
- copied runner 中的 `REPO`、`EXP`、`SRT`、guard 和 heartbeat skill path；
- disaggregate 的 `submit.py`、dataset、launcher、work_dir、container mount 和 output path。

只允许调整部署路径和目标集群 Slurm 语法。TP/EP/ADP、backend、token budgets、KV、MTP、
concurrency、ISL/OSL/RR、warmup/formal 倍数及 metrics flags 必须保持冻结；任何变化都应生成单独标记的 arm。

原始 `submit_*.sh` 含历史绝对路径和已存在 ledger 防覆盖保护，**不能在新集群原样执行**。

### 7.4 Preflight 和污染检查

对每个要提交的 recipe：

```bash
cd "$EXPERIMENT_ROOT"
"$SRTCTL" preflight -f recipes/<recipe>.yaml
"$SRTCTL" dry-run -f recipes/<recipe>.yaml -o outputs
grep -RInE '(/metrics|pip install -e|PYTHONPATH=.*TensorRT-LLM)' recipes scripts setup outputs || true
```

提交前人工/Codex 检查 dry-run 展开的 image、model、mount、nodes、GPU denominator、backend、MTP、
EPLB map、workload 和 metrics。`/metrics` 命中若只是本文禁令文字不构成问题；任何实际 curl/client/
probe 请求必须停止提交。

### 7.5 建议执行顺序和关键点

Phase A，最小正确性与 low-latency parity：

1. 用 LL no-MTP recipe 暂时只保留 C1/C128，跑非正式 porting gate。
2. 用 LL MTP3 recipe 暂时只保留 C1/C128，确认输出非垃圾且 MTP 服务路径健康。
3. 通过后恢复完整 C1-C512 recipes，作为正式 LL curves。

Phase B，aggregate headline maximum：

1. no-MTP：`nomtp-budget-confirmation/chunk-budget16896-curve.yaml`，先跑 C2048；通过后跑
   C1536/1792/2048/2304。目标 C2048 在同 image 上约 6.06k-6.10k total TPS/GPU。
2. MTP3：`mtp3-budget33792-curve/static528-budget33792-curve.yaml`，先跑 C512/C640；通过后跑
   C384/512/640/768/896/1024。目标为 5.7k-5.9k 稳定平台，不以单次 6036.81 为硬门槛。
3. 至少在独立 allocation 重复 MTP3 C640；保留所有慢点，报告 min/median/mean/max。

Phase C，baseline/因果诊断，只在 headline 通过后运行：

- `aggregate-ht-*-noeplb.yaml` 与 `aggregate-ht-*-static528.yaml` 做 TRTLLM5-parity baseline。
- `mtp3-nochunk-ab/{chunk1024,chunk8448,nochunk8448}.yaml` 复核 no-chunk 因果关系。
- KV V2 recipes 只做诊断，不替换 V1 maximum baseline。

Phase D，disaggregate：

- aggregate 通过后再运行 `best-noeplb-*-client-oversample-retry2.yaml`。
- 必须验证所有 CTX/GEN roles、NIXL transfer、客户端 oversampling、完成请求数、GPU denominator 和 MTP
  mean accepted length。目标是 TRTLLM5 anchor ±5%，不是 aggregate maximum 的绝对值。

在支持直接 srtctl submission 的目标集群，典型命令为：

```bash
cd "$EXPERIMENT_ROOT"
"$SRTCTL" apply -f recipes/<recipe>.yaml -o outputs --json -y \
  >> submissions/submissions.jsonl
```

若使用历史 allocation wrapper 来满足单 NVL-domain/heartbeat 约束，必须先完成 7.3 的路径重绑定并
dry-run `sbatch --test-only`；不得让 heartbeat 进入 JIT、graph capture、warmup 或 formal window。

## 8. 每个新结果的验收门槛

只有同时满足以下条件才标记 `ACCEPTED`：

- Slurm `COMPLETED 0:0`；资源数正确；要求 NVL domain 的运行没有跨 domain；
- 实际 runtime config 与提交 recipe 一致，模型/image/map hash 已记录；
- image-contained runtime，或明确标注其他 source activation；不得混入未知 host source；
- 每个 concurrency 的 completed==planned、errors=0、ISL 全为 8192、OSL 全为 1024；
- throughput 使用全部 serving GPUs；disaggregate 使用 CTX+GEN 总 GPU 数；
- loading heartbeat 已在 KV/JIT/graph/warmup/formal 前停止；
- 每个正式点前有独立 warmup，JIT/autotune/graph capture 不与正式窗口重叠；
- 所有 server access logs 中真实 `/metrics` 请求数为 0；
- MTP3 配置 draft=3、forced drafts=2.3；若有可审计字段，mean total acceptance 接近 3.3；
- 保存 recipe lock、Slurm metadata、node list、fingerprint、server config/log、raw JSON、rollup 和 auditor。

以下情况必须拒绝：cross-domain、部分请求、错误输出、错误 GPU denominator、formal 前无 warmup、
`/metrics` polling、heartbeat/JIT 污染、没有正式结果的启动/kernel failure。

## 9. 常见 failure pattern

| 现象 | 第一检查项 | 处理 |
|---|---|---|
| C1 或低并发异常慢 | JIT/cubin/cache cold start、wrong image、GDN replay、TP/EP | 独立 warmup；检查 fingerprint/hash；不要接受首轮污染点 |
| HT 全曲线低但请求完整 | nodes/domain、backend、comm method、Static map、token budget | same-allocation matched A/B；不要直接归因“节点坏” |
| TTFT 恶化而 TPOT 稳定 | residency、requests/rank、scheduler population ramp | 检查 batch/KV capacity/chunking，不先改 kernel |
| MTP3 波动 5%-7% | routing trajectory、allocation variance、acceptance | 做独立 repeat，报告分布而不是只选最快点 |
| budget67584 启动后死亡 | `moeComputeRouteDevice` launch failure | 作为 invalid boundary；退回 33792，不报告 TPS |
| KV V2 MTP3 回退 | policy 被强制、capacity/residency 差异 | 匹配容量后再归因；maximum 保持 V1 |
| 结果和 TRTLLM5 接近但与 maximum 差大 | 使用了原始 1024/8448 baseline，而非最大 budget recipe | 比较 recipe lock，不把不同合同混为 regression |
| 日志出现 `/metrics` | 外部 monitor/readiness/reporter | 整个 server 生命周期结果作废，修复后新 job |

## 10. 对外发布措辞

- no-MTP aggregate NVFP4：最大观察值 6096.89，独立复测 6062.45 total tok/s/GPU，可以称为
  “在冻结 image/config 下稳定超过 6k”。
- MTP3 aggregate NVFP4：最高观察值 6036.81；独立曲线峰值 5874.82；C640 三次中位 5820.72、
  范围 5677.05-6036.81。应称为“稳定约 5.7k-5.9k，单次达到 6k”，不能称稳定超过 6k。
- 性能提升来自 chunked scheduling、合适的 scheduler/MoE token budget 和 Static528 组合，
  不应写成 no-chunk 修复或 NVFP4 kernel code fix。
- 历史 image 结果、当前 branch rebuild 结果和不同集群同配置结果必须分别标注。

## 11. 权威入口

- 本文：`OAKHAVEN_NVFP4_REPRODUCTION.md`
- [完整 maximum-throughput 分析](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/NVFP4_FINAL_AGGREGATE_DISAGG_AND_NOCHUNK_MAX_THROUGHPUT_ANALYSIS_20260820.md)
- [冻结计划与 reference lock](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/PLAN_AND_REFERENCE_LOCK.md)
- [机器可读最终 summary](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/nvfp4-final-aggregate-disagg-and-nochunk-max-summary.json)
- [Aggregate V1 full curves](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/aggregate-v1-campaign-audit.json)
- [KV V2 diagnostics](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/aggregate-kv-manager-v2-diagnostics.json)
- [Disaggregate accepted audit](final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/disaggregate-best-client-oversample-retry2-audit.json)

本文冻结的是 2026-08-20 已完成的 NVFP4 campaign。新集群复现必须写新报告并保留所有新 job；
不得回写、覆盖或重新解释历史 raw outputs。
