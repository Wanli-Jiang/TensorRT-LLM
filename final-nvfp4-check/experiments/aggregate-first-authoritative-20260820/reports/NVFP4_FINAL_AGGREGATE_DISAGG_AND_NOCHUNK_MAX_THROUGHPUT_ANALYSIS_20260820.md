<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Oakhaven Max NVFP4：aggregate 最大吞吐、no-chunk 因果验证及 disaggregate 对照

时间：2026-08-20 13:55:13 PDT（America/Los_Angeles）

跨集群复现、完整 LL/HT/V1/V2 数据表、关键日志索引和 Codex 执行合同统一见仓库根目录
`OAKHAVEN_NVFP4_REPRODUCTION.md`。本文继续作为 maximum-throughput 因果分析的权威细节报告。

## 一、结论先行

本轮已经完成 no-MTP、MTP3 的 aggregate 高吞吐上限探索，并把最初的
“no-chunk 可能提高吞吐”假设拆成 matched A/B 进行了验证。最终结论是：

1. **no-chunk 假设不成立。** 真正有效的做法是保留
   `enable_chunked_prefill: true`，同时提高 scheduler 和 MoE 的
   `max_num_tokens`。MTP3 在相同 8448-token budget 下关闭 chunking，所有
   C384-C1024 点都回退 5.50%-9.11%。no-MTP 在相同 8448 budget 下，chunk
   开关仅相差 +0.145%，属于中性。
2. **no-MTP 可以稳定超过 6000 total tok/s/GPU。** Static528、CUTEDSL、
   ADP16/EP16、chunked prefill、budget16896、C2048 的两次独立结果为
   **6096.89** 和 **6062.45 tok/s/GPU**，差异仅 -0.56%。C2304 降到
   5905.02，因此 C2048 是可信饱和点。
3. **MTP3 的可信稳定区间是约 5.7k-5.9k tok/s/GPU，单次最高为
   6036.81。** 同一 Static528、budget33792、C640 配方三次结果为
   6036.81、5677.05、5820.72；中位数 5820.72、均值 5844.86，最大到
   最小跨度 6.34%。因此 6036.81 是有效的“最高观察值”，但不能称作稳定
   6k floor。
4. MTP3 的独立完整曲线在 C1024 得到 **5874.82 tok/s/GPU**；C512-C1024
   整体是一个有集群/路由波动的高吞吐平台。若只追求 total throughput，
   使用 C512-C1024；若也考虑 TTFT，优先 C512-C640。
5. budget 不是越大越好。no-MTP 从 16896 再增到 33792 没有收益
   （-0.53%）；MTP3 的 67584 budget 在 warmup 阶段触发
   `moeComputeRouteDevice / moe_load_balance_routing` CUDA launch failure，
   没有正式结果，必须作为无效配置而不是性能回退点。

机器可读的最终汇总位于：

`/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820/reports/nvfp4-final-aggregate-disagg-and-nochunk-max-summary.json`

## 二、冻结的实验身份

| 项目 | 冻结值 |
|---|---|
| Checkpoint | `/lustre/fsw/portfolios/coreai/users/williamj/models/oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean` |
| Checkpoint 精度 | NVFP4；服务 KV cache 为 FP8 |
| `config.json` SHA-256 | `3b0153fad68686977da18f663455c4164499d0cf752d0767d19e2b46090b34e9` |
| `.sqsh` | `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh` |
| Image SHA-256 | `08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59` |
| TensorRT-LLM version | `1.3.0rc25` |
| TensorRT-LLM repo HEAD | `e49b205b0423ec611a5b1062423f9598674ba0e8` |
| srt-slurm HEAD | `922f005de4674cd51cfdc6f6b361ad07b1893014`，本轮未修改 |
| Source overlay | 无 |
| GPU/拓扑 | 4 个 GB300 节点，16 GPU，单 NVL72 domain |
| Parallelism | TP16/EP16/ADP16/PP1，attention DP 开启 |
| MoE backend | CUTEDSL；NVLINK one-sided；dispatch/combine block 256；workspace 2304 MiB |
| Workload | random，精确 ISL8192/OSL1024，RR=1.0，request rate=inf |
| 采样 | 每个 concurrency 1 wave warmup + 3 waves formal，HTTP connection reuse |
| KV | manager V1，FP8，fraction=0.92，block reuse 关闭 |
| Perf metrics | `return_perf_metrics`、iter perf/request stats、iter log 全部关闭；从不访问 `/metrics` |

Static placement map：

- no-MTP：`maps/nomtp-static528.yaml`，SHA-256
  `68ef80e1b996cc900f29c859221155bf6a3b7e7c1e5c83faf476d1c12b2f88be`。
- MTP3：`maps/mtp3-static528.yaml`，SHA-256
  `cbc575a8d0bf221a76d3a04552a123d27964b788dee233be4eced6b6a71b30dc`。

## 三、最终 aggregate 最大吞吐

### 3.1 no-MTP

最终推荐配置：

```yaml
tensor_parallel_size: 16
moe_expert_parallel_size: 16
enable_attention_dp: true
max_batch_size: 128
max_num_tokens: 16896
enable_chunked_prefill: true
cuda_graph_config:
  max_batch_size: 128
kv_cache_config:
  use_kv_cache_manager_v2: false
  free_gpu_memory_fraction: 0.92
  dtype: fp8
moe_config:
  backend: CUTEDSL
  max_num_tokens: 16896
  load_balancer: /eplb.yaml
```

同时设置 `TRTLLM_USE_GDN_REPLAY=0`，不启用 speculative decoding。

#### no-MTP 上限筛选

| Job | Chunked prefill | Scheduler/MoE budget | C | total tok/s/GPU | 相对同日 no-chunk 8448 |
|---:|:---:|---:|---:|---:|---:|
| 527441 | 否 | 8448 | 2048 | 5486.55 | 基线 |
| 527442 | 是 | 8448 | 2048 | 5494.49 | +0.145% |
| 527443 | 是 | 16896 | 2048 | **6096.89** | **+11.124%** |

这个表把两个因素分开了：仅打开 chunking 没有显著收益；把 budget 从一个
8192-token prompt 级别提高到可以容纳约两个完整 prompt，才产生约 11% 的
端到端吞吐提升。

#### no-MTP 独立完整尾部曲线

Job 527727，所有点均为 3×concurrency 正式请求：

| C | 完成请求 | total tok/s/GPU | Median TTFT | Median TPOT |
|---:|---:|---:|---:|---:|
| 1536 | 4608 | 5777.31 | 12.579 s | 135.50 ms |
| 1792 | 5376 | 5970.15 | 12.662 s | 155.28 ms |
| 2048 | 6144 | **6062.45** | 14.798 s | 174.86 ms |
| 2304 | 6912 | 5905.02 | 23.127 s | 182.46 ms |

结论：

- C2048 的首次/复测为 6096.89 / 6062.45，差 -0.56%，是稳定结果。
- C2304 相比 C2048 回退 2.60%，同时 TTFT 明显增加，说明已经越过饱和点。
- budget33792、C2048 的 Job 527728 为 6030.29，比 budget16896 的独立
  复测低 0.53%。所以 no-MTP 的最佳 budget 取 16896，不继续增大。
- 相比当前 image 原始 Static528 峰值 5506.88，独立复测提高 10.09%；
  相比 TRTLLM5 Static528 参考 5476.81，提高 10.69%。

### 3.2 MTP3

最终推荐配置：

```yaml
tensor_parallel_size: 16
moe_expert_parallel_size: 16
enable_attention_dp: true
max_batch_size: 32
max_num_tokens: 33792
enable_chunked_prefill: true
cuda_graph_config:
  max_batch_size: 32
kv_cache_config:
  use_kv_cache_manager_v2: false
  free_gpu_memory_fraction: 0.92
  dtype: fp8
moe_config:
  backend: CUTEDSL
  max_num_tokens: 33792
  load_balancer: /eplb.yaml
speculative_config:
  decoding_type: MTP
  max_draft_len: 3
```

同时设置：

```text
TRTLLM_USE_GDN_REPLAY=1
TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=2.3
TLLM_SPEC_SKIP_IDENTITY_DRAFT_GATHER=0
```

`2.3` 表示强制平均接受 2.3 个 draft tokens，目标总 accept length 约 3.3。
由于 aggregate run 按要求关闭 TensorRT-LLM perf metrics，aggregate 结果 JSON
没有可审计的 measured accepted-length 字段；但相同模型的已接受 disaggregate
复现独立测得 3.287，和目标一致。

#### MTP3 budget 与 placement 筛选

同日 C640 matched screen：

| Job | Placement | Scheduler/MoE budget | total tok/s/GPU | 相对 no-EPLB 8448 |
|---:|---|---:|---:|---:|
| 527208 | no-EPLB | 8448 | 4735.09 | 基线 |
| 527209 | no-EPLB | 16896 | 4935.07 | +4.22% |
| 527210 | no-EPLB | 33792 | 5113.23 | +7.99% |
| 527211 | Static528 | 8448 | **5456.98** | **+15.25%** |

这说明对当前 image、当前路由数据和已经修正的 scheduler budget，Static528
是有收益的。不能把早期 1024-token scheduler 配置下“Static 没收益”的结论
直接外推到新配置。

#### MTP3 budget33792 独立完整曲线

Job 527804：

| C | 完成请求 | total tok/s/GPU | Median TTFT | Median TPOT |
|---:|---:|---:|---:|---:|
| 384 | 1152 | 5501.20 | 12.321 s | 25.33 ms |
| 512 | 1536 | 5793.83 | 11.240 s | 35.14 ms |
| 640 | 1920 | 5677.05 | 16.207 s | 43.12 ms |
| 768 | 2304 | 5525.36 | 29.351 s | 42.02 ms |
| 896 | 2688 | 5627.11 | 45.558 s | 43.64 ms |
| 1024 | 3072 | **5874.82** | 55.262 s | 42.31 ms |

独立曲线的最高点在测试边界 C1024，但 C512-C1024 的 total throughput 波动
只有一个宽平台的量级，并且 TTFT 从 C512 的 11.2 秒增至 C1024 的 55.3 秒。
因此：

- 只追求 aggregate total throughput：使用 C512-C1024，并以 5.7k-5.9k
  tok/s/GPU 作为可复现能力区间。
- 还在意 tail latency：优先 C512-C640。
- 不把单次 6036.81 当作稳定保证；它是合法、完整、可审计的最高观察值。

#### MTP3 C640 三次同配置复测

| Job | 场景 | total tok/s/GPU |
|---:|---|---:|
| 527689 | 首次 budget33792 focused point | **6036.81** |
| 527804 | 完整曲线中的独立重复 | 5677.05 |
| 528007 | 第三次独立重复 | 5820.72 |

统计：最小 5677.05，中位数 5820.72，均值 5844.86，最大 6036.81，
max/min 跨度 6.34%。即使取最差一次，仍比同配方 budget8448 的复测
5360.03 高 5.91%；中位数提高 8.59%。因此 budget33792 的收益可复现，
但节点/路由轨迹会影响单次绝对值。

#### 67584 budget 为什么不能使用

Job 527805 在正式 benchmark 之前失败：

- warmup 计划 640 请求，仅完成 114；
- `moe_load_balance_routing` / `moeComputeRouteDevice` 触发
  `CUDA unspecified launch failure`；
- Slurm `FAILED/87:0`；
- 没有正式结果文件；
- 心跳已在 warmup 前停止，并非心跳干扰；日志没有可用的正式吞吐窗口。

因此 67584 是当前 stack 的无效上限配置。它只用于把安全 budget bracket 在
33792 与 67584 之间，不能作为“67584 性能差”的数据点。

## 四、为什么 no-chunk 假设是错的

### 4.1 matched causal A/B

MTP3 的三臂实验保持模型、image、backend、parallelism、KV、请求和节点约束
一致，只改变 chunk flag 与 budget：

| C | chunk+1024 | chunk+8448 | no-chunk+8448 | 1024→8448 | 关闭 chunk 的影响 |
|---:|---:|---:|---:|---:|---:|
| 384 | 2757.64 | 4623.25 | 4368.99 | +67.65% | -5.50% |
| 512 | 2827.43 | 4910.95 | 4463.78 | +73.69% | -9.11% |
| 640 | 2897.29 | 4938.49 | 4499.30 | +70.45% | -8.89% |
| 768 | 2849.78 | 4915.28 | 4621.37 | +72.48% | -5.98% |
| 896 | 2885.98 | 4877.59 | 4564.54 | +69.01% | -6.42% |
| 1024 | 2788.49 | 4886.61 | 4459.48 | +75.24% | -8.74% |

因果结论非常明确：此前看起来像“no-chunk 带来的巨大收益”，实际上来自
`max_num_tokens` 从 1024 提到能够容纳一个 8192-token prompt 的量级；在同一
8448 budget 下，no-chunk 始终更差。

### 4.2 scheduler 控制流解释

当前 image 内的 scheduler 证据保存在：

- `evidence/image-scheduler-chunk-control-flow.txt`
- `evidence/image-scheduler.py.sha256`
- `evidence/image-native-scheduler-strings.txt`
- `evidence/image-libtensorrt_llm.so.sha256`

控制流的核心区别：

1. `enable_chunked_prefill=false` 时，context request 必须把完整的剩余 context
   放进当前 iteration 的剩余 `max_num_tokens`；放不下时 scheduler 直接
   `break`。在已有 generation requests 驻留的高并发阶段，这会造成 context
   head-of-line wait 和 batch 空洞。
2. `enable_chunked_prefill=true` 时，scheduler 先保留 generation requests，
   再用 context chunks 填满剩余 token capacity。这样既不破坏 decode
   residency，又能让 prefill/population ramp 持续推进。
3. ISL=8192 时，8448、16896、33792 分别大致对应每 iteration 可容纳一个、
   两个、四个完整 prompt 加少量 generation/draft token。扩大 budget 会增加
   每轮有效 token 数，并为 CUTEDSL MoE 提供更大的 token/M shape，从而改善
   GPU 与通信利用率。

需要保留一个边界：最大吞吐搜索把 scheduler 和 MoE 的 `max_num_tokens` 同步
调整，因此 8448 以上的增益还没有完全拆成“scheduler 占多少、MoE cap 占多少”。
但 1024→8448 的 A/B 中 MoE cap 只从 8192 变到 8448，而 scheduler cap 从
1024 变到 8448，足以说明第一阶主因是 scheduler token capacity，而不是
no-chunk。

## 五、Static EPLB、KV manager 与 disaggregate 的位置

### 5.1 Static528

- 原始 no-MTP aggregate campaign：Static528 峰值 5506.88，相比 no-EPLB
  4984.08 提高 10.49%。
- 修正 budget 后的 MTP3 同日 C640 screen：Static528+8448 为 5456.98，
  no-EPLB+8448 为 4735.09，提高 15.25%。
- 最大吞吐配方因此使用 Static528。该结论只适用于当前模型、当前路由 map、
  当前 image 与 ISL8192/OSL1024/RR1.0，不应泛化为所有 MTP workload。

### 5.2 KV manager V2

单独的 V2 诊断已经完成，但最大吞吐探索继续使用与 TRTLLM5 参考一致的 V1：

- low-latency no-MTP：无回退；
- low-latency MTP3：无回退；
- high-throughput no-MTP：约中性，但 V2 可用容量更小，属于 capacity-confounded；
- high-throughput MTP3：C640/C768/C1024 分别为 -10.31%/-1.20%/-4.49%，
  存在随并发变化的回退。

所以本轮 maximum result 不能替换成 V2 后再与 V1 基线比较。

### 5.3 已接受的 disaggregate E2E 对照

| 模式 | Job | 拓扑 | C | 当前结果 | TRTLLM5 | 差异 |
|---|---:|---|---:|---:|---:|---:|
| no-MTP | 526474 | 3×CTX8-ADP8/EP8 + 1×GEN16-ADP16/EP16，40 GPU | 1536 | 5341.00 | 5321.71 | +0.363% |
| MTP3 | 526475 | 4×CTX8-ADP8/EP8 + 1×GEN16-ADP16/EP16，48 GPU | 768 | 5298.25 | 5296.64 | +0.030% |

两项均与 TRTLLM5 最佳 no-EPLB disaggregate 参考对齐。MTP3 measured mean
accepted length 为 3.287。

注意 aggregate 和 disaggregate 的 GPU denominator、拓扑、CTX/GEN 比例不同，
不能只用绝对 tok/s/GPU 判断哪一种部署“更好”；它们回答的是不同资源布局下
的吞吐问题。

## 六、可靠性与 failure classification

所有被接受的正式结果均通过以下检查：

- Slurm 主 Job 为 `COMPLETED/0:0`，GPU 数正确，所有节点来自一个 NVL72 domain；
- runtime config 与提交 recipe 一致，模型/image/map 哈希冻结；
- planned/completed request 数一致；输入长度数组全部为 8192，输出长度全部为
  1024；错误字符串全部为空；
- total throughput 使用全部 16 serving GPUs 作为 denominator；
- loading heartbeat 仅覆盖长时间权重加载，并在 KV-cache、CUDA graph、warmup、
  formal measurement 之前自动停止；
- 每个正式点前有独立 warmup，JIT/autotune 不落入 formal window；
- TensorRT-LLM perf/iteration/request metrics 全部关闭，日志中 `/metrics` 请求数为 0。

少量 SA-Bench JSON 中出现 1-2 个 `empty_decoded_texts`。它们来自 tokenizer 对
special-token 序列的文本解码，不是服务失败：对应请求仍返回精确 1024 output
tokens，错误数组为空，planned/completed 数一致。它不改变性能完整性判定；但本
报告是性能报告，不代替独立的模型 accuracy 评估。

failure 分类必须遵守：

- 有完整 formal result、请求/token/错误审计通过：可用性能点；
- 服务在 warmup/JIT/kernel 阶段死亡、没有 formal result：配置或 kernel failure；
- cross-domain allocation、heartbeat 进入 formal、访问 `/metrics`、JIT 与 formal
  重叠：实验污染，结果作废；
- 单次性能低但完整性通过：先看同配方重复分布，不能自动归因于节点重启或代码。

## 七、可复现文件

实验根目录：

`/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820`

关键 recipes：

- no-MTP 最终曲线：
  `recipes/nomtp-budget-confirmation/chunk-budget16896-curve.yaml`
- no-MTP budget 上限对照：
  `recipes/nomtp-budget-confirmation/chunk-budget33792-c2048.yaml`
- MTP3 最终曲线：
  `recipes/mtp3-budget33792-curve/static528-budget33792-curve.yaml`
- MTP3 无效上限：
  `recipes/mtp3-budget33792-curve/static528-budget67584-c640.yaml`
- MTP3 第三次 C640 复测使用：
  `recipes/mtp3-static-confirmation/static528-budget33792-c640.yaml`

提交脚本：

- `scripts/submit_nomtp_budget_confirmation.sh`
- `scripts/submit_mtp3_budget33792_curve.sh`
- `scripts/submit_mtp3_static33792_c640_repeat2.sh`

审计脚本与最终审计产物：

- `scripts/audit_nomtp_budget_confirmation.py` →
  `reports/nomtp-budget-confirmation.json`
- `scripts/audit_mtp3_budget33792_curve.py` →
  `reports/mtp3-budget33792-curve.json`
- `scripts/audit_mtp3_c640_repeat2.py` →
  `reports/mtp3-static33792-c640-repeat2.json`
- `reports/mtp3-nochunk-causal-ab.json`
- `reports/aggregate-kv-manager-v2-diagnostics.json`
- `reports/disaggregate-best-client-oversample-retry2-audit.json`

对应 raw output/job：

- no-MTP final curve：
  `outputs/nomtp-budget-confirm-chunk-budget16896-curve-v1/527727`
- no-MTP budget33792：
  `outputs/nomtp-budget-confirm-chunk-budget33792-c2048-v1/527728`
- MTP3 final curve：
  `outputs/mtp3-record-static528-budget33792-curve-v1/527804`
- MTP3 invalid budget67584：
  `outputs/mtp3-record-static528-budget67584-c640-v1/527805`
- MTP3 third C640 repeat：
  `outputs/mtp3-static33792-c640-repeat2-v1/528007`

## 八、建议采用的公开表述

对外分享时，建议把“最大值”和“稳定复现值”分开：

- **no-MTP aggregate NVFP4：最大观察值 6096.89，独立复测 6062.45
  total tok/s/GPU；可以称为稳定超过 6k。**
- **MTP3 aggregate NVFP4：最大观察值 6036.81；独立完整曲线峰值 5874.82；
  C640 三次中位数 5820.72、范围 5677.05-6036.81。应称为稳定 5.7k-5.9k，
  偶尔达到 6k，而不是稳定超过 6k。**
- **性能提升来自 chunked scheduling + 合适的 scheduler/MoE token budget +
  Static528 的组合；不应写成 no-chunk 修复。**

这些措辞同时保留了最高能力、重复性和集群/路由波动，避免只选择最好的一次
结果造成不可复现的承诺。

最后一个范围边界需要明确：no-MTP 已通过 C2304 的下降确认 C2048 饱和；MTP3
按照冻结的 TRTLLM5 饱和 ladder 扫到 C1024，独立曲线在 C1024 有一次回升，
但仍低于 C640 的最高观察值 6036.81。因而本报告给出的是**当前冻结 C384-C1024
范围内的审计最大值和稳定平台**，不是对所有未测试 C>1024 的数学全局最优证明。
如果未来扩展 C>1024，应作为新的尾部实验单独报告，不能回填或覆盖本报告的
冻结结果。
