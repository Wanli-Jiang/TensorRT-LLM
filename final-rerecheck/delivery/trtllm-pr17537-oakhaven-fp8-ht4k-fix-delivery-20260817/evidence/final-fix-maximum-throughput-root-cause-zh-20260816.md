<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MTP3 最高吞吐最终修复：完整根因分析

> **因果归因更新（2026-08-19）：** 本文保留的运行数据以及 replay/residency/KV-cap
> 结论仍然有效，但旧文把 FlashInfer static T=1 归因为 6.462% serving 收益的说法不再接受，
> 因为旧对照来自不同 allocation。后续同 allocation production A/B/A 的
> dynamic/static/dynamic 为 4049.545/4064.498/4096.202 tok/s/GPU，说明 static T=1
> 不是跨 4k 必需条件。请以仓库中的
> `GDN_T1_AND_HIGH_THROUGHPUT_4000TPS_ROOT_CAUSE_REAUDIT_ZH_20260818.md` 和本交付包
> `TECHNICAL_CLARIFICATION.md` 为最终解释。

生成时间：`2026-08-16T21:27:21-07:00`（`America/Los_Angeles`）

证据截止时间：`2026-08-16T21:20:26-07:00`（Job 491712 完成时间）

术语澄清（`2026-08-17`）：原报告中的“DeepGEMM Split-K”混合了两条独立路径。
DeepGEMM 是 FP8 MoE backend；`TRTLLM_LOW_M_GEMM_BACKEND=auto` 将 eligible BF16、
`M<=32` 的 Linear 路由到 FlashInfer `mm_bf16`，再由其内部 heuristic 选择 direct 或
Split-K。观察到的 830–838 MiB 失败 allocation 是 DeepGEMM MoE
`triton_fused_gather_finalize` 的 output tensor，不是已经证明的 FlashInfer Split-K workspace。
另外，`d8d10ab` wide-value-head replay tuning 虽存在于被测 PR #17537 source 中，但没有
isolated peak-throughput A/B，不属于已证明的 >4k 必需项。完整澄清见交付包 README 和当前
canonical clarification report。

## 1. 结论

最终稳定恢复到 `>4000 total tok/s/GPU` 不是某个单独参数带来的偶然收益，也不能归因于节点重启。完整证据表明，原先存在两个相互独立的软件问题，随后还需要两个资源/调度条件，才能把代码修复转化为最高吞吐：

1. **FlashInfer GDN 的 T=1 编译缓存和 specialization 不完整。**旧路径从 `B=1` 的 batch-dynamic 模板编译，但 Python cache key 和 CuTe compile signature 都没有完整区分实际 batch、shape 和 stride。`B=1,T=1` 的 singleton dimension 使动态 layout 描述不足以唯一表示实际视图；CUDA graph 可能持续 replay 一个没有按实际 batch/layout specialization 的 cubin。修复后，生产 shape 的 GDN CUDA-graph micro-benchmark 延迟降低约 8%，端到端隔离实验提高 6.462%，输出和 recurrent state bit-exact。
2. **旧 MTP GDN verify 路径错误地物化了随完整 speculative state 规模增长的巨大中间张量。**在生产配置 `M=104,T=4` 下，仅 intermediate SSM state 就需要 112.125 GiB/GPU；加 convolution scratch 后约 115.410 GiB/GPU。cached replay 只保存 prefix-invariant 的 causal update，并只重算最多 8 个新 token，使 verify scratch 降至 11.388 GiB/GPU，节省 104.022 GiB/GPU（90.133%）。这是 resident batch 能够从 M36 提高到至少 M48 的必要容量修复。
3. **突破 4k 的最后一步主要是 request residency/population-ramp 效应，而不是单用户 decode 更快。**`C=1536, ADP=32` 意味着每个 ADP rank 初始有 48 个请求。M36 全局只能 resident 1152 个请求，另外 384 个必须进入第二波；M48 正好能 resident 全部 1536 个请求。同一 allocation 的直接 A/B 中，M48 相比 M36 吞吐提高 3.746%，但 median TPOT 反而恶化 29.021%。因此收益不可能来自 decode latency 改善，而是来自消除第二波请求并缩短固定工作量的 makespan。
4. **no-chunk prefill 和绝对 KV token cap 是最终配置中的两个 enabling choice。**no-chunk 避免 8192-token prompt 被切成至少 8 个 1024-token 调度/collective 阶段，加快所有请求进入 decode 的速度。另一方面，只设置 `free_gpu_memory_fraction=0.88` 并不能保证 transient headroom：state memory 释放后，KV allocator 会把释放出来的空间再次占满。再设置 `max_tokens=479232`，可固定为每 rank 52 个平均 9216-token sequence，同时满足 48-request residency，并为 DeepGEMM MoE output、CUDA graph、communication workspace 和其他 transient 留出空间。

所以，根因可以精确表述为：

> **T=1 GDN specialization/cache identity 缺陷造成每个 decode cycle 的真实性能损失；旧 GDN speculative-state 生命周期又使达到 C1536 所需的 resident batch 不可行。cached replay 修复容量后，M48 消除了 384 个请求的第二调度波；no-chunk 和绝对 KV cap 分别保证快速 population ramp 与稳定的 transient memory headroom。**

FlashInfer low-M auto 不是 regression 根因；在 headroom 相同的端到端匹配实验中，它相对 off 带来 1.470% 的正收益，但已有 low-M off run 超过 4k，因此它不是超过 4k 的必要条件。`/metrics`、heartbeat、JIT 冷启动、节点重启和错误输出也已被排除。

## 2. 被分析的生产配置

最终直接 A/B 使用如下共同配置：

| 项目 | 设置 |
|---|---|
| Model | `/lustre/fsw/portfolios/coreai/users/williamj/models/oakhaven-max-final-fp8_vv3` |
| Container | `trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh` |
| Container SHA256 | `08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59` |
| GPU | 32 × GB300，8 nodes × 4 GPUs |
| Parallelism | TP32 / EP32 / ADP32 / PP1 |
| Precision | FP8 weights，FP8 attention KV，BF16 Mamba/GDN state |
| MoE | DeepGEMM，Static544 EPLB，max tokens 65536 |
| MTP | draft length 3；forced accepted draft tokens 2.3，即总 accept length 3.3 |
| Workload | ISL=8192，OSL=1024，RR=1.0，C=1536 |
| Requests | 4608 warmup + 4608 formal |
| Prefill | `max_num_tokens=8512`，chunked prefill disabled |
| KV manager | V2，fraction=0.88，absolute `max_tokens=479232` |
| GDN | T=1 static specialization + cached replay V2 |
| FlashInfer low-M BF16 | `TRTLLM_LOW_M_GEMM_BACKEND=auto`；内部 direct/Split-K heuristic |
| Formal metric | total token throughput / 32 GPUs |

M36 和 M48 两个 arm 的语义差异仅为：

- `max_batch_size=36` 对比 `48`；
- CUDA graph `max_batch_size=36` 对比 `48`；
- 为避免 cache 污染而使用不同的临时 cache 目录和实验名称。

两个 arm 使用同一 Job 491712、同一 8-node segment、同一 image/model、同一绝对 KV capacity，并且顺序执行。这样可以把节点、镜像、模型、网络拓扑和大部分集群时间变化排除在直接因果比较之外。

## 3. 吞吐指标和工作量校验

每个 formal arm 的固定总工作量为：

```text
4608 requests × (8192 input + 1024 output)
= 42,467,328 total tokens
```

报告中的总吞吐重新计算为：

```text
total tok/s/GPU = 42,467,328 / measured_duration_seconds / 32
```

输入 token 占 total-token metric 的比例为：

```text
8192 / (8192 + 1024) = 88.8889%
```

这点非常重要：该 workload 的 total-token throughput 主要由 prompt population 多快完成 prefill、进入 decode，以及整个固定 workload 多快结束决定。它并不等价于单用户 decode token latency。因为工作量完全相同，所以 total throughput 提高就是 makespan 缩短；不能只看 TPOT 判断总吞吐。

所有被接受的 formal JSON 都满足：

- `completed == num_prompts`；
- zero nonempty errors；
- 所有 generated text 非空；
- 每个请求实际 ISL 精确为 8192；
- 每个请求实际 OSL 精确为 1024；
- throughput 可以从固定总 token 数和 measured duration 重新计算。

## 4. 根因一：T=1 GDN compile-cache specialization 缺陷

### 4.1 旧实现的问题

FlashInfer 修复 commit 为：

```text
baad0dca27d165341d188b895f3ab161e8098344
```

对应文件：

```text
flashinfer/gdn_kernels/gdn_decode_bf16_state.py
```

旧的 T=1 路径具有两个 cache 层次的问题：

1. Python kernel cache key 没有包含 concrete `B`，也没有完整包含所有 caller-visible tensor 的 shape/stride；
2. CuTe compile signature 没有包含 static batch size，导致即使 Python 侧区分了某些调用，CuTe 内部仍可能复用先前的 dynamic cubin。

在 `B=1,T=1` 下，size-one dimension 的 stride 并不能唯一确定它所代表的逻辑 layout。packed/non-compact view、padded state 和 CUDA-graph batch bucket 可能映射到相同的不充分 cache identity。因此这里不是简单的“cache miss 太多”，而是**cache identity 不足造成错误 specialization 复用**：数值结果在已测 shape 上仍可以正确，但生成的执行路径比 exact static specialization 慢。

### 4.2 修复内容

修复同时关闭两个 cache 层次的碰撞：

- 在 T=1 Python cache key 中加入 concrete batch `B`；
- 加入所有相关 tensor 的 shape 和 stride；
- T=1 使用 concrete tensor descriptor，而不是完全从 `B=1` dynamic template 推导 runtime layout；
- 把 `static_batch_size` 加入 CuTe compile signature；
- 对真正的 MTP `T>=2` 路径继续保留 dynamic-batch 行为。

只修改 Python key 不够，因为 CuTe 内部仍可能复用 cubin；只修改 CuTe signature 也不够，因为 packed 和 non-compact view 仍可能在 Python cache 层碰撞。最终修复同时处理了两个层次。

### 4.3 exact-shape micro-benchmark

Job 467068 在 GB300 上使用生产 shape：

```text
B=36, T=1, H=16, HV=128, K=128, V=128
50 warmup + 500 measured CUDA-graph replays
```

结果：

| 路径 | 单次延迟 | 对 fixed/static 的差异 |
|---|---:|---:|
| 旧 dynamic specialization | 0.055303169 ms | baseline |
| 新 static specialization | 0.051204033 ms | **8.0055% speedup** |

eager 模式的独立测量方向一致，speedup 为 8.4460%。新旧路径的 output 和 recurrent state 在 exact-shape control 中 bit-exact，说明这是性能修复，不是用数值误差换取速度。

Job 467826 做了 immutable fixed-vs-fixed repeat：

- CUDA graph ratio：1.000086；
- eager ratio：0.999928；
- output/state bit-exact。

这个 repeat 证明 micro-benchmark 本身稳定，8% 差异不是 timer noise 或偶然运行顺序造成的。

### 4.4 端到端隔离实验

仅替换 GDN overlay 的端到端对照：

| Job | GDN 路径 | total tok/s/GPU | Duration | Median TTFT | Median TPOT |
|---:|---|---:|---:|---:|---:|
| 466450 | dynamic/old | 3784.483 | 350.670 s | 34.886 s | 73.713 ms |
| 467197 | static/fixed | 4029.055 | 329.383 s | 32.731 s | 69.562 ms |

变化：

- total tok/s/GPU：**+6.4625%**；
- duration：-6.0702%；
- median TTFT：-6.1783%；
- median TPOT：-5.6309%。

该模型 92 层中有 69 个 GDN layer。普通 decode 部分的每个 speculative cycle 都执行 T=1，因此单个 kernel 约 8% 的稳定差异会跨 69 层和大量 decode iteration 累积。micro-benchmark 和端到端隔离实验方向一致，构成了该根因的直接证据。

修复后的 M36 独立重复 Job 467781 得到 3998.881 tok/s/GPU；两个 static M36 run 平均 4013.968，CV=0.376%，说明提升可重复，但 M36 本身仍处在 4k 附近而非稳定显著超过 4k。

## 5. 根因二：旧 GDN speculative-state 生命周期造成容量爆炸

### 5.1 旧路径的精确内存公式

生产模型有 69 个 GDN layer。旧 verify 路径在 `M=104,T=4,K=128,V=128,HV=128` 下物化 intermediate SSM state：

```text
69 layers × M104 × T4 × K128 × V128 × HV128 × 2 bytes
= 120,393,302,016 bytes
= 112.125 GiB/GPU
```

再加 convolution scratch：

```text
112.125 GiB + 3.284912 GiB = 115.409912 GiB/GPU
```

Job 488785 正好在尝试分配 112.12 GiB 时 OOM，与公式精确吻合。因此这不是“不明原因集群回退”，而是可以由 tensor geometry 直接预测的确定性内存缺陷。

### 5.2 cached replay 如何改变状态复杂度

cached replay 保存：

- `old_u`：prefix-invariant update vector；
- `old_k`：已经 normalized 的 prefix key；
- `old_G`：prefix cumulative log-decay。

每次 verify 不再重新物化整个历史 speculative-state tensor，而是只对最多 `T<=8` 个新 update 做 forward substitution。V2 all-layer commit 再通过一个 partitioned path 一次性推进 checkpoint state。

相关 TensorRT-LLM commits：

```text
ee241d25f43973ad52495119d6536176b91c0aec
57f2781e4e9f679cfa429400b64e447fbefa253e
```

其中 PR #16464 / `ee241d25f4` 首次引入功能性 GDN MTP cached replay；PR #16768 /
`57f2781e4e` 在此基础上改进 low-batch kernel、default enablement、V2 manager/state layout
和 all-layer commit。最终被测 V2 stack 同时包含两者。PR #17537 中后续的 `d8d10ab`
wide-value-head tuning 是精确源码 provenance；其 ratio-8 主 kernel tuning 对 Oakhaven
`H=16,HV=128` 只在 `N<=4` 触发，且没有 isolated C1536 A/B，因此不列为已证明的峰值吞吐
必要项。

内存审计结果：

| Verify state | GiB/GPU |
|---|---:|
| Legacy intermediate SSM | 112.125 |
| Legacy convolution scratch | 3.285 |
| **Legacy total** | **115.410** |
| Replay history buffers + common convolution scratch | **11.388** |
| **节省** | **104.022（90.133%）** |

这里必须谨慎区分作用范围：现有证据证明 replay 是**容量和 state-lifecycle 修复**，使 M48/M104 成为可能；没有声称 replay 会让同一个 M36 shape 的 kernel latency 本身更快。

### 5.3 “只增加 max batch”为什么不足够

旧路径的两个 M48/no-replay formal run（Jobs 488761、488762）只有：

```text
2656.712, 2669.285 tok/s/GPU
mean = 2662.998 tok/s/GPU
```

这说明单纯把 scheduler capacity 从 M36 提到 M48 并不能绕过旧 state path。由于这些旧 run 与最终 stack 在 replay version 和 memory setting 上还有其他差异，它们是强方向性证据，不作为严格的 matched replay A/B；精确因果证据仍是内存公式、112.12-GiB OOM 和 replay footprint audit。

## 6. 为什么 M48 是突破 4k 的直接阈值

### 6.1 请求几何

`C=1536, ADP=32`：

```text
1536 / 32 = 48 requests/rank
```

| max batch | 全局 resident slots | 初始 queued requests | 结果 |
|---:|---:|---:|---|
| M36 | 36 × 32 = 1152 | 1536 - 1152 = 384，即 12/rank | 至少需要第二波 |
| M48 | 48 × 32 = 1536 | 0 | 一次 resident 全部初始 population |
| M104 | 104 × 32 = 3328 | 0 | 有额外容量 margin |

M36 的 384 个 queued request 不能与第一波同时完成 prefill/decode；这会拉长 TTFT 分布和整个固定 workload 的尾部。M48 正好跨过 `48 requests/rank` 的 residency cliff。

### 6.2 同 allocation 的最终直接 A/B

Job 491712 的结果：

| C1536 formal metric | M36 | M48 | M48 相对变化 |
|---|---:|---:|---:|
| Total tok/s/GPU | 3971.592 | **4120.374** | **+3.746%** |
| Fixed-work makespan | 334.149 s | 322.083 s | **-3.611%** |
| Median TTFT | 27.539 s | 7.978 s | **-71.032%** |
| Median TPOT | 74.296 ms | 95.857 ms | **+29.021%（变差）** |
| Peak aggregate output | 51,864 tok/s | 57,370 tok/s | **+10.616%** |
| Total manager quota | 34.189 GiB | 37.694 GiB | +3.504 GiB |
| Rank-0 PyTorch allowance | 236.454 GiB | 232.925 GiB | -3.529 GiB |

这是最终根因链中最关键的实验：

- 如果收益来自 decode kernel 变快，TPOT 应该改善；实际 TPOT 恶化 29.021%。
- 如果收益来自消除第二波 request，应该观察到 TTFT 大幅下降、peak aggregate output 上升、固定工作量 makespan 缩短；实际数据完全符合这一签名。
- M48 在直接 control 中达到 4120.374，而 headroom-safe 的 M36 只有 3971.592。

因此，突破 4k 的直接机制是**resident population 从 36 提升到每 rank 48，消除第二调度波并提前建立更高的 aggregate decode occupancy**。

M48 多出的 3.504 GiB quota 是每 rank 多 12 个 resident slot 所需的 batch-dependent state/context 成本，不是 attention KV 增大：两个 arm 的 attention KV absolute cap 都是 479232 tokens。

## 7. no-chunk prefill 的作用和边界

Job 488694 在同一 allocation 中比较：

- chunked：`max_num_tokens=1024`；
- no-chunk：`max_num_tokens=8512`，chunking disabled；
- 其他语义设置保持一致，cache name 隔离。

| Arm | Total tok/s/GPU | Makespan | Median TTFT | Median TPOT | Peak output |
|---|---:|---:|---:|---:|---:|
| Chunked | 3552.008 | 249.081 s | 35.308 s | 73.439 ms | 39,632 |
| No-chunk B1 | 3889.608 | 227.462 s | 26.446 s | 74.427 ms | 47,903 |
| No-chunk B2 | 3701.809 | 239.001 s | 29.560 s | 77.433 ms | 48,025 |

相对 chunked：

- B1 total throughput +9.504%，TTFT -25.099%，peak output +20.870%；
- B2 total throughput +4.217%，TTFT -16.279%，peak output +21.177%；
- 两个 no-chunk arm 的 TPOT 都略微变差。

8192-token prompt 在 1024-token budget 下至少需要 8 个 chunk scheduling/collective boundary；8512 budget 可以在一个调度单元中完成 context。结果再次呈现“TTFT/ramp 改善、TPOT 不改善”的 population-ramp 特征。

但是 no-chunk M36 没有稳定超过 4k，所以它是必要的方向性优化，不是最终 bug fix 的独立根因。

## 8. 为什么仅设置 KV fraction 会再次吃掉修复收益

### 8.1 M104 上观察到的 fraction/headroom 区间

| `free_gpu_memory_fraction` | Logged device quota | 平均 9216-token sequences/rank | 结果 |
|---:|---:|---:|---|
| 0.85 | 49.620 GiB | 47.91 | 低于 48/rank residency cliff |
| **0.88** | **51.371 GiB** | **52.06** | resident capacity 和 transient headroom 均满足 |
| 0.90 | 52.547 GiB | 54.85 | DeepGEMM MoE gather/finalize output OOM；同时 low-M auto 已启用 |

M104、KV=0.90 时，`triton_fused_gather_finalize` 需要额外分配 838 MiB；此时完整 CUDA 进程只剩约 1.21 GiB 可用空间，考虑 fragmentation、graph private pools 和其他 rank 差异后发生 OOM。把 fraction 降为 0.88：

- KV quota 减少 1.176 GiB；
- rank-0 PyTorch allowance 增加 1.219 GiB；
- 仍保持超过 48 sequences/rank 的 capacity。

### 8.2 uncapped M36 反例证明 fraction 不是固定 headroom

Job 491622 使用与最终 stack 相同的 replay 和 fraction=0.88，但不设置绝对 `max_tokens`。M36 的 recurrent state 比 M104 少，KV manager 将释放出来的空间重新用于 attention KV：

| Matched M36 case | 仅 fraction、无上限 | `max_tokens=479232` | cap 的效果 |
|---|---:|---:|---:|
| Total manager quota | 62.918 GiB | 34.189 GiB | **-28.728 GiB** |
| Attention-KV tokens | 1,075,436（反推） | 479,232 | -55.44% |
| 9216-token sequences/rank | 116.69（反推） | 52.00 | -64.69 |
| Rank-0 PyTorch allowance | 207.735 GiB | 236.454 GiB | **+28.719 GiB** |
| DeepGEMM gather finalize | 申请 830 MiB，仅余 236.94 MiB，OOM | formal run 成功 | transient headroom 恢复 |

反推使用 manager V2 的真实 `max_util_for_resume=0.95` 和 49152 bytes/token：

```text
capped attention resumable cost
= 479232 tokens × 49152 bytes/token
= 21.9375 GiB

capped attention quota-space cost
= 21.9375 / 0.95
= 23.092105 GiB

M36 fixed/context resumable intercept
= 34.189212 × 0.95 - 21.9375
= 10.542252 GiB

uncapped attention resumable bytes
= 62.917689 × 0.95 - 10.542252
= 49.229553 GiB

uncapped attention tokens
= 49.229553 GiB / 49152 bytes
= 1,075,436 tokens
= 116.692 average 9216-token sequences/rank
```

M36 scheduler 最多只能 resident 36 requests/rank，但 uncapped KV 却为约 116.7 个平均 sequence/rank 预留空间，超过 scheduler capacity 三倍。`max_batch_size` 不会自动约束 KV token pool。

所以：

> `free_gpu_memory_fraction` 是 allocator 对“当前剩余内存”的反馈，不是稳定的 non-KV headroom contract。任何 state-memory 优化都可能被 KV allocator 立即重新占用。

可靠配置必须同时使用：

- fraction 作为设备级软预算；
- absolute `max_tokens` 作为 attention KV 硬上限；
- `max_tokens >= concurrency/ADP × average_sequence_length` 保证 residency；
- cap 后剩余空间必须覆盖 graph pool、communication workspace 和最大 GEMM transient。

最终使用：

```text
max_tokens = 52 × 9216 = 479232 tokens/rank
```

它比所需 48 sequence/rank 多 4 个 sequence margin，同时避免让 KV 吞掉约 28.7 GiB transient headroom。

## 9. FlashInfer low-M direct/Split-K 是否是 regression 根因

结论：不是。但此前把失败称为“Split-K OOM”不准确。实际 traceback 显示
`triton_fused_gather_finalize` 在 `torch.empty((num_rows, unpadded_hidden_size))` 分配
DeepGEMM MoE final output 时 OOM；并没有证据表明 830–838 MiB 是 FlashInfer Split-K
partial workspace。

### 9.1 matched KV=0.85 端到端对照

| 配置 | Total tok/s/GPU |
|---|---:|
| FlashInfer low-M auto，direct/Split-K available，Job 489081 | 3974.603 |
| low-M off，Job 489194 | 3917.025 |

low-M auto 相对 off：

- total throughput：**+1.470%**；
- duration：-1.449%；
- median TPOT：-3.167%；
- median TTFT：+1.571%。

在 KV/headroom 相同的条件下，FlashInfer low-M auto 是正收益。这里的 `auto` 只表示允许
FlashInfer 内部 direct/Split-K heuristic；serving log 没有记录每个 shape 的最终 tactic，不能
声称所有 route 都实际选择了 Split-K。

### 9.2 production-shape micro-benchmark

Job 489183：

- device：GB300；
- 72 个 shape；
- CUDA graph；
- PDL enabled；
- 每个 shape 100 replays × 7 trials；
- FlashInfer/low-M 在 63/72 shape 上更快；
- representative existing/FlashInfer median ratios：
  - N=256：1.3308；
  - N=4096：1.1997；
  - N=8192：1.0718；
- 最大 BF16 absolute error：0.03125。

该 microbenchmark 比较的是 `torch.nn.functional.linear` 与 FlashInfer `mm_bf16`，不是
DeepGEMM microbenchmark。正确做法不是因这次 OOM 关闭 low-M auto，而是用绝对 KV cap 为
已观察到的 DeepGEMM MoE output、CUDA graph pools、communication workspace、可能的 low-M
temporary 和 fragmentation 共同留出空间。已有 low-M off 的 M104 runs 达到 4114.062 和
4110.993 tok/s/GPU，因此 low-M auto 是有匹配正收益的可选优化，不是 >4k 必需项。

## 10. 稳定性和复现性

修复后的 M104 family 在四个独立 allocation 上得到：

| Job | Low-M | KV fraction | Total tok/s/GPU |
|---:|---|---:|---:|
| 488899 | off | 0.90 | 4114.062 |
| 488993 | off | 0.90 | 4110.993 |
| 489192 | auto | 0.88 | 4112.350 |
| 489281 | auto | 0.88 | 4065.823 |
| **Mean** | — | — | **4100.807** |

统计量：

- 四个 repaired-family run：CV=0.493%，range/mean=1.176%，全部超过 4k；
- low-M-off 两次 repeat：CV=0.037%；
- 最终选择的 low-M-auto/KV=0.88 两次：mean=4089.086，CV=0.569%，range/mean=1.138%。

四个 run 不是完全相同 recipe，因此 family CV 用于证明“修复后的机制跨 allocation 稳定”，而不是把四个值错误地当作严格重复。严格的单变量 residency 证明来自同 allocation Job 491712。

这一组数据足以排除“只有某一次节点重启后才达到 4k”的解释：同一修复机制在不同 allocation、不同安全 headroom 组合上重复超过 4k。

## 11. 排除的混杂因素

### 11.1 `/metrics`

所有正式窗口均未请求 TRT-LLM JSON `/metrics` endpoint。full-log audit 和 formal-window audit 都没有发现 `GET /metrics`。response-level performance metrics 也保持 disabled。因此之前确认的 metrics statistics-consumer stall/race 不存在于这些结果中。

### 11.2 loading heartbeat

heartbeat 只在合法 Slurm allocation 的模型加载阶段使用：

- M36 在权重加载 70%–72% 时自动停止；
- M48 在权重加载 71%–74% 时自动停止。

它在 KV allocation、JIT/tuning、CUDA graph capture、warmup 和 formal measurement 之前已经停止，不会污染正式窗口。

### 11.3 JIT 和 cache 冷启动

- 两个 A/B arm 使用隔离的 XDG/CUDA/Triton/TorchInductor/FlashInfer cache；
- JIT、tuning 和 graph capture 全部在 formal 前完成；
- 每个 arm 有三倍 concurrency 的独立 warmup population；
- formal-window audit 没有 JIT/tuning/graph-capture signature。

GDN micro-benchmark 也有 50 warmup + 500 replay，并有 fixed-vs-fixed immutable repeat。因此收益不是把 JIT 冷启动错误计入 baseline。

### 11.4 节点和集群波动

- 最关键 M36/M48 control 在同一 Job、同一 node segment 内顺序执行；
- repaired family 又跨四个 allocation 重复；
- 直接结论依赖单 allocation A/B、exact-shape microbench 和确定性内存公式，而不是历史 cross-node 差值。

节点状态仍可能解释约 1% 的普通 run-to-run noise，但不能解释 8% GDN microbench、112.12-GiB 精确 OOM、90.13% replay memory reduction 或 M36/M48 residency cliff。

### 11.5 输出轨迹变化

MTP 生成轨迹会影响 expert routing 和 scheduler timing。一组 audited pair 的逐 index exact generated-text fraction 只有 13.997%。因此，本报告不把任意 cross-allocation 的约 1% 差异当作因果证据，也不会把各个百分比简单相加。

## 12. 为什么各项收益不能直接相加

以下数字作用在不同瓶颈和不同 operating point：

- GDN T=1 exact-shape microbench：约 +8%；
- GDN-only 端到端隔离：+6.462%；
- no-chunk directional control：+4.217% 到 +9.504%；
- M36 → M48 residency：+3.746%；
- low-M auto → off：+1.470%；
- replay scratch：减少 90.133%，是容量变化，不是同 batch latency 百分比。

它们不能相加，因为：

- T=1 修复改变每个 decode cycle；
- replay 改变可行 batch 和 allocator geometry；
- no-chunk 改变 prefill population ramp；
- M48 改变请求波次；
- KV cap 改变可用 transient headroom；
- FlashInfer direct/Split-K heuristic 改变特定 low-M BF16 GEMM latency。

每次修复后 bottleneck 会转移。最终 4120 tok/s/GPU 是这些机制在同一 operating point 下的组合结果，不是独立 speedup 的线性和。

## 13. 证据强度分级

### 已直接证明

- T=1 static GDN 在生产 shape 上比旧 dynamic specialization 快约 8%，数值 bit-exact；
- 只替换 GDN overlay 的端到端吞吐提高 6.462%；
- 旧 M104 intermediate state 的公式为 112.125 GiB，并与 112.12-GiB OOM 匹配；
- replay 将 verify scratch 从 115.410 降至 11.388 GiB；
- 同 allocation、同 KV cap 的 M48 比 M36 高 3.746%，同时 TPOT 变差，证明最高吞吐来自 residency/ramp；
- fraction-only 的 uncapped M36 会把释放内存转成过大的 KV pool 并在 DeepGEMM gather finalize OOM；
- headroom 足够时 FlashInfer low-M BF16 auto 比 off 快 1.470%；
- 修复后的 family 可以重复达到 4065–4114 tok/s/GPU。

### 强方向性证据，但不单独作为根因证明

- no-chunk 在同 allocation 的两个 arm 都明显改善 TTFT 和 aggregate ramp；
- legacy M48/no-replay 很慢，说明只改 max batch 不足；
- M104 KV fraction 0.85/0.88/0.90 显示 residency capacity 与 transient headroom 的窄窗口。

### 当前没有声称

- 没有声称 replay 会使同一 M36 kernel latency 更快；
- 没有声称 0.88 对所有 batch/state geometry 都安全；
- 没有声称每个 1% 左右的跨 allocation 差异都能归因于某个 kernel；
- 没有把不同实验的百分比相加；
- 没有把节点重启作为性能恢复原因。

## 14. 面向未来的 failure-check 方法

### 14.1 先检查 workload/correctness invariant

吞吐分析前必须拒绝：

- incomplete requests；
- nonempty errors；
- empty/garbage generated text；
- 实际 ISL/OSL 与目标不一致；
- GPU denominator 或 token definition 不一致；
- formal window 内出现 `/metrics`、JIT、graph capture 或 heartbeat。

### 14.2 区分 kernel regression 与 population-ramp regression

| 观察签名 | 优先检查 |
|---|---|
| TPOT 和 exact-shape kernel 同时退化 | specialization、cache identity、kernel selection |
| TTFT/peak output/makespan 明显变化，但 TPOT 不改善或变差 | resident slots、prefill chunking、request waves |
| 服务器 ready 后在 GEMM transient OOM | KV over-provision、graph pool、workspace、allocator feedback |
| 只有第一次运行慢 | JIT/tuning/cache cold start 是否进入 formal |
| 约 1% 跨节点差异且 output trajectory 不同 | 先视为 run-to-run/routing noise，不作因果归因 |

### 14.3 每次运行前计算容量

至少记录：

```text
requests_per_rank = concurrency / ADP
global_resident_slots = max_batch_size × ADP
initial_queued = max(0, concurrency - global_resident_slots)
required_kv_tokens_per_rank = requests_per_rank × (ISL + OSL)
```

同时检查 absolute KV cap 是否高于 required tokens、又是否给 transient 留有空间。不要在 residency cliff 两侧比较配置却不明确标注。

### 14.4 联合记录所有 GPU memory owner

必须同时审计：

- model weights；
- attention KV；
- Mamba/GDN state；
- PyTorch allocated/reserved；
- CUDA graph private pools；
- frontend CUDA context；
- NCCL/NVLink/communication workspace；
- replay/verification scratch；
- DeepGEMM MoE output/workspace、FlashInfer low-M 可能的 temporary，以及其他最大 serving transient；

只看 `nvidia-smi free` 或一个 allocator 的 quota 不足以证明 headroom。

### 14.5 推荐的因果实验层级

1. 同 allocation、单变量 A/B；
2. exact production-shape micro-benchmark + correctness/bit-exact check；
3. 确定性 tensor-memory 公式和 OOM allocation 对齐；
4. 独立 allocation repeats；
5. 历史 cross-allocation 结果只作背景，不作为第一因果证据。

## 15. 最终推荐配置原则

对于当前 FP8 high-throughput、MTP3、RR=1.0、C1536 workload：

- 保留 T=1 GDN exact static specialization/cache fix；
- 启用 GDN cached replay V2；
- `max_batch_size >= concurrency / ADP = 48`；
- no-chunk prefill，`max_num_tokens` 足以容纳 8192 context 加调度余量；
- KV 同时设置 fraction 和 absolute token cap；当前验证值为 0.88 + 479232；
- 可保留 FlashInfer low-M BF16 auto（direct/Split-K heuristic）；必须给已观察到的 830–838 MiB DeepGEMM MoE output、graph pools、communication workspace 和 fragmentation 留有额外余量；
- 使用独立 cache，完成 JIT/graph capture 和明确 warmup 后才开始 formal；
- formal job 全程禁止 `/metrics`；
- 每个结果都重新检查固定工作量、完成请求数、实际长度和 GPU denominator。

## 16. 原始证据和可复现入口

### 机器可读审计

- `final-rerecheck/audits/ht4000/final-fix-causal-decomposition-20260816.json`
- `final-rerecheck/scripts/analyze_final_fix_root_cause.py`
- `final-rerecheck/audits/ht4000/final-fix-root-cause-artifact-hashes-20260816.md`

### 最终 M36/M48 直接 A/B

- Recipes：`final-rerecheck/recipes/perf/final-fix-residency-kvcap-ab-20260816/`
- Runner：`final-rerecheck/scripts/run_mtp3_final_fix_residency_kvcap_ab_20260816.sbatch`
- Job 491712 output：`final-rerecheck/outputs/final-fix-residency-kvcap-ab-20260816/`
- M36 result：`final-rerecheck/outputs/final-fix-residency-kvcap-ab-20260816/m36-kvcap-a/491712/logs/sa-bench_isl_8192_osl_1024/results_concurrency_1536_gpus_32.json`
- M48 result：`final-rerecheck/outputs/final-fix-residency-kvcap-ab-20260816/m48-kvcap-b/491712/logs/sa-bench_isl_8192_osl_1024/results_concurrency_1536_gpus_32.json`

### 负向和 micro-benchmark 证据

- Uncapped fraction negative control，Job 491622：`final-rerecheck/outputs/final-fix-residency-ab-20260816/m36-a/491622/`
- GDN exact-shape micro-benchmark，Job 467068：`final-rerecheck/outputs/ht4000/gdn-t1-exact-shape-microbench/467068/`
- GDN immutable repeat，Job 467826：`final-rerecheck/outputs/ht4000/gdn-t1-immutable-exact-shape/467826/`
- Replay memory audit：`final-rerecheck/audits/ht4000/gdn-replay-resource-audit.json`
- FlashInfer low-M BF16 graph micro-benchmark，Job 489183：`final-rerecheck/outputs/microbench/low-m-graph/job-489183/`
- no-chunk A/B/A，Job 488694：`final-rerecheck/outputs/mtp3-nochunk-factorial-20260816/`

### 关联报告

- 英文根因报告：`final-rerecheck/reports/final-fix-maximum-throughput-root-cause-20260816.md`
- MTP3 >4k full curve 报告：`final-rerecheck/reports/mtp3-over4k-full-curve-and-root-cause-20260816.md`
- Oakhaven publication summary：`/lustre/fsw/portfolios/coreai/users/williamj/oakhaven-kernels/data/trtllm/agg/oakhaven-max/stack-final-rerecheck-20260816/ROOT_CAUSE.md`

## 17. 最终判定

最终修复之所以能有效提高最高吞吐，是因为它先修复了一个会在每个 decode cycle 重复支付的 T=1 GDN specialization 性能缺陷，又修复了一个阻止大 resident batch 的 GDN state-lifecycle 容量缺陷。容量修复使每 rank 至少 resident 48 个请求成为可能；M48 由此消除了 C1536 下的第二调度波。no-chunk 加快初始 population ramp，绝对 KV cap 则阻止 allocator 把释放内存重新变成无用的过量 KV，并为 DeepGEMM MoE output、CUDA graph、communication workspace、FlashInfer low-M temporary 和其他 serving transient 保留空间。

最强的直接证据是 Job 491712：M48 在 TPOT 恶化 29.021% 的情况下，仍把固定工作量吞吐从 3971.592 提高到 4120.374 tok/s/GPU，并将 TTFT 降低 71.032%。这个现象只能由更好的 population residency/ramp 解释，不能由“decode kernel 更快”、JIT、节点重启或 metrics 干扰解释。

因此，该修复应被视为一个真实的 **kernel specialization bug fix + speculative-state capacity/lifecycle bug fix**；最终 `>4k tok/s/GPU` 是修复后的 stack 跨过 resident-batch threshold 后的可重复表现，而不是一次偶然的集群高点。
