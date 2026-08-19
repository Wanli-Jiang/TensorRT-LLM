<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashInfer GDN T=1 与 FP8 High-Throughput >4000 TPS/GPU 根因复审

生成日期：2026-08-18（America/Los_Angeles）
实验模型：`oakhaven-max-final-fp8_vv3`
主要配置：TP32/EP32/ADP32、DeepGEMM MoE、Static544、RR=1.0、ISL8192/OSL1024
证据截止：2026-08-18T04:16:18-07:00
结论状态：历史数据复审、同 GPU microbenchmark 和同 allocation E2E A/B/A 均已完成。

## 1. 最终结论

之前把 `3784.483 -> 4029.055 tok/s/GPU` 的 `+6.462%` 全部归因于 FlashInfer GDN T=1
static specialization，是错误的因果归因。两个数字来自不同 Slurm allocation；后续同为 static
GDN 的等价运行覆盖 `3781.575--4029.055 tok/s/GPU`，其波动本身已经与所谓 GDN E2E 收益处于
同一量级。旧对照没有控制生成轨迹、expert routing、请求波次和 allocation 状态，不能作为单变量
A/B。

GDN T=1 static specialization 在 GB300 exact shape microbenchmark 上确实更快，而且结果 bit-exact：

- dynamic median：`55.288 us`；
- static median：`51.186 us`；
- 单次调用节省：`4.102 us`；
- kernel latency 减少：`7.420%`。

但这是一个只有约 55 us 的单 kernel。该模型有 69 个 GDN layer，MTP3 每个 speculative cycle
大约执行 3 次 T=1 draft pass。即便假设 207 次调用都完全处于关键路径，每 cycle 也只节省：

```text
4.102 us/call * 69 layers * 3 T=1 passes = 0.849 ms/cycle
```

相对 M36 和 M48 实测 decode cycle 时间，理论占比分别只有约 `0.349%` 和 `0.268%`；即便保守按
4 次 T=1 pass 计算也小于 `0.5%`。因此 7%--8% 的 kernel microbenchmark 改善不可能直接产生
6.462% 的整个 serving E2E 改善。

新的同 allocation A/B/A Job 505210 给出最终直接证据：

```text
dynamic A  = 4049.545 TPS/GPU
static B   = 4064.498 TPS/GPU
dynamic A2 = 4096.202 TPS/GPU
dynamic mean = 4072.874 TPS/GPU
static vs dynamic mean = -0.206%
```

static B 位于 dynamic bracket `[4049.545, 4096.202]` 内；dynamic 自身 range/mean 为 `1.146%`。
static 相对 A 为 `+0.369%`，但相对 A2 为 `-0.774%`，相对 dynamic 均值反而略慢。static 的
median TPOT 相对 dynamic 均值也变差 `0.885%`。因此在该完整 production contract 下，没有从
运行噪声中分离出 static T=1 的正向 E2E 收益，更不支持旧的 `+6.462%`。

稳定超过 4000 tok/s/GPU 的真正主链条是：

1. TensorRT-LLM GDN cached replay 修复 speculative state 的容量和生命周期，使 M48/M104 resident
   batch 成为可能；它不是 FlashInfer T=1 kernel fix。
2. `C1536 / ADP32 = 48 requests/rank`。M36 只能同时 resident 36 个请求，留下 12 个请求/rank
   进入第二波；M48 正好一次 resident 全部请求。
3. 同 allocation 的 M36 -> M48 直接 A/B 把吞吐从 `3971.592` 提高到 `4120.374`，即
   `+3.746%`；同时 median TPOT 反而变差 `29.021%`，但 TTFT 改善 `71.032%`。这是明确的
   residency/population-ramp 收益，不是 decode kernel 变快。
4. no-chunk prefill 让 8192-token prompt 避免至少 8 个 1024-token chunk 的调度和 collective
   边界。同 allocation 的方向性 A/B/A 改善 `4.217%--9.504%`，但波动较大，不能把这个区间
   当作可加到其他收益上的独立固定百分比。
5. 绝对 KV cap `max_tokens=479232` 防止 allocator 把 replay/M48 释放的空间重新全部用于 KV，
   为 DeepGEMM MoE output、CUDA graph pool、通信 workspace 和 fragmentation 留出 transient
   headroom。它主要防 OOM 和保持配置可复现，不是直接 kernel 加速。
6. FlashInfer low-M direct/Split-K auto 在 matched KV=0.85 对照中有 `+1.470%`，但 low-M off
   也有两个 run 达到 `4114.062` 和 `4110.993`，所以它是可选小优化，不是跨 4k 必需条件。

简言之：**真正稳定跨 4k 的“fix”是 state capacity + resident batch geometry + prefill ramp +
allocator headroom 的组合；FlashInfer GDN T=1 static kernel 只是一个微秒级、平台相关的局部优化，
并不是旧报告声称的 6.462% E2E 主因。**

### 1.1 通俗解释：为什么单个请求变慢，系统总吞吐反而提高

需要先区分三个概念：

- **TPOT**：单个请求生成第一个 token 之后，平均每生成一个 token 要等多久；越小表示单请求
  decode 越快。
- **Total TPS/GPU**：所有请求的 input 和 output tokens 除以整个 workload 的完成时间；越大表示
  整个系统吞吐越高。
- **Population ramp**：完成 prefill、已经进入 decode 的请求数量随时间增长的过程。越快达到高
  decode population，GPU 越早进入高吞吐区间。

在 `C1536/ADP32` 下，每个 rank 收到 48 个请求。

M36 最多同时 resident 36 个请求，另外 12 个请求必须排队：

```text
第一波：36 requests/rank
排队：  12 requests/rank
```

第一波竞争较少，因此单请求 TPOT 较好；但排队的 12 个请求要等 slot 释放后才能开始，最终形成
第二波并拉长 workload 尾部。

M48 可以一次 resident 全部 48 个请求：

```text
第一波：48 requests/rank
排队：   0 requests/rank
```

48 个请求同时运行会增加资源竞争，所以单请求 TPOT 变差；但所有请求都更早开始，不再需要第二
调度波，因此全部请求反而更早完成。

| Metric | M36 | M48 | 解释 |
|---|---:|---:|---|
| Median TPOT | 74.296 ms | 95.857 ms | 单个请求慢 29.021% |
| Median TTFT | 27.539 s | 7.978 s | 请求更早进入生成 |
| Fixed-work makespan | 334.149 s | 322.083 s | 全部工作更早完成 |
| TPS/GPU | 3971.592 | 4120.374 | 系统总吞吐提高 3.746% |

可以把它理解成电梯：M36 每趟人少、单趟轻快，但剩余乘客必须等第二趟；M48 一趟更拥挤，
每个人的体验略慢，却能一次运完全部乘客。因此“TPOT 变差、TTFT 和 makespan 改善、总 TPS
提高”是 residency/消除第二波的典型签名，不是 decode kernel 变快的签名。

### 1.2 通俗解释：no-chunk 为什么改善 population ramp

每个请求有 8192 个 input tokens。Chunked 配置的 `max_num_tokens=1024` 会让单个 prompt 被
强制拆成至少 8 个 context chunk：

```text
1024 tokens -> 调度/同步 -> 1024 tokens -> 调度/同步 -> ...
-> 完成 prefill -> 进入 decode
```

每个 chunk 之间都有 scheduler 和 collective 边界。很多请求会长时间停留在 prefill 阶段，
所以 decode population 只能缓慢增长：

```text
decode population: 0 -> 少量 -> 中等 -> 高 occupancy
```

No-chunk 配置使用 `max_num_tokens=8512`，单个 8192-token prompt 不再被 1024-token ceiling
强制拆成 8 段，可以在一个 context scheduling unit 中完成：

```text
8192 tokens -> 完成 prefill -> 进入 decode
```

这样更多请求会更早进入 decode，aggregate output 更快爬升到高位。它改善的是从 prefill 到
decode 的 population ramp，而不是单请求 decode kernel latency。

Job 488694 的结果：

| 配置 | TPS/GPU | Median TTFT | Median TPOT | Peak output |
|---|---:|---:|---:|---:|
| Chunked，token budget 1024 | 3552.008 | 35.308 s | 73.439 ms | 39,632 |
| No-chunk B1，token budget 8512 | 3889.608 | 26.446 s | 74.427 ms | 47,903 |
| No-chunk B2，token budget 8512 | 3701.809 | 29.560 s | 77.433 ms | 48,025 |

这里同样能看到：

- TTFT 明显改善，说明请求更早完成 prefill；
- peak output 提高约 21%，说明更早建立了较大的 decode population；
- TPOT 略微变差，说明收益不是来自单请求 decode 更快；
- total TPS 分别提高 9.504% 和 4.217%，说明 fixed workload 更早完成。

“方向性收益 `4.217%--9.504%`”表示两个 no-chunk repeat 都比 chunked 快，因此收益方向可信；
但两个 repeat 的幅度不同，说明精确数值受到生成轨迹、调度和 tail completion 影响。该实验还
同时改变了 chunking 和 full-prompt 所需 token budget，因此不能宣称 no-chunk 固定提高 9.5%，
也不能把它与 M48 的 3.746% 直接相加。

一句话总结：**M48 解决“还有请求在门外排队”；no-chunk 解决“请求在 prefill 流程中推进太慢”。
两者都让系统更早进入高 occupancy 并更早完成全部工作，但不一定让单个请求的每个 decode token
更快。**

## 2. 必须区分的两个 GDN 改动

“GDN fix”这个简称曾经混合了两个完全不同的层次。

| 改动 | 所在项目 | 解决的问题 | 已证明的主要作用 |
|---|---|---|---|
| GDN T=1 batch/layout specialization 与 compile-cache identity | FlashInfer | T=1 cubin 的静态 batch/layout 条件没有被 cache key 和 compile signature 完整表达 | correctness；某些 GPU/compiler/shape 上有数微秒 kernel 收益 |
| GDN cached replay | TensorRT-LLM | MTP verify 物化巨大的 intermediate SSM state，且 state 生命周期不适合大 resident batch | 将 verify scratch 从约 115.410 GiB/GPU 降到 11.388 GiB/GPU，使 M48/M104 可行 |

两者都与 GDN 有关，但不能把 cached replay 带来的容量收益记到 FlashInfer T=1 kernel 上。

### 2.1 交付包与历史 overlay 也不是同一源码对

`gdn-b1-fix/flashinfer-gdn-t1-cache-fix-delivery-20260818` 中最终交付补丁的源码身份是：

| 交付包 arm | SHA256 |
|---|---|
| baseline | `c4268cd8dfb14648c1212ad789e79da8fd63112eb0cf3facdaa2f28d36c5a844` |
| fixed | `b8248980f5064d4851a36318a7b878cc82426dc270e30b45f50b0f2eb1f381b6` |

历史 >4k E2E 归因实际使用的是：

| 历史 arm | SHA256 | 含义 |
|---|---|---|
| dynamic | `61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61` | FlashInfer 0.6.17 路径 |
| static | `4982b5a9d20d9b18588020ab3e938238c9692ffab6265c3533f4b7cf8309a8fe` | 0.6.15-style static T=1 overlay |

两对代码解决的是同一类 dynamic descriptor/static specialization 问题，但 commit base、周边代码和
测试覆盖不同。其他 cluster 对交付包做 microbenchmark 时，不能直接拿它与旧 E2E 的 61de/4982
数字等同。

### 2.2 为什么 cache fix 本身不应被理解成稳态吞吐优化

补丁做了三类事情：

- cache key 加入具体 B、shape、stride、dtype 和 index descriptor；
- T=1 placeholder/descriptor 使用实际 B 和真实 layout；
- T=1 launch grid 和 compile signature 显式包含 static batch；T>=2 继续 batch-dynamic。

其中 cache-key 修复主要保证“当前 cubin 是否适用于当前调用”。在 CUDA graph capture 完成后，
正式 replay 不再每 token 经过 Python compile-cache lookup，所以仅仅把 cache key 写对不会产生
持续的 graph-replay E2E 收益。稳态性能只可能来自新生成 cubin 中的 concrete descriptor、constexpr
batch/grid 等 codegen 差异。

这也解释了为什么该补丁可以同时满足以下事实：

- 它是必要的 correctness fix；
- GB300 某形状上 kernel 快约 7.4%；
- B200 交付包测到 CUDA graph 只快 2.90%；
- 另一个 cluster 在其 compiler/hardware/shape 下可能测不到显著收益。

## 3. 新的严格 GDN microbenchmark

### 3.1 实验设计

Job `505209` 在单个 GB300、同一进程中同时加载 dynamic 和 static module：

- shape：`B36,T1,H16,HV128,K128,V128`；
- 两个 arm 各自先做 100 次 warmup；
- 每 trial 2000 次 CUDA graph replay；
- 共 11 个 trial；
- 顺序交替为 AB、BA、AB、BA，抑制温度、频率和测量顺序漂移；
- output 和 state 都要求逐元素 exact equal。

### 3.2 结果

| Metric | Dynamic | Static | Static 相对变化 |
|---|---:|---:|---:|
| Median CUDA graph latency | 0.055288 ms | 0.051186 ms | -7.420% |
| Absolute latency | 55.288 us | 51.186 us | -4.102 us |
| Trial CV | 0.00440% | 0.00490% | 两边都非常稳定 |
| Output exact equal | — | true | pass |
| State exact equal | — | true | pass |

这个结果重复证明了“GB300 上该 exact shape 的 static cubin 确实更快”，同时也把收益的绝对量级
钉在 `4.102 us/call`。相对百分比看起来很大，是因为分母只有约 55 us。

### 3.3 Amdahl 分解

MTP3 accept length 为 3.3，一个 speculative cycle 的近似时间可由 `TPOT * 3.3` 得到。

| Serving point | TPOT | 近似 cycle time | 207 个 T=1 GDN call 的总节省 | 占 cycle 比例 |
|---|---:|---:|---:|---:|
| 历史 M36 | 73.713 ms/token | 243.253 ms | 0.849 ms | 0.349% |
| 最终 M48 | 95.857 ms/token | 316.328 ms | 0.849 ms | 0.268% |

这是偏向高估 GDN 收益的计算：它假设所有 kernel saving 都严格串行且处于关键路径，没有被其他
GPU work、通信或调度覆盖。即使如此，结果也比旧的 `6.462%` 小一个数量级以上。

## 4. 为什么历史 +6.462% 不是有效单变量证据

历史配对为：

| Job | GDN | TPS/GPU | Duration | Median TTFT | Median TPOT | Peak output |
|---:|---|---:|---:|---:|---:|---:|
| 466450 | dynamic | 3784.483 | 350.670 s | 34.886 s | 73.713 ms | 39,640.8 |
| 467197 | static | 4029.055 | 329.383 s | 32.731 s | 69.562 ms | 39,192.2 |

表面上 total TPS 提高 6.462%，但 peak output 反而下降 1.132%。这不是“decode kernel 全面变快”
应有的清晰签名；它更像固定 workload 的 population/tail/trajectory 发生了变化。

后续 static 运行如下：

| Job | Static 路径身份 | TPS/GPU |
|---:|---|---:|
| 467197 | source overlay | 4029.055 |
| 467781 | source overlay repeat | 3998.881 |
| 468021 | exact worktree overlay | 3781.575 |
| 467553 | immutable image | 3847.456 |
| 467716 | immutable image repeat | 3862.037 |

最严格的三个 exact/immutable static 运行 `468021/467553/467716`：

- mean：`3830.356 TPS/GPU`；
- 相对单个 dynamic 466450：仅 `+1.212%`；
- 三个 static run 的 range/mean：`2.101%`。

如果把五个 static 运行全部放在一起：

- mean：`3903.801 TPS/GPU`；
- 相对 dynamic 466450：`+3.153%`；
- static 内部 range/mean：`6.339%`。

也就是说，static 组内部波动已经与旧的 `+6.462%` 几乎相同。更直接地，Jobs 467781 与
468021 的配置/source semantics 等价，但 TPS 相差 `5.434%`；逐 index generated-text 只有
`644/4608 = 13.976%` 完全相同，而两者 peak output 都约 40k。生成 token 不同会改变 expert
routing、负载不均、MTP 调度和 tail completion，足以制造看起来像 kernel 优化的 cross-allocation
差异。

因此，旧的 466450/467197 对照只能作为“曾观察到相关性”的历史点，不能继续作为 GDN E2E
因果百分比。

## 5. 同 allocation E2E A/B/A 复测

Job `505210` 使用同一组 8 个 GB300 节点，按以下顺序重启完整 server：

```text
dynamic A -> static B -> dynamic A2
```

固定合同：

- TP32/EP32/ADP32、DeepGEMM MoE、Static544；
- MTP3，forced accepted draft tokens=2.3，accept length=3.3；
- C1536、RR=1.0、ISL8192/OSL1024、4608 formal requests；
- M48、no-chunk、`max_num_tokens=8512`；
- GDN cached replay on；
- KV V2，fraction=0.88，absolute `max_tokens=479232`；
- low-M auto、DeepGEMM max tokens 65536、A2A workspace 2304 MiB；
- 每个 arm 使用独立 JIT/cache 目录，JIT 和 CUDA graph capture 全部在 formal window 之前完成；
- 不调用 `/metrics`；loading heartbeat 在权重加载边界自动停止。

三个 recipe 除 arm 名、GDN module mount 和必须隔离的 cache 路径之外相同。

### 5.1 最终结果

<!-- E2E_ABA_RESULT_START -->

Job 505210 于 `2026-08-18T04:16:18-07:00` 以 `COMPLETED 0:0` 结束，总 elapsed
`01:35:24`。三个 formal arm 的结果为：

| Metric | Dynamic A | Static B | Dynamic A2 | Dynamic mean | Static vs dynamic mean |
|---|---:|---:|---:|---:|---:|
| TPS/GPU | 4049.545 | 4064.498 | 4096.202 | 4072.874 | **-0.206%** |
| Duration | 327.717 s | 326.511 s | 323.984 s | 325.850 s | +0.203%（更慢） |
| Median TTFT | 9.482 s | 9.387 s | 9.235 s | 9.359 s | +0.304%（更差） |
| P99 TTFT | 79.605 s | 80.935 s | 78.861 s | 79.233 s | +2.148%（更差） |
| Median TPOT | 97.403 ms | 97.765 ms | 96.411 ms | 96.907 ms | +0.885%（更差） |
| P99 TPOT | 107.339 ms | 106.986 ms | 105.175 ms | 106.257 ms | +0.687%（更差） |
| Peak aggregate output | 57,234.5 | 57,441.1 | 57,381.3 | 57,307.9 | +0.232% |

关键比较：

- dynamic bracket：`[4049.545, 4096.202] TPS/GPU`；
- dynamic range/mean：`1.146%`；
- static B 位于 dynamic bracket 内：`true`；
- static vs dynamic A：`+0.369%`；
- static vs dynamic A2：`-0.774%`；
- static vs dynamic mean：`-0.206%`。

生成轨迹的逐 index exact match：

| Pair | Match | Fraction |
|---|---:|---:|
| Dynamic A vs Static B | 670/4608 | 14.540% |
| Static B vs Dynamic A2 | 664/4608 | 14.410% |
| Dynamic A vs Dynamic A2 | 670/4608 | 14.540% |

static 与 dynamic 的轨迹差异并不高于 dynamic 两次自身的差异。这个结果与 TPOT/TTFT 一起说明，
三个点的约 1% 摆动由 serving trajectory/tail 主导，不能解析成 static kernel 的 E2E 收益。

三个 arm 均通过以下审计：

- `completed=num_prompts=4608`；
- 4608 个 input length 全为 8192，output length 全为 1024；
- 4608 个 generated texts 全部非空；
- error count=0；
- 固定工作量重算 TPS 与 JSON 报告完全一致；
- full-log 和 formal-window audit 都没有 fatal signature 或 `GET /metrics`；
- loading heartbeat 分别在权重加载 70%、70%--72%、72% 时自动停止，早于 KV setup、autotune、
  warm-up 和 formal window；
- dynamic A/A2 module SHA256 均为 `61de9ffa...e61`，static B 为 `4982b5a9...a8fe`；
- srt-slurm HEAD 和完整 worktree status 相对提交前 guard 未变化。

最直接的结论是：**不使用 static T=1 overlay 的 dynamic A 和 A2 都已经超过 4000 TPS/GPU；
static B 不仅没有超出 dynamic bracket，相对 dynamic 均值还略慢 0.206%。**

<!-- E2E_ABA_RESULT_END -->

## 6. 真正超过 4000 TPS/GPU 的 cause breakdown

这些因素相互作用，百分比不能相加。下面按“直接性能因果、容量前提、稳定性前提、可选优化、
噪声源”分类。

### 6.1 直接主因：M48 消除第二调度波

`C1536 / ADP32 = 48 requests/rank`：

| max batch | 全局 resident slots | 初始 queued requests | 调度结构 |
|---:|---:|---:|---|
| M36 | 1152 | 384，即 12/rank | 至少两波 |
| M48 | 1536 | 0 | 一波 resident 全部请求 |
| M104 | 3328 | 0 | 有更多容量 margin |

Job 491712 是目前最强的同 allocation 单变量证据。两个 arm 使用同一模型、镜像、节点、GDN
static/replay、no-chunk、KV absolute cap 和 workload，只改变 M36/M48 及对应 graph max batch：

| C1536 formal metric | M36 | M48 | M48 相对变化 |
|---|---:|---:|---:|
| Total TPS/GPU | 3971.592 | 4120.374 | +3.746% |
| Fixed-work makespan | 334.149 s | 322.083 s | -3.611% |
| Median TTFT | 27.539 s | 7.978 s | -71.032% |
| Median TPOT | 74.296 ms | 95.857 ms | +29.021%（变差） |
| Peak aggregate output | 51,864 | 57,370 | +10.616% |

如果收益来自 GDN 或其他 decode kernel 变快，TPOT 应改善；实际 TPOT 明显变差。与此同时，
TTFT、peak output 和 fixed-work makespan 呈现出消除第二波请求的完整签名。因此跨过 4k 的
直接主因是 resident population，而不是单 token latency。

### 6.2 MTP-specific 容量前提：GDN cached replay

旧 verify path 在 `69 layers, M104, T4, HV128, K128, V128, BF16` 下物化：

```text
69 * 104 * 4 * 128 * 128 * 128 * 2 bytes = 112.125 GiB/GPU
```

加 convolution scratch 后总计约 `115.410 GiB/GPU`。Job 488785 的 `112.12 GiB` OOM 与公式
吻合。cached replay 的 audited verify scratch 为 `11.388 GiB/GPU`：

| 项目 | GiB/GPU |
|---|---:|
| Legacy verify scratch | 115.410 |
| Cached replay scratch | 11.388 |
| 节省 | 104.022（90.133%） |

这是决定 M48/M104 是否能运行的容量和 state-lifecycle bug fix。现有证据没有证明 replay 会让
相同 M36 的单个 GDN kernel 更快，也不应把这 90% memory saving 写成 90% 性能收益。

### 6.3 Population-ramp 因素：no-chunk prefill

Job 488694 在同一 allocation 顺序比较 chunked 与两个 no-chunk arm：

| Arm | TPS/GPU | TTFT | TPOT | Peak output |
|---|---:|---:|---:|---:|
| Chunked, token budget 1024 | 3552.008 | 35.308 s | 73.439 ms | 39,632 |
| No-chunk B1, token budget 8512 | 3889.608 | 26.446 s | 74.427 ms | 47,903 |
| No-chunk B2, token budget 8512 | 3701.809 | 29.560 s | 77.433 ms | 48,025 |

相对 chunked，两个 no-chunk arm 的 TPS 分别 `+9.504%`、`+4.217%`，TTFT 和 peak output
明显改善，但 TPOT 都略差。这也是 ramp 改善而非 decode kernel 改善。

这个 benchmark 的总 token 指标中，input 占：

```text
8192 / (8192 + 1024) = 88.889%
```

因此快速完成 prefill 并尽快建立完整 decode population，会显著改善 total-token throughput。
这也是为什么只盯 T=1 decode kernel 容易误判整个 high-throughput 结果。

边界：该实验同时改变 chunking 和 full-prompt 所需 token budget，且两个 repeat 有明显差异；它
证明方向和机制，但不是一个可移植的固定 `+X%` 常数，也未严格证明最终 M48 下 no-chunk 的独立
必要性。

### 6.4 稳定性前提：绝对 KV cap

仅设置 `free_gpu_memory_fraction=0.88` 不是固定 headroom。replay 或较小 recurrent batch 释放
显存后，KV manager 会把释放空间再次用于 attention KV。Job 491622 的 uncapped M36 control
最终为约 1,075,436 KV tokens/rank 预留空间，远高于 M36 实际 resident 需要，并在 DeepGEMM
`triton_fused_gather_finalize` 申请 830 MiB 时只剩 236.94 MiB，随后 OOM。

`max_tokens=479232 = 52 * 9216` 把每 rank attention KV 固定为 52 个平均 sequence，覆盖 M48 的
48-request residency，同时留下 transient headroom。它的价值是防止 allocator feedback 抵消
replay 的内存收益，使 A/B 和生产复现稳定；不是直接提高 GEMM/GDN latency。

### 6.5 可选小优化：FlashInfer low-M direct/Split-K

Matched KV=0.85 的 endpoint control：

| 配置 | TPS/GPU |
|---|---:|
| low-M auto | 3974.603 |
| low-M off | 3917.025 |

auto 相对 off 为 `+1.470%`。但以下 low-M off M104 runs 已超过 4k：

- Job 488899：`4114.062`；
- Job 488993：`4110.993`。

因此 low-M/Split-K 是有 matched 正收益的可选优化，而不是超过 4k 的必要条件。830--838 MiB
OOM 的实际 traceback 位于 DeepGEMM MoE gather/finalize output，不能错误归因成 FlashInfer
Split-K partial workspace。

### 6.6 不能忽略的噪声：生成轨迹和 expert routing

MTP/high-concurrency run 即使 input/output length 和 request count 完全相同，也可能生成不同 token。
token 内容改变 MoE expert routing、通信不均、batch compaction、slot reuse 和 tail completion。

已经观察到：

- static equivalent Jobs 467781/468021 只有 `13.976%` generated texts 对齐，但 TPS 相差
  `5.434%`；
- M36/M48 Job 491712 只有 `634/4608 = 13.759%` generated texts 对齐；
- final M104 repaired family 的四次 TPS range/mean 为 `1.176%`，即使机制稳定也仍有约 1%
  普通跨 allocation 摆动。

因此任何约 1% 的 cross-allocation 差异都不应单独宣布为根因；约 5%--6% 的结果也必须先有
同 allocation A/B/A、trajectory 审计和机制签名才能归因。

## 7. 为什么该组合能稳定超过 4k

四个独立 M104 repaired-family run：

| Job | Low-M | KV fraction | TPS/GPU |
|---:|---|---:|---:|
| 488899 | off | 0.90 | 4114.062 |
| 488993 | off | 0.90 | 4110.993 |
| 489192 | auto | 0.88 | 4112.350 |
| 489281 | auto | 0.88 | 4065.823 |
| Mean | — | — | 4100.807 |

四次均超过 4k，CV=`0.493%`；其中 low-M-off 两次 CV 只有 `0.037%`。稳定性不是来自节点重启，
而是来自把系统移出 M36 的 residency cliff：当所有 C1536 request 一开始都能 resident，尾部不再
依赖第二波何时进入，生成轨迹对 makespan 的放大作用随之减小。

no-MTP full curve 也给出独立的几何佐证：

- C1536 只有 `3500.194 TPS/GPU`；
- C3264 达到 `4055.616 TPS/GPU`；
- `3264/ADP32 = 102 requests/rank`，与 M104 的 104 slots 基本对齐；
- C3328 正好是 `104 requests/rank` 的 capacity edge，并在显存边界失败。

no-MTP 不需要 MTP cached replay，却仍然只在 resident population 接近 M104 饱和时跨过 4k。
这是“4k 来自 occupancy/residency，而不是 MTP GDN T=1 fix”的另一条强证据。

## 8. 为什么其他 cluster 的 microbenchmark 可能没有收益

当前 `gdn-b1-fix` 目录包含交付包、补丁、报告和 B200 结果，没有包含用户所说的另一个 cluster
原始 log，所以不能对那个具体 zero-gain run 的唯一原因做事实判定。但以下解释都符合源码和现有
数据：

1. **硬件差异。**同一思路在 GB300 测到 `7.420%`，交付包 B200 CUDA graph 只有 `2.895%`。
2. **compiler/CuTe/CUTLASS 差异。**收益来自 dynamic descriptor 与 static constexpr 的 codegen
   差异；不同 CUDA、nvidia-cutlass-dsl 或编译器可能把 dynamic 路径优化到相同机器码。
3. **shape 不同。**必须核对 B、T、H、HV、K、V、stride、packed QKV、state pool layout 和 fallback
   路径。只写“GDN T=1”不足以保证命中同一个 kernel。
4. **baseline 已包含等效特化。**若目标分支或缓存中已经是 static B/layout cubin，再应用 cache
   correctness fix 不会产生第二次性能收益。
5. **JIT/缓存污染。**新修复会为不同 T=1 B/layout 产生独立 cubin；若把首次编译混入测量，或者
   baseline/fixed 复用错误 cache 目录，结果会被启动成本或旧 cubin 污染。
6. **测量分辨率。**绝对差只有数微秒。需要同进程 AB/BA、CUDA Events、足够 replay 次数和 CV，
   单次 profiler sample 很容易看不到稳定差异。

最重要的是：即使目标 cluster 的严格 microbenchmark 最终确认 steady-state gain=0，这也不会推翻
cached replay 的 correctness/capacity 价值，更不会推翻 M48 residency 的 +3.746% matched E2E
证据。它只会进一步确认 FlashInfer T=1 部分不是 >4k 主因。

## 9. 修正后的因果等级

| 因素 | 证据等级 | 可下结论 | 不可下结论 |
|---|---|---|---|
| FlashInfer T=1 cache/static fix | exact-shape same-GPU interleaved microbench + correctness tests + Job 505210 E2E A/B/A | correctness 必要；局部 kernel 收益硬件相关；当前 contract 未分离出正向 E2E 收益 | 固定 +6.462% E2E；所有 cluster 都有收益 |
| GDN cached replay | tensor geometry + exact 112.12-GiB OOM + footprint audit | 容量/state-lifecycle 因果，使 M48/M104 可行 | 同 M36 kernel 一定更快 |
| M36 -> M48 | Job 491712 same-allocation matched A/B | 直接 +3.746% residency 收益 | 由 decode latency 改善导致 |
| no-chunk | Job 488694 same-allocation directional A/B/A | 明显改善 prefill/population ramp | 固定可加百分比；最终 M48 下独立必要性已证明 |
| absolute KV cap | uncapped OOM + capped matched success + quota audit | 保留 transient headroom、提高可复现性 | 直接增加 kernel TPS |
| low-M auto | matched KV=0.85 endpoint + 72-shape microbench | 小幅正收益、可选 | 超过 4k 必须开启 |
| 生成轨迹 | output-text match audit + static-equivalent run spread | 是重要 cross-run 混杂因素 | 任意差异都可以用“节点问题”解释 |

## 10. 对旧报告的修正

本报告保留旧报告中已经被严格证明的内容：cached replay memory reduction、M36/M48 residency、
no-chunk 方向、KV cap、low-M matched control、`/metrics` 排除和 heartbeat/JIT 边界。

本报告明确取代下列旧结论：

- `final-fix-maximum-throughput-root-cause-zh-20260816.md` 中把 Jobs 466450/467197 的
  `+6.462%` 作为 GDN T=1 E2E 单变量因果；
- `mtp3-over4k-full-curve-and-root-cause-20260816.md` 中把约 8% GDN microbenchmark 直接延伸成
  high-throughput 主因的表述。

旧文件不覆盖，以保留审计历史；今后引用 root cause 时应以本报告为准。

## 11. 推荐的复现和 failure check 方法

### 11.1 判断是否真的跨过 residency cliff

```text
requests_per_rank = concurrency / ADP
```

先检查 `requests_per_rank <= max_batch_size`，再检查 KV token cap、GDN/recurrent state 和 graph pool
是否支持该 resident population。若只差几个 slot，结果可能因第二波请求出现断崖。

### 11.2 把 capacity fix 和 speed fix 分开

- replay/KV cap/OOM：用 footprint、allocation traceback 和最大可 resident batch 验证；
- kernel speed：用 exact shape、同进程 AB/BA、CUDA graph replay 验证；
- E2E speed：同 allocation A/B/A，固定 request count 和实际 lengths，并审计 generated-text
  trajectory。

### 11.3 E2E 必查签名

- TPOT 改善：更可能是 steady-state decode/kernel；
- TTFT/makespan 改善但 TPOT 变差：更可能是 population ramp/residency；
- peak output 不变但 total TPS 大幅变化：检查 tail、request wave 和 trajectory；
- OOM allocation 与预期 workspace 不一致：按 traceback 归属，不按同时开启的 feature 猜测；
- 不请求 `/metrics`，response perf metrics 关闭；
- heartbeat 必须在 KV setup、graph capture、warmup 和 formal window 之前停止；
- 每个 arm 使用独立 JIT/cache 路径，并把首次编译排除在 formal window 外。

## 12. 证据路径

### 新复审

- 实验根目录：`final-rerecheck/gdn-t1-attribution-20260818/`
- GB300 interleaved microbenchmark：
  `final-rerecheck/gdn-t1-attribution-20260818/outputs/microbench/505209/results.json`
- E2E A/B/A runner：
  `final-rerecheck/gdn-t1-attribution-20260818/scripts/run_gdn_t1_e2e_aba.sbatch`
- E2E analyzer：
  `final-rerecheck/gdn-t1-attribution-20260818/scripts/analyze_e2e_aba.py`
- E2E machine-readable comparison：
  `final-rerecheck/gdn-t1-attribution-20260818/outputs/e2e/job-505210/comparison.json`

### 历史 matched controls

- M36/M48 Job 491712：
  `final-rerecheck/outputs/final-fix-residency-kvcap-ab-20260816/`
- no-chunk Job 488694：
  `final-rerecheck/outputs/mtp3-nochunk-factorial-20260816/`
- replay resource audit：
  `final-rerecheck/audits/ht4000/gdn-replay-resource-audit.json`
- causal decomposition：
  `final-rerecheck/audits/ht4000/final-fix-causal-decomposition-20260816.json`
- no-MTP C3264 Job 472461：
  `final-rerecheck/outputs/paired-retest-20260815/nomtp/472461/`
- stable M104 runs：
  `final-rerecheck/outputs/mtp3-v2-m104-replay-splitk-20260816/` 和
  `final-rerecheck/outputs/mtp3-v2-m104-replay-splitk-headroom-20260816/`

### FlashInfer 交付包

- `gdn-b1-fix/flashinfer-gdn-t1-cache-fix-delivery-20260818/`
