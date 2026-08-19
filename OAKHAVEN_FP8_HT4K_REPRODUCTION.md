<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Oakhaven-Max FP8 no-MTP / MTP3 超过 4000 tok/s/GPU 复现指南

更新时间：2026-08-19（America/Los_Angeles）

这是本交付分支的主入口。目标是在 32 张 GB300 上，以相同 FP8 checkpoint、并行策略、
DeepGEMM Static544 placement 和固定 8K/1K 请求合同，复现以下经过严格审计的结果：

| 模式 | 选定并发 | 可信参考值（total input+output tok/s/GPU） | 验收线 |
|---|---:|---:|---:|
| no-MTP | 3264 | 4055.616 | >= 4000 |
| MTP3 | 1536 | 4065.823 / 4112.350 / 4110.993 / 4114.062 | 每次 >= 4000；三次独立 allocation CV <= 1.5% |
| MTP3 reference mean | 1536 | 4100.807，population CV 0.493% | 仅作跨集群参考，不是硬编码等值要求 |

`tok/s/GPU` 在本文中始终指：

```text
(全部 input tokens + 全部 output tokens) / formal duration / 32 GPUs
```

它不是 output-only TPS，也不是单用户 decode TPS。

## 1. 交付身份

- 分享分支：`user/williamj/oakhaven-fp8-ht4k-repro-20260819`
- 用户 fork：`https://github.com/Wanli-Jiang/TensorRT-LLM`
- 性能栈基线：`9a6889b2a2aba6f6e44483999dd972bc157c297b`
- 本分支额外 runtime 修复：`f572594361`，把显式 `max_num_tokens` 传递到所有 hybrid
  Mamba/GDN cache-manager construction path。
- 便携交付包：
  `final-rerecheck/delivery/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817`

推荐直接 checkout 本交付分支，不要在此分支上再次应用
`patches/trtllm-pr17537-max-num-tokens-propagation.patch`。该 patch 仅为仍停留在
`9a6889b2a2` 的旧 checkout 提供兼容性。

FlashInfer T=1 static specialization patch 也保留在交付包中，但它不是超过 4k 的必要条件。
最新同 allocation A/B/A 中，未使用 static overlay 的 dynamic GDN 两次分别达到
4049.545 和 4096.202 tok/s/GPU。只有在目标 FlashInfer 版本、correctness 或 kernel
specialization 需要它时，才按 byte-level source hash 显式应用；不要把“microbenchmark
没有收益”当作 high-throughput 失败。

## 2. 复现范围与不可混用项

固定合同：

| 字段 | 值 |
|---|---|
| Checkpoint | `oakhaven-max-final-fp8_vv3`，FP8 only |
| GPU | 32 x GB300，8 nodes x 4 GPUs |
| Parallelism | TP32 / EP32 / ADP32 / PP1 |
| Attention DP | enabled |
| MoE | TensorRT-LLM `DEEPGEMM` backend |
| Placement | Static544；no-MTP 和 MTP3 使用不同 map |
| KV / recurrent state | FP8 attention KV；BF16 Mamba/GDN state |
| Workload | exact ISL=8192、OSL=1024、RR=1.0、request rate=`inf` |
| Measurement | 1 x concurrency warmup；3 x concurrency formal |
| Metrics | response perf metrics off；iteration stats/log off；禁止 `/metrics` |

以下变化都形成新的实验合同，不能直接与 4k reference 比较：RR=0.8、NVFP4、不同
checkpoint、不同 ISL/OSL、不同 GPU generation、不同 GPU denominator、output-only TPS、
不同 placement map 或自然 MTP acceptance。

## 3. 两个选定点的完整配置

### 3.1 MTP3 C1536

```text
TP/EP/ADP/PP                 32/32/32/1
concurrency                  1536 = 48 requests/rank
max_batch_size               48
cuda_graph max_batch_size    48
max_num_tokens               8512
max_seq_len                  9472
chunked prefill              false
MTP max_draft_len            3
forced accepted drafts       2.3
total accept length          about 3.3
GDN cached replay            on
KV manager                   V2
KV free_gpu_memory_fraction  0.88
KV absolute max_tokens       479232 = 52 x 9216 tokens/rank
KV avg_seq_len               9216
low-M BF16 backend           auto (optional small optimization)
DeepGEMM max_num_tokens      65536
MoE A2A workspace            2304 MiB
```

M48 是关键 resident geometry：`1536 / ADP32 = 48 requests/rank`。M36 会让每 rank
12 个请求进入第二波；同 allocation A/B 中，M36 -> M48 将吞吐从 3971.592 提高到
4120.374（+3.746%），即使 median TPOT 反而变差 29.021%。因此主收益来自消除第二波和
改善 population ramp，而不是单 token kernel 变快。

### 3.2 no-MTP C3264

```text
TP/EP/ADP/PP                 32/32/32/1
concurrency                  3264 = 102 requests/rank
max_batch_size               104
cuda_graph max_batch_size    104
max_num_tokens               8448
max_seq_len                  9472
chunked prefill              false
GDN cached replay            off
KV manager                   V2
KV free_gpu_memory_fraction  0.92
KV avg_seq_len               9216
low-M BF16 backend           auto
DeepGEMM max_num_tokens      65536
MoE A2A workspace            2304 MiB
```

C3328 等于 104 requests/rank，已在显存容量边界 OOM，必须标记为 rejected boundary，
不能选择性发布为有效结果。no-MTP 的可信选定点是 C3264。

## 4. 获取分支并验证源码

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

如果 fork 被配置成其他 remote 名称，请替换 `origin`。复现时建议使用 detached commit 或
在本地建立只用于复现的 branch；不要把未知本地修改混入构建身份。

校验交付包：

```bash
DELIVERY_ROOT="$PWD/final-rerecheck/delivery/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817"
"$DELIVERY_ROOT/scripts/verify_bundle.sh"
```

## 5. 构建不可变 wheel / Docker / sqsh

正式结果不能使用 editable install、`.pth`、host `PYTHONPATH` 或 source overlay。它们只允许
用于诊断。详细要求见交付包 `BUILD_AND_PORTING.md`。

最低流程：

1. 记录 TensorRT-LLM commit、完整 `git status`、submodule/FlashInfer revision、base image
   digest、CUDA/driver/PyTorch/TensorRT/Python ABI、GPU/SM。
2. 为 source、build、wheel、Docker、sqsh 和 runtime cache 分配唯一目录；不要复用另一 checkout
   的 editable metadata 或 build tree。
3. 先检查当前 revision 的 `python3 scripts/build_wheel.py --help`，然后执行正常 native build。
   不使用 `--fast_build`、`--skip_building_wheel` 或 `--linking_install_binary`。
4. 用普通 `pip install` 将 wheel 安装进 pinned runtime base，移除旧 editable package。
5. Docker 和 sqsh 必须来自同一 wheel；嵌入 build manifest，并记录 wheel、native library、
   Docker archive/digest、sqsh 的 SHA-256 和大小。
6. 在 `/tmp`、unset `PYTHONPATH`、没有 TensorRT-LLM/FlashInfer host source mount 的 Docker 和
   sqsh 中分别验证 import、`ldd`、loaded library maps 和 `trtllm-serve --help`。

当前 retained sqsh 仅用于原集群 provenance：

```text
/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/containers/
trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh
SHA256 08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59
```

如果目标集群可以访问并且 runtime ABI 兼容，可以先用它做 reference；正式跨集群交付仍建议
从本 branch 构建新的普通 wheel/image/sqsh。

## 6. 准备 target-cluster 实验目录

所有 recipe、dry-run、logs、raw JSON 和报告必须放在本 TensorRT-LLM checkout 下面，禁止写入
srt-slurm repo：

```bash
export TRTLLM_REPO="$PWD"
export DELIVERY_ROOT="$TRTLLM_REPO/final-rerecheck/delivery/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817"
export EXPERIMENT_ROOT="$TRTLLM_REPO/experiments/oakhaven-fp8-ht4k-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$EXPERIMENT_ROOT"/{recipes,outputs,notes,guards}
```

先对 srt-slurm 保存 HEAD 和完整 porcelain status；benchmark 前后必须完全一致。不要清理它的
既有 dirty state，也不要在其中创建 recipe 或 output。

## 7. 渲染 portable recipes

替换下列路径和 Slurm 参数：

```bash
export MODEL_PATH=/shared/models/oakhaven-max-final-fp8_vv3
export CONTAINER_IMAGE=/shared/images/oakhaven-fp8-ht4k.sqsh
export SLURM_ACCOUNT=ACCOUNT
export SLURM_PARTITION=batch
export SLURM_QOS=short
```

MTP3：

```bash
python3 "$DELIVERY_ROOT/scripts/render_recipe.py" \
  "$DELIVERY_ROOT/configs/mtp3-c1536-m48-kvcap479232.yaml.in" \
  "$EXPERIMENT_ROOT/recipes/mtp3-c1536.yaml" \
  --model-path "$MODEL_PATH" \
  --container-image "$CONTAINER_IMAGE" \
  --delivery-root "$DELIVERY_ROOT" \
  --slurm-account "$SLURM_ACCOUNT" \
  --slurm-partition "$SLURM_PARTITION" \
  --slurm-qos "$SLURM_QOS" \
  --gpu-type gb300
```

no-MTP：

```bash
python3 "$DELIVERY_ROOT/scripts/render_recipe.py" \
  "$DELIVERY_ROOT/configs/nomtp-c3264-m104.yaml.in" \
  "$EXPERIMENT_ROOT/recipes/nomtp-c3264.yaml" \
  --model-path "$MODEL_PATH" \
  --container-image "$CONTAINER_IMAGE" \
  --delivery-root "$DELIVERY_ROOT" \
  --slurm-account "$SLURM_ACCOUNT" \
  --slurm-partition "$SLURM_PARTITION" \
  --slurm-qos "$SLURM_QOS" \
  --gpu-type gb300
```

在提交前检查两份渲染 recipe：模型、镜像、8x4 GPU、TP/EP/ADP、placement map、MTP、KV、
environment 和 benchmark 参数必须与第 2/3 节完全一致。

## 8. preflight、dry-run 和 `/metrics` 禁令

```bash
export SRT_REPO=/path/to/srt-slurm
export SRTCTL="$SRT_REPO/.venv/bin/srtctl"
cd "$EXPERIMENT_ROOT"

"$SRTCTL" preflight -f recipes/mtp3-c1536.yaml
"$SRTCTL" dry-run -f recipes/mtp3-c1536.yaml -o outputs
"$SRTCTL" preflight -f recipes/nomtp-c3264.yaml
"$SRTCTL" dry-run -f recipes/nomtp-c3264.yaml -o outputs

grep -RIn '/metrics' recipes outputs || true
```

如果任何 readiness helper、sidecar、reporter、dashboard 或 shell loop 会在 server 生命周期中
请求 JSON `/metrics`，停止提交并移除该行为。即使低频请求也可能等待/消费 engine iteration
statistics，污染 token delivery。健康检查必须使用 documented health endpoint 或独立的非测量
completion。

同时检查 dry-run 展开的 sbatch：不得有 host source overlay、editable install、未解析 marker、
错误 GPU denominator 或 measurement-window probe。

## 9. 执行顺序

建议顺序：

1. image/sqsh independence gate；
2. 一个小的 deterministic semantic serving smoke；
3. MTP3 C1536 selected point；
4. MTP3 C1536 在三个独立 allocation 重复；
5. no-MTP C3264 selected point；
6. 只有 selected point 全部通过后，才扩展完整 curve。

提交示例：

```bash
cd "$EXPERIMENT_ROOT"
"$SRTCTL" apply -f recipes/mtp3-c1536.yaml -o outputs --json -y
"$SRTCTL" apply -f recipes/nomtp-c3264.yaml -o outputs --json -y
```

每个 job 必须监控到 Slurm terminal state；只有 `COMPLETED 0:0` 才可能被接受。不要在 server
刚 ready、warmup 完成或部分 benchmark 完成时停止监控。

Loading heartbeat 只能在合法 allocation 的 checkpoint loading 阶段运行，并必须在 KV cache
setup、JIT/autotune、CUDA graph capture、correctness、warmup 和 formal measurement 之前停止。

## 10. 正式结果验收

每个 accepted JSON 必须满足：

- `completed == planned`；
- 每个 input length 恰好 8192；
- 每个 output length 恰好 1024；
- error list 为空，generated text 全部非空；
- token totals 与 request-level sum 一致；
- 使用 32 GPU 重新计算 total tok/s/GPU，与 JSON 一致；
- formal window 内无 `/metrics`、heartbeat、health probe、JIT/tuning、graph capture、OOM、
  traceback、CUDA/NCCL/MPI fatal signature 或 silent fallback。

对每个 job 运行 conservative formal-window audit：

```bash
python3 "$DELIVERY_ROOT/scripts/audit_formal_measurement_windows_corrected.py" \
  "$EXPERIMENT_ROOT/outputs/<job-id>" \
  --output "$EXPERIMENT_ROOT/notes/<job-id>-formal-window-audit.json"
```

MTP3 要报告所有 independent repeat，不得只保留超过 4k 的快点：

```text
values, mean, median, min, max, range/mean, population CV
```

跨集群判定：

- `PASS`：请求/日志/身份门槛全部通过；MTP3 三次均 >=4000 且 CV<=1.5%；no-MTP
  C3264 >=4000。
- `FAIL`：身份和请求合同正确，但重复结果持续低于门槛，且 failure analysis 已排除冷启动、
  `/metrics`、容量和明显 allocation fault。
- `INCONCLUSIVE`：OOM、incomplete requests、错误长度、source/image 混用、formal contamination、
  集群故障或不足三次 MTP3 repeat。

## 11. 常见 failure pattern

| 现象 | 优先检查 |
|---|---|
| 只有第一次慢 | JIT/tuning/graph capture 是否进入 formal；cache 是否跨 GPU/revision 误用 |
| TTFT/makespan 变差，TPOT 平或更好 | resident slots、M36 第二波、chunked prefill、population ramp |
| MTP3 约 2.7-2.9k，而 no-MTP 正常 | allocation-sensitive MTP small-token/high-iteration failure；换 chassis 做相同 immutable repeat |
| DeepGEMM ready 后 OOM | KV 是否抢占 transient headroom、absolute cap、graph pool、A2A workspace、gather/finalize output、fragmentation |
| 约 1% 跨 allocation 波动 | 先视为生成轨迹/expert routing/普通噪声，不宣布代码根因 |
| GDN microbench 无收益但 E2E >4k | 正常；static T=1 不是跨 4k 必要条件 |
| `/metrics` access log 存在 | 拒绝该 server 进程产生的所有性能结果 |

不要把 DeepGEMM MoE 与 FlashInfer low-M direct/Split-K 混称为“DeepGEMM Split-K”。保留的
830-838 MiB OOM traceback 位于 DeepGEMM `triton_fused_gather_finalize` output allocation，
不是已证明的 Split-K workspace。

## 12. 交给其他 Codex agent

完整可复制 prompt 位于：

```text
final-rerecheck/delivery/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817/PROMPT.md
```

Agent 必须先阅读本文件、交付包 README、BUILD_AND_PORTING、ACCEPTANCE_CHECKLIST 和
TECHNICAL_CLARIFICATION，再执行构建或 Slurm 提交。它不得修改 srt-slurm，不得轮询
`/metrics`，也不得把 NVFP4 或 RR=0.8 数据混入本次验收。

## 13. 保留证据

- 完整 45-point curve：
  `final-rerecheck/results/full-curves-20260816/nomtp-mtp3-full-curve.json`
- MTP3 稳定复现与完整曲线：
  `final-rerecheck/reports/mtp3-over4k-full-curve-and-root-cause-20260816.md`
- 最新 GDN T=1 因果修正和同 allocation A/B/A：
  `final-rerecheck/reports/GDN_T1_AND_HIGH_THROUGHPUT_4000TPS_ROOT_CAUSE_REAUDIT_ZH_20260818.md`
- no-MTP/MTP3 paired retest：
  `final-rerecheck/reports/paired-retest-20260815.md`
- 便携 package 内的 Static544 maps、templates、auditor、evidence 和 source patch：
  `final-rerecheck/delivery/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817/`

分享结果时至少同时给出 branch/commit、image SHA、checkpoint identity、raw JSON、formal-window
audit 和所有 MTP3 repeat，不能只分享一个聚合数字。
