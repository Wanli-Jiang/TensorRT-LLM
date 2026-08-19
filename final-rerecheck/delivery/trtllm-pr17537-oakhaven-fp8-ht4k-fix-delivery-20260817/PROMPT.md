<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Target-cluster execution prompt

把下面内容连同本压缩包一起交给目标集群上的 Codex/开发者。先替换尖括号占位符。

---

你正在目标 GB300 Slurm 集群上复测 Oakhaven-Max FP8 no-MTP/MTP3 high-throughput
交付分支。交付包位于：

```text
<DELIVERY_ROOT>/trtllm-pr17537-oakhaven-fp8-ht4k-fix-delivery-20260817
```

目标输入：

```text
TensorRT-LLM repo: <TRTLLM_REPO>
Delivery branch: user/williamj/oakhaven-fp8-ht4k-repro-20260819
Delivery fork: https://github.com/Wanli-Jiang/TensorRT-LLM
FlashInfer repo: <FLASHINFER_REPO>
Model: <MODEL_PATH>/oakhaven-max-final-fp8_vv3
Base/build image: <BASE_IMAGE_OR_SQSH>
srt-slurm repo: <SRT_SLURM_REPO>
Slurm account/partition/qos: <ACCOUNT>/<PARTITION>/<QOS>
Experiment root: <TRTLLM_REPO>/experiments/pr17537-oakhaven-fp8-ht4k-<TIMESTAMP>
```

任务目标：checkout 指定交付分支，构建独立、不可变、非 editable 的 TensorRT-LLM/FlashInfer
wheel + Docker image + Pyxis/Enroot sqsh，然后验证 artifact identity、serving correctness、
MTP3 C1536 和 no-MTP C3264。最终判断该集群是否能够稳定复现 >4000 total input+output
tok/s/GPU。

必须遵守：

1. 完整阅读 `<TRTLLM_REPO>/AGENTS.md`、`CODING_GUIDELINES.md`、仓库根目录
   `OAKHAVEN_FP8_HT4K_REPRODUCTION.md`、交付包 `README.md`、
   `BUILD_AND_PORTING.md`、`ACCEPTANCE_CHECKLIST.md`、`TECHNICAL_CLARIFICATION.md`，以及目标
   checkout 中相关组件指南。
2. 先运行交付包 `scripts/verify_bundle.sh`；保留输出。
3. 记录所有初始身份和 dirty state，不得清理、reset、stash 或覆盖用户已有修改。
4. 确认 TensorRT-LLM HEAD 位于交付分支，并包含 PR-stack head
   `9a6889b2a2aba6f6e44483999dd972bc157c297b` 和 propagation commit `f572594361`。该 branch
   已经包含/继承 replay functional/V2
   commits `ee241d25f4`、`57f2781e4e`，以及 wide-head follow-up `d8d10ab354`，不要重复
   cherry-pick。`d8d10ab354` 的 ancestor check 只证明精确 PR provenance，不表示其峰值
   high-throughput 收益已被 isolated A/B 证明。
5. 不要在交付 branch 上再次应用 TensorRT-LLM propagation patch。FlashInfer static T=1 patch
   是可选 correctness/specialization hardening，不是跨 4k 前提；只有目标版本需要且 source hash
   精确匹配时才应用。未知 FlashInfer hash 必须停下并显式 rebase，禁止 `--reject` 或强行覆盖。
6. 审查 source identity，运行 focused hybrid cache-manager tests。若修改或提交代码，必须
   使用 `git commit -s`，不要加入 AI attribution/co-author。
7. 构建普通 wheel；禁止 `pip install -e`、`.pth`、`PYTHONPATH` source overlay、
   `--skip_building_wheel`、`--linking_install_binary` 或 `--fast_build` 作为最终 QA artifact。
8. Docker 和 sqsh 必须来自同一组 wheel。嵌入 source/base/runtime/native/cache hash manifest。
   从 `/tmp`、无 host source mount、unset `PYTHONPATH` 的环境验证 import 与 native library identity。
9. 使用交付包 portable recipes；只把渲染后的 recipe、dry-run、logs、results 写入
   `<TRTLLM_REPO>/experiments/...`。srt-slurm 必须保持只读，benchmark 前后保存完整 git status。
10. GPU loading heartbeat 只能用于合法 allocation 的加载阶段，并必须在 KV init、JIT/tuning、
    CUDA graph capture、correctness、warmup 和 formal measurement 前停止。
11. 正式窗口禁止请求 `/metrics`，response-level performance metrics 保持 disabled；不要让 health
    probe、heartbeat、JIT、tuning 或 graph capture 混入 formal interval。
12. FlashInfer low-M BF16 direct/Split-K heuristic、GDN specialization 和 CUDA graph 都可能有
   冷启动：为每个 job/node/variant 使用独立 cache，先完成所有 tuning/capture，再用
    `1 × concurrency` warmup 和 `3 × concurrency` formal population。

按以下顺序执行并持续推进，除非出现确实需要用户授权的外部阻塞：

A. 身份/补丁

- 校验压缩包和内部 SHA256SUMS。
- 记录 TensorRT-LLM、FlashInfer、base image、CUDA、driver、PyTorch、TensorRT、Python ABI、GPU/SM。
- 证明 `9a6889b2a2` 和 `f572594361` 都是 HEAD ancestor。记录安装后的 GDN 文件 SHA-256；
  dynamic 或 static 都可用于 >4k 复现。若选择 optional static patch，则 hash 必须是
  `4982b5a9d20d9b18588020ab3e938238c9692ffab6265c3533f4b7cf8309a8fe`。

B. 构建/独立性

- 构建普通 wheels、Docker、Docker-save 和 sqsh，记录完整命令、大小、SHA-256/digest。
- 审计没有 editable marker、`.pth`、`.egg-link`、source symlink、missing `ldd` dependency 或混合库。
- 禁网/全新 writable runtime cache 下至少验证启动路径；若仍需 JIT，记录并在正式测量前预热。

C. optional kernel characterization

- 使用交付包 `scripts/bench_flashinfer_gdn_t1_exact_shape.py` 或等价 exact-shape test，覆盖
  B=36,T=1,H=16,HV=128,K=V=128，50 warmup + 500 CUDA-graph replays。
- 运行 B=24/16/4/2/1 descending graph ladder，packed QKV 和 padded pool guard。
- 如果使用 optional static patch，patched 与 trusted output/state 必须匹配并记录同 GPU matched
  A/B latency。参考值为 0.0553032 -> 0.0512040 ms，但不设置跨集群最低 speedup；最新 production
  A/B/A 已证明 static T=1 不是 >4k 必要条件。

D. serving integrity

- 在性能测试前运行确定性小样本 correctness smoke。
- 每个 formal JSON 必须满足 completed==planned、errors=0、generated text nonempty、实际
  ISL=8192、OSL=1024。

E. MTP3 selected point

- 32×GB300，8 nodes×4 GPUs，TP32/EP32/ADP32/PP1，DeepGEMM Static544。
- RR=1.0、ISL=8192、OSL=1024、request rate=inf、C1536。
- draft=3、forced accepted drafts=2.3、总 accept length=3.3。
- M48、T8512、no chunked prefill、GDN replay on、FlashInfer low-M BF16 auto；direct/Split-K
  由 FlashInfer 内部 heuristic 选择，不要把它称为 DeepGEMM Split-K。
- KV manager V2：fraction=0.88，absolute max_tokens=479232，avg_seq_len=9216。
- 同一 selected point 做 3 个独立 allocation repeat。目标：每次 >=4000 total tok/s/GPU，
  population CV <=1.5%。保留全部结果，不选择性丢弃慢点。

F. no-MTP selected point

- 同硬件/并行/MoE/workload，C3264。
- M104、T8448、no chunked prefill、GDN replay off、FlashInfer low-M BF16 auto、KV
  fraction=0.92。
- 目标 >=4000 total tok/s/GPU。C3328 是已知 OOM capacity boundary，不作为有效点发布。

G. failure analysis

- 如果 kernel 和 TPOT 同时退化，查 GDN source/hash、cache/cubin identity、shape/layout/tactic。
- 如果 TTFT/makespan 退化但 TPOT 不退化，查 resident slots、request waves 和 chunking。
- 如果 ready 后 DeepGEMM OOM，先从 traceback 确认实际 allocation owner，再联合审计 KV、
  Mamba/GDN state、PyTorch allocated/reserved、graph private pools、communication workspace、
  DeepGEMM MoE output/workspace、可能的 FlashInfer low-M temporary 和 fragmentation。保留的
  830–838 MiB failure 是 DeepGEMM `triton_fused_gather_finalize` output allocation，不是已证明
  的 Split-K workspace。
- 第一次慢优先判为 JIT/tuning contamination；约 1% 跨 allocation 差异先视为 routing/noise。
- 不允许仅凭节点名或重启历史归因集群问题。优先做 same-allocation A/B 和 exact-shape microbench。

最终交付一个自包含报告和机器可读 manifest，至少包含：

- 所有 source/build/image/cache 身份与 hash；
- branch/commit identity、任何实际应用的 optional patch 与 `git diff --check`/unit/GPU test 结果；
- exact-shape before/after 数据与 correctness；
- MTP3/no-MTP 每个 raw result、recomputed tok/s/GPU、mean/median/range/CV；
- exact formal-window log audit；
- 失败点和 rejected boundary；
- 与保留参考 `MTP3 mean=4100.807`、`no-MTP peak=4055.616` 的比较；
- 明确的 PASS/FAIL/INCONCLUSIVE 和下一步。

不要在目标集群探索 NVFP4；本任务仅限 FP8 checkpoint。

---
