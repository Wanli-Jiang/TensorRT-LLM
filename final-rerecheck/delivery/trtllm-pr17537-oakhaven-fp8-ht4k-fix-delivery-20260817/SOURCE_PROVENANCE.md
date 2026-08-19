<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Source and evidence provenance

## TensorRT-LLM

- Repository: `https://github.com/NVIDIA/TensorRT-LLM`
- Delivery fork: `https://github.com/Wanli-Jiang/TensorRT-LLM`
- Delivery branch: `user/williamj/oakhaven-fp8-ht4k-repro-20260819`
- PR: `https://github.com/NVIDIA/TensorRT-LLM/pull/17537`
- PR head at packaging time: `9a6889b2a2aba6f6e44483999dd972bc157c297b`
- Branch observed locally: `user/williamj/support-new-model-pr-stacks`
- PR status observed on 2026-08-17: draft, 18 commits
- The PR page says the stack is intended to support optimized Qwen3.8-2.4T-A95B-FP8 performance
  and will be split into smaller upstream PRs.

The original three-file TensorRT-LLM patch is the exact diff above that head at package creation.
It is now committed on the delivery branch as `f572594361`. Its files are `_util.py`,
`mamba_cache_manager.py`, and the focused unit test. The compatibility patch remains in the bundle
for older checkouts; it must not be reapplied on the delivery branch.

### Replay lineage and evidence scope

- PR #16464 / `ee241d25f4`: functional GDN MTP cached replay mechanism.
- PR #16768 / `57f2781e4e`: low-batch tuning, eligible default enablement, runtime
  bookkeeping, and V2 manager/all-layer commit support used by the tested stack.
- PR #17537 / `d8d10ab354`: wide-value-head follow-up and metadata preparation tuning.

The verification script requires all three because the delivery pins exact PR #17537 source.
That ancestry check is source provenance, not a causal performance classification. The retained
evidence proves the replay mechanism's capacity role but has no isolated revert A/B for
`d8d10ab354`; its peak-high-throughput benefit remains unproven.

## FlashInfer

- Repository: `https://github.com/flashinfer-ai/flashinfer`
- Fixed local branch: `oak-gdn-t1-cache-fix`
- Original fix commit: `baad0dca27d165341d188b895f3ab161e8098344`
- Subject: `fix: key GDN T=1 compile cache by batch and layout`
- Audited bad package sources:
  - 0.6.16 source `8da13a29c85f7e5b1c81878d933f84ae9fc4afa9`
  - 0.6.17.dev20260806 source `e493ed8c496432d668b4bfad703427abfc832c1f`
  - both have the same bad module SHA-256 `61de9ffa...e61`
- Fixed module SHA-256: `4982b5a9...a8fe`

The version-targeted code patch was mechanically validated against the 0.6.16 source tree and
produced a file byte-identical to the retained fixed module. Because the 0.6.17 nightly module is
byte-identical to 0.6.16, the guarded source hash is the authoritative applicability check.

The patch is optional for reproducing >4k. Job `505210` measured dynamic/static/dynamic at
4049.545/4064.498/4096.202 tok/s/GPU on the same allocation; static was inside the dynamic bracket.
The retained fixed module and patch remain for provenance and optional correctness/specialization
hardening, not as a mandatory serving-performance dependency.

## Runtime and retained result

- Retained QA image:
  `/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh`
- Image SHA-256:
  `08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59`
- Retained model:
  `/lustre/fsw/portfolios/coreai/users/williamj/models/oakhaven-max-final-fp8_vv3`
- Model identity hashes are preserved in the causal evidence report.

The target cluster should not depend on these absolute paths. They are provenance only; render the
portable recipes with paths valid on that cluster.

## Patch validation performed while packaging

- TensorRT-LLM patch dry-run/applied successfully to a clean `git archive` of PR head `9a6889b2a2`.
- Patched archive files were byte-compared with the measured worktree files.
- FlashInfer code patch dry-run/applied successfully to source commit `8da13a29...`.
- Patched GDN source became SHA-256 `4982b5a9...a8fe` and was byte-compared with the retained fixed
  module.
- All changed Python files passed `python3 -m py_compile` on the packaging host.
- Host `pytest` was unavailable, so the focused TensorRT-LLM tests must be run in the target build
  environment; this is an explicit acceptance gate, not silently marked passed.
