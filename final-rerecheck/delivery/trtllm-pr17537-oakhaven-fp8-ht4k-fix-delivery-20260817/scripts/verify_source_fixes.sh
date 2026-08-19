#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly REQUIRED_TRTLLM_COMMIT="9a6889b2a2aba6f6e44483999dd972bc157c297b"
readonly REQUIRED_REPLAY_COMMITS=(
    "ee241d25f43973ad52495119d6536176b91c0aec"
    "57f2781e4e9f679cfa429400b64e447fbefa253e"
    "d8d10ab3540c4341852075a45b6b35bcfa0a23cf"
)
readonly BAD_GDN_SHA256="61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61"
readonly FIXED_GDN_SHA256="4982b5a9d20d9b18588020ab3e938238c9692ffab6265c3533f4b7cf8309a8fe"

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 TRTLLM_REPO [FLASHINFER_REPO]" >&2
    exit 2
fi

readonly trtllm_repo="$(git -C "$1" rev-parse --show-toplevel)"
readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly delivery_root="$(cd "${script_dir}/.." && pwd)"
readonly trtllm_patch="${delivery_root}/patches/trtllm-pr17537-max-num-tokens-propagation.patch"
readonly flashinfer_patch="${delivery_root}/patches/flashinfer-gdn-t1-static-cache-vs-0.6.16-0.6.17.patch"

git -C "${trtllm_repo}" merge-base --is-ancestor "${REQUIRED_TRTLLM_COMMIT}" HEAD
for commit in "${REQUIRED_REPLAY_COMMITS[@]}"; do
    git -C "${trtllm_repo}" merge-base --is-ancestor "${commit}" HEAD
done

git -C "${trtllm_repo}" apply --reverse --check "${trtllm_patch}"
gdn_path=""
actual_gdn_sha256="not-checked"
if [[ $# -eq 2 ]]; then
    readonly flashinfer_repo="$(git -C "$2" rev-parse --show-toplevel)"
    gdn_path="${flashinfer_repo}/flashinfer/gdn_kernels/gdn_decode_bf16_state.py"
    actual_gdn_sha256="$(sha256sum "${gdn_path}" | awk '{print $1}')"
    case "${actual_gdn_sha256}" in
    "${FIXED_GDN_SHA256}")
        git -C "${flashinfer_repo}" apply --reverse --check "${flashinfer_patch}"
        echo "GDN_MODE=static-optional"
        ;;
    "${BAD_GDN_SHA256}")
        echo "GDN_MODE=dynamic-accepted-for-ht4k"
        ;;
    *)
        echo "ERROR: unsupported GDN source identity ${actual_gdn_sha256}." >&2
        exit 1
        ;;
    esac
fi

compile_paths=(
    "${trtllm_repo}/tensorrt_llm/_torch/pyexecutor/_util.py"
    "${trtllm_repo}/tensorrt_llm/_torch/pyexecutor/mamba_cache_manager.py"
)
if [[ -n "${gdn_path}" ]]; then
    compile_paths+=("${gdn_path}")
fi
python3 -m py_compile "${compile_paths[@]}"

if command -v pytest >/dev/null 2>&1; then
    pytest -q "${trtllm_repo}/tests/unittest/_torch/executor/test_mamba_cache_manager.py" \
        -k "kimi_explicit_v2_manager_geometry or qwen3_gdn_replay_supports_cpp_and_v2_managers or cpp_hybrid_zero_local_mamba_layers"
else
    echo "WARNING: pytest is unavailable; TensorRT-LLM unit tests were not run." >&2
fi

echo "SOURCE_FIX_VERIFICATION=PASS"
echo "TRTLLM_HEAD=$(git -C "${trtllm_repo}" rev-parse HEAD)"
if [[ $# -eq 2 ]]; then
    echo "FLASHINFER_HEAD=$(git -C "${flashinfer_repo}" rev-parse HEAD)"
fi
echo "GDN_SHA256=${actual_gdn_sha256}"
