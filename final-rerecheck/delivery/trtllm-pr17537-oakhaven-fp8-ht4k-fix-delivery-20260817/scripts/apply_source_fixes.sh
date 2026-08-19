#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly REQUIRED_TRTLLM_COMMIT="9a6889b2a2aba6f6e44483999dd972bc157c297b"
readonly BAD_GDN_SHA256="61de9ffa703962cb1ddb73823100550138708bbcbb535a3efcac608940e67e61"
readonly FIXED_GDN_SHA256="4982b5a9d20d9b18588020ab3e938238c9692ffab6265c3533f4b7cf8309a8fe"

usage() {
    echo "Usage: $0 --trtllm-repo PATH [--flashinfer-repo PATH --apply-optional-flashinfer-gdn-fix]" >&2
}

trtllm_repo=""
flashinfer_repo=""
apply_optional_flashinfer_gdn_fix="0"
while [[ $# -gt 0 ]]; do
    case "$1" in
    --trtllm-repo)
        trtllm_repo="$2"
        shift 2
        ;;
    --flashinfer-repo)
        flashinfer_repo="$2"
        shift 2
        ;;
    --apply-optional-flashinfer-gdn-fix)
        apply_optional_flashinfer_gdn_fix="1"
        shift
        ;;
    *)
        usage
        exit 2
        ;;
    esac
done

if [[ -z "${trtllm_repo}" ]]; then
    usage
    exit 2
fi
if [[ "${apply_optional_flashinfer_gdn_fix}" == "1" && -z "${flashinfer_repo}" ]]; then
    echo "ERROR: --flashinfer-repo is required with --apply-optional-flashinfer-gdn-fix." >&2
    exit 2
fi

readonly script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly delivery_root="$(cd "${script_dir}/.." && pwd)"
readonly trtllm_patch="${delivery_root}/patches/trtllm-pr17537-max-num-tokens-propagation.patch"
readonly flashinfer_patch="${delivery_root}/patches/flashinfer-gdn-t1-static-cache-vs-0.6.16-0.6.17.patch"

trtllm_repo="$(git -C "${trtllm_repo}" rev-parse --show-toplevel)"

if ! git -C "${trtllm_repo}" merge-base --is-ancestor \
    "${REQUIRED_TRTLLM_COMMIT}" HEAD; then
    echo "ERROR: TensorRT-LLM HEAD does not contain PR #17537 head ${REQUIRED_TRTLLM_COMMIT}." >&2
    exit 1
fi

if git -C "${trtllm_repo}" apply --reverse --check "${trtllm_patch}" 2>/dev/null; then
    echo "TensorRT-LLM propagation patch is already applied."
else
    git -C "${trtllm_repo}" apply --check "${trtllm_patch}"
    git -C "${trtllm_repo}" apply "${trtllm_patch}"
    echo "Applied TensorRT-LLM max_num_tokens propagation patch."
fi

gdn_path=""
gdn_sha256="not-checked"
if [[ "${apply_optional_flashinfer_gdn_fix}" == "1" ]]; then
    flashinfer_repo="$(git -C "${flashinfer_repo}" rev-parse --show-toplevel)"
    readonly gdn_relative_path="flashinfer/gdn_kernels/gdn_decode_bf16_state.py"
    gdn_path="${flashinfer_repo}/${gdn_relative_path}"
    if [[ ! -f "${gdn_path}" ]]; then
        echo "ERROR: expected FlashInfer source file is missing: ${gdn_path}" >&2
        exit 1
    fi

    gdn_sha256="$(sha256sum "${gdn_path}" | awk '{print $1}')"
    case "${gdn_sha256}" in
    "${FIXED_GDN_SHA256}")
        echo "Optional FlashInfer GDN T=1 patch is already applied."
        ;;
    "${BAD_GDN_SHA256}")
        git -C "${flashinfer_repo}" apply --check "${flashinfer_patch}"
        git -C "${flashinfer_repo}" apply "${flashinfer_patch}"
        echo "Applied optional FlashInfer GDN T=1 batch/layout cache patch."
        ;;
    *)
        echo "ERROR: unsupported GDN source identity ${gdn_sha256}." >&2
        echo "Expected dynamic ${BAD_GDN_SHA256} or static ${FIXED_GDN_SHA256}." >&2
        echo "Rebase explicitly; do not force-apply to an unknown FlashInfer revision." >&2
        exit 1
        ;;
    esac

    gdn_sha256="$(sha256sum "${gdn_path}" | awk '{print $1}')"
    if [[ "${gdn_sha256}" != "${FIXED_GDN_SHA256}" ]]; then
        echo "ERROR: patched GDN hash mismatch: ${gdn_sha256}" >&2
        exit 1
    fi
fi

compile_paths=(
    "${trtllm_repo}/tensorrt_llm/_torch/pyexecutor/_util.py"
    "${trtllm_repo}/tensorrt_llm/_torch/pyexecutor/mamba_cache_manager.py"
)
if [[ -n "${gdn_path}" ]]; then
    compile_paths+=("${gdn_path}")
fi
python3 -m py_compile "${compile_paths[@]}"

echo "Source patches applied and syntax-checked."
echo "TensorRT-LLM repo: ${trtllm_repo}"
echo "Optional FlashInfer patch requested: ${apply_optional_flashinfer_gdn_fix}"
echo "GDN SHA256: ${gdn_sha256}"
echo "No commit was created. Review, test, and commit with git commit -s if desired."
