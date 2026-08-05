#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Run focused lint and unit coverage for the completed optimization port.

#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --segment=1
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/williamj/TRTLLM5/oak-nvfp4-sweep/logs/%x-%j.slurm.log

set -euo pipefail

BASE_IMAGE="${1:?usage: $0 BASE_IMAGE TRTLLM5_REPO}"
TRTLLM5_REPO="${2:?usage: $0 BASE_IMAGE TRTLLM5_REPO}"

EXPECTED_REPO=/lustre/fsw/portfolios/coreai/users/williamj/TRTLLM5
HANDOFF_ROOT="${TRTLLM5_REPO}/oak-nvfp4-sweep"
CONTAINER_REPO=/opt/trtllm5-dev
CONTAINER_HANDOFF=/opt/trtllm5-handoff

if [[ "$(realpath "${TRTLLM5_REPO}")" != "$(realpath "${EXPECTED_REPO}")" ]]; then
    echo "ERROR: refusing to test a non-TRTLLM5 checkout" >&2
    exit 2
fi
test -f "${BASE_IMAGE}"
test -x "${HANDOFF_ROOT}/venv/bin/python3"
mkdir -p "${HANDOFF_ROOT}/cache" "${HANDOFF_ROOT}/logs"

export CONTAINER_HANDOFF CONTAINER_REPO
export ENROOT_ALLOW_DEV=yes
srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${TRTLLM5_REPO}:${CONTAINER_REPO},${HANDOFF_ROOT}:${CONTAINER_HANDOFF}" \
    --container-workdir="${CONTAINER_REPO}" \
    --export=ALL \
    bash -c '
        set -euo pipefail
        unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
        source "${CONTAINER_HANDOFF}/venv/bin/activate"

        cache_root="${CONTAINER_HANDOFF}/cache/test-${SLURM_JOB_ID}"
        export PRE_COMMIT_HOME="${CONTAINER_HANDOFF}/cache/pre-commit"
        export RUFF_CACHE_DIR="${CONTAINER_HANDOFF}/cache/ruff"
        export XDG_CACHE_HOME="${cache_root}/xdg"
        export CUDA_CACHE_PATH="${cache_root}/cuda"
        export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
        export TRITON_CACHE_DIR="${cache_root}/triton"
        export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
        export TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions"
        export NUMBA_CACHE_DIR="${cache_root}/numba"
        mkdir -p "${PRE_COMMIT_HOME}" "${RUFF_CACHE_DIR}" "${XDG_CACHE_HOME}" \
            "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" \
            "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
            "${TORCH_EXTENSIONS_DIR}" "${NUMBA_CACHE_DIR}"

        python3 -m pip install --upgrade \
            mako parameterized pre-commit pytest==9.0.3 pytest-mock

        pre-commit run --files \
            cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu \
            cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.h \
            cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.cu \
            cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/runner.h \
            cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp \
            tensorrt_llm/_torch/custom_ops/trtllm_gen_custom_ops.py \
            tensorrt_llm/_torch/models/modeling_qwen3_next.py \
            tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py \
            tensorrt_llm/_torch/modules/fused_moe/moe_scheduler.py \
            tensorrt_llm/_torch/pyexecutor/_util.py \
            tests/unittest/_torch/executor/test_kv_cache_estimation.py \
            tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py \
            tests/unittest/_torch/modules/moe/test_moe_backend.py \
            tests/unittest/_torch/modules/moe/test_moe_scheduler.py \
            oak-nvfp4-sweep/build_trtllm5_dev.sbatch \
            oak-nvfp4-sweep/test_trtllm5_changes.sbatch \
            oak-nvfp4-sweep/validate_trtllm5_dev.sbatch

        python3 -m pytest -q \
            tests/unittest/_torch/executor/test_kv_cache_estimation.py::test_separate_one_model_draft_normalizes_target_pool_ratio \
            tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py::test_mtp_forward_uses_and_restores_draft_rank_token_counts \
            tests/unittest/_torch/models/test_qwen3_next_eager_fusion.py::test_mtp_forward_restores_rank_token_counts_after_failure \
            tests/unittest/_torch/modules/moe/test_moe_backend.py::test_fp8_block_scale_moe_fallback_tactic_is_explicit_and_deterministic \
            tests/unittest/_torch/modules/moe/test_moe_scheduler.py
    ' | tee "${HANDOFF_ROOT}/logs/test-${SLURM_JOB_ID}.log"

git -C "${TRTLLM5_REPO}" diff --check
