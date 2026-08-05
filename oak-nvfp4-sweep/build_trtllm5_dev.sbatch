#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Build TRTLLM5 with repository-local native artifacts, caches, and a
# persistent editable Python environment.

#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --segment=1

set -euo pipefail

BASE_IMAGE="${1:?usage: $0 BASE_IMAGE TRTLLM5_REPO [clean|incremental]}"
TRTLLM5_REPO="${2:?usage: $0 BASE_IMAGE TRTLLM5_REPO [clean|incremental]}"
BUILD_MODE="${3:-incremental}"

EXPECTED_REPO=/lustre/fsw/portfolios/coreai/users/williamj/TRTLLM5
HANDOFF_ROOT="${TRTLLM5_REPO}/oak-nvfp4-sweep"
CONTAINER_REPO=/opt/trtllm5-dev
CONTAINER_HANDOFF=/opt/trtllm5-handoff

if [[ "$(realpath "${TRTLLM5_REPO}")" != "$(realpath "${EXPECTED_REPO}")" ]]; then
    echo "ERROR: refusing to build a non-TRTLLM5 checkout: ${TRTLLM5_REPO}" >&2
    exit 2
fi
if [[ ! -f "${BASE_IMAGE}" || ! -d "${TRTLLM5_REPO}/.git" ]]; then
    echo "ERROR: base image or TRTLLM5 checkout is missing" >&2
    exit 2
fi
if [[ "${BUILD_MODE}" != "clean" && "${BUILD_MODE}" != "incremental" ]]; then
    echo "ERROR: build mode must be clean or incremental" >&2
    exit 2
fi

mkdir -p "${HANDOFF_ROOT}/cache/ccache" "${HANDOFF_ROOT}/logs"
SOURCE_HEAD="$(git -C "${TRTLLM5_REPO}" rev-parse HEAD)"
SOURCE_DIFF_SHA256="$(git -C "${TRTLLM5_REPO}" diff --binary HEAD | sha256sum | cut -d ' ' -f 1)"
IMAGE_SHA256="$(sha256sum "${BASE_IMAGE}" | cut -d ' ' -f 1)"

export BUILD_MODE CONTAINER_HANDOFF CONTAINER_REPO IMAGE_SHA256
export SOURCE_DIFF_SHA256 SOURCE_HEAD
export ENROOT_ALLOW_DEV=yes
srun --mpi=pmix --kill-on-bad-exit=1 \
    --container-image="${BASE_IMAGE}" \
    --container-mounts="${TRTLLM5_REPO}:${CONTAINER_REPO},${HANDOFF_ROOT}:${CONTAINER_HANDOFF}" \
    --container-workdir="${CONTAINER_REPO}" \
    --export=ALL \
    bash -c '
        set -euo pipefail
        unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

        cache_root="${CONTAINER_HANDOFF}/cache/job-${SLURM_JOB_ID}"
        export CCACHE_DIR="${CONTAINER_HANDOFF}/cache/ccache"
        export XDG_CACHE_HOME="${cache_root}/xdg"
        export CUDA_CACHE_PATH="${cache_root}/cuda"
        export FLASHINFER_WORKSPACE_BASE="${cache_root}/flashinfer"
        export TRITON_CACHE_DIR="${cache_root}/triton"
        export TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor"
        export TORCH_EXTENSIONS_DIR="${cache_root}/torch-extensions"
        export NUMBA_CACHE_DIR="${cache_root}/numba"
        export HF_HOME="${cache_root}/hf"
        mkdir -p "${CCACHE_DIR}" "${XDG_CACHE_HOME}" "${CUDA_CACHE_PATH}" \
            "${FLASHINFER_WORKSPACE_BASE}" "${TRITON_CACHE_DIR}" \
            "${TORCHINDUCTOR_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" \
            "${NUMBA_CACHE_DIR}" "${HF_HOME}"

        # --linking_install_binary replaces these generated Python trees with
        # symlinks.  A previous copy-based build can leave an ignored real
        # directory behind, which build_wheel.py cannot remove with unlink().
        # Quarantine only untracked generated trees; never move source files.
        stale_root="${CONTAINER_HANDOFF}/cache/stale-install/job-${SLURM_JOB_ID}"
        for relative_path in \
            tensorrt_llm/deep_ep \
            tensorrt_llm/deep_gemm \
            tensorrt_llm/flash_mla; do
            install_path="${CONTAINER_REPO}/${relative_path}"
            if [[ -e "${install_path}" && ! -L "${install_path}" ]]; then
                tracked_files="$(git ls-files -- "${relative_path}")"
                if [[ -n "${tracked_files}" ]]; then
                    echo "ERROR: refusing to replace tracked tree ${relative_path}" >&2
                    exit 2
                fi
                mkdir -p "${stale_root}"
                quarantine_name="${relative_path//\//__}"
                mv -- "${install_path}" "${stale_root}/${quarantine_name}"
                echo "Quarantined stale generated tree: ${relative_path}"
            fi
        done

        build_args=(
            --use_ccache
            --cuda_architectures 100-real
            --skip_building_wheel
            --linking_install_binary
            --job_count 64
            --no-venv
            --yes
        )
        if [[ "${BUILD_MODE}" == "clean" ]]; then
            build_args+=(--clean)
        fi

        python3 scripts/build_wheel.py "${build_args[@]}"

        test -f cpp/build/tensorrt_llm/libtensorrt_llm.so
        test -f cpp/build/tensorrt_llm/thop/libth_common.so

        if [[ ! -x "${CONTAINER_HANDOFF}/venv/bin/python3" ]]; then
            python3 -m venv --system-site-packages "${CONTAINER_HANDOFF}/venv"
        fi
        source "${CONTAINER_HANDOFF}/venv/bin/activate"
        python3 -m pip install --no-build-isolation --no-deps --force-reinstall \
            --editable "${CONTAINER_REPO}"

        python3 - <<PY
import pathlib

import tensorrt_llm
from tensorrt_llm import _common

expected_root = pathlib.Path("${CONTAINER_REPO}")
package_path = pathlib.Path(tensorrt_llm.__file__).resolve()
assert package_path.is_relative_to(expected_root), package_path
assert _common is not None

mapped = {
    pathlib.Path(line.split()[-1]).resolve()
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
    if "libth_common.so" in line
}
assert len(mapped) == 1, mapped
assert next(iter(mapped)).is_relative_to(expected_root), mapped
print(f"tensorrt_llm={package_path}")
print(f"libth_common={next(iter(mapped))}")
PY

        sha256sum cpp/build/tensorrt_llm/libtensorrt_llm.so \
            cpp/build/tensorrt_llm/thop/libth_common.so
        trtllm-serve --help >/dev/null

        cat > "${CONTAINER_HANDOFF}/logs/build-${SLURM_JOB_ID}.identity" <<EOF
slurm_job_id=${SLURM_JOB_ID}
source_head=${SOURCE_HEAD}
source_diff_sha256=${SOURCE_DIFF_SHA256}
image_sha256=${IMAGE_SHA256}
build_mode=${BUILD_MODE}
container_repo=${CONTAINER_REPO}
container_venv=${CONTAINER_HANDOFF}/venv
EOF
    '

git -C "${TRTLLM5_REPO}" diff --check
