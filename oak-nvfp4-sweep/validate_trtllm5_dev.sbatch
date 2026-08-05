#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Validate that the persistent editable environment loads only TRTLLM5 Python
# and native artifacts.

#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --segment=1

set -euo pipefail

BASE_IMAGE="${1:?usage: $0 BASE_IMAGE TRTLLM5_REPO}"
TRTLLM5_REPO="${2:?usage: $0 BASE_IMAGE TRTLLM5_REPO}"

EXPECTED_REPO=/lustre/fsw/portfolios/coreai/users/williamj/TRTLLM5
HANDOFF_ROOT="${TRTLLM5_REPO}/oak-nvfp4-sweep"
CONTAINER_REPO=/opt/trtllm5-dev
CONTAINER_HANDOFF=/opt/trtllm5-handoff

if [[ "$(realpath "${TRTLLM5_REPO}")" != "$(realpath "${EXPECTED_REPO}")" ]]; then
    echo "ERROR: refusing to validate a non-TRTLLM5 checkout" >&2
    exit 2
fi
test -f "${BASE_IMAGE}"
test -x "${HANDOFF_ROOT}/venv/bin/python3"
mkdir -p "${HANDOFF_ROOT}/logs"

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

        python3 - <<PY
import pathlib
import sys

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm import _common

expected_root = pathlib.Path("${CONTAINER_REPO}")
package_path = pathlib.Path(tensorrt_llm.__file__).resolve()
assert package_path.is_relative_to(expected_root), package_path
assert _common is not None

loaded_modules = {
    name: pathlib.Path(module.__file__).resolve()
    for name, module in sys.modules.items()
    if name.startswith("tensorrt_llm") and getattr(module, "__file__", None)
}
outside = {
    name: path
    for name, path in loaded_modules.items()
    if not path.is_relative_to(expected_root)
}
assert not outside, outside

mapped = {
    pathlib.Path(line.split()[-1]).resolve()
    for line in pathlib.Path("/proc/self/maps").read_text().splitlines()
    if "libth_common.so" in line or "libtensorrt_llm.so" in line
}
assert mapped, mapped
assert all(path.is_relative_to(expected_root) for path in mapped), mapped

forbidden = ("TRTLLM3", "TRTLLM4", "/workspace/TensorRT-LLM")
assert not any(
    marker in entry
    for entry in sys.path
    for marker in forbidden
), sys.path

print(f"tensorrt_llm={package_path}")
for path in sorted(mapped):
    print(f"native_library={path}")
PY

        if ldd "${CONTAINER_REPO}/tensorrt_llm/libs/libtensorrt_llm.so" | grep -q "not found"; then
            ldd "${CONTAINER_REPO}/tensorrt_llm/libs/libtensorrt_llm.so"
            exit 1
        fi
        if ldd "${CONTAINER_REPO}/tensorrt_llm/libs/libth_common.so" | grep -q "not found"; then
            ldd "${CONTAINER_REPO}/tensorrt_llm/libs/libth_common.so"
            exit 1
        fi
        trtllm-serve --help >/dev/null
    ' | tee "${HANDOFF_ROOT}/logs/validate-${SLURM_JOB_ID}.log"

