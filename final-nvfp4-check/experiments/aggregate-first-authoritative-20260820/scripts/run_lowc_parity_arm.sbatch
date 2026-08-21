#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --account=coreai_comparch_trtllm
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --switches=1@10:00
#SBATCH --comment={"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"model_loading","description":"Oakhaven NVFP4 MTP3 low-concurrency parity diagnostic; loading-only heartbeat; metrics endpoint forbidden."}}

set -euo pipefail

readonly REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM
readonly EXP=${REPO}/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly RUNNER=${EXP}/scripts/run_perf_arm_in_allocation.sh
readonly WINDOW_AUDITOR=${EXP}/scripts/audit_formal_measurement_windows.py
readonly GUARD=/home/williamj/.codex/skills/run-trtllm-srt-slurm-benchmarks/scripts/srt_repo_guard.py
readonly SRT=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/srt-slurm
readonly BASELINE=${REPO}/final-nvfp4-check/guards/srt-repo.before.json
readonly CONFIG="${NVFP4_CONFIG:?NVFP4_CONFIG is required}"
readonly IMAGE="${NVFP4_IMAGE:?NVFP4_IMAGE is required}"
readonly ARM="${NVFP4_ARM:?NVFP4_ARM is required}"
readonly OUTPUT_BASE=${EXP}/outputs/${ARM}
readonly OUTPUT_DIR=${OUTPUT_BASE}/${SLURM_JOB_ID}

if [[ "${SLURM_NNODES}" -ne 2 ]]; then
    echo "Expected 2 nodes, got ${SLURM_NNODES}." >&2
    exit 3
fi

mapfile -t NVL72_DOMAINS < <(
    scontrol show hostnames "${SLURM_NODELIST}" \
        | sed -E 's/^(nvl72d[0-9]+).*/\1/' \
        | sort -u
)
if [[ "${#NVL72_DOMAINS[@]}" -ne 1 ]]; then
    echo "Cross-domain allocation is invalid: ${SLURM_NODELIST}." >&2
    exit 4
fi

mkdir -p "${OUTPUT_DIR}"
exec > >(tee "${OUTPUT_DIR}/allocation.log") 2>&1

python3 "${GUARD}" check --repo "${SRT}" --baseline "${BASELINE}"
python3 "${GUARD}" assert-external --repo "${SRT}" "${EXP}"
scontrol show job "${SLURM_JOB_ID}" -o > "${OUTPUT_DIR}/slurm-job.txt"
scontrol show hostnames "${SLURM_NODELIST}" > "${OUTPUT_DIR}/nodes.txt"
stat -c '%n|%s|%y|%i' "${IMAGE}" > "${OUTPUT_DIR}/image-stat.txt"
sha256sum "${CONFIG}" "${RUNNER}" "${WINDOW_AUDITOR}" "$0" \
    > "${OUTPUT_DIR}/artifact-sha256.txt"

bash "${RUNNER}" "${CONFIG}" "${OUTPUT_BASE}" "${IMAGE}" "${ARM}"
python3 "${WINDOW_AUDITOR}" "${OUTPUT_DIR}" \
    --output "${OUTPUT_DIR}/audit-formal-window.json"

if grep -R -n -E '"(GET|POST|HEAD|PUT|OPTIONS) /metrics([? ]|$)' \
    "${OUTPUT_DIR}/logs"; then
    echo "ERROR: forbidden /metrics request found in ${ARM}."
    exit 88
fi

python3 "${GUARD}" check --repo "${SRT}" --baseline "${BASELINE}"
date --iso-8601=seconds > "${OUTPUT_DIR}/completed-at.txt"
printf '%s\n' 8 > "${OUTPUT_DIR}/serving-gpu-count.txt"
