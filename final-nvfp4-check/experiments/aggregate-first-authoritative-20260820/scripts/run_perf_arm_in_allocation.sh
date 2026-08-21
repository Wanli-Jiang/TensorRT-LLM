#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM
readonly SRT=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/srt-slurm
readonly HEARTBEAT_SKILL=/home/williamj/.codex/skills/run-slurm-loading-heartbeat
readonly CONFIG="${1:?config path is required}"
readonly OUTPUT_BASE="${2:?output base is required}"
readonly IMAGE="${3:?container image is required}"
readonly ARM_NAME="${4:?arm name is required}"
readonly OUTPUT_DIR="${OUTPUT_BASE}/${SLURM_JOB_ID}"
readonly LOG_DIR="${OUTPUT_DIR}/logs"
readonly HEARTBEAT_STATE="${OUTPUT_DIR}/loading-heartbeat"
readonly STOP_REGEX='Loading (safetensors )?weights( concurrently| in parallel)?:[[:space:]]+7[0-9]%'

mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/sweep_${SLURM_JOB_ID}.log") 2>&1

test -s "${CONFIG}"
test -s "${IMAGE}"
test -x "${SRT}/.venv-compute/bin/python"
cp "${CONFIG}" "${OUTPUT_DIR}/config.yaml"
git -C "${REPO}" status --short > "${OUTPUT_DIR}/git_state.txt"

export SRTCTL_OUTPUT_DIR="${OUTPUT_DIR}"
export SRTCTL_SOURCE_DIR="${SRT}"
export PYTHONPATH="${SRT}/src"

readonly HEAD_NODE="$(scontrol show hostnames "${SLURM_NODELIST}" | head -n1)"
readonly SERVER_LOG="${LOG_DIR}/${HEAD_NODE}_agg_w0.out"
readonly EXPECTED_TASKS="$((SLURM_NNODES * 4))"

echo "Arm: ${ARM_NAME}"
echo "Job: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES} (${SLURM_NODELIST})"
echo "Config: ${CONFIG}"
echo "Image: ${IMAGE}"
echo "Output: ${OUTPUT_DIR}"
date --iso-8601=seconds > "${OUTPUT_DIR}/arm-start.txt"

heartbeat_stop() {
    "${HEARTBEAT_SKILL}/scripts/slurm_loading_heartbeat.sh" stop \
        --state-dir "${HEARTBEAT_STATE}" || true
}

benchmark_complete() {
    grep -Fq 'Benchmark completed successfully' \
        "${LOG_DIR}/sweep_${SLURM_JOB_ID}.log" \
        && test -s "${LOG_DIR}/benchmark-rollup.json"
}

trap heartbeat_stop EXIT INT TERM

"${HEARTBEAT_SKILL}/scripts/slurm_loading_heartbeat.sh" dry-run \
    --phase model-loading \
    --state-dir "${HEARTBEAT_STATE}" \
    --server-log "${SERVER_LOG}" \
    --container-image "${IMAGE}" \
    --nodes "${SLURM_NNODES}" \
    --gpus-per-node 4 \
    --stop-regex "${STOP_REGEX}" \
    --job-id "${SLURM_JOB_ID}" \
    > "${OUTPUT_DIR}/heartbeat.dry-run.txt"

grep -q "^total_tasks=${EXPECTED_TASKS}$" \
    "${OUTPUT_DIR}/heartbeat.dry-run.txt"

"${HEARTBEAT_SKILL}/scripts/slurm_loading_heartbeat.sh" start \
    --phase model-loading \
    --state-dir "${HEARTBEAT_STATE}" \
    --server-log "${SERVER_LOG}" \
    --container-image "${IMAGE}" \
    --nodes "${SLURM_NNODES}" \
    --gpus-per-node 4 \
    --stop-regex "${STOP_REGEX}" \
    --job-id "${SLURM_JOB_ID}"

set +e
cd "${EXP}"
"${SRT}/.venv-compute/bin/python" \
    -m srtctl.cli.do_sweep "${OUTPUT_DIR}/config.yaml"
readonly SWEEP_EXIT_CODE=$?
set -e

FINAL_EXIT_CODE="${SWEEP_EXIT_CODE}"
if [[ "${SWEEP_EXIT_CODE}" -ne 0 ]] && benchmark_complete; then
    echo "Normalizing cleanup-only sweep exit ${SWEEP_EXIT_CODE}."
    FINAL_EXIT_CODE=0
fi

heartbeat_stop
"${HEARTBEAT_SKILL}/scripts/slurm_loading_heartbeat.sh" status \
    --state-dir "${HEARTBEAT_STATE}" \
    | tee "${OUTPUT_DIR}/heartbeat.final-status.txt"

HEARTBEAT_AUDIT_OK=1
grep -qx 'state=stopped' "${OUTPUT_DIR}/heartbeat.final-status.txt" \
    || HEARTBEAT_AUDIT_OK=0
grep -qx 'supervisor_alive=no' "${OUTPUT_DIR}/heartbeat.final-status.txt" \
    || HEARTBEAT_AUDIT_OK=0
grep -qx "total_tasks=${EXPECTED_TASKS}" \
    "${OUTPUT_DIR}/heartbeat.final-status.txt" \
    || HEARTBEAT_AUDIT_OK=0
grep -qx 'stop_reason=automatic model-loading boundary matched' \
    "${OUTPUT_DIR}/heartbeat.final-status.txt" \
    || HEARTBEAT_AUDIT_OK=0
test -s "${HEARTBEAT_STATE}/boundary.match" || HEARTBEAT_AUDIT_OK=0

if [[ "${HEARTBEAT_AUDIT_OK}" -ne 1 ]]; then
    echo 'ERROR: loading heartbeat audit failed; invalidating this arm.'
    FINAL_EXIT_CODE=86
fi

if ! benchmark_complete; then
    echo 'ERROR: benchmark completion marker or rollup is missing.'
    FINAL_EXIT_CODE=87
fi

date --iso-8601=seconds > "${OUTPUT_DIR}/arm-end.txt"
printf '%s\n' "${FINAL_EXIT_CODE}" > "${OUTPUT_DIR}/arm-exit-code.txt"
trap - EXIT INT TERM
exit "${FINAL_EXIT_CODE}"
