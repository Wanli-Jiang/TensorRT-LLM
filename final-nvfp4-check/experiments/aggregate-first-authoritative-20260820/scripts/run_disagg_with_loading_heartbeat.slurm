#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM
readonly ROOT=${REPO}/final-nvfp4-check
readonly EXP=${ROOT}/experiments/aggregate-first-authoritative-20260820
readonly REFERENCE_RUNNER=/lustre/fsw/portfolios/coreai/users/williamj/nemotron-ultra-benchmarking/disagg/bin/harness/trtllm-disagg-benchmark/disaggr_torch.slurm
readonly HEARTBEAT_CONTROLLER=/home/williamj/.codex/skills/run-slurm-loading-heartbeat/scripts/slurm_loading_heartbeat.sh
readonly LOCAL_BIN=${EXP}/setup/bin
readonly SRT=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/srt-slurm
readonly SRT_GUARD=/home/williamj/.codex/skills/run-trtllm-srt-slurm-benchmarks/scripts/srt_repo_guard.py
readonly SRT_BASELINE=${ROOT}/guards/srt-repo.before.json

full_logdir=""
container_image=""
benchmark_mode=""
args=("$@")
while (($#)); do
    case "$1" in
        --full-logdir)
            full_logdir=$2
            shift 2
            ;;
        --container-image)
            container_image=$2
            shift 2
            ;;
        --benchmark-mode)
            benchmark_mode=$2
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

test -n "${full_logdir}"
test -n "${container_image}"
test -n "${benchmark_mode}"
test -s "${REFERENCE_RUNNER}"
test -x "${HEARTBEAT_CONTROLLER}"
test -x "${LOCAL_BIN}/scancel"
test -x "${LOCAL_BIN}/srun"
case "${full_logdir}" in
    "${EXP}"/outputs/disaggregate/*) ;;
    *)
        echo "Output directory is outside the owning experiment: ${full_logdir}" >&2
        exit 2
        ;;
esac

readonly HEARTBEAT_STATE=${full_logdir}/loading-heartbeat
readonly SERVER_LOG=${full_logdir}/3_output_GEN_0.log
readonly STOP_REGEX='Loading (safetensors )?weights( concurrently| in parallel)?:[[:space:]]+7[0-9]%|KVCacheManager(V2)?|warm[ -]?up|CUDA graph[^[:cntrl:]]*captur|Application startup complete'
readonly WORKER_LOG_SNAPSHOT_DIR=${full_logdir}/worker-log-snapshots
worker_log_tail_pids=()
mkdir -p "${HEARTBEAT_STATE}"

python3 "${SRT_GUARD}" check --repo "${SRT}" --baseline "${SRT_BASELINE}"
python3 "${SRT_GUARD}" assert-external --repo "${SRT}" "${ROOT}"
scontrol show job "${SLURM_JOB_ID}" -o > "${full_logdir}/slurm-job.txt"
scontrol show hostnames "${SLURM_NODELIST}" > "${full_logdir}/nodes.txt"
sha256sum "${REFERENCE_RUNNER}" "${HEARTBEAT_CONTROLLER}" "${LOCAL_BIN}/srun" \
    "${LOCAL_BIN}/scancel" "${EXP}/setup/trtllm-llmapi-launch" \
    "${EXP}/setup/current-image-trtllm-llmapi-launch.original" "$0" \
    > "${full_logdir}/runner-sha256.txt"
stat -c '%n|%s|%y|%i' "${container_image}" > "${full_logdir}/image-stat.txt"

start_worker_log_snapshots()
{
    local role
    local source_log
    mkdir -p "${WORKER_LOG_SNAPSHOT_DIR}"
    for role in GEN_0 CTX_0 CTX_1 CTX_2 CTX_3 CTX_4 CTX_5 CTX_6; do
        source_log=${full_logdir}/3_output_${role}.log
        tail -n +1 --retry -F "${source_log}" \
            > "${WORKER_LOG_SNAPSHOT_DIR}/3_output_${role}.log" \
            2> "${WORKER_LOG_SNAPSHOT_DIR}/3_output_${role}.tail.err" &
        worker_log_tail_pids+=("$!")
    done
}

stop_worker_log_snapshots()
{
    local tail_pid
    if ((${#worker_log_tail_pids[@]} == 0)); then
        return
    fi
    sleep 2
    for tail_pid in "${worker_log_tail_pids[@]}"; do
        kill "${tail_pid}" 2>/dev/null || true
    done
    for tail_pid in "${worker_log_tail_pids[@]}"; do
        wait "${tail_pid}" 2>/dev/null || true
    done
    worker_log_tail_pids=()
}

cleanup()
{
    set +e
    stop_worker_log_snapshots
    "${HEARTBEAT_CONTROLLER}" stop --state-dir "${HEARTBEAT_STATE}" \
        >> "${HEARTBEAT_STATE}/exit-trap.log" 2>&1
}
trap cleanup EXIT INT TERM

"${HEARTBEAT_CONTROLLER}" dry-run \
    --phase model-loading \
    --state-dir "${HEARTBEAT_STATE}" \
    --server-log "${SERVER_LOG}" \
    --container-image "${container_image}" \
    --nodes "${SLURM_JOB_NUM_NODES}" \
    --gpus-per-node 4 \
    --stop-regex "${STOP_REGEX}" \
    --job-id "${SLURM_JOB_ID}" \
    > "${HEARTBEAT_STATE}/dry-run.txt"
grep -qx "total_tasks=$((SLURM_JOB_NUM_NODES * 4))" "${HEARTBEAT_STATE}/dry-run.txt"

"${HEARTBEAT_CONTROLLER}" start \
    --phase model-loading \
    --state-dir "${HEARTBEAT_STATE}" \
    --server-log "${SERVER_LOG}" \
    --container-image "${container_image}" \
    --nodes "${SLURM_JOB_NUM_NODES}" \
    --gpus-per-node 4 \
    --stop-regex "${STOP_REGEX}" \
    --job-id "${SLURM_JOB_ID}"

export PATH="${LOCAL_BIN}:${PATH}"
start_worker_log_snapshots
set +e
bash "${REFERENCE_RUNNER}" "${args[@]}"
runner_rc=$?
set -e
stop_worker_log_snapshots

"${HEARTBEAT_CONTROLLER}" stop --state-dir "${HEARTBEAT_STATE}" \
    > "${HEARTBEAT_STATE}/explicit-stop.log" 2>&1 || true
"${HEARTBEAT_CONTROLLER}" status --state-dir "${HEARTBEAT_STATE}" \
    | tee "${HEARTBEAT_STATE}/final-status.txt"

grep -qx 'state=stopped' "${HEARTBEAT_STATE}/final-status.txt"
grep -qx 'supervisor_alive=no' "${HEARTBEAT_STATE}/final-status.txt"
grep -qx "total_tasks=$((SLURM_JOB_NUM_NODES * 4))" \
    "${HEARTBEAT_STATE}/final-status.txt"
grep -qx 'stop_reason=automatic model-loading boundary matched' \
    "${HEARTBEAT_STATE}/final-status.txt"
test -s "${HEARTBEAT_STATE}/boundary.match"

if grep -R -n -E '"(GET|POST|HEAD|PUT|OPTIONS) /metrics([? ]|$)' \
    "${full_logdir}" --include='*.log' --include='*.out'; then
    echo "Forbidden /metrics request found." >&2
    runner_rc=88
fi
python3 "${SRT_GUARD}" check --repo "${SRT}" --baseline "${SRT_BASELINE}"

trap - EXIT INT TERM
exit "${runner_rc}"
