#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly SCRIPT=${EXP}/scripts/run_aggregate_arm.sbatch
readonly SUBMISSIONS=${EXP}/submissions/aggregate-v1-nomtp-tail.tsv

mkdir -p "${EXP}/submissions"
printf 'arm\tjob_id\tconfig\tnodes\tsubmitted_at\n' > "${SUBMISSIONS}"

submit_arm() {
    local arm="${1:?arm required}"
    local config="${2:?config required}"
    local job_id
    mkdir -p "${EXP}/outputs/${arm}"
    job_id="$(sbatch --parsable \
        --job-name="nv4-${arm}" \
        --output="${EXP}/outputs/${arm}/slurm-%j.out" \
        --nodes=4 \
        --ntasks=16 \
        --ntasks-per-node=4 \
        --gpus-per-node=4 \
        --segment=4 \
        --export="ALL,NVFP4_CONFIG=${config},NVFP4_ARM=${arm},NVFP4_NODES=4" \
        "${SCRIPT}")"
    printf '%s\t%s\t%s\t4\t%s\n' \
        "${arm}" "${job_id}" "${config}" "$(date --iso-8601=seconds)" \
        | tee -a "${SUBMISSIONS}"
}

submit_arm \
    ht-nomtp-noeplb-tail-v1 \
    "${EXP}/recipes/aggregate-ht-nomtp-adp16-cutedsl-noeplb-tail.yaml"
submit_arm \
    ht-nomtp-static528-tail-v1 \
    "${EXP}/recipes/aggregate-ht-nomtp-adp16-cutedsl-static528-tail.yaml"
