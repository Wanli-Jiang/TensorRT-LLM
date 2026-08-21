#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly SCRIPT=${EXP}/scripts/run_aggregate_arm.sbatch
readonly SUBMISSIONS=${EXP}/submissions/aggregate-v1.tsv

mkdir -p "${EXP}/outputs" "${EXP}/submissions"
printf 'arm\tjob_id\tconfig\tnodes\tsubmitted_at\n' > "${SUBMISSIONS}"

submit_arm() {
    local arm="${1:?arm required}"
    local config="${2:?config required}"
    local nodes="${3:?nodes required}"
    local ntasks="$((nodes * 4))"
    local job_id
    mkdir -p "${EXP}/outputs/${arm}"
    job_id="$(sbatch --parsable \
        --job-name="nv4-${arm}" \
        --output="${EXP}/outputs/${arm}/slurm-%j.out" \
        --nodes="${nodes}" \
        --ntasks="${ntasks}" \
        --ntasks-per-node=4 \
        --gpus-per-node=4 \
        --segment="${nodes}" \
        --export="ALL,NVFP4_CONFIG=${config},NVFP4_ARM=${arm},NVFP4_NODES=${nodes}" \
        "${SCRIPT}")"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${arm}" "${job_id}" "${config}" "${nodes}" \
        "$(date --iso-8601=seconds)" | tee -a "${SUBMISSIONS}"
}

submit_arm \
    ll-nomtp-v1 \
    "${EXP}/recipes/aggregate-ll-nomtp-tp8-trtllm.yaml" \
    2
submit_arm \
    ll-mtp3-v1 \
    "${EXP}/recipes/aggregate-ll-mtp3-tp8-trtllm.yaml" \
    2
submit_arm \
    ht-nomtp-noeplb-v1 \
    "${EXP}/recipes/aggregate-ht-nomtp-adp16-cutedsl-noeplb.yaml" \
    4
submit_arm \
    ht-nomtp-static528-v1 \
    "${EXP}/recipes/aggregate-ht-nomtp-adp16-cutedsl-static528.yaml" \
    4
submit_arm \
    ht-mtp3-noeplb-v1 \
    "${EXP}/recipes/aggregate-ht-mtp3-adp16-cutedsl-noeplb.yaml" \
    4
submit_arm \
    ht-mtp3-static528-v1 \
    "${EXP}/recipes/aggregate-ht-mtp3-adp16-cutedsl-static528.yaml" \
    4
