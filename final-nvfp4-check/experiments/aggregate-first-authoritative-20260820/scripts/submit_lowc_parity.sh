#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly SCRIPT=${EXP}/scripts/run_lowc_parity_arm.sbatch
readonly SUBMISSIONS=${EXP}/submissions/lowc-parity.tsv
readonly CURRENT_IMAGE=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh
readonly TRTLLM5_IMAGE=/lustre/fsw/portfolios/coreai/users/williamj/containers/trtllm-sbsa-16694-base-20260731-built3.sqsh

mkdir -p "${EXP}/outputs" "${EXP}/submissions"
printf 'arm\tjob_id\tconfig\timage\tsubmitted_at\n' > "${SUBMISSIONS}"

submit_arm() {
    local arm="${1:?arm required}"
    local config="${2:?config required}"
    local image="${3:?image required}"
    local job_id
    mkdir -p "${EXP}/outputs/${arm}"
    job_id="$(sbatch --parsable \
        --job-name="nv4-${arm}" \
        --output="${EXP}/outputs/${arm}/slurm-%j.out" \
        --nodes=2 \
        --ntasks=8 \
        --ntasks-per-node=4 \
        --gpus-per-node=4 \
        --segment=2 \
        --export="ALL,NVFP4_CONFIG=${config},NVFP4_IMAGE=${image},NVFP4_ARM=${arm}" \
        "${SCRIPT}")"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${arm}" "${job_id}" "${config}" "${image}" \
        "$(date --iso-8601=seconds)" | tee -a "${SUBMISSIONS}"
}

submit_arm \
    mtp3-current-lowc-parity-v1 \
    "${EXP}/recipes/parity/mtp3-current-lowc-v1.yaml" \
    "${CURRENT_IMAGE}"
submit_arm \
    mtp3-trtllm5-lowc-parity-v1 \
    "${EXP}/recipes/parity/mtp3-trtllm5-lowc-v1.yaml" \
    "${TRTLLM5_IMAGE}"
