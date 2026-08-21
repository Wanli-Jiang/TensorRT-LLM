#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly SCRIPT=${EXP}/scripts/run_aggregate_arm.sbatch
readonly SUBMISSIONS=${EXP}/submissions/mtp3-max-screen.tsv
readonly CONFIG_DIR=${EXP}/recipes/mtp3-max-screen

if [[ -e "${SUBMISSIONS}" ]]; then
    echo "Refusing to overwrite existing submission ledger: ${SUBMISSIONS}" >&2
    exit 3
fi

mkdir -p "${EXP}/outputs" "${EXP}/submissions"
printf 'arm\tjob_id\tconfig\tnodes\tsubmitted_at\n' > "${SUBMISSIONS}"

submit_arm()
{
    local variant="${1:?variant required}"
    local arm="mtp3-max-screen-${variant}-v1"
    local config="${CONFIG_DIR}/${variant}.yaml"
    local nodes=4
    local ntasks=16
    local job_id
    test -s "${config}"
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

submit_arm noeplb-budget8448-control
submit_arm noeplb-budget16896
submit_arm noeplb-budget33792
submit_arm static528-budget8448
