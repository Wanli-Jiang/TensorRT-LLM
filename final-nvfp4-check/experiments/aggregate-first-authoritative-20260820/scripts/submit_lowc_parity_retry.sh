#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly SCRIPT=${EXP}/scripts/run_lowc_parity_arm.sbatch
readonly CONFIG=${EXP}/recipes/parity/mtp3-trtllm5-lowc-v1.yaml
readonly IMAGE=/lustre/fsw/portfolios/coreai/users/williamj/containers/trtllm-sbsa-16694-base-20260731-built3.sqsh
readonly ARM=mtp3-trtllm5-lowc-parity-v1-retry1
readonly SUBMISSIONS=${EXP}/submissions/lowc-parity-retries.tsv

mkdir -p "${EXP}/outputs/${ARM}" "${EXP}/submissions"
if [[ ! -s "${SUBMISSIONS}" ]]; then
    printf 'arm\tjob_id\tconfig\timage\tsubmitted_at\treplaces\n' > "${SUBMISSIONS}"
fi

job_id="$(sbatch --parsable \
    --job-name="nv4-${ARM}" \
    --output="${EXP}/outputs/${ARM}/slurm-%j.out" \
    --nodes=2 \
    --ntasks=8 \
    --ntasks-per-node=4 \
    --gpus-per-node=4 \
    --segment=2 \
    --export="ALL,NVFP4_CONFIG=${CONFIG},NVFP4_IMAGE=${IMAGE},NVFP4_ARM=${ARM}" \
    "${SCRIPT}")"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${ARM}" "${job_id}" "${CONFIG}" "${IMAGE}" \
    "$(date --iso-8601=seconds)" 525531 | tee -a "${SUBMISSIONS}"
