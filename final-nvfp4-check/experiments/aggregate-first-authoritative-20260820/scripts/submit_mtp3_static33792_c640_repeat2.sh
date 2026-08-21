#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly SCRIPT=${EXP}/scripts/run_aggregate_arm.sbatch
readonly CONFIG=${EXP}/recipes/mtp3-static-confirmation/static528-budget33792-c640.yaml
readonly ARM=mtp3-static33792-c640-repeat2-v1
readonly LEDGER=${EXP}/submissions/mtp3-static33792-c640-repeat2.tsv

if [[ -e "${LEDGER}" ]]; then
    echo "Refusing to overwrite existing submission ledger: ${LEDGER}" >&2
    exit 3
fi

mkdir -p "${EXP}/outputs/${ARM}" "${EXP}/submissions"
printf 'arm\tjob_id\tconfig\tnodes\tsubmitted_at\n' > "${LEDGER}"
job_id="$(sbatch --parsable \
    --job-name=nv4-mtp3-static33792-repeat2 \
    --output="${EXP}/outputs/${ARM}/slurm-%j.out" \
    --nodes=4 \
    --ntasks=16 \
    --ntasks-per-node=4 \
    --gpus-per-node=4 \
    --segment=4 \
    --export="ALL,NVFP4_CONFIG=${CONFIG},NVFP4_ARM=${ARM},NVFP4_NODES=4" \
    "${SCRIPT}")"
printf '%s\t%s\t%s\t4\t%s\n' \
    "${ARM}" "${job_id}" "${CONFIG}" "$(date --iso-8601=seconds)" \
    | tee -a "${LEDGER}"
