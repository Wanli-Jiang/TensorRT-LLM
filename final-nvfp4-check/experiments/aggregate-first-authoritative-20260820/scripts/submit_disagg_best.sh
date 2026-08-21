#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly EXP=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/williamj/TensorRT-LLM/final-nvfp4-check/experiments/aggregate-first-authoritative-20260820
readonly SUBMIT_PY=/lustre/fsw/portfolios/coreai/users/williamj/nemotron-ultra-benchmarking/disagg/bin/harness/trtllm-disagg-benchmark/submit.py
readonly LEDGER=${EXP}/submissions/disaggregate-best.tsv

mode="${1:-}"
case "${mode}" in
    nomtp|mtp3) ;;
    *)
        echo "Usage: $0 {nomtp|mtp3}" >&2
        exit 2
        ;;
esac

readonly CONFIG=${EXP}/recipes/disaggregate/best-noeplb-${mode}.yaml
readonly OUTPUT=${EXP}/outputs/disaggregate/${mode}-best-noeplb-v1
readonly SUBMIT_LOG=${EXP}/submissions/disaggregate-${mode}-best.submit.log

test -s "${SUBMIT_PY}"
test -s "${CONFIG}"
if [[ -e "${OUTPUT}" ]]; then
    echo "Refusing to overwrite existing output directory: ${OUTPUT}" >&2
    exit 3
fi
if [[ -s "${LEDGER}" ]] && awk -F '\t' -v mode="${mode}" 'NR > 1 && $1 == mode {found=1} END {exit !found}' "${LEDGER}"; then
    echo "A ${mode} submission is already recorded in ${LEDGER}" >&2
    exit 4
fi

mkdir -p "${EXP}/outputs/disaggregate" "${EXP}/submissions"
set +e
submit_output="$(python3 "${SUBMIT_PY}" \
    --config "${CONFIG}" \
    --log-dir "${OUTPUT}" 2>&1)"
submit_rc=$?
set -e
printf '%s\n' "${submit_output}" | tee "${SUBMIT_LOG}"
if ((submit_rc != 0)); then
    exit "${submit_rc}"
fi

job_id="$(printf '%s\n' "${submit_output}" \
    | sed -n 's/^Submitted batch job \([0-9][0-9]*\)$/\1/p' \
    | tail -1)"
if [[ -z "${job_id}" ]]; then
    echo "Submission returned no Slurm job ID." >&2
    exit 5
fi

if [[ ! -e "${LEDGER}" ]]; then
    printf 'mode\tjob_id\tconfig\toutput\tdeployed_gpus\tsubmitted_at\n' > "${LEDGER}"
fi
case "${mode}" in
    nomtp) deployed_gpus=40 ;;
    mtp3) deployed_gpus=48 ;;
esac
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${mode}" "${job_id}" "${CONFIG}" "${OUTPUT}" "${deployed_gpus}" \
    "$(date --iso-8601=seconds)" | tee -a "${LEDGER}"
