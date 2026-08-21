#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the two completed chunk8448 no-EPLB MTP3 C640 controls."""

from __future__ import annotations

import json
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
MAIN = (
    EXPERIMENT
    / "outputs"
    / "mtp3-chunk8448-v1"
    / "526770"
    / "logs"
    / "sa-bench_isl_8192_osl_1024"
    / "results_concurrency_640_gpus_16.json"
)
REPEAT = (
    EXPERIMENT
    / "outputs"
    / "mtp3-max-screen-noeplb-budget8448-control-v1"
    / "527208"
    / "logs"
    / "sa-bench_isl_8192_osl_1024"
    / "results_concurrency_640_gpus_16.json"
)
OUTPUT = EXPERIMENT / "reports" / "mtp3-c640-control-repeat-trajectory.json"
METRICS = (
    "total_token_throughput",
    "output_throughput",
    "peak_output_tokens_per_s",
    "median_ttft_ms",
    "median_tpot_ms",
    "median_itl_ms",
    "median_e2el_ms",
    "duration",
)


def _change(current: float, baseline: float) -> float:
    return 100.0 * (current / baseline - 1.0)


def main() -> None:
    main_result = json.loads(MAIN.read_text(encoding="utf-8"))
    repeat_result = json.loads(REPEAT.read_text(encoding="utf-8"))
    for result in (main_result, repeat_result):
        if result["completed"] != 1920:
            raise ValueError("C640 control did not complete exactly 1920 requests")
        if result["total_input_tokens"] != 1920 * 8192:
            raise ValueError("C640 control input-token invariant failed")
        if result["total_output_tokens"] != 1920 * 1024:
            raise ValueError("C640 control output-token invariant failed")
        if any(result["errors"]):
            raise ValueError("C640 control contains a non-empty request error")
    main_texts = main_result["generated_texts"]
    repeat_texts = repeat_result["generated_texts"]
    if len(main_texts) != len(repeat_texts):
        raise ValueError("C640 controls contain different generated-text counts")
    comparisons = {}
    for metric in METRICS:
        baseline = float(main_result[metric])
        current = float(repeat_result[metric])
        comparisons[metric] = {
            "main": baseline,
            "repeat": current,
            "repeat_change_percent": _change(current, baseline),
        }
    report = {
        "integrity_status": "pass",
        "configuration": "MTP3 no-EPLB chunked prefill, scheduler/MoE budget 8448, C640",
        "main": {"job": "526770", "result": str(MAIN), "nodes": "nvl72d167-T[12,15-17]"},
        "repeat": {
            "job": "527208",
            "result": str(REPEAT),
            "nodes": "nvl72d156-T[10-12,16]",
        },
        "comparisons": comparisons,
        "generated_text_exact_match_fraction": sum(
            left == right for left, right in zip(main_texts, repeat_texts)
        )
        / len(main_texts),
        "interpretation": (
            "The repeat has a slightly higher instantaneous peak output rate but "
            "lower average throughput and low exact-text agreement, which is the "
            "known allocation/trajectory-variance phenotype rather than evidence "
            "of a lower kernel compute ceiling."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
