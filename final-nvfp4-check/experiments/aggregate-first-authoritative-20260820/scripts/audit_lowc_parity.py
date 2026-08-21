#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the focused MTP3 low-concurrency diagnostic reruns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from audit_current_aggregate import _audit_logs, _audit_result, _sha256, _slurm

EXPERIMENT = Path(__file__).resolve().parents[1]
CURRENT_ARM = "mtp3-current-lowc-parity-v1"
CURRENT_JOB = "525530"
DIAGNOSTIC_ARM = "mtp3-trtllm5-lowc-parity-v1-retry1"
DIAGNOSTIC_JOB = "525534"
CURRENT_RECIPE = EXPERIMENT / "recipes" / "parity" / "mtp3-current-lowc-v1.yaml"
DIAGNOSTIC_RECIPE = EXPERIMENT / "recipes" / "parity" / "mtp3-trtllm5-lowc-v1.yaml"
REFERENCE_REPORT = (
    EXPERIMENT / "reference-evidence" / "trtllm5-aggregate-audit.json"
)
OUTPUT = EXPERIMENT / "reports" / "mtp3-lowc-parity-audit.json"
CONCURRENCIES = (1, 2, 4, 8)
GPUS = 8
PROMPT_MULTIPLIER = 20


def _root(arm: str, job: str) -> Path:
    return EXPERIMENT / "outputs" / arm / job


def _result_path(root: Path, concurrency: int) -> Path:
    return (
        root
        / "logs"
        / "sa-bench_isl_8192_osl_1024"
        / f"results_concurrency_{concurrency}_gpus_{GPUS}.json"
    )


def _historical_points() -> dict[int, dict[str, Any]]:
    report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    if report["status"] != "pass":
        raise ValueError("historical TRTLLM5 reference audit did not pass")
    reference = next(
        item for item in report["references"] if item["label"] == "ll-mtp3"
    )
    return {
        int(point["concurrency"]): point
        for point in reference["points"]
        if int(point["concurrency"]) in CONCURRENCIES
    }


def _audit_run(
    arm: str,
    job: str,
    recipe: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    root = _root(arm, job)
    runtime_config = root / "config.yaml"
    if _sha256(runtime_config) != _sha256(recipe):
        raise ValueError(f"{arm}: runtime config differs from recipe")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    benchmark = config["benchmark"]
    if tuple(int(value) for value in benchmark["concurrencies"]) != CONCURRENCIES:
        raise ValueError(f"{arm}: concurrency matrix differs from the frozen diagnostic")
    if int(benchmark["num_prompts_mult"]) != PROMPT_MULTIPLIER:
        raise ValueError(f"{arm}: prompt multiplier mismatch")
    if int(benchmark["num_warmup_mult"]) != 3:
        raise ValueError(f"{arm}: warmup multiplier mismatch")
    if benchmark["reuse_http_connections"] is not False:
        raise ValueError(f"{arm}: HTTP connection mode mismatch")

    serving = config["backend"]["trtllm_config"]["aggregated"]
    for key in (
        "return_perf_metrics",
        "enable_iter_perf_stats",
        "enable_iter_req_stats",
        "print_iter_log",
    ):
        if serving[key] is not False:
            raise ValueError(f"{arm}: {key} was not disabled")

    points = {
        concurrency: _audit_result(
            _result_path(root, concurrency),
            concurrency,
            PROMPT_MULTIPLIER,
            GPUS,
        )
        for concurrency in CONCURRENCIES
    }
    result_paths = list(
        root.glob("logs/sa-bench_isl_8192_osl_1024/results_concurrency_*_gpus_8.json")
    )
    if len(result_paths) != len(CONCURRENCIES):
        raise ValueError(f"{arm}: result-file count mismatch")

    heartbeat = (root / "heartbeat.final-status.txt").read_text(encoding="utf-8")
    for marker in (
        "state=stopped",
        "supervisor_alive=no",
        "total_tasks=8",
        "stop_reason=automatic model-loading boundary matched",
    ):
        if marker not in heartbeat:
            raise ValueError(f"{arm}: heartbeat missing {marker!r}")
    formal = json.loads((root / "audit-formal-window.json").read_text(encoding="utf-8"))
    if formal["status"] != "pass":
        raise ValueError(f"{arm}: formal-window audit failed")
    return (
        {
            "arm": arm,
            "job": job,
            "slurm": _slurm(job, GPUS),
            "recipe": str(recipe),
            "recipe_sha256": _sha256(recipe),
            "metrics_hits": _audit_logs(root),
            "points": [points[concurrency] for concurrency in CONCURRENCIES],
        },
        points,
    )


def _change_percent(value: float, baseline: float) -> float:
    return 100.0 * (value / baseline - 1.0)


def main() -> None:
    current_run, current = _audit_run(
        CURRENT_ARM,
        CURRENT_JOB,
        CURRENT_RECIPE,
    )
    diagnostic_run, diagnostic = _audit_run(
        DIAGNOSTIC_ARM,
        DIAGNOSTIC_JOB,
        DIAGNOSTIC_RECIPE,
    )
    historical = _historical_points()
    comparisons = []
    historical_outliers = []
    for concurrency in CONCURRENCIES:
        current_tps = float(current[concurrency]["total_tps_per_gpu"])
        diagnostic_tps = float(diagnostic[concurrency]["total_tps_per_gpu"])
        historical_tps = float(historical[concurrency]["total_tps_per_gpu"])
        historical_delta = _change_percent(current_tps, historical_tps)
        diagnostic_delta = _change_percent(current_tps, diagnostic_tps)
        if abs(historical_delta) > 5.0:
            historical_outliers.append(concurrency)
        comparisons.append(
            {
                "concurrency": concurrency,
                "current_total_tps_per_gpu": current_tps,
                "historical_trtllm5_total_tps_per_gpu": historical_tps,
                "current_vs_historical_percent": historical_delta,
                "same_day_diagnostic_total_tps_per_gpu": diagnostic_tps,
                "current_vs_same_day_diagnostic_percent": diagnostic_delta,
                "historical_parity_status": (
                    "pass" if abs(historical_delta) <= 5.0 else "localized-regression"
                ),
            }
        )

    report = {
        "integrity_status": "pass",
        "performance_status": (
            "localized-c2-regression"
            if historical_outliers == [2]
            else "review-required"
            if historical_outliers
            else "parity"
        ),
        "historical_outlier_concurrencies": historical_outliers,
        "diagnostic_scope": {
            "authoritative_baseline": "TRTLLM5 job 422739",
            "same_day_diagnostic": (
                "TRTLLM5 current checkout mounted read-only over its older runtime image; "
                "not a bitwise 2026-08-11 source snapshot"
            ),
        },
        "current_run": current_run,
        "same_day_diagnostic_run": diagnostic_run,
        "comparisons": comparisons,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
