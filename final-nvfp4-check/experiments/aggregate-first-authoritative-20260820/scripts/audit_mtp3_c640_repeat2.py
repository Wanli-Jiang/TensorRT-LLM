#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the third MTP3 Static528/budget33792 C640 repeat."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import yaml
from audit_current_aggregate import _audit_logs, _audit_result, _sha256, _slurm

EXPERIMENT = Path(__file__).resolve().parents[1]
LEDGER = EXPERIMENT / "submissions" / "mtp3-static33792-c640-repeat2.tsv"
MANIFEST = EXPERIMENT / "recipes" / "mtp3-static-confirmation" / "manifest.json"
PRIOR_REPORT = EXPERIMENT / "reports" / "mtp3-static-confirmation.json"
CURVE_REPORT = EXPERIMENT / "reports" / "mtp3-budget33792-curve.json"
OUTPUT = EXPERIMENT / "reports" / "mtp3-static33792-c640-repeat2.json"
EXPECTED_ARM = "mtp3-static33792-c640-repeat2-v1"


def _heartbeat(root: Path) -> dict[str, str]:
    text = (root / "heartbeat.final-status.txt").read_text(encoding="utf-8")
    expected = {
        "state": "stopped",
        "supervisor_alive": "no",
        "total_tasks": "16",
        "stop_reason": "automatic model-loading boundary matched",
    }
    for key, value in expected.items():
        if f"{key}={value}" not in text:
            raise ValueError(f"{root}: heartbeat {key} mismatch")
    return expected


def _change(current: float, baseline: float) -> float:
    return 100.0 * (current / baseline - 1.0)


def _read_submission() -> dict[str, str]:
    with LEDGER.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1 or rows[0]["arm"] != EXPECTED_ARM:
        raise ValueError("repeat2 submission ledger is not the frozen one-arm design")
    return rows[0]


def _audit_repeat(row: dict[str, str], manifest: dict[str, Any]) -> dict[str, Any]:
    recipe = Path(row["config"])
    if recipe != Path(manifest["path"]) or _sha256(recipe) != manifest["sha256"]:
        raise ValueError("repeat2 submitted recipe differs from manifest")
    for mounted in manifest["mounted_files"]:
        mounted_path = Path(mounted["path"])
        if _sha256(mounted_path) != mounted["sha256"]:
            raise ValueError("repeat2 mounted file differs from manifest")
    root = EXPERIMENT / "outputs" / row["arm"] / row["job_id"]
    if _sha256(root / "config.yaml") != _sha256(recipe):
        raise ValueError("repeat2 runtime config differs from recipe")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    serving = config["backend"]["trtllm_config"]["aggregated"]
    if serving["enable_chunked_prefill"] is not True:
        raise ValueError("repeat2 chunked prefill was not enabled")
    if int(serving["max_num_tokens"]) != 33792:
        raise ValueError("repeat2 scheduler token budget mismatch")
    if int(serving["moe_config"]["max_num_tokens"]) != 33792:
        raise ValueError("repeat2 MoE token budget mismatch")
    if not serving["moe_config"].get("load_balancer"):
        raise ValueError("repeat2 Static528 placement is missing")
    if serving["kv_cache_config"]["use_kv_cache_manager_v2"] is not False:
        raise ValueError("repeat2 KV manager differs from V1")
    point = _audit_result(
        root
        / "logs"
        / "sa-bench_isl_8192_osl_1024"
        / "results_concurrency_640_gpus_16.json",
        640,
        3,
        16,
    )
    formal = json.loads((root / "audit-formal-window.json").read_text(encoding="utf-8"))
    if formal["status"] != "pass":
        raise ValueError("repeat2 formal-window audit failed")
    _audit_logs(root)
    return {
        "job": row["job_id"],
        "output": str(root),
        "point": point,
        "heartbeat": _heartbeat(root),
        "slurm": _slurm(row["job_id"], 16),
    }


def main() -> None:
    row = _read_submission()
    manifest_rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest = next(
        item
        for item in manifest_rows
        if item["variant"] == "static528-budget33792-c640"
    )
    repeat2 = _audit_repeat(row, manifest)
    prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    curve = json.loads(CURVE_REPORT.read_text(encoding="utf-8"))
    first = next(
        arm
        for arm in prior["arms"]
        if arm["variant"] == "static528-budget33792-c640"
    )["points"][0]
    curve_repeat = next(
        point for point in curve["curve"]["points"] if point["concurrency"] == 640
    )
    budget8448_repeat = prior["repeat_static528_budget8448_c640_tps_per_gpu"]
    runs = [
        {
            "label": "first-screen",
            "job": next(
                arm["job"]
                for arm in prior["arms"]
                if arm["variant"] == "static528-budget33792-c640"
            ),
            "total_tps_per_gpu": first["total_tps_per_gpu"],
        },
        {
            "label": "full-curve-repeat",
            "job": curve["curve"]["job"],
            "total_tps_per_gpu": curve_repeat["total_tps_per_gpu"],
        },
        {
            "label": "third-independent-repeat",
            "job": repeat2["job"],
            "total_tps_per_gpu": repeat2["point"]["total_tps_per_gpu"],
        },
    ]
    values = [run["total_tps_per_gpu"] for run in runs]
    report = {
        "integrity_status": "pass",
        "configuration": {
            "placement": "Static528",
            "enable_chunked_prefill": True,
            "max_num_tokens": 33792,
            "moe_max_num_tokens": 33792,
            "concurrency": 640,
            "kv_manager": "V1",
        },
        "runs": runs,
        "repeat_statistics": {
            "minimum_tps_per_gpu": min(values),
            "maximum_tps_per_gpu": max(values),
            "mean_tps_per_gpu": statistics.fmean(values),
            "median_tps_per_gpu": statistics.median(values),
            "max_to_min_spread_percent": _change(max(values), min(values)),
            "minimum_change_vs_budget8448_repeat_percent": _change(
                min(values), budget8448_repeat
            ),
        },
        "repeat2": repeat2,
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
