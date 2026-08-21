#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the focused MTP3 Static528 confirmation and interaction jobs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml
from audit_current_aggregate import _audit_logs, _audit_result, _sha256, _slurm

EXPERIMENT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "static528-budget8448-curve",
    "static528-budget33792-c640",
)
SCREEN_REPORT = EXPERIMENT / "reports" / "mtp3-aggregate-max-screen.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "mtp3-static-confirmation.tsv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            EXPERIMENT / "recipes" / "mtp3-static-confirmation" / "manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "reports" / "mtp3-static-confirmation.json",
    )
    return parser.parse_args()


def _read_submissions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    prefix = "mtp3-static-confirm-"
    result = {
        row["arm"].removeprefix(prefix).removesuffix("-v1"): row for row in rows
    }
    if tuple(result) != VARIANTS or len(rows) != len(VARIANTS):
        raise ValueError("submission ledger differs from the frozen two-arm design")
    return result


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


def _audit_arm(
    variant: str, row: dict[str, str], manifest: dict[str, Any]
) -> dict[str, Any]:
    recipe = Path(row["config"])
    if recipe != Path(manifest["path"]) or _sha256(recipe) != manifest["sha256"]:
        raise ValueError(f"{variant}: submitted recipe differs from manifest")
    for mounted in manifest["mounted_files"]:
        mounted_path = Path(mounted["path"])
        if _sha256(mounted_path) != mounted["sha256"]:
            raise ValueError(f"{variant}: mounted file differs from manifest")
    root = EXPERIMENT / "outputs" / row["arm"] / row["job_id"]
    runtime = root / "config.yaml"
    if _sha256(runtime) != _sha256(recipe):
        raise ValueError(f"{variant}: runtime config differs from recipe")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    serving = config["backend"]["trtllm_config"]["aggregated"]
    if serving["enable_chunked_prefill"] is not True:
        raise ValueError(f"{variant}: chunked prefill was not enabled")
    if int(serving["max_num_tokens"]) != int(manifest["max_num_tokens"]):
        raise ValueError(f"{variant}: scheduler token budget mismatch")
    if int(serving["moe_config"]["max_num_tokens"]) != int(
        manifest["moe_max_num_tokens"]
    ):
        raise ValueError(f"{variant}: MoE token budget mismatch")
    if not serving["moe_config"].get("load_balancer"):
        raise ValueError(f"{variant}: Static528 placement is missing")
    if serving["kv_cache_config"]["use_kv_cache_manager_v2"] is not False:
        raise ValueError(f"{variant}: KV manager differs from V1")
    for key in (
        "return_perf_metrics",
        "enable_iter_perf_stats",
        "enable_iter_req_stats",
        "print_iter_log",
    ):
        if serving[key] is not False:
            raise ValueError(f"{variant}: {key} was not disabled")
    points = [
        _audit_result(
            root
            / "logs"
            / "sa-bench_isl_8192_osl_1024"
            / f"results_concurrency_{concurrency}_gpus_16.json",
            concurrency,
            3,
            16,
        )
        for concurrency in manifest["concurrencies"]
    ]
    formal = json.loads((root / "audit-formal-window.json").read_text(encoding="utf-8"))
    if formal["status"] != "pass":
        raise ValueError(f"{variant}: formal-window audit failed")
    _audit_logs(root)
    return {
        "variant": variant,
        "role": manifest["role"],
        "placement": manifest["placement"],
        "max_num_tokens": manifest["max_num_tokens"],
        "moe_max_num_tokens": manifest["moe_max_num_tokens"],
        "job": row["job_id"],
        "output": str(root),
        "heartbeat": _heartbeat(root),
        "slurm": _slurm(row["job_id"], 16),
        "points": points,
    }


def main() -> None:
    args = _parse_args()
    submissions = _read_submissions(args.submissions)
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = {row["variant"]: row for row in manifest_rows}
    if tuple(manifest) != VARIANTS:
        raise ValueError("manifest differs from the frozen two-arm design")
    arms = [
        _audit_arm(variant, submissions[variant], manifest[variant])
        for variant in VARIANTS
    ]
    screen = json.loads(SCREEN_REPORT.read_text(encoding="utf-8"))
    screen_static = next(
        arm for arm in screen["arms"] if arm["variant"] == "static528-budget8448"
    )["point"]
    curve_c640 = next(
        point for point in arms[0]["points"] if point["concurrency"] == 640
    )
    interaction_c640 = arms[1]["points"][0]
    all_points = [point for arm in arms for point in arm["points"]]
    peak = max(all_points, key=lambda point: point["total_tps_per_gpu"])
    report = {
        "integrity_status": "pass",
        "screen_static528_budget8448_c640_tps_per_gpu": screen_static[
            "total_tps_per_gpu"
        ],
        "repeat_static528_budget8448_c640_tps_per_gpu": curve_c640[
            "total_tps_per_gpu"
        ],
        "repeat_change_percent": _change(
            curve_c640["total_tps_per_gpu"], screen_static["total_tps_per_gpu"]
        ),
        "budget33792_interaction_change_vs_repeat_budget8448_percent": _change(
            interaction_c640["total_tps_per_gpu"], curve_c640["total_tps_per_gpu"]
        ),
        "peak": peak,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
