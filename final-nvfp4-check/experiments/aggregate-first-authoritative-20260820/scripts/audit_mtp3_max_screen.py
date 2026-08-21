#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the aggregate MTP3 token-budget and Static-EPLB C640 screen."""

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
    "noeplb-budget8448-control",
    "noeplb-budget16896",
    "noeplb-budget33792",
    "static528-budget8448",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "mtp3-max-screen.tsv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT / "recipes" / "mtp3-max-screen" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "reports" / "mtp3-aggregate-max-screen.json",
    )
    return parser.parse_args()


def _read_submissions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {
        row["arm"].removeprefix("mtp3-max-screen-").removesuffix("-v1"): row
        for row in rows
    }
    if tuple(result) != VARIANTS or len(rows) != len(VARIANTS):
        raise ValueError("submission ledger differs from the frozen four-arm screen")
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
    has_placement = bool(serving["moe_config"].get("load_balancer"))
    if has_placement != (manifest["placement"] == "Static528"):
        raise ValueError(f"{variant}: placement mode mismatch")
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
        "point": point,
    }


def main() -> None:
    args = _parse_args()
    submissions = _read_submissions(args.submissions)
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = {row["variant"]: row for row in manifest_rows}
    if tuple(manifest) != VARIANTS:
        raise ValueError("manifest differs from the frozen four-arm screen")
    arms = [
        _audit_arm(variant, submissions[variant], manifest[variant])
        for variant in VARIANTS
    ]
    control = arms[0]["point"]["total_tps_per_gpu"]
    for arm in arms:
        arm["change_vs_same_day_control_percent"] = _change(
            arm["point"]["total_tps_per_gpu"], control
        )
    peak = max(arms, key=lambda arm: arm["point"]["total_tps_per_gpu"])
    report = {
        "integrity_status": "pass",
        "control": VARIANTS[0],
        "peak": peak,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
