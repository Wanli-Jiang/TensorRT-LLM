#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit and decompose the matched MTP3 chunked-prefill experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml
from audit_current_aggregate import _audit_logs, _audit_result, _sha256, _slurm

EXPERIMENT = Path(__file__).resolve().parents[1]
VARIANTS = ("chunk1024", "chunk8448", "nochunk8448")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "mtp3-nochunk-ab.tsv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=EXPERIMENT / "recipes" / "mtp3-nochunk-ab" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "reports" / "mtp3-nochunk-causal-ab.json",
    )
    return parser.parse_args()


def _read_submissions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {row["arm"].removeprefix("mtp3-").removesuffix("-v1"): row for row in rows}
    if tuple(result) != VARIANTS or len(rows) != len(VARIANTS):
        raise ValueError("submission ledger differs from the frozen three-arm matrix")
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


def _audit_arm(
    variant: str,
    row: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    recipe = Path(row["config"])
    if recipe != Path(manifest["path"]) or _sha256(recipe) != manifest["sha256"]:
        raise ValueError(f"{variant}: submitted recipe differs from manifest")
    root = EXPERIMENT / "outputs" / row["arm"] / row["job_id"]
    runtime = root / "config.yaml"
    if _sha256(runtime) != _sha256(recipe):
        raise ValueError(f"{variant}: runtime config differs from recipe")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    serving = config["backend"]["trtllm_config"]["aggregated"]
    if serving["enable_chunked_prefill"] is not manifest["enable_chunked_prefill"]:
        raise ValueError(f"{variant}: chunked-prefill setting mismatch")
    if int(serving["max_num_tokens"]) != int(manifest["max_num_tokens"]):
        raise ValueError(f"{variant}: scheduler token budget mismatch")
    if int(serving["moe_config"]["max_num_tokens"]) != int(
        manifest["moe_max_num_tokens"]
    ):
        raise ValueError(f"{variant}: MoE token budget mismatch")
    if serving["kv_cache_config"]["use_kv_cache_manager_v2"] is not False:
        raise ValueError(f"{variant}: KV manager differs from the V1 baseline")
    for key in (
        "return_perf_metrics",
        "enable_iter_perf_stats",
        "enable_iter_req_stats",
        "print_iter_log",
    ):
        if serving[key] is not False:
            raise ValueError(f"{variant}: {key} was not disabled")
    points = []
    for concurrency_value in config["benchmark"]["concurrencies"]:
        concurrency = int(concurrency_value)
        path = (
            root
            / "logs"
            / "sa-bench_isl_8192_osl_1024"
            / f"results_concurrency_{concurrency}_gpus_16.json"
        )
        points.append(_audit_result(path, concurrency, 3, 16))
    formal = json.loads((root / "audit-formal-window.json").read_text(encoding="utf-8"))
    if formal["status"] != "pass":
        raise ValueError(f"{variant}: formal-window audit failed")
    _audit_logs(root)
    slurm = _slurm(row["job_id"], 16)
    return {
        "variant": variant,
        "role": manifest["role"],
        "job": row["job_id"],
        "output": str(root),
        "enable_chunked_prefill": manifest["enable_chunked_prefill"],
        "max_num_tokens": manifest["max_num_tokens"],
        "moe_max_num_tokens": manifest["moe_max_num_tokens"],
        "heartbeat": _heartbeat(root),
        "slurm": slurm,
        "points": points,
    }


def _change(current: float, baseline: float) -> float:
    return 100.0 * (current / baseline - 1.0)


def main() -> None:
    args = _parse_args()
    submissions = _read_submissions(args.submissions)
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = {row["variant"]: row for row in manifest_rows}
    if tuple(manifest) != VARIANTS:
        raise ValueError("manifest differs from the frozen three-arm matrix")
    arms = {
        variant: _audit_arm(variant, submissions[variant], manifest[variant])
        for variant in VARIANTS
    }
    indexed = {
        variant: {int(point["concurrency"]): point for point in arm["points"]}
        for variant, arm in arms.items()
    }
    comparisons = []
    for concurrency in manifest["chunk1024"]["concurrencies"]:
        baseline = indexed["chunk1024"][concurrency]
        large_budget = indexed["chunk8448"][concurrency]
        no_chunk = indexed["nochunk8448"][concurrency]
        comparisons.append(
            {
                "concurrency": concurrency,
                "chunk1024_total_tps_per_gpu": baseline["total_tps_per_gpu"],
                "chunk8448_total_tps_per_gpu": large_budget["total_tps_per_gpu"],
                "nochunk8448_total_tps_per_gpu": no_chunk["total_tps_per_gpu"],
                "large_budget_effect_percent": _change(
                    large_budget["total_tps_per_gpu"], baseline["total_tps_per_gpu"]
                ),
                "chunk_flag_effect_percent": _change(
                    no_chunk["total_tps_per_gpu"], large_budget["total_tps_per_gpu"]
                ),
                "combined_nochunk_effect_percent": _change(
                    no_chunk["total_tps_per_gpu"], baseline["total_tps_per_gpu"]
                ),
                "chunk_flag_ttft_change_percent": _change(
                    no_chunk["median_ttft_ms"], large_budget["median_ttft_ms"]
                ),
                "chunk_flag_tpot_change_percent": _change(
                    no_chunk["median_tpot_ms"], large_budget["median_tpot_ms"]
                ),
            }
        )
    all_points = [
        {"variant": variant, **point}
        for variant, arm in arms.items()
        for point in arm["points"]
    ]
    peak = max(all_points, key=lambda point: point["total_tps_per_gpu"])
    report = {
        "integrity_status": "pass",
        "causal_design": {
            "large_budget_effect": "chunk8448 / chunk1024",
            "chunk_flag_effect": "nochunk8448 / chunk8448",
            "combined_nochunk_effect": "nochunk8448 / chunk1024",
        },
        "peak": peak,
        "comparisons": comparisons,
        "arms": [arms[variant] for variant in VARIANTS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
