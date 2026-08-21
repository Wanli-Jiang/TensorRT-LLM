#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit the MTP3 Static528 budget33792 curve and budget67584 bracket."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml
from audit_current_aggregate import _audit_logs, _audit_result, _sha256, _slurm

EXPERIMENT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "static528-budget33792-curve",
    "static528-budget67584-c640",
)
PRIOR_REPORT = EXPERIMENT / "reports" / "mtp3-static-confirmation.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "mtp3-budget33792-curve.tsv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(EXPERIMENT / "recipes" / "mtp3-budget33792-curve" / "manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "reports" / "mtp3-budget33792-curve.json",
    )
    return parser.parse_args()


def _read_submissions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    prefix = "mtp3-record-"
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


def _failed_slurm(job: str) -> dict[str, str]:
    proc = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            job,
            "--format=JobIDRaw,State,ExitCode,AllocTRES,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split("|") for line in proc.stdout.splitlines() if line]
    row = next((item for item in rows if item[0] == job), None)
    if row is None:
        raise ValueError(f"missing Slurm allocation row for {job}")
    if row[1:3] != ["FAILED", "87:0"]:
        raise ValueError(f"job {job} does not have the expected failure: {row}")
    if "gres/gpu=16" not in row[3]:
        raise ValueError(f"job {job} GPU allocation mismatch: {row[3]}")
    domains = set(re.findall(r"nvl72d[0-9]+", row[4]))
    if len(domains) != 1:
        raise ValueError(f"job {job} crossed NVL72 domains: {row[4]}")
    return {
        "state": row[1],
        "exit_code": row[2],
        "allocation": row[3],
        "nodes": row[4],
        "domain": next(iter(domains)),
    }


def _audit_expected_upper_failure(
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
    if _sha256(root / "config.yaml") != _sha256(recipe):
        raise ValueError(f"{variant}: runtime config differs from recipe")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    serving = config["backend"]["trtllm_config"]["aggregated"]
    if serving["enable_chunked_prefill"] is not True:
        raise ValueError(f"{variant}: chunked prefill was not enabled")
    if int(serving["max_num_tokens"]) != 67584:
        raise ValueError(f"{variant}: scheduler token budget mismatch")
    if int(serving["moe_config"]["max_num_tokens"]) != 67584:
        raise ValueError(f"{variant}: MoE token budget mismatch")
    if not serving["moe_config"].get("load_balancer"):
        raise ValueError(f"{variant}: Static528 placement is missing")
    result = (
        root
        / "logs"
        / "sa-bench_isl_8192_osl_1024"
        / "results_concurrency_640_gpus_16.json"
    )
    if result.exists():
        raise ValueError(f"{variant}: unexpected formal result exists")
    benchmark_text = (root / "logs" / "benchmark.out").read_text(
        encoding="utf-8", errors="replace"
    )
    successful = [
        int(value)
        for value in re.findall(r"Successful requests:\s+([0-9]+)", benchmark_text)
    ]
    if successful != [114]:
        raise ValueError(f"{variant}: unexpected warmup completion counts {successful}")
    if "Initial test run failed" not in benchmark_text:
        raise ValueError(f"{variant}: missing formal initial-gate failure")
    server_logs = list((root / "logs").glob("*_agg_w0.out"))
    if len(server_logs) != 1:
        raise ValueError(f"{variant}: expected one server log, got {server_logs}")
    server_text = server_logs[0].read_text(encoding="utf-8", errors="replace")
    for needle in (
        "moe_load_balance_routing",
        "moeComputeRouteDevice",
        "CUDA runtime error in cudaLaunchKernelEx",
        "unspecified launch failure",
    ):
        if needle not in server_text:
            raise ValueError(f"{variant}: missing failure evidence {needle!r}")
    if (root / "arm-exit-code.txt").read_text(encoding="utf-8").strip() != "87":
        raise ValueError(f"{variant}: unexpected arm exit code")
    if (root / "audit-formal-window.json").exists():
        raise ValueError(f"{variant}: failed arm unexpectedly has a formal-window audit")
    _audit_logs(root)
    return {
        "variant": variant,
        "role": manifest["role"],
        "placement": manifest["placement"],
        "max_num_tokens": manifest["max_num_tokens"],
        "moe_max_num_tokens": manifest["moe_max_num_tokens"],
        "job": row["job_id"],
        "output": str(root),
        "status": "invalid-upper-budget-kernel-failure",
        "warmup_planned_requests": 640,
        "warmup_completed_requests": 114,
        "formal_result_present": False,
        "failure_kernel": "moeComputeRouteDevice / moe_load_balance_routing",
        "failure": "CUDA unspecified launch failure",
        "heartbeat": _heartbeat(root),
        "slurm": _failed_slurm(row["job_id"]),
        "server_log": str(server_logs[0]),
        "server_log_sha256": _sha256(server_logs[0]),
    }


def main() -> None:
    args = _parse_args()
    submissions = _read_submissions(args.submissions)
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = {row["variant"]: row for row in manifest_rows}
    if tuple(manifest) != VARIANTS:
        raise ValueError("manifest differs from the frozen two-arm design")
    curve = _audit_arm(VARIANTS[0], submissions[VARIANTS[0]], manifest[VARIANTS[0]])
    upper_failure = _audit_expected_upper_failure(
        VARIANTS[1], submissions[VARIANTS[1]], manifest[VARIANTS[1]]
    )
    prior = json.loads(PRIOR_REPORT.read_text(encoding="utf-8"))
    prior_c640 = curve["points"][0]
    for arm in prior["arms"]:
        if arm["variant"] == "static528-budget33792-c640":
            prior_c640 = arm["points"][0]
            break
    curve_c640 = next(
        point for point in curve["points"] if point["concurrency"] == 640
    )
    peak = max(curve["points"], key=lambda point: point["total_tps_per_gpu"])
    report = {
        "integrity_status": "pass",
        "prior_budget33792_c640_tps_per_gpu": prior_c640["total_tps_per_gpu"],
        "repeat_budget33792_c640_tps_per_gpu": curve_c640["total_tps_per_gpu"],
        "repeat_change_percent": _change(
            curve_c640["total_tps_per_gpu"], prior_c640["total_tps_per_gpu"]
        ),
        "budget67584_status": upper_failure["status"],
        "peak": peak,
        "curve": curve,
        "upper_budget_failure": upper_failure,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
