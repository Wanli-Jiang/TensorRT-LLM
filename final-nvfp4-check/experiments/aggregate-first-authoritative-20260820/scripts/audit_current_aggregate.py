#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit current aggregate NVFP4 arms and compare them with TRTLLM5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT = Path(__file__).resolve().parents[1]
IMAGE = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh"
)
IMAGE_SHA256 = "08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59"
MODEL = (
    "/lustre/fsw/portfolios/coreai/users/williamj/models/"
    "oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean"
)
METRICS = re.compile(r'"(?:GET|POST|HEAD|PUT|OPTIONS) /metrics(?:[? ]|$)', re.I)
REFERENCE_LABELS = {
    "ll-nomtp-v1": ("ll-nomtp",),
    "ll-mtp3-v1": ("ll-mtp3",),
    "ht-nomtp-noeplb-v1": ("ht-nomtp-noeplb",),
    "ht-nomtp-static528-v1": ("ht-nomtp-static528",),
    "ht-mtp3-noeplb-v1": ("ht-mtp3-noeplb-main", "ht-mtp3-noeplb-tail"),
    "ht-mtp3-static528-v1": ("ht-mtp3-static528",),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "aggregate-v1.tsv",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=EXPERIMENT / "reference-evidence" / "trtllm5-aggregate-audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "reports" / "aggregate-v1-audit.json",
    )
    parser.add_argument("--parity-percent", type=float, default=5.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slurm(job: str, expected_gpus: int) -> dict[str, str]:
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
    if row[1:3] != ["COMPLETED", "0:0"]:
        raise ValueError(f"job {job} is not COMPLETED 0:0: {row}")
    if f"gres/gpu={expected_gpus}" not in row[3]:
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


def _load_reference_points(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw["status"] != "pass":
        raise ValueError("TRTLLM5 reference audit did not pass")
    return {
        reference["label"]: {
            int(point["concurrency"]): point for point in reference["points"]
        }
        for reference in raw["references"]
    }


def _audit_logs(root: Path) -> int:
    hits = 0
    for path in set(root.rglob("*.log")) | set(root.rglob("*.out")):
        hits += len(METRICS.findall(path.read_text(encoding="utf-8", errors="replace")))
    if hits:
        raise ValueError(f"found {hits} forbidden /metrics requests under {root}")
    return hits


def _audit_result(path: Path, concurrency: int, multiplier: int, gpus: int) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    planned = concurrency * multiplier
    if int(raw["max_concurrency"]) != concurrency:
        raise ValueError(f"{path}: embedded concurrency mismatch")
    if int(raw["num_prompts"]) != planned or int(raw["completed"]) != planned:
        raise ValueError(f"{path}: planned/completed request mismatch")
    input_lens = raw.get("input_lens") or []
    output_lens = raw.get("output_lens") or []
    errors = raw.get("errors") or []
    generated = raw.get("generated_texts") or []
    if len(input_lens) != planned or set(input_lens) != {8192}:
        raise ValueError(f"{path}: exact input-length invariant failed")
    if len(output_lens) != planned or set(output_lens) != {1024}:
        raise ValueError(f"{path}: exact output-length invariant failed")
    if len(errors) != planned or any(bool(value) for value in errors):
        raise ValueError(f"{path}: request error invariant failed")
    if len(generated) != planned:
        raise ValueError(f"{path}: decoded-text count mismatch")
    input_tokens = planned * 8192
    output_tokens = planned * 1024
    if int(raw["total_input_tokens"]) != input_tokens:
        raise ValueError(f"{path}: total input tokens mismatch")
    if int(raw["total_output_tokens"]) != output_tokens:
        raise ValueError(f"{path}: total output tokens mismatch")
    calculated = (input_tokens + output_tokens) / float(raw["duration"]) / gpus
    reported = float(raw["total_token_throughput"]) / gpus
    if not math.isclose(calculated, reported, rel_tol=1e-10):
        raise ValueError(f"{path}: throughput arithmetic mismatch")
    return {
        "concurrency": concurrency,
        "planned_requests": planned,
        "completed_requests": int(raw["completed"]),
        "empty_decoded_texts": sum(not text for text in generated),
        "total_tps_per_gpu": reported,
        "median_ttft_ms": float(raw["median_ttft_ms"]),
        "median_tpot_ms": float(raw["median_tpot_ms"]),
        "duration_seconds": float(raw["duration"]),
        "path": str(path),
        "sha256": _sha256(path),
    }


def _reference_comparisons(
    arm: str,
    points: list[dict[str, Any]],
    references: dict[str, dict[int, dict[str, Any]]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    current = {int(point["concurrency"]): point for point in points}
    comparisons = []
    failures = []
    for label in REFERENCE_LABELS[arm]:
        for concurrency, reference in references[label].items():
            point = current.get(concurrency)
            if point is None:
                raise ValueError(f"{arm}: missing reference concurrency {concurrency}")
            delta = 100.0 * (
                float(point["total_tps_per_gpu"]) / float(reference["total_tps_per_gpu"]) - 1.0
            )
            status = "pass" if abs(delta) <= threshold else "retest"
            if status != "pass":
                failures.append(
                    f"{arm} C{concurrency} differs from {label} by {delta:+.3f}%"
                )
            comparisons.append(
                {
                    "reference_label": label,
                    "concurrency": concurrency,
                    "current_total_tps_per_gpu": point["total_tps_per_gpu"],
                    "reference_total_tps_per_gpu": reference["total_tps_per_gpu"],
                    "change_percent": delta,
                    "threshold_percent": threshold,
                    "status": status,
                }
            )
    return comparisons, failures


def _audit_arm(
    row: dict[str, str],
    references: dict[str, dict[int, dict[str, Any]]],
    threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    arm = row["arm"]
    job = row["job_id"]
    recipe = Path(row["config"])
    nodes = int(row["nodes"])
    gpus = nodes * 4
    root = EXPERIMENT / "outputs" / arm / job
    runtime_config = root / "config.yaml"
    if _sha256(runtime_config) != _sha256(recipe):
        raise ValueError(f"{arm}: runtime config differs from recipe")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    if config["model"]["path"] != MODEL or config["model"]["container"] != IMAGE:
        raise ValueError(f"{arm}: model or image mismatch")
    serving = config["backend"]["trtllm_config"]["aggregated"]
    if serving["kv_cache_config"]["use_kv_cache_manager_v2"] is not False:
        raise ValueError(f"{arm}: authoritative V1 arm did not use KV manager V1")
    for key in ("return_perf_metrics", "enable_iter_perf_stats", "enable_iter_req_stats", "print_iter_log"):
        if serving[key] is not False:
            raise ValueError(f"{arm}: {key} was not disabled")
    if any("TensorRT-LLM" in mount and ":/eplb.yaml" not in mount for mount in config.get("extra_mount", [])):
        raise ValueError(f"{arm}: unexpected TensorRT-LLM source overlay")
    benchmark = config["benchmark"]
    expected = [int(value) for value in benchmark["concurrencies"]]
    multiplier = int(benchmark["num_prompts_mult"])
    points = []
    for concurrency in expected:
        path = (
            root
            / "logs"
            / "sa-bench_isl_8192_osl_1024"
            / f"results_concurrency_{concurrency}_gpus_{gpus}.json"
        )
        points.append(_audit_result(path, concurrency, multiplier, gpus))
    actual_paths = list(root.glob("logs/sa-bench_*/results_concurrency_*_gpus_*.json"))
    if len(actual_paths) != len(expected):
        raise ValueError(f"{arm}: result-file count mismatch")
    heartbeat = (root / "heartbeat.final-status.txt").read_text(encoding="utf-8")
    for marker in (
        "state=stopped",
        "supervisor_alive=no",
        f"total_tasks={gpus}",
        "stop_reason=automatic model-loading boundary matched",
    ):
        if marker not in heartbeat:
            raise ValueError(f"{arm}: heartbeat missing {marker!r}")
    formal = json.loads((root / "audit-formal-window.json").read_text(encoding="utf-8"))
    if formal["status"] != "pass":
        raise ValueError(f"{arm}: formal-window audit failed")
    comparisons, failures = _reference_comparisons(arm, points, references, threshold)
    return (
        {
            "status": "pass" if not failures else "retest",
            "arm": arm,
            "job": job,
            "slurm": _slurm(job, gpus),
            "recipe": str(recipe),
            "recipe_sha256": _sha256(recipe),
            "image_sha256": IMAGE_SHA256,
            "metrics_hits": _audit_logs(root),
            "points": points,
            "reference_comparisons": comparisons,
        },
        failures,
    )


def main() -> None:
    args = _parse_args()
    references = _load_reference_points(args.references)
    with args.submissions.open(encoding="utf-8", newline="") as submissions:
        rows = list(csv.DictReader(submissions, delimiter="\t"))
    if tuple(row["arm"] for row in rows) != tuple(REFERENCE_LABELS):
        raise ValueError("submission ledger arms/order differ from frozen V1 matrix")
    audited = []
    failures = []
    for row in rows:
        arm, arm_failures = _audit_arm(row, references, args.parity_percent)
        audited.append(arm)
        failures.extend(arm_failures)
    report = {
        "status": "pass" if not failures else "retest",
        "parity_threshold_percent": args.parity_percent,
        "failures": failures,
        "arms": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
