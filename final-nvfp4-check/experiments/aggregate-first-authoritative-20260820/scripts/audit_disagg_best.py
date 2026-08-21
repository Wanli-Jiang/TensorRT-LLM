#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit current-image no-EPLB disaggregated results against TRTLLM5 anchors."""

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
MODEL = Path(
    "/lustre/fsw/portfolios/coreai/users/williamj/models/"
    "oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean"
)
IMAGE = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh"
)
METRICS = re.compile(r'"(?:GET|POST|HEAD|PUT|OPTIONS) /metrics(?:[? ]|$)')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "disaggregate-best.tsv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            EXPERIMENT
            / "recipes"
            / "disaggregate"
            / "best-noeplb-manifest.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "reports" / "disaggregate-best-audit.json",
    )
    parser.add_argument("--parity-percent", type=float, default=5.0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_ledger(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    result = {row["mode"]: row for row in rows}
    if set(result) != {"nomtp", "mtp3"} or len(rows) != 2:
        raise ValueError("submission ledger must contain exactly nomtp and mtp3")
    return result


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
    if row is None or row[1:3] != ["COMPLETED", "0:0"]:
        raise ValueError(f"job {job} did not complete successfully: {row}")
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


def _audit_no_metrics(root: Path) -> int:
    hits = 0
    for path in set(root.rglob("*.log")) | set(root.rglob("*.out")):
        hits += len(METRICS.findall(path.read_text(encoding="utf-8", errors="replace")))
    if hits:
        raise ValueError(f"found {hits} forbidden /metrics requests under {root}")
    return hits


def _audit_heartbeat(root: Path, expected_gpus: int) -> dict[str, str]:
    path = root / "loading-heartbeat" / "final-status.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    status = dict(line.split("=", 1) for line in lines if "=" in line)
    expected = {
        "state": "stopped",
        "supervisor_alive": "no",
        "total_tasks": str(expected_gpus),
        "stop_reason": "automatic model-loading boundary matched",
    }
    for key, value in expected.items():
        if status.get(key) != value:
            raise ValueError(f"heartbeat {key} mismatch: {status.get(key)!r}")
    if not (root / "loading-heartbeat" / "boundary.match").is_file():
        raise ValueError("heartbeat boundary evidence is missing")
    return {key: status[key] for key in expected}


def _audit_generated_configs(root: Path) -> None:
    for role in ("ctx", "gen"):
        config = yaml.safe_load((root / f"{role}_config.yaml").read_text(encoding="utf-8"))
        for key in (
            "return_perf_metrics",
            "enable_iter_perf_stats",
            "enable_iter_req_stats",
            "print_iter_log",
        ):
            if config.get(key) is not False:
                raise ValueError(f"{role}_config.yaml did not disable {key}")


def _audit_result(
    root: Path,
    concurrency: int,
    rounds: int,
    deployed_gpus: int,
    dataset: Path,
    client_oversample_expected: bool,
) -> dict[str, Any]:
    path = root / f"concurrency_{concurrency}" / "result.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    planned = concurrency * rounds
    dataset_records = sum(
        bool(line.strip()) for line in dataset.read_text(encoding="utf-8").splitlines()
    )
    expected_completed = (
        planned if client_oversample_expected else min(planned, dataset_records)
    )
    expected_input = expected_completed * 8192
    expected_output = expected_completed * 1024
    if int(raw["max_concurrency"]) != concurrency:
        raise ValueError(f"{path}: concurrency mismatch")
    if (
        int(raw["num_prompts"]) != planned
        or int(raw["completed"]) != expected_completed
    ):
        raise ValueError(f"{path}: planned/completed request mismatch")
    if int(raw["total_input_tokens"]) != expected_input:
        raise ValueError(f"{path}: exact input-token invariant failed")
    if int(raw["total_output_tokens"]) != expected_output:
        raise ValueError(f"{path}: exact output-token invariant failed")
    calculated = (expected_input + expected_output) / float(raw["duration"])
    reported = float(raw["total_token_throughput"])
    if not math.isclose(calculated, reported, rel_tol=1e-10):
        raise ValueError(f"{path}: throughput arithmetic mismatch")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "concurrency": concurrency,
        "rounds": rounds,
        "planned_requests": planned,
        "dataset_records": dataset_records,
        "client_oversample_expected": client_oversample_expected,
        "expected_completed_requests": expected_completed,
        "completed_requests": int(raw["completed"]),
        "duration_seconds": float(raw["duration"]),
        "total_tps": reported,
        "total_tps_per_gpu": reported / deployed_gpus,
        "median_ttft_ms": float(raw["median_ttft_ms"]),
        "median_tpot_ms": float(raw["median_tpot_ms"]),
        "mean_accepted_length": raw.get("mean_avg_decoded_tokens_per_iter"),
    }


def _audit_arm(
    mode: str,
    row: dict[str, str],
    reference: dict[str, Any],
    parity_percent: float,
) -> dict[str, Any]:
    recipe = Path(row["config"])
    root = Path(row["output"])
    job = row["job_id"]
    deployed_gpus = int(row["deployed_gpus"])
    if recipe != Path(reference["path"]) or _sha256(recipe) != reference["sha256"]:
        raise ValueError(f"{mode}: submitted recipe differs from frozen manifest")
    if deployed_gpus != int(reference["deployed_gpus"]):
        raise ValueError(f"{mode}: deployed GPU count differs from manifest")
    config = yaml.safe_load(recipe.read_text(encoding="utf-8"))
    environment = config["environment"]
    if Path(environment["model_path"]) != MODEL or Path(environment["container_image"]) != IMAGE:
        raise ValueError(f"{mode}: model or image mismatch")
    if environment["trtllm_repo"] != "none":
        raise ValueError(f"{mode}: source overlay is enabled")
    for client_path, expected_sha256 in reference.get(
        "client_file_sha256", {}
    ).items():
        path = Path(client_path)
        if _sha256(path) != expected_sha256:
            raise ValueError(f"{mode}: client helper changed after manifest freeze: {path}")
    _audit_generated_configs(root)
    result = _audit_result(
        root,
        int(reference["concurrency"]),
        int(reference["rounds"]),
        deployed_gpus,
        Path(reference["dataset"]),
        bool(reference.get("client_oversample_expected", False)),
    )
    if reference.get("client_oversample_expected", False):
        marker = (
            f"NVFP4_CLIENT_OVERSAMPLE before=4096 "
            f"requested={result['planned_requests']} "
            f"after={result['planned_requests']}"
        )
        benchmark_log = (root / "6_bench.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if marker not in benchmark_log:
            raise ValueError(f"{mode}: client oversampling marker is missing")
    reference_tps = float(reference["reference_total_tps_per_gpu"])
    change = 100.0 * (result["total_tps_per_gpu"] / reference_tps - 1.0)
    done = root / f"8_done_{job}.txt"
    if not done.is_file():
        raise ValueError(f"{mode}: completion marker is missing")
    return {
        "mode": mode,
        "job": job,
        "output": str(root),
        "reference_job": str(reference["reference_job"]),
        "reference_total_tps_per_gpu": reference_tps,
        "change_percent": change,
        "parity_threshold_percent": parity_percent,
        "performance_status": "pass" if abs(change) <= parity_percent else "retest",
        "no_metrics_requests": _audit_no_metrics(root),
        "heartbeat": _audit_heartbeat(root, deployed_gpus),
        "slurm": _slurm(job, deployed_gpus),
        "result": result,
    }


def main() -> None:
    args = _parse_args()
    rows = _read_ledger(args.submissions)
    manifest_rows = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest = {row["mode"]: row for row in manifest_rows}
    if set(manifest) != {"nomtp", "mtp3"}:
        raise ValueError("frozen manifest must contain exactly nomtp and mtp3")
    arms = [
        _audit_arm(mode, rows[mode], manifest[mode], args.parity_percent)
        for mode in ("nomtp", "mtp3")
    ]
    report = {
        "integrity_status": "pass",
        "performance_status": (
            "pass"
            if all(arm["performance_status"] == "pass" for arm in arms)
            else "retest"
        ),
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
