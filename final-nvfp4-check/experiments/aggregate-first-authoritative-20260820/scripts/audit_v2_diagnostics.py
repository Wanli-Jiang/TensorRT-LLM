#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit KV manager V2 diagnostics against the accepted V1 aggregate runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml
from audit_current_aggregate import _audit_logs, _audit_result, _sha256, _slurm

EXPERIMENT = Path(__file__).resolve().parents[1]
V2_ARMS = {
    "ll-nomtp-v2": ("ll-nomtp-v1", "latency", 8),
    "ll-mtp3-v2": ("ll-mtp3-v1", "latency", 8),
    "ht-nomtp-noeplb-v2": ("ht-nomtp-noeplb-tail-v1", "throughput", 16),
    "ht-mtp3-noeplb-v2": ("ht-mtp3-noeplb-v1", "throughput", 16),
}
V1_CAPACITY = re.compile(r"Allocated [0-9.]+ GiB for max tokens in paged KV cache \(([0-9]+)\)")
V2_CAP_ENFORCEMENT = re.compile(
    r"max_tokens ([0-9]+) is provided\. Allowed quota from max_tokens is "
    r"([0-9.]+)GiB\. New quota is ([0-9.]+)GiB"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "aggregate-v1.tsv",
    )
    parser.add_argument(
        "--v2-submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "aggregate-v2-diagnostics.tsv",
    )
    parser.add_argument(
        "--v1-tail-submissions",
        type=Path,
        default=EXPERIMENT / "submissions" / "aggregate-v1-nomtp-tail.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT / "reports" / "aggregate-kv-manager-v2-diagnostics.json",
    )
    return parser.parse_args()


def _read_submissions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    return {row["arm"]: row for row in rows}


def _server_log(root: Path) -> tuple[Path, str]:
    logs = list((root / "logs").glob("*_agg_w0.out"))
    if len(logs) != 1:
        raise ValueError(f"{root}: expected one aggregate server log, found {logs}")
    return logs[0], logs[0].read_text(encoding="utf-8", errors="replace")


def _v1_capacity(root: Path) -> int:
    _, server_text = _server_log(root)
    values = [int(value) for value in V1_CAPACITY.findall(server_text)]
    if not values:
        raise ValueError(f"{root}: no V1 KV token-capacity evidence")
    # The server first creates a small profiling/warmup pool. MTP can then
    # create separate target and draft pools; request admission is constrained
    # by the smaller final pool, not the larger and cheaper draft-model pool.
    final_pool_values = [value for value in values if value >= 100_000]
    if not final_pool_values:
        raise ValueError(f"{root}: no V1 formal KV token-capacity evidence")
    return min(final_pool_values)


def _result_path(root: Path, concurrency: int, gpus: int) -> Path:
    return (
        root
        / "logs"
        / "sa-bench_isl_8192_osl_1024"
        / f"results_concurrency_{concurrency}_gpus_{gpus}.json"
    )


def _performance_comparison(
    objective: str,
    v1: dict[str, Any],
    v2: dict[str, Any],
) -> dict[str, Any]:
    throughput_delta = 100.0 * (
        float(v2["total_tps_per_gpu"]) / float(v1["total_tps_per_gpu"]) - 1.0
    )
    ttft_delta = 100.0 * (float(v2["median_ttft_ms"]) / float(v1["median_ttft_ms"]) - 1.0)
    tpot_delta = 100.0 * (float(v2["median_tpot_ms"]) / float(v1["median_tpot_ms"]) - 1.0)
    if objective == "throughput":
        regression = throughput_delta < -5.0
    else:
        regression = int(v2["concurrency"]) == 1 and (ttft_delta > 10.0 or tpot_delta > 5.0)
    return {
        "concurrency": v2["concurrency"],
        "v1_total_tps_per_gpu": v1["total_tps_per_gpu"],
        "v2_total_tps_per_gpu": v2["total_tps_per_gpu"],
        "total_tps_change_percent": throughput_delta,
        "v1_median_ttft_ms": v1["median_ttft_ms"],
        "v2_median_ttft_ms": v2["median_ttft_ms"],
        "median_ttft_change_percent": ttft_delta,
        "v1_median_tpot_ms": v1["median_tpot_ms"],
        "v2_median_tpot_ms": v2["median_tpot_ms"],
        "median_tpot_change_percent": tpot_delta,
        "regression": regression,
    }


def _audit_arm(
    v1_row: dict[str, str],
    v2_row: dict[str, str],
    objective: str,
    gpus: int,
) -> dict[str, Any]:
    v1_root = EXPERIMENT / "outputs" / v1_row["arm"] / v1_row["job_id"]
    v2_root = EXPERIMENT / "outputs" / v2_row["arm"] / v2_row["job_id"]
    v2_recipe = Path(v2_row["config"])
    runtime_config = v2_root / "config.yaml"
    if _sha256(v2_recipe) != _sha256(runtime_config):
        raise ValueError(f"{v2_row['arm']}: runtime config differs from recipe")
    config = yaml.safe_load(v2_recipe.read_text(encoding="utf-8"))
    serving = config["backend"]["trtllm_config"]["aggregated"]
    kv_cache = serving["kv_cache_config"]
    v1_capacity = _v1_capacity(v1_root)
    if kv_cache.get("use_kv_cache_manager_v2") is not True:
        raise ValueError(f"{v2_row['arm']}: V2 manager is not enabled")
    if int(kv_cache.get("avg_seq_len", 0)) != 9216:
        raise ValueError(f"{v2_row['arm']}: avg_seq_len is not 9216")
    if int(kv_cache.get("max_tokens", 0)) != v1_capacity:
        raise ValueError(
            f"{v2_row['arm']}: max_tokens {kv_cache.get('max_tokens')} "
            f"does not match V1 capacity {v1_capacity}"
        )
    for key in (
        "return_perf_metrics",
        "enable_iter_perf_stats",
        "enable_iter_req_stats",
        "print_iter_log",
    ):
        if serving[key] is not False:
            raise ValueError(f"{v2_row['arm']}: {key} was not disabled")

    _, server_text = _server_log(v2_root)
    managers = set(re.findall(r"Selected hybrid KV cache manager: ([A-Za-z0-9_]+)", server_text))
    if managers != {"MambaHybridCacheManagerV2"}:
        raise ValueError(f"{v2_row['arm']}: unexpected managers {managers}")
    if "KVCacheV2Scheduler:" not in server_text:
        raise ValueError(f"{v2_row['arm']}: KVCacheV2Scheduler evidence is missing")
    requested_scheduler_policy = serving.get("scheduler_config", {}).get(
        "capacity_scheduler_policy", "GUARANTEED_NO_EVICT"
    )
    if requested_scheduler_policy != "GUARANTEED_NO_EVICT":
        raise ValueError(
            f"{v2_row['arm']}: unexpected requested scheduler policy "
            f"{requested_scheduler_policy}"
        )
    scheduler_override_marker = (
        "KVCacheV2Scheduler only supports MAX_UTILIZATION for now, got "
        "GUARANTEED_NO_EVICT, setting to MAX_UTILIZATION"
    )
    scheduler_override_evidence = server_text.count(scheduler_override_marker)
    if scheduler_override_evidence < gpus:
        raise ValueError(
            f"{v2_row['arm']}: expected at least {gpus} scheduler override records, "
            f"found {scheduler_override_evidence}"
        )
    if f"max_tokens {v1_capacity} is provided" not in server_text:
        raise ValueError(f"{v2_row['arm']}: runtime did not acknowledge the V1 token cap")
    all_quota_evidence = [
        (int(max_tokens), float(allowed_quota), float(effective_quota))
        for max_tokens, allowed_quota, effective_quota in V2_CAP_ENFORCEMENT.findall(server_text)
    ]
    target_allowed_quota_gib = max(
        (
            allowed_quota
            for max_tokens, allowed_quota, _ in all_quota_evidence
            if max_tokens == v1_capacity
        ),
        default=None,
    )
    if target_allowed_quota_gib is None:
        raise ValueError(f"{v2_row['arm']}: V2 capacity-enforcement evidence is missing")
    # MTP without attention DP can build both target and draft managers. They
    # receive the same max_tokens value but have very different byte costs;
    # select the largest allowed quota, which is the serving target manager.
    quota_evidence = [
        (allowed_quota, effective_quota)
        for max_tokens, allowed_quota, effective_quota in all_quota_evidence
        if max_tokens == v1_capacity
        and math.isclose(allowed_quota, target_allowed_quota_gib, rel_tol=1e-9)
    ]
    quota_ratios = [
        effective_quota / allowed_quota
        for allowed_quota, effective_quota in quota_evidence
    ]
    minimum_effective_quota_gib = min(
        effective_quota for _, effective_quota in quota_evidence
    )
    calibration_by_tokens: dict[int, float] = {}
    for max_tokens, allowed_quota, _ in all_quota_evidence:
        if max_tokens != v1_capacity:
            calibration_by_tokens[max_tokens] = max(
                calibration_by_tokens.get(max_tokens, 0.0), allowed_quota
            )
    calibration_candidates = [
        (max_tokens, allowed_quota)
        for max_tokens, allowed_quota in calibration_by_tokens.items()
        if max_tokens < v1_capacity and allowed_quota < target_allowed_quota_gib
    ]
    if not calibration_candidates:
        raise ValueError(f"{v2_row['arm']}: no target-manager calibration quota")
    calibration_tokens, calibration_quota_gib = max(calibration_candidates)
    effective_token_capacity = math.floor(
        calibration_tokens
        + (minimum_effective_quota_gib - calibration_quota_gib)
        / (target_allowed_quota_gib - calibration_quota_gib)
        * (v1_capacity - calibration_tokens)
    )
    if effective_token_capacity <= 0:
        raise ValueError(f"{v2_row['arm']}: invalid effective V2 token capacity")

    benchmark = config["benchmark"]
    multiplier = int(benchmark["num_prompts_mult"])
    comparisons = []
    adp_size = int(serving["tensor_parallel_size"]) if serving["enable_attention_dp"] else 1
    for concurrency_value in benchmark["concurrencies"]:
        concurrency = int(concurrency_value)
        v1_result = _audit_result(
            _result_path(v1_root, concurrency, gpus),
            concurrency,
            multiplier,
            gpus,
        )
        v2_result = _audit_result(
            _result_path(v2_root, concurrency, gpus),
            concurrency,
            multiplier,
            gpus,
        )
        comparison = _performance_comparison(objective, v1_result, v2_result)
        requests_per_adp_rank = math.ceil(concurrency / adp_size)
        required_resident_tokens = requests_per_adp_rank * 9216
        v1_resident_sequences = v1_capacity // 9216
        v2_resident_sequences = effective_token_capacity // 9216
        comparison.update(
            {
                "requests_per_adp_rank": requests_per_adp_rank,
                "required_resident_tokens": required_resident_tokens,
                "v1_full_residency": required_resident_tokens <= v1_capacity,
                "v2_full_residency": (
                    required_resident_tokens <= effective_token_capacity
                ),
                "capacity_confounded": (
                    requests_per_adp_rank
                    > min(v1_resident_sequences, v2_resident_sequences)
                    and v1_resident_sequences != v2_resident_sequences
                ),
            }
        )
        comparisons.append(comparison)

    heartbeat = (v2_root / "heartbeat.final-status.txt").read_text(encoding="utf-8")
    for marker in (
        "state=stopped",
        "supervisor_alive=no",
        f"total_tasks={gpus}",
        "stop_reason=automatic model-loading boundary matched",
    ):
        if marker not in heartbeat:
            raise ValueError(f"{v2_row['arm']}: heartbeat missing {marker!r}")
    formal = json.loads((v2_root / "audit-formal-window.json").read_text(encoding="utf-8"))
    if formal["status"] != "pass":
        raise ValueError(f"{v2_row['arm']}: formal-window audit failed")
    _audit_logs(v2_root)
    _slurm(v2_row["job_id"], gpus)
    return {
        "arm": v2_row["arm"],
        "job": v2_row["job_id"],
        "matched_v1_arm": v1_row["arm"],
        "matched_v1_job": v1_row["job_id"],
        "objective": objective,
        "requested_v1_token_capacity": v1_capacity,
        "effective_v2_token_capacity": effective_token_capacity,
        "v1_resident_sequences_at_avg_9216": v1_capacity // 9216,
        "v2_resident_sequences_at_avg_9216": effective_token_capacity // 9216,
        "capacity_match_status": (
            "exact"
            if effective_token_capacity == v1_capacity
            else "v2-device-memory-limited"
        ),
        "capacity_enforcement_records": len(quota_evidence),
        "minimum_effective_to_requested_quota_ratio": min(quota_ratios),
        "target_allowed_quota_gib": target_allowed_quota_gib,
        "minimum_effective_quota_gib": minimum_effective_quota_gib,
        "calibration_tokens": calibration_tokens,
        "calibration_allowed_quota_gib": calibration_quota_gib,
        "manager": "MambaHybridCacheManagerV2",
        "scheduler": "KVCacheV2Scheduler",
        "requested_scheduler_policy": requested_scheduler_policy,
        "effective_scheduler_policy": "MAX_UTILIZATION",
        "scheduler_policy_forced_by_v2": True,
        "scheduler_override_evidence_records": scheduler_override_evidence,
        "performance_status": (
            "regression" if any(point["regression"] for point in comparisons) else "no-regression"
        ),
        "comparisons": comparisons,
    }


def main() -> None:
    args = _parse_args()
    v1_rows = _read_submissions(args.v1_submissions)
    v1_rows.update(_read_submissions(args.v1_tail_submissions))
    v2_rows = _read_submissions(args.v2_submissions)
    if tuple(v2_rows) != tuple(V2_ARMS):
        raise ValueError("V2 submission ledger differs from the frozen diagnostic matrix")
    arms = []
    for v2_arm, (v1_arm, objective, gpus) in V2_ARMS.items():
        arms.append(_audit_arm(v1_rows[v1_arm], v2_rows[v2_arm], objective, gpus))
    report = {
        "integrity_status": "pass",
        "performance_status": (
            "regression"
            if any(arm["performance_status"] == "regression" for arm in arms)
            else "no-regression"
        ),
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
