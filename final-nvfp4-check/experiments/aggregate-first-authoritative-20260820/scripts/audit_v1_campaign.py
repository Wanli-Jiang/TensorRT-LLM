#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit and merge the authoritative aggregate KV-manager-V1 campaign."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audit_current_aggregate import _audit_logs, _audit_result, _sha256, _slurm

EXPERIMENT = Path(__file__).resolve().parents[1]
REFERENCE_REPORT = (
    EXPERIMENT / "reference-evidence" / "trtllm5-aggregate-audit.json"
)
OUTPUT = EXPERIMENT / "reports" / "aggregate-v1-campaign-audit.json"
PARITY_PERCENT = 5.0


@dataclass(frozen=True)
class Source:
    arm: str
    job: str
    recipe: str
    gpus: int
    multiplier: int
    expected_state: str
    formal_report: str

    @property
    def root(self) -> Path:
        return EXPERIMENT / "outputs" / self.arm / self.job

    @property
    def recipe_path(self) -> Path:
        return EXPERIMENT / "recipes" / self.recipe


SOURCES = {
    "ll-nomtp": Source(
        "ll-nomtp-v1",
        "525297",
        "aggregate-ll-nomtp-tp8-trtllm.yaml",
        8,
        5,
        "COMPLETED",
        "audit-formal-window.json",
    ),
    "ll-mtp3": Source(
        "ll-mtp3-v1",
        "525298",
        "aggregate-ll-mtp3-tp8-trtllm.yaml",
        8,
        5,
        "COMPLETED",
        "audit-formal-window.json",
    ),
    "ht-nomtp-noeplb-main": Source(
        "ht-nomtp-noeplb-v1",
        "525299",
        "aggregate-ht-nomtp-adp16-cutedsl-noeplb.yaml",
        16,
        3,
        "TIMEOUT",
        "audit-formal-window-partial.json",
    ),
    "ht-nomtp-noeplb-tail": Source(
        "ht-nomtp-noeplb-tail-v1",
        "525325",
        "aggregate-ht-nomtp-adp16-cutedsl-noeplb-tail.yaml",
        16,
        3,
        "COMPLETED",
        "audit-formal-window.json",
    ),
    "ht-nomtp-static-main": Source(
        "ht-nomtp-static528-v1",
        "525300",
        "aggregate-ht-nomtp-adp16-cutedsl-static528.yaml",
        16,
        3,
        "TIMEOUT",
        "audit-formal-window-partial.json",
    ),
    "ht-nomtp-static-tail": Source(
        "ht-nomtp-static528-tail-v1",
        "525326",
        "aggregate-ht-nomtp-adp16-cutedsl-static528-tail.yaml",
        16,
        3,
        "COMPLETED",
        "audit-formal-window.json",
    ),
    "ht-mtp3-noeplb": Source(
        "ht-mtp3-noeplb-v1",
        "525301",
        "aggregate-ht-mtp3-adp16-cutedsl-noeplb.yaml",
        16,
        3,
        "COMPLETED",
        "audit-formal-window.json",
    ),
    "ht-mtp3-static": Source(
        "ht-mtp3-static528-v1",
        "525302",
        "aggregate-ht-mtp3-adp16-cutedsl-static528.yaml",
        16,
        3,
        "COMPLETED",
        "audit-formal-window.json",
    ),
}

LL_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
HT_NOMTP_CONCURRENCIES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    768,
    1024,
    1280,
    1536,
    1792,
    2048,
    2304,
)
HT_MTP3_CONCURRENCIES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
)


def _slurm_source(source: Source) -> dict[str, str]:
    if source.expected_state == "COMPLETED":
        return _slurm(source.job, source.gpus)
    proc = subprocess.run(
        [
            "sacct",
            "-X",
            "-n",
            "-P",
            "-j",
            source.job,
            "--format=JobIDRaw,State,ExitCode,AllocTRES,NodeList",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split("|") for line in proc.stdout.splitlines() if line]
    row = next((item for item in rows if item[0] == source.job), None)
    if row is None or row[1] != "TIMEOUT" or row[2] != "0:0":
        raise ValueError(f"{source.job}: expected controlled TIMEOUT 0:0, got {rows}")
    if f"gres/gpu={source.gpus}" not in row[3]:
        raise ValueError(f"{source.job}: GPU allocation mismatch")
    domains = set(re.findall(r"nvl72d[0-9]+", row[4]))
    if len(domains) != 1:
        raise ValueError(f"{source.job}: cross-domain allocation {row[4]}")
    return {
        "state": row[1],
        "exit_code": row[2],
        "allocation": row[3],
        "nodes": row[4],
        "domain": next(iter(domains)),
        "acceptance_scope": "completed result files only; later planned points excluded",
    }


def _audit_source(source: Source) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    runtime_config = source.root / "config.yaml"
    if _sha256(runtime_config) != _sha256(source.recipe_path):
        raise ValueError(f"{source.arm}: runtime config differs from recipe")
    result_paths = sorted(
        source.root.glob(
            f"logs/sa-bench_isl_8192_osl_1024/"
            f"results_concurrency_*_gpus_{source.gpus}.json"
        )
    )
    if not result_paths:
        raise ValueError(f"{source.arm}: no result files")
    points = {}
    for path in result_paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        concurrency = int(raw["max_concurrency"])
        points[concurrency] = _audit_result(
            path,
            concurrency,
            source.multiplier,
            source.gpus,
        )
    formal = json.loads(
        (source.root / source.formal_report).read_text(encoding="utf-8")
    )
    if formal["status"] != "pass":
        raise ValueError(f"{source.arm}: formal-window audit failed")
    windows = formal.get("windows", [])
    if len(windows) != len(points):
        raise ValueError(f"{source.arm}: formal-window/result count mismatch")
    return (
        {
            "arm": source.arm,
            "job": source.job,
            "slurm": _slurm_source(source),
            "recipe": str(source.recipe_path),
            "recipe_sha256": _sha256(source.recipe_path),
            "formal_report": str(source.root / source.formal_report),
            "formal_window_count": len(windows),
            "metrics_hits": _audit_logs(source.root),
            "result_concurrencies": sorted(points),
        },
        points,
    )


def _load_references() -> dict[str, dict[int, dict[str, Any]]]:
    report = json.loads(REFERENCE_REPORT.read_text(encoding="utf-8"))
    if report["status"] != "pass":
        raise ValueError("TRTLLM5 reference audit did not pass")
    return {
        item["label"]: {
            int(point["concurrency"]): point for point in item["points"]
        }
        for item in report["references"]
    }


def _merge(
    concurrencies: tuple[int, ...],
    sources: list[tuple[dict[int, dict[str, Any]], set[int]]],
) -> dict[int, dict[str, Any]]:
    merged = {}
    for points, selected in sources:
        for concurrency in selected:
            if concurrency not in points:
                raise ValueError(f"missing selected concurrency {concurrency}")
            merged[concurrency] = points[concurrency]
    if tuple(sorted(merged)) != concurrencies:
        raise ValueError(f"merged concurrency mismatch: {tuple(sorted(merged))}")
    return merged


def _comparisons(
    points: dict[int, dict[str, Any]],
    reference_labels: tuple[str, ...],
    references: dict[str, dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for label in reference_labels:
        for concurrency, reference in references[label].items():
            current = points[concurrency]
            delta = 100.0 * (
                float(current["total_tps_per_gpu"])
                / float(reference["total_tps_per_gpu"])
                - 1.0
            )
            output.append(
                {
                    "reference_label": label,
                    "concurrency": concurrency,
                    "current_total_tps_per_gpu": current["total_tps_per_gpu"],
                    "reference_total_tps_per_gpu": reference["total_tps_per_gpu"],
                    "change_percent": delta,
                    "status": (
                        "regression"
                        if delta < -PARITY_PERCENT
                        else "improvement"
                        if delta > PARITY_PERCENT
                        else "parity"
                    ),
                }
            )
    return output


def _peak(points: dict[int, dict[str, Any]]) -> dict[str, Any]:
    point = max(points.values(), key=lambda item: float(item["total_tps_per_gpu"]))
    return {
        "concurrency": point["concurrency"],
        "total_tps_per_gpu": point["total_tps_per_gpu"],
    }


def main() -> None:
    source_reports = {}
    source_points = {}
    for label, source in SOURCES.items():
        source_reports[label], source_points[label] = _audit_source(source)

    curves = {
        "ll-nomtp": _merge(
            LL_CONCURRENCIES,
            [(source_points["ll-nomtp"], set(LL_CONCURRENCIES))],
        ),
        "ll-mtp3": _merge(
            LL_CONCURRENCIES,
            [(source_points["ll-mtp3"], set(LL_CONCURRENCIES))],
        ),
        "ht-nomtp-noeplb": _merge(
            HT_NOMTP_CONCURRENCIES,
            [
                (
                    source_points["ht-nomtp-noeplb-main"],
                    set(HT_NOMTP_CONCURRENCIES[:13]),
                ),
                (
                    source_points["ht-nomtp-noeplb-tail"],
                    set(HT_NOMTP_CONCURRENCIES[13:]),
                ),
            ],
        ),
        "ht-nomtp-static528": _merge(
            HT_NOMTP_CONCURRENCIES,
            [
                (
                    source_points["ht-nomtp-static-main"],
                    set(HT_NOMTP_CONCURRENCIES[:13]),
                ),
                (
                    source_points["ht-nomtp-static-tail"],
                    set(HT_NOMTP_CONCURRENCIES[13:]),
                ),
            ],
        ),
        "ht-mtp3-noeplb": _merge(
            HT_MTP3_CONCURRENCIES,
            [(source_points["ht-mtp3-noeplb"], set(HT_MTP3_CONCURRENCIES))],
        ),
        "ht-mtp3-static528": _merge(
            HT_MTP3_CONCURRENCIES,
            [(source_points["ht-mtp3-static"], set(HT_MTP3_CONCURRENCIES))],
        ),
    }
    reference_labels = {
        "ll-nomtp": ("ll-nomtp",),
        "ll-mtp3": ("ll-mtp3",),
        "ht-nomtp-noeplb": ("ht-nomtp-noeplb",),
        "ht-nomtp-static528": ("ht-nomtp-static528",),
        "ht-mtp3-noeplb": (
            "ht-mtp3-noeplb-main",
            "ht-mtp3-noeplb-tail",
        ),
        "ht-mtp3-static528": ("ht-mtp3-static528",),
    }
    references = _load_references()
    curve_reports = {}
    statuses = []
    for label, points in curves.items():
        comparisons = _comparisons(
            points,
            reference_labels[label],
            references,
        )
        statuses.extend(item["status"] for item in comparisons)
        curve_reports[label] = {
            "points": [points[concurrency] for concurrency in sorted(points)],
            "peak": _peak(points),
            "reference_comparisons": comparisons,
        }

    repeated_points = []
    for prefix in ("ht-nomtp-noeplb", "ht-nomtp-static"):
        main_points = source_points[f"{prefix}-main"]
        tail_points = source_points[f"{prefix}-tail"]
        for concurrency in sorted(set(main_points) & set(tail_points)):
            main_tps = float(main_points[concurrency]["total_tps_per_gpu"])
            tail_tps = float(tail_points[concurrency]["total_tps_per_gpu"])
            repeated_points.append(
                {
                    "arm": prefix,
                    "concurrency": concurrency,
                    "main_total_tps_per_gpu": main_tps,
                    "tail_total_tps_per_gpu": tail_tps,
                    "tail_vs_main_percent": 100.0 * (tail_tps / main_tps - 1.0),
                }
            )

    nomtp_noeplb_peak = curve_reports["ht-nomtp-noeplb"]["peak"]
    nomtp_static_peak = curve_reports["ht-nomtp-static528"]["peak"]
    mtp_noeplb_peak = curve_reports["ht-mtp3-noeplb"]["peak"]
    mtp_static_peak = curve_reports["ht-mtp3-static528"]["peak"]
    report = {
        "integrity_status": "pass",
        "performance_status": (
            "review-localized-differences"
            if "regression" in statuses
            else "parity-or-improvement"
        ),
        "merge_policy": (
            "No-MTP C1-C1280 use completed formal windows from the main sweeps; "
            "C1536-C2304 use COMPLETED tail jobs. Main sweep TIMEOUT occurred only "
            "after the accepted files and is retained in source provenance."
        ),
        "sources": source_reports,
        "curves": curve_reports,
        "repeatability": repeated_points,
        "eplb_peak_effect": {
            "nomtp_static_vs_noeplb_percent": 100.0
            * (
                float(nomtp_static_peak["total_tps_per_gpu"])
                / float(nomtp_noeplb_peak["total_tps_per_gpu"])
                - 1.0
            ),
            "mtp3_static_vs_noeplb_percent": 100.0
            * (
                float(mtp_static_peak["total_tps_per_gpu"])
                / float(mtp_noeplb_peak["total_tps_per_gpu"])
                - 1.0
            ),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
