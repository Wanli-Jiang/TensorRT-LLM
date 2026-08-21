#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed request-integrity audit of the frozen TRTLLM5 references."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


T5 = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "williamj/TRTLLM5/oak-nvfp4-sweep/experiments"
)
OUTPUT = Path(__file__).resolve().parents[1] / "reference-evidence" / "trtllm5-aggregate-audit.json"


@dataclass(frozen=True)
class Reference:
    label: str
    job: str
    root: Path
    gpus: int
    prompt_multiplier: int
    expected_concurrencies: tuple[int, ...]


REFERENCES = (
    Reference(
        "ll-nomtp",
        "422737",
        T5
        / "oakhaven-max-nvfp4-gb300x8-8k1k-rr1-20260811/outputs/422737",
        8,
        5,
        (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    ),
    Reference(
        "ll-mtp3",
        "422739",
        T5
        / "oakhaven-max-nvfp4-gb300x8-8k1k-rr1-20260811/outputs/422739",
        8,
        5,
        (1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    ),
    Reference(
        "ht-nomtp-noeplb",
        "431407",
        T5
        / "oakhaven-max-nvfp4-adp16-no-eplb-20260812/outputs/"
        "nomtp-adp16-cutedsl-no-eplb-m128-v1/431407",
        16,
        3,
        (1536, 1792, 2048, 2304),
    ),
    Reference(
        "ht-nomtp-static528",
        "432389",
        T5
        / "oakhaven-max-nvfp4-adp16-static-eplb-20260812/outputs/"
        "nomtp-static528-v1/432389",
        16,
        3,
        (2048,),
    ),
    Reference(
        "ht-mtp3-noeplb-main",
        "430411",
        T5
        / "oakhaven-max-nvfp4-adp16-no-eplb-20260812/outputs/"
        "mtp3-adp16-cutedsl-no-eplb-m32-v1/430411",
        16,
        3,
        (256, 320, 384, 448, 512, 640),
    ),
    Reference(
        "ht-mtp3-noeplb-tail",
        "431024",
        T5
        / "oakhaven-max-nvfp4-adp16-no-eplb-20260812/outputs/"
        "mtp3-adp16-cutedsl-no-eplb-m32-tail-v1/431024",
        16,
        3,
        (768, 896, 1024),
    ),
    Reference(
        "ht-mtp3-static528",
        "431623",
        T5
        / "oakhaven-max-nvfp4-adp16-static-eplb-20260812/outputs/"
        "mtp3-static528-v1/431623",
        16,
        3,
        (640,),
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _slurm(job: str, gpus: int) -> dict[str, str]:
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
        raise ValueError(f"job {job} is not COMPLETED 0:0: {rows}")
    if f"gres/gpu={gpus}" not in row[3]:
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


def _result(reference: Reference, path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    concurrency = int(raw["max_concurrency"])
    planned = concurrency * reference.prompt_multiplier
    for field in ("num_prompts", "completed"):
        if int(raw[field]) != planned:
            raise ValueError(f"{reference.job} C{concurrency} {field} != {planned}")
    input_lens = raw.get("input_lens") or []
    output_lens = raw.get("output_lens") or []
    errors = raw.get("errors") or []
    generated = raw.get("generated_texts") or []
    if len(input_lens) != planned or set(input_lens) != {8192}:
        raise ValueError(f"{reference.job} C{concurrency} input-length failure")
    if len(output_lens) != planned or set(output_lens) != {1024}:
        raise ValueError(f"{reference.job} C{concurrency} output-length failure")
    if len(errors) != planned or any(bool(value) for value in errors):
        raise ValueError(f"{reference.job} C{concurrency} request errors")
    if len(generated) != planned:
        raise ValueError(f"{reference.job} C{concurrency} decoded-text count mismatch")
    expected_input = planned * 8192
    expected_output = planned * 1024
    if int(raw["total_input_tokens"]) != expected_input:
        raise ValueError(f"{reference.job} C{concurrency} input-token mismatch")
    if int(raw["total_output_tokens"]) != expected_output:
        raise ValueError(f"{reference.job} C{concurrency} output-token mismatch")
    calculated = (expected_input + expected_output) / float(raw["duration"]) / reference.gpus
    reported = float(raw["total_token_throughput"]) / reference.gpus
    if not math.isclose(calculated, reported, rel_tol=1e-10):
        raise ValueError(f"{reference.job} C{concurrency} throughput arithmetic mismatch")
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


def main() -> None:
    audited = []
    for reference in REFERENCES:
        paths = sorted(
            reference.root.glob(
                f"logs/sa-bench_isl_8192_osl_1024/"
                f"results_concurrency_*_gpus_{reference.gpus}.json"
            )
        )
        points = sorted((_result(reference, path) for path in paths), key=lambda item: item["concurrency"])
        actual = tuple(int(point["concurrency"]) for point in points)
        if actual != reference.expected_concurrencies:
            raise ValueError(
                f"{reference.job} concurrency mismatch: {actual} != {reference.expected_concurrencies}"
            )
        config = reference.root / "config.yaml"
        audited.append(
            {
                "label": reference.label,
                "job": reference.job,
                "gpus": reference.gpus,
                "slurm": _slurm(reference.job, reference.gpus),
                "config": str(config),
                "config_sha256": _sha256(config),
                "points": points,
            }
        )
    report = {"status": "pass", "references": audited}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
