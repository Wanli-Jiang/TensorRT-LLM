#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render current-image copies of the best TRTLLM5 no-EPLB disagg recipes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
TRTLLM5_EXPERIMENTS = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "williamj/TRTLLM5/oak-nvfp4-disagg-sweep/experiments"
)
REFERENCE_SWEEP = TRTLLM5_EXPERIMENTS / "oakhaven-max-nvfp4-disagg-8k1k-20260812"
RIGHT_EDGE = TRTLLM5_EXPERIMENTS / "oakhaven-max-nvfp4-disagg-mtp3-right-edge-20260820"
IMAGE = Path(
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh"
)
IMAGE_SHA256 = "08c33698800171f1836c17346d4e8c6ef72705f360d925f1ab075ed035e3fb59"
RUNNER = EXPERIMENT / "scripts/run_disagg_with_loading_heartbeat.slurm"
LAUNCHER = EXPERIMENT / "setup/trtllm-llmapi-launch"
ORIGINAL_LAUNCHER = EXPERIMENT / "setup/current-image-trtllm-llmapi-launch.original"
DATASET = REFERENCE_SWEEP / "data/oakhaven-max-8192-1024-4096-ratio-1.0_for_serve.json"
RECIPES: dict[str, dict[str, Any]] = {
    "nomtp": {
        "source": REFERENCE_SWEEP
        / "recipes/e2e-nomtp-ctx3adp8-gen1-dep16-c1536-r24-sol90-v1.yaml",
        "reference_job": "500677",
        "reference_total_tps_per_gpu": 5321.710735990,
        "concurrency": 1536,
        "rounds": 24,
        "ctx_instances": 3,
        "gen_instances": 1,
    },
    "mtp3": {
        "source": RIGHT_EDGE
        / "recipes/e2e-mtp3-ctx4adp8-gen1-dep16-c768-r12-noeplb-mbs48-formal-v1.yaml",
        "reference_job": "524759",
        "reference_total_tps_per_gpu": 5296.636804728,
        "concurrency": 768,
        "rounds": 12,
        "ctx_instances": 4,
        "gen_instances": 1,
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _disable_metrics(role_config: dict[str, Any]) -> None:
    role_config.update(
        {
            "return_perf_metrics": False,
            "enable_iter_perf_stats": False,
            "enable_iter_req_stats": False,
            "print_iter_log": False,
        }
    )


def _render(mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = RECIPES[mode]
    source = Path(spec["source"])
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["slurm"]["script_file"] = str(RUNNER)
    config["slurm"]["job_name"] = f"nv4-disagg-{mode}-best-noeplb"
    config["benchmark"].update(
        {
            "multi_round": int(spec["rounds"]),
            "benchmark_ratio": 1.0,
            "concurrency_list": str(spec["concurrency"]),
            "input_length": 8192,
            "output_length": 1024,
            "dataset_file": str(DATASET),
        }
    )
    config["hardware"]["num_ctx_servers"] = int(spec["ctx_instances"])
    config["hardware"]["num_gen_servers"] = int(spec["gen_instances"])
    mounts = (
        "/lustre/:/lustre",
        f"{EXPERIMENT}:{EXPERIMENT}",
        f"{LAUNCHER}:/usr/local/bin/trtllm-llmapi-launch:ro",
        f"{ORIGINAL_LAUNCHER}:/opt/current-image-trtllm-llmapi-launch.original:ro",
        f"{DATASET}:{DATASET}:ro",
    )
    environment = config["environment"]
    environment.update(
        {
            "container_mount": ",".join(mounts),
            "container_image": str(IMAGE),
            "trtllm_repo": "none",
            "build_wheel": False,
            "cuda_architectures": "",
            "trtllm_wheel_path": "",
            "server_env_var": (
                "TRTLLM_SERVER_DISABLE_GC=1 TRTLLM_DISAGG_SERVER_DISABLE_GC=1 "
                "TLLM_DISABLE_MPI=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1"
            ),
        }
    )
    worker_env = str(environment["worker_env_var"])
    if "TLLM_ALL_RANK_LOG=" not in worker_env:
        worker_env += " TLLM_ALL_RANK_LOG=1"
    environment["worker_env_var"] = worker_env
    for role in ("ctx", "gen"):
        _disable_metrics(config["worker_config"][role])

    metadata = {
        "mode": mode,
        "placement": "no-EPLB",
        "source": str(source),
        "source_sha256": _sha256(source),
        "reference_job": str(spec["reference_job"]),
        "reference_total_tps_per_gpu": float(spec["reference_total_tps_per_gpu"]),
        "concurrency": int(spec["concurrency"]),
        "rounds": int(spec["rounds"]),
        "ctx_instances": int(spec["ctx_instances"]),
        "gen_instances": int(spec["gen_instances"]),
        "deployed_gpus": 8 * int(spec["ctx_instances"])
        + 16 * int(spec["gen_instances"]),
        "image": str(IMAGE),
        "image_sha256": IMAGE_SHA256,
        "dataset": str(DATASET),
        "dataset_sha256": _sha256(DATASET),
        "controlled_differences": [
            "current immutable image replaces old image plus TRTLLM5 source overlay",
            "all TensorRT-LLM performance/iteration/request metrics and iteration logs disabled",
            "experiment-local loading-heartbeat runner and launcher cache isolation",
        ],
    }
    return config, metadata


def main() -> None:
    output_dir = EXPERIMENT / "recipes/disaggregate"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    header = (
        "# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    for mode in RECIPES:
        config, metadata = _render(mode)
        path = output_dir / f"best-noeplb-{mode}.yaml"
        path.write_text(
            header + yaml.safe_dump(copy.deepcopy(config), sort_keys=False),
            encoding="utf-8",
        )
        metadata["path"] = str(path)
        metadata["sha256"] = _sha256(path)
        manifest.append(metadata)
        print(path)
    manifest_path = output_dir / "best-noeplb-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
