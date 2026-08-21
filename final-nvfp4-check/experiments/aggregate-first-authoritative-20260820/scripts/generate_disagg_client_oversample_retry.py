#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate disagg retries with the TRTLLM5 client oversampling behavior."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
RECIPE_DIR = EXPERIMENT / "recipes" / "disaggregate"
BASE_MANIFEST = RECIPE_DIR / "best-noeplb-manifest.json"
HARNESS = EXPERIMENT / "setup" / "disagg-client-oversample-harness"
OVERLAY = EXPERIMENT / "setup" / "disagg-client-oversample"
ATTEMPT = "client-oversample-retry2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base_rows = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    header = (
        "# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    retry_rows: list[dict[str, Any]] = []
    client_files = [
        OVERLAY / "sitecustomize.py",
        HARNESS / "run_benchmark.sh",
        HARNESS / "start_server.sh",
        HARNESS / "start_worker.sh",
        HARNESS / "wait_server.sh",
        HARNESS / "get_env.py",
        HARNESS / "process_power_logs.py",
    ]
    client_file_hashes = {str(path): _sha256(path) for path in client_files}
    for base_row in base_rows:
        mode = str(base_row["mode"])
        base_recipe = Path(base_row["path"])
        config = yaml.safe_load(base_recipe.read_text(encoding="utf-8"))
        config["slurm"]["job_name"] = f"nv4-disagg-{mode}-{ATTEMPT}"
        config["environment"]["work_dir"] = str(HARNESS)
        path = RECIPE_DIR / f"best-noeplb-{mode}-{ATTEMPT}.yaml"
        path.write_text(
            header + yaml.safe_dump(copy.deepcopy(config), sort_keys=False),
            encoding="utf-8",
        )

        metadata = copy.deepcopy(base_row)
        metadata.update(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "base_current_recipe": str(base_recipe),
                "base_current_recipe_sha256": _sha256(base_recipe),
                "client_oversample_expected": True,
                "client_oversample_harness": str(HARNESS),
                "client_file_sha256": client_file_hashes,
            }
        )
        metadata["controlled_differences"] = list(
            metadata["controlled_differences"]
        ) + [
            "client-only CustomDataset oversampling restored to the TRTLLM5 behavior"
        ]
        retry_rows.append(metadata)
        print(path)

    manifest = RECIPE_DIR / f"best-noeplb-{ATTEMPT}-manifest.json"
    manifest.write_text(
        json.dumps(retry_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest)


if __name__ == "__main__":
    main()
