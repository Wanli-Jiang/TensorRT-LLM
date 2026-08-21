#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a three-arm MTP3 chunking/token-budget causal decomposition."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE = EXPERIMENT / "recipes" / "aggregate-ht-mtp3-adp16-cutedsl-noeplb.yaml"
CONCURRENCIES = [384, 512, 640, 768, 896, 1024]
VARIANTS: dict[str, dict[str, Any]] = {
    "chunk1024": {
        "enable_chunked_prefill": True,
        "max_num_tokens": 1024,
        "moe_max_num_tokens": 8192,
        "role": "TRTLLM5-matched chunked baseline",
    },
    "chunk8448": {
        "enable_chunked_prefill": True,
        "max_num_tokens": 8448,
        "moe_max_num_tokens": 8448,
        "role": "large-budget control",
    },
    "nochunk8448": {
        "enable_chunked_prefill": False,
        "max_num_tokens": 8448,
        "moe_max_num_tokens": 8448,
        "role": "no-chunk treatment",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(variant: str, spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    config["name"] = f"nvfp4-current-agg-ht-mtp3-noeplb-{variant}"
    serving = config["backend"]["trtllm_config"]["aggregated"]
    serving["enable_chunked_prefill"] = bool(spec["enable_chunked_prefill"])
    serving["max_num_tokens"] = int(spec["max_num_tokens"])
    serving["moe_config"]["max_num_tokens"] = int(spec["moe_max_num_tokens"])
    config["benchmark"]["concurrencies"] = list(CONCURRENCIES)
    environment = config["backend"]["aggregated_environment"]
    cache_root = f"/tmp/nvfp4-mtp3-chunk-ab-{variant}/{{node}}"
    environment.update(
        {
            "XDG_CACHE_HOME": f"{cache_root}/xdg",
            "CUDA_CACHE_PATH": f"{cache_root}/cuda",
            "TORCHINDUCTOR_CACHE_DIR": f"{cache_root}/torchinductor",
            "TRITON_CACHE_DIR": f"{cache_root}/triton",
            "FLASHINFER_WORKSPACE_BASE": f"{cache_root}/fi-workspace",
            "FLASHINFER_CUBIN_DIR": f"{cache_root}/fi-cubins",
        }
    )
    return config


def main() -> None:
    output_dir = EXPERIMENT / "recipes" / "mtp3-nochunk-ab"
    output_dir.mkdir(parents=True, exist_ok=True)
    header = (
        "# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    manifest = []
    for variant, spec in VARIANTS.items():
        config = _render(variant, spec)
        path = output_dir / f"{variant}.yaml"
        path.write_text(
            header + yaml.safe_dump(config, sort_keys=False),
            encoding="utf-8",
        )
        manifest.append(
            {
                "variant": variant,
                "role": spec["role"],
                "path": str(path),
                "sha256": _sha256(path),
                "enable_chunked_prefill": spec["enable_chunked_prefill"],
                "max_num_tokens": spec["max_num_tokens"],
                "moe_max_num_tokens": spec["moe_max_num_tokens"],
                "concurrencies": CONCURRENCIES,
                "base": str(BASE),
                "base_sha256": _sha256(BASE),
            }
        )
        print(path)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
