#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the focused no-MTP budget16896 curve and budget33792 check."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
BASE = EXPERIMENT / "recipes" / "aggregate-ht-nomtp-adp16-cutedsl-static528-tail.yaml"
VARIANTS: dict[str, dict[str, Any]] = {
    "chunk-budget16896-curve": {
        "max_num_tokens": 16896,
        "moe_max_num_tokens": 16896,
        "concurrencies": [1536, 1792, 2048, 2304],
        "role": "independent winner confirmation and saturation curve",
    },
    "chunk-budget33792-c2048": {
        "max_num_tokens": 33792,
        "moe_max_num_tokens": 33792,
        "concurrencies": [2048],
        "role": "focused four-full-prefill budget interaction check",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render(variant: str, spec: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    config["name"] = f"nvfp4-current-agg-ht-nomtp-confirm-{variant}"
    serving = config["backend"]["trtllm_config"]["aggregated"]
    serving["enable_chunked_prefill"] = True
    serving["max_num_tokens"] = int(spec["max_num_tokens"])
    serving["moe_config"]["max_num_tokens"] = int(spec["moe_max_num_tokens"])
    serving["kv_cache_config"]["use_kv_cache_manager_v2"] = False
    if "speculative_config" in serving:
        raise ValueError("no-MTP base unexpectedly contains speculative_config")
    config["benchmark"]["concurrencies"] = list(spec["concurrencies"])
    cache_root = f"/tmp/nvfp4-nomtp-budget-confirm-{variant}/{{node}}"
    config["backend"]["aggregated_environment"].update(
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
    output_dir = EXPERIMENT / "recipes" / "nomtp-budget-confirmation"
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
            header + yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        mounted_files = []
        for mount in config.get("extra_mount", []):
            source = Path(mount.split(":", maxsplit=1)[0])
            mounted_files.append({"path": str(source), "sha256": _sha256(source)})
        manifest.append(
            {
                "variant": variant,
                "role": spec["role"],
                "path": str(path),
                "sha256": _sha256(path),
                "base": str(BASE),
                "base_sha256": _sha256(BASE),
                "placement": "Static528",
                "enable_chunked_prefill": True,
                "max_num_tokens": spec["max_num_tokens"],
                "moe_max_num_tokens": spec["moe_max_num_tokens"],
                "concurrencies": spec["concurrencies"],
                "mounted_files": mounted_files,
            }
        )
        print(path)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
