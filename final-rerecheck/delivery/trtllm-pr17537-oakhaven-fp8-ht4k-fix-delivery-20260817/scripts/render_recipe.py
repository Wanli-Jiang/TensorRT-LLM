#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a portable srt-slurm recipe without modifying srt-slurm."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--delivery-root", required=True)
    parser.add_argument("--slurm-account", required=True)
    parser.add_argument("--slurm-partition", required=True)
    parser.add_argument("--slurm-qos", required=True)
    parser.add_argument("--gpu-type", default="gb300")
    args = parser.parse_args()

    replacements = {
        "@@MODEL_PATH@@": args.model_path,
        "@@CONTAINER_IMAGE@@": args.container_image,
        "@@DELIVERY_ROOT@@": str(Path(args.delivery_root).resolve()),
        "@@SLURM_ACCOUNT@@": args.slurm_account,
        "@@SLURM_PARTITION@@": args.slurm_partition,
        "@@SLURM_QOS@@": args.slurm_qos,
        "@@GPU_TYPE@@": args.gpu_type,
    }
    rendered = args.template.read_text()
    for marker, value in replacements.items():
        rendered = rendered.replace(marker, value)
    unresolved = sorted({part.split("@@", 1)[0] for part in rendered.split("@@")[1::2]})
    if "@@" in rendered:
        raise RuntimeError(f"Unresolved template marker(s): {unresolved}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
