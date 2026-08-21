# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restore TRTLLM5 CustomDataset oversampling for the disagg load generator."""

from __future__ import annotations

from typing import Any

from tensorrt_llm.serve.scripts.benchmark_dataset import CustomDataset

_ORIGINAL_SAMPLE = CustomDataset.sample


def _sample_with_oversampling(
    self: CustomDataset, tokenizer: Any, num_requests: int
) -> list[Any]:
    requests = _ORIGINAL_SAMPLE(self, tokenizer, num_requests)
    before = len(requests)
    self.maybe_oversample_requests(requests, num_requests)
    print(
        "NVFP4_CLIENT_OVERSAMPLE "
        f"before={before} requested={num_requests} after={len(requests)}",
        flush=True,
    )
    return requests


CustomDataset.sample = _sample_with_oversampling
