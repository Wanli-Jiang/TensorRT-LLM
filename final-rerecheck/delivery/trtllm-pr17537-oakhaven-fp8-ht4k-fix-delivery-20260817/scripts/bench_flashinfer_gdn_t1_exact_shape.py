#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flashinfer.gdn_kernels import gdn_decode_bf16_state as gdn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--reference-state", type=Path)
    return parser.parse_args()


def event_time_ms(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    key_heads = 16
    value_heads = 128
    key_dim = 128
    value_dim = 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(1234)
    q = 0.05 * torch.randn(
        batch_size, 1, key_heads, key_dim, device=device, dtype=dtype
    )
    k = 0.05 * torch.randn_like(q)
    v = 0.05 * torch.randn(
        batch_size, 1, value_heads, value_dim, device=device, dtype=dtype
    )
    a = 0.05 * torch.randn(
        batch_size, 1, value_heads, device=device, dtype=dtype
    )
    b = 0.05 * torch.randn_like(a)
    a_log = torch.zeros(value_heads, device=device, dtype=torch.float32)
    dt_bias = torch.zeros(value_heads, device=device, dtype=torch.float32)
    initial_state_seed = 0.02 * torch.randn(
        batch_size,
        value_heads,
        value_dim,
        key_dim,
        device=device,
        dtype=dtype,
    )
    initial_state = initial_state_seed.clone()
    initial_state_indices = torch.arange(
        batch_size, device=device, dtype=torch.int32
    )
    output = torch.empty_like(v)

    def run_kernel() -> None:
        gdn.gated_delta_rule(
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state,
            initial_state_indices=initial_state_indices,
            use_qk_l2norm_in_kernel=True,
            scale=key_dim**-0.5,
            output=output,
        )

    # Compile once, then reset and preserve the result of exactly one production
    # entry-point invocation for the cross-version numerical comparison.
    run_kernel()
    torch.cuda.synchronize()
    initial_state.copy_(initial_state_seed)
    run_kernel()
    torch.cuda.synchronize()
    correctness_output = output.detach().cpu().clone()
    correctness_state = initial_state.detach().cpu().clone()

    initial_state.copy_(initial_state_seed)
    eager_ms = event_time_ms(run_kernel, args.warmup, args.repeats)

    initial_state.copy_(initial_state_seed)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_kernel()
    graph_ms = event_time_ms(graph.replay, args.warmup, args.repeats)

    import flashinfer

    module_path = Path(gdn.__file__)
    source = module_path.read_text()
    result = {
        "label": args.label,
        "flashinfer_version": flashinfer.__version__,
        "module_path": str(module_path),
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "has_static_t1_specialization": "static_batch_size" in source,
        "gpu": torch.cuda.get_device_name(),
        "shape": {
            "batch_size": batch_size,
            "decode_tokens": 1,
            "key_heads": key_heads,
            "value_heads": value_heads,
            "key_dim": key_dim,
            "value_dim": value_dim,
        },
        "warmup": args.warmup,
        "repeats": args.repeats,
        "eager_ms": eager_ms,
        "cuda_graph_replay_ms": graph_ms,
    }
    (args.output_dir / f"{args.label}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    torch.save(correctness_output, args.output_dir / f"{args.label}-output.pt")
    torch.save(correctness_state, args.output_dir / f"{args.label}-state.pt")

    if args.reference_output is not None and args.reference_state is not None:
        reference_output = torch.load(args.reference_output)
        reference_state = torch.load(args.reference_state)
        output_diff = (reference_output.float() - correctness_output.float()).abs()
        state_diff = (reference_state.float() - correctness_state.float()).abs()
        dynamic_metrics = json.loads((args.output_dir / "dynamic.json").read_text())
        comparison = {
            "output_max_abs_diff": output_diff.max().item(),
            "output_mean_abs_diff": output_diff.mean().item(),
            "output_exact_equal": torch.equal(reference_output, correctness_output),
            "state_max_abs_diff": state_diff.max().item(),
            "state_mean_abs_diff": state_diff.mean().item(),
            "state_exact_equal": torch.equal(reference_state, correctness_state),
            "eager_speedup_static_over_dynamic": (
                dynamic_metrics["eager_ms"] / result["eager_ms"]
            ),
            "cuda_graph_speedup_static_over_dynamic": (
                dynamic_metrics["cuda_graph_replay_ms"] / result["cuda_graph_replay_ms"]
            ),
        }
        (args.output_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2) + "\n"
        )
        print(json.dumps(comparison, indent=2), flush=True)

    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
