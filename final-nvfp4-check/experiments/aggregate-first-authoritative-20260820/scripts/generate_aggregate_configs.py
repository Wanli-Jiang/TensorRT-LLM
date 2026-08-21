#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the frozen current-image aggregate NVFP4 reproduction configs."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

EXPERIMENT = Path(__file__).resolve().parents[1]
RECIPE_DIR = EXPERIMENT / "recipes"
MODEL = (
    "/lustre/fsw/portfolios/coreai/users/williamj/models/"
    "oakhaven-max-final-nvfp4-routed-experts-experimental_vv1-clean"
)
IMAGE = (
    "/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/"
    "williamj/containers/trtllm-9a6889b-worktree-gdnstatic-crossmap-qa-20260814.sqsh"
)
MAPS = {
    "nomtp": EXPERIMENT / "maps" / "nomtp-static528.yaml",
    "mtp3": EXPERIMENT / "maps" / "mtp3-static528.yaml",
}
LL_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
HT_CONCURRENCIES = {
    "nomtp": [
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
    ],
    "mtp3": [
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
    ],
}
HT_V2_DIAGNOSTIC_CONCURRENCIES = {
    "nomtp": [2048, 2304],
    "mtp3": [640, 768, 1024],
}
V2_MATCHED_MAX_TOKENS = {
    ("latency", "nomtp"): 6_361_024,
    ("latency", "mtp3"): 5_220_000,
    ("throughput", "nomtp"): 1_295_328,
    ("throughput", "mtp3"): 1_606_784,
}


def _common(name: str, nodes: int) -> dict[str, Any]:
    return {
        "name": name,
        "slurm": {
            "account": "coreai_comparch_trtllm",
            "partition": "batch",
            "time_limit": "02:00:00",
        },
        "sbatch_directives": {
            "qos": "short",
            "cpus-per-task": "16",
            "mem": "0",
            "exclusive": "",
            "switches": "1@10:00",
            "comment": (
                '{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60",'
                '"reason":"model_loading","description":"Current-image Oakhaven '
                "NVFP4 aggregate reference reproduction; audited loading-only heartbeat; "
                'metrics endpoint forbidden."}}'
            ),
        },
        "model": {"path": MODEL, "container": IMAGE, "precision": "fp4"},
        "identity": {
            "container": {"image": IMAGE},
            "frameworks": {"tensorrt_llm": "1.3.0rc25"},
        },
        "resources": {
            "gpu_type": "gb300",
            "gpus_per_node": 4,
            "agg_nodes": nodes,
            "agg_workers": 1,
            "gpus_per_agg": nodes * 4,
        },
        "srun_options": {
            "mem": "0",
            "container-remap-root": "",
            "cpu-bind": "none",
        },
        "frontend": {"type": "trtllm_serve", "enable_multiple_frontends": False},
        "backend": {
            "type": "trtllm",
            "aggregated_environment": {
                "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "NCCL_DEBUG": "WARN",
                "NCCL_GRAPH_MIXING_SUPPORT": "0",
                "NCCL_MNNVL_ENABLE": "1",
                "NCCL_CUMEM_ENABLE": "1",
                "OMPI_MCA_coll_ucc_enable": "0",
                "TLLM_ALL_RANK_LOG": "1",
                "TLLM_LOG_LEVEL": "INFO",
                "TRTLLM_ENABLE_PDL": "1",
                "TRTLLM_SERVER_DISABLE_GC": "1",
                "TRTLLM_WORKER_DISABLE_GC": "1",
                "TRTLLM_LOW_M_GEMM_BACKEND": "off",
                "TRTLLM_USE_GDN_REPLAY": "0",
                "TRTLLM_HF_WEIGHT_CACHE": "0",
                "TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL": "False",
                "TRT_LLM_LOAD_WEIGHTS_NUM_WORKERS": "4",
            },
            "trtllm_config": {"aggregated": {}},
        },
        "health_check": {"max_attempts": 480, "interval_seconds": 10},
        "benchmark": {
            "type": "sa-bench",
            "dataset_name": "random",
            "isl": 8192,
            "osl": 1024,
            "req_rate": "inf",
            "random_range_ratio": 1.0,
            "use_chat_template": False,
        },
    }


def _disable_metrics(serving: dict[str, Any]) -> None:
    serving.update(
        {
            "stream_interval": 10,
            "return_perf_metrics": False,
            "enable_iter_perf_stats": False,
            "enable_iter_req_stats": False,
            "print_iter_log": False,
            "trust_remote_code": True,
            "num_postprocess_workers": 4,
        }
    )


def _set_unique_caches(config: dict[str, Any], arm: str) -> None:
    environment = config["backend"]["aggregated_environment"]
    root = f"/tmp/nvfp4-authoritative-{arm}/{{node}}"
    environment.update(
        {
            "XDG_CACHE_HOME": f"{root}/xdg",
            "CUDA_CACHE_PATH": f"{root}/cuda",
            "TORCHINDUCTOR_CACHE_DIR": f"{root}/torchinductor",
            "TRITON_CACHE_DIR": f"{root}/triton",
            "FLASHINFER_WORKSPACE_BASE": f"{root}/fi-workspace",
            "FLASHINFER_CUBIN_DIR": f"{root}/fi-cubins",
        }
    )


def _enable_mtp3(config: dict[str, Any]) -> None:
    environment = config["backend"]["aggregated_environment"]
    environment.update(
        {
            "TRTLLM_USE_GDN_REPLAY": "1",
            "TLLM_SPEC_SKIP_IDENTITY_DRAFT_GATHER": "0",
            "TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS": "2.3",
        }
    )
    serving = config["backend"]["trtllm_config"]["aggregated"]
    serving["speculative_config"] = {"decoding_type": "MTP", "max_draft_len": 3}


def _ll(mode: str) -> dict[str, Any]:
    config = _common(f"nvfp4-current-agg-ll-{mode}-tp8-trtllm", 2)
    serving = config["backend"]["trtllm_config"]["aggregated"]
    max_num_tokens = 8704 if mode == "mtp3" else 8448
    serving.update(
        {
            "tensor_parallel_size": 8,
            "moe_expert_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "enable_attention_dp": False,
            "enable_lm_head_tp_in_adp": False,
            "disable_mm_encoder": True,
            "max_batch_size": 128,
            "max_num_tokens": max_num_tokens,
            "max_seq_len": 9472,
            "enable_chunked_prefill": False,
            "allreduce_strategy": "MNNVL",
            "cuda_graph_config": {"enable_padding": True, "max_batch_size": 128},
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": False,
                "free_gpu_memory_fraction": 0.70 if mode == "mtp3" else 0.74,
                "dtype": "fp8",
                "mamba_ssm_cache_dtype": "bfloat16",
                "mamba_ssm_stochastic_rounding": False,
            },
            "moe_config": {
                "backend": "TRTLLM",
                "max_num_tokens": max_num_tokens,
                "use_low_precision_moe_combine": True,
            },
            "nvfp4_gemm_config": {
                "allowed_backends": ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
            },
        }
    )
    _disable_metrics(serving)
    _set_unique_caches(config, f"ll-{mode}")
    if mode == "mtp3":
        _enable_mtp3(config)
    config["benchmark"].update(
        {
            "concurrencies": LL_CONCURRENCIES,
            "num_prompts_mult": 5,
            "num_warmup_mult": 1,
            "reuse_http_connections": False,
        }
    )
    return config


def _enable_kv_manager_v2(
    config: dict[str, Any], arm: str, matched_max_tokens: int
) -> None:
    serving = config["backend"]["trtllm_config"]["aggregated"]
    kv_cache = serving["kv_cache_config"]
    kv_cache["use_kv_cache_manager_v2"] = True
    kv_cache["avg_seq_len"] = 9216
    kv_cache["max_tokens"] = matched_max_tokens
    config["name"] = f"{config['name']}-kv-v2"
    _set_unique_caches(config, arm)


def _ll_v2(mode: str) -> dict[str, Any]:
    config = copy.deepcopy(_ll(mode))
    _enable_kv_manager_v2(
        config,
        f"ll-{mode}-kv-v2",
        V2_MATCHED_MAX_TOKENS[("latency", mode)],
    )
    return config


def _ht(mode: str, static_eplb: bool) -> dict[str, Any]:
    eplb = "static528" if static_eplb else "noeplb"
    config = _common(f"nvfp4-current-agg-ht-{mode}-adp16-cutedsl-{eplb}", 4)
    environment = config["backend"]["aggregated_environment"]
    environment.update(
        {
            "ENABLE_CONFIGURABLE_MOE": "1",
            "TRTLLM_FORCE_COMM_METHOD": "NVLINK_ONE_SIDED",
            "TLLM_MOE_A2A_DISPATCH_BLOCK_SIZE": "256",
            "TLLM_MOE_A2A_COMBINE_BLOCK_SIZE": "256",
            "TRTLLM_MOE_A2A_WORKSPACE_MB": "2304",
        }
    )
    mtp3 = mode == "mtp3"
    max_batch_size = 32 if mtp3 else 128
    graph_batch_sizes = (
        [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
        if mtp3
        else [
            1,
            2,
            4,
            8,
            12,
            16,
            24,
            32,
            40,
            48,
            56,
            64,
            72,
            80,
            88,
            96,
            104,
            112,
            120,
            128,
        ]
    )
    moe_config: dict[str, Any] = {
        "backend": "CUTEDSL",
        "max_num_tokens": 8192 if mtp3 else 8448,
        "use_low_precision_moe_combine": True,
    }
    if static_eplb:
        moe_config["load_balancer"] = "/eplb.yaml"
        config["extra_mount"] = [f"{MAPS[mode]}:/eplb.yaml:ro"]
    serving = config["backend"]["trtllm_config"]["aggregated"]
    serving.update(
        {
            "tensor_parallel_size": 16,
            "moe_expert_parallel_size": 16,
            "pipeline_parallel_size": 1,
            "enable_attention_dp": True,
            "enable_lm_head_tp_in_adp": True,
            "disable_mm_encoder": True,
            "max_batch_size": max_batch_size,
            "max_num_tokens": 1024 if mtp3 else 8448,
            "max_seq_len": 9472,
            "enable_chunked_prefill": mtp3,
            "allreduce_strategy": "AUTO",
            "cuda_graph_config": {
                "enable_padding": True,
                "max_batch_size": max_batch_size,
                "batch_sizes": graph_batch_sizes,
            },
            "kv_cache_config": {
                "enable_block_reuse": False,
                "use_kv_cache_manager_v2": False,
                "free_gpu_memory_fraction": 0.92,
                "dtype": "fp8",
                "mamba_ssm_cache_dtype": "bfloat16",
                "mamba_ssm_stochastic_rounding": False,
            },
            "moe_config": moe_config,
            "nvfp4_gemm_config": {
                "allowed_backends": ["cutlass", "cublaslt", "cutedsl", "cuda_core"]
            },
            "attention_dp_config": {
                "enable_balance": True,
                "batching_wait_iters": 0,
                "timeout_iters": 60,
            },
        }
    )
    _disable_metrics(serving)
    _set_unique_caches(config, f"ht-{mode}-{eplb}")
    if mtp3:
        _enable_mtp3(config)
    config["benchmark"].update(
        {
            "concurrencies": HT_CONCURRENCIES[mode],
            "num_prompts_mult": 3,
            "num_warmup_mult": 1,
            "reuse_http_connections": True,
        }
    )
    return config


def _nomtp_ht_tail(static_eplb: bool) -> dict[str, Any]:
    config = copy.deepcopy(_ht("nomtp", static_eplb))
    eplb = "static528" if static_eplb else "noeplb"
    config["name"] = f"nvfp4-current-agg-ht-nomtp-adp16-cutedsl-{eplb}-tail"
    config["benchmark"]["concurrencies"] = [1536, 1792, 2048, 2304]
    _set_unique_caches(config, f"ht-nomtp-{eplb}-tail")
    return config


def _ht_v2(mode: str) -> dict[str, Any]:
    config = copy.deepcopy(_ht(mode, False))
    config["benchmark"]["concurrencies"] = HT_V2_DIAGNOSTIC_CONCURRENCIES[mode]
    _enable_kv_manager_v2(
        config,
        f"ht-{mode}-noeplb-kv-v2",
        V2_MATCHED_MAX_TOKENS[("throughput", mode)],
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Generate only an exact output filename; repeat as needed.",
    )
    args = parser.parse_args()
    configs = {
        "aggregate-ll-nomtp-tp8-trtllm.yaml": _ll("nomtp"),
        "aggregate-ll-mtp3-tp8-trtllm.yaml": _ll("mtp3"),
        "aggregate-ht-nomtp-adp16-cutedsl-noeplb.yaml": _ht("nomtp", False),
        "aggregate-ht-nomtp-adp16-cutedsl-static528.yaml": _ht("nomtp", True),
        "aggregate-ht-mtp3-adp16-cutedsl-noeplb.yaml": _ht("mtp3", False),
        "aggregate-ht-mtp3-adp16-cutedsl-static528.yaml": _ht("mtp3", True),
        "aggregate-ht-nomtp-adp16-cutedsl-noeplb-tail.yaml": _nomtp_ht_tail(False),
        "aggregate-ht-nomtp-adp16-cutedsl-static528-tail.yaml": _nomtp_ht_tail(True),
        "aggregate-ll-nomtp-tp8-trtllm-kv-v2.yaml": _ll_v2("nomtp"),
        "aggregate-ll-mtp3-tp8-trtllm-kv-v2.yaml": _ll_v2("mtp3"),
        "aggregate-ht-nomtp-adp16-cutedsl-noeplb-kv-v2.yaml": _ht_v2("nomtp"),
        "aggregate-ht-mtp3-adp16-cutedsl-noeplb-kv-v2.yaml": _ht_v2("mtp3"),
    }
    RECIPE_DIR.mkdir(parents=True, exist_ok=True)
    header = (
        "# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n\n"
    )
    for name, config in configs.items():
        if args.only and name not in args.only:
            continue
        path = RECIPE_DIR / name
        path.write_text(
            header + yaml.safe_dump(copy.deepcopy(config), sort_keys=False),
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
