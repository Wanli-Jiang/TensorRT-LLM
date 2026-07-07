# Accuracy Evaluation Tool `trtllm-eval`

We provide a CLI tool `trtllm-eval` for evaluating model accuracy. It shares the core evaluation logics with the [accuracy test suite](../../tests/integration/defs/accuracy) of TensorRT-LLM.

`trtllm-eval` is built on the offline API -- [LLM API](https://nvidia.github.io/TensorRT-LLM/llm-api/index.html). It provides developers a unified entrypoint for accuracy evaluation. Compared with the online API [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html), offline API provides clearer error messages and simplifies the debugging workflow.

`trtllm-eval` follows the CLI interface of [`trtllm-serve`](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-serve.html).

```bash
pip install -r requirements.txt

# Evaluate Llama-3.1-8B-Instruct on MMLU
trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct mmlu

# Evaluate Llama-3.1-8B-Instruct on GSM8K
trtllm-eval --model meta-llama/Llama-3.1-8B-Instruct gsm8k

# Evaluate Llama-3.3-70B-Instruct on GPQA Diamond
trtllm-eval --model meta-llama/Llama-3.3-70B-Instruct gpqa_diamond
```

The `--model` argument accepts either a Hugging Face model ID or a local checkpoint path. By default, `trtllm-eval` runs the model with the PyTorch backend; pass `--backend tensorrt` to switch to the TensorRT backend. Alternatively, the `--model` argument also accepts a local path to pre-built TensorRT engines; in that case, please pass the Hugging Face tokenizer path to the `--tokenizer` argument.

See more details by `trtllm-eval --help`.

## NeMo-Gym benchmarks

`trtllm-eval` exposes [NeMo-Gym](https://github.com/NVIDIA-NeMo/Gym) Artificial-Analysis (AA)
benchmarks as in-process subcommands:

| Subcommand   | Benchmark | Headline metric |
|--------------|-----------|-----------------|
| `gpqa_ng`    | GPQA-diamond — multiple choice | `pass@1` accuracy |
| `ifbench_ng` | AllenAI IFBench — instruction following (300 prompts) | instruction- / prompt-level accuracy |
| `scicode_ng` | SciCode — multi-step scientific code (`test_aai` split) | `subtask_accuracy` / `problem_accuracy` |

NeMo-Gym drives the model over an **OpenAI-compatible endpoint** rather than loading it
directly. `trtllm-eval` loads the model once into the in-process `LLM`, exposes that same LLM on
a localhost endpoint (TensorRT-LLM's `OpenAIServer`, no second copy of the weights), and drives
NeMo-Gym's `gym eval run` against it. The score is logged as
`[evaluate] NeMo-Gym <bench> results: ...`.

### 1. Install (once)

`uv` must be on `PATH` (NeMo-Gym builds a small venv per server). Run inside the TensorRT-LLM
container:

```bash
bash examples/trtllm-eval/install_nemo_gym.sh
gym list benchmarks | grep gpqa      # sanity check
```

[`install_nemo_gym.sh`](install_nemo_gym.sh) installs NeMo-Gym (pure-Python, no torch/cuda) with
`--no-deps` plus only the deps TensorRT-LLM does not already ship. To leave the container's
Python untouched, install into a dedicated venv instead — `INSTALL_MODE=venv bash
examples/trtllm-eval/install_nemo_gym.sh` — and point `TRTLLM_NEMO_GYM_VENV` at the printed path.

### 2. Run

```bash
# Single GPU
trtllm-eval --model <hf_or_path> gpqa_ng --num_samples 20

# Multi-GPU (TP/EP)
trtllm-eval --model <path> --tp_size 4 --ep_size 4 \
  --kv_cache_free_gpu_memory_fraction 0.7 --max_batch_size 16 \
  gpqa_ng

# ifbench / scicode run the same way
trtllm-eval --model <path> ifbench_ng --num_samples 20
trtllm-eval --model <path> scicode_ng --num_samples 20
```

**Data.** Each subcommand stages its dataset on first use. `gpqa_ng` sources it from
`ns_acc_bench_infra` when available (offline, no token); otherwise it downloads the HF-gated
`Idavidrein/gpqa`, which needs `export HF_TOKEN=<token>`. `ifbench_ng` (AllenAI GitHub) and
`scicode_ng` (public HF `SciCode1/SciCode`) download without any token.

**Options** (all subcommands): `--num_samples`, `--num_repeats`, `--num_samples_in_parallel`,
`--max_output_length`, `--temperature` / `--top_p` / `--top_k`, `--output_dir` (keep the
rollouts / metrics / NeMo-Gym log).

**Environment**:

| Variable | Purpose |
|----------|---------|
| `NS_ACC_BENCH_INFRA` | Shared datasets folder (default `<LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra`); lets `gpqa_ng` run offline / token-free and provides SciCode's `test_data.h5`. |
| `HF_TOKEN` | Only for `gpqa_ng` when the infra copy is unavailable (HF-gated `Idavidrein/gpqa`). |
| `TRTLLM_NEMO_GYM_VENV` | Use a dedicated NeMo-Gym venv (from the `INSTALL_MODE=venv` install). |
| `TRTLLM_NEMO_GYM_UV_VENV_DIR` | Persistent dir for NeMo-Gym's per-server `uv` venvs, so they are built once and reused across runs. |
| `TRTLLM_NEMO_GYM_SCICODE_H5` | Explicit path to SciCode's `test_data.h5` (else taken from `ns_acc_bench_infra`). |

### 3. Per-benchmark notes

- **`gpqa_ng`** — GPQA-diamond multiple choice, rule-based answer extraction. Data comes from
  `ns_acc_bench_infra` (no token) or the HF-gated `Idavidrein/gpqa` (needs `HF_TOKEN`).
- **`ifbench_ng`** — AllenAI IFBench strict grader (checks the literal response).
  `trtllm-eval` sends `enable_thinking=true` + `skip_special_tokens=true`, and
  `install_nemo_gym.sh` patches the grader to strip the leading reasoning whitespace (a `\n\n`
  left after the `<think>` block is split out) before scoring — otherwise exact-format
  instructions fail.
- **`scicode_ng`** — problems download from public HF (no token); code is executed in a Ray
  subprocess worker, so it also needs SciCode's `test_data.h5` ground-truth asset. That comes
  from `ns_acc_bench_infra` (`datasets/test_data.h5`) automatically, or set
  `TRTLLM_NEMO_GYM_SCICODE_H5` explicitly.

> On first use NeMo-Gym builds a small `uv` venv per server (downloads ray/scipy/…, a few
> minutes). Set `TRTLLM_NEMO_GYM_UV_VENV_DIR` to a persistent dir to build them once and reuse.

### 4. Example: Nemotron-3-Super-120B-A12B-FP8

Serving config `super-v3-fp8.yaml` (passed via `--extra_llm_api_options`), on 4×H200:

```yaml
cuda_graph_config:
  enable_padding: true
  max_batch_size: 8
enable_attention_dp: true
enable_chunked_prefill: true
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.6
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
moe_config:
  backend: CUTLASS
```

Run GPQA-diamond (temperature 1.0 / top_p 0.95, matching the AA methodology):

```bash
# HF_TOKEN only needed if ns_acc_bench_infra is unavailable (gpqa data source).
trtllm-eval --model /path/to/NVIDIA-Nemotron-3-Super-120B-A12B-FP8 \
  --backend pytorch --tp_size 4 --ep_size 4 --max_batch_size 8 \
  --kv_cache_free_gpu_memory_fraction 0.6 --trust_remote_code \
  --extra_llm_api_options super-v3-fp8.yaml \
  gpqa_ng --temperature 1.0 --top_p 0.95
```

NeMo-Gym results on this model (temperature 1.0 / top_p 0.95, single sample):

| Benchmark | NeMo-Gym |
|-----------|----------|
| GPQA-diamond `pass@1`         | 78.3% |
| ifbench instruction / prompt  | 74.8% / 72.0% |
| scicode subtask / problem     | 40.1% / 17.9% |

For long-reasoning models, raise `--max_output_length` to avoid truncating GPQA answers.
