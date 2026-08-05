# TRTLLM5 NVFP4 sweep handoff

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
-->

This directory prepares an isolated TensorRT-LLM development baseline for the
NVFP4 sweep in `/lustre/fsw/portfolios/coreai/users/williamj/TRTLLM5`.

## Scope

- Upstream base: `6d72d24b8fb6c922ff896f51140f5a8a4f4d7f90`
- Development branch: `user/williamj/nvfp4-sweep-base`
- Stable source: the six completed commits from TensorRT-LLM PR #17192
- Container source mount: `/opt/trtllm5-dev`
- Persistent development environment: `/opt/trtllm5-handoff/venv`
- Persistent caches: `/opt/trtllm5-handoff/cache`

The port was applied onto the TRTLLM5 base rather than copying files from an
older checkout. This preserves changes already present on the newer main
branch and makes conflict resolution reviewable in Git.

### Included commits

| TRTLLM5 commit | Original commit | Purpose |
| --- | --- | --- |
| `a396fca185` | `d225dcbc89` | Normalize separate one-model draft KV-cache pool ratios. |
| `7c9a06db3d` | `360a39222b` | Select a deterministic valid FP8 MoE fallback tactic. |
| `2773a8edf4` | `4b03486e67` | Publish and restore Qwen3Next MTP ADP token metadata. |
| `77ec2e55b3` | `d1d35c8fde` | Make DeepGEMM chunking graph-safe and configurable. |
| `5e28b2d5a2` | `fb46819680` | Optimize FP8 block-scale MoE prefill activation. |
| `1f90ff8d30` | `df443cea09` | Size FP8 MoE activation backing independently. |

The optional FlashInfer v0.6.15 GDN T=1 cache-key fix is archived in
[`patches/flashinfer`](patches/flashinfer). Apply it only when the selected
toolchain image does not already contain that fix.

### Explicitly excluded

- The in-progress FP8 top-k10 group-4 combine experiment and EP64 validation.
- All files and results under TRTLLM3 `oak-combine-opt`.
- Uncommitted TRTLLM4 MTP experiments in attention, linear, model-engine, and
  Eagle3 code.
- Benchmark results, model weights, generated cubins, native libraries, build
  directories, Python environments, and caches from any other checkout.

In particular,
`cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu` is
unchanged from TRTLLM5 main.

## One-time preparation

Confirm the branch and initialize the repository-owned submodules:

```bash
cd /lustre/fsw/portfolios/coreai/users/williamj/TRTLLM5
git switch user/williamj/nvfp4-sweep-base
git submodule update --init --recursive
git status --short
```

Choose a development image with the compiler and dependencies required by the
TRTLLM5 commit. The following image is a known GB300/SM100 toolchain starting
point; its installed TensorRT-LLM package is not used as the runtime identity:

```bash
export TRTLLM5_BASE_IMAGE=/lustre/fsw/portfolios/coreai/users/williamj/containers/trtllm-sbsa-16694-base-20260731-built3.sqsh
export TRTLLM5_REPO=/lustre/fsw/portfolios/coreai/users/williamj/TRTLLM5
export TRTLLM5_HANDOFF=${TRTLLM5_REPO}/oak-nvfp4-sweep
```

Do not use the smaller `moe-opt-native-fix.sqsh` as the build toolchain: it is
a runtime image and does not contain `nvcc`. It remains suitable as a serving
base only after the TRTLLM5 native artifacts have been packaged separately.

## Clean local-development build

Submit the supplied build job:

```bash
sbatch ${TRTLLM5_HANDOFF}/build_trtllm5_dev.sbatch \
  ${TRTLLM5_BASE_IMAGE} ${TRTLLM5_REPO} clean
```

The job performs an SM100-only native build using:

```bash
python3 scripts/build_wheel.py \
  --clean \
  --use_ccache \
  --cuda_architectures 100-real \
  --skip_building_wheel \
  --linking_install_binary \
  --job_count 64 \
  --no-venv \
  --yes
```

It then creates a persistent `venv --system-site-packages` under this handoff
directory and installs TRTLLM5 in editable mode. The editable registration and
all generated native links resolve to `/opt/trtllm5-dev`, so every later
container must mount the host checkout at that exact path.

Build products remain in the TRTLLM5 checkout:

- `cpp/build/tensorrt_llm/libtensorrt_llm.so`
- `cpp/build/tensorrt_llm/thop/libth_common.so`
- `tensorrt_llm/libs/` links created by `--linking_install_binary`

No wheel or sqsh is produced by this workflow.

## Interactive local development

After the build succeeds, start an interactive allocation with both TRTLLM5
paths mounted at their stable container locations:

```bash
srun -A coreai_comparch_trtllm -p batch --qos=interactive \
  --nodes=1 --gres=gpu:4 --time=01:00:00 \
  --container-image=${TRTLLM5_BASE_IMAGE} \
  --container-mounts=${TRTLLM5_REPO}:/opt/trtllm5-dev,${TRTLLM5_HANDOFF}:/opt/trtllm5-handoff \
  --container-workdir=/opt/trtllm5-dev --pty bash

unset PYTHONPATH PYTHONHOME
source /opt/trtllm5-handoff/venv/bin/activate
python3 -c 'import pathlib, tensorrt_llm; print(pathlib.Path(tensorrt_llm.__file__).resolve())'
```

The printed path must start with `/opt/trtllm5-dev`. Python edits are live
immediately. After C++ or CUDA changes, leave the shell and run an incremental
build:

```bash
sbatch ${TRTLLM5_HANDOFF}/build_trtllm5_dev.sbatch \
  ${TRTLLM5_BASE_IMAGE} ${TRTLLM5_REPO} incremental
```

Use `clean` again after changing CMake inputs, submodule revisions, compiler
options, or CUDA architectures.

## Validation

Validate the persistent editable environment without modifying the base image:

```bash
sbatch ${TRTLLM5_HANDOFF}/validate_trtllm5_dev.sbatch \
  ${TRTLLM5_BASE_IMAGE} ${TRTLLM5_REPO}
```

The validation checks:

- Python imports resolve to `/opt/trtllm5-dev`;
- loaded `libth_common.so` resolves to the TRTLLM5 checkout;
- the two primary native libraries have no unresolved dynamic dependencies;
- no loaded TensorRT-LLM module comes from TRTLLM3, TRTLLM4, or a stock image;
- `trtllm-serve --help` works from the persistent development environment.

Run the focused lint and regression suite before using this checkout as a
benchmark baseline:

```bash
sbatch ${TRTLLM5_HANDOFF}/test_trtllm5_changes.sbatch \
  ${TRTLLM5_BASE_IMAGE} ${TRTLLM5_REPO}
```

The suite covers draft KV sizing, fallback tactic selection, MTP metadata
restoration, DeepGEMM chunking, and the associated MoE scheduler behavior.

## Isolation rules for the next Codex

1. Read the repository `AGENTS.md`, `CLAUDE.md`, and
   `CODING_GUIDELINES.md` before changing code.
2. Do not mount another TensorRT-LLM checkout into the same container.
3. Do not reuse another checkout's `cpp/build`, venv, ccache, TorchInductor,
   Triton, FlashInfer, or CUDA cache.
4. Do not set `PYTHONPATH` to TRTLLM3, TRTLLM4, `/workspace/TensorRT-LLM`, or
   another image-owned source tree.
5. Record branch, HEAD, source diff, image path, image SHA256, native-library
   SHA256, Slurm job ID, and nodes for every benchmark.
6. Keep NVFP4 sweep code separate from the excluded group-4 FP8 combine work
   until that experiment has completed and is explicitly selected for porting.
