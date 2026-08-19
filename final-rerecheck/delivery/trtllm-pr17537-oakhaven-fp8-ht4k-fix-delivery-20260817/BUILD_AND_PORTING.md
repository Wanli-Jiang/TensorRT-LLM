<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Build and porting procedure

## 1. Freeze the target identities

Before building or applying an optional compatibility patch, save:

```bash
git -C /path/to/TensorRT-LLM rev-parse HEAD
git -C /path/to/TensorRT-LLM status --short
git -C /path/to/flashinfer rev-parse HEAD
git -C /path/to/flashinfer status --short
sha256sum /path/to/flashinfer/flashinfer/gdn_kernels/gdn_decode_bf16_state.py
```

The preferred delivery branch already includes TensorRT-LLM propagation commit `f572594361` on top
of PR-stack head `9a6889b2a2`. Do not apply the TensorRT-LLM compatibility patch again. The old
application helper intentionally aborts unless TensorRT-LLM contains that base and the FlashInfer GDN
module has an audited source identity. Do not bypass the guard on a newer, unknown FlashInfer file.

The source changes are Python-only, but the delivery image must still be rebuilt as one immutable
artifact. Reusing a wheel while mounting source files makes package identity ambiguous.

## 2. Verify the branch and optional source changes

On the delivery branch, verify that both `9a6889b2a2` and `f572594361` are ancestors, then run the
focused hybrid cache-manager unit tests. The target checkout's `AGENTS.md`,
`CODING_GUIDELINES.md`, and component guides remain authoritative.

The FlashInfer static T=1 patch is optional for high-throughput reproduction. Apply it only when the
target revision matches its guarded source hash and the team wants the corresponding
correctness/specialization hardening. Dynamic GDN is a valid >4k configuration: latest
same-allocation C1536 runs reached 4049.545 and 4096.202 tok/s/GPU without the static overlay.

If committing the changes, use DCO sign-off:

```bash
git add <reviewed-files>
git commit -s -m "[None][fix] restore static GDN serving specialization"
```

Do not add AI attribution or co-authors. If pre-commit changes files, review and restage them.

## 3. Build ordinary packages

Use fresh, build-specific paths. For TensorRT-LLM PR #17537, inspect `scripts/build_wheel.py --help`
in the target checkout. The audited command shape is:

```bash
python3 scripts/build_wheel.py \
  --clean \
  --use_ccache \
  --cuda_architectures "103-real" \
  --build_dir /external/build-id/build/cpp \
  --dist_dir /external/build-id/wheel \
  --job_count CPUS \
  --yes
```

Use the exact architecture value supported by the target checkout/toolchain; record it rather than
blindly copying `103-real`. Do not use `--fast_build`, `--skip_building_wheel`,
`--linking_install_binary`, or an editable install for QA delivery.

Build/install FlashInfer using that revision's documented wheel procedure. Record the installed GDN
module SHA-256. If the optional static patch is used, require
`4982b5a9d20d9b18588020ab3e938238c9692ffab6265c3533f4b7cf8309a8fe`; otherwise record the dynamic
identity rather than treating it as a failure.

## 4. Build one self-contained runtime image

Install both ordinary wheels into a pinned runtime base. Remove any previous editable packages.
Embed a build manifest containing:

- TensorRT-LLM HEAD and patch SHA-256;
- FlashInfer HEAD and GDN module SHA-256;
- base-image digest;
- Python ABI, CUDA, PyTorch, TensorRT, driver floor, and target SM;
- native library hashes;
- Docker digest/archive hash and `.sqsh` hash.

From `/tmp` inside Docker and Pyxis/Enroot, with `PYTHONPATH` unset and no source mounts, verify:

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path
import tensorrt_llm
import flashinfer

gdn = Path(flashinfer.__file__).parent / "gdn_kernels/gdn_decode_bf16_state.py"
print("tensorrt_llm", tensorrt_llm.__file__)
print("flashinfer", flashinfer.__file__)
print("gdn_sha256", hashlib.sha256(gdn.read_bytes()).hexdigest())
PY
```

Reject `.pth`, `.egg-link`, `__editable__`, host-source dependency, missing `ldd` dependency, or
mixed native-library identities.

## 5. Warm caches without contaminating measurement

The optional static GDN fix intentionally creates separate T=1 variants for actual batch/layout buckets.
FlashInfer low-M BF16 `mm_bf16` may compile/tune its internally selected direct or
Split-K tactic on first use; CUDA graphs also have first-capture costs. This path is
separate from the DeepGEMM FP8 MoE backend.

- assign unique cache roots per job, node, and variant;
- finish all compilation/tuning before formal requests;
- capture/replay the full batch ladder, including B=1;
- preserve the cache manifest and hashes;
- do not let multiple ranks concurrently overwrite a shared JSON/cache file;
- keep the existing implementation as fallback outside measured low-M shapes.

Do not reuse a cache generated on another GPU generation or dependency revision.

## 6. Run and retain evidence

Store target-cluster configs, dry-run output, Slurm logs, raw SA-Bench JSON, cache manifests, identity
logs, and reports in the target TensorRT-LLM experiment directory. Keep srt-slurm unchanged.

Start with MTP3 C1536, repeat it on three independent allocations, then run no-MTP C3264. Use one
warmup wave and three formal waves. Only after those selected-point gates pass should a full
concurrency curve be submitted. See `ACCEPTANCE_CHECKLIST.md` and `PROMPT.md`.
