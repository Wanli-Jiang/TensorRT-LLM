<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ModelingV2 contract used by Staircase

The in-tree `tensorrt_llm/_torch/modeling_v2/README.md` and its tests are
authoritative. This file is a compact role-facing map.

## Product boundary

```text
tensorrt_llm/_torch/modeling_v2/
  _router_index.py
  catalog/index.yaml
  catalog/<category>/<entry>.{md,py}
  models/<family>/routing.py
  models/<family>/targets/<checkpoint>/<gpu_arch>/<parallel>/
    modeling.py
    weights.py

tests/unittest/_torch/modeling_v2/
tests/integration/defs/accuracy/test_modeling_v2_*.py
tests/integration/defs/accuracy/references/
```

Staircase must not create a second model tree, runtime flag, routing registry,
test hierarchy, or receipt vocabulary. Agents may inspect Hugging Face,
built-in TensorRT-LLM per-model implementations, and existing ModelingV2 code.
The final target remains self-contained: it may use catalog entries but not
import helpers from a built-in model or a sibling target.

## Smith

One Smith plan item owns one catalog entry. A non-torch entry includes its
contract, thin wrapper, focused GPU test, and architecture-specific observed
evidence. Independent entries may be probed concurrently through bounded
Slurm jobs, coordinated by the single AgentTeam Coder. Multi-node collectives
need explicit node/rank evidence. Scheduler placement alone does not certify
the operation.

## Assembler

Assembler consumes reviewed catalog surfaces to produce a flat target,
weights, and routing changes. Every tensor computation in the target belongs
to the catalog vocabulary. Checkpoint shape, GPU architecture, structural
features, and parallel topology may select a target; batch shape, sequence
length, and tuning knobs may not.

## Tuner

Tuner changes one declared variable, holds checkpoint/workload/topology/build
and correctness gates fixed, and reports uncertainty as well as measured
delta. A faster incorrect or fallback run fails.

## Runtime and evidence

Set `TRTLLM_MODELING_V2=require` before ranks start. Use independent source or
golden references, then focused GPU parity, boot, forward/generation parity,
accuracy, and feature-specific signals. Skips, stale artifacts, wrong GPU
architecture, and fallback routing are not passing evidence.

Staircase follows the current `modeling-bringup` operational model: AgentTeam
runs on the login node and its roles submit resource-intensive work through
self-contained `srun` or `sbatch` commands. This provides AgentTeam checkpoint
and human-feedback recovery, not a typed Slurm controller. Agents retain
scheduler access, authoritative Stage/Goal detail remains in Markdown, and
parallel Smith means concurrent Slurm work coordinated by one Coder rather
than concurrent autonomous agent sessions.
