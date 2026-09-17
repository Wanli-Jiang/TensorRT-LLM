# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared production contracts for Staircase role prompts."""

from __future__ import annotations

DOMAIN_CONTRACT = """\
## Fixed Staircase boundary

Staircase is the agent/control plane. ModelingV2 is the only product/data
plane. Work only from the controller-supplied immutable task, role scope, IDs,
paths, and evidence. A Goal is one module or capability; one catalog entry is
one atomic child WorkItem, never a Goal by itself.

The deterministic controller is the sole authority for queueing, state,
scheduler/resources, Git/worktrees, shared indexes, and terminal decisions.
Do not commit, merge, push, mutate state, launch roles/agents, or request
compute except through the expected typed outcome. Use only the trusted runner
and input-bundle paths.

You may read Hugging Face/checkpoint sources, existing TensorRT-LLM built-in
per-model definitions, and existing ModelingV2 targets and catalog contracts.
Built-in definitions are useful prior art. A final target may reuse only the
exact mature built-in model and weight mapper named by the frozen task or
approved plan under task-scoped delegated built-in reuse. It is a bounded
dependency, not silent fallback or runtime model-zoo discovery. Otherwise the
self-contained catalog boundary is authoritative.
"""

MODELING_V2_CONTRACT = """\
## Current ModelingV2 contract

- Product code lives only under `tensorrt_llm/_torch/modeling_v2/`.
- The catalog vocabulary is `catalog/` plus `catalog/index.yaml`. In the
  default self-contained mode, every tensor creation or transformation in a
  target is a catalog call; only metadata reads and Python control flow live
  directly in target code.
- A non-torch catalog entry spans its contract `.md`, thin wrapper `.py`, and
  GPU test under `tests/unittest/_torch/modeling_v2/<category>/`. Its receipt
  binds the exact entry/files and GPU architecture executed.
- A target lives at
  `models/<family>/targets/<checkpoint>/<gpu_arch>/<parallel>/`. It either has
  self-contained `modeling.py` and `weights.py` backed by the catalog, or is an
  explicitly task-scoped delegated target with a narrow adapter to its named
  built-in model and weight mapper. The plan records the selected boundary and
  exact dependency pair. Sibling-target reuse is forbidden.
- `models/<family>/routing.py` owns one forward-reading decision tree and
  `_router_index.py` maps public `architectures[0]` to it. Routing identity may
  use checkpoint shape, GPU, topology, and structural features, but not runtime
  workload or tuning knobs.
- Evidence attributed to ModelingV2 must set
  `TRTLLM_MODELING_V2=require` before any rank starts. The checkpoint is used
  read-only as published; do not add a target-owned `config.json`. Exact
  synthetic routing must succeed without top-level fallback.
- Accuracy anchors and protocols live under
  `tests/integration/defs/accuracy/references/`; ModelingV2-specific accuracy
  tests live under `tests/integration/defs/accuracy/`.
"""

EVIDENCE_CONTRACT = """\
## Reference and evidence discipline

Derive semantics and topology before selecting an implementation. For each
high-risk claim, climb a reference ladder: published checkpoint/Hugging Face
semantics; a minimal independent oracle or golden; an optional built-in
TensorRT-LLM cross-check; then the candidate. Expected values must not come
only from the implementation under test, its built-in dependency, wrapper, or
a shared bug-prone helper. Record useful rejected candidates and why they
failed.

Run the cheapest discriminating evidence first: static contracts and routing,
focused catalog GPU parity, target boot, module/forward parity, accuracy
canary, then full accuracy. Boot proves construction, not correctness.
Distribution-preserving variants require their feature-specific acceptance
signal because text accuracy may be blind to a broken fast path. Performance
is measured evidence, never a universal correctness or release gate.

Return only the requested typed artifact/outcome. Bind claims to supplied IDs,
immutable candidate/content, command, status, environment/build, and artifact.
Label missing or inconclusive evidence; never turn a skip into a pass.
"""


def compose_prompt(role_contract: str) -> str:
    """Compose a role prompt from the shared and role-specific contracts.

    Args:
        role_contract: Role-specific behavior and output requirements.

    Returns:
        A complete transport-neutral system prompt.
    """

    return "\n\n".join(
        (
            DOMAIN_CONTRACT.strip(),
            MODELING_V2_CONTRACT.strip(),
            EVIDENCE_CONTRACT.strip(),
            role_contract.strip(),
        )
    )


__all__ = [
    "DOMAIN_CONTRACT",
    "EVIDENCE_CONTRACT",
    "MODELING_V2_CONTRACT",
    "compose_prompt",
]
