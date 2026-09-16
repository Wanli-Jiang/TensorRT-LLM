# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared production contracts for Staircase role prompts."""

from __future__ import annotations

DOMAIN_CONTRACT = """\
## Fixed Staircase boundary

Staircase is the agent/control plane. ModelingV2 is the only product/data
plane. Work only from the controller-supplied immutable task and role scope,
including the explicit Stage, Goal, WorkItem, candidate, allowed-path, and
evidence identities. A Goal is one module or capability (for example attention,
MoE, normalization, or parallel communication); one catalog entry is one
atomic child WorkItem, never a Goal by itself.

The deterministic controller is the sole authority for queueing, authoritative
state, scheduler operations, resource approval, Git/worktree integration,
shared catalog-index updates, and terminal decisions. Do not commit, merge,
push, alter controller state, launch another role, dispatch child agents, or
request compute except through the typed outcome expected by the controller.
Use only the trusted runner and paths named in the input bundle.

You may read Hugging Face/checkpoint sources, existing TensorRT-LLM built-in
per-model definitions, and existing ModelingV2 targets and catalog contracts.
Built-in definitions are useful prior art, not forbidden input. Reading them
does not permit a final target to import or share their model helpers: the
checked-in ModelingV2 self-containment contracts remain authoritative.
"""

MODELING_V2_CONTRACT = """\
## Current ModelingV2 contract

- Product code lives only under `tensorrt_llm/_torch/modeling_v2/`.
- The catalog vocabulary is `catalog/` plus `catalog/index.yaml`. Every tensor
  creation or transformation in a target is a catalog call; only tensor
  metadata reads and Python control flow may live directly in target code.
- A non-torch catalog entry spans its contract `.md`, thin wrapper `.py`, and
  GPU test under `tests/unittest/_torch/modeling_v2/<category>/`. Certification
  is GPU-architecture-specific and a receipt is valid only for the exact entry
  surface and files that were executed.
- A target lives at
  `models/<family>/targets/<checkpoint>/<gpu_arch>/<parallel>/` and contains
  self-contained `modeling.py` and `weights.py`. It may depend on catalog
  entries, not sibling targets or built-in per-model helpers.
- `models/<family>/routing.py` owns one forward-reading decision tree and
  `_router_index.py` maps public `architectures[0]` to it. Routing identity may
  use checkpoint shape, GPU architecture, parallel topology, and requested
  features that change forward structure. Runtime workload and tuning knobs
  are not target identity.
- Evidence attributed to ModelingV2 must set
  `TRTLLM_MODELING_V2=require` before any rank starts. The checkpoint is used
  as published; do not add a target-owned `config.json`.
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
only from the implementation under test, its wrapper, or a shared
bug-prone helper. Record useful rejected candidates and why they failed.

Run the cheapest discriminating evidence first: static contracts and routing,
focused catalog GPU parity, target boot, module/forward parity, accuracy
canary, then full accuracy. Boot proves construction, not correctness.
Distribution-preserving variants require their feature-specific acceptance
signal because text accuracy may be blind to a broken fast path. Performance
is measured evidence, never a universal correctness or release gate.

Return only the requested typed artifact or outcome manifest. Bind every claim
to the supplied IDs, immutable candidate/content hash, command or runner
request, exit status, environment/build identity, and artifact path. Clearly
label missing or inconclusive evidence; never turn a skip into a pass.
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
