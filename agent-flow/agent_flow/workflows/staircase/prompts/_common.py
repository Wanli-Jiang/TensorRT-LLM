# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared domain and execution guidance for Staircase prompts."""

from __future__ import annotations

DOMAIN_CONTRACT = """\
## Staircase and ModelingV2 boundary

Staircase is an AgentTeam specialization for bringing up and tuning
TensorRT-LLM ModelingV2 targets. Product code lives under
`tensorrt_llm/_torch/modeling_v2/`; do not create an `_torch/staircase` data
plane or an alternate runtime flag. Evidence attributed to ModelingV2 sets
`TRTLLM_MODELING_V2=require` before every participating rank starts.

You may read the checkpoint's Hugging Face source, built-in TensorRT-LLM
per-model definitions, existing ModelingV2 targets, and catalog contracts.
Built-in models are useful implementation references, not forbidden input.
The final ModelingV2 target must still satisfy its checked-in
self-containment and catalog-vocabulary contracts.

A non-torch catalog entry consists of its contract `.md`, thin wrapper `.py`,
and entry-specific GPU test. A target lives at
`models/<family>/targets/<checkpoint>/<gpu_arch>/<parallel>/` and contains
self-contained `modeling.py` and `weights.py`. Routing belongs in the family
`routing.py` and `_router_index.py`; workload and tuning knobs are not target
identity.
"""

ROLE_MAPPING = """\
## Staircase phases inside the AgentTeam lifecycle

The Python orchestrator remains the existing five-role AgentTeam workflow:
PlanDrafter, PlanReviewer, Coder, Reviewer, and QA. Smith, Assembler, and
Tuner are domain phases, not extra orchestrator processes:

- a `[Smith]` plan item makes the current Coder analyze, verify, or onboard
  exactly one catalog entry, and makes Reviewer apply the Smith checklist;
- an `[Assembler]` item makes Coder assemble one target/routing/weight slice,
  and makes Reviewer apply the Assembler checklist;
- a `[Tuner]` item makes Coder test one performance hypothesis with one
  changed variable, and makes Reviewer check comparability and correctness.

PlanDrafter owns this decomposition, PlanReviewer validates it, and QA checks
the closed Stage. Keep the active phase and item visible in `status.md` so a
context reset or resume does not silently switch roles. Do not claim that the
workflow ran multiple autonomous Smith agents: one Coder session coordinates
the work unless the AgentTeam implementation itself is extended later.
"""

EVIDENCE_CONTRACT = """\
## Reference and evidence discipline

Derive checkpoint semantics and parallel topology before choosing an
implementation. Expected values must not come only from the candidate, its
wrapper, or a shared helper. Use the cheapest discriminating ladder: static
contracts and routing, focused catalog GPU parity, target boot, module or
forward parity, accuracy canary, full accuracy, and feature-specific signals.
Boot is not correctness. A skipped, CPU-only, wrong-architecture, fallback,
or stale run is missing evidence rather than a pass.

Bind each reported result to the exact command, repository/build identity,
checkpoint, GPU architecture, topology, exit status, and artifact path.
Accuracy, routing, self-containment, catalog coverage, and requested feature
signals are hard gates. Performance cannot compensate for a correctness
failure.
"""

SLURM_LOGIN_NODE_EXECUTION = """\
## Login-node orchestration and Slurm execution

This workflow itself runs on the Slurm login node, matching the existing
`modeling-bringup` operating model. Read `slurm-environment` from `task.yaml`.
Use its `slurm_partition` and `docker_image` verbatim, and include
`slurm_account` and `slurm_qos` when present. Agent reasoning and AgentTeam
checkpointing stay on the login node; GPU, multi-rank, long-running, or
resource-intensive commands run through explicit `srun` or `sbatch` jobs.
Do not silently fall back to a local GPU or CPU-only substitute.

Do not install packages or rebuild TensorRT-LLM on the login node. Bind-mount
the repository, checkpoint, references, workspace, and build/cache paths at
the same absolute paths in the container. Reuse the existing local-dev build
when its Python import, native libraries, source checkout, and container image
all match the task. Rebuild inside an allocated node/container only when the
change requires it or the identity check fails. Never download an agent SDK in
a compute job.

Before a new resource shape is trusted, check the effective account,
partition, QoS, nodes, GPUs, time limit, image, and mounts; use
`sbatch --test-only` when available. Scheduler acceptance proves only that the
request is admissible. It does not prove the test, model, or evidence passed.
"""

TEST_COMMAND_CACHE = """\
## Shared `test_command.md`

Coder, Reviewer, and QA share `<workspace>/test_command.md` as a cache of
verified, self-contained Slurm commands. Each entry records purpose, exact
`srun` or `sbatch` command, account/partition/QoS, nodes/tasks/GPUs/time,
container image and mounts, working directory, environment, expected output,
artifact/log paths, and the last observed result. A command is reusable only
when those identities still match. Reviewer and QA independently rerun the
load-bearing command; they do not accept Coder prose as evidence.

For Smith, `task.yaml` may bound `max_parallel_jobs`, `max_nodes_per_job`, and
`max_total_nodes`. One Coder may submit multiple independent per-entry catalog
analysis or validation jobs concurrently and then collect every result. A
single vertical collective probe may request multiple nodes. Stay within all
bounds and keep one artifact/log namespace per entry and job. This is
parallel Slurm work coordinated by one Smith-mode Coder, not parallel
autonomous Smith agent sessions.
"""


def compose_extension(role_contract: str) -> str:
    """Compose one role extension from shared Staircase contracts."""
    return "\n\n".join(
        (
            DOMAIN_CONTRACT.strip(),
            ROLE_MAPPING.strip(),
            EVIDENCE_CONTRACT.strip(),
            role_contract.strip(),
        )
    )


__all__ = [
    "DOMAIN_CONTRACT",
    "EVIDENCE_CONTRACT",
    "ROLE_MAPPING",
    "SLURM_LOGIN_NODE_EXECUTION",
    "TEST_COMMAND_CACHE",
    "compose_extension",
]
