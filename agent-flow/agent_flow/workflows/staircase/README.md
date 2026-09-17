<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Staircase workflow

`agent_flow.workflows.staircase` is a ModelingV2 specialization of the
existing AgentTeam/modeling-bringup workflow. It does not implement a separate
Slurm controller. Run it directly on a login node; Coder, Reviewer, and QA
submit resource-intensive validation with explicit `srun` or `sbatch`
commands.

## Role mapping

| AgentTeam role | Staircase responsibility |
|---|---|
| PlanDrafter | derives the model and creates ordered Smith, Assembler, and optional Tuner items |
| PlanReviewer | red-teams the derivation, dependencies, gates, and Slurm ceilings |
| Coder | adopts the active Smith, Assembler, or Tuner phase and performs the work |
| Reviewer | independently reviews and reruns the active phase's decisive evidence |
| QA | validates the closed Stage and hard correctness/accuracy/feature gates |

Smith, Assembler, and Tuner are domain phases within the existing Coder and
Reviewer loop. They are not additional orchestrator processes.

## Run from a login node

Install or update the editable AgentFlow package from this checkout, then copy
one example:

```bash
cd agent-flow
python -m pip install -e .

cp agent_flow/workflows/staircase/task.slurm.example.yaml /shared/tasks/my-model.yaml
$EDITOR /shared/tasks/my-model.yaml

staircase \
  --task /shared/tasks/my-model.yaml \
  --workspace /shared/staircase-runs/my-model
```

The wrapper automatically enables AgentTeam's post-QA replan cycle and uses a
five-iteration Coder context reset interval. Explicit generic AgentTeam flags
such as `--num-iterations`, `--plan-human-review`, `--build-human-review`,
`--feedback`, and `--clean` remain available.

Including `slurm-environment` in the task enables login-node Slurm guidance:

```yaml
slurm-environment:
  slurm_account: coreai_comparch_trtllm
  slurm_partition: batch
  slurm_qos: normal
  docker_image: /shared/containers/trtllm-local-dev.sqsh
```

The roles maintain `<workspace>/test_command.md` with complete, reusable
commands and observed results. The repository, checkpoint, references,
workspace, and build/cache paths should be mounted at identical absolute paths
inside the container. An existing local-dev build may be reused only after its
Python import, native libraries, source checkout, and image identity match.
Install and build work must not run on the login node.

## Parallel and multi-node Smith

The optional `smith` mapping sets ceilings:

```yaml
smith:
  max_parallel_jobs: 4
  max_nodes_per_job: 2
  max_total_nodes: 8
```

One Smith-mode Coder may submit several independent per-entry jobs and collect
them concurrently, or use several nodes for one collective probe. Each job
needs its own logs/artifacts and exact resource identity. Current AgentTeam
does not start multiple autonomous Smith agent sessions; extending the Python
orchestrator would be required for that stronger form of parallelism.

## Resume and intervention

AgentTeam checkpoints the generic role transition in `.agent_team_state.json`
and keeps the task, plan, acceptance criteria, progress, and status in the
workspace. Interrupt with Ctrl-C and rerun the same command to resume. To
course-correct a run:

```bash
staircase \
  --task /shared/tasks/my-model.yaml \
  --workspace /shared/staircase-runs/my-model \
  --feedback /shared/tasks/reviewer-feedback.txt \
  --trigger-replan-with-feedback
```

The stored workspace task is authoritative on resume and is used to select
local versus Slurm prompts.

## Task contract

Required inputs are the source-model code, checkpoint, TensorRT-LLM checkout,
and target identity (`family`, `checkpoint`, `gpu_arch`, and `parallel`). An
accuracy anchor is optional but should include benchmark, score, tolerance,
and provenance. Unknown task keys are preserved for the agents, matching
modeling-bringup's extensible task style.

Packaged examples:

- `task.example.yaml`: local/development shape;
- `task.slurm.example.yaml`: login-node orchestration with Slurm execution;
- `references/modeling_v2_contract.md`: compact product boundary.

## Claim boundary

This mode inherits modeling-bringup's strengths and limitations. It provides
the established AgentTeam lifecycle, workspace resume, human feedback, and
agent-authored Slurm execution. It does not provide scheduler-free agent
images, controller-owned queue admission, typed attempt receipts, isolated
candidate worktrees, automatic exact-job recovery, or deterministic fan-in.
Therefore a successful run can provide onboarding evidence, but it must not be
described as certification of the previously proposed isolated Slurm control
plane.
