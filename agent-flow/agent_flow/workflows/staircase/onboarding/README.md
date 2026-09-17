<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Staircase onboarding

Onboarding creates and validates one ModelingV2 deployment target for an
explicit checkpoint, GPU architecture, structural feature set, and parallel
topology. The target is self-contained/catalog-backed by default, or may use
task-scoped delegated built-in reuse of one exact model and weight mapper when
the frozen task or approved plan says so. Staircase is the control plane; product code
stays in `tensorrt_llm/_torch/modeling_v2/` and its existing test surfaces.

## Prepare a schema-v2 task

Copy [`task.example.yaml`](task.example.yaml) and replace every placeholder.
The example shows the credentialed deployment mode; see the credential section
below before launching it. The task owner must supply facts that agents cannot
safely infer:

- canonical repository root, exact 40-character base commit, dirty policy,
  and a non-overlapping shared workspace root;
- checkpoint path/provenance, public `architectures[0]`, and any independent
  reference sources;
- family/checkpoint/SM identity, full TP/PP/MoE-EP/MoE-TP/attention-DP
  mapping, requested structural features, and expected ModelingV2 route;
- repository-relative pytest selectors for boot, component, collective,
  accuracy, and feature-signal gates, including the exact accuracy reference,
  protocol, and tolerance;
- explicit certification mode (`SYNTHETIC` or `REAL` for Slurm; `LOCAL` only
  through the CLI local override) and, for `REAL`, exact
  product/build/image/device/collective/transport identity;
- Slurm controller account, partition, optional QoS/reservation, image,
  mounts, build identity, `scheduler_clients: true`,
  `container_launch_mode: in_allocation_srun`, nested-submission policy,
  timeouts, and requeue policy;
- a distinct `agent_worker` image/build identity with
  `scheduler_clients: false` and
  `network_enforcement: certified_worker_image`;
- all seven role classes (`plan_drafter`, `plan_reviewer`, `smith_coder`,
  `assembler_coder`, `tuner_coder`, `reviewer`, and `qa`), each bound to a
  single-node/single-task resource, `backend_api_only`, non-secret environment,
  and one admitted credential mode;
- horizontal Smith concurrency and aggregate node/GPU caps, plus
  `distinct_nodes_required: true` with `exclusive: true` when each role wave
  must occupy distinct physical nodes; the closed
  `coder_analysis`, `exploratory_probe`, `deterministic_gate`,
  `reviewer_analysis`, and `reviewer_rerun` resource classes; per-item
  escalation bounds; and infrastructure retries;
- `diff_only` or `signed_commits` delivery, always with merge and push disabled.

Resource classes inherit the controller `partition` and `qos` unless they
declare an override. A site whose CPU QoS and GPU QoS are disjoint can use the
following shape for a one-rank target:

```yaml
execution:
  slurm:
    controller:
      partition: cpu
      qos: cpu-short
    role_classes:
      # Keep each role's existing network/environment/credential policy.
      qa:
        resource_class: reviewer_analysis
    smith:
      resource_classes:
        coder_analysis:
          nodes: 1
          tasks_per_node: 1
          gpus_per_node: 0
          cpus_per_task: 8
          memory: 32GiB
          time_limit: 01:00:00
        deterministic_gate:
          nodes: 1
          tasks_per_node: 1
          gpus_per_node: 4
          gpu_allocation_padding: true
          partition: batch
          qos: normal
          cpus_per_task: 16
          memory: 64GiB
          time_limit: 01:00:00
```

The example omits the unchanged role policy bodies. Here `target.world_size`
must be `1`. Exactly one gate task participates and sees one GPU; the other
three GPUs are allocation padding required by the site QoS. Padding is rejected
for agent role resources, multi-rank targets, multi-node requests, or an exact
one-GPU request. Mapping QA to the CPU `reviewer_analysis` class is appropriate
when QA audits immutable gate receipts and does not execute a GPU test itself.

All repository, checkpoint, image, mount, and workspace paths are canonical
host paths visible where the task declares them. Gate fields contain selectors,
not `pytest ...` command strings or acceptance prose. The selected accuracy
test owns the evaluation harness and must bind its receipt to the frozen
reference, protocol, and tolerance.

Controller mounts are identity-only: every normalized `container_path` equals
its canonical `host_path`, and the mount set must cover every required
controller-visible repository, workspace, checkpoint, and reference path with
the required access. Staircase does not translate frozen task paths.

The controller and worker images are separate trust domains. Only the
controller image may contain Slurm clients. The task-bound worker policy and
its SHA-256 are checked again in each worker; `certified_worker_image` means
the site certifies that image/build as enforcing `backend_api_only`. Staircase
does not create a network namespace or firewall inside the worker process.
Controller environment entries with credential-like names or recognizable
secret-like values are rejected before normalized task persistence.

Each attempt exposes only its exact immutable `input.json` or
`role-input.json` as a read-only file and its sibling `output/`/evidence tree
as the writable mailbox. Authoritative credential descriptors and revocation
ledgers remain controller-owned. Code, checkpoint, and references are
read-only; only a Coder's exact candidate overlay is separately writable.

The loader batches validation errors, normalizes the task once, and persists a
SHA-256 digest. Reusing a workspace with different normalized semantics fails
closed. Use a new workspace for a changed target or plan envelope unless a
state migration explicitly accepts it.

## Choose one credential mode

For an externally brokered deployment, keep the example's shape:

- every one of the seven role policies is byte-identical after normalization;
- the credential names are allowed for the selected backend (exactly one of
  `CODEX_AUTH_JSON` or `OPENAI_API_KEY` for `codex`, or the supported
  Anthropic names for `claude-code`);
- the broker ID is not `preauthenticated`; and
- `controller.credential_broker_socket` is an absolute private AF_UNIX path;
- `controller.source_auth_json_path` names one canonical, same-UID, singly
  linked mode-`0400`/`0600` file mounted read-only at the exact same path; no
  broader controller mount may cover it.

The controller starts the packaged broker sidecar before client preflight and
always shuts it down. Use a short private controller-local path such as
`/tmp/staircase-credential-broker-qwen/broker.sock` when the shared workspace
path would exceed the AF_UNIX limit. Only the broker reads the source value;
controller state and worker manifests contain handles and descriptors only.

`CODEX_AUTH_JSON` carries one bounded, strict Codex `auth.json` document inside
the attempt-bound bundle; it is not an environment-variable handoff to the
agent. The worker creates a mode-`0700` temporary `CODEX_HOME`, writes only a
mode-`0600` `auth.json`, invokes the agent once, validates any refreshed JSON,
redacts refreshed token leaves, and removes the temporary home. Do not request
`CODEX_AUTH_JSON` and `OPENAI_API_KEY` together.

For a site where the agent backend is already authenticated without an
environment credential, change all seven policies to:

```yaml
credential_broker:
  broker_id: preauthenticated
  allowed_credential_names: []
  per_attempt_ttl_seconds: 14400
```

and remove both `controller.credential_broker_socket` and
`controller.source_auth_json_path`. Mixing preauthenticated and credentialed
role policies is rejected.

## Validate and launch

Create a strict preflight fact manifest for the exact launcher selection. It
names the repository, workspace root, container image, build identity, and
mounts; Staircase separately observes the Git state, command availability, and
phase-bound nested-`sbatch` capability.

```bash
staircase onboard \
    --task /shared/tasks/qwen3.yaml \
    --workspace /shared/staircase-runs/qwen3-sm100-tp8 \
    --execution slurm \
    --preflight-facts /shared/tasks/qwen3.preflight.json
```

The login process submits one deterministic controller. The controller repeats
preflight in its allocation and owns all later submissions. There is no login-
node or local fallback when nested submission, the shared mount, the broker, or
the declared build identity is unavailable.

For every containerized controller or worker, `sbatch` requests only the bare
allocation/resources. The fixed submitted script then `exec`s one
in-allocation
`srun --nodes=1 --ntasks=1 --overlap --cpu-bind=none --no-container-mount-home`
carrying the validated `--container-image`, optional `--container-mounts`, and
fixed internal argv.
Disabling the inner step's affinity binding does not relax the outer
allocation's cgroup CPU limits or GPU GRES/device isolation.
The validated non-secret environment and controller-owned agent launch-policy
digest are repeated in the fixed inner argv, so worker admission does not
depend on Pyxis propagating the batch-job environment.
For admitted single-rank GPU allocation padding, that fixed `srun` additionally
uses `--gpus-per-task=1`.
Pyxis flags never appear on `sbatch`; user task text cannot inject shell or
scheduler fragments, and the inner process exit status becomes the batch-job
exit status.

`--execution local` exists only for CPU/fake development. It cannot certify
Slurm policy, credentials, multi-node placement, GPU architecture, catalog
receipts, or target gates.

Submission and recovery are exact-token and append-only for launcher,
planning, generic-worker, and successor jobs. A unique owned match is adopted.
By default, an accepted call that disappears from queue/accounting enters
durable manual recovery. A second/final submit or cancellation of a no-job
intent requires site-certified accounting retention plus bounded,
time-separated absence samples.

`staircase status` only projects controller/child owned observations, lease,
checkpoint, launcher/successor, and mailbox drift; it never adopts,
quarantines, mutates state, or exposes credential values. Dead-controller
emergency cancellation holds the lease lock, reconciles hidden `SUBMITTING`
intents, revokes exact planning/Coder/Reviewer/QA credentials, cancels exact
owned children, and writes its full receipt last. Ambiguity or broker failure
requires manual reconciliation.

## Onboarding lifecycle

1. PlanDrafter derives checkpoint semantics, route identity, topology,
   capabilities, independently QA-able Stages, module/capability Goals, atomic
   WorkItems, references, gates, and resource classes. It records whether the
   target is self-contained or uses a bounded, named delegated model and weight mapper.
2. PlanReviewer accepts that exact immutable plan digest or requests a bounded
   new revision. Rejected or superseded revisions never dispatch.
3. For a self-contained target, the controller forms bounded horizontal Smith
   waves. Each Smith Coder analyzes, verifies, or onboards exactly one catalog
   entry in an isolated overlay. Delegated targets need Smith items only for
   catalog surfaces they actually own; delegation is not fabricated catalog work.
4. Each candidate passes ordered deterministic gates, fresh Reviewer analysis,
   and a fresh Reviewer rerun pinned to the same candidate and overlay digest.
5. A modifying Smith item passes independent QA against that candidate and the
   controller-ingested gate digests. Reviewer and QA recompute the frozen
   overlay digest before backend invocation and fail closed on mismatch.
6. The controller alone integrates approved siblings, consumes semantic index
   deltas, updates `catalog/index.yaml`, and records the new integration head.
7. Assembler implements one target core, feature, weight, or routing item. The
   self-contained path uses integrated catalog dependencies and turns a gap
   into a Smith item. The delegated path creates only a narrow adapter to the
   exact authorized mature model and weight mapper; undeclared reuse, sibling imports,
   and fallback are rejected. Both paths follow the same gate, Reviewer-rerun,
   and required-QA lifecycle before controller integration.
8. After all required Goals integrate, the controller projects exact per-item
   QA receipts into the Stage verdict; only it can advance `READY_FOR_QA` to
   `QA_PASSED` and then `CLOSED`.
9. The controller emits a verified diff-only patch or an isolated DCO-signed
   branch and a terminal report. It never merges or pushes.

A successful Coder result scans declared changed-file bytes for exact injected
secrets and central secret patterns while computing its content digest.
Controller import and staging bind that digest, reject undeclared changes, and
fail closed on symlinks, special files, hard links, races, or digest drift.

Horizontal Smith work can occupy several independent single-node allocations
under aggregate limits. With `distinct_nodes_required: true`,
`exclusive: true` is mandatory. Every horizontal Coder, Reviewer-analysis, and
Reviewer-rerun attempt gets its own exclusive one-node allocation and writes
`items/<item>/attempts/<sequence:04d>/placement-receipt.json`. The receipt binds
the exact run/item/attempt/task/generation and Slurm job/array/cluster identity
to one observed hostname/`NodeList`, exclusivity, and a content digest.

Before fan-in, the controller validates distinct hosts separately within the
approved plan's Coder, Reviewer-analysis, and Reviewer-rerun role waves. It
waits without writing fan-in intent for incomplete active peers and blocks on
terminal missing evidence, receipt/identity/exclusivity mismatch, or duplicate
hosts within a role wave. Cross-stage node reuse is allowed; stage-level QA is
not part of this proof. A role stage never launched for an already-terminal
peer is omitted because there is no job to prove.

Vertical catalog validation is different: it uses one deterministic-gate
attempt, one multi-node allocation, one immutable `srun` rank plan, exact
world-size/topology checks, and collective rank-to-node receipts. A synthetic
body proves runner plumbing only; `REAL` evidence requires the real ModelingV2
product body and exact product identity.

Known scheduler observations validate cluster/job/array, user, generated job
name, comment, and token. Collective launch resolves absolute immutable
`srun` and rank-side `scontrol` executables from non-worker-owned,
non-writable trust paths and invokes fixed argv with `shell=False`.

## Product acceptance boundary

Every rank of evidence attributed to ModelingV2 must start with:

```bash
export TRTLLM_MODELING_V2=require
```

`auto` may fall back to the built-in model and therefore cannot support a
ModelingV2 claim. A delegated target still resolves its exact synthetic class
first; its explicit internal dependency is not top-level routing fallback.
Expected outputs for high-risk modules require an independent reference or
extra cross-check, not only the wrapper or implementation under test.

Use the evidence ladder in
[`../references/modeling_v2_contract.md`](../references/modeling_v2_contract.md):
static contract/routing checks, catalog parity or exact delegation-boundary
checks, boot, module/forward parity, accuracy canary/full accuracy, and
feature-specific acceptance. Boot proves construction, not correctness. Every
target requires exact no-fallback synthetic routing, a read-only published
checkpoint, real boot and generation, independent reference evidence,
Reviewer, and QA. Performance cannot override correctness, routing, the
selected implementation boundary, accuracy, or feature-signal failure.

Onboarding is complete only when the terminal report contains the receipts
required by the task's certification mode, target identity, and topology. CPU
tests, a fake scheduler, a synthetic multi-node runner, a broker-client test,
or an editable build smoke are development evidence, not live-agent/backend,
broker-daemon, collective, or REAL-model GPU correctness, accuracy, or
performance certification. The external broker daemon/socket, backend egress,
accounting-retention policy, live backend, and real product tests require site
certification. A blocked live Slurm canary is not a successful result.
