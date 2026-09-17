<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Staircase workflow

`agent_flow.workflows.staircase` is the Slurm-first agent/control plane for
bringing up and tuning TensorRT-LLM ModelingV2 targets. Staircase owns plans,
agent attempts, scheduling, evidence, recovery, and delivery. Product code and
tests remain in `tensorrt_llm/_torch/modeling_v2/` and its existing test trees;
there is no second `_torch/staircase` data plane.

## Production architecture

```text
login node: validate, submit one controller, observe
  `- deterministic Slurm controller (no agent SDK session)
       |- immutable task + generation-fenced authoritative state
       |- PlanDrafter -> PlanReviewer
       |- Smith Coder / Assembler Coder / Tuner Coder
       |- deterministic gates -> Reviewer analysis -> Reviewer rerun
       |- mandatory QA where selected (always for Tuner)
       |- serial controller-only Git/catalog fan-in
       `- diff-only patch or isolated signed branch + terminal report
```

The normal AgentFlow lifecycle remains authoritative. Smith, Assembler, and
Tuner are pinned Coder domain profiles, not recursive orchestrators. The task
binds every LLM invocation to exactly one of seven typed Slurm role classes:

1. `plan_drafter`
2. `plan_reviewer`
3. `smith_coder`
4. `assembler_coder`
5. `tuner_coder`
6. `reviewer`
7. `qa`

Each role class is single-node, single-task, uses the closed
`backend_api_only` network policy, and has an explicit non-secret environment
and credential policy. A fresh process/session handles every attempt. The
controller never initializes an agent SDK and does not share a persistent
agent session across items.

The controller and agent workers use deliberately different images. The
controller image/build includes Slurm clients and sets
`scheduler_clients: true` because it owns nested submission. Every agent role instead uses the
single task-level `execution.slurm.agent_worker` image and build identity,
which must differ from the controller and declares `scheduler_clients: false`.
That exact worker-image policy is digest-bound into each role manifest and
exported to the job as controller-owned `STAIRCASE_AGENT_POLICY_DIGEST` for
worker verification. `network_enforcement: certified_worker_image` means the site
certifies that immutable image/build as enforcing `backend_api_only`; it is a
deployment trust assertion, not an in-process network namespace or firewall
implemented by Staircase.

The Coder role is specialized by the Smith, Assembler, or Tuner profile. The
shared Reviewer role receives the same domain profile's review checklist but
runs in a different immutable snapshot and session. QA is a separate role and
cannot reuse either Coder or Reviewer state.

The controller is the sole writer of authoritative state and the sole owner
of queue admission, scheduler submissions, exact job identities, resource
approval, candidate binding, Git integration, catalog-index updates,
cancellation, tuning promotion, and terminal decisions. Workers receive an
immutable input bundle, write an atomic result, and cannot submit jobs or
mutate controller state, shared Git metadata, or the catalog index.

## Public commands

Production starts require independently observed preflight facts:

```bash
staircase onboard \
    --task /shared/tasks/model.yaml \
    --workspace /shared/staircase-runs/model-sm100-tp8 \
    --execution slurm \
    --preflight-facts /shared/tasks/model.preflight.json

staircase tune \
    --task /shared/tasks/model-tune.yaml \
    --workspace /shared/staircase-runs/model-tune-001 \
    --execution slurm \
    --preflight-facts /shared/tasks/model-tune.preflight.json

staircase status  --workspace /shared/staircase-runs/model-sm100-tp8
staircase status  --workspace /shared/staircase-runs/model-sm100-tp8 --json
staircase cancel  --workspace /shared/staircase-runs/model-sm100-tp8 \
    --reason "operator request"
staircase respond --workspace /shared/staircase-runs/model-sm100-tp8 \
    --request-id request-g1-0123456789ab --response-file response.txt
```

The preflight manifest identifies the repository, shared workspace,
controller image/build, and exact mounts; the normalized task separately
binds the distinct agent-worker image/build. Staircase independently checks
the repository HEAD and dirty state, command availability, and a phase-bound
`sbatch --test-only` nested-submission probe. The controller repeats the
production check from inside its allocation. `nested_submission` is the only
admitted dispatch topology; dispatcher and preallocated-pool modes fail schema
validation.

Controller mounts are identity-only. Every normalized `container_path` must
equal its canonical `host_path`, and the mount set must cover the repository,
workspace, checkpoint, and additional references with the required access.
There is no host-to-container path translation of frozen task identities.

Container launch is two-level and fixed. The trusted scheduler adapter invokes
`sbatch` with allocation/resource flags only and submits a generated script on
stdin. Inside the allocation that script uses `exec srun` with fixed
`--nodes=1 --ntasks=1 --overlap --cpu-bind=none`, the validated
`--container-image` and `--container-mounts`, and the fixed internal controller
or worker argv. Disabling the inner step's affinity binding does not relax the
outer allocation's cgroup CPU limits or GPU GRES/device isolation. Pyxis flags
are not placed on `sbatch`, user text cannot add shell fragments or scheduler
flags, and `exec` preserves the `srun`/container process exit status as the
batch-job exit status.

Each resource class may explicitly override the controller's `partition` and
`qos`; omitted values continue to inherit the controller defaults. This lets a
CPU controller and agent roles use a CPU partition while deterministic GPU
gates use a GPU partition without giving workers scheduler control. A
single-rank `deterministic_gate` may also set
`gpu_allocation_padding: true` and request more than one GPU when a site QoS
requires a minimum GPU allocation. This narrow exception is admitted only for
one rank on one node. The inner `srun` then adds `--gpus-per-task=1`, so the
extra GPUs satisfy allocation policy but never become ranks or visible gate
devices. Without the explicit field, rank and GPU totals remain exact.

`--execution local` is a CLI override for CPU/fake-scheduler development only:

```bash
staircase onboard --task task.yaml --workspace /tmp/staircase-test \
    --execution local
```

Local execution does not prove shared mounts, nested submission, credentials,
GPU identity, rank placement, or ModelingV2 correctness. Import and every
`--help` path remain free of SDK startup, scheduler calls, and GPU probes.

## Task, state, and recovery

Copy and edit one packaged schema-v2 example:

- [`onboarding/task.example.yaml`](onboarding/task.example.yaml)
- [`tuning/task.example.yaml`](tuning/task.example.yaml)

The loader rejects unknown keys, duplicate YAML keys, unsafe paths and slugs,
non-finite numbers, inconsistent topology/resource totals, obsolete data-plane
names, and implicit merge or push. It canonicalizes the task once and persists
its SHA-256 digest. Resume uses that frozen task; a semantic change requires a
new workspace unless an explicit state migration exists.

Durable state records Stages, Goals, WorkItems, Attempts, exact Slurm job IDs,
submission tokens, integration history, controller generation, owner nonce,
heartbeat, and lease. The launcher writes a frozen submission intent before
submitting and can adopt the exact job by token after a crash instead of
creating a duplicate.

Launcher, planning, generic-worker, and successor submissions share one
append-only, exact-token recovery journal. A unique scheduler match is adopted.
The default Slurm adapter does not treat an invisible historical job as absent:
it persists a durable manual-recovery boundary. A second and final automatic
submit, or cancellation of a token-only/no-job intent, is permitted only when
the site certifies queue/accounting retention and the journal records bounded,
time-separated clean absence samples after the visibility grace period.

Controller advance or preemption signals stop new dispatch, checkpoint state,
and, when configured, submit a dependency-fenced successor generation rather
than asking Slurm to requeue the same process. A successor cannot acquire the
lease until accounting proves the exact predecessor terminal. It then adopts
every active old-generation attempt by its persisted token/job identity before
dispatching new work and revokes credentials for non-adopted attempts.

An unexpected dead controller can be recovered by restarting `onboard` or
`tune` against the same task/workspace. Recovery requires terminal accounting,
exact controller ownership, and no live lease before it submits a successor.
Human input is also detached: `WAITING_FOR_INPUT` is checkpointed, `respond`
publishes a write-once response, and a successor consumes it; no controller
waits on stdin.

`status` is a read-only projection over authoritative state, exact owned
controller/child scheduler observations, lease lock/record, checkpoints,
launcher/successor receipts, and result/human-input mailbox drift. It never
adopts or quarantines work, mutates state, or emits credential values.

`cancel` first writes a durable cancellation request for a live controller.
Dead-controller emergency cleanup holds the lease lock while it revalidates
state, lease, controller ownership, and terminal accounting; reconciles hidden
`SUBMITTING`/no-job intents by their append-only journals; and preflights every
exact child identity. It then revokes exact planning, Coder, Reviewer, and QA
credential handles before cancelling exact owned child jobs, and writes the
full idempotent receipt last. Ambiguous scheduler evidence, uncertified absence,
or broker/receipt failure stops with manual reconciliation and no false cleanup
receipt. Names, ranges, wildcards, and user-wide cancellation targets are never
used.

## Credentials and worker isolation

Credential configuration has exactly two admitted deployment modes:

- **Preauthenticated:** every role uses broker ID `preauthenticated`, every
  `allowed_credential_names` list is empty, and
  `controller.credential_broker_socket` is omitted. No environment credential
  is materialized.
- **Credentialed:** all seven role classes use byte-identical normalized
  broker policy, name only credentials allowed for the selected backend, and
  configure one absolute `controller.credential_broker_socket` plus an exact
  private `controller.source_auth_json_path`. The broker ID cannot be
  `preauthenticated`.

The credentialed mode is a narrow controller-local trust boundary. The task
mounts only the source auth file, read-only at the same absolute path; no
broader mount may cover it. The controller starts the packaged bounded AF_UNIX
broker in a private mode-`0700` runtime directory before client preflight and
always shuts it down. The source value never enters controller environment,
arguments, state, evidence, logs, or reports. Canonical length-framed requests
exchange only public attempt bindings, opaque handles, paths, and receipts.
The broker materializes an owned mode-`0600` bundle below the
controller-private credential root; workers receive it at a fixed read-only
mount only after admission. Terminal, cancellation, replacement,
generation-fence, and expired-orphan paths revoke the exact handle.

For a Codex backend, a credentialed policy selects exactly one transport:
`OPENAI_API_KEY` or `CODEX_AUTH_JSON`; combining them is rejected. The OAuth
form is still delivered by the same exact-attempt broker bundle. A worker
strictly validates the bounded JSON, writes it only to a private temporary
`CODEX_HOME/auth.json`, permits one agent invocation to refresh that file,
adds original and refreshed token leaves to in-memory redaction, and removes
the temporary home before publishing any result. OAuth JSON is never exported
as an environment value. Every containerized inner `srun` also disables the
implicit host-home mount.

The task loader also rejects both credential-like controller environment
names and recognizable secret-like values before the normalized task can be
persisted. This is defense in depth; the controller environment is never a
supported credential transport.

Agent workers also run a negative preflight before loading credentials. They
fail closed if scheduler authentication material or scheduler clients are
available. A task role environment cannot admit scheduler-owned names or
credential-like names.

Each attempt mounts its exact immutable `input.json` or `role-input.json` as a
read-only file. Within the attempt mailbox, only the sibling `output/` tree
(including declared evidence) is writable; the authoritative credential
descriptor and revocation ledger remain controller-owned and are never derived
from worker output. Code, checkpoint, and reference mounts are read-only;
only a Coder's exact metadata-free candidate overlay is separately writable.

Reviewer and QA jobs consume an immutable candidate overlay, never a mutable
Coder worktree. Their input carries the controller-computed overlay digest;
the worker recomputes the metadata-free tree digest before loading credentials
or invoking the backend and fails closed on any mismatch.

Before a successful Coder candidate can be bound, the worker streams declared
changed-file bytes through exact injected-secret and central secret-pattern
scans while computing the overlay digest. Controller import and staging verify
that same content identity and reject undeclared changes. Symlinks, special
files, multiple hard links, metadata/read races, or digest drift fail closed
without publishing a candidate patch or commit.

## Parallel Smith and controller-only fan-in

Horizontal Smith parallelism is a bounded wave of independent catalog-entry
WorkItems. Each Smith Coder owns one entry and one metadata-free overlay; its
candidate passes its own ordered gates, Reviewer analysis, and fresh Reviewer
rerun. A modifying item also passes its required QA before integration.
`max_parallel_items`, `max_nodes_total`, `max_gpus_total`, dependency,
path/claim locks, and per-item escalation bounds constrain the wave. This can
consume several single-node allocations concurrently. When
`distinct_nodes_required: true`, the task must also set `exclusive: true`.
Every horizontal Smith Coder, Reviewer-analysis, and Reviewer-rerun attempt is
then a separate exclusive one-node Slurm allocation. The worker writes
`items/<item>/attempts/<sequence:04d>/placement-receipt.json`, binding the exact
schema/run/item/attempt/task/generation and Slurm job/array/cluster identity to
its observed hostname, single-node `NodeList`, exclusivity request, and
content digest.

The approved plan deterministically freezes each horizontal item cohort.
Placement is proved separately for the Coder, Reviewer-analysis, and
Reviewer-rerun attempt sets. The controller waits without writing fan-in
intent while a nonterminal peer has not closed that role stage or an active
attempt lacks its exact job/receipt. It blocks fail-closed on cohort/state
drift; missing terminal evidence; malformed, symlinked, or digest-invalid
receipts; any identity, hostname/`NodeList`, or exclusivity mismatch; and any
host/`NodeList` reuse within the same role stage. A role stage never launched
for an already-terminal peer has no job and is omitted. Nodes may be reused
across different role stages, and stage-level QA is intentionally outside
this horizontal Smith placement proof.

Vertical Smith validation is one deterministic-gate attempt for one WorkItem,
with one multi-node allocation, one trusted supervisor, and a
controller-generated `srun` rank plan. It is not recursive Smith and not a set
of independently mutable array elements. The resource topology must exactly
match target world size, use one GPU per rank, and record distinct rank-to-node
placement. Its collective rank receipt is independent of the horizontal
per-agent placement receipts above.

All observations of known jobs use the owned path bound to exact
cluster/job/array, scheduler user, generated job name, comment, and submission
token. For collective execution, the trusted supervisor resolves `srun` and
rank-side `scontrol` to absolute executable regular files through immutable,
non-worker-owned and non-writable paths, then executes fixed argv with
`shell=False`.

Certification remains explicit:

- `LOCAL` is the explicit local/developer override; where it records product
  evidence, the strongest multi-rank scope is the canonical real
  single-node/four-GPU body;
- `SYNTHETIC` proves multi-node launch, placement, and manifest plumbing only;
- `REAL` requires the real ModelingV2 product rank body and exact repository,
  Python package, native build, image, compute capability, collective backend,
  transport, topology, and placement evidence.

After independent siblings are approved, only the controller integrates them,
in deterministic order, under its Git lock. It verifies every candidate patch
again, applies each successful sibling independently, consumes semantic
`IndexDelta` records, updates `catalog/index.yaml` once, and runs aggregate
validators. Workers never patch the shared index. Intent/receipt journals and
linear patch-sequence verification make pre-receipt and receipt-before-ledger
crashes restart-safe.

## Assembler, Tuner, Reviewer, and QA

Assembler implements one target core, feature, weights, or routing item. The
default self-contained mode uses already integrated catalog dependencies; a
missing surface becomes a planned Smith dependency. When the frozen task or
approved plan explicitly selects task-scoped delegated built-in reuse,
Assembler may instead create a narrow adapter to the exact named mature model
and weight mapper. It cannot add undeclared model-zoo dependencies, use a
sibling target, silently fall back, or create an uncertified catalog shortcut.

Tuner starts from an immutable, correctness-gated baseline and changes exactly
one declared variable. The enforced path is:

```text
matched baseline/candidate measurement
  -> immutable tuning artifacts and verification candidate
  -> ordered deterministic hard gates
  -> fresh Reviewer analysis
  -> fresh Reviewer rerun of decisive evidence
  -> mandatory independent QA
  -> controller replay and KEEP or REJECT
```

Coder, Reviewer, and QA may report evidence or a recommendation, but none can
promote a candidate. The controller writes the promotion decision only after
Reviewer rerun and QA approve the same digest. It deterministically replays the
measurement identity, curves, uncertainty, gates, campaign, and decision. A
hard correctness/feature failure always rejects the candidate; performance
cannot override it.

Per-item QA is pinned to the immutable candidate and trusted gate-result
digests. A Stage advances `ACTIVE -> READY_FOR_QA -> QA_PASSED -> CLOSED` only
after all required Goals and WorkItems integrate and required QA evidence is
valid. The controller, not the QA agent, closes the Stage and run.

## Delivery boundary

`delivery.mode` selects one review-ready handoff:

- `diff_only` uses a detached checkout under the run workspace. Internal
  controller commits provide restart-safe identities, then terminal success
  publishes `delivery/staircase.patch` plus an immutable receipt bound to the
  run, task digest, base commit, integration head, SHA-256, size, and path.
- `signed_commits` requires a safe branch name and creates/adopts that branch
  only in an isolated workspace checkout. Integrated commits include a DCO
  `Signed-off-by` trailer, and the terminal report records the exact branch,
  base, and integration head.

Neither mode changes the user's source checkout. Both require `merge: false`
and `push: false`; Staircase does not merge, push, or create a pull request.

## ModelingV2 evidence boundary

Agents may read Hugging Face sources, built-in TensorRT-LLM per-model
definitions, and existing ModelingV2 targets and catalog entries. Targets are
self-contained and catalog-backed by default. A frozen task/plan may explicitly
admit task-scoped delegated built-in reuse for one exact synthetic target and
name one mature model/weight-mapper pair. This does not admit sibling targets,
other model-zoo helpers, silent fallback, or the built-in implementation as its
own correctness oracle.

See [`references/modeling_v2_contract.md`](references/modeling_v2_contract.md)
for paths, routing identity, catalog entry shape, required runtime mode, and
the evidence ladder. In particular, every result attributed to ModelingV2
must start every rank with `TRTLLM_MODELING_V2=require`. Boot is not
correctness: both modes require exact no-fallback routing, a read-only published
checkpoint, real boot and generation, an independent reference, Reviewer, and
QA. A high-risk expected output cannot come only from the implementation under
test or a helper that shares its bug.

## Certification limitations

The implementation and CPU/fake-scheduler tests establish deterministic
contracts, recovery, exact-ID ownership, credential isolation, packaging, and
synthetic rank plumbing. They are not evidence that a deployment has:

- run real Codex or Claude role workers on the target cluster;
- supplied and certified the external AF_UNIX broker daemon/socket, backend
  egress policy, or a live backend;
- certified scheduler accounting retention for absence proofs;
- exercised live nested worker submission, preemption, or node failure there;
- run a real multi-node ModelingV2 collective; or
- passed target GPU boot, component, accuracy, feature, or performance gates.

Those claims require immutable receipts from the declared site, image, build,
checkpoint, GPU topology, live backend, and real product rank body. A fake
scheduler, synthetic canary/runner, or editable local build cannot establish
REAL-model GPU correctness, accuracy, or performance. A blocked live Slurm
canary is not a successful certification receipt.
