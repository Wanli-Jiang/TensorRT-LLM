<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Staircase tuning

Tuning evaluates bounded, one-variable hypotheses on an already integrated and
correctness-gated ModelingV2 target. It does not replace onboarding and cannot
use performance to excuse a correctness, routing, catalog, or feature-signal
failure.

## Preconditions and approved-plan inputs

Copy [`task.example.yaml`](task.example.yaml). It uses the same strict
schema-v2 execution, seven-role, credential, recovery, certification, and
delivery contracts as onboarding. In addition, the accepted PlanDrafter output
must freeze, and PlanReviewer must validate:

- the integrated target, base commit, build, image, and checkpoint;
- route identity and complete parallel topology;
- workload, hardware, metric direction, and measurement protocol;
- one code or configuration knob and its baseline/candidate values;
- decisive-effect, noise, and maximum-uncertainty rules;
- matched correctness and feature hard gates for both arms;
- feature-specific acceptance for speculative or other distribution-preserving
  fast paths; and
- Slurm resource/concurrency bounds that cannot oversubscribe the approved
  allocation envelope.

A structural-forward or routing-identity change is onboarding/assembly work,
not tuning. Code and configuration changes cannot share one hypothesis. Gate
fields are repository-relative pytest selectors, not shell commands or prose.

Choose exactly one credential mode for all seven roles. A credentialed run
requires byte-identical broker policy and the site-owned AF_UNIX socket;
Staircase implements only the client, not the daemon. A preauthenticated run
uses broker ID `preauthenticated`, empty credential-name lists, and no socket.

The controller image/build must set `scheduler_clients: true` and
`container_launch_mode: in_allocation_srun`. Agent roles use the distinct
task-level `agent_worker` image/build with `scheduler_clients: false` and
`network_enforcement: certified_worker_image`. The latter is a site
certification that the immutable worker image enforces `backend_api_only`, not
a network namespace or firewall created in the Python worker. Credential-like
controller environment names and recognizable secret-like values are rejected
before the normalized task is persisted.

Controller mounts are identity-only: every normalized container path equals
its canonical host path and covers each required controller-visible surface;
frozen repository, workspace, checkpoint, and reference paths are not
translated inside the container.

## Launch

Production execution requires the exact launcher/build/mount fact manifest:

```bash
staircase tune \
    --task /shared/tasks/qwen3-tune.yaml \
    --workspace /shared/staircase-runs/qwen3-tune-001 \
    --execution slurm \
    --preflight-facts /shared/tasks/qwen3-tune.preflight.json
```

The task digest is immutable across resume. Use a new workspace for a changed
hypothesis envelope or baseline unless a state migration explicitly accepts
it. `--execution local` is limited to CPU/fake-runner development; a local
timing result is not production evidence.

Slurm launch uses a bare allocation `sbatch` with no Pyxis flags. Its fixed
script ends in `exec srun --nodes=1 --ntasks=1 --overlap` with the exact
`--container-image`, optional `--container-mounts`, and fixed internal argv.
Task text cannot add shell fragments or scheduler flags, and `exec` propagates
the inner process exit status to the batch job.

Launcher, planning, Tuner/Reviewer/QA, and successor submissions use the same
append-only exact-token recovery. A unique match is adopted; default
uncertified queue/accounting absence becomes durable manual recovery. Only
site-certified retention plus bounded, separated absence samples permits a
second/final submit or no-job intent cancellation.

`staircase status` is a read-only projection of owned controller/child jobs,
lease/checkpoint, launcher/successor, and mailbox drift. It does not adopt,
quarantine, mutate state, or emit credentials. Dead-controller emergency
cancel holds the lease lock, reconciles hidden `SUBMITTING` intents, revokes
exact role credentials, cancels exact owned children, and writes its complete
receipt last; ambiguity or broker failure requires manual reconciliation.

## Enforced hypothesis lifecycle

The controller does not accept a Tuner's recommendation as promotion. The
implemented path is:

1. Tuner Coder measures baseline and candidate under the same checkpoint,
   target, route, workload, topology, build, protocol, and hardware, changing
   exactly one declared variable.
2. The controller validates matched gate identities, finite samples, bounded
   uncertainty, and distinct arm digests. It writes immutable baseline,
   candidate, evaluation, and evidence-ready campaign artifacts.
3. The controller binds those artifacts and any permitted code change to one
   immutable verification candidate.
4. The candidate passes the ordered deterministic hard-gate suite.
5. A fresh Reviewer performs analysis against the frozen candidate and
   immutable measurement envelopes.
6. A second fresh Reviewer attempt reruns the decisive evidence. Only this
   digest-pinned rerun can approve the WorkItem.
7. Independent QA is mandatory for every Tuner item, even if the general
   workflow did not select that item for QA. QA revalidates the same candidate
   and trusted gate-result digests.
8. Only after Reviewer rerun and QA approval does the controller replay the
   complete campaign and write a `KEEP` or `REJECT` promotion decision.
9. A kept patch is integrated by the controller-only fan-in path. A rejected
   or no-change hypothesis remains an honest terminal result; it is not
   promoted.

The deterministic outcomes are `keep`, `invalid_baseline`,
`hard_gate_failure`, `regression`, `no_change`, and `noise`. Baseline hard-gate
failure invalidates the comparison. Candidate hard-gate failure rejects it.
Only a conservative improvement bound meeting the decisive-effect threshold
can produce `keep`; neither Coder, Reviewer, nor QA can override or publish the
controller decision.

Reviewer and QA inputs also carry the controller-computed candidate-overlay
digest. Each worker recomputes the metadata-free worktree digest before
loading credentials or calling the backend, so a mutable or substituted
overlay fails closed.

Every attempt mounts exact `input.json`/`role-input.json` read-only and exposes
only sibling `output/`/evidence as its writable mailbox; authoritative
credential descriptors and revocation ledgers remain controller-owned. A
successful changed candidate is stream-scanned for exact injected secrets and
central secret patterns while its digest is computed. Controller import/stage
rejects undeclared changes, symlinks, special files, hard links, races, and
digest drift.

Prefer paired same-session measurements when session variance can dominate the
expected delta. Baseline and candidate hard-gate sets must match exactly.
Feature gates are required when emitted-text accuracy can remain correct while
the optimized path silently stops working.

## Slurm and evidence boundaries

Tuner Coder is one single-node/single-task typed role job. Its deterministic
gate may be a real multi-node allocation when the frozen target topology
requires it. The trusted supervisor executes the controller-generated rank
plan and records placement; the Tuner agent does not call `srun` or submit
children. All Git integration remains serialized by the controller.

Known-job observations bind cluster/job/array, user, generated job name,
comment, and submission token. Collective launch resolves absolute immutable
`srun` and rank-side `scontrol` executables through non-worker-owned,
non-writable trust paths and invokes fixed argv with `shell=False`.

The task's `smith.distinct_nodes_required`/`exclusive` contract applies to
horizontal catalog Smith role waves, not to a Tuner role or this vertical
collective. When such a Smith wave exists in an onboarding plan, Coder,
Reviewer-analysis, and Reviewer-rerun attempts are checked as three separate
exclusive one-node placement sets; cross-stage reuse is allowed and QA is
excluded. A vertical collective instead proves all ranks from one multi-node
deterministic-gate allocation.

`SYNTHETIC` collective evidence proves launch and manifest plumbing only. A
production scaling or performance claim requires the declared real Slurm/GPU
topology, real ModelingV2 rank body, exact build/product identity, and
candidate-bound receipts. Likewise, CPU/fake tests, a broker-client test, or a
local editable build do not certify live agent/backend execution, the external
broker daemon, a product collective, or REAL-model GPU correctness, accuracy,
or performance. Broker daemon/socket and backend egress, accounting retention,
the live backend, and real product tests remain external certifications; a
blocked live Slurm canary is not success evidence.

Delivery follows the task: `diff_only` publishes a content-verified terminal
patch and receipt; `signed_commits` uses an isolated DCO-signed branch. Neither
mode merges, pushes, creates a pull request, or mutates the user's checkout.
