<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ModelingV2 contract used by Staircase

This is the compact agent-facing product contract for Staircase. The in-tree
[`tensorrt_llm/_torch/modeling_v2/README.md`](../../../../../tensorrt_llm/_torch/modeling_v2/README.md)
and ModelingV2 tests remain authoritative if this summary drifts.

## Ownership and artifact boundary

```text
tensorrt_llm/_torch/modeling_v2/
  _router_index.py
  explain.py
  catalog/
    index.yaml
    <category>/<entry>.md
    <category>/<entry>.py
  models/<family>/
    routing.py
    targets/<checkpoint>/<gpu_arch>/<parallel>/
      modeling.py
      weights.py

tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<entry>.py
tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py
tests/unittest/_torch/modeling_v2/test_modeling_v2_routing.py
tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py
tests/integration/defs/accuracy/test_modeling_v2_*.py
tests/integration/defs/accuracy/references/
```

Staircase is only the control plane. It must not create a second product tree,
runtime flag, routing registry, test hierarchy, or receipt vocabulary.
Staircase agents may read Hugging Face sources, built-in TensorRT-LLM
per-model definitions, and existing ModelingV2 targets/catalog entries as
references. That permission does not weaken the checked-in boundary:

- a target is flat and self-contained for one checkpoint/GPU/topology triple;
- it may import catalog entries and siblings inside its own target directory;
- it cannot import model helpers or target files from the built-in zoo or a
  sibling ModelingV2 target; and
- the checkpoint is read as published, without a target-owned `config.json`.

Targets contain `modeling.py` and `weights.py`. Smoke programs, benchmark
configuration, and accuracy references live in the existing examples/tests
surfaces, not in the target directory.

## Catalog and Smith contract

The catalog is the target's tensor-computation vocabulary. Every operation
that creates or transforms a tensor is a catalog call; only tensor metadata
reads and Python control flow live directly in a target.

A non-torch catalog entry spans both product and test trees:

1. `catalog/<category>/<entry>.md` defines the semantic contract and receipts;
2. `catalog/<category>/<entry>.py` is a thin, normally one-invocation wrapper;
3. `tests/unittest/_torch/modeling_v2/<category>/test_modeling_v2_<entry>.py`
   is the collected GPU parity test.

Torch mirror entries are the documented exception: upstream owns their
correctness, so they do not carry the same per-architecture receipt shape.

One Smith WorkItem owns one catalog entry. A read-only `CatalogVerifyItem`
cannot return a patch. A `CatalogOnboardItem` returns only its allowed
entry-specific paths and, when needed, a semantic `IndexDelta`; it cannot patch
`catalog/index.yaml`. After the candidate's gates and two fresh Reviewer
attempts succeed, the controller verifies the patch again and integrates
siblings in deterministic order under its Git lock. Only the controller
applies all semantic deltas to the shared index and runs aggregate validators.

A receipt is valid only for the exact entry surface, tests, GPU architecture,
and content it names. It is invalidated by a later change to the contract,
wrapper, or test and cannot be generalized to another SM or to a target that
merely imports the entry.

## Routing identity

`_router_index.py` maps public `architectures[0]` to one family `routing.py`.
The family routing tree returns a synthetic registered class for an exact
target. A routing criterion must be known when `_resolve_class` runs, remain
constant for the engine lifetime, and change forward structure. Admitted
examples are:

- checkpoint shape fingerprint;
- GPU architecture;
- explicit parallel topology; and
- a requested feature that structurally changes the forward.

Batch composition, sequence lengths, token limits, CUDA-graph batch bounds,
and other workload/tuning knobs are not target identity. Per-step variation is
dispatch inside the selected target. Required mode must fail loudly when no
exact target matches.

The checkpoint fingerprint cannot distinguish every fine-tune with the same
shape. Consequently, a gate certifies the exact checkpoint provenance named in
its receipt, not every checkpoint that routes to that target.

## Required runtime mode

Use:

```bash
export TRTLLM_MODELING_V2=require
```

Set it before every rank starts. Setting it only in a driver after MPI or
another launcher initializes can split driver and worker routing. Evidence
from `auto` is insufficient for a ModelingV2 claim because it may fall back to
a built-in implementation. The older `MODELING_V2_TARGET` and the obsolete
`TRTLLM_STAIRCASE` names are not part of the contract.

## References and gates

For high-risk semantics, expected output cannot come only from the candidate,
its wrapper, or a shared bug-prone helper. Build the reference ladder from:

1. published checkpoint/Hugging Face semantics and provenance;
2. a minimal independent oracle or checked golden;
3. optional built-in TensorRT-LLM behavior as prior art or a cross-check; and
4. the ModelingV2 candidate under test.

Reading built-in definitions is allowed; importing or sharing their model
helpers in the final target is not.

Use the cheapest gate capable of falsifying the claim:

1. static catalog, routing, target-identity, required-op, and self-containment
   checks;
2. entry-specific GPU parity on the declared architecture;
3. target boot with weight coverage and deterministic continuations;
4. module/forward, logit, or generation parity against an independent source;
5. an accuracy canary followed by the task's full accuracy protocol; and
6. a feature-specific acceptance signal where output accuracy may remain
   correct while the feature is broken, such as draft-path acceptance.

Boot catches construction failures; it is not accuracy. Performance is
measured evidence, never the universal release gate. Required correctness,
accuracy, self-containment, routing, catalog, and feature-acceptance failures
are hard failures regardless of score or speed.

## Execution isolation and horizontal placement

The Slurm controller and agent roles do not share an execution image. The
controller image/build has scheduler clients and is the only component that
submits jobs. Agent roles use the distinct digest-bound `agent_worker`
image/build with `scheduler_clients: false`. Its
`network_enforcement: certified_worker_image` value records the site's trusted
certification that the image enforces `backend_api_only`; Staircase does not
implement an in-process namespace or firewall.

Controller mounts preserve path identity: each normalized container target is
the canonical host source and covers every required controller-visible path.
Worker attempts then narrow that surface to exact read-only
`input.json`/`role-input.json`, code, checkpoint, and references, plus only the
sibling writable `output/`/evidence mailbox and, for a Coder, its exact writable
candidate overlay. Authoritative credential descriptors and revocation ledgers
remain controller-owned.

Containerized jobs request a bare Slurm allocation first. The fixed submitted
script then `exec`s an in-allocation `srun` with the validated image, mounts,
and internal command. Pyxis flags belong to `srun`, not `sbatch`, and the
`exec` boundary preserves the inner process exit status.

All known-job observations bind cluster/job/array, user, generated job name,
comment, and exact submission token. Submission recovery is append-only;
uncertified accounting absence becomes durable manual recovery rather than a
duplicate job. `status` only projects owned scheduler, lease, checkpoint,
launcher/successor, and mailbox drift. Dead-controller cleanup holds the lease
lock, reconciles hidden intents, revokes exact credentials, cancels exact owned
jobs, and writes its full receipt last; ambiguity remains manual.

With horizontal Smith `distinct_nodes_required: true`, `exclusive: true` is
mandatory. Each one-node Smith Coder, Reviewer-analysis, and Reviewer-rerun
attempt emits an immutable placement receipt bound to its exact attempt and
Slurm identity. Before controller-only fan-in, host/`NodeList` uniqueness is
proved independently inside each of those three approved-plan role waves.
Incomplete active peers cause WAIT without a fan-in intent; terminal missing,
tampered, mismatched, non-exclusive, or duplicate-node evidence blocks fail
closed. Cross-stage node reuse is permitted, and stage-level QA is excluded.
This horizontal proof is separate from the collective rank evidence below.

## Collective and certification boundary

Staircase freezes the evidence scope before a collective starts:

- `LOCAL` is the explicit local/developer boundary; its canonical multi-rank
  product evidence is a real single-node/four-GPU rank body;
- `SYNTHETIC` uses a synthetic multi-node rank body and can certify only runner
  launch/placement/manifest behavior; and
- `REAL` uses the real product rank body and requires exact expected product
  identity.

The deterministic-gate resource must match target world size, allocate one GPU
per rank, and have distinct named nodes. The trusted supervisor executes only
the controller-generated `srun` plan. Completion verifies the immutable plan,
rank-command and input digests, all rank reports, rank/local-rank placement,
hostnames, body kind, and evidence scope.

The supervisor resolves `srun` and rank-side `scontrol` to absolute executable
regular files through immutable, non-worker-owned and non-writable trust paths,
then invokes fixed argv with `shell=False`.

For `REAL`, the receipt must additionally match repository commit, imported
`tensorrt_llm` path, native build, image, compute capability, collective
backend, transport, full Mapping, and actual product output. Node count or an
`srun` exit code alone is never a collective certificate.

## Candidate and delivery receipts

Candidate evidence binds run/task/generation, base commit, attempt, immutable
input, candidate content, changed paths, command, build, device/topology, and
evidence hashes. Coder overlays contain no shared Git metadata. The controller
creates the candidate commit and requires a valid DCO `Signed-off-by` trailer;
Reviewer and QA receive that frozen candidate plus a controller-computed
metadata-free overlay digest, which their workers recompute before backend
invocation. Fan-in uses the same immutable candidate identity rather than
mutable Coder logs or prose.

Successful Coder changed-file bytes are stream-scanned for exact injected
credential values and central secret patterns while the content digest is
computed. Controller import and staging verify the bound digest and reject
undeclared changes, symlinks, special files, multiple hard links, races, and
digest drift before publishing a candidate commit or patch.

Diff-only delivery is a terminal patch from the declared base to the recorded
integration head. A successful terminal report accepts it only when the patch
and immutable receipt agree on run ID, task digest, base, head, path, SHA-256,
and byte size. Signed-commit delivery uses a task-named branch in an isolated
workspace checkout and reports the exact base and head. Neither delivery mode
merges or pushes.

## Honest claim boundary

CPU/unit tests, a fake scheduler, a synthetic runner, an AF_UNIX client test,
or a local editable build can prove their respective contracts but cannot
certify live agent/backend execution, the external broker daemon, a real
ModelingV2 collective, a target checkpoint, or REAL-model GPU correctness,
accuracy, or performance. Those claims exist only when the terminal report
contains the required immutable receipts from the exact declared site, build,
image, checkpoint, topology, live backend, and real product body. The certified
agent-worker image, broker daemon/socket and egress, accounting retention, live
backend, and REAL product tests remain external deployment responsibilities. A
blocked live Slurm canary is not certification evidence.
