# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only contract tests for the Staircase Slurm boundary."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from agent_flow.workflows.staircase.common.slurm import (
    CommandResult,
    DependencyType,
    FakeScheduler,
    InternalCommand,
    InternalEntrypoint,
    JobDependency,
    JobIdentity,
    JobStatus,
    Mount,
    ObservationSource,
    ResourceRequest,
    SchedulerError,
    SlurmCommandError,
    SlurmScheduler,
    normalize_environment,
    redact_environment,
    render_internal_script,
)

_TOKEN = "attempt-00000001"


class ScriptedExecutor:
    """Return scripted command results while retaining every exact invocation."""

    def __init__(self, results: Sequence[CommandResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def run(self, argv: Sequence[str], *, input_text: str | None = None) -> CommandResult:
        self.calls.append((tuple(argv), input_text))
        if not self._results:
            raise AssertionError(f"unexpected scheduler invocation: {argv}")
        return self._results.pop(0)


@pytest.fixture
def resources(tmp_path: Path) -> ResourceRequest:
    """Return a representative, fully typed worker resource request."""
    return ResourceRequest(
        label="smith-attention",
        account="coreai",
        partition="batch",
        qos="normal",
        reservation="bringup",
        time_limit="00:20:00",
        nodes=2,
        tasks_per_node=8,
        cpus_per_task=4,
        gpus_per_node=8,
        memory_mb=32_768,
        output_path=tmp_path / "logs" / "%j.out",
        error_path=tmp_path / "logs" / "%j.err",
        signal_seconds=60,
        exclusive=True,
        container_image="/images/trtllm.sqsh",
        mounts=(Mount(tmp_path / "source", Path("/workspace")),),
    )


@pytest.fixture
def worker_command(tmp_path: Path) -> InternalCommand:
    """Return a fixed internal worker command."""
    return InternalCommand(
        InternalEntrypoint.WORKER,
        input_bundle=tmp_path / "workspace with spaces" / "input.json",
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("label", "smith;scancel"),
        ("account", "acct\n--wrap=evil"),
        ("partition", "batch $(id)"),
        ("qos", "normal,*"),
        ("reservation", "../other"),
    ],
)
def test_resource_request_rejects_scheduler_field_injection(
    resources: ResourceRequest,
    field_name: str,
    bad_value: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported characters"):
        replace(resources, **{field_name: bad_value})


@pytest.mark.parametrize("field_name", ["nodes", "tasks_per_node", "cpus_per_task"])
def test_resource_request_rejects_bool_as_integer(
    resources: ResourceRequest,
    field_name: str,
) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        replace(resources, **{field_name: True})


@pytest.mark.parametrize("time_limit", ["20", "00:60:00", "00:00:60", "00:00:00"])
def test_resource_request_rejects_noncanonical_time(
    resources: ResourceRequest,
    time_limit: str,
) -> None:
    with pytest.raises(ValueError, match="time_limit"):
        replace(resources, time_limit=time_limit)


def test_resource_request_rejects_raw_container_options(resources: ResourceRequest) -> None:
    with pytest.raises(ValueError, match="container_image"):
        replace(resources, container_image="image.sqsh --container-mounts=/etc:/host")
    with pytest.raises(ValueError, match="cannot contain"):
        Mount(Path("/src,other"), Path("/target"))
    with pytest.raises(ValueError, match="cannot contain '..'"):
        replace(resources, container_image="/images/../secrets.sqsh")
    with pytest.raises(ValueError, match="in_allocation_srun"):
        replace(resources, container_launch_mode="sbatch_pyxis")


def test_gpu_allocation_padding_is_explicit_and_single_rank(
    resources: ResourceRequest,
) -> None:
    with pytest.raises(ValueError, match="requires one node and one task"):
        replace(resources, gpu_allocation_padding=True)

    single_rank = replace(
        resources,
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=4,
        gpu_allocation_padding=True,
    )
    assert single_rank.gpu_allocation_padding is True

    with pytest.raises(ValueError, match="requires explicit gpu_allocation_padding"):
        replace(single_rank, gpu_allocation_padding=False)

    with pytest.raises(ValueError, match="fixed in-allocation container launch"):
        replace(single_rank, container_image=None, mounts=())

    with pytest.raises(ValueError, match="requires more than one allocated GPU"):
        replace(single_rank, gpus_per_node=1)


@pytest.mark.parametrize("raw_identity", ["123*", "123_[1-4]", "123.batch", "1;scancel", "0"])
def test_job_identity_rejects_non_exact_targets(raw_identity: str) -> None:
    with pytest.raises(ValueError):
        JobIdentity.parse(raw_identity)


def test_exact_array_identity() -> None:
    identity = JobIdentity.parse("12345_7", cluster="alpha")
    assert identity.job_id == "12345"
    assert identity.array_task_id == "7"
    assert identity.scheduler_id == "12345_7"


def test_fixed_script_quotes_paths_without_accepting_raw_fragments(tmp_path: Path) -> None:
    workspace = tmp_path / "space and ' quote; $(touch nope)"
    command = InternalCommand(
        InternalEntrypoint.CONTROLLER,
        workspace=workspace,
        generation=1,
        owner_nonce="controller-0001",
    )

    script = render_internal_script(command)

    assert script.startswith("#!/bin/bash\nset -euo pipefail\numask 077\nexec python3 -m ")
    assert "agent_flow.workflows.staircase.internal controller" in script
    assert "--owner-nonce controller-0001" in script
    assert "'" in script
    assert script.count("exec ") == 1


def test_fixed_script_uses_resolved_container_launcher(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    script = render_internal_script(
        worker_command,
        resources,
        container_launcher="/cm/local/apps/slurm/25.11/bin/srun",
    )

    assert "exec /cm/local/apps/slurm/25.11/bin/srun " in script


def test_cpu_container_unsets_only_injected_placement_environment(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    cpu_resources = replace(
        resources,
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=0,
        gpu_allocation_padding=False,
    )

    cpu_script = render_internal_script(worker_command, cpu_resources)
    gpu_script = render_internal_script(worker_command, resources)

    python_index = cpu_script.index("python3")
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "NVIDIA_VISIBLE_DEVICES",
        "RANK",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_GPUS_PER_NODE",
        "SLURM_GPUS_PER_TASK",
        "SLURM_JOB_GPUS",
        "SLURM_STEP_GPUS",
        "WORLD_SIZE",
    ):
        marker = f"-u {name}"
        assert marker in cpu_script
        assert cpu_script.index(marker) < python_index
        assert marker not in gpu_script

    assert "-u SLURM_LOCALID" not in cpu_script
    assert "-u SLURM_NTASKS" not in cpu_script
    assert "-u SLURM_PROCID" not in cpu_script


def test_internal_command_has_role_specific_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot have an input bundle"):
        InternalCommand(
            InternalEntrypoint.CONTROLLER,
            workspace=tmp_path,
            generation=1,
            owner_nonce="controller-0001",
            input_bundle=tmp_path / "input.json",
        )
    with pytest.raises(ValueError, match="requires an input bundle"):
        InternalCommand(InternalEntrypoint.WORKER)
    with pytest.raises(ValueError, match="requires an owner nonce"):
        InternalCommand(InternalEntrypoint.CONTROLLER, workspace=tmp_path, generation=1)
    with pytest.raises(ValueError, match="accepts only an input bundle"):
        InternalCommand(
            InternalEntrypoint.WORKER,
            workspace=tmp_path,
            input_bundle=tmp_path / "input.json",
        )


def test_environment_is_allowlisted_deterministic_and_nonsecret() -> None:
    normalized = normalize_environment(
        {
            "PYTHONUNBUFFERED": "1",
            "HF_HOME": "/shared/cache",
            "TLLM_FMHA_LIBS": "fallback",
            "QWEN3_5_HF_REFERENCE_DIR": "/shared/qwen-reference",
            "QWEN3_5_BUILTIN_REFERENCE_DIR": "/shared/qwen-builtin-reference",
        }
    )
    assert normalized == (
        ("HF_HOME", "/shared/cache"),
        ("PYTHONUNBUFFERED", "1"),
        ("QWEN3_5_BUILTIN_REFERENCE_DIR", "/shared/qwen-builtin-reference"),
        ("QWEN3_5_HF_REFERENCE_DIR", "/shared/qwen-reference"),
        ("TLLM_FMHA_LIBS", "fallback"),
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        normalize_environment({"BASH_ENV": "/tmp/execute-me"})
    with pytest.raises(ValueError, match="prohibited"):
        normalize_environment({"HF_TOKEN": "secret"})
    with pytest.raises(ValueError, match="safely serialized"):
        normalize_environment({"HF_HOME": "/cache,PATH=/attacker"})

    assert redact_environment({"HF_TOKEN": "secret", "HF_HOME": "/cache"}) == {
        "HF_TOKEN": "<redacted>",
        "HF_HOME": "/cache",
    }


def test_submit_builds_only_structured_argv_and_parses_identity(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    executor = ScriptedExecutor([CommandResult(0, "12345;alpha\n")])
    scheduler = SlurmScheduler(executor=executor, cluster="alpha", username="tester")

    identity = scheduler.submit(
        resources,
        worker_command,
        _TOKEN,
        {"PYTHONUNBUFFERED": "1", "HF_HOME": "/shared/cache"},
    )

    assert identity == JobIdentity("12345", cluster="alpha")
    argv, script = executor.calls[0]
    assert argv[0] == "sbatch"
    assert "--parsable" in argv
    assert f"--job-name=staircase-{_TOKEN}" in argv
    assert f"--comment=staircase:{_TOKEN};label=smith-attention" in argv
    assert "--nodes=2" in argv
    assert "--gpus-per-node=8" in argv
    assert "--exclusive" in argv
    assert not any(value.startswith("--container-image=") for value in argv)
    assert not any(value.startswith("--container-mounts=") for value in argv)
    assert "--export=HF_HOME=/shared/cache,PYTHONUNBUFFERED=1" in argv
    assert "--clusters=alpha" in argv
    assert not any(value == "--wrap" or value.startswith("--wrap=") for value in argv)
    assert script == render_internal_script(worker_command, resources)
    assert "exec srun --nodes=1 --ntasks=1 --overlap" in script
    assert "--no-container-mount-home" in script
    assert "--container-image=/images/trtllm.sqsh" in script
    assert "--container-mounts=" in script


def test_submit_uses_controller_resolved_container_launcher(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    executor = ScriptedExecutor([CommandResult(0, "12345;alpha\n")])
    scheduler = SlurmScheduler(
        executor=executor,
        cluster="alpha",
        username="tester",
        container_launcher="/cm/local/apps/slurm/25.11/bin/srun",
    )

    scheduler.submit(resources, worker_command, _TOKEN)

    _argv, script = executor.calls[0]
    assert script is not None
    assert "exec /cm/local/apps/slurm/25.11/bin/srun " in script


def test_submit_padded_single_rank_binds_only_one_gpu_to_the_worker(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    padded = replace(
        resources,
        nodes=1,
        tasks_per_node=1,
        gpus_per_node=4,
        gpu_allocation_padding=True,
    )
    executor = ScriptedExecutor([CommandResult(0, "12345;alpha\n")])
    scheduler = SlurmScheduler(executor=executor, cluster="alpha", username="tester")

    scheduler.submit(padded, worker_command, _TOKEN)

    argv, script = executor.calls[0]
    assert "--ntasks-per-node=1" in argv
    assert "--gpus-per-node=4" in argv
    assert "--gpus-per-task=1" in script
    assert script.count("--gpus-per-task=1") == 1


def test_agent_policy_binding_is_controller_owned_and_exported(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    bound = replace(resources, agent_policy_digest="a" * 64)
    executor = ScriptedExecutor([CommandResult(0, "12345;alpha\n")])
    scheduler = SlurmScheduler(executor=executor, cluster="alpha", username="tester")

    scheduler.submit(bound, worker_command, _TOKEN)

    argv, _script = executor.calls[0]
    assert f"--export=STAIRCASE_AGENT_POLICY_DIGEST={'a' * 64}" in argv


def test_submit_renders_exact_typed_afterany_dependency(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    executor = ScriptedExecutor([CommandResult(0, "12345;alpha\n")])
    scheduler = SlurmScheduler(executor=executor, cluster="alpha", username="tester")
    dependency = JobDependency(
        DependencyType.AFTERANY,
        JobIdentity("12000", cluster="alpha"),
    )

    scheduler.submit(resources, worker_command, _TOKEN, dependency=dependency)

    assert "--dependency=afterany:12000" in executor.calls[0][0]


def test_dependency_rejects_array_and_cross_cluster_jobs(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    with pytest.raises(ValueError, match="non-array"):
        JobDependency(
            DependencyType.AFTERANY,
            JobIdentity("12000", array_task_id="1", cluster="alpha"),
        )

    scheduler = FakeScheduler(cluster="alpha")
    dependency = JobDependency(
        DependencyType.AFTERANY,
        JobIdentity("12000", cluster="beta"),
    )
    with pytest.raises(SchedulerError, match="cluster"):
        scheduler.submit(resources, worker_command, _TOKEN, dependency=dependency)


def test_fake_scheduler_captures_typed_dependency(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    scheduler = FakeScheduler(cluster="alpha")
    dependency = JobDependency(
        DependencyType.AFTERANY,
        JobIdentity("12000", cluster="alpha"),
    )

    scheduler.submit(resources, worker_command, _TOKEN, dependency=dependency)

    assert scheduler.submissions[0].dependency == dependency


def test_submit_rejects_malformed_or_wrong_cluster_output(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    malformed = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, "not-a-job\n")]), username="tester"
    )
    with pytest.raises(SlurmCommandError, match="unexpected sbatch"):
        malformed.submit(resources, worker_command, _TOKEN)

    wrong_cluster = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, "12345;beta\n")]),
        cluster="alpha",
        username="tester",
    )
    with pytest.raises(SlurmCommandError, match="expected 'alpha'"):
        wrong_cluster.submit(resources, worker_command, _TOKEN)


def test_command_failure_redacts_exported_environment(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    executor = ScriptedExecutor([CommandResult(1, stderr="submission rejected")])
    scheduler = SlurmScheduler(executor=executor, username="tester")

    with pytest.raises(SlurmCommandError) as error:
        scheduler.submit(resources, worker_command, _TOKEN, {"HF_HOME": "/private/cache"})

    assert "--export=<redacted>" in str(error.value)
    assert "/private/cache" not in str(error.value)


def test_observe_prefers_live_queue_record() -> None:
    executor = ScriptedExecutor(
        [CommandResult(0, "12345|RUNNING|node-a|staircase:attempt-00000001\n")]
    )
    scheduler = SlurmScheduler(executor=executor, username="tester")

    observation = scheduler.observe(JobIdentity("12345"))

    assert observation.status is JobStatus.RUNNING
    assert observation.reason == "node-a"
    assert observation.source is ObservationSource.QUEUE
    assert len(executor.calls) == 1


@pytest.mark.parametrize(
    ("raw_state", "expected_status"),
    [
        ("COMPLETED", JobStatus.COMPLETED),
        ("CANCELLED by 1000", JobStatus.CANCELLED),
        ("FAILED", JobStatus.FAILED),
        ("BOOT_FAIL", JobStatus.FAILED),
        ("PREEMPTED", JobStatus.PREEMPTED),
        ("NODE_FAIL", JobStatus.NODE_FAIL),
        ("OUT_OF_MEMORY", JobStatus.OUT_OF_MEMORY),
        ("TIMEOUT", JobStatus.TIMEOUT),
    ],
)
def test_observe_terminal_accounting_states(
    raw_state: str,
    expected_status: JobStatus,
) -> None:
    executor = ScriptedExecutor(
        [
            CommandResult(0, ""),
            CommandResult(0, f"12345|{raw_state}|test reason|staircase:{_TOKEN}\n"),
        ]
    )
    scheduler = SlurmScheduler(executor=executor, username="tester")

    observation = scheduler.observe(JobIdentity("12345"))

    assert observation.status is expected_status
    assert observation.status.terminal is True
    assert observation.source is ObservationSource.ACCOUNTING


def test_observe_accounting_lag_is_unknown_not_failure() -> None:
    scheduler = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, ""), CommandResult(0, "")]),
        username="tester",
    )

    observation = scheduler.observe(JobIdentity("12345"))

    assert observation.status is JobStatus.UNKNOWN
    assert observation.status.terminal is False
    assert observation.source is ObservationSource.UNKNOWN
    assert "may not yet" in (observation.reason or "")


def test_ambiguous_observation_is_unknown() -> None:
    duplicate_rows = "12345|RUNNING|node-a|x\n12345|PENDING|Resources|x\n"
    scheduler = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, duplicate_rows)]), username="tester"
    )

    observation = scheduler.observe(JobIdentity("12345"))

    assert observation.status is JobStatus.UNKNOWN
    assert observation.source is ObservationSource.QUEUE
    assert "ambiguous" in (observation.reason or "")


def test_cancel_uses_only_exact_array_element() -> None:
    executor = ScriptedExecutor([CommandResult(0)])
    scheduler = SlurmScheduler(executor=executor, cluster="alpha", username="tester")

    scheduler.cancel(JobIdentity("12345", array_task_id="7", cluster="alpha"))

    argv, input_text = executor.calls[0]
    assert argv == ("scancel", "--clusters=alpha", "12345_7")
    assert input_text is None
    assert not any("*" in value or "[" in value for value in argv)


def test_ownership_requires_exact_cluster_user_comment_and_token() -> None:
    row = f"12345|tester|staircase:{_TOKEN};label=smith\n"
    executor = ScriptedExecutor([CommandResult(0, row)])
    scheduler = SlurmScheduler(
        executor=executor,
        cluster="alpha",
        username="tester",
    )

    ownership = scheduler.verify_ownership(
        JobIdentity("12345", cluster="alpha"),
        _TOKEN,
    )

    assert ownership.matched
    assert ownership.username == "tester"
    assert ownership.source is ObservationSource.QUEUE
    assert executor.calls[0][0] == (
        "squeue",
        "--noheader",
        "--jobs=12345",
        "--format=%i|%u|%k",
        "--clusters=alpha",
    )


def test_ownership_mismatch_fails_closed_after_accounting_lookup() -> None:
    executor = ScriptedExecutor(
        [
            CommandResult(0, ""),
            CommandResult(0, f"12345|other-user|staircase:{_TOKEN};label=smith\n"),
        ]
    )
    scheduler = SlurmScheduler(executor=executor, username="tester")

    ownership = scheduler.verify_ownership(JobIdentity("12345"), _TOKEN)

    assert not ownership.matched
    assert "other-user" in ownership.reason
    assert ownership.source is ObservationSource.ACCOUNTING

    cluster_mismatch = SlurmScheduler(
        executor=ScriptedExecutor([]),
        cluster="alpha",
        username="tester",
    ).verify_ownership(JobIdentity("12345", cluster="beta"), _TOKEN)
    assert not cluster_mismatch.matched
    assert "cluster" in cluster_mismatch.reason


def test_fake_owned_observation_accepts_exact_scheduler_metadata(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    scheduler = FakeScheduler(cluster="alpha", username="tester")
    identity = scheduler.submit(resources, worker_command, _TOKEN)

    observation = scheduler.observe_owned(identity, _TOKEN)

    assert observation.status is JobStatus.PENDING
    assert observation.source is ObservationSource.QUEUE


@pytest.mark.parametrize(
    "mutation",
    ["token", "user", "comment", "name", "cluster"],
)
def test_fake_owned_observation_rejects_reused_or_forged_job_metadata(
    resources: ResourceRequest,
    worker_command: InternalCommand,
    mutation: str,
) -> None:
    scheduler = FakeScheduler(cluster="alpha", username="tester")
    identity = scheduler.submit(resources, worker_command, _TOKEN)
    observed_identity = identity
    username = "tester"
    comment = f"staircase:{_TOKEN};label={resources.label}"
    job_name = f"staircase-{_TOKEN}"
    observed_token = _TOKEN
    if mutation == "token":
        observed_token = "attempt-00000002"
    elif mutation == "user":
        username = "other-user"
    elif mutation == "comment":
        comment = "unowned-comment"
    elif mutation == "name":
        job_name = "staircase-another-job"
    else:
        observed_identity = JobIdentity(identity.job_id, cluster="beta")
    scheduler.set_ownership_evidence(
        identity,
        observed_identity=observed_identity,
        username=username,
        comment=comment,
        job_name=job_name,
    )

    observation = scheduler.observe_owned(identity, observed_token)

    assert observation.status is JobStatus.UNKNOWN
    assert observation.source is ObservationSource.UNKNOWN


def test_single_node_placement_prefers_exact_live_queue_evidence() -> None:
    row = f"12345|tester|staircase:{_TOKEN};label=smith|1|node-a\n"
    executor = ScriptedExecutor([CommandResult(0, row)])
    scheduler = SlurmScheduler(executor=executor, cluster="alpha", username="tester")

    evidence = scheduler.verify_single_node_placement(
        JobIdentity("12345", cluster="alpha"),
        _TOKEN,
    )

    assert evidence.matched
    assert evidence.node_count == 1
    assert evidence.node_list == "node-a"
    assert evidence.source is ObservationSource.QUEUE
    assert len(executor.calls) == 1
    assert executor.calls[0][0] == (
        "squeue",
        "--noheader",
        "--jobs=12345",
        "--format=%i|%u|%k|%D|%N",
        "--clusters=alpha",
    )


def test_single_node_placement_falls_back_to_exact_accounting_allocation() -> None:
    row = f"12345_7|tester|staircase:{_TOKEN};label=smith|1|node-a\n"
    executor = ScriptedExecutor([CommandResult(0, ""), CommandResult(0, row)])
    scheduler = SlurmScheduler(executor=executor, cluster="alpha", username="tester")
    identity = JobIdentity("12345", array_task_id="7", cluster="alpha")

    evidence = scheduler.verify_single_node_placement(identity, _TOKEN)

    assert evidence.matched
    assert evidence.identity == identity
    assert evidence.source is ObservationSource.ACCOUNTING
    assert executor.calls[1][0] == (
        "sacct",
        "--noheader",
        "--parsable2",
        "--allocations",
        "--jobs=12345_7",
        "--format=JobIDRaw,User,Comment,NNodes,NodeList",
        "--clusters=alpha",
    )


def test_single_node_placement_accounting_lag_is_unmatched_unknown() -> None:
    executor = ScriptedExecutor([CommandResult(0, ""), CommandResult(0, "")])
    scheduler = SlurmScheduler(executor=executor, username="tester")

    evidence = scheduler.verify_single_node_placement(JobIdentity("12345"), _TOKEN)

    assert not evidence.matched
    assert evidence.source is ObservationSource.UNKNOWN
    assert evidence.node_list is None


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ("malformed\n", "malformed"),
        (
            f"12345|tester|staircase:{_TOKEN}|1|node-a\n12345|tester|staircase:{_TOKEN}|1|node-a\n",
            "ambiguous",
        ),
        (f"12346|tester|staircase:{_TOKEN}|1|node-a\n", "exact job identity"),
        (f"12345|other|staircase:{_TOKEN}|1|node-a\n", "owner"),
        ("12345|tester|staircase:attempt-00000002|1|node-a\n", "submission token"),
        (f"12345|tester|staircase:{_TOKEN}|2|node-a,node-b\n", "canonical"),
        (f"12345|tester|staircase:{_TOKEN}|1|node[01-02]\n", "compressed"),
    ],
)
def test_nonempty_untrusted_queue_placement_fails_without_stale_accounting_fallback(
    rows: str,
    reason: str,
) -> None:
    historical = f"12345|tester|staircase:{_TOKEN}|1|historical-node\n"
    executor = ScriptedExecutor([CommandResult(0, rows), CommandResult(0, historical)])
    scheduler = SlurmScheduler(executor=executor, username="tester")

    evidence = scheduler.verify_single_node_placement(JobIdentity("12345"), _TOKEN)

    assert not evidence.matched
    assert reason in evidence.reason
    assert evidence.source is ObservationSource.QUEUE
    assert len(executor.calls) == 1


def test_fake_scheduler_placement_requires_exact_identity_cluster_and_token() -> None:
    scheduler = FakeScheduler(cluster="alpha", username="tester")
    identity = JobIdentity("12345", cluster="alpha")
    scheduler.set_placement(identity, _TOKEN, "node-a")

    exact = scheduler.verify_single_node_placement(identity, _TOKEN)
    wrong_token = scheduler.verify_single_node_placement(identity, "attempt-00000002")
    missing = scheduler.verify_single_node_placement(
        JobIdentity("12346", cluster="alpha"),
        _TOKEN,
    )
    wrong_cluster = scheduler.verify_single_node_placement(
        JobIdentity("12345", cluster="beta"),
        _TOKEN,
    )

    assert exact.matched and exact.node_list == "node-a"
    assert not wrong_token.matched
    assert not missing.matched
    assert not wrong_cluster.matched


def test_lookup_adopts_unique_live_submission() -> None:
    row = f"12345|RUNNING|node-a|staircase:{_TOKEN};label=smith-attention\n"
    executor = ScriptedExecutor([CommandResult(0, row)])
    scheduler = SlurmScheduler(executor=executor, username="tester")

    lookup = scheduler.lookup_submission(_TOKEN)

    assert lookup.identity == JobIdentity("12345")
    assert lookup.status is JobStatus.RUNNING
    assert lookup.source is ObservationSource.QUEUE
    assert executor.calls[0][0][3] == f"--name=staircase-{_TOKEN}"
    assert "--format=%i|%T|%R|%k" in executor.calls[0][0]


def test_lookup_falls_back_to_accounting() -> None:
    row = f"12345|COMPLETED|None|staircase:{_TOKEN};label=smith-attention\n"
    executor = ScriptedExecutor([CommandResult(0, ""), CommandResult(0, row)])
    scheduler = SlurmScheduler(executor=executor, username="tester")

    lookup = scheduler.lookup_submission(_TOKEN)

    assert lookup.identity == JobIdentity("12345")
    assert lookup.status is JobStatus.COMPLETED
    assert lookup.source is ObservationSource.ACCOUNTING
    accounting_argv = executor.calls[1][0]
    assert "--allocations" in accounting_argv
    assert "--user=tester" in accounting_argv


def test_lookup_ambiguity_and_accounting_lag_are_unknown() -> None:
    rows = (
        f"12345|RUNNING|node-a|staircase:{_TOKEN};label=smith\n"
        f"12346|PENDING|Resources|staircase:{_TOKEN};label=smith\n"
    )
    ambiguous = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, rows)]), username="tester"
    ).lookup_submission(_TOKEN)
    assert ambiguous.identity is None
    assert ambiguous.status is JobStatus.UNKNOWN
    assert len(ambiguous.matches) == 2

    lag = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, ""), CommandResult(0, "")]),
        username="tester",
    ).lookup_submission(_TOKEN)
    assert lag.identity is None
    assert lag.status is JobStatus.UNKNOWN
    assert lag.source is ObservationSource.UNKNOWN


def test_lookup_ignores_similar_but_nonmatching_token() -> None:
    row = "12345|RUNNING|node-a|staircase:attempt-00000002;label=smith\n"
    scheduler = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, row), CommandResult(0, "")]),
        username="tester",
    )

    lookup = scheduler.lookup_submission(_TOKEN)

    assert lookup.identity is None
    assert lookup.status is JobStatus.UNKNOWN


def test_strict_probe_unions_queue_and_accounting_and_detects_duplicates() -> None:
    job_name = f"staircase-{_TOKEN}"
    queue = f"12345|RUNNING|node-a|staircase:{_TOKEN};label=smith|tester|{job_name}\n"
    accounting = (
        f"12345|RUNNING|node-a|staircase:{_TOKEN};label=smith|tester|{job_name}\n"
        f"12346|COMPLETED|None|staircase:{_TOKEN};label=smith|tester|{job_name}\n"
    )
    executor = ScriptedExecutor([CommandResult(0, queue), CommandResult(0, accounting)])
    scheduler = SlurmScheduler(
        executor=executor,
        cluster="alpha",
        username="tester",
        submission_absence_certified=True,
    )

    probe = scheduler.probe_submission(_TOKEN)

    assert probe.identity is None
    assert probe.trustworthy
    assert probe.matches == (
        JobIdentity("12345", cluster="alpha"),
        JobIdentity("12346", cluster="alpha"),
    )
    assert len(executor.calls) == 2
    assert "--format=%i|%T|%R|%k|%u|%j" in executor.calls[0][0]
    assert "--format=JobIDRaw,State,Reason,Comment,User,JobName" in executor.calls[1][0]
    assert "--clusters=alpha" in executor.calls[0][0]
    assert "--clusters=alpha" in executor.calls[1][0]


def test_owned_observation_reuses_strict_two_surface_token_evidence() -> None:
    job_name = f"staircase-{_TOKEN}"
    row = f"12345|RUNNING|node-a|staircase:{_TOKEN};label=smith|tester|{job_name}\n"
    executor = ScriptedExecutor([CommandResult(0, row), CommandResult(0, row)])
    scheduler = SlurmScheduler(
        executor=executor,
        cluster="alpha",
        username="tester",
        submission_absence_certified=True,
    )

    observation = scheduler.observe_owned(
        JobIdentity("12345", cluster="alpha"),
        _TOKEN,
    )

    assert observation.status is JobStatus.RUNNING
    assert observation.source is ObservationSource.QUEUE
    assert "--format=%i|%T|%R|%k|%u|%j" in executor.calls[0][0]
    assert "--format=JobIDRaw,State,Reason,Comment,User,JobName" in executor.calls[1][0]


def test_owned_observation_rejects_job_id_reuse_with_wrong_name() -> None:
    wrong_name = "staircase-reused-job"
    row = f"12345|RUNNING|node-a|staircase:{_TOKEN};label=smith|tester|{wrong_name}\n"
    scheduler = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, row), CommandResult(0, row)]),
        username="tester",
        submission_absence_certified=True,
    )

    observation = scheduler.observe_owned(JobIdentity("12345"), _TOKEN)

    assert observation.status is JobStatus.UNKNOWN
    assert observation.source is ObservationSource.UNKNOWN


def test_strict_probe_proves_only_certified_clean_two_surface_absence() -> None:
    uncertified = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, ""), CommandResult(0, "")]),
        username="tester",
    ).probe_submission(_TOKEN)
    assert not uncertified.absence_proven
    assert "retention" in (uncertified.reason or "")

    certified = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, ""), CommandResult(0, "")]),
        username="tester",
        submission_absence_certified=True,
    ).probe_submission(_TOKEN)
    assert certified.absence_proven


@pytest.mark.parametrize(
    "row",
    (
        "malformed\n",
        f"12345|RUNNING|node|staircase:{_TOKEN};label=x|other|staircase-{_TOKEN}\n",
        f"12345|RUNNING|node|staircase:other-token;label=x|tester|staircase-{_TOKEN}\n",
        f"invalid|RUNNING|node|staircase:{_TOKEN};label=x|tester|staircase-{_TOKEN}\n",
    ),
)
def test_strict_probe_treats_nonempty_untrusted_rows_as_blocked(row: str) -> None:
    probe = SlurmScheduler(
        executor=ScriptedExecutor([CommandResult(0, row), CommandResult(0, "")]),
        username="tester",
        submission_absence_certified=True,
    ).probe_submission(_TOKEN)

    assert not probe.absence_proven
    assert not probe.trustworthy


def test_fake_scheduler_lifecycle_and_exact_cancel(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    scheduler = FakeScheduler(first_job_id=200)
    identity = scheduler.submit(resources, worker_command, _TOKEN, {"PYTHONUNBUFFERED": "1"})

    assert identity == JobIdentity("200")
    assert scheduler.observe(identity).status is JobStatus.PENDING
    assert scheduler.lookup_submission(_TOKEN).identity == identity

    scheduler.transition(identity, JobStatus.RUNNING, reason="node-a")
    assert scheduler.observe(identity).status is JobStatus.RUNNING
    scheduler.transition(
        identity,
        JobStatus.COMPLETED,
        source=ObservationSource.ACCOUNTING,
    )
    assert scheduler.observe(identity).status is JobStatus.COMPLETED

    scheduler.cancel(identity)
    assert scheduler.cancelled == (identity,)
    assert scheduler.observe(identity).status is JobStatus.CANCELLED


def test_fake_scheduler_models_lag_and_duplicate_submission(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    scheduler = FakeScheduler()
    first = scheduler.submit(resources, worker_command, _TOKEN)
    scheduler.transition(first, JobStatus.PENDING, visible=False)
    assert scheduler.observe(first).status is JobStatus.UNKNOWN
    assert scheduler.lookup_submission(_TOKEN).status is JobStatus.UNKNOWN

    scheduler.transition(first, JobStatus.RUNNING, visible=True)
    second = scheduler.submit(resources, worker_command, _TOKEN)
    lookup = scheduler.lookup_submission(_TOKEN)
    assert lookup.identity is None
    assert lookup.status is JobStatus.UNKNOWN
    assert lookup.matches == (first, second)


def test_fake_scheduler_rejects_unknown_cancel() -> None:
    with pytest.raises(SchedulerError, match="unknown fake job"):
        FakeScheduler().cancel(JobIdentity("999"))


def test_protocol_environment_input_is_a_mapping(
    resources: ResourceRequest,
    worker_command: InternalCommand,
) -> None:
    """Document that arbitrary mapping implementations normalize identically."""

    class Environment(Mapping[str, str]):
        def __getitem__(self, key: str) -> str:
            return {"HF_HOME": "/cache"}[key]

        def __iter__(self) -> Iterator[str]:
            return iter(("HF_HOME",))

        def __len__(self) -> int:
            return 1

    scheduler = FakeScheduler()
    scheduler.submit(resources, worker_command, _TOKEN, Environment())
    assert scheduler.submissions[0].environment == (("HF_HOME", "/cache"),)
