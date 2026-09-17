# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private process entry points for controller, role, supervisor, and rank jobs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .common.credential_broker_server import UnixCredentialBrokerServer
    from .common.credentials import CredentialBroker
    from .task_schema import NormalizedTask


_BROKER_SHUTDOWN_TIMEOUT_SECONDS = 5.0


def _build_managed_credential_broker_server(
    task: NormalizedTask,
    *,
    workspace: Path,
) -> UnixCredentialBrokerServer | None:
    """Build the task-owned OAuth broker without reading credential contents."""
    from .common.credential_broker_server import (
        CredentialBrokerServerConfig,
        UnixCredentialBrokerServer,
    )

    controller = task.execution.slurm.controller
    source_auth_json_path = controller.source_auth_json_path
    if source_auth_json_path is None:
        return None
    policies = tuple(
        role_class.credential_broker for role_class in task.execution.slurm.role_classes
    )
    if not policies or any(policy != policies[0] for policy in policies[1:]):
        raise RuntimeError("managed credential broker policies are missing or inconsistent")
    policy = policies[0]
    if policy.allowed_credential_names != ("CODEX_AUTH_JSON",):
        raise RuntimeError("managed credential broker permits only CODEX_AUTH_JSON")
    socket_value = controller.credential_broker_socket
    if socket_value is None:
        raise RuntimeError("managed credential broker has no socket path")

    canonical_workspace = workspace.resolve(strict=True)
    socket_path = Path(socket_value)
    workspace_socket = socket_path.is_relative_to(canonical_workspace)
    private_runtime_socket = (
        socket_path.parent.parent == Path("/tmp")
        and socket_path.parent.name.startswith("staircase-credential-broker-")
        and socket_path.name == "broker.sock"
    )
    if not workspace_socket and not private_runtime_socket:
        raise RuntimeError("managed credential broker socket does not use an admitted location")
    if workspace_socket and socket_path.parent == canonical_workspace:
        raise RuntimeError("managed credential broker socket requires a private workspace child")
    try:
        socket_path.parent.mkdir(mode=0o700, parents=False, exist_ok=False)
    except FileExistsError:
        if private_runtime_socket:
            raise RuntimeError(
                "private credential broker runtime directory already exists"
            ) from None
    return UnixCredentialBrokerServer(
        CredentialBrokerServerConfig(
            socket_path=socket_path,
            broker_id=policy.broker_id,
            allowed_credential_names=policy.allowed_credential_names,
            source_auth_json_path=source_auth_json_path,
            workspace_roots=(canonical_workspace,),
        )
    )


def _build_credential_broker(task: NormalizedTask, *, workspace: Path) -> CredentialBroker:
    """Construct and preflight the one task-authorized broker instance."""
    from .common.credential_broker_client import (
        UnixCredentialBrokerClient,
        UnixCredentialBrokerConfig,
    )
    from .common.credentials import NoCredentialBroker
    from .task_schema import SlurmRole

    policies = tuple(
        role_class.credential_broker for role_class in task.execution.slurm.role_classes
    )
    if len(policies) != len(SlurmRole):
        raise RuntimeError("normalized task does not define every role credential policy")
    controller = task.execution.slurm.controller
    credentialed = any(policy.allowed_credential_names for policy in policies)
    if not credentialed:
        if controller.credential_broker_socket is not None or any(
            policy.broker_id != "preauthenticated" or policy.allowed_credential_names
            for policy in policies
        ):
            raise RuntimeError("invalid preauthenticated credential broker deployment")
        return NoCredentialBroker()

    policy = policies[0]
    if any(candidate != policy for candidate in policies[1:]):
        raise RuntimeError("role credential broker policies are not identical")
    socket_path = controller.credential_broker_socket
    if socket_path is None:
        raise RuntimeError("credentialed deployment has no broker socket")
    broker = UnixCredentialBrokerClient(
        UnixCredentialBrokerConfig(
            socket_path=Path(socket_path),
            broker_id=policy.broker_id,
            allowed_credential_names=policy.allowed_credential_names,
            attempt_ttl_seconds=policy.per_attempt_ttl_seconds,
            expected_socket_owner_uid=os.geteuid(),
            expected_socket_group_gid=os.getegid(),
        ),
        controller_workspace=workspace,
        handle_ledger_path=workspace / "credential-broker-state" / "handle-bindings.jsonl",
    )
    broker.preflight()
    return broker


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m agent_flow.workflows.staircase.internal")
    commands = parser.add_subparsers(dest="command", required=True)
    controller = commands.add_parser("controller")
    controller.add_argument("--workspace", type=Path, required=True)
    controller.add_argument("--generation", type=int, required=True)
    controller.add_argument("--owner-nonce", required=True)
    worker = commands.add_parser("role-worker")
    worker.add_argument("--input", type=Path, required=True)
    rank_worker = commands.add_parser("rank-worker")
    rank_worker.add_argument("--input", type=Path, required=True)
    rank_supervisor = commands.add_parser("rank-supervisor")
    rank_supervisor.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one private controller, role-worker, or rank-worker process."""
    args = _build_parser().parse_args(argv)
    if args.command == "controller":
        from .common.gitops import ControllerGitOps
        from .controller.collective import build_task_collective_adapter
        from .controller.runtime import ProductionRuntimeFactory
        from .task_schema import load_normalized_task
        from .workflow import NORMALIZED_TASK_FILENAME, run_controller

        task = load_normalized_task(args.workspace / NORMALIZED_TASK_FILENAME)
        server = _build_managed_credential_broker_server(task, workspace=args.workspace)
        try:
            if server is not None:
                server.start_in_background()
            credential_broker = _build_credential_broker(task, workspace=args.workspace)
            collective_git = ControllerGitOps(
                task.repository.root,
                args.workspace / "locks" / "git.lock",
                f"staircase-collective:{args.owner_nonce}",
            )
            collective_adapter = build_task_collective_adapter(
                workspace=args.workspace,
                task=task,
                git=collective_git,
            )
            run_controller(
                args.workspace,
                generation=args.generation,
                owner_nonce=args.owner_nonce,
                runtime_factory=ProductionRuntimeFactory(
                    collective_gate_adapter=collective_adapter,
                    credential_broker=credential_broker,
                ),
                credential_broker=credential_broker,
            )
        finally:
            if server is not None:
                server.shutdown()
                server.wait(_BROKER_SHUTDOWN_TIMEOUT_SECONDS)
                socket_parent = server.config.socket_path.parent
                if not socket_parent.is_relative_to(args.workspace.resolve(strict=True)):
                    try:
                        socket_parent.rmdir()
                    except FileNotFoundError:
                        pass
                    except OSError as error:
                        raise RuntimeError(
                            "private credential broker runtime directory was not empty"
                        ) from error
        return

    if args.command == "rank-worker":
        from .common.rank_worker import execute_rank

        returncode = execute_rank(args.input)
        if returncode != 0:
            raise SystemExit(returncode)
        return

    if args.command == "rank-supervisor":
        from .common.rank_supervisor_worker import execute_rank_supervisor

        execute_rank_supervisor(args.input)
        return

    from .common.worker import execute_worker, is_worker_input_manifest

    if is_worker_input_manifest(args.input):
        execute_worker(args.input)
        return

    from .common.runners import execute_role, load_role_spec

    execute_role(load_role_spec(args.input))


if __name__ == "__main__":
    main()
