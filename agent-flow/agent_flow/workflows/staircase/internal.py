# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private process entry points for controller, role, supervisor, and rank jobs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .common.credentials import CredentialBroker
    from .task_schema import NormalizedTask


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
