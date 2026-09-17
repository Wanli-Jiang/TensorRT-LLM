# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public command-line interface for the Staircase workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


def _add_start_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task", type=Path, required=True, help="Versioned Staircase task YAML.")
    parser.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Shared durable workspace visible from controller and worker nodes.",
    )
    parser.add_argument(
        "--execution",
        choices=("slurm", "local"),
        default="slurm",
        help="Execution plane. Slurm is production; local is CPU development/test only.",
    )
    parser.add_argument(
        "--preflight-facts",
        type=Path,
        help=(
            "Strict JSON manifest of independently observed repository, workspace, image, "
            "build, and mount facts. Required by production Slurm execution."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staircase",
        description="Run the Slurm-first Staircase control plane over TensorRT-LLM ModelingV2.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    onboard = commands.add_parser("onboard", help="Bring up and validate a ModelingV2 target.")
    _add_start_arguments(onboard)

    tune = commands.add_parser("tune", help="Tune an integrated ModelingV2 target.")
    _add_start_arguments(tune)

    status = commands.add_parser(
        "status", help="Read workflow and scheduler status without mutation."
    )
    status.add_argument("--workspace", type=Path, required=True)
    status.add_argument("--json", action="store_true", dest="as_json")

    cancel = commands.add_parser("cancel", help="Request cancellation of exactly owned jobs.")
    cancel.add_argument("--workspace", type=Path, required=True)
    cancel.add_argument("--reason", default="operator requested cancellation")

    respond = commands.add_parser("respond", help="Answer a durable human-input request.")
    respond.add_argument("--workspace", type=Path, required=True)
    respond.add_argument("--request-id", required=True)
    respond.add_argument("--response-file", type=Path, required=True)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the public CLI exactly once."""
    return _build_parser().parse_args(argv)


def _run_start(args: argparse.Namespace) -> int:
    from .task_schema import TaskSchemaError, load_and_normalize_task

    try:
        task = load_and_normalize_task(args.task, execution_override=args.execution)
    except (OSError, TaskSchemaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    from .workflow import load_observed_preflight_facts, start_run

    facts_provider = None
    if args.execution == "slurm" and args.preflight_facts is not None:
        facts = load_observed_preflight_facts(args.preflight_facts)

        def facts_provider(_task: object, _phase: str):  # type: ignore[no-untyped-def]
            return facts

    start_run(
        mode=args.command,
        task=task,
        workspace=args.workspace,
        preflight_facts_provider=facts_provider,
    )
    return 0


def _run_status(args: argparse.Namespace) -> int:
    from .workflow import render_status

    print(render_status(args.workspace, as_json=args.as_json))
    return 0


def _run_cancel(args: argparse.Namespace) -> int:
    from .workflow import request_cancellation

    request_cancellation(args.workspace, reason=args.reason)
    return 0


def _run_respond(args: argparse.Namespace) -> int:
    try:
        response = args.response_file.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"error: unable to read response file: {exc}", file=sys.stderr)
        return 2
    if not response.strip():
        print("error: response file must not be empty", file=sys.stderr)
        return 2

    from .workflow import record_human_response

    record_human_response(args.workspace, request_id=args.request_id, response=response)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    """Run one public Staircase command."""
    args = _parse_args(argv)
    handlers = {
        "onboard": _run_start,
        "tune": _run_start,
        "status": _run_status,
        "cancel": _run_cancel,
        "respond": _run_respond,
    }
    try:
        result = handlers[args.command](args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        result = 2
    raise SystemExit(result)


if __name__ == "__main__":
    main()
