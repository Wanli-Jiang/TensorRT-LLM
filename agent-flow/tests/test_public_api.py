# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the lazy public API exposed by :mod:`agent_flow`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PUBLIC_NAMES = {
    "AgentLayer",
    "AgentLayerConfig",
    "AgentRequest",
    "AgentResponse",
    "AgentTextEvent",
    "BackendConfig",
    "CLAUDE_CODE_DEFAULT_MODEL",
    "CODEX_DEFAULT_MODEL",
    "CompactBoundaryEvent",
    "Module",
    "RateLimitWarningEvent",
    "Sequential",
    "ServerToolCallEvent",
    "SessionConfig",
    "SessionInitEvent",
    "ThinkingEvent",
    "ToolCallEvent",
    "UsageInfo",
    "called_required_tool_this_turn",
    "require_tool_call_stop_hook",
}


def test_root_import_preserves_public_api_without_loading_agent_sdk() -> None:
    script = f"""
import sys

sys.path.insert(0, {_PROJECT_ROOT.as_posix()!r})
import agent_flow

expected = {_PUBLIC_NAMES!r}
assert set(agent_flow.__all__) == expected
assert expected <= set(dir(agent_flow))
assert "anyio" not in sys.modules
assert "claude_agent_sdk" not in sys.modules

from agent_flow import AgentRequest, BackendConfig, SessionConfig
from agent_flow.config import BackendConfig as BackendConfigImplementation
from agent_flow.types import AgentRequest as AgentRequestImplementation

assert AgentRequest is AgentRequestImplementation
assert BackendConfig is BackendConfigImplementation
assert SessionConfig.__module__ == "agent_flow.config"
assert "anyio" not in sys.modules
assert "claude_agent_sdk" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
