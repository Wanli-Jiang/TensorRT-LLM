# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public AgentFlow API with lazily loaded implementation modules."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import (
        CLAUDE_CODE_DEFAULT_MODEL,
        CODEX_DEFAULT_MODEL,
        AgentLayerConfig,
        BackendConfig,
        SessionConfig,
    )
    from .hooks import called_required_tool_this_turn, require_tool_call_stop_hook
    from .layers import AgentLayer
    from .module import Module, Sequential
    from .types import (
        AgentRequest,
        AgentResponse,
        AgentTextEvent,
        CompactBoundaryEvent,
        RateLimitWarningEvent,
        ServerToolCallEvent,
        SessionInitEvent,
        ThinkingEvent,
        ToolCallEvent,
        UsageInfo,
    )

_LAZY_EXPORTS = {
    "AgentLayer": (".layers", "AgentLayer"),
    "AgentLayerConfig": (".config", "AgentLayerConfig"),
    "AgentRequest": (".types", "AgentRequest"),
    "AgentResponse": (".types", "AgentResponse"),
    "AgentTextEvent": (".types", "AgentTextEvent"),
    "BackendConfig": (".config", "BackendConfig"),
    "CLAUDE_CODE_DEFAULT_MODEL": (".config", "CLAUDE_CODE_DEFAULT_MODEL"),
    "CODEX_DEFAULT_MODEL": (".config", "CODEX_DEFAULT_MODEL"),
    "CompactBoundaryEvent": (".types", "CompactBoundaryEvent"),
    "Module": (".module", "Module"),
    "RateLimitWarningEvent": (".types", "RateLimitWarningEvent"),
    "Sequential": (".module", "Sequential"),
    "ServerToolCallEvent": (".types", "ServerToolCallEvent"),
    "SessionConfig": (".config", "SessionConfig"),
    "SessionInitEvent": (".types", "SessionInitEvent"),
    "ThinkingEvent": (".types", "ThinkingEvent"),
    "ToolCallEvent": (".types", "ToolCallEvent"),
    "UsageInfo": (".types", "UsageInfo"),
    "called_required_tool_this_turn": (".hooks", "called_required_tool_this_turn"),
    "require_tool_call_stop_hook": (".hooks", "require_tool_call_stop_hook"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public symbol only when a caller first requests it."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public symbols in interactive discovery."""
    return sorted(set(globals()) | set(__all__))
