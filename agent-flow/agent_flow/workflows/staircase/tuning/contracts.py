# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-neutral input contracts for one-variable Staircase tuning."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_KNOB = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,255}$")
_ROUTE_OR_EXECUTION_IDENTITY_SEGMENTS = frozenset(
    {
        "architecture",
        "attention_data_parallel_size",
        "checkpoint",
        "checkpoint_id",
        "expected_route",
        "family",
        "mapping",
        "model_class",
        "moe_expert_parallel_size",
        "moe_tensor_parallel_size",
        "pipeline_parallel_size",
        "route",
        "router_index",
        "synthetic_target",
        "target",
        "tensor_parallel_size",
        "topology",
        "world_size",
    }
)


class TuningContractError(ValueError):
    """Raised when tuning evidence or a controller decision is unsafe."""


class KnobKind(str, Enum):
    """Supported kinds of one-variable tuning changes."""

    CODE = "code"
    CONFIGURATION = "configuration"


class MetricDirection(str, Enum):
    """Direction in which a benchmark metric improves."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True, kw_only=True)
class KnobChange:
    """One scalar code or configuration variable changed by a hypothesis."""

    name: str
    kind: KnobKind
    baseline_value: JsonPrimitive
    candidate_value: JsonPrimitive

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SAFE_KNOB.fullmatch(self.name):
            raise TuningContractError("knob name must be a safe dotted identifier")
        segments = self.name.lower().split(".")
        if any(segment in _ROUTE_OR_EXECUTION_IDENTITY_SEGMENTS for segment in segments):
            raise TuningContractError(
                "tuning knobs cannot change route, checkpoint, target, or topology identity"
            )
        if not isinstance(self.kind, KnobKind):
            raise TuningContractError("knob kind must be code or configuration")
        _validate_json_primitive("baseline knob value", self.baseline_value)
        _validate_json_primitive("candidate knob value", self.candidate_value)
        if type(self.baseline_value) is not type(self.candidate_value):
            raise TuningContractError("baseline and candidate knob values must have the same type")
        if self.baseline_value == self.candidate_value:
            raise TuningContractError("a tuning knob must actually change")


@dataclass(frozen=True, kw_only=True)
class UncertaintyRule:
    """Benchmark-specific absolute effect and uncertainty thresholds."""

    minimum_effect: float
    noise_threshold: float
    maximum_combined_uncertainty: float

    def __post_init__(self) -> None:
        for name, value in (
            ("minimum_effect", self.minimum_effect),
            ("noise_threshold", self.noise_threshold),
            ("maximum_combined_uncertainty", self.maximum_combined_uncertainty),
        ):
            _require_non_negative_finite(name, value)

    @property
    def decisive_effect(self) -> float:
        """Return the minimum improvement that can be kept."""
        return max(self.minimum_effect, self.noise_threshold)


@dataclass(frozen=True, kw_only=True)
class TuningHypothesis:
    """One typed Tuner hypothesis with exactly one changed variable."""

    hypothesis_id: str
    item_id: str
    statement: str
    metric: str
    direction: MetricDirection
    changes: tuple[KnobChange, ...]
    uncertainty: UncertaintyRule

    def __post_init__(self) -> None:
        _require_safe_id("hypothesis_id", self.hypothesis_id)
        _require_safe_id("item_id", self.item_id)
        _require_text("hypothesis statement", self.statement)
        _require_text("metric", self.metric)
        if not isinstance(self.direction, MetricDirection):
            raise TuningContractError("direction must be a MetricDirection")
        if len(self.changes) != 1:
            raise TuningContractError("a tuning hypothesis must change exactly one variable")
        if not isinstance(self.changes[0], KnobChange):
            raise TuningContractError("hypothesis changes must be KnobChange values")
        if not isinstance(self.uncertainty, UncertaintyRule):
            raise TuningContractError("uncertainty must be an UncertaintyRule")

    @property
    def change(self) -> KnobChange:
        """Return the hypothesis's one changed variable."""
        return self.changes[0]


def _require_safe_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise TuningContractError(f"{name} must be a safe non-empty identifier")


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TuningContractError(f"{name} must be non-empty")


def _require_finite_number(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TuningContractError(f"{name} must be a finite number")


def _require_non_negative_finite(name: str, value: float) -> None:
    _require_finite_number(name, value)
    if value < 0:
        raise TuningContractError(f"{name} must be non-negative")


def _validate_json_primitive(name: str, value: JsonPrimitive) -> None:
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise TuningContractError(f"{name} must be a JSON primitive")
    if isinstance(value, float) and not math.isfinite(value):
        raise TuningContractError(f"{name} cannot contain NaN or infinity")


__all__ = [
    "JsonPrimitive",
    "KnobChange",
    "KnobKind",
    "MetricDirection",
    "TuningContractError",
    "TuningHypothesis",
    "UncertaintyRule",
]
