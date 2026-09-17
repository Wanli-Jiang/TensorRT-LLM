# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed ModelingV2 gate plans and evidence-scope validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import PurePosixPath
from typing import Mapping, Sequence

MODELING_V2_REQUIRED_ENV = (("TRTLLM_MODELING_V2", "require"),)

_STATIC_TESTS = (
    "tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py",
    "tests/unittest/_torch/modeling_v2/test_modeling_v2_routing.py",
    "tests/unittest/_torch/modeling_v2/test_modeling_v2_no_stale_claims.py",
)
_NATIVE_CONTRACT_TEST = "tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py"
_SAFE_GATE_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
_SAFE_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_NODE_PART = re.compile(r"^[A-Za-z0-9_.+\-\[\],=]+$")


class GatePolicyError(ValueError):
    """Raised when a gate or receipt would overstate its evidence."""


class GatePhase(IntEnum):
    """Required ModelingV2 gate order."""

    CLAIMS_ROUTING_NO_STALE = 10
    NATIVE_CONTRACT = 20
    ENTRY_GPU = 30
    COLLECTIVE = 40
    BOOT = 50
    ACCURACY = 60
    FEATURE_SIGNAL = 70
    PERFORMANCE = 80


class GatePurpose(str, Enum):
    """Kind of proposition a gate can support."""

    STRUCTURE = "structure"
    CORRECTNESS = "correctness"
    ACCURACY = "accuracy"
    FEATURE = "feature"
    PERFORMANCE = "performance"


class EvidenceScope(str, Enum):
    """Execution scope represented by a receipt."""

    CPU_STATIC = "cpu_static"
    SINGLE_GPU_PRODUCT = "single_gpu_product"
    LOCAL_FOUR_GPU_PRODUCT = "local_four_gpu_product"
    SYNTHETIC_MULTI_NODE_RUNNER = "synthetic_multi_node_runner"
    REAL_MULTI_NODE_PRODUCT = "real_multi_node_product"


class ClaimScope(str, Enum):
    """Scope of a claim a receipt is proposed to support."""

    STRUCTURAL = "structural"
    PRODUCT_CORRECTNESS = "product_correctness"
    ACCURACY = "accuracy"
    LOCAL_FOUR_GPU_PRODUCT = "local_four_gpu_product"
    SYNTHETIC_RUNNER = "synthetic_runner"
    MULTI_NODE_PRODUCT = "multi_node_product"
    PERFORMANCE = "performance"


@dataclass(frozen=True, slots=True)
class AccuracyCriteria:
    """Frozen accuracy selector and acceptance semantics from the task.

    Args:
        selector: Exact pytest file or node selector implementing the protocol.
        reference: Immutable reference identity or artifact named by the task.
        protocol: Non-prose protocol identifier interpreted by the accuracy test.
        tolerance: Maximum task-approved non-negative comparison tolerance.
    """

    selector: str
    reference: str
    protocol: str
    tolerance: float

    def __post_init__(self) -> None:
        _require_pytest_selector("accuracy selector", self.selector)
        for name, value in (("reference", self.reference), ("protocol", self.protocol)):
            if not isinstance(value, str) or not value.strip() or "\x00" in value or "\n" in value:
                raise GatePolicyError(f"accuracy {name} must be a non-empty single-line value")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not math.isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise GatePolicyError("accuracy tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class GateCommand:
    """Shell-free process invocation with an explicit environment."""

    argv: tuple[str, ...]
    environment: tuple[tuple[str, str], ...] = MODELING_V2_REQUIRED_ENV

    def __post_init__(self) -> None:
        if not self.argv or not all(
            isinstance(argument, str) and argument for argument in self.argv
        ):
            raise GatePolicyError("gate argv must contain non-empty strings")
        if any("\x00" in argument or "\n" in argument for argument in self.argv):
            raise GatePolicyError("gate argv cannot contain NUL or newline characters")
        names: set[str] = set()
        for name, value in self.environment:
            if not _SAFE_ENV_NAME.fullmatch(name) or "\x00" in value or "\n" in value:
                raise GatePolicyError("gate environment contains an unsafe name or value")
            if name in names:
                raise GatePolicyError(f"duplicate gate environment variable {name!r}")
            names.add(name)
        if dict(self.environment).get("TRTLLM_MODELING_V2") != "require":
            raise GatePolicyError("every ModelingV2 gate must set TRTLLM_MODELING_V2=require")


@dataclass(frozen=True, slots=True)
class GateSpec:
    """One deterministic gate in an ordered release suite."""

    gate_id: str
    phase: GatePhase
    purpose: GatePurpose
    command: GateCommand
    hard_gate: bool = True
    accuracy: AccuracyCriteria | None = None

    def __post_init__(self) -> None:
        if not _SAFE_GATE_ID.fullmatch(self.gate_id):
            raise GatePolicyError("gate_id must be a safe lowercase identifier")
        if not isinstance(self.phase, GatePhase) or not isinstance(self.purpose, GatePurpose):
            raise GatePolicyError("gate phase and purpose must use their typed enums")
        if not isinstance(self.command, GateCommand) or not isinstance(self.hard_gate, bool):
            raise GatePolicyError("gate command and hard_gate must use their typed values")
        if self.purpose is GatePurpose.PERFORMANCE and self.hard_gate:
            raise GatePolicyError(
                "performance evidence cannot be a universal ModelingV2 correctness gate"
            )
        if (self.purpose is GatePurpose.ACCURACY) != (self.accuracy is not None):
            raise GatePolicyError("only an accuracy gate must carry accuracy criteria")


@dataclass(frozen=True, slots=True)
class RankPlacement:
    """Explicit placement of one product or canary rank."""

    rank: int
    node: str
    local_rank: int

    def __post_init__(self) -> None:
        for name, value in (("rank", self.rank), ("local_rank", self.local_rank)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise GatePolicyError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.node, str)
            or not self.node.strip()
            or any(character in self.node for character in "\x00\n")
        ):
            raise GatePolicyError("node must be a non-empty scheduler node name")


@dataclass(frozen=True, slots=True)
class GateReceipt:
    """Topology and purpose carried by one immutable gate result."""

    gate_id: str
    purpose: GatePurpose
    scope: EvidenceScope
    placements: tuple[RankPlacement, ...]
    product_rank_body: bool
    passed: bool
    accuracy: AccuracyCriteria | None = None

    def __post_init__(self) -> None:
        if not _SAFE_GATE_ID.fullmatch(self.gate_id):
            raise GatePolicyError("receipt gate_id must be a safe lowercase identifier")
        if not isinstance(self.purpose, GatePurpose) or not isinstance(self.scope, EvidenceScope):
            raise GatePolicyError("receipt purpose and scope must use typed enums")
        if not isinstance(self.product_rank_body, bool) or not isinstance(self.passed, bool):
            raise GatePolicyError("receipt product_rank_body and passed must be booleans")
        ranks = [placement.rank for placement in self.placements]
        if ranks != list(range(len(ranks))):
            raise GatePolicyError("receipt placements must enumerate every rank in rank order")
        node_rank_pairs = {(placement.node, placement.local_rank) for placement in self.placements}
        if len(node_rank_pairs) != len(self.placements):
            raise GatePolicyError("receipt placements contain a duplicate node-local rank")
        node_count = len({placement.node for placement in self.placements})
        if self.scope is EvidenceScope.CPU_STATIC and self.placements:
            raise GatePolicyError("CPU-static evidence cannot claim rank placement")
        if self.scope is EvidenceScope.SINGLE_GPU_PRODUCT and (
            len(self.placements) != 1 or node_count != 1 or not self.product_rank_body
        ):
            raise GatePolicyError("single-GPU product evidence requires one real product rank")
        if self.scope is EvidenceScope.LOCAL_FOUR_GPU_PRODUCT and (
            len(self.placements) != 4 or node_count != 1 or not self.product_rank_body
        ):
            raise GatePolicyError("local four-GPU evidence requires four product ranks on one node")
        if self.scope is EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER and (
            len(self.placements) < 2 or node_count < 2 or self.product_rank_body
        ):
            raise GatePolicyError(
                "synthetic runner evidence requires multiple nodes and a non-product rank body"
            )
        if self.scope is EvidenceScope.REAL_MULTI_NODE_PRODUCT and (
            len(self.placements) < 2 or node_count < 2 or not self.product_rank_body
        ):
            raise GatePolicyError(
                "real multi-node evidence requires topology-bearing product rank placements"
            )
        if (self.purpose is GatePurpose.ACCURACY) != (self.accuracy is not None):
            raise GatePolicyError("only an accuracy receipt must carry accuracy criteria")


def build_gate_suite(
    *,
    entry_gpu_tests: Sequence[str],
    collective_tests: Sequence[str],
    boot_tests: Sequence[str],
    accuracy: AccuracyCriteria,
    feature_signal_tests: Sequence[str],
    base_environment: Mapping[str, str] | None = None,
) -> tuple[GateSpec, ...]:
    """Build the ordered deterministic ModelingV2 acceptance suite.

    Args:
        entry_gpu_tests: Focused catalog/component GPU pytest selectors.
        collective_tests: Product collective pytest selectors, if applicable.
        boot_tests: Target boot pytest selectors.
        accuracy: Task-selected test selector and typed acceptance criteria.
        feature_signal_tests: Feature-specific acceptance pytest selectors.
        base_environment: Additional non-secret environment for every command.

    Returns:
        Gates ordered from cheapest structural checks through feature signals.
    """
    if not isinstance(accuracy, AccuracyCriteria):
        raise GatePolicyError("accuracy must use AccuracyCriteria")
    environment = _normalize_environment(base_environment)
    specs = [
        _pytest_gate(
            "claims-routing-no-stale",
            GatePhase.CLAIMS_ROUTING_NO_STALE,
            GatePurpose.STRUCTURE,
            _STATIC_TESTS,
            environment,
        ),
        _pytest_gate(
            "native-contract",
            GatePhase.NATIVE_CONTRACT,
            GatePurpose.CORRECTNESS,
            (_NATIVE_CONTRACT_TEST,),
            environment,
        ),
    ]
    if entry_gpu_tests:
        specs.append(
            _pytest_gate(
                "entry-gpu",
                GatePhase.ENTRY_GPU,
                GatePurpose.CORRECTNESS,
                entry_gpu_tests,
                environment,
            )
        )
    if collective_tests:
        specs.append(
            _pytest_gate(
                "collective",
                GatePhase.COLLECTIVE,
                GatePurpose.CORRECTNESS,
                collective_tests,
                environment,
            )
        )
    specs.append(
        _pytest_gate(
            "boot",
            GatePhase.BOOT,
            GatePurpose.CORRECTNESS,
            _require_selectors("boot_tests", boot_tests),
            environment,
        )
    )
    accuracy_gate = _pytest_gate(
        "accuracy",
        GatePhase.ACCURACY,
        GatePurpose.ACCURACY,
        (accuracy.selector,),
        environment,
        accuracy=accuracy,
    )
    specs.append(accuracy_gate)
    if feature_signal_tests:
        specs.append(
            _pytest_gate(
                "feature-signal",
                GatePhase.FEATURE_SIGNAL,
                GatePurpose.FEATURE,
                feature_signal_tests,
                environment,
            )
        )
    _validate_suite_order(specs)
    return tuple(specs)


def validate_receipt_claim(
    receipt: GateReceipt,
    claim: ClaimScope,
    *,
    gate_spec: GateSpec | None = None,
) -> None:
    """Reject a claim that is stronger than the receipt's execution scope.

    Args:
        receipt: Immutable gate result with explicit topology.
        claim: Proposed claim scope.
        gate_spec: Frozen expected gate. Required for an accuracy claim so the
            controller compares the receipt with task-owned criteria.
    """
    if not receipt.passed:
        raise GatePolicyError("a failed gate receipt cannot support a claim")
    if claim is ClaimScope.PERFORMANCE:
        if receipt.purpose is not GatePurpose.PERFORMANCE:
            raise GatePolicyError("a performance claim requires performance evidence")
        return
    if receipt.purpose is GatePurpose.PERFORMANCE:
        raise GatePolicyError("performance evidence cannot support a correctness claim")
    if claim is ClaimScope.ACCURACY:
        if receipt.purpose is not GatePurpose.ACCURACY:
            raise GatePolicyError("an accuracy claim requires accuracy evidence")
        if gate_spec is None:
            raise GatePolicyError("an accuracy claim requires its frozen gate specification")
        if gate_spec.gate_id != receipt.gate_id or gate_spec.purpose is not GatePurpose.ACCURACY:
            raise GatePolicyError("accuracy receipt does not identify the expected gate")
        if gate_spec.accuracy != receipt.accuracy:
            raise GatePolicyError("accuracy receipt criteria do not match the frozen task")
        if receipt.scope in {
            EvidenceScope.CPU_STATIC,
            EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
        }:
            raise GatePolicyError("static or synthetic evidence cannot certify accuracy")
        return
    if claim is ClaimScope.STRUCTURAL:
        return
    if claim is ClaimScope.PRODUCT_CORRECTNESS:
        if receipt.scope in {
            EvidenceScope.CPU_STATIC,
            EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER,
        }:
            raise GatePolicyError("static or synthetic evidence cannot certify product correctness")
        return
    if claim is ClaimScope.LOCAL_FOUR_GPU_PRODUCT:
        if receipt.scope is not EvidenceScope.LOCAL_FOUR_GPU_PRODUCT:
            raise GatePolicyError("local four-GPU claims require an exact local four-GPU receipt")
        return
    if claim is ClaimScope.SYNTHETIC_RUNNER:
        if receipt.scope is not EvidenceScope.SYNTHETIC_MULTI_NODE_RUNNER:
            raise GatePolicyError("runner-canary claims require synthetic multi-node evidence")
        return
    if claim is ClaimScope.MULTI_NODE_PRODUCT:
        if receipt.scope is not EvidenceScope.REAL_MULTI_NODE_PRODUCT:
            raise GatePolicyError(
                "a multi-node product claim requires a real topology-bearing product receipt"
            )
        return
    raise GatePolicyError(f"unsupported claim scope {claim!r}")


def _pytest_gate(
    gate_id: str,
    phase: GatePhase,
    purpose: GatePurpose,
    selectors: Sequence[str],
    environment: tuple[tuple[str, str], ...],
    *,
    accuracy: AccuracyCriteria | None = None,
) -> GateSpec:
    normalized = _require_selectors(gate_id, selectors)
    return GateSpec(
        gate_id=gate_id,
        phase=phase,
        purpose=purpose,
        command=GateCommand(argv=("python3", "-m", "pytest", *normalized), environment=environment),
        accuracy=accuracy,
    )


def _normalize_environment(environment: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    result = dict(environment or {})
    if not all(isinstance(name, str) and isinstance(value, str) for name, value in result.items()):
        raise GatePolicyError("gate environment names and values must be strings")
    configured_mode = result.get("TRTLLM_MODELING_V2")
    if configured_mode not in {None, "require"}:
        raise GatePolicyError("TRTLLM_MODELING_V2 cannot be overridden from require")
    result["TRTLLM_MODELING_V2"] = "require"
    return tuple(sorted(result.items()))


def _require_selectors(name: str, selectors: Sequence[str]) -> tuple[str, ...]:
    if not selectors:
        raise GatePolicyError(f"{name} must contain at least one pytest selector")
    normalized = tuple(_require_pytest_selector(name, selector) for selector in selectors)
    if len(set(normalized)) != len(normalized):
        raise GatePolicyError(f"{name} contains duplicate selectors")
    return normalized


def _require_pytest_selector(name: str, selector: str) -> str:
    if not isinstance(selector, str) or not selector or "\x00" in selector or "\n" in selector:
        raise GatePolicyError(f"{name} contains an unsafe pytest selector")
    path_text, *node_parts = selector.split("::")
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or str(path) != path_text
        or not path.parts
        or path.parts[0] != "tests"
        or ".." in path.parts
        or "." in path.parts
        or path.suffix != ".py"
    ):
        raise GatePolicyError(
            f"{name} must contain repository-relative tests/*.py selectors, not commands"
        )
    if any(not part or _SAFE_NODE_PART.fullmatch(part) is None for part in node_parts):
        raise GatePolicyError(f"{name} contains an unsafe pytest node identifier")
    return selector


def validate_pytest_selector(selector: str) -> str:
    """Validate one task-supplied repository-relative pytest selector."""
    return _require_pytest_selector("pytest selector", selector)


def _validate_suite_order(specs: Sequence[GateSpec]) -> None:
    phases = [spec.phase for spec in specs]
    if phases != sorted(phases) or len(phases) != len(set(phases)):
        raise GatePolicyError("gate suite phases must be unique and ordered")


__all__ = [
    "MODELING_V2_REQUIRED_ENV",
    "AccuracyCriteria",
    "ClaimScope",
    "EvidenceScope",
    "GateCommand",
    "GatePhase",
    "GatePolicyError",
    "GatePurpose",
    "GateReceipt",
    "GateSpec",
    "RankPlacement",
    "build_gate_suite",
    "validate_receipt_claim",
    "validate_pytest_selector",
]
