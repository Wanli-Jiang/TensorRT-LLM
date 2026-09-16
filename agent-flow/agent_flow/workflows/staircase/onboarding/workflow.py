# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure ModelingV2 onboarding and Assembler orchestration contracts.

This module does not run agents, mutate product files, or contact a scheduler.
It turns a normalized task and authoritative controller state into immutable
Assembler inputs, typed catalog-gap proposals, and a static reconciliation of
the ModelingV2 routing tree.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import TypeAlias

from ..state import DomainProfile, RunState, WorkItemKind, WorkItemRecord, WorkItemStatus
from ..task_schema import NormalizedTask, ParallelMapping, ReferenceConfig, TargetConfig

MODELING_V2_ROOT = PurePosixPath("tensorrt_llm/_torch/modeling_v2")
MODELS_ROOT = MODELING_V2_ROOT / "models"
ROUTER_INDEX_PATH = MODELING_V2_ROOT / "_router_index.py"
CATALOG_ROOT = MODELING_V2_ROOT / "catalog"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PYTHON_SEGMENT = re.compile(r"^[a-z_][a-z0-9_]*$")


class OnboardingContractError(ValueError):
    """Raised when a proposed onboarding operation violates ModelingV2 policy."""


class DependencyNotIntegratedError(OnboardingContractError):
    """Raised when Assembler would consume non-integrated authoritative state."""


class ReconciliationDisposition(str, Enum):
    """Static result of reconciling one requested route with the product tree."""

    SHADOW_NOOP = "shadow_noop"
    MISSING_TARGET = "missing_target"


class AssemblerUnit(str, Enum):
    """One isolated product responsibility assigned to Assembler."""

    CORE = "core"
    FEATURE = "feature"
    WEIGHTS = "weights"
    ROUTING = "routing"


@dataclass(frozen=True, slots=True)
class MappingIdentity:
    """Every parallel axis that participates in ModelingV2 route identity."""

    world_size: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    moe_expert_parallel_size: int
    moe_tensor_parallel_size: int
    attention_data_parallel_size: int

    @classmethod
    def from_target(cls, target: TargetConfig) -> MappingIdentity:
        """Build and revalidate the complete mapping carried by a normalized task."""
        mapping = target.mapping
        identity = cls(
            world_size=target.world_size,
            tensor_parallel_size=mapping.tensor_parallel_size,
            pipeline_parallel_size=mapping.pipeline_parallel_size,
            moe_expert_parallel_size=mapping.moe_expert_parallel_size,
            moe_tensor_parallel_size=mapping.moe_tensor_parallel_size,
            attention_data_parallel_size=mapping.attention_data_parallel_size,
        )
        identity._validate(mapping)
        return identity

    def _validate(self, mapping: ParallelMapping) -> None:
        values = (
            self.world_size,
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.moe_expert_parallel_size,
            self.moe_tensor_parallel_size,
            self.attention_data_parallel_size,
        )
        if any(isinstance(value, bool) or value < 1 for value in values):
            raise OnboardingContractError("every Mapping identity axis must be positive")
        expected_world_size = self.tensor_parallel_size * self.pipeline_parallel_size
        if self.world_size != expected_world_size:
            raise OnboardingContractError(
                "world_size must equal tensor_parallel_size * pipeline_parallel_size"
            )
        if mapping != ParallelMapping(
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            moe_expert_parallel_size=self.moe_expert_parallel_size,
            moe_tensor_parallel_size=self.moe_tensor_parallel_size,
            attention_data_parallel_size=self.attention_data_parallel_size,
        ):
            raise OnboardingContractError(
                "Mapping identity does not preserve every normalized axis"
            )

    @property
    def parallel_segment(self) -> str:
        """Return a readable path segment without discarding any non-default axis."""
        tp = self.tensor_parallel_size
        if self.pipeline_parallel_size == 1:
            if (
                self.moe_expert_parallel_size == 1
                and self.moe_tensor_parallel_size == 1
                and self.attention_data_parallel_size == 1
            ):
                return f"tp{tp}"
            if self.moe_expert_parallel_size == tp and self.moe_tensor_parallel_size == 1:
                if self.attention_data_parallel_size == tp:
                    return f"dep{self.world_size}"
                if self.attention_data_parallel_size == 1:
                    return f"tep{self.world_size}"
        return (
            f"tp{tp}_pp{self.pipeline_parallel_size}_"
            f"ep{self.moe_expert_parallel_size}_mtp{self.moe_tensor_parallel_size}_"
            f"adp{self.attention_data_parallel_size}"
        )


@dataclass(frozen=True, slots=True)
class RouteIdentity:
    """Fully derived ModelingV2 identity for one deployment target."""

    architecture: str
    family: str
    checkpoint_id: str
    sm: int
    mapping: MappingIdentity
    structural_features: tuple[str, ...]
    parallel_segment: str
    target_segment: str
    family_root: PurePosixPath
    target_directory: PurePosixPath
    routing_module: str
    target_module: str
    synthetic_class: str

    @property
    def route_table_key(self) -> tuple[str, str]:
        """Return the routing-table key proposed for this target."""
        return (self.checkpoint_id, self.target_segment)

    @property
    def target_products(self) -> tuple[PurePosixPath, PurePosixPath]:
        """Return the only two product files owned by a target."""
        return (
            self.target_directory / "modeling.py",
            self.target_directory / "weights.py",
        )


def derive_route_identity(reference: ReferenceConfig, target: TargetConfig) -> RouteIdentity:
    """Derive and validate a route from normalized checkpoint and target facts.

    Args:
        reference: Normalized checkpoint identity, including ``architectures[0]``.
        target: Normalized GPU, feature, path, and complete Mapping request.

    Returns:
        An immutable route identity suitable for planning and reconciliation.
    """
    mapping = MappingIdentity.from_target(target)
    features = tuple(
        sorted(_python_slug(feature, "structural feature") for feature in target.features)
    )
    if len(set(features)) != len(features):
        raise OnboardingContractError(
            "structural feature names must remain unique after route normalization"
        )
    parallel_segment = mapping.parallel_segment
    target_segment = parallel_segment
    if features:
        target_segment = f"{parallel_segment}_{'_'.join(features)}"
    family_root = MODELS_ROOT / target.family
    target_directory = (
        family_root / "targets" / target.checkpoint_id / f"sm_{target.sm}" / target_segment
    )
    expected_route = PurePosixPath(target.expected_route)
    if expected_route != target_directory / "modeling.py" and expected_route not in (
        target_directory,
        *target_directory.parents,
    ):
        raise OnboardingContractError(
            f"expected_route {expected_route} conflicts with derived target {target_directory}"
        )
    if expected_route == MODELS_ROOT or family_root not in (
        expected_route,
        *expected_route.parents,
    ):
        raise OnboardingContractError(
            "expected_route must identify the requested architecture family"
        )

    routing_module = f"models.{target.family}.routing"
    relative_target = target_directory.relative_to(MODELING_V2_ROOT)
    target_module = ".".join((*relative_target.parts, "modeling"))
    identity_tokens = (
        target.family,
        target.checkpoint_id,
        f"sm_{target.sm}",
        target_segment,
    )
    synthetic_class = "ModelingV2" + "".join(_camel_case(token) for token in identity_tokens)
    return RouteIdentity(
        architecture=reference.architecture,
        family=target.family,
        checkpoint_id=target.checkpoint_id,
        sm=target.sm,
        mapping=mapping,
        structural_features=features,
        parallel_segment=parallel_segment,
        target_segment=target_segment,
        family_root=family_root,
        target_directory=target_directory,
        routing_module=routing_module,
        target_module=target_module,
        synthetic_class=synthetic_class,
    )


def route_identity_from_task(task: NormalizedTask) -> RouteIdentity:
    """Derive a route directly from an immutable normalized task."""
    return derive_route_identity(task.reference, task.target)


@dataclass(frozen=True, slots=True)
class CatalogRequirement:
    """One catalog surface the requested assembly must consume."""

    entry_id: str
    catalog_path: PurePosixPath
    reason: str

    def __post_init__(self) -> None:
        _require_safe_id("catalog entry_id", self.entry_id)
        _validate_catalog_path(self.catalog_path)
        if not self.reason.strip():
            raise OnboardingContractError("catalog requirement reason must be non-empty")


@dataclass(frozen=True, slots=True)
class IntegratedCatalogSurface:
    """Catalog surface attributed to one integrated Smith dependency."""

    source_item_id: str
    entry_id: str
    catalog_path: PurePosixPath

    def __post_init__(self) -> None:
        _require_safe_id("source_item_id", self.source_item_id)
        _require_safe_id("catalog entry_id", self.entry_id)
        _validate_catalog_path(self.catalog_path)


@dataclass(frozen=True, slots=True)
class CatalogGapProposal:
    """Typed Smith work proposed when Assembler lacks a required surface."""

    proposal_id: str
    consumer_item_id: str
    goal_id: str
    entry_id: str
    catalog_path: PurePosixPath
    reason: str
    kind: WorkItemKind = WorkItemKind.CATALOG_ONBOARD

    def __post_init__(self) -> None:
        for name, value in (
            ("proposal_id", self.proposal_id),
            ("consumer_item_id", self.consumer_item_id),
            ("goal_id", self.goal_id),
            ("entry_id", self.entry_id),
        ):
            _require_safe_id(name, value)
        _validate_catalog_path(self.catalog_path)
        if not self.reason.strip():
            raise OnboardingContractError("catalog gap reason must be non-empty")
        if self.kind is not WorkItemKind.CATALOG_ONBOARD:
            raise OnboardingContractError("a catalog gap must propose catalog_onboard work")


@dataclass(frozen=True, slots=True)
class AssemblyScope:
    """Common immutable input shared by every Assembler unit."""

    item_id: str
    route: RouteIdentity
    dependency_item_ids: tuple[str, ...]
    catalog_surfaces: tuple[IntegratedCatalogSurface, ...]


@dataclass(frozen=True, slots=True)
class CoreAssemblerInput:
    """Input for the target's flat modeling core."""

    scope: AssemblyScope


@dataclass(frozen=True, slots=True)
class FeatureAssemblerInput:
    """Input for one structure-changing feature increment."""

    scope: AssemblyScope
    feature: str

    def __post_init__(self) -> None:
        if self.feature not in self.scope.route.structural_features:
            raise OnboardingContractError("feature input is not part of route identity")


@dataclass(frozen=True, slots=True)
class WeightsAssemblerInput:
    """Input for the target-owned weight manifest and loader."""

    scope: AssemblyScope


@dataclass(frozen=True, slots=True)
class RoutingAssemblerInput:
    """Input for family routing and a conditional router-index update."""

    scope: AssemblyScope
    new_architecture_family: bool


AssemblerInput: TypeAlias = (
    CoreAssemblerInput | FeatureAssemblerInput | WeightsAssemblerInput | RoutingAssemblerInput
)


@dataclass(frozen=True, slots=True)
class AssemblyPreparation:
    """Either a dispatchable Assembler input or typed catalog-gap proposals."""

    assembler_input: AssemblerInput | None
    catalog_gaps: tuple[CatalogGapProposal, ...]

    def __post_init__(self) -> None:
        if (self.assembler_input is None) == (not self.catalog_gaps):
            raise OnboardingContractError(
                "assembly preparation must contain exactly one of input or catalog gaps"
            )


@dataclass(frozen=True, slots=True)
class AssemblerOutput:
    """Validated, product-only output boundary for one Assembler unit."""

    item_id: str
    unit: AssemblerUnit
    route: RouteIdentity
    changed_paths: tuple[PurePosixPath, ...]
    synthetic_class: str
    target_module: str
    route_table_key: tuple[str, str]
    updates_router_index: bool


def prepare_assembler_input(
    state: RunState,
    item_id: str,
    route: RouteIdentity,
    unit: AssemblerUnit,
    requirements: tuple[CatalogRequirement, ...] = (),
    catalog_surfaces: tuple[IntegratedCatalogSurface, ...] = (),
    *,
    feature: str | None = None,
    new_architecture_family: bool = False,
) -> AssemblyPreparation:
    """Prepare one Assembler input from authoritative integrated dependencies.

    Missing catalog surfaces are returned as typed Smith proposals. Existing
    dependency items that have not reached ``INTEGRATED`` are a hard block,
    not a gap and never a surface Assembler may consume.
    """
    try:
        item = state.item(item_id)
    except KeyError as exc:
        raise OnboardingContractError(f"unknown Assembler item {item_id!r}") from exc
    _validate_assembler_item(item, unit)
    if unit is not AssemblerUnit.FEATURE and feature is not None:
        raise OnboardingContractError("only feature assembly accepts a feature identity")
    requirement_ids = [requirement.entry_id for requirement in requirements]
    if len(set(requirement_ids)) != len(requirement_ids):
        raise OnboardingContractError("Assembler catalog requirements contain duplicate entries")
    dependencies = {dependency_id: state.item(dependency_id) for dependency_id in item.dependencies}
    not_integrated = sorted(
        dependency_id
        for dependency_id, dependency in dependencies.items()
        if dependency.status is not WorkItemStatus.INTEGRATED
    )
    if not_integrated:
        raise DependencyNotIntegratedError(
            f"Assembler item {item_id!r} has non-integrated dependencies: {not_integrated!r}"
        )

    surfaces_by_entry: dict[str, IntegratedCatalogSurface] = {}
    for surface in catalog_surfaces:
        dependency = dependencies.get(surface.source_item_id)
        if dependency is None:
            raise OnboardingContractError(
                f"catalog surface {surface.entry_id!r} is not an item dependency"
            )
        if dependency.kind not in {
            WorkItemKind.CATALOG_VERIFY,
            WorkItemKind.CATALOG_ONBOARD,
        }:
            raise OnboardingContractError("catalog surfaces must come from Smith catalog items")
        if surface.entry_id in surfaces_by_entry:
            raise OnboardingContractError(
                f"multiple integrated dependencies claim catalog entry {surface.entry_id!r}"
            )
        surfaces_by_entry[surface.entry_id] = surface

    for requirement in requirements:
        surface = surfaces_by_entry.get(requirement.entry_id)
        if surface is not None and surface.catalog_path != requirement.catalog_path:
            raise OnboardingContractError(
                f"catalog entry {requirement.entry_id!r} resolved to an unexpected surface"
            )

    gaps = tuple(
        CatalogGapProposal(
            proposal_id=f"gap-{item.item_id}-{requirement.entry_id}",
            consumer_item_id=item.item_id,
            goal_id=item.goal_id,
            entry_id=requirement.entry_id,
            catalog_path=requirement.catalog_path,
            reason=requirement.reason,
        )
        for requirement in requirements
        if requirement.entry_id not in surfaces_by_entry
    )
    if gaps:
        return AssemblyPreparation(None, gaps)

    required_ids = {requirement.entry_id for requirement in requirements}
    used_surfaces = tuple(
        sorted(
            (
                surface
                for entry_id, surface in surfaces_by_entry.items()
                if entry_id in required_ids
            ),
            key=lambda surface: surface.entry_id,
        )
    )
    scope = AssemblyScope(item.item_id, route, item.dependencies, used_surfaces)
    if unit is AssemblerUnit.CORE:
        assembler_input: AssemblerInput = CoreAssemblerInput(scope)
    elif unit is AssemblerUnit.WEIGHTS:
        assembler_input = WeightsAssemblerInput(scope)
    elif unit is AssemblerUnit.FEATURE:
        if feature is None:
            raise OnboardingContractError("feature assembly requires one feature identity")
        assembler_input = FeatureAssemblerInput(scope, _python_slug(feature, "feature"))
    else:
        assembler_input = RoutingAssemblerInput(scope, new_architecture_family)
    return AssemblyPreparation(assembler_input, ())


def validate_assembler_output(
    assembler_input: AssemblerInput,
    changed_paths: tuple[PurePosixPath, ...],
    *,
    updates_router_index: bool = False,
) -> AssemblerOutput:
    """Validate that an Assembler result stays inside its product boundary."""
    scope = assembler_input.scope
    route = scope.route
    modeling_path, weights_path = route.target_products
    if isinstance(assembler_input, CoreAssemblerInput):
        unit = AssemblerUnit.CORE
        allowed = {modeling_path}
        required = {modeling_path}
    elif isinstance(assembler_input, WeightsAssemblerInput):
        unit = AssemblerUnit.WEIGHTS
        allowed = {weights_path}
        required = {weights_path}
    elif isinstance(assembler_input, FeatureAssemblerInput):
        unit = AssemblerUnit.FEATURE
        allowed = {modeling_path, weights_path}
        required = {modeling_path}
    else:
        unit = AssemblerUnit.ROUTING
        routing_path = route.family_root / "routing.py"
        allowed = {routing_path}
        required = {routing_path}
        if assembler_input.new_architecture_family:
            allowed.add(ROUTER_INDEX_PATH)
        if updates_router_index != assembler_input.new_architecture_family:
            raise OnboardingContractError(
                "_router_index.py must change exactly when onboarding a new architecture family"
            )
    changed = set(changed_paths)
    if len(changed) != len(changed_paths):
        raise OnboardingContractError("Assembler output contains duplicate paths")
    if not required.issubset(changed) or not changed.issubset(allowed):
        raise OnboardingContractError(
            f"{unit.value} output paths must stay within {sorted(map(str, allowed))!r}"
        )
    if unit is not AssemblerUnit.ROUTING and updates_router_index:
        raise OnboardingContractError("only Routing may update _router_index.py")
    return AssemblerOutput(
        item_id=scope.item_id,
        unit=unit,
        route=route,
        changed_paths=tuple(sorted(changed, key=str)),
        synthetic_class=route.synthetic_class,
        target_module=route.target_module,
        route_table_key=route.route_table_key,
        updates_router_index=updates_router_index,
    )


@dataclass(frozen=True, slots=True)
class TargetReconciliation:
    """No-op shadow result or product-only proposal for a missing target."""

    disposition: ReconciliationDisposition
    route: RouteIdentity
    synthetic_class: str
    new_architecture_family: bool
    proposed_paths: tuple[PurePosixPath, ...]


def reconcile_modeling_v2_target(
    repository_root: Path, route: RouteIdentity
) -> TargetReconciliation:
    """Statically reconcile a route without importing targets or writing files."""
    product_root = repository_root / MODELING_V2_ROOT
    router_index = product_root / "_router_index.py"
    routers = _literal_assignment(router_index, "MODELING_V2_ROUTERS")
    if not isinstance(routers, dict):
        raise OnboardingContractError("MODELING_V2_ROUTERS must be a literal mapping")
    mapped_module = routers.get(route.architecture)
    family_routing = repository_root / route.family_root / "routing.py"
    family_directory = repository_root / route.family_root

    if mapped_module is None:
        if family_directory.exists() or route.routing_module in routers.values():
            raise OnboardingContractError(
                "_router_index.py may change only for a genuinely new architecture family"
            )
        return _missing_target(route, new_family=True)
    if mapped_module != route.routing_module:
        raise OnboardingContractError(
            f"architecture {route.architecture!r} maps to {mapped_module!r}, not "
            f"{route.routing_module!r}"
        )
    if not family_routing.is_file():
        raise OnboardingContractError("router index names a missing family routing module")

    checkpoints = _literal_assignment(family_routing, "_CHECKPOINTS")
    targets = _literal_assignment(family_routing, "_TARGETS")
    target_modules = _literal_assignment(family_routing, "TARGET_MODULES")
    if (
        not isinstance(checkpoints, dict)
        or not isinstance(targets, dict)
        or not isinstance(target_modules, dict)
    ):
        raise OnboardingContractError("family route tables must be literal mappings")

    synthetic_class = targets.get(route.route_table_key)
    target_dir = repository_root / route.target_directory
    if synthetic_class is None:
        if target_dir.exists():
            raise OnboardingContractError(
                "target path exists without an authoritative route-table row"
            )
        return _missing_target(route, new_family=False)
    if not isinstance(synthetic_class, str):
        raise OnboardingContractError("route-table target must be a synthetic class string")
    if route.checkpoint_id not in checkpoints.values():
        raise OnboardingContractError("route table names a checkpoint absent from _CHECKPOINTS")

    sm_value = _literal_assignment(family_routing, "_SM")
    expected_sm = divmod(route.sm, 10)
    if sm_value != expected_sm:
        raise OnboardingContractError(
            f"target path sm_{route.sm} disagrees with family routing _SM={sm_value!r}"
        )
    target_module = target_modules.get(synthetic_class)
    if target_module != route.target_module:
        raise OnboardingContractError(
            "synthetic class, target module path, and route table do not agree"
        )
    _validate_synthetic_class(synthetic_class, route)
    _validate_existing_target(repository_root, route, synthetic_class)
    return TargetReconciliation(
        disposition=ReconciliationDisposition.SHADOW_NOOP,
        route=route,
        synthetic_class=synthetic_class,
        new_architecture_family=False,
        proposed_paths=(),
    )


def _missing_target(route: RouteIdentity, *, new_family: bool) -> TargetReconciliation:
    proposed = [*route.target_products, route.family_root / "routing.py"]
    if new_family:
        proposed.append(ROUTER_INDEX_PATH)
    return TargetReconciliation(
        disposition=ReconciliationDisposition.MISSING_TARGET,
        route=route,
        synthetic_class=route.synthetic_class,
        new_architecture_family=new_family,
        proposed_paths=tuple(proposed),
    )


def _validate_assembler_item(item: WorkItemRecord, unit: AssemblerUnit) -> None:
    if item.profile is not DomainProfile.ASSEMBLER:
        raise OnboardingContractError("Assembler input requires the assembler domain profile")
    expected_kind = {
        AssemblerUnit.CORE: WorkItemKind.ASSEMBLE_CORE,
        AssemblerUnit.WEIGHTS: WorkItemKind.ASSEMBLE_CORE,
        AssemblerUnit.FEATURE: WorkItemKind.ASSEMBLE_FEATURE,
        AssemblerUnit.ROUTING: WorkItemKind.ROUTING,
    }[unit]
    if item.kind is not expected_kind:
        raise OnboardingContractError(
            f"{unit.value} input requires work-item kind {expected_kind.value!r}"
        )


def _validate_existing_target(
    repository_root: Path, route: RouteIdentity, synthetic_class: str
) -> None:
    target_dir = repository_root / route.target_directory
    modeling_path = target_dir / "modeling.py"
    weights_path = target_dir / "weights.py"
    if not modeling_path.is_file() or not weights_path.is_file():
        raise OnboardingContractError("routed target must contain both modeling.py and weights.py")
    allowed_files = {"__init__.py", "modeling.py", "weights.py"}
    unexpected = sorted(
        entry.name
        for entry in target_dir.iterdir()
        if entry.is_file() and entry.name not in allowed_files
    )
    if unexpected:
        raise OnboardingContractError(
            f"target directory contains non-product files: {unexpected!r}"
        )
    source = modeling_path.read_text(encoding="utf-8")
    decorator = f'@register_auto_model("{synthetic_class}")'
    if decorator not in source:
        raise OnboardingContractError("modeling.py does not register the routed synthetic class")


def _validate_synthetic_class(synthetic_class: str, route: RouteIdentity) -> None:
    if not synthetic_class.startswith("ModelingV2"):
        raise OnboardingContractError("synthetic class must use the ModelingV2 prefix")
    lowered = synthetic_class.lower()
    for segment in (route.checkpoint_id, f"sm_{route.sm}", route.target_segment):
        token = _camel_case(segment).lower()
        if token not in lowered:
            raise OnboardingContractError(
                f"synthetic class {synthetic_class!r} does not carry path segment {segment!r}"
            )


def _literal_assignment(path: Path, name: str) -> object:
    if not path.is_file():
        raise OnboardingContractError(f"missing ModelingV2 route source: {path}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise OnboardingContractError(
            f"cannot parse ModelingV2 route source {path}: {exc}"
        ) from exc
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value_node = node.value
                if value_node is None:
                    break
                try:
                    return ast.literal_eval(value_node)
                except (ValueError, TypeError, SyntaxError) as exc:
                    raise OnboardingContractError(
                        f"{path}:{name} must be statically auditable literal data"
                    ) from exc
    raise OnboardingContractError(f"{path} does not define {name}")


def _validate_catalog_path(path: PurePosixPath) -> None:
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise OnboardingContractError("catalog path must be a normalized relative Python path")
    try:
        path.relative_to(CATALOG_ROOT)
    except ValueError as exc:
        raise OnboardingContractError(f"catalog path must be under {CATALOG_ROOT}") from exc


def _python_slug(value: str, name: str) -> str:
    result = value.replace("-", "_")
    if not _PYTHON_SEGMENT.fullmatch(result):
        raise OnboardingContractError(f"{name} {value!r} is not a Python-safe route segment")
    return result


def _camel_case(value: str) -> str:
    words = tuple(filter(None, re.split(r"[^A-Za-z0-9]+", value)))
    if not words:
        raise OnboardingContractError(f"cannot derive a class-name token from {value!r}")
    return "".join(word[:1].upper() + word[1:] for word in words)


def _require_safe_id(name: str, value: str) -> None:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise OnboardingContractError(f"{name} must be a safe non-empty identifier")
