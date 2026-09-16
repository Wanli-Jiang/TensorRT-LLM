# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure policy contracts for deterministic Staircase work dispatch.

This module intentionally knows nothing about agents, durable state, Git, or
Slurm.  The controller validates agent-authored proposals here before mapping
them into its own authoritative records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Collection, Sequence, TypeAlias

from ..tuning.contracts import TuningHypothesis

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_CLAIM_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

INITIAL_CODER_RESOURCE_CLASS = "coder_analysis"
ESCALATED_CODER_RESOURCE_CLASS = "exploratory_probe"
CODER_RESOURCE_ESCALATION_TRANSITIONS = frozenset(
    {(INITIAL_CODER_RESOURCE_CLASS, ESCALATED_CODER_RESOURCE_CLASS)}
)
NON_CODER_RESOURCE_CLASSES = frozenset(
    {"deterministic_gate", "reviewer_analysis", "reviewer_rerun"}
)


class PolicyViolation(ValueError):
    """Raised when a proposed controller action violates a locked policy."""


class WorkItemKind(str, Enum):
    """Kinds accepted from a reviewed Staircase plan."""

    FEASIBILITY = "feasibility"
    SEARCH = "search"
    CATALOG_VERIFY = "catalog_verify"
    CATALOG_ONBOARD = "catalog_onboard"
    ASSEMBLE_CORE = "assemble_core"
    ASSEMBLE_FEATURE = "assemble_feature"
    ROUTING = "routing"
    TUNE_HYPOTHESIS = "tune_hypothesis"
    GATE = "gate"


class VerdictScope(str, Enum):
    """Granularity of the deterministic verdict produced by an item."""

    ITEM = "item"
    ALL_RANKS = "all_ranks"


CATALOG_ITEM_KINDS = frozenset({WorkItemKind.CATALOG_VERIFY, WorkItemKind.CATALOG_ONBOARD})
ASSEMBLER_ITEM_KINDS = frozenset(
    {WorkItemKind.ASSEMBLE_CORE, WorkItemKind.ASSEMBLE_FEATURE, WorkItemKind.ROUTING}
)
ARRAY_ELIGIBLE_ITEM_KINDS = frozenset(
    {WorkItemKind.FEASIBILITY, WorkItemKind.SEARCH, WorkItemKind.CATALOG_VERIFY, WorkItemKind.GATE}
)


@dataclass(frozen=True, slots=True, kw_only=True)
class AssemblerFeatureInput:
    """Exact task-owned feature assigned to one Assembler WorkItem."""

    feature: str

    def __post_init__(self) -> None:
        if not isinstance(self.feature, str) or not self.feature.strip():
            raise PolicyViolation("Assembler feature identity must be non-empty")
        if self.feature != self.feature.strip():
            raise PolicyViolation("Assembler feature identity must not contain edge whitespace")


WorkItemDomainInput: TypeAlias = AssemblerFeatureInput | TuningHypothesis | None


@dataclass(frozen=True)
class CertifiedClaimCell:
    """One opaque, entry-owned certification cell used as a scheduler lock."""

    entry_id: str
    cell_id: str

    def __post_init__(self) -> None:
        _require_safe_id("claim entry_id", self.entry_id)
        if not isinstance(self.cell_id, str) or not _SAFE_CLAIM_ID.fullmatch(self.cell_id):
            raise PolicyViolation("claim cell_id must be a safe non-empty token")

    @property
    def lock_key(self) -> tuple[str, str]:
        """Return the stable lock identity for this certified cell."""
        return (self.entry_id, self.cell_id)


@dataclass(frozen=True)
class ExecutionShape:
    """Resource topology and dispatch form of one atomic work item."""

    nodes: int = 1
    ranks_per_node: int = 1
    gpus_per_node: int = 0
    array_element: bool = False
    verdict_scope: VerdictScope = VerdictScope.ITEM

    def __post_init__(self) -> None:
        _require_positive_int("nodes", self.nodes)
        _require_positive_int("ranks_per_node", self.ranks_per_node)
        _require_non_negative_int("gpus_per_node", self.gpus_per_node)
        if not isinstance(self.array_element, bool):
            raise PolicyViolation("array_element must be a boolean")
        if not isinstance(self.verdict_scope, VerdictScope):
            raise PolicyViolation("verdict_scope must be a VerdictScope")
        if self.collective and self.verdict_scope is not VerdictScope.ALL_RANKS:
            raise PolicyViolation("a collective execution requires one all-rank verdict")
        if self.array_element and self.collective:
            raise PolicyViolation("a collective item cannot be an array element")

    @property
    def collective(self) -> bool:
        """Whether this shape executes more than one rank."""
        return self.nodes > 1 or self.ranks_per_node > 1

    @property
    def world_size(self) -> int:
        """Return the exact number of ranks in the item."""
        return self.nodes * self.ranks_per_node

    @property
    def total_gpus(self) -> int:
        """Return the aggregate GPUs reserved by the item."""
        return self.nodes * self.gpus_per_node


@dataclass(frozen=True, kw_only=True)
class WorkItemProposal:
    """Frozen plan output for one controller-owned atomic work item.

    Catalog proposals carry exactly one ``entry_id``.  Non-catalog proposals
    carry none, which prevents an Assembler item from silently onboarding an
    uncertified catalog surface.
    """

    item_id: str
    goal_id: str
    kind: WorkItemKind
    resource_class: str
    execution: ExecutionShape
    modifies_files: bool
    domain_input: WorkItemDomainInput = None
    dependencies: tuple[str, ...] = ()
    entry_ids: tuple[str, ...] = ()
    allowed_paths: tuple[str, ...] = ()
    certified_claim_cells: tuple[CertifiedClaimCell, ...] = ()

    def __post_init__(self) -> None:
        _require_safe_id("item_id", self.item_id)
        _require_safe_id("goal_id", self.goal_id)
        _require_safe_id("resource_class", self.resource_class)
        if not isinstance(self.kind, WorkItemKind):
            raise PolicyViolation("kind must be a WorkItemKind")
        if not isinstance(self.execution, ExecutionShape):
            raise PolicyViolation("execution must be an ExecutionShape")
        if not isinstance(self.modifies_files, bool):
            raise PolicyViolation("modifies_files must be a boolean")
        if self.kind is WorkItemKind.ASSEMBLE_FEATURE:
            if not isinstance(self.domain_input, AssemblerFeatureInput):
                raise PolicyViolation(
                    "an assemble_feature work item requires exactly one feature domain_input"
                )
        elif self.kind is WorkItemKind.TUNE_HYPOTHESIS:
            if not isinstance(self.domain_input, TuningHypothesis):
                raise PolicyViolation(
                    "a tune_hypothesis work item requires one typed hypothesis domain_input"
                )
            if self.domain_input.item_id != self.item_id:
                raise PolicyViolation("tuning hypothesis item_id must match its WorkItem identity")
        elif self.domain_input is not None:
            raise PolicyViolation(f"{self.kind.value} work requires a null domain_input")
        _require_unique_safe_ids("dependencies", self.dependencies)
        _require_unique_safe_ids("entry_ids", self.entry_ids)
        if self.item_id in self.dependencies:
            raise PolicyViolation(f"work item {self.item_id!r} cannot depend on itself")
        if self.kind in CATALOG_ITEM_KINDS:
            if len(self.entry_ids) != 1:
                raise PolicyViolation("a catalog work item must own exactly one entry")
        elif self.entry_ids:
            raise PolicyViolation("only catalog work items may own catalog entries")
        if self.kind in ASSEMBLER_ITEM_KINDS and self.certified_claim_cells:
            raise PolicyViolation("Assembler items cannot create uncertified catalog claims")
        if self.execution.array_element:
            if self.modifies_files:
                raise PolicyViolation("file-modifying work cannot be an array element")
            if self.kind not in ARRAY_ELIGIBLE_ITEM_KINDS:
                raise PolicyViolation(f"{self.kind.value} work cannot be an array element")
        if self.kind is WorkItemKind.CATALOG_ONBOARD and not self.modifies_files:
            raise PolicyViolation("catalog onboarding must own file modifications")
        _require_unique_paths(self.allowed_paths)
        if self.modifies_files and not self.allowed_paths:
            raise PolicyViolation("file-modifying work must declare at least one allowed path")
        claim_keys = [claim.lock_key for claim in self.certified_claim_cells]
        if len(claim_keys) != len(set(claim_keys)):
            raise PolicyViolation(f"work item {self.item_id!r} contains duplicate claim cells")
        if self.kind in CATALOG_ITEM_KINDS:
            entry_id = self.entry_ids[0]
            if any(claim.entry_id != entry_id for claim in self.certified_claim_cells):
                raise PolicyViolation("every certified claim cell must belong to the atomic entry")
        elif self.certified_claim_cells:
            raise PolicyViolation("only catalog work items may reserve certified claim cells")

    @property
    def entry_id(self) -> str | None:
        """Return the one atomic catalog entry, when this is a catalog item."""
        return self.entry_ids[0] if self.entry_ids else None


@dataclass(frozen=True, kw_only=True)
class ResourceEscalationRequest:
    """Typed request for a controller-approved resource-class change."""

    request_id: str
    item_id: str
    attempt_id: str
    current_resource_class: str
    requested_resource_class: str
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "request_id",
            "item_id",
            "attempt_id",
            "current_resource_class",
            "requested_resource_class",
        ):
            _require_safe_id(name, getattr(self, name))
        if self.current_resource_class == self.requested_resource_class:
            raise PolicyViolation("resource escalation must request a different class")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise PolicyViolation("resource escalation reason must be non-empty")


@dataclass(frozen=True, kw_only=True)
class ReviewerLinkage:
    """Proof that a fresh Reviewer inspected the exact frozen candidate."""

    item_id: str
    candidate_attempt_id: str
    reviewer_attempt_id: str
    candidate_digest: str
    reviewed_candidate_digest: str

    def __post_init__(self) -> None:
        _require_safe_id("item_id", self.item_id)
        _require_safe_id("candidate_attempt_id", self.candidate_attempt_id)
        _require_safe_id("reviewer_attempt_id", self.reviewer_attempt_id)
        if self.candidate_attempt_id == self.reviewer_attempt_id:
            raise PolicyViolation("Reviewer must run as a distinct attempt")
        _require_digest("candidate_digest", self.candidate_digest)
        _require_digest("reviewed_candidate_digest", self.reviewed_candidate_digest)
        if self.reviewed_candidate_digest != self.candidate_digest:
            raise PolicyViolation("Reviewer linkage does not match the frozen candidate digest")


@dataclass(frozen=True, kw_only=True)
class IndexDeltaProposal:
    """Semantic catalog-index row proposed by one reviewed Smith item.

    This is deliberately not a free-form patch.  The controller owns the
    comment-preserving index updater and serial integration order.
    """

    proposal_id: str
    item_id: str
    entry_id: str
    catalog_path: str
    implementation: str
    summary: str
    candidate_digest: str
    reviewer: ReviewerLinkage
    certified_claim_cells: tuple[CertifiedClaimCell, ...] = ()

    def __post_init__(self) -> None:
        _require_safe_id("proposal_id", self.proposal_id)
        _require_safe_id("item_id", self.item_id)
        _require_safe_id("entry_id", self.entry_id)
        _require_relative_path("catalog_path", self.catalog_path)
        if not self.catalog_path.endswith(".py"):
            raise PolicyViolation("catalog_path must identify a Python wrapper")
        if (
            not isinstance(self.implementation, str)
            or not self.implementation.strip()
            or not isinstance(self.summary, str)
            or not self.summary.strip()
        ):
            raise PolicyViolation("index implementation and summary must be non-empty")
        _require_digest("candidate_digest", self.candidate_digest)
        if self.reviewer.item_id != self.item_id:
            raise PolicyViolation("index delta Reviewer must belong to the source item")
        if self.reviewer.candidate_digest != self.candidate_digest:
            raise PolicyViolation("index delta must name the reviewed candidate digest")
        claim_keys = [claim.lock_key for claim in self.certified_claim_cells]
        if len(claim_keys) != len(set(claim_keys)):
            raise PolicyViolation("index delta contains duplicate claim cells")
        if any(claim.entry_id != self.entry_id for claim in self.certified_claim_cells):
            raise PolicyViolation("index delta claim cells must belong to its one entry")


def validate_work_item_proposals(
    items: Sequence[WorkItemProposal],
    *,
    allowed_resource_classes: Collection[str] | None = None,
    allowed_path_roots: Sequence[str] | None = None,
) -> None:
    """Validate graph-wide identities, claims, dependencies, and boundaries.

    Args:
        items: Frozen proposals to validate as one plan graph.
        allowed_resource_classes: Optional task-normalized resource class names.
        allowed_path_roots: Optional task-scoped repository-relative write roots.

    Raises:
        PolicyViolation: If the graph is unsafe or internally inconsistent.
    """
    item_ids = [item.item_id for item in items]
    _require_unique_values("work-item IDs", item_ids)
    known_ids = set(item_ids)
    entry_owners: dict[str, str] = {}
    claim_owners: dict[tuple[str, str], str] = {}
    allowed_classes = (
        set(allowed_resource_classes) if allowed_resource_classes is not None else None
    )
    if allowed_classes is not None:
        for resource_class in allowed_classes:
            _require_safe_id("allowed resource class", resource_class)
    roots = _validated_roots(allowed_path_roots)

    for item in items:
        if allowed_classes is not None and item.resource_class not in allowed_classes:
            raise PolicyViolation(
                f"work item {item.item_id!r} requests unknown resource class "
                f"{item.resource_class!r}"
            )
        unknown = set(item.dependencies) - known_ids
        if unknown:
            raise PolicyViolation(
                f"work item {item.item_id!r} has unknown dependencies: {sorted(unknown)!r}"
            )
        if item.entry_id is not None:
            previous = entry_owners.setdefault(item.entry_id, item.item_id)
            if previous != item.item_id:
                raise PolicyViolation(
                    f"catalog entry {item.entry_id!r} is owned by both {previous!r} "
                    f"and {item.item_id!r}"
                )
        for claim in item.certified_claim_cells:
            previous = claim_owners.setdefault(claim.lock_key, item.item_id)
            if previous != item.item_id:
                raise PolicyViolation(
                    f"certified claim cell {claim.lock_key!r} is owned by both "
                    f"{previous!r} and {item.item_id!r}"
                )
        if roots is not None:
            for path in item.allowed_paths:
                if not any(_path_contains(root, path) for root in roots):
                    raise PolicyViolation(
                        f"work item {item.item_id!r} path {path!r} is outside allowed roots"
                    )
    _reject_dependency_cycles(items)


def validate_plan_domain_inputs(
    items: Sequence[WorkItemProposal],
    *,
    workflow_mode: str,
    target_features: Collection[str],
) -> None:
    """Bind special WorkItem inputs to the frozen run mode and task target.

    This deliberately validates only identities available before production
    dispatch.  It never infers a feature or hypothesis from prose, paths, or
    ambient configuration.
    """
    if workflow_mode not in {"onboard", "tune"}:
        raise PolicyViolation("workflow_mode must be 'onboard' or 'tune'")
    features = tuple(target_features)
    if any(
        not isinstance(feature, str) or not feature.strip() or feature != feature.strip()
        for feature in features
    ):
        raise PolicyViolation("target features must be canonical non-empty strings")
    _require_unique_values("target features", features)
    known_features = set(features)
    assigned_features: dict[str, str] = {}
    hypothesis_owners: dict[str, str] = {}

    for item in items:
        if item.kind is WorkItemKind.ASSEMBLE_FEATURE:
            if workflow_mode != "onboard":
                raise PolicyViolation("assemble_feature work is invalid in tune mode")
            domain_input = item.domain_input
            if not isinstance(domain_input, AssemblerFeatureInput):
                raise PolicyViolation("assemble_feature work lacks a typed feature identity")
            if domain_input.feature not in known_features:
                raise PolicyViolation(
                    f"work item {item.item_id!r} names unknown target feature "
                    f"{domain_input.feature!r}"
                )
            previous = assigned_features.setdefault(domain_input.feature, item.item_id)
            if previous != item.item_id:
                raise PolicyViolation(
                    f"target feature {domain_input.feature!r} is ambiguously assigned to "
                    f"both {previous!r} and {item.item_id!r}"
                )
        elif item.kind is WorkItemKind.TUNE_HYPOTHESIS:
            if workflow_mode != "tune":
                raise PolicyViolation("tune_hypothesis work is invalid in onboard mode")
            domain_input = item.domain_input
            if not isinstance(domain_input, TuningHypothesis):
                raise PolicyViolation("tune_hypothesis work lacks a typed hypothesis")
            previous = hypothesis_owners.setdefault(domain_input.hypothesis_id, item.item_id)
            if previous != item.item_id:
                raise PolicyViolation(
                    f"tuning hypothesis {domain_input.hypothesis_id!r} is ambiguously assigned "
                    f"to both {previous!r} and {item.item_id!r}"
                )


def validate_resource_escalation(
    request: ResourceEscalationRequest,
    item: WorkItemProposal,
    *,
    current_attempt_id: str,
    current_resource_class: str,
    current_role: str | Enum,
    resource_class_history: Sequence[str],
    consumed_request_ids: Collection[str],
    allowed_resource_classes: Collection[str],
) -> None:
    """Validate one initial Coder's one-way resource-class transition.

    The current class is supplied from the authoritative AttemptRecord rather
    than inferred from the frozen WorkItem proposal.  ``resource_class_history``
    is the ordered predecessor lineage ending at that current attempt.
    """
    if request.item_id != item.item_id:
        raise PolicyViolation("resource escalation is linked to another work item")
    _require_safe_id("current attempt_id", current_attempt_id)
    if request.attempt_id != current_attempt_id:
        raise PolicyViolation("resource escalation is linked to another attempt")
    _require_safe_id("current resource class", current_resource_class)
    if request.current_resource_class != current_resource_class:
        raise PolicyViolation("resource escalation does not name the actual attempt-selected class")
    role = current_role.value if isinstance(current_role, Enum) else current_role
    if role != "coder":
        raise PolicyViolation("resource escalation is allowed only for a Coder attempt")
    if item.resource_class != INITIAL_CODER_RESOURCE_CLASS:
        raise PolicyViolation(
            f"initial Coder resource class must be {INITIAL_CODER_RESOURCE_CLASS!r}"
        )

    allowed = set(allowed_resource_classes)
    for resource_class in allowed:
        _require_safe_id("allowed resource class", resource_class)
    if current_resource_class not in allowed:
        raise PolicyViolation(f"current resource class {current_resource_class!r} is not allowed")
    if request.requested_resource_class not in allowed:
        raise PolicyViolation(
            f"requested resource class {request.requested_resource_class!r} is not allowed"
        )
    if (
        current_resource_class in NON_CODER_RESOURCE_CLASSES
        or request.requested_resource_class in NON_CODER_RESOURCE_CLASSES
    ):
        raise PolicyViolation("deterministic and Reviewer resource classes cannot be escalated")
    transition = (current_resource_class, request.requested_resource_class)
    if transition not in CODER_RESOURCE_ESCALATION_TRANSITIONS:
        raise PolicyViolation(
            "unsupported resource escalation transition; only "
            f"{INITIAL_CODER_RESOURCE_CLASS} -> {ESCALATED_CODER_RESOURCE_CLASS} is allowed"
        )

    history = tuple(resource_class_history)
    if not history:
        raise PolicyViolation("resource escalation requires predecessor resource lineage")
    for resource_class in history:
        _require_safe_id("resource lineage class", resource_class)
    if history[-1] != current_resource_class:
        raise PolicyViolation("resource lineage does not end at the current attempt class")
    if history[0] != INITIAL_CODER_RESOURCE_CLASS:
        raise PolicyViolation("resource lineage does not start at the initial Coder class")
    if len(history) != len(set(history)):
        raise PolicyViolation("resource escalation lineage contains a cycle")
    if request.requested_resource_class in history:
        raise PolicyViolation("resource escalation would reuse a prior class or create a cycle")

    consumed = tuple(consumed_request_ids)
    for request_id in consumed:
        _require_safe_id("consumed escalation request ID", request_id)
    if len(consumed) != len(set(consumed)):
        raise PolicyViolation("consumed escalation request IDs contain duplicates")
    if request.request_id in set(consumed):
        raise PolicyViolation("resource escalation request was already consumed")


def validate_index_delta(
    delta: IndexDeltaProposal,
    item: WorkItemProposal,
    *,
    catalog_root: str = "tensorrt_llm/_torch/modeling_v2/catalog",
) -> None:
    """Validate an index delta against its reviewed atomic catalog item."""
    _require_relative_path("catalog_root", catalog_root)
    if item.kind not in CATALOG_ITEM_KINDS or item.entry_id is None:
        raise PolicyViolation("only a catalog work item may propose an index delta")
    if delta.item_id != item.item_id or delta.entry_id != item.entry_id:
        raise PolicyViolation("index delta does not match its source item and atomic entry")
    if set(delta.certified_claim_cells) != set(item.certified_claim_cells):
        raise PolicyViolation("index delta must preserve the item's exact certified claim cells")
    wrapper_path = f"{catalog_root}/{delta.catalog_path}"
    if wrapper_path not in item.allowed_paths:
        raise PolicyViolation("index delta wrapper is not an item-owned allowed path")


def paths_overlap(left: str, right: str) -> bool:
    """Return whether two validated repository-relative path locks overlap."""
    _require_relative_path("left path", left)
    _require_relative_path("right path", right)
    left_path = PurePosixPath(left)
    right_path = PurePosixPath(right)
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def _require_safe_id(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise PolicyViolation(f"{name} must be a safe non-empty identifier")


def _require_unique_safe_ids(name: str, values: Sequence[str]) -> None:
    for value in values:
        _require_safe_id(name, value)
    _require_unique_values(name, values)


def _require_unique_values(name: str, values: Sequence[object]) -> None:
    if len(values) != len(set(values)):
        raise PolicyViolation(f"duplicate {name} are not allowed")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PolicyViolation(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PolicyViolation(f"{name} must be a non-negative integer")


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise PolicyViolation(f"{name} must be a lowercase SHA-256 digest")


def _require_relative_path(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or value != PurePosixPath(value).as_posix()
    ):
        raise PolicyViolation(f"{name} must be a canonical repository-relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PolicyViolation(f"{name} must not be absolute or contain dot segments")


def _require_unique_paths(paths: Sequence[str]) -> None:
    for path in paths:
        _require_relative_path("allowed path", path)
    _require_unique_values("allowed paths", paths)


def _validated_roots(roots: Sequence[str] | None) -> tuple[str, ...] | None:
    if roots is None:
        return None
    for root in roots:
        _require_relative_path("allowed path root", root)
    _require_unique_values("allowed path roots", roots)
    return tuple(roots)


def _path_contains(root: str, child: str) -> bool:
    root_path = PurePosixPath(root)
    child_path = PurePosixPath(child)
    return child_path == root_path or root_path in child_path.parents


def _reject_dependency_cycles(items: Sequence[WorkItemProposal]) -> None:
    dependencies = {item.item_id: item.dependencies for item in items}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(item_id: str) -> None:
        if item_id in visited:
            return
        if item_id in visiting:
            raise PolicyViolation(f"dependency graph contains a cycle through {item_id!r}")
        visiting.add(item_id)
        for dependency in dependencies[item_id]:
            visit(dependency)
        visiting.remove(item_id)
        visited.add(item_id)

    for item_id in sorted(dependencies):
        visit(item_id)
