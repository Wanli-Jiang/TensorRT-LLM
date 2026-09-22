# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic ModelingV2 artifact boundaries for Staircase.

The control plane calculates file ownership before an agent is dispatched.
This module keeps that calculation independent of Git, Slurm, and agent
output, and provides import-free checks for the flat target contract.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Mapping, Sequence

MODELING_V2_ROOT = "tensorrt_llm/_torch/modeling_v2"
MODELING_V2_UNIT_ROOT = "tests/unittest/_torch/modeling_v2"
ACCURACY_ROOT = "tests/integration/defs/accuracy"
TEST_LIST_ROOT = "tests/integration/test_lists/test-db"

_SAFE_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_ARCHITECTURE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_SYNTHETIC_CLASS = re.compile(r"^ModelingV2[A-Za-z0-9]+$")
_PRODUCT_FILES = frozenset({"modeling.py", "weights.py"})
_PACKAGE_MARKER = "__init__.py"
_OBSOLETE_PRODUCT_TOKENS = ("_torch/staircase", "TRTLLM_STAIRCASE")
_STALE_CLAIMS = (
    (re.compile(r"\b\d+\.\d+\.\d+rc\d+\b"), "pinned TensorRT-LLM release"),
    (
        re.compile(r"(?i)\bflashinfer[-\w]*\s+v?\d+\.\d+\.\d+"),
        "pinned FlashInfer release",
    ),
    (
        re.compile(r"(?i)\btensorrt[-_]llm\s*[=<>!]=\s*\d"),
        "pinned TensorRT-LLM requirement",
    ),
    (re.compile(r"(?i)\bmeasured\s+(?:on\s+)?20\d\d-\d\d-\d\d"), "measurement date"),
)


class ModelingV2PolicyError(ValueError):
    """Raised when an artifact proposal violates the ModelingV2 boundary."""


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    """Path and registry identity of one ModelingV2 target.

    Args:
        family: Family package below ``models/``.
        checkpoint: Checkpoint identity below ``targets/``.
        sm: CUDA SM identity without the ``sm_`` prefix.
        parallel: Structural parallel-layout identity, such as ``tp1``.
        public_architecture: Hugging Face ``architectures[0]`` value.
        synthetic_class: Registry name returned by the routing tree.
        new_architecture_family: Whether ``_router_index.py`` lacks this
            public architecture and therefore needs a serialized update.
    """

    family: str
    checkpoint: str
    sm: int
    parallel: str
    public_architecture: str
    synthetic_class: str
    new_architecture_family: bool

    def __post_init__(self) -> None:
        for name in ("family", "checkpoint", "parallel"):
            _require_segment(name, getattr(self, name))
        if isinstance(self.sm, bool) or not isinstance(self.sm, int) or self.sm < 10:
            raise ModelingV2PolicyError("sm must be an integer architecture identity")
        if not _ARCHITECTURE.fullmatch(self.public_architecture):
            raise ModelingV2PolicyError("public_architecture must be a Python class identifier")
        if not _SYNTHETIC_CLASS.fullmatch(self.synthetic_class):
            raise ModelingV2PolicyError(
                "synthetic_class must start with ModelingV2 and contain only alphanumerics"
            )
        if not isinstance(self.new_architecture_family, bool):
            raise ModelingV2PolicyError("new_architecture_family must be a boolean")
        for segment in (self.checkpoint, f"sm_{self.sm}", self.parallel):
            if _camel(segment).lower() not in self.synthetic_class.lower():
                raise ModelingV2PolicyError(
                    f"synthetic_class does not encode target path segment {segment!r}"
                )

    @property
    def target_directory(self) -> str:
        """Return the repository-relative target directory."""
        return (
            f"{MODELING_V2_ROOT}/models/{self.family}/targets/{self.checkpoint}/"
            f"sm_{self.sm}/{self.parallel}"
        )

    @property
    def target_module(self) -> str:
        """Return the package-relative dotted target module."""
        return (
            f"models.{self.family}.targets.{self.checkpoint}.sm_{self.sm}.{self.parallel}.modeling"
        )

    @property
    def routing_path(self) -> str:
        """Return the repository-relative family routing module."""
        return f"{MODELING_V2_ROOT}/models/{self.family}/routing.py"


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One atomic catalog entry owned by a Smith work item."""

    category: str
    entry: str
    torch_primitive: bool = False

    def __post_init__(self) -> None:
        _require_segment("catalog category", self.category)
        _require_segment("catalog entry", self.entry)
        if not isinstance(self.torch_primitive, bool):
            raise ModelingV2PolicyError("torch_primitive must be a boolean")


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """Task-specific file boundaries grouped by integration authority."""

    catalog: tuple[str, ...]
    catalog_index: tuple[str, ...]
    models: tuple[str, ...]
    routing: tuple[str, ...]
    unit_tests: tuple[str, ...]
    accuracy: tuple[str, ...]
    test_lists: tuple[str, ...]
    docs: tuple[str, ...]

    @property
    def all_paths(self) -> tuple[str, ...]:
        """Return every allowed path in stable de-duplicated order."""
        paths = (
            self.catalog
            + self.catalog_index
            + self.models
            + self.routing
            + self.unit_tests
            + self.accuracy
            + self.test_lists
            + self.docs
        )
        return tuple(dict.fromkeys(paths))


def calculate_artifact_paths(
    identity: TargetIdentity,
    *,
    catalog_entries: Sequence[CatalogEntry] = (),
    accuracy_test: str | None = None,
    accuracy_references: Sequence[str] = (),
    test_lists: Sequence[str] = (),
    docs: Sequence[str] = (),
) -> ArtifactPaths:
    """Calculate exact paths that may participate in one onboarding task.

    Shared catalog-index integration is deliberately included as its own path
    and remains controller-owned. Callers select the applicable group for an
    individual work item instead of handing every path to every worker.

    Args:
        identity: Target route and path identity.
        catalog_entries: Atomic entries needed by the target.
        accuracy_test: Optional test filename below the accuracy test root.
        accuracy_references: Exact files below ``accuracy/references``.
        test_lists: Exact CI test-list filenames below the test-list root.
        docs: Exact repository-relative ModelingV2 documentation paths.

    Returns:
        Stable task-specific path groups.
    """
    catalog: list[str] = []
    unit_tests: list[str] = []
    for catalog_entry in catalog_entries:
        prefix = f"{MODELING_V2_ROOT}/catalog/{catalog_entry.category}/{catalog_entry.entry}"
        catalog.append(f"{prefix}.py")
        if not catalog_entry.torch_primitive:
            catalog.append(f"{prefix}.md")
            unit_tests.append(
                f"{MODELING_V2_UNIT_ROOT}/{catalog_entry.category}/"
                f"test_modeling_v2_{catalog_entry.entry}.py"
            )
    catalog_index = (f"{MODELING_V2_ROOT}/catalog/index.yaml",) if catalog_entries else ()

    models = tuple(f"{identity.target_directory}/{name}" for name in sorted(_PRODUCT_FILES))
    routing = [identity.routing_path]
    if identity.new_architecture_family:
        routing.append(f"{MODELING_V2_ROOT}/_router_index.py")

    accuracy: list[str] = []
    if accuracy_test is not None:
        test_name = _require_filename("accuracy_test", accuracy_test, suffix=".py")
        if not test_name.startswith("test_modeling_v2_"):
            raise ModelingV2PolicyError("accuracy_test must start with 'test_modeling_v2_'")
        accuracy.append(f"{ACCURACY_ROOT}/{test_name}")
    for reference in accuracy_references:
        accuracy.append(
            _bounded_relative_file(
                reference,
                root=f"{ACCURACY_ROOT}/references",
                name="accuracy reference",
            )
        )

    normalized_test_lists = tuple(
        _require_suffix(
            _bounded_relative_file(path, root=TEST_LIST_ROOT, name="test list"),
            suffixes=(".yaml", ".yml"),
            name="test list",
        )
        for path in test_lists
    )
    normalized_docs = tuple(_validate_doc_path(path) for path in docs)
    return ArtifactPaths(
        catalog=tuple(catalog),
        catalog_index=catalog_index,
        models=models,
        routing=tuple(routing),
        unit_tests=tuple(unit_tests),
        accuracy=tuple(accuracy),
        test_lists=normalized_test_lists,
        docs=normalized_docs,
    )


def validate_changed_paths(
    identity: TargetIdentity,
    changed_paths: Sequence[str],
    allowed: ArtifactPaths,
) -> None:
    """Reject changes outside the calculated task boundary.

    Args:
        identity: Target identity used to enforce router-index ownership.
        changed_paths: Repository-relative paths in a candidate.
        allowed: Task-specific path calculation.
    """
    normalized = tuple(_relative_path(path, "changed path") for path in changed_paths)
    extras = sorted(set(normalized) - set(allowed.all_paths))
    if extras:
        raise ModelingV2PolicyError(f"candidate changed paths outside its boundary: {extras!r}")
    router_index = f"{MODELING_V2_ROOT}/_router_index.py"
    if router_index in normalized and not identity.new_architecture_family:
        raise ModelingV2PolicyError(
            "_router_index.py may change only for a new architecture family"
        )
    target_prefix = f"{identity.target_directory}/"
    target_products = {
        path.removeprefix(target_prefix) for path in normalized if path.startswith(target_prefix)
    }
    if not target_products <= _PRODUCT_FILES:
        raise ModelingV2PolicyError("a target may contain only modeling.py and weights.py products")


def validate_complete_target_products(identity: TargetIdentity, paths: Sequence[str]) -> None:
    """Require exactly the two target products, ignoring empty package markers.

    Args:
        identity: Target whose complete file listing is being checked.
        paths: Repository-relative regular files under the target directory.
    """
    prefix = f"{identity.target_directory}/"
    members: set[str] = set()
    for path in paths:
        normalized = _relative_path(path, "target path")
        if not normalized.startswith(prefix):
            raise ModelingV2PolicyError(f"target file is outside {identity.target_directory!r}")
        member = normalized.removeprefix(prefix)
        if "/" in member:
            raise ModelingV2PolicyError("a target cannot contain nested product directories")
        if member != _PACKAGE_MARKER:
            members.add(member)
    if members != _PRODUCT_FILES:
        raise ModelingV2PolicyError(
            f"target products must be exactly modeling.py and weights.py; found {sorted(members)!r}"
        )


def validate_route_sources(
    identity: TargetIdentity,
    *,
    routing_source: str,
    modeling_source: str,
) -> None:
    """Check route table, module, decorator, and path identity agreement.

    Args:
        identity: Expected target identity.
        routing_source: Source text of the family ``routing.py``.
        modeling_source: Source text of the target ``modeling.py``.
    """
    route_strings = _string_literals(routing_source, "routing.py")
    if identity.synthetic_class not in route_strings:
        raise ModelingV2PolicyError("routing.py does not name the synthetic class")
    if identity.target_module not in route_strings:
        raise ModelingV2PolicyError(
            "routing.py does not map the class to the expected target module"
        )

    try:
        tree = ast.parse(modeling_source, filename="modeling.py")
    except SyntaxError as error:
        raise ModelingV2PolicyError(f"modeling.py is not valid Python: {error}") from error
    registrations: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            function = decorator.func
            if isinstance(function, ast.Name) and function.id == "register_auto_model":
                value = decorator.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    registrations.add(value.value)
    if registrations != {identity.synthetic_class}:
        raise ModelingV2PolicyError(
            "modeling.py must register exactly its expected synthetic class; "
            f"found {sorted(registrations)!r}"
        )


def validate_target_self_containment(sources: Mapping[str, str]) -> None:
    """Reject imports from built-in or sibling model implementations.

    General engine interfaces and catalog entries remain legal. Relative
    imports may only name a sibling file inside the same flat target.

    Args:
        sources: Target product filename to Python source text.
    """
    if set(sources) != _PRODUCT_FILES:
        raise ModelingV2PolicyError("self-containment audit requires modeling.py and weights.py")
    violations: list[str] = []
    for filename, source in sorted(sources.items()):
        try:
            tree = ast.parse(source, filename=filename)
        except SyntaxError as error:
            raise ModelingV2PolicyError(f"{filename} is not valid Python: {error}") from error
        for node in ast.walk(tree):
            module: str | None = None
            level = 0
            if isinstance(node, ast.ImportFrom):
                module = node.module
                level = node.level
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    violation = _forbidden_absolute_import(alias.name)
                    if violation is not None:
                        violations.append(f"{filename}:{node.lineno}: {violation}")
                continue
            if level > 1:
                violations.append(f"{filename}:{node.lineno}: relative import escapes target")
            elif level == 0 and module is not None:
                violation = _forbidden_absolute_import(module)
                if violation is not None:
                    violations.append(f"{filename}:{node.lineno}: {violation}")
    if violations:
        raise ModelingV2PolicyError("target is not self-contained:\n" + "\n".join(violations))


def find_stale_claims(sources: Mapping[str, str]) -> tuple[str, ...]:
    """Return obsolete product names and version/date claims with locations.

    Args:
        sources: Repository-relative path to source/prose text.

    Returns:
        Stable human-readable violations. An empty tuple means the scan passed.
    """
    violations: list[str] = []
    for path, source in sorted(sources.items()):
        for line_number, line in enumerate(source.splitlines(), 1):
            for token in _OBSOLETE_PRODUCT_TOKENS:
                if token in line:
                    violations.append(f"{path}:{line_number}: obsolete product token {token!r}")
            for pattern, description in _STALE_CLAIMS:
                if (match := pattern.search(line)) is not None:
                    violations.append(
                        f"{path}:{line_number}: {match.group(0)!r} is a stale {description}"
                    )
    return tuple(violations)


def _forbidden_absolute_import(module: str) -> str | None:
    built_in_models = "tensorrt_llm._torch.models"
    sibling_targets = "tensorrt_llm._torch.modeling_v2.models"
    if module == built_in_models or (
        module.startswith(f"{built_in_models}.") and module != f"{built_in_models}.modeling_utils"
    ):
        return f"imports built-in model implementation {module!r}"
    if module == sibling_targets or module.startswith(f"{sibling_targets}."):
        return f"imports sibling ModelingV2 target {module!r}"
    return None


def _string_literals(source: str, filename: str) -> set[str]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        raise ModelingV2PolicyError(f"{filename} is not valid Python: {error}") from error
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _camel(segment: str) -> str:
    return "".join(part.capitalize() for part in segment.split("_"))


def _require_segment(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_SEGMENT.fullmatch(value):
        raise ModelingV2PolicyError(f"{name} must be a safe lowercase path segment")


def _require_filename(name: str, value: str, *, suffix: str) -> str:
    path = PurePosixPath(value)
    if path.name != value or not value.endswith(suffix) or value in {".", ".."}:
        raise ModelingV2PolicyError(f"{name} must be one {suffix} filename")
    return value


def _relative_path(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelingV2PolicyError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or str(path) != value:
        raise ModelingV2PolicyError(f"{name} must be a normalized repository-relative path")
    return value


def _bounded_relative_file(value: str, *, root: str, name: str) -> str:
    normalized = _relative_path(value, name)
    if normalized.startswith(f"{root}/"):
        return normalized
    if "/" not in normalized:
        return f"{root}/{normalized}"
    raise ModelingV2PolicyError(f"{name} must be a filename or resolve beneath {root!r}")


def _validate_doc_path(value: str) -> str:
    normalized = _relative_path(value, "documentation path")
    if normalized in {"CODEOWNERS", ".github/CODEOWNERS"}:
        return normalized
    allowed_roots = (f"{MODELING_V2_ROOT}/", "docs/")
    if not normalized.startswith(allowed_roots) or not normalized.endswith((".md", ".rst")):
        raise ModelingV2PolicyError(
            "documentation paths must be Markdown/RST under ModelingV2 or docs/"
        )
    return normalized


def _require_suffix(value: str, *, suffixes: tuple[str, ...], name: str) -> str:
    if not value.endswith(suffixes):
        raise ModelingV2PolicyError(f"{name} must end in one of {suffixes!r}")
    return value


__all__ = [
    "ACCURACY_ROOT",
    "MODELING_V2_ROOT",
    "MODELING_V2_UNIT_ROOT",
    "TEST_LIST_ROOT",
    "ArtifactPaths",
    "CatalogEntry",
    "ModelingV2PolicyError",
    "TargetIdentity",
    "calculate_artifact_paths",
    "find_stale_claims",
    "validate_changed_paths",
    "validate_complete_target_products",
    "validate_route_sources",
    "validate_target_self_containment",
]
