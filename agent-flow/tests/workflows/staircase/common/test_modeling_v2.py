# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for task-specific ModelingV2 artifact policy."""

from __future__ import annotations

import pytest

from agent_flow.workflows.staircase.common.modeling_v2 import (
    CatalogEntry,
    ModelingV2PolicyError,
    TargetIdentity,
    calculate_artifact_paths,
    find_stale_claims,
    validate_changed_paths,
    validate_complete_target_products,
    validate_route_sources,
    validate_target_self_containment,
)


def _identity(*, new_family: bool = True) -> TargetIdentity:
    return TargetIdentity(
        family="qwen3_moe",
        checkpoint="qwen3_30b_a3b",
        sm=100,
        parallel="tp1",
        public_architecture="Qwen3MoeForCausalLM",
        synthetic_class="ModelingV2Qwen330bA3bSm100Tp1",
        new_architecture_family=new_family,
    )


def test_calculate_artifact_paths_separates_every_authority_boundary() -> None:
    paths = calculate_artifact_paths(
        _identity(),
        catalog_entries=(
            CatalogEntry(category="norm", entry="new_rmsnorm"),
            CatalogEntry(category="torch", entry="multiply", torch_primitive=True),
        ),
        accuracy_test="test_modeling_v2_qwen3.py",
        accuracy_references=("qwen3.yaml",),
        test_lists=("l0_gb200.yml",),
        docs=("tensorrt_llm/_torch/modeling_v2/README.md", "docs/source/models/qwen3.md"),
    )

    assert paths.catalog == (
        "tensorrt_llm/_torch/modeling_v2/catalog/norm/new_rmsnorm.py",
        "tensorrt_llm/_torch/modeling_v2/catalog/norm/new_rmsnorm.md",
        "tensorrt_llm/_torch/modeling_v2/catalog/torch/multiply.py",
    )
    assert paths.catalog_index == ("tensorrt_llm/_torch/modeling_v2/catalog/index.yaml",)
    assert paths.models == (
        "tensorrt_llm/_torch/modeling_v2/models/qwen3_moe/targets/"
        "qwen3_30b_a3b/sm_100/tp1/modeling.py",
        "tensorrt_llm/_torch/modeling_v2/models/qwen3_moe/targets/"
        "qwen3_30b_a3b/sm_100/tp1/weights.py",
    )
    assert paths.routing == (
        "tensorrt_llm/_torch/modeling_v2/models/qwen3_moe/routing.py",
        "tensorrt_llm/_torch/modeling_v2/_router_index.py",
    )
    assert paths.unit_tests == (
        "tests/unittest/_torch/modeling_v2/norm/test_modeling_v2_new_rmsnorm.py",
    )
    assert paths.accuracy == (
        "tests/integration/defs/accuracy/test_modeling_v2_qwen3.py",
        "tests/integration/defs/accuracy/references/qwen3.yaml",
    )
    assert paths.test_lists == ("tests/integration/test_lists/test-db/l0_gb200.yml",)


def test_existing_family_cannot_modify_router_index() -> None:
    identity = _identity(new_family=False)
    paths = calculate_artifact_paths(identity)
    assert "tensorrt_llm/_torch/modeling_v2/_router_index.py" not in paths.routing

    with pytest.raises(ModelingV2PolicyError, match="outside its boundary"):
        validate_changed_paths(
            identity,
            ("tensorrt_llm/_torch/modeling_v2/_router_index.py",),
            paths,
        )


def test_target_products_are_exactly_modeling_and_weights() -> None:
    identity = _identity()
    prefix = f"{identity.target_directory}/"
    validate_complete_target_products(
        identity,
        (f"{prefix}__init__.py", f"{prefix}modeling.py", f"{prefix}weights.py"),
    )

    with pytest.raises(ModelingV2PolicyError, match="exactly modeling.py and weights.py"):
        validate_complete_target_products(
            identity,
            (f"{prefix}modeling.py", f"{prefix}weights.py", f"{prefix}smoke.py"),
        )


def test_changed_target_path_rejects_a_third_product() -> None:
    identity = _identity()
    paths = calculate_artifact_paths(identity)
    with pytest.raises(ModelingV2PolicyError, match="outside its boundary"):
        validate_changed_paths(identity, (f"{identity.target_directory}/config.json",), paths)


def test_route_class_module_and_registration_must_agree() -> None:
    identity = _identity()
    routing_source = f'''\
_TARGETS = {{("qwen3_30b_a3b", "tp1"): "{identity.synthetic_class}"}}
TARGET_MODULES = {{"{identity.synthetic_class}": "{identity.target_module}"}}
'''
    modeling_source = f'''\
def register_auto_model(name):
    return lambda value: value

@register_auto_model("{identity.synthetic_class}")
class {identity.synthetic_class}:
    pass
'''
    validate_route_sources(
        identity,
        routing_source=routing_source,
        modeling_source=modeling_source,
    )

    wrong_module = routing_source.replace(identity.target_module, f"{identity.target_module}.other")
    with pytest.raises(ModelingV2PolicyError, match="expected target module"):
        validate_route_sources(
            identity,
            routing_source=wrong_module,
            modeling_source=modeling_source,
        )


def test_synthetic_class_must_encode_checkpoint_sm_and_parallel_path() -> None:
    with pytest.raises(ModelingV2PolicyError, match="target path segment"):
        TargetIdentity(
            family="qwen3_moe",
            checkpoint="qwen3_30b_a3b",
            sm=100,
            parallel="tp1",
            public_architecture="Qwen3MoeForCausalLM",
            synthetic_class="ModelingV2Qwen3Sm100Tp1",
            new_architecture_family=True,
        )


def test_self_containment_allows_engine_and_catalog_but_not_model_helpers() -> None:
    validate_target_self_containment(
        {
            "modeling.py": """
from tensorrt_llm._torch.modeling_v2.catalog.norm.rms import rms
from tensorrt_llm._torch.models.modeling_utils import register_auto_model
from . import weights
""",
            "weights.py": "import torch\n",
        }
    )

    with pytest.raises(ModelingV2PolicyError, match="built-in model implementation"):
        validate_target_self_containment(
            {
                "modeling.py": (
                    "from tensorrt_llm._torch.models.modeling_qwen3 import Qwen3Model\n"
                ),
                "weights.py": "import torch\n",
            }
        )
    with pytest.raises(ModelingV2PolicyError, match="sibling ModelingV2 target"):
        validate_target_self_containment(
            {
                "modeling.py": (
                    "from tensorrt_llm._torch.modeling_v2.models.other.targets.x import Helper\n"
                ),
                "weights.py": "import torch\n",
            }
        )


def test_stale_scan_reports_obsolete_product_versions_and_measurement_dates() -> None:
    violations = find_stale_claims(
        {
            "target.md": """
Use TRTLLM_STAIRCASE under _torch/staircase.
Requires tensorrt_llm>=1.2.3 and flashinfer v0.4.2.
Measured on 2026-09-16.
"""
        }
    )
    assert len(violations) == 5
    assert any("TRTLLM_STAIRCASE" in violation for violation in violations)
    assert any("measurement date" in violation for violation in violations)
