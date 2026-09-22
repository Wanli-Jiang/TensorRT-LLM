# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic contracts for compact Staircase role prompts."""

from __future__ import annotations

import re
from pathlib import Path

from agent_flow.workflows.agent_team.prompts import PromptBundle
from agent_flow.workflows.staircase.prompts import (
    DEFAULT_PROMPTS,
    STAIRCASE_PROMPTS,
    DomainProfile,
    build_staircase_prompts,
)

_ROLE_NAMES = ("plan_drafter", "plan_reviewer", "coder", "reviewer", "qa")
_OBSOLETE_PRODUCT_TOKENS = (
    "_torch/staircase",
    "TRTLLM_STAIRCASE",
    "MODELING_V2_TARGET",
    "staircase_bringup",
)
_RAW_SCHEDULER_COMMANDS = ("sbatch", "srun", "squeue", "sacct", "scancel")
_PROMPT_STATE_TOKENS = ("status.md", "progress.yaml", "[Doing]", "[Undo]")


def _prompts(bundle: PromptBundle) -> dict[str, str]:
    """Return a role-to-prompt mapping for compact assertions."""
    return {name: getattr(bundle, name) for name in _ROLE_NAMES}


def _norm(text: str) -> str:
    """Collapse wrapping whitespace for semantic substring assertions."""
    return re.sub(r"\s+", " ", text)


def test_base_bundle_is_five_role_and_bounded() -> None:
    assert isinstance(STAIRCASE_PROMPTS, PromptBundle)
    assert DEFAULT_PROMPTS is STAIRCASE_PROMPTS

    prompts = _prompts(STAIRCASE_PROMPTS)
    assert all(prompt.strip() for prompt in prompts.values())
    assert all(len(prompt) < 8_000 for prompt in prompts.values())
    assert sum(map(len, prompts.values())) < 30_000


def test_every_role_carries_control_data_and_reference_boundaries() -> None:
    for role, raw_prompt in _prompts(STAIRCASE_PROMPTS).items():
        prompt = _norm(raw_prompt)
        assert "Staircase is the agent/control plane" in prompt, role
        assert "ModelingV2 is the only product/data plane" in prompt, role
        assert "A Goal is one module or capability" in prompt, role
        assert "one catalog entry is one atomic child WorkItem" in prompt, role
        assert "controller is the sole authority" in prompt, role
        assert "built-in per-model definitions" in prompt, role
        assert "task-scoped delegated built-in reuse" in prompt, role
        assert "exact mature built-in model and weight mapper" in prompt, role
        assert "self-contained catalog boundary is authoritative" in prompt, role
        assert "implementation under test" in prompt, role
        assert "Performance is measured evidence" in prompt, role


def test_every_role_uses_only_current_modeling_v2_contract() -> None:
    for role, raw_prompt in _prompts(STAIRCASE_PROMPTS).items():
        prompt = _norm(raw_prompt)
        assert "tensorrt_llm/_torch/modeling_v2/" in prompt, role
        assert "catalog/index.yaml" in prompt, role
        assert "models/<family>/routing.py" in prompt, role
        assert "models/<family>/targets/<checkpoint>/<gpu_arch>/<parallel>/" in prompt, role
        assert "tests/unittest/_torch/modeling_v2/<category>/" in prompt, role
        assert "tests/integration/defs/accuracy/references/" in prompt, role
        assert "TRTLLM_MODELING_V2=require" in prompt, role
        assert "before any rank starts" in prompt, role
        assert "target-owned `config.json`" in prompt, role


def test_plan_roles_enforce_capability_first_atomic_planning() -> None:
    drafter = _norm(STAIRCASE_PROMPTS.plan_drafter)
    reviewer = _norm(STAIRCASE_PROMPTS.plan_reviewer)

    assert "capability-first execution plan" in drafter
    assert "derivation table" in drafter
    assert "capability map" in drafter
    assert "catalog candidate or gap" in drafter
    assert "Split every catalog entry into its own Smith item" in drafter
    assert "Decision records for accepted and rejected candidates" in drafter
    assert "Do not encode lifecycle transitions" in drafter
    assert "Every WorkItem carries an explicit `domain_input`" in drafter
    assert "complete typed one-variable hypothesis" in drafter
    assert "Respect the frozen workflow mode" in drafter
    assert "plan records the selected boundary and exact dependency pair" in drafter

    assert "Goal is an operation instead of a module/capability" in reviewer
    assert "multiple catalog entries share one atomic item" in reviewer
    assert "consume an entry before its independent review" in reviewer
    assert "resource requests stay inside" in reviewer
    assert "missing, unknown, or ambiguous WorkItem `domain_input`" in reviewer
    assert "Approval of the frozen plan digest explicitly attests" in reviewer


def test_execution_roles_preserve_isolation_and_authority() -> None:
    coder = _norm(STAIRCASE_PROMPTS.coder)
    reviewer = _norm(STAIRCASE_PROMPTS.reviewer)

    assert "exactly the supplied atomic WorkItem" in coder
    assert "Do not edit sibling item paths or shared indexes" in coder
    assert "typed delta" in coder
    assert "typed escalation" in coder
    assert "Do not approve or integrate your own candidate" in coder

    assert "fresh read-only snapshot" in reviewer
    assert "controller-supplied candidate hash" in reviewer
    assert "Do not reuse a mutable Coder workspace" in reviewer
    assert "independently rerun" in reviewer
    assert "Never change Stage, Goal, WorkItem, or attempt" in reviewer


def test_qa_hard_gates_override_scores_and_performance() -> None:
    prompt = _norm(STAIRCASE_PROMPTS.qa)
    assert "Hard gates override all scores" in prompt
    assert "no weighted average can turn it into `PASS`" in prompt
    assert "Performance may be reported when requested" in prompt
    assert "performance alone cannot prove correctness" in prompt
    assert "feature-specific acceptance gate" in prompt
    assert "routing did not fall back" in prompt


def test_roles_publish_only_the_normative_outcome_vocabularies() -> None:
    expected = {
        "plan_drafter": ("DRAFTED", "BLOCKED"),
        "plan_reviewer": ("ACCEPT", "REVISE", "BLOCK"),
        "coder": ("CANDIDATE", "RESOURCE_ESCALATION", "BLOCKED"),
        "reviewer": ("APPROVE", "REJECT", "BLOCK"),
        "qa": ("PASS", "FAIL", "BLOCK"),
    }
    for role, outcomes in expected.items():
        prompt = _norm(_prompts(STAIRCASE_PROMPTS)[role])
        for outcome in outcomes:
            assert f"`{outcome}`" in prompt, (role, outcome)
        assert "lowercase aliases" in prompt, role


def test_domain_profiles_extend_only_coder_and_reviewer() -> None:
    base = STAIRCASE_PROMPTS
    for profile in DomainProfile:
        specialized = build_staircase_prompts(profile)
        assert specialized.plan_drafter == base.plan_drafter
        assert specialized.plan_reviewer == base.plan_reviewer
        assert specialized.qa == base.qa
        assert specialized.coder.startswith(base.coder)
        assert specialized.reviewer.startswith(base.reviewer)
        assert f"Domain profile: {profile.value.capitalize()}" in specialized.coder
        assert f"Domain profile: {profile.value.capitalize()}" in specialized.reviewer

    assert build_staircase_prompts() is STAIRCASE_PROMPTS


def test_smith_profile_is_atomic_and_multi_node_safe() -> None:
    prompts = build_staircase_prompts(DomainProfile.SMITH)
    coder = _norm(prompts.coder)
    reviewer = _norm(prompts.reviewer)
    assert "exactly one catalog entry" in coder
    assert "contract `.md`, thin wrapper `.py`, and collected GPU test" in coder
    assert "typed catalog-index delta" in coder
    assert "Do not inspect or edit another entry" in coder
    assert "only the trusted supervisor may run across the assigned nodes" in coder
    assert "Independently rerun the decisive cell or matrix" in reviewer
    assert "never edit the shared index" in reviewer


def test_assembler_profile_enforces_selected_implementation_boundary() -> None:
    prompts = build_staircase_prompts(DomainProfile.ASSEMBLER)
    coder = _norm(prompts.coder)
    reviewer = _norm(prompts.reviewer)
    assert "already integrated catalog dependencies" in coder
    assert "typed catalog-gap request" in coder
    assert "keep the target self-contained and flat" in coder
    assert "task-scoped delegated built-in reuse" in coder
    assert "exact named mature model and weight mapper" in coder
    assert "do not widen that dependency or turn it into fallback" in coder
    assert "workload/tuning knobs out of routing" in coder
    assert "audit flat-forward catalog coverage" in reviewer
    assert "exact named model/mapper boundary" in reviewer
    assert "undeclared, unbounded, sibling-target, or fallback imports" in reviewer


def test_delegated_target_keeps_independent_plan_review_and_qa_gates() -> None:
    plan_reviewer = _norm(STAIRCASE_PROMPTS.plan_reviewer)
    reviewer = _norm(STAIRCASE_PROMPTS.reviewer)
    qa = _norm(STAIRCASE_PROMPTS.qa)

    assert "explicitly authorized for task-scoped delegated built-in reuse" in plan_reviewer
    assert "name the exact mature built-in model and weight mapper" in plan_reviewer
    assert "real boot/generation, independent reference, Reviewer, and QA gates" in plan_reviewer

    assert "frozen task or approved plan names the exact model and weight mapper" in reviewer
    assert "no other built-in dependency is imported" in reviewer
    assert "built-in implementation is not its own oracle" in reviewer

    assert "published checkpoint stayed read-only" in qa
    assert "real checkpoint boot and generation" in qa
    assert "exact synthetic route used ModelingV2" in qa
    assert "routing did not fall back" in qa
    assert "independent Reviewer and QA evidence" in qa


def test_tuner_profile_changes_one_variable_and_cannot_promote() -> None:
    prompts = build_staircase_prompts(DomainProfile.TUNER)
    coder = _norm(prompts.coder)
    reviewer = _norm(prompts.reviewer)
    assert "change one declared variable" in coder
    assert "Do not combine code and configuration changes" in coder
    assert "Do not" in coder and "promote your own result" in coder
    assert "baseline and candidate are comparable" in reviewer
    assert "A faster incorrect candidate fails" in reviewer
    assert "controller, which alone decides promotion" in reviewer


def test_prompts_exclude_obsolete_products_site_commands_and_text_state() -> None:
    bundles = [STAIRCASE_PROMPTS, *(build_staircase_prompts(profile) for profile in DomainProfile)]
    for bundle in bundles:
        for role, prompt in _prompts(bundle).items():
            for token in (
                *_OBSOLETE_PRODUCT_TOKENS,
                *_RAW_SCHEDULER_COMMANDS,
                *_PROMPT_STATE_TOKENS,
            ):
                assert token not in prompt, (role, token)


def test_staircase_docs_define_bounded_delegated_reuse_without_weakening_gates() -> None:
    import agent_flow.workflows.staircase as staircase_package

    package_file = staircase_package.__file__
    assert package_file is not None
    package_dir = Path(package_file).parent
    documents = (
        package_dir / "README.md",
        package_dir / "onboarding" / "README.md",
        package_dir / "references" / "modeling_v2_contract.md",
    )
    for document in documents:
        text = _norm(document.read_text(encoding="utf-8"))
        assert "task-scoped delegated built-in reuse" in text.lower(), document
        assert "exact" in text and "model" in text and "weight mapper" in text, document
        assert "read-only" in text and "published" in text, document
        assert "Reviewer" in text and "QA" in text, document


def test_prompt_modules_have_no_live_backend_or_subprocess_imports() -> None:
    import agent_flow.workflows.staircase.prompts as prompt_package

    package_file = prompt_package.__file__
    assert package_file is not None
    for source in Path(package_file).parent.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        assert "claude_agent_sdk" not in text
        assert "openai" not in text
        assert "import subprocess" not in text
