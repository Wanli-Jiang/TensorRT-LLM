# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for pure Staircase proposal policy."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent_flow.workflows.staircase.common.policy import (
    CertifiedClaimCell,
    ExecutionShape,
    IndexDeltaProposal,
    PolicyViolation,
    ResourceEscalationRequest,
    ReviewerLinkage,
    VerdictScope,
    WorkItemKind,
    WorkItemProposal,
    paths_overlap,
    validate_index_delta,
    validate_resource_escalation,
    validate_work_item_proposals,
)

_DIGEST = "a" * 64
_CATALOG_ROOT = "tensorrt_llm/_torch/modeling_v2/catalog"


def _catalog_item(
    item_id: str = "smith-attention",
    *,
    entry_id: str = "flash-attention",
    dependencies: tuple[str, ...] = (),
    resource_class: str = "smith-gpu",
    path: str = f"{_CATALOG_ROOT}/attention/flash_attention.py",
    claim_id: str = "sm_100:bf16",
    execution: ExecutionShape | None = None,
    modifies_files: bool = True,
    kind: WorkItemKind = WorkItemKind.CATALOG_ONBOARD,
) -> WorkItemProposal:
    return WorkItemProposal(
        item_id=item_id,
        goal_id="attention",
        kind=kind,
        resource_class=resource_class,
        execution=execution or ExecutionShape(),
        modifies_files=modifies_files,
        dependencies=dependencies,
        entry_ids=(entry_id,),
        allowed_paths=(path,),
        certified_claim_cells=(CertifiedClaimCell(entry_id, claim_id),),
    )


def _reviewer(item_id: str = "smith-attention") -> ReviewerLinkage:
    return ReviewerLinkage(
        item_id=item_id,
        candidate_attempt_id="coder-1",
        reviewer_attempt_id="reviewer-1",
        candidate_digest=_DIGEST,
        reviewed_candidate_digest=_DIGEST,
    )


def test_work_item_proposal_is_frozen() -> None:
    item = _catalog_item()
    with pytest.raises(FrozenInstanceError):
        item.item_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize("unsafe_id", ["../entry", "entry/name", "entry name", "", "-entry"])
def test_safe_identifiers_reject_path_and_shell_shapes(unsafe_id: str) -> None:
    with pytest.raises(PolicyViolation, match="safe non-empty identifier"):
        _catalog_item(item_id=unsafe_id)


@pytest.mark.parametrize("entry_ids", [(), ("one", "two")])
def test_catalog_item_owns_exactly_one_entry(entry_ids: tuple[str, ...]) -> None:
    with pytest.raises(PolicyViolation, match="exactly one entry"):
        WorkItemProposal(
            item_id="smith-entry",
            goal_id="attention",
            kind=WorkItemKind.CATALOG_VERIFY,
            resource_class="smith-gpu",
            execution=ExecutionShape(),
            modifies_files=False,
            entry_ids=entry_ids,
        )


def test_claim_cells_must_belong_to_atomic_entry() -> None:
    with pytest.raises(PolicyViolation, match="belong to the atomic entry"):
        WorkItemProposal(
            item_id="smith-entry",
            goal_id="attention",
            kind=WorkItemKind.CATALOG_VERIFY,
            resource_class="smith-gpu",
            execution=ExecutionShape(),
            modifies_files=False,
            entry_ids=("entry-one",),
            certified_claim_cells=(CertifiedClaimCell("entry-two", "sm_100:bf16"),),
        )


def test_collective_is_one_all_rank_item_and_never_an_array_element() -> None:
    with pytest.raises(PolicyViolation, match="all-rank verdict"):
        ExecutionShape(nodes=2, ranks_per_node=1)
    with pytest.raises(PolicyViolation, match="cannot be an array element"):
        ExecutionShape(
            nodes=2,
            ranks_per_node=1,
            array_element=True,
            verdict_scope=VerdictScope.ALL_RANKS,
        )

    shape = ExecutionShape(
        nodes=2,
        ranks_per_node=4,
        gpus_per_node=4,
        verdict_scope=VerdictScope.ALL_RANKS,
    )
    assert shape.collective
    assert shape.world_size == 8
    assert shape.total_gpus == 8


def test_arrays_are_limited_to_immutable_analysis_or_verification() -> None:
    with pytest.raises(PolicyViolation, match="file-modifying"):
        _catalog_item(execution=ExecutionShape(array_element=True), modifies_files=True)
    with pytest.raises(PolicyViolation, match="cannot be an array element"):
        WorkItemProposal(
            item_id="assemble-core",
            goal_id="model",
            kind=WorkItemKind.ASSEMBLE_CORE,
            resource_class="assembler-gpu",
            execution=ExecutionShape(array_element=True),
            modifies_files=False,
        )

    item = _catalog_item(
        kind=WorkItemKind.CATALOG_VERIFY,
        execution=ExecutionShape(array_element=True),
        modifies_files=False,
    )
    assert item.execution.array_element


def test_file_modifying_work_requires_an_explicit_write_boundary() -> None:
    with pytest.raises(PolicyViolation, match="at least one allowed path"):
        WorkItemProposal(
            item_id="assemble-core",
            goal_id="model",
            kind=WorkItemKind.ASSEMBLE_CORE,
            resource_class="assembler-gpu",
            execution=ExecutionShape(),
            modifies_files=True,
        )


def test_graph_validates_resources_paths_and_dependencies() -> None:
    first = _catalog_item()
    second = _catalog_item(
        "smith-norm",
        entry_id="rmsnorm",
        dependencies=(first.item_id,),
        path=f"{_CATALOG_ROOT}/norm/rmsnorm.py",
    )
    validate_work_item_proposals(
        (second, first),
        allowed_resource_classes={"smith-gpu"},
        allowed_path_roots=(_CATALOG_ROOT,),
    )

    with pytest.raises(PolicyViolation, match="unknown resource class"):
        validate_work_item_proposals((first,), allowed_resource_classes={"cpu"})
    with pytest.raises(PolicyViolation, match="outside allowed roots"):
        validate_work_item_proposals(
            (first,), allowed_path_roots=("tests/unittest/_torch/modeling_v2",)
        )


def test_graph_rejects_duplicate_ids_entries_claims_and_cycles() -> None:
    first = _catalog_item()
    with pytest.raises(PolicyViolation, match="duplicate work-item IDs"):
        validate_work_item_proposals((first, first))

    duplicate_entry = _catalog_item(
        "smith-copy", path=f"{_CATALOG_ROOT}/attention/copy.py", claim_id="sm_103:bf16"
    )
    with pytest.raises(PolicyViolation, match="catalog entry"):
        validate_work_item_proposals((first, duplicate_entry))

    with pytest.raises(PolicyViolation, match="duplicate claim cells"):
        WorkItemProposal(
            item_id="smith-entry",
            goal_id="attention",
            kind=WorkItemKind.CATALOG_VERIFY,
            resource_class="smith-gpu",
            execution=ExecutionShape(),
            modifies_files=False,
            entry_ids=("entry",),
            certified_claim_cells=(
                CertifiedClaimCell("entry", "sm_100:bf16"),
                CertifiedClaimCell("entry", "sm_100:bf16"),
            ),
        )

    left = _catalog_item("left", entry_id="left-entry", dependencies=("right",))
    right = _catalog_item("right", entry_id="right-entry", dependencies=("left",))
    with pytest.raises(PolicyViolation, match="cycle"):
        validate_work_item_proposals((left, right))


def test_resource_escalation_is_typed_and_bounded() -> None:
    item = _catalog_item(resource_class="coder_analysis")
    request = ResourceEscalationRequest(
        request_id="escalate-1",
        item_id=item.item_id,
        attempt_id="coder-1",
        current_resource_class="coder_analysis",
        requested_resource_class="exploratory_probe",
        reason="The verified collective requires two nodes.",
    )
    validate_resource_escalation(
        request,
        item,
        current_attempt_id="coder-1",
        current_resource_class="coder_analysis",
        current_role="coder",
        resource_class_history=("coder_analysis",),
        consumed_request_ids=(),
        allowed_resource_classes={"coder_analysis", "exploratory_probe"},
    )
    with pytest.raises(PolicyViolation, match="is not allowed"):
        validate_resource_escalation(
            request,
            item,
            current_attempt_id="coder-1",
            current_resource_class="coder_analysis",
            current_role="coder",
            resource_class_history=("coder_analysis",),
            consumed_request_ids=(),
            allowed_resource_classes={"coder_analysis"},
        )


def test_resource_escalation_uses_actual_attempt_class_role_and_lineage() -> None:
    item = _catalog_item(resource_class="coder_analysis")
    request = ResourceEscalationRequest(
        request_id="escalate-1",
        item_id=item.item_id,
        attempt_id="coder-1",
        current_resource_class="coder_analysis",
        requested_resource_class="exploratory_probe",
        reason="The initial analysis class cannot run the bounded probe.",
    )
    common = {
        "current_attempt_id": "coder-1",
        "current_resource_class": "coder_analysis",
        "current_role": "coder",
        "resource_class_history": ("coder_analysis",),
        "consumed_request_ids": (),
        "allowed_resource_classes": {
            "coder_analysis",
            "exploratory_probe",
            "deterministic_gate",
            "reviewer_analysis",
        },
    }
    with pytest.raises(PolicyViolation, match="actual attempt-selected"):
        validate_resource_escalation(
            request,
            item,
            **{**common, "current_resource_class": "exploratory_probe"},
        )
    with pytest.raises(PolicyViolation, match="only for a Coder"):
        validate_resource_escalation(
            request,
            item,
            **{**common, "current_role": "reviewer"},
        )
    with pytest.raises(PolicyViolation, match="already consumed"):
        validate_resource_escalation(
            request,
            item,
            **{**common, "consumed_request_ids": (request.request_id,)},
        )
    with pytest.raises(PolicyViolation, match="contains a cycle"):
        validate_resource_escalation(
            request,
            item,
            **{
                **common,
                "resource_class_history": ("coder_analysis", "coder_analysis"),
            },
        )

    disallowed = ResourceEscalationRequest(
        request_id="escalate-gate",
        item_id=item.item_id,
        attempt_id="coder-1",
        current_resource_class="coder_analysis",
        requested_resource_class="deterministic_gate",
        reason="invalid class",
    )
    with pytest.raises(PolicyViolation, match="cannot be escalated"):
        validate_resource_escalation(disallowed, item, **common)


def test_resource_escalation_rejects_same_class_at_the_typed_boundary() -> None:
    with pytest.raises(PolicyViolation, match="different class"):
        ResourceEscalationRequest(
            request_id="same-class",
            item_id="smith-attention",
            attempt_id="coder-1",
            current_resource_class="coder_analysis",
            requested_resource_class="coder_analysis",
            reason="no transition",
        )


def test_reviewer_linkage_pins_distinct_attempt_and_candidate_hash() -> None:
    with pytest.raises(PolicyViolation, match="distinct attempt"):
        ReviewerLinkage(
            item_id="smith-attention",
            candidate_attempt_id="attempt-1",
            reviewer_attempt_id="attempt-1",
            candidate_digest=_DIGEST,
            reviewed_candidate_digest=_DIGEST,
        )
    with pytest.raises(PolicyViolation, match="frozen candidate digest"):
        ReviewerLinkage(
            item_id="smith-attention",
            candidate_attempt_id="coder-1",
            reviewer_attempt_id="reviewer-1",
            candidate_digest=_DIGEST,
            reviewed_candidate_digest="b" * 64,
        )


def test_index_delta_is_semantic_and_matches_reviewed_atomic_item() -> None:
    item = _catalog_item()
    delta = IndexDeltaProposal(
        proposal_id="index-delta-1",
        item_id=item.item_id,
        entry_id=item.entry_id or "",
        catalog_path="attention/flash_attention.py",
        implementation="torch.ops.trtllm.flash_attention",
        summary="Paged attention wrapper.",
        candidate_digest=_DIGEST,
        reviewer=_reviewer(),
        certified_claim_cells=item.certified_claim_cells,
    )
    validate_index_delta(delta, item)

    wrong_item = _catalog_item(
        "smith-other",
        entry_id="other-entry",
        path=f"{_CATALOG_ROOT}/attention/other.py",
    )
    with pytest.raises(PolicyViolation, match="source item and atomic entry"):
        validate_index_delta(delta, wrong_item)


def test_index_delta_wrapper_must_be_owned_by_item() -> None:
    item = _catalog_item()
    delta = IndexDeltaProposal(
        proposal_id="index-delta-1",
        item_id=item.item_id,
        entry_id=item.entry_id or "",
        catalog_path="attention/not_owned.py",
        implementation="torch.ops.trtllm.flash_attention",
        summary="Paged attention wrapper.",
        candidate_digest=_DIGEST,
        reviewer=_reviewer(),
        certified_claim_cells=item.certified_claim_cells,
    )
    with pytest.raises(PolicyViolation, match="not an item-owned allowed path"):
        validate_index_delta(delta, item)


def test_path_lock_overlap_includes_ancestor_directories() -> None:
    assert paths_overlap("root/catalog", "root/catalog/entry.py")
    assert paths_overlap("root/catalog/entry.py", "root/catalog/entry.py")
    assert not paths_overlap("root/catalog/a.py", "root/catalog/b.py")
