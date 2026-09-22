# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic Staircase ready-wave selection."""

from __future__ import annotations

import pytest

from agent_flow.workflows.staircase.common.dispatch import (
    AttemptReservation,
    ResourceCaps,
    select_ready_wave,
    validate_candidate_capacity,
)
from agent_flow.workflows.staircase.common.policy import (
    CertifiedClaimCell,
    ExecutionShape,
    PolicyViolation,
    VerdictScope,
    WorkItemKind,
    WorkItemProposal,
)

_CAPS = ResourceCaps(max_jobs=8, max_nodes=8, max_gpus=32)


def _item(
    item_id: str,
    *,
    entry_id: str | None = None,
    path: str | None = None,
    claim_id: str | None = None,
    dependencies: tuple[str, ...] = (),
    kind: WorkItemKind = WorkItemKind.CATALOG_ONBOARD,
    execution: ExecutionShape | None = None,
) -> WorkItemProposal:
    catalog_entry = entry_id or f"{item_id}-entry"
    catalog_path = path or f"catalog/{catalog_entry}.py"
    claims = (CertifiedClaimCell(catalog_entry, claim_id),) if claim_id is not None else ()
    is_catalog = kind in {WorkItemKind.CATALOG_ONBOARD, WorkItemKind.CATALOG_VERIFY}
    return WorkItemProposal(
        item_id=item_id,
        goal_id="goal",
        kind=kind,
        resource_class="smith-gpu",
        execution=execution or ExecutionShape(nodes=1, gpus_per_node=1),
        modifies_files=kind is not WorkItemKind.CATALOG_VERIFY,
        dependencies=dependencies,
        entry_ids=(catalog_entry,) if is_catalog else (),
        allowed_paths=(catalog_path,),
        certified_claim_cells=claims,
    )


def _active_attempt(
    item_id: str,
    *,
    attempt_id: str | None = None,
    status: str = "running",
    jobs: int = 1,
    nodes: int = 1,
    gpus: int = 1,
    paths: tuple[str, ...] = (),
    claims: tuple[CertifiedClaimCell, ...] = (),
) -> AttemptReservation:
    return AttemptReservation(
        attempt_id=attempt_id or f"attempt-{item_id}",
        item_id=item_id,
        status=status,
        jobs=jobs,
        nodes=nodes,
        gpus=gpus,
        path_locks=paths,
        claim_locks=claims,
    )


def test_selects_stable_item_id_order_independent_of_input_order() -> None:
    items = (_item("charlie"), _item("alpha"), _item("bravo"))
    statuses = {item.item_id: "ready" for item in items}
    selected = select_ready_wave(items, item_statuses=statuses, attempts=(), caps=_CAPS)
    assert [item.item_id for item in selected] == ["alpha", "bravo", "charlie"]


def test_path_conflicts_serialize_in_stable_order() -> None:
    first = _item("alpha", path="catalog/attention")
    second = _item("bravo", path="catalog/attention/wrapper.py")
    selected = select_ready_wave(
        (second, first),
        item_statuses={"alpha": "ready", "bravo": "ready"},
        attempts=(),
        caps=_CAPS,
    )
    assert [item.item_id for item in selected] == ["alpha"]


def test_active_claim_and_path_locks_block_only_conflicting_items() -> None:
    blocked_path = _item("path-item", path="catalog/attention/entry.py")
    blocked_claim = _item("claim-item", claim_id="sm_100:bf16")
    independent = _item("independent")
    active = _active_attempt(
        "older-item",
        paths=("catalog/attention",),
        claims=(CertifiedClaimCell("claim-item-entry", "sm_100:bf16"),),
    )
    statuses = {item.item_id: "ready" for item in (blocked_path, blocked_claim, independent)}
    selected = select_ready_wave(
        (blocked_path, blocked_claim, independent),
        item_statuses=statuses,
        attempts=(active,),
        caps=_CAPS,
    )
    assert [item.item_id for item in selected] == ["independent"]


def test_caps_include_every_controller_owned_nonterminal_attempt_phase() -> None:
    attempts = tuple(
        _active_attempt(
            f"active-{index}",
            attempt_id=f"attempt-{index}",
            status=status,
            nodes=1,
            gpus=2,
        )
        for index, status in enumerate(
            (
                "prepared",
                "submitting",
                "submitted",
                "pending",
                "running",
                "terminal_observed",
                "collecting",
            )
        )
    )
    candidate = _item("candidate")
    selected = select_ready_wave(
        (candidate,),
        item_statuses={"candidate": "ready"},
        attempts=attempts,
        caps=ResourceCaps(max_jobs=7, max_nodes=8, max_gpus=16),
    )
    assert selected == ()


def test_terminal_attempts_release_resources_and_locks() -> None:
    candidate = _item("candidate", path="catalog/shared.py")
    finished = _active_attempt(
        "finished", status="validated", jobs=99, nodes=99, gpus=99, paths=("catalog",)
    )
    selected = select_ready_wave(
        (candidate,),
        item_statuses={"candidate": "ready"},
        attempts=(finished,),
        caps=ResourceCaps(max_jobs=1, max_nodes=1, max_gpus=1),
    )
    assert selected == (candidate,)


def test_oversized_item_is_skipped_without_blocking_later_smaller_item() -> None:
    large = _item(
        "alpha-large",
        execution=ExecutionShape(
            nodes=4,
            ranks_per_node=1,
            gpus_per_node=4,
            verdict_scope=VerdictScope.ALL_RANKS,
        ),
    )
    small = _item("bravo-small")
    selected = select_ready_wave(
        (large, small),
        item_statuses={"alpha-large": "ready", "bravo-small": "ready"},
        attempts=(),
        caps=ResourceCaps(max_jobs=2, max_nodes=2, max_gpus=4),
    )
    assert selected == (small,)


def test_collective_consumes_one_job_but_all_nodes_and_gpus() -> None:
    collective = _item(
        "collective",
        execution=ExecutionShape(
            nodes=2,
            ranks_per_node=4,
            gpus_per_node=4,
            verdict_scope=VerdictScope.ALL_RANKS,
        ),
    )
    selected = select_ready_wave(
        (collective,),
        item_statuses={"collective": "ready"},
        attempts=(),
        caps=ResourceCaps(max_jobs=1, max_nodes=2, max_gpus=8),
    )
    assert selected == (collective,)


def test_assembler_ready_requires_integrated_dependencies() -> None:
    smith = _item("smith")
    assembler = _item(
        "assembler",
        kind=WorkItemKind.ASSEMBLE_CORE,
        dependencies=(smith.item_id,),
        path="models/family/modeling.py",
    )
    with pytest.raises(PolicyViolation, match="before every dependency was INTEGRATED"):
        select_ready_wave(
            (smith, assembler),
            item_statuses={"smith": "approved", "assembler": "ready"},
            attempts=(),
            caps=_CAPS,
        )

    selected = select_ready_wave(
        (smith, assembler),
        item_statuses={"smith": "integrated", "assembler": "ready"},
        attempts=(),
        caps=_CAPS,
    )
    assert selected == (assembler,)


def test_failed_sibling_does_not_cancel_independent_ready_item() -> None:
    failed = _item("failed")
    independent = _item("independent")
    dependent = _item("dependent", dependencies=(failed.item_id,))
    selected = select_ready_wave(
        (failed, independent, dependent),
        item_statuses={"failed": "rejected", "independent": "ready", "dependent": "ready"},
        attempts=(),
        caps=_CAPS,
    )
    assert selected == (independent,)


def test_already_active_item_is_not_dispatched_twice() -> None:
    item = _item("smith")
    selected = select_ready_wave(
        (item,),
        item_statuses={"smith": "ready"},
        attempts=(_active_attempt("smith"),),
        caps=_CAPS,
    )
    assert selected == ()


def test_dependency_cycle_is_rejected_before_selection() -> None:
    left = _item("left", dependencies=("right",))
    right = _item("right", dependencies=("left",))
    with pytest.raises(PolicyViolation, match="cycle"):
        select_ready_wave(
            (left, right),
            item_statuses={"left": "ready", "right": "ready"},
            attempts=(),
            caps=_CAPS,
        )


def test_active_usage_over_caps_fails_closed() -> None:
    with pytest.raises(PolicyViolation, match="already exceed"):
        select_ready_wave(
            (_item("candidate"),),
            item_statuses={"candidate": "ready"},
            attempts=(_active_attempt("active", gpus=2),),
            caps=ResourceCaps(max_jobs=1, max_nodes=1, max_gpus=1),
        )


def test_unknown_item_status_fails_closed() -> None:
    item = _item("candidate")
    with pytest.raises(PolicyViolation, match="unknown work-item status"):
        select_ready_wave(
            (item,),
            item_statuses={"candidate": "almost_ready"},
            attempts=(),
            caps=_CAPS,
        )


def test_multiple_active_attempts_for_one_item_fail_closed() -> None:
    item = _item("candidate")
    with pytest.raises(PolicyViolation, match="multiple active attempts"):
        select_ready_wave(
            (item,),
            item_statuses={"candidate": "ready"},
            attempts=(
                _active_attempt("candidate", attempt_id="attempt-1"),
                _active_attempt("candidate", attempt_id="attempt-2"),
            ),
            caps=_CAPS,
        )


def test_direct_candidate_capacity_counts_active_plus_candidate() -> None:
    active = _active_attempt("active", nodes=2, gpus=2)
    candidate = _active_attempt(
        "candidate",
        attempt_id="candidate-attempt",
        status="prepared",
        nodes=2,
        gpus=2,
    )
    validate_candidate_capacity(
        (active,),
        candidate,
        caps=ResourceCaps(max_jobs=2, max_nodes=4, max_gpus=4),
    )

    for caps, expected in (
        (ResourceCaps(max_jobs=1, max_nodes=4, max_gpus=4), "jobs"),
        (ResourceCaps(max_jobs=2, max_nodes=3, max_gpus=4), "nodes"),
        (ResourceCaps(max_jobs=2, max_nodes=4, max_gpus=3), "gpus"),
    ):
        with pytest.raises(PolicyViolation, match=expected):
            validate_candidate_capacity((active,), candidate, caps=caps)


def test_direct_candidate_capacity_rejects_duplicate_or_active_item() -> None:
    active = _active_attempt("active")
    duplicate_id = _active_attempt(
        "candidate",
        attempt_id=active.attempt_id,
        status="prepared",
    )
    with pytest.raises(PolicyViolation, match="already reserved"):
        validate_candidate_capacity((active,), duplicate_id, caps=_CAPS)

    same_item = _active_attempt(
        "active",
        attempt_id="new-attempt",
        status="prepared",
    )
    with pytest.raises(PolicyViolation, match="already has an active"):
        validate_candidate_capacity((active,), same_item, caps=_CAPS)
