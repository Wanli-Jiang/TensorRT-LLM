# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for the Qwen3.8-27B-NVFP4 ModelingV2 target."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ..modeling_v2.test_qwen3_5_modeling_v2_onboarding import (
    EXPECTED_GENERATED_TOKENS,
    OnboardingRuntime,
    _assert_checkpoint_unchanged,
    _assert_prompt_tokenization,
    _assert_target_identity,
    _greedy_params,
)
from ..modeling_v2.test_qwen3_5_modeling_v2_onboarding import (
    onboarding_runtime as _shared_onboarding_runtime,
)

# Re-export the shared module fixture under its canonical pytest name.  The
# test below resolves it through ``FixtureRequest`` so static analysis does not
# mistake the test argument for an import redefinition.
onboarding_runtime = _shared_onboarding_runtime

# The imported module-scoped fixture keeps its response thread alive until its
# finalizer runs after this test, so the per-call thread-leak sampler is not an
# appropriate gate here.
pytestmark = pytest.mark.threadleak(enabled=False)

REFERENCE = Path(__file__).with_name("references") / "qwen3_8_27b_nvfp4_builtin_generation.json"


def _load_frozen_builtin_tokens(
    runtime: OnboardingRuntime,
) -> dict[str, list[int]]:
    """Load the checked-in, zero-tolerance compatibility anchor."""
    assert REFERENCE.is_file(), f"frozen built-in fixture is missing: {REFERENCE}"
    value = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert value["schema_version"] == 1
    assert value["source"] == "mature built-in Qwen3.5 TensorRT-LLM path"
    assert value["protocol"] == "greedy-token-exact-builtin-five-prompts-32-tokens"

    frozen = value["records"]
    assert isinstance(frozen, list)
    by_prompt = {record["prompt"]: record["token_ids"] for record in frozen}
    assert set(by_prompt) == {record.prompt for record in runtime.fixture_records}
    for tokens in by_prompt.values():
        assert isinstance(tokens, list)
        assert len(tokens) == EXPECTED_GENERATED_TOKENS
        assert all(isinstance(token, int) and token >= 0 for token in tokens)
    return by_prompt


def test_generation_matches_frozen_builtin_fixture(
    request: pytest.FixtureRequest,
) -> None:
    """The synthetic target is token-exact with the mature built-in path."""
    runtime = request.getfixturevalue("onboarding_runtime")
    assert isinstance(runtime, OnboardingRuntime)
    expected = _load_frozen_builtin_tokens(runtime)
    mismatches: list[str] = []
    for record in runtime.fixture_records:
        _assert_prompt_tokenization(runtime, record)
        response = runtime.llm.generate(
            record.prompt_token_ids,
            sampling_params=_greedy_params(return_logits=False),
            use_tqdm=False,
        )
        assert response.finished
        assert len(response.outputs) == 1
        actual = response.outputs[0].token_ids
        assert actual is not None
        if actual != expected[record.prompt]:
            mismatches.append(
                f"{record.prompt!r}: target={actual}, built_in={expected[record.prompt]}"
            )
    assert not mismatches, (
        "ModelingV2 delegated target differs from the mature built-in "
        "TensorRT-LLM path:\n" + "\n".join(mismatches)
    )
    _assert_target_identity(runtime.live_model, runtime.target_class)
    _assert_checkpoint_unchanged(runtime)
