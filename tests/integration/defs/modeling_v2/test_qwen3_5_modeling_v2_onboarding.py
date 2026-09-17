# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint gates for the Qwen3.8-27B-NVFP4 ModelingV2 target.

These gates are deliberately fail-closed.  A successful engine boot is not
enough: the process must run in ``TRTLLM_MODELING_V2=require`` mode on one
sm_103 GPU, the live model object must be the synthetic target rather than its
built-in base class, and the checkpoint must remain byte-identical.

The accuracy fixture is the immutable output of the independent HF reference
run from Slurm job 772151.  Its content hash and the accompanying checkpoint
witness are pinned below.  The runner must supply their directory through
``QWEN3_5_HF_REFERENCE_DIR``; no committed test depends on a developer's
scratch tree.  If either artifact changes, this test does not silently bless
the replacement.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modeling_v2 import MODELING_V2_ENV
from tensorrt_llm._torch.models.modeling_auto import AutoModelForCausalLM
from tensorrt_llm.mapping import Mapping

from ._qwen3_5_checkpoint import (
    EXPECTED_CHECKPOINT_FILE_COUNT,
    CheckpointReader,
    assert_manifest_inventory,
    checkpoint_digest,
    checkpoint_path,
    compare_digests,
    hash_file,
)
from ._qwen3_5_hf_reference import GenerationRecord, load_generation_fixture

# The module-scoped LLM intentionally keeps its asynchronous response thread
# alive across all four selectors and closes it in the fixture finalizer.
# pytest-threadleak samples after each test call, before that finalizer runs.
pytestmark = pytest.mark.threadleak(enabled=False)

REFERENCE_DIR_ENV = "QWEN3_5_HF_REFERENCE_DIR"
BUILTIN_REFERENCE_DIR_ENV = "QWEN3_5_BUILTIN_REFERENCE_DIR"

EXPECTED_TARGET = "ModelingV2Qwen3827BNvfp4Sm103Tp1"
EXPECTED_TARGET_MODULE = (
    "tensorrt_llm._torch.modeling_v2.models.qwen3_5.targets.qwen3_8_27b_nvfp4.sm_103.tp1.modeling"
)
EXPECTED_INNER_TARGET = "ModelingV2Qwen3827BNvfp4TextSm103Tp1"
EXPECTED_CAPABILITY = (10, 3)
EXPECTED_GENERATED_TOKENS = 32
EXPECTED_FIXTURE_RECORDS = 5
MIN_HF_LOGIT_COSINE = 0.99
MAX_HF_LOGIT_MEAN_ABS_ERROR = 0.50

# BLAKE2b-256 of the exact artifacts emitted by independent HF job 772151.
EXPECTED_FIXTURE_DIGEST = (
    "blake2b256:46c0849b141f505f58e5a1a8163a14d1037968c610a7d683c5d6445f98b204bb"
)
EXPECTED_HF_LOGITS_DIGEST = (
    "blake2b256:f24e9fd491e498e4ceaf30da608fba019a44e51ba96b76a9adf4c43faf00fe2c"
)
EXPECTED_BUILTIN_FIXTURE_DIGEST = (
    "blake2b256:563631628780ba19205a563060d3c0f92d3c17d6ed0805c830242da3089be23f"
)
EXPECTED_WITNESS_DIGEST = (
    "blake2b256:63e2f09ab46082af6593947567561faba7176aea7743a2c245d1882c99816ddd"
)
EXPECTED_CONFIG_DIGEST = (
    "blake2b256:48bb42d6988f56e541b7a5ae4b0c2e2ddf0b6caa6cd00fb103be7e6d4449cf50"
)


@dataclass(frozen=True)
class OnboardingRuntime:
    """One real engine boot and the evidence that identifies it."""

    llm: LLM
    checkpoint: str
    target_class: type[torch.nn.Module]
    live_model: torch.nn.Module
    digest_before: dict[str, str]
    digest_after_load: dict[str, str]
    fixture_meta: dict[str, object]
    fixture_records: list[GenerationRecord]


def _require_real_environment() -> None:
    """Refuse to turn a CPU, wrong-SM, or fallback run into green evidence."""
    mode = os.environ.get(MODELING_V2_ENV)
    assert mode == "require", (
        f"{MODELING_V2_ENV} must be exported as 'require' before Python starts; got {mode!r}"
    )
    if not torch.cuda.is_available():
        pytest.fail("Qwen3.8 onboarding gates require one real CUDA GPU")
    assert torch.cuda.device_count() == 1, (
        f"TP1 gate requires exactly one visible GPU, got {torch.cuda.device_count()} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
    )
    capability = torch.cuda.get_device_capability(0)
    assert capability == EXPECTED_CAPABILITY, (
        f"gate is certified only for sm_103, got {torch.cuda.get_device_name(0)} "
        f"with capability {capability}"
    )


def _load_frozen_reference() -> tuple[dict[str, object], list[GenerationRecord], dict[str, str]]:
    """Load the pinned HF fixture and its independently recorded checkpoint."""
    reference_dir_value = os.environ.get(REFERENCE_DIR_ENV)
    assert reference_dir_value, (
        f"set {REFERENCE_DIR_ENV} to the external directory containing the "
        "job-772151 hf_generation_fixture.json and checkpoint_hashes.json artifacts"
    )
    reference_dir = Path(reference_dir_value)
    fixture = reference_dir / "hf_generation_fixture.json"
    checkpoint_witness = reference_dir / "checkpoint_hashes.json"
    assert fixture.is_file(), f"frozen HF fixture is missing: {fixture}"
    assert checkpoint_witness.is_file(), (
        f"frozen checkpoint witness is missing: {checkpoint_witness}"
    )
    assert hash_file(str(fixture)) == EXPECTED_FIXTURE_DIGEST, (
        "the HF generation fixture changed after job 772151; review and pin a "
        "new independent reference rather than accepting it implicitly"
    )
    assert hash_file(str(checkpoint_witness)) == EXPECTED_WITNESS_DIGEST, (
        "the checkpoint witness changed after job 772151"
    )

    meta, records = load_generation_fixture(str(fixture))
    assert meta["run_id"] == "772151"
    assert meta["source"] == "stock transformers Qwen3_5ForCausalLM.generate()"
    assert meta["capability"] == [10, 3]
    assert meta["world_size"] == 1
    assert meta["tp_size"] == 1
    assert meta["pp_size"] == 1
    assert meta["decoding"] == {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": EXPECTED_GENERATED_TOKENS,
        "min_new_tokens": EXPECTED_GENERATED_TOKENS,
    }
    assert len(records) == EXPECTED_FIXTURE_RECORDS
    for record in records:
        assert record.prompt
        assert record.prompt_token_ids
        assert len(record.generated_token_ids) == EXPECTED_GENERATED_TOKENS
        assert record.text.strip()

    witness = json.loads(checkpoint_witness.read_text(encoding="utf-8"))
    assert witness["changes"] == []
    assert witness["before"] == witness["after"]
    assert len(witness["before"]) == EXPECTED_CHECKPOINT_FILE_COUNT
    return meta, records, witness["before"]


def _resolve_checkpoint() -> str:
    """Resolve the repository-standard model path and verify its fixed identity."""
    selected = Path(checkpoint_path())
    assert selected.is_dir(), f"checkpoint is missing: {selected}"
    config = selected / "config.json"
    assert config.is_file(), f"checkpoint config is missing: {config}"
    assert hash_file(str(config)) == EXPECTED_CONFIG_DIGEST, (
        f"{selected} is not the pinned Qwen3.8-27B-NVFP4 checkpoint: "
        "config.json content identity differs"
    )
    return str(selected)


def _live_model(llm: LLM) -> torch.nn.Module:
    """Return the in-process engine model, failing if its identity is hidden."""
    executor = getattr(llm, "_executor", None)
    engine = getattr(executor, "engine", None)
    model_engine = getattr(engine, "model_engine", None)
    model = getattr(model_engine, "model", None)
    assert isinstance(model, torch.nn.Module), (
        "the TP1 gate could not inspect the live PyTorch model; accepting only "
        "a preflight resolver result would not prove that the booted engine did "
        "not fall back"
    )
    return model


def _assert_target_identity(model: torch.nn.Module, target_class: type[torch.nn.Module]) -> None:
    """Prove that the actual engine contains the synthetic target class."""
    assert type(model) is target_class, (
        f"live engine built {type(model).__module__}.{type(model).__name__}, "
        f"expected {EXPECTED_TARGET_MODULE}.{EXPECTED_TARGET}"
    )
    assert type(model).__name__ == EXPECTED_TARGET
    assert type(model).__module__ == EXPECTED_TARGET_MODULE
    assert getattr(model, "mm_encoder", object()) is None
    model_config = model.model_config
    assert model_config.disable_mm_encoder is True
    assert model_config.mapping.world_size == 1
    assert model_config.mapping.tp_size == 1
    assert model_config.mapping.pp_size == 1
    assert model_config.mapping.cp_size == 1
    assert not model_config.mapping.enable_attention_dp
    inner_model = getattr(model, "llm", None)
    assert isinstance(inner_model, torch.nn.Module)
    assert type(inner_model).__name__ == EXPECTED_INNER_TARGET
    assert type(inner_model).__module__ == EXPECTED_TARGET_MODULE


@pytest.fixture(scope="module")
def onboarding_runtime() -> Iterator[OnboardingRuntime]:
    """Boot the target once while keeping every selector independently strict."""
    _require_real_environment()
    checkpoint = _resolve_checkpoint()
    fixture_meta, fixture_records, frozen_checkpoint_digest = _load_frozen_reference()

    reader = CheckpointReader(checkpoint)
    assert_manifest_inventory(reader)
    del reader

    digest_before = checkpoint_digest(checkpoint)
    assert digest_before == frozen_checkpoint_digest, (
        "the live checkpoint bytes differ from the checkpoint used by frozen HF job 772151"
    )

    # This is the same production config path used by the engine.  Resolve it
    # before boot so the asserted class object can be compared by identity with
    # the model object held by the live executor.
    model_config = ModelConfig.from_pretrained(
        checkpoint,
        mapping=Mapping(),
        disable_mm_encoder=True,
        max_num_tokens=256,
        max_seq_len=128,
    )
    target_class = AutoModelForCausalLM._resolve_class(model_config)
    assert target_class.__name__ == EXPECTED_TARGET
    assert target_class.__module__ == EXPECTED_TARGET_MODULE

    with LLM(
        checkpoint,
        backend="pytorch",
        disable_mm_encoder=True,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        max_batch_size=5,
        max_num_tokens=256,
        max_seq_len=128,
        enable_chunked_prefill=False,
        disable_overlap_scheduler=True,
        gather_generation_logits=True,
    ) as llm:
        live_model = _live_model(llm)
        _assert_target_identity(live_model, target_class)
        digest_after_load = checkpoint_digest(checkpoint)
        yield OnboardingRuntime(
            llm=llm,
            checkpoint=checkpoint,
            target_class=target_class,
            live_model=live_model,
            digest_before=digest_before,
            digest_after_load=digest_after_load,
            fixture_meta=fixture_meta,
            fixture_records=fixture_records,
        )


def _assert_checkpoint_unchanged(runtime: OnboardingRuntime) -> None:
    digest_after = checkpoint_digest(runtime.checkpoint)
    changes = compare_digests(runtime.digest_before, digest_after)
    assert not changes, f"the gate modified checkpoint-owned files: {changes}"


def _assert_prompt_tokenization(runtime: OnboardingRuntime, record: GenerationRecord) -> None:
    actual = runtime.llm.tokenizer.encode(record.prompt)
    assert actual == record.prompt_token_ids, (
        f"tokenizer drift for {record.prompt!r}: target={actual}, "
        f"frozen HF={record.prompt_token_ids}"
    )


def _greedy_params(*, return_logits: bool) -> SamplingParams:
    return SamplingParams(
        max_tokens=EXPECTED_GENERATED_TOKENS,
        min_tokens=EXPECTED_GENERATED_TOKENS,
        temperature=0.0,
        top_k=1,
        ignore_eos=True,
        detokenize=False,
        return_generation_logits=return_logits,
    )


def _one_token_greedy_params() -> SamplingParams:
    return SamplingParams(
        max_tokens=1,
        min_tokens=1,
        temperature=0.0,
        top_k=1,
        ignore_eos=True,
        detokenize=False,
        return_generation_logits=True,
    )


def _load_frozen_hf_logits(records: list[GenerationRecord]) -> dict[str, torch.Tensor]:
    reference_dir = Path(os.environ[REFERENCE_DIR_ENV])
    path = reference_dir / "hf_step_logits.pt"
    assert path.is_file(), f"frozen HF step logits are missing: {path}"
    assert hash_file(str(path)) == EXPECTED_HF_LOGITS_DIGEST, (
        "the HF step-logit artifact changed after job 772151"
    )
    value = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(value, dict)
    assert set(value) == {record.prompt for record in records}
    expected_shape = (EXPECTED_GENERATED_TOKENS, 248320)
    for prompt, logits in value.items():
        assert isinstance(prompt, str)
        assert isinstance(logits, torch.Tensor)
        assert tuple(logits.shape) == expected_shape
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
    return value


def _load_frozen_builtin_tokens(records: list[GenerationRecord]) -> dict[str, list[int]]:
    reference_dir_value = os.environ.get(BUILTIN_REFERENCE_DIR_ENV)
    assert reference_dir_value, (
        f"set {BUILTIN_REFERENCE_DIR_ENV} to the external job-774852 artifact directory"
    )
    path = Path(reference_dir_value) / "builtin-generation.json"
    assert path.is_file(), f"frozen built-in fixture is missing: {path}"
    assert hash_file(str(path)) == EXPECTED_BUILTIN_FIXTURE_DIGEST, (
        "the mature built-in TensorRT-LLM fixture changed after job 774852"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == 1
    assert value["source"] == "mature built-in Qwen3.5 TensorRT-LLM path"
    frozen = value["records"]
    assert isinstance(frozen, list)
    by_prompt = {record["prompt"]: record["token_ids"] for record in frozen}
    assert set(by_prompt) == {record.prompt for record in records}
    for tokens in by_prompt.values():
        assert isinstance(tokens, list)
        assert len(tokens) == EXPECTED_GENERATED_TOKENS
        assert all(isinstance(token, int) and token >= 0 for token in tokens)
    return by_prompt


def test_read_only_weight_load(onboarding_runtime: OnboardingRuntime) -> None:
    """Load every real checkpoint shard without changing any published byte."""
    runtime = onboarding_runtime
    _assert_target_identity(runtime.live_model, runtime.target_class)
    assert runtime.digest_after_load == runtime.digest_before

    parameters = list(runtime.live_model.named_parameters())
    assert parameters, "the live target has no parameters"
    meta_parameters = [name for name, parameter in parameters if parameter.is_meta]
    assert not meta_parameters, f"weight load left meta parameters: {meta_parameters[:20]}"
    assert all(parameter.device.type == "cuda" for _, parameter in parameters)

    # Check representative values without scanning the complete 27 B model a
    # second time.  Loader coverage is enforced by the production loader; this
    # catches a target that merely materialized empty/non-finite placeholders.
    floating = [
        (name, parameter)
        for name, parameter in parameters
        if parameter.numel() and parameter.is_floating_point()
    ]
    assert floating, "the loaded target has no floating-point parameters"
    for name, parameter in floating[:32]:
        samples = parameter.detach().reshape(-1)[:: max(1, parameter.numel() // 8)][:8].float()
        assert torch.isfinite(samples).all(), f"{name} contains non-finite sampled weights"

    _assert_checkpoint_unchanged(runtime)


def test_real_checkpoint_boot_and_generation(onboarding_runtime: OnboardingRuntime) -> None:
    """The selected target boots and emits finite, non-empty real-model output."""
    runtime = onboarding_runtime
    record = runtime.fixture_records[0]
    _assert_prompt_tokenization(runtime, record)

    response = runtime.llm.generate(
        record.prompt_token_ids,
        sampling_params=_greedy_params(return_logits=True),
        use_tqdm=False,
    )
    assert response.finished
    assert len(response.outputs) == 1
    completion = response.outputs[0]
    assert completion.token_ids
    assert len(completion.token_ids) == EXPECTED_GENERATED_TOKENS
    assert all(isinstance(token_id, int) and token_id >= 0 for token_id in completion.token_ids)
    logits = completion.generation_logits
    assert isinstance(logits, torch.Tensor)
    assert logits.numel() > 0
    assert torch.isfinite(logits).all(), "generation produced non-finite logits"

    _assert_target_identity(runtime.live_model, runtime.target_class)
    _assert_checkpoint_unchanged(runtime)


def test_teacher_forced_logits_match_frozen_hf_reference(
    onboarding_runtime: OnboardingRuntime,
) -> None:
    """Quantized target logits stay within the frozen HF numerical bars."""
    # Job 772151 deliberately dequantized the published FP8/NVFP4 weights to
    # BF16 and used HF eager attention.  It is an independent semantic oracle,
    # not a bit-exact decode oracle.  Replay every HF prefix so autoregressive
    # divergence after a close argmax cannot invalidate later comparisons.
    runtime = onboarding_runtime
    reference_logits = _load_frozen_hf_logits(runtime.fixture_records)
    failures: list[str] = []
    for record in runtime.fixture_records:
        _assert_prompt_tokenization(runtime, record)
        prefixes = [
            record.prompt_token_ids + record.generated_token_ids[:step]
            for step in range(EXPECTED_GENERATED_TOKENS)
        ]
        responses = []
        for start in range(0, len(prefixes), 5):
            batch = runtime.llm.generate(
                prefixes[start : start + 5],
                sampling_params=_one_token_greedy_params(),
                use_tqdm=False,
            )
            assert isinstance(batch, list)
            responses.extend(batch)
        assert len(responses) == EXPECTED_GENERATED_TOKENS
        target_rows: list[torch.Tensor] = []
        for response in responses:
            assert response.finished
            assert len(response.outputs) == 1
            logits = response.outputs[0].generation_logits
            assert isinstance(logits, torch.Tensor)
            assert tuple(logits.shape) == (1, 248320)
            target_rows.append(logits[0].float().cpu())
        target = torch.stack(target_rows)
        reference = reference_logits[record.prompt]
        cosine = F.cosine_similarity(target, reference, dim=-1)
        mean_abs = (target - reference).abs().mean(dim=-1)
        assert torch.isfinite(cosine).all()
        assert torch.isfinite(mean_abs).all()
        min_cosine = cosine.min().item()
        max_mean_abs = mean_abs.max().item()
        if min_cosine < MIN_HF_LOGIT_COSINE or max_mean_abs > MAX_HF_LOGIT_MEAN_ABS_ERROR:
            failures.append(
                f"{record.prompt!r}: min_cosine={min_cosine:.6f}, max_mean_abs={max_mean_abs:.6f}"
            )

    assert not failures, (
        "ModelingV2 teacher-forced logits exceed the independently defined "
        f"HF parity bars (cosine >= {MIN_HF_LOGIT_COSINE:.2f} and mean abs "
        f"error <= {MAX_HF_LOGIT_MEAN_ABS_ERROR:.2f}):\n" + "\n".join(failures)
    )
    _assert_target_identity(runtime.live_model, runtime.target_class)
    _assert_checkpoint_unchanged(runtime)


def test_generation_matches_frozen_builtin_fixture(
    onboarding_runtime: OnboardingRuntime,
) -> None:
    """The synthetic target is token-exact with the mature built-in path."""
    runtime = onboarding_runtime
    expected = _load_frozen_builtin_tokens(runtime.fixture_records)
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
