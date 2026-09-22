# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The HF reference tier for Qwen3.8-27B-NVFP4: native modules, native generate.

Every parity claim in this onboarding is against ``transformers``' own
``Qwen3_5ForCausalLM`` -- its modules, its cache, its ``generate()`` -- on the
real checkpoint. Only the *weight load* is ours, and not by preference:
``transformers`` 5.5.4 has no ``modelopt`` entry in ``AUTO_QUANTIZER_MAPPING``,
so ``from_pretrained`` logs "Unknown quantization type, got modelopt ... we will
skip the quantization" and then reinitializes every quantized weight on the
resulting shape mismatch (``ckpt: [17408, 2560] vs model: [17408, 5120]``).
A reference built that way would be random numbers with a straight face.

So the model is constructed from the checkpoint's own ``text_config`` and every
parameter is filled from the published tensors, dequantized per the algorithm
that tensor was published under. The forward that runs on them is stock
``transformers`` with nothing patched, and generation is stock
``model.generate()`` rather than a hand-written decode loop -- which is what
makes the emitted token ids usable as a golden fixture.

Dequantized bf16 is the reference *semantics*, not a claim of bit equality with
the quantized target: NVFP4/FP8 rounding is exactly the residual the parity
bars (cosine >= 0.99, mean abs err <= 0.10) are set to tolerate.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ._qwen3_5_checkpoint import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_MODEL_TYPE,
    EXPECTED_NUM_LAYERS,
    EXPECTED_QUANT_ALGO,
    EXPECTED_TEXT_MODEL_TYPE,
    EXPECTED_VOCAB_SIZE,
    CheckpointReader,
    checkpoint_path,
    classify_key,
    dequantize_module,
    hf_text_name,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

#: The fixed prompts every parity tier replays. Five, plain text, no chat
#: template: the same string has to reach both paths, and a template is one
#: more thing that can differ between them while both look reasonable.
REFERENCE_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "Water boils at a temperature of",
    "The first person to walk on the Moon was",
    "In mathematics, the square root of 144 is",
    "The largest ocean on Earth is the",
)

#: Layers the activation replay hooks. One Gated DeltaNet and one full
#: attention layer from each of the first, middle and last blocks of the
#: (linear, linear, linear, full) x 16 sequence -- a transform that is wrong
#: only at depth is otherwise invisible.
REFERENCE_LAYERS: tuple[int, ...] = (0, 3, 30, 31, 62, 63)


@dataclass
class LoadReport:
    """Bidirectional accounting of the dequantized load.

    A one-sided check passes while half the model is uninitialized: HF's own
    loader reported every scale as UNEXPECTED *and* reinitialized four tensor
    families, and still returned a model. Both directions, or neither.
    """

    parameters_written: list[str] = field(default_factory=list)
    keys_consumed: list[str] = field(default_factory=list)
    keys_excluded: dict[str, int] = field(default_factory=dict)
    algo_counts: dict[str, int] = field(default_factory=dict)

    def assert_complete(self, model: torch.nn.Module, reader: CheckpointReader) -> None:
        written = set(self.parameters_written)
        if len(written) != len(self.parameters_written):
            duplicates = sorted(
                {n for n in self.parameters_written if self.parameters_written.count(n) > 1}
            )
            raise AssertionError(f"parameters written more than once: {duplicates}")
        expected = {name for name, _ in model.named_parameters()}
        missing = sorted(expected - written)
        extra = sorted(written - expected)
        if missing or extra:
            raise AssertionError(
                f"parameter coverage mismatch: {len(missing)} never written "
                f"{missing[:8]}, {len(extra)} written but not a parameter {extra[:8]}"
            )

        consumed = set(self.keys_consumed)
        if len(consumed) != len(self.keys_consumed):
            raise AssertionError("a checkpoint key was consumed more than once")
        active = {name for name in reader.keys() if classify_key(name) == "text"}
        unconsumed = sorted(active - consumed)
        if unconsumed:
            raise AssertionError(
                f"{len(unconsumed)} active text tensors were never consumed: {unconsumed[:8]}"
            )
        foreign = sorted(consumed - active)
        if foreign:
            raise AssertionError(f"consumed non-text tensors: {foreign[:8]}")


def assert_checkpoint_identity(reader: CheckpointReader) -> None:
    """Refuse to build a reference on a checkpoint that is not this one."""
    config = reader.config
    text = config["text_config"]
    facts = {
        "architecture": (config["architectures"][0], EXPECTED_ARCHITECTURE),
        "model_type": (config["model_type"], EXPECTED_MODEL_TYPE),
        "text_model_type": (text["model_type"], EXPECTED_TEXT_MODEL_TYPE),
        "num_hidden_layers": (text["num_hidden_layers"], EXPECTED_NUM_LAYERS),
        "hidden_size": (text["hidden_size"], EXPECTED_HIDDEN_SIZE),
        "vocab_size": (text["vocab_size"], EXPECTED_VOCAB_SIZE),
        "quant_algo": (config["quantization_config"]["quant_algo"], EXPECTED_QUANT_ALGO),
    }
    wrong = {k: v for k, v in facts.items() if v[0] != v[1]}
    if wrong:
        raise AssertionError(f"checkpoint is not Qwen3.8-27B-NVFP4: {wrong}")


def build_hf_text_model(
    path: str | None = None,
    device: str = "cuda",
    attn_implementation: str = "eager",
    reader: CheckpointReader | None = None,
) -> tuple[torch.nn.Module, LoadReport, CheckpointReader]:
    """Construct the native text model and fill it from the checkpoint.

    ``attn_implementation="eager"`` on purpose: the reference's job is to be
    auditable, and ``eager_attention_forward`` is the one path whose softmax
    precision and mask handling are visible in the source being compared to.
    """
    from transformers import AutoConfig
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

    path = path or checkpoint_path()
    reader = reader or CheckpointReader(path)
    assert_checkpoint_identity(reader)

    config = AutoConfig.from_pretrained(path)
    text_config = config.text_config
    text_config._attn_implementation = attn_implementation

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with _no_init_weights(), torch.device(device):
            model = Qwen3_5ForCausalLM(text_config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()

    report = load_dequantized_weights(model, reader, device=device)
    report.assert_complete(model, reader)
    return model, report, reader


@contextlib.contextmanager
def _no_init_weights() -> Iterator[None]:
    """Skip HF's random init; every parameter is overwritten and checked.

    Worth the import dance: initializing 27 B parameters that are about to be
    overwritten costs minutes per run, and the coverage assertions -- not the
    initializer -- are what prove nothing was left unset.
    """
    try:
        from transformers.modeling_utils import no_init_weights
    except ImportError:
        yield
        return
    with no_init_weights():
        yield


def load_dequantized_weights(
    model: torch.nn.Module, reader: CheckpointReader, device: str = "cuda"
) -> LoadReport:
    """Fill every parameter from the published tensors, dequantized.

    Walks the *module* tree rather than the checkpoint: a projection is read as
    one unit (weight plus whichever scales it published) so the algorithm comes
    from the tensors themselves, and a scale that no module claims shows up as
    an unconsumed key rather than being quietly dropped.
    """
    report = LoadReport()
    target = torch.device(device)

    parameters = dict(model.named_parameters())
    ckpt_of_param = {}
    for name in reader.keys("text"):
        if name.endswith(".weight") or name.endswith(".A_log") or name.endswith(".dt_bias"):
            mapped = hf_text_name(name)
            if mapped in parameters:
                ckpt_of_param[mapped] = name

    for param_name, param in parameters.items():
        ckpt_name = ckpt_of_param.get(param_name)
        if ckpt_name is None:
            raise AssertionError(f"no checkpoint tensor maps onto parameter {param_name!r}")
        prefix, _, suffix = ckpt_name.rpartition(".")
        if suffix == "weight":
            module = reader.read_module(prefix)
            value = dequantize_module(module, device=target)
            report.algo_counts[module.algo] = report.algo_counts.get(module.algo, 0) + 1
            report.keys_consumed.append(ckpt_name)
            for extra in ("weight_scale", "weight_scale_2", "input_scale"):
                if reader.has(f"{prefix}.{extra}"):
                    report.keys_consumed.append(f"{prefix}.{extra}")
        else:
            value = reader.get(ckpt_name).to(device=target)
            report.algo_counts["BF16"] = report.algo_counts.get("BF16", 0) + 1
            report.keys_consumed.append(ckpt_name)

        if tuple(value.shape) != tuple(param.shape):
            raise AssertionError(
                f"{param_name}: checkpoint gives {tuple(value.shape)}, module wants "
                f"{tuple(param.shape)}"
            )
        with torch.no_grad():
            param.copy_(value.to(param.dtype))
        report.parameters_written.append(param_name)
        del value

    for name in reader.keys():
        namespace = classify_key(name)
        if namespace != "text":
            report.keys_excluded[namespace] = report.keys_excluded.get(namespace, 0) + 1

    torch.cuda.empty_cache()
    return report


def load_tokenizer(path: str | None = None) -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path or checkpoint_path())


# ----------------------------------------------------------------------
# Activation capture
# ----------------------------------------------------------------------


@dataclass
class ActivationCapture:
    """Hidden states at the boundaries ``source_activation_replay`` compares.

    Keyed ``layer<i>.<site>``; ``site`` is one of ``layer_in``, ``mixer_in``,
    ``mixer_out``, ``layer_out``. ``mixer_in`` is post-``input_layernorm``,
    which is the tensor a target-side module replay actually needs -- feeding
    it ``layer_in`` would fold the norm into the comparison and blur which
    side is wrong.
    """

    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict[str, object] = field(default_factory=dict)

    def add(self, key: str, value: torch.Tensor) -> None:
        self.tensors[key] = value.detach().to("cpu", torch.float32)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"tensors": self.tensors, "meta": self.meta}, path)

    @staticmethod
    def load(path: str) -> ActivationCapture:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        capture = ActivationCapture()
        capture.tensors = blob["tensors"]
        capture.meta = blob["meta"]
        return capture


def _mixer_of(layer: torch.nn.Module) -> tuple[str, torch.nn.Module]:
    if hasattr(layer, "linear_attn"):
        return "linear_attention", layer.linear_attn
    return "full_attention", layer.self_attn


@contextlib.contextmanager
def capture_layers(
    model: torch.nn.Module,
    capture: ActivationCapture,
    layer_indices: Sequence[int] = REFERENCE_LAYERS,
    tag: str = "",
) -> Iterator[ActivationCapture]:
    """Hook the named layers for one forward, then remove every hook."""
    handles: list[torch.utils.hooks.RemovableHandle] = []
    prefix = f"{tag}." if tag else ""

    def make_layer_hooks(index: int, layer: torch.nn.Module) -> None:
        kind, mixer = _mixer_of(layer)
        capture.meta[f"{prefix}layer{index}.kind"] = kind

        def layer_pre(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            capture.add(f"{prefix}layer{index}.layer_in", hidden)
            embeddings = kwargs.get("position_embeddings")
            if embeddings is not None:
                capture.add(f"{prefix}layer{index}.cos", embeddings[0])
                capture.add(f"{prefix}layer{index}.sin", embeddings[1])
            return None

        def layer_post(_module, _args, _kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            capture.add(f"{prefix}layer{index}.layer_out", hidden)

        def mixer_pre(_module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            capture.add(f"{prefix}layer{index}.mixer_in", hidden)
            return None

        def mixer_post(_module, _args, _kwargs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            capture.add(f"{prefix}layer{index}.mixer_out", hidden)

        handles.append(layer.register_forward_pre_hook(layer_pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(layer_post, with_kwargs=True))
        handles.append(mixer.register_forward_pre_hook(mixer_pre, with_kwargs=True))
        handles.append(mixer.register_forward_hook(mixer_post, with_kwargs=True))

    for index in layer_indices:
        make_layer_hooks(index, model.model.layers[index])

    try:
        yield capture
    finally:
        for handle in handles:
            handle.remove()


def module_parameters(model: torch.nn.Module, layer_index: int) -> dict[str, torch.Tensor]:
    """Every parameter of one decoder layer, keyed by its in-layer name."""
    layer = model.model.layers[layer_index]
    return {name: param.detach() for name, param in layer.named_parameters()}


# ----------------------------------------------------------------------
# Native generation
# ----------------------------------------------------------------------


@dataclass
class GenerationRecord:
    """One prompt's native-``generate()`` result, as a re-runnable fixture."""

    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    text: str

    def as_dict(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "text": self.text,
        }


def greedy_generate(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str] = REFERENCE_PROMPTS,
    max_new_tokens: int = 32,
    logits_sink: Callable[[str, torch.Tensor], None] | None = None,
) -> list[GenerationRecord]:
    """Greedy-decode each prompt with stock ``generate()``.

    One prompt per call -- batching would introduce padding, and a padded
    reference is a different computation from the single-sequence one the
    target runs.
    """
    records: list[GenerationRecord] = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_k=None,
                top_p=None,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_logits=True,
                use_cache=True,
            )
        prompt_ids = encoded["input_ids"][0].tolist()
        generated = output.sequences[0, len(prompt_ids) :].tolist()
        if logits_sink is not None:
            logits_sink(prompt, torch.stack([step[0] for step in output.logits]).float().cpu())
        records.append(
            GenerationRecord(
                prompt=prompt,
                prompt_token_ids=prompt_ids,
                generated_token_ids=generated,
                text=tokenizer.decode(generated, skip_special_tokens=False),
            )
        )
    return records


def save_generation_fixture(
    records: Sequence[GenerationRecord], path: str, meta: dict[str, object]
) -> None:
    """Write the golden token fixture that later tiers assert against."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(
            {"meta": meta, "records": [record.as_dict() for record in records]},
            handle,
            indent=2,
        )


def load_generation_fixture(path: str) -> tuple[dict, list[GenerationRecord]]:
    with open(path) as handle:
        blob = json.load(handle)
    return blob["meta"], [GenerationRecord(**record) for record in blob["records"]]
