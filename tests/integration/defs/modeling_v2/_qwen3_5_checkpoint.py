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
"""Read-only reader for the published Qwen3.8-27B-NVFP4 checkpoint.

The checkpoint is a ModelOpt ``MIXED_PRECISION`` export: 208 per-tensor FP8
projections (full attention and Gated DeltaNet) and 193 NVFP4 projections
(dense MLP plus the language head), published as three safetensors shards that
this module never writes to.

Two consumers share it. The HF reference tier
(``_qwen3_5_hf_reference.py``) needs every active tensor *dequantized* to
bf16, because ``transformers`` has no ``modelopt`` quantizer -- its
``AUTO_QUANTIZER_MAPPING`` raises ``Unknown quantization type, got modelopt``
and a plain ``from_pretrained`` then reinitializes every quantized weight on a
shape mismatch. The target tier consumes the same tensors quantized, so the
manifest and the namespace split live here once rather than in each.

Namespaces, observed in ``model.safetensors.index.json`` (2194 tensors):

* 1846 active text tensors -- ``model.language_model.*`` (1840 in layers, plus
  embeddings and the final norm) and the four ``lm_head.*`` entries;
* 333 ``model.visual.*`` vision tensors, excluded by the text-only scope;
* 15 ``mtp.*`` speculative tensors, excluded (and named in the checkpoint's own
  ``exclude_modules``).

Those counts are asserted by :func:`assert_manifest_inventory`, not merely
reported: a checkpoint whose namespace or quantization inventory has moved is a
different checkpoint, and every downstream number in this onboarding is quoted
against this one.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch

#: Default location of the published checkpoint. ``LLM_MODELS_ROOT`` is the
#: repository's convention for weight-bearing tests; the environment variable
#: below overrides it outright for a checkout elsewhere.
CHECKPOINT_ENV = "QWEN3_5_NVFP4_CHECKPOINT"
CHECKPOINT_DIRNAME = "Qwen3.8-27B-NVFP4"

#: The 16 values an e2m1 nibble can take (1 sign, 2 exponent, 1 mantissa).
E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

#: NVFP4 block length along the input dimension, from the checkpoint's own
#: ``quantized_layers[*].group_size``.
NVFP4_BLOCK = 16

#: Read-only witness: BLAKE2b-256 over whole file contents, 16 MiB at a time.
CHECKPOINT_HASH_NAME = "blake2b256"
CHECKPOINT_HASH_BYTES = 32
CHECKPOINT_HASH_CHUNK = 16 << 20

#: Publisher metadata rather than published checkpoint files.
VCS_DIRS = frozenset({".git"})

#: What the published checkpoint declares. Asserting these rather than reading
#: them keeps a silently different checkpoint from being measured as this one.
EXPECTED_ARCHITECTURE = "Qwen3_5ForConditionalGeneration"
EXPECTED_MODEL_TYPE = "qwen3_5"
EXPECTED_TEXT_MODEL_TYPE = "qwen3_5_text"
EXPECTED_QUANT_ALGO = "MIXED_PRECISION"
EXPECTED_NUM_LAYERS = 64
EXPECTED_HIDDEN_SIZE = 5120
EXPECTED_VOCAB_SIZE = 248320
EXPECTED_TENSOR_COUNT = 2194
EXPECTED_TEXT_TENSOR_COUNT = 1846
EXPECTED_VISION_TENSOR_COUNT = 333
EXPECTED_MTP_TENSOR_COUNT = 15
EXPECTED_FP8_MODULE_COUNT = 208
EXPECTED_NVFP4_MODULE_COUNT = 193

#: Published files under the checkpoint root, ``.git`` excluded. The read-only
#: witness has to cover a fixed set: a witness that silently hashes 18 files is
#: not weaker by a little, it stops covering whichever file was dropped.
EXPECTED_CHECKPOINT_FILE_COUNT = 19


def checkpoint_path() -> str:
    """Resolve the checkpoint directory, preferring an explicit override."""
    explicit = os.environ.get(CHECKPOINT_ENV)
    if explicit:
        return explicit
    models_root = os.environ.get("LLM_MODELS_ROOT")
    if models_root:
        return os.path.join(models_root, CHECKPOINT_DIRNAME)
    raise RuntimeError(
        f"set {CHECKPOINT_ENV} to the {CHECKPOINT_DIRNAME} directory, or "
        f"LLM_MODELS_ROOT to the directory holding it"
    )


def classify_key(name: str) -> str:
    """Return the namespace of a checkpoint key: text, vision or mtp.

    The split is the text-only scope's whole basis, so it is one function with
    one caller-visible vocabulary rather than a prefix test repeated per site.
    """
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.language_model.") or name.startswith("lm_head."):
        return "text"
    raise ValueError(f"unclassified checkpoint key {name!r}")


def hf_text_name(name: str) -> str:
    """Map an active checkpoint key onto its ``Qwen3_5ForCausalLM`` name.

    ``Qwen3_5ForCausalLM`` is the text-only class: its ``model`` *is* the
    ``Qwen3_5TextModel`` that sits at ``model.language_model`` in the
    multimodal class the checkpoint declares. ``lm_head`` is already top level.
    """
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    if name.startswith("lm_head."):
        return name
    raise ValueError(f"{name!r} is not an active text tensor")


@dataclass(frozen=True)
class QuantizedModule:
    """One projection's published tensors, whatever its algorithm.

    ``weight_scale_2`` is present exactly for NVFP4 (it is the global scale
    that the per-block fp8 scales are relative to) and absent for per-tensor
    FP8, which is how ``algo`` is decided -- the layer's own tensors rather
    than a name pattern.
    """

    prefix: str
    algo: str
    weight: torch.Tensor
    weight_scale: torch.Tensor | None
    weight_scale_2: torch.Tensor | None
    input_scale: torch.Tensor | None


class CheckpointReader:
    """Random-access, read-only reader over the published safetensors shards.

    Shards are memory mapped and tensors fetched by name, so a caller that
    wants one projection pays for one projection -- the alternative, loading a
    10 GiB shard to reach a 2 KiB scale, is what makes a naive manifest pass
    cost more than the model it checks.
    """

    def __init__(self, path: str | None = None) -> None:
        from safetensors import safe_open

        self.path = path or checkpoint_path()
        index_file = os.path.join(self.path, "model.safetensors.index.json")
        with open(index_file) as handle:
            self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        self._handles = {
            shard: safe_open(os.path.join(self.path, shard), framework="pt")
            for shard in sorted(set(self.weight_map.values()))
        }
        with open(os.path.join(self.path, "config.json")) as handle:
            self.config: dict = json.load(handle)

    # -- manifest -------------------------------------------------------

    def keys(self, namespace: str | None = None) -> list[str]:
        """Checkpoint keys, optionally restricted to one namespace."""
        names = sorted(self.weight_map)
        if namespace is None:
            return names
        return [name for name in names if classify_key(name) == namespace]

    def namespace_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for name in self.weight_map:
            key = classify_key(name)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def quantized_layers(self) -> dict[str, dict]:
        """The checkpoint's own per-layer quantization declarations."""
        return self.config["quantization_config"]["quantized_layers"]

    def quant_algo_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.quantized_layers().values():
            algo = record["quant_algo"]
            counts[algo] = counts.get(algo, 0) + 1
        return counts

    # -- tensors --------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        """Fetch one tensor exactly as published."""
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{name!r} is not in this checkpoint")
        return self._handles[shard].get_tensor(name)

    def module_prefixes(self, suffix: str = ".weight") -> Iterator[str]:
        for name in sorted(self.weight_map):
            if name.endswith(suffix):
                yield name[: -len(suffix)]

    def read_module(self, prefix: str) -> QuantizedModule:
        """Read one projection's weight and whichever scales it publishes."""
        weight = self.get(f"{prefix}.weight")
        scale = self.get(f"{prefix}.weight_scale") if self.has(f"{prefix}.weight_scale") else None
        scale_2 = (
            self.get(f"{prefix}.weight_scale_2") if self.has(f"{prefix}.weight_scale_2") else None
        )
        input_scale = (
            self.get(f"{prefix}.input_scale") if self.has(f"{prefix}.input_scale") else None
        )
        if scale is None:
            algo = "BF16"
        elif scale_2 is None:
            algo = "FP8"
        else:
            algo = "NVFP4"
        return QuantizedModule(prefix, algo, weight, scale, scale_2, input_scale)


def assert_manifest_inventory(reader: CheckpointReader) -> dict[str, object]:
    """Hold the published manifest to its exact declared inventory.

    Every count below is a load-bearing claim somewhere downstream: the text
    total is what the bidirectional load accounting is checked against, the
    vision and MTP totals are the text-only scope's exclusion budget, and the
    FP8/NVFP4 split is the ``MIXED_PRECISION`` intent the target must preserve
    rather than collapse. Printing them proves nothing -- a checkpoint that
    gained a vision tensor or moved one projection from FP8 to NVFP4 would
    print a different number and still pass -- so drift fails here, before the
    27 B load, rather than being discovered as an unexplained parity residual.

    The key *sets* are compared too, not just the totals: a fourth namespace or
    a third algorithm is inventory drift even when the counts of the ones we
    know about happen to be unchanged.
    """
    namespaces = reader.namespace_counts()
    algos = reader.quant_algo_counts()
    total = len(reader.weight_map)

    expected_namespaces = {
        "text": EXPECTED_TEXT_TENSOR_COUNT,
        "vision": EXPECTED_VISION_TENSOR_COUNT,
        "mtp": EXPECTED_MTP_TENSOR_COUNT,
    }
    expected_algos = {
        "FP8": EXPECTED_FP8_MODULE_COUNT,
        "NVFP4": EXPECTED_NVFP4_MODULE_COUNT,
    }

    wrong: dict[str, str] = {}
    if total != EXPECTED_TENSOR_COUNT:
        wrong["total_tensors"] = f"{total} != {EXPECTED_TENSOR_COUNT}"
    if namespaces != expected_namespaces:
        wrong["namespaces"] = f"{namespaces} != {expected_namespaces}"
    if algos != expected_algos:
        wrong["quantization"] = f"{algos} != {expected_algos}"
    declared = sum(algos.values())
    if declared != EXPECTED_FP8_MODULE_COUNT + EXPECTED_NVFP4_MODULE_COUNT:
        wrong["declared_modules"] = (
            f"{declared} != {EXPECTED_FP8_MODULE_COUNT + EXPECTED_NVFP4_MODULE_COUNT}"
        )
    if wrong:
        raise AssertionError(
            f"checkpoint manifest inventory drifted from the published "
            f"Qwen3.8-27B-NVFP4 declaration: {wrong}"
        )

    return {
        "total_tensors": total,
        "namespaces": namespaces,
        "quantization": algos,
        "asserted": {
            "total_tensors": EXPECTED_TENSOR_COUNT,
            "namespaces": expected_namespaces,
            "quantization": expected_algos,
        },
    }


def dequantize_fp8_per_tensor(
    weight: torch.Tensor, weight_scale: torch.Tensor, device: torch.device | None = None
) -> torch.Tensor:
    """Dequantize a per-tensor FP8 (e4m3) weight to bf16.

    ModelOpt stores ``w_fp8 = round(w / weight_scale)``, so the inverse is a
    multiply. Getting this direction wrong is finite and wrong rather than
    loud, which is why the negative control in the S1/S2 catalog tests exists.
    """
    target = device or weight.device
    scale = weight_scale.to(device=target, dtype=torch.float32)
    return (weight.to(device=target, dtype=torch.float32) * scale).to(torch.bfloat16)


def dequantize_nvfp4(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dequantize a block-scaled NVFP4 weight to bf16.

    ``weight`` is ``[N, K/2]`` uint8 with two e2m1 nibbles per byte, low nibble
    first; ``block_scale`` is ``[N, K/16]`` fp8_e4m3 relative to the scalar
    ``global_scale``. The element value is
    ``e2m1(nibble) * block_scale * global_scale``.
    """
    target = device or weight.device
    packed = weight.to(device=target)
    if packed.dtype != torch.uint8:
        packed = packed.view(torch.uint8)
    k = packed.shape[-1] * 2
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=target)

    values = torch.empty(*packed.shape[:-1], k, dtype=torch.float32, device=target)
    values[..., 0::2] = lut[(packed & 0x0F).long()]
    values[..., 1::2] = lut[((packed >> 4) & 0x0F).long()]

    scale = block_scale.to(device=target, dtype=torch.float32) * global_scale.to(
        device=target, dtype=torch.float32
    )
    values = values.view(*packed.shape[:-1], k // NVFP4_BLOCK, NVFP4_BLOCK) * scale.unsqueeze(-1)
    return values.view(*packed.shape[:-1], k).to(torch.bfloat16)


def dequantize_module(module: QuantizedModule, device: torch.device | None = None) -> torch.Tensor:
    """Dequantize whichever algorithm this module was published under."""
    if module.algo == "BF16":
        return module.weight.to(device=device or module.weight.device)
    if module.algo == "FP8":
        return dequantize_fp8_per_tensor(module.weight, module.weight_scale, device)
    if module.algo == "NVFP4":
        return dequantize_nvfp4(module.weight, module.weight_scale, module.weight_scale_2, device)
    raise ValueError(f"unknown algorithm {module.algo!r} for {module.prefix}")


def hash_file(path: str) -> str:
    """BLAKE2b-256 over a file's whole content."""
    digest = hashlib.blake2b(digest_size=CHECKPOINT_HASH_BYTES)
    with open(path, "rb", buffering=0) as handle:
        for chunk in iter(lambda: handle.read(CHECKPOINT_HASH_CHUNK), b""):
            digest.update(chunk)
    return f"{CHECKPOINT_HASH_NAME}:{digest.hexdigest()}"


def checkpoint_files(path: str | None = None) -> list[str]:
    """Every published checkpoint file, as paths relative to the root.

    ``.git`` is skipped: it is the publisher's VCS metadata, not a published
    checkpoint file, and its LFS object store is a second 21 GiB copy of the
    same three shards.
    """
    root = path or checkpoint_path()
    names: list[str] = []
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d not in VCS_DIRS)
        for name in files:
            full = os.path.join(directory, name)
            if os.path.isfile(full) and not os.path.islink(full):
                names.append(os.path.relpath(full, root))
    return sorted(names)


def checkpoint_digest(path: str | None = None, max_workers: int = 4) -> dict[str, str]:
    """Content hash of every published checkpoint file.

    "The checkpoint stayed read-only" is a claim the tests make about
    themselves, so the witness has to be one an in-place rewrite cannot
    reproduce: size and mtime are both trivially preserved by a writer that
    keeps the layout and restores the timestamps, which is exactly the silent
    failure this guards. Hashing all 21 GiB costs ~30 s per pass at four
    threads -- ``hashlib`` releases the GIL, so the pool is bounded by Lustre
    read bandwidth, not by Python.
    """
    root = path or checkpoint_path()
    names = checkpoint_files(root)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        hashes = pool.map(hash_file, (os.path.join(root, name) for name in names))
    return dict(zip(names, hashes))


def compare_digests(before: dict[str, str], after: dict[str, str]) -> list[tuple[str, str]]:
    """Name every file whose content, presence or absence changed."""
    changes: list[tuple[str, str]] = []
    for name in sorted(set(before) | set(after)):
        if name not in after:
            changes.append((name, "removed"))
        elif name not in before:
            changes.append((name, "added"))
        elif before[name] != after[name]:
            changes.append((name, f"{before[name]} -> {after[name]}"))
    return changes
