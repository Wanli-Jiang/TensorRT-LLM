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
"""Where onboarding evidence lands, and what a report has to carry.

Every gate in this onboarding is judged by someone who did not run it, so a
number without its provenance is not evidence: the same test passes on the
wrong GPU, against a stale wheel, or with a built-in fallback silently
standing in for the target. The closeout criterion enumerates the fields a
report must record; :func:`provenance` collects them once so each test reports
the same set and a missing one is visible as ``null`` rather than as an
omission nobody notices.

Fields that do not apply to a tier are recorded as ``"n/a: <why>"`` rather than
dropped -- an absent ``attention_backend`` reads the same whether the tier had
none or the test forgot to look.

Two of those fields are content hashes rather than names, because a name is not
an identity. :func:`source_digests` binds the report to the bytes of the
*untracked* launcher and test sources that actually ran -- ``git HEAD`` plus a
dirty-file count says nothing about what is in those files -- and
:func:`native_libraries` binds it to the TensorRT-LLM ``.so`` objects the
process actually mapped, which is the only way a stale native build is
distinguishable from the rebuilt one.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import torch

#: Repo root, from this file's location: tests/integration/defs/modeling_v2/.
REPO_ROOT = Path(__file__).resolve().parents[4]

#: Evidence root. The environment variable lets a Reviewer rerun into their own
#: directory without editing a test.
EVIDENCE_ENV = "QWEN3_5_EVIDENCE_DIR"
DEFAULT_EVIDENCE = REPO_ROOT / "qwen38-27b-nvfp4-onboard" / "workspace" / "evidence" / "stage1"

#: Set by the launcher so the report can name the image it actually ran in;
#: enroot/pyxis does not export the ``.sqsh`` path itself.
CONTAINER_ENV = "QWEN3_5_CONTAINER_IMAGE"

#: Set by the launcher to the exact command line under test.
COMMAND_ENV = "QWEN3_5_TEST_COMMAND"

#: Set by the launcher to its own path, so the shell that performed the
#: bootstrap is hashed alongside the Python that ran the assertions.
LAUNCHER_ENV = "QWEN3_5_LAUNCHER"

#: Overrides the run identity; otherwise the Slurm job id, otherwise a local
#: timestamp. One run, one directory.
RUN_ID_ENV = "QWEN3_5_RUN_ID"

_RUN_ID: str | None = None

#: Hashing the mapped native objects reads ~1 GiB; the set cannot change once
#: the process has mapped them, so it is computed once per process.
_NATIVE_LIBRARIES: list[dict[str, object]] | None = None


def evidence_dir() -> Path:
    directory = Path(os.environ.get(EVIDENCE_ENV, DEFAULT_EVIDENCE))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_id() -> str:
    """Stable identifier for this process's run, resolved once."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = (
            os.environ.get(RUN_ID_ENV)
            or os.environ.get("SLURM_JOB_ID")
            or time.strftime("local-%Y%m%d-%H%M%S")
        )
    return _RUN_ID


def run_dir() -> Path:
    directory = evidence_dir() / "runs" / run_id()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def artifacts_dir() -> Path:
    """Artifacts live under the run that produced them, never at a shared path.

    A rerun used to overwrite ``hf_activations.pt`` and ``hf_step_logits.pt``
    while the archived report of the *previous* run went on citing those paths,
    so the report and the bytes it described could silently drift apart. Writing
    them per run removes the shared path entirely rather than racing to copy it.
    """
    directory = run_dir() / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def artifact_digests(paths: list[Path]) -> dict[str, str]:
    """Bind a report to the exact bytes it cites.

    Callers pass the artifacts the run is *required* to have produced, so a
    path that is not there is a failure rather than a row to omit: skipping it
    would publish a report whose artifact table is short by exactly the file
    that went missing, which reads as "that artifact was not part of this tier".
    """
    from ._qwen3_5_checkpoint import hash_file

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AssertionError(f"required artifacts were not written: {missing}")
    return {str(path): hash_file(str(path)) for path in paths}


def source_digests() -> dict[str, str]:
    """Hash the launcher and test sources this run actually executed.

    All of them are untracked, so ``git rev-parse HEAD`` plus a dirty count
    identifies the *repository* the run sat in and says nothing about the code
    that ran. Hashing the bytes closes that gap: a report and a rerun that
    disagree can be told apart from a report and a rerun of different sources.
    """
    from ._qwen3_5_checkpoint import hash_file

    paths = sorted(Path(__file__).resolve().parent.glob("*.py"))
    launcher = os.environ.get(LAUNCHER_ENV)
    if launcher:
        paths.append(Path(launcher).resolve())

    digests: dict[str, str] = {}
    for path in paths:
        try:
            name = str(path.relative_to(REPO_ROOT))
        except ValueError:
            name = str(path)
        digests[name] = hash_file(str(path))
    return digests


def native_libraries() -> list[dict[str, object]]:
    """The TensorRT-LLM native objects this process has actually mapped.

    Read from ``/proc/self/maps`` rather than from the package directory: the
    question a reader has is not "which ``.so`` files exist on disk" but "which
    ones did this interpreter load", and after a rebuild those can differ. Each
    is reported with its size and content hash, so a run against a stale native
    build is distinguishable from one against the rebuilt one -- which
    ``tensorrt_llm.__file__`` alone, a pure-Python path, can never show.
    """
    from ._qwen3_5_checkpoint import hash_file

    global _NATIVE_LIBRARIES
    if _NATIVE_LIBRARIES is not None:
        return _NATIVE_LIBRARIES

    try:
        lines = Path("/proc/self/maps").read_text().splitlines()
    except OSError as exc:
        return [{"error": f"/proc/self/maps unavailable: {exc}"}]

    seen: list[str] = []
    for line in lines:
        fields = line.split(maxsplit=5)
        if len(fields) < 6:
            continue
        path = fields[5].strip()
        if not path.startswith("/") or "tensorrt_llm" not in path:
            continue
        if not (path.endswith(".so") or ".so." in path):
            continue
        if path not in seen:
            seen.append(path)

    libraries: list[dict[str, object]] = []
    for path in sorted(seen):
        try:
            libraries.append(
                {
                    "path": path,
                    "size_bytes": os.path.getsize(path),
                    "blake2b256": hash_file(path),
                }
            )
        except OSError as exc:
            libraries.append({"path": path, "error": str(exc)})
    _NATIVE_LIBRARIES = libraries
    return libraries


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, cwd=REPO_ROOT, timeout=120)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def _trtllm_identity() -> dict[str, object]:
    """Which TensorRT-LLM the process actually imported, if any.

    Best effort on purpose: the HF reference tier must not require the wheel,
    but when the wheel *is* loaded the report has to say which one, because a
    stale install is the failure this field exists to catch.
    """
    import importlib.util

    if importlib.util.find_spec("tensorrt_llm") is None:
        return {"tensorrt_llm": "not installed"}
    try:
        import tensorrt_llm
    except ImportError as exc:
        return {"tensorrt_llm": f"import failed: {exc}"}

    package = getattr(tensorrt_llm, "__file__", None)
    editable = bool(package) and Path(package).resolve().is_relative_to(REPO_ROOT)
    return {
        "tensorrt_llm_version": getattr(tensorrt_llm, "__version__", None),
        "tensorrt_llm_file": package,
        "tensorrt_llm_is_editable_mount": editable,
        "tensorrt_llm_native_libraries": native_libraries(),
    }


def provenance(**extra: object) -> dict[str, object]:
    """The identity a reader needs before trusting any number below it."""
    import transformers

    record: dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run_id": run_id(),
        "command": os.environ.get(COMMAND_ENV, "unset"),
        "launcher": os.environ.get(LAUNCHER_ENV, "unset"),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_qos": os.environ.get("SLURM_JOB_QOS"),
        "slurm_nodes": os.environ.get("SLURM_JOB_NUM_NODES"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "container_image": os.environ.get(CONTAINER_ENV, "unset"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability())
        if torch.cuda.is_available()
        else None,
        "device_uuid": _run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"]),
        "world_size": 1,
        "tp_size": 1,
        "pp_size": 1,
        "trtllm_modeling_v2": os.environ.get("TRTLLM_MODELING_V2", "unset"),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "git_head": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty_files": len(_run(["git", "status", "--porcelain"]).splitlines()),
        "repo_root": str(REPO_ROOT),
        "executed_source_digests": source_digests(),
    }
    record.update(_trtllm_identity())
    record.update(extra)
    return record


def json_block(title: str, payload: object) -> list[str]:
    return ["", f"## {title}", "", "```json", json.dumps(payload, indent=2, default=str), "```"]


def write_report(path: Path, title: str, lines: list[str], elapsed: float | None = None) -> Path:
    """Write one markdown report, with the elapsed time the closeout asks for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# {title}", ""]
    if elapsed is not None:
        body.append(f"Elapsed: {elapsed:.1f}s")
        body.append("")
    body.extend(lines)
    body.append("")
    path.write_text("\n".join(body))
    return path
