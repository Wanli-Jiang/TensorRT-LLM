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
"""In-process NeMo-Gym accuracy benches for ``trtllm-eval`` (gpqa_ng / ifbench_ng / scicode_ng).

NeMo-Gym (https://github.com/NVIDIA-NeMo/Gym) drives a model over an OpenAI HTTP endpoint
rather than calling the in-process ``LLM`` directly. This module serves that same in-process
``LLM`` on a localhost endpoint (TensorRT-LLM's ``OpenAIServer``, no second copy of the
weights), drives NeMo-Gym v0.4.0's ``gym eval run`` against it via the ``vllm_model``
model-type, and reports the score from ``*_aggregate_metrics.json``.

Install and usage: see ``examples/trtllm-eval/README.md``.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess  # nosec B404
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional, Tuple, Union

import click

from ..logger import logger
from ..sampling_params import SamplingParams
from .interface import Evaluator

_PIP_HINT = (
    "NeMo-Gym is required for this benchmark. Install it in the container with "
    "`bash examples/trtllm-eval/install_nemo_gym.sh` (or into a dedicated venv, then point "
    "TRTLLM_NEMO_GYM_VENV at it). The `gym` CLI must be importable/on PATH."
)

_ROOT_SNIPPET = "import nemo_gym, os; print(os.path.dirname(os.path.dirname(nemo_gym.__file__)))"


def _resolve_gym() -> Tuple[str, str]:
    """Return ``(gym_executable, cwd_root)``.

    Honours ``TRTLLM_NEMO_GYM_VENV`` (dedicated venv), else the ``gym`` entry on ``PATH``.
    ``cwd_root`` holds NeMo-Gym's bundled ``benchmarks/`` (its configs resolve relative to CWD).
    """
    venv = os.environ.get("TRTLLM_NEMO_GYM_VENV")
    if venv:
        gym_bin = os.path.join(venv, "bin", "gym")
        if not os.path.isfile(gym_bin):
            raise RuntimeError(_PIP_HINT)
        py = os.path.join(venv, "bin", "python")
        root = subprocess.run(  # nosec B603
            [py, "-c", _ROOT_SNIPPET], capture_output=True, text=True, check=True
        ).stdout.strip()
        return gym_bin, root
    # find_spec avoids importing (and paying the Ray import cost of) nemo_gym.
    gym_bin = shutil.which("gym")
    if not gym_bin:
        raise RuntimeError(_PIP_HINT)
    import importlib.util

    spec = importlib.util.find_spec("nemo_gym")
    if spec is None or not spec.origin:
        raise RuntimeError(_PIP_HINT)
    return gym_bin, os.path.dirname(os.path.dirname(spec.origin))


@contextmanager
def _serve_llm_in_process(llm: Any, model_name: str, host: str = "127.0.0.1"):
    """Serve ``llm`` as an OpenAI endpoint on a free localhost port; yield its ``/v1`` URL.

    Runs ``OpenAIServer`` in a daemon thread; on exit its lifespan calls ``llm.shutdown()``,
    so the caller must NOT shut the LLM down again.
    """
    import uvicorn

    from ..serve import OpenAIServer

    # Bind now and hand the bound socket to uvicorn to avoid a bind race.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]

    server = OpenAIServer(
        generator=llm,
        model=model_name,
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
    )
    server.binding_addr = f"http://{host}:{port}"
    server.host = host
    server.port = port

    config = uvicorn.Config(server.app, host=host, port=port, log_level="warning")
    userver = uvicorn.Server(config)
    userver.install_signal_handlers = lambda: None  # not on the main thread

    thread = threading.Thread(
        target=lambda: userver.run(sockets=[sock]),
        name="trtllm-nemo-gym-openai",
        daemon=True,
    )
    thread.start()

    base = f"http://{host}:{port}"
    _wait_for_health(f"{base}/health")
    logger.info(f"NeMo-Gym: serving the in-process LLM at {base}/v1 (model={model_name})")
    try:
        yield f"{base}/v1"
    finally:
        userver.should_exit = True
        thread.join(timeout=60)


def _wait_for_health(health_url: str, timeout_s: float = 600.0) -> None:
    """Block until the local OpenAI server answers ``/health`` (or time out)."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as resp:  # nosec B310
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"In-process OpenAI server did not become healthy at {health_url}")


class NemoGymRunner:
    """Thin wrapper over NeMo-Gym's ``gym eval run`` / ``gym eval prepare`` CLIs."""

    def __init__(self, gym_bin: Optional[str] = None, root: Optional[str] = None):
        if gym_bin is None or root is None:
            gym_bin, root = _resolve_gym()
        self.gym_bin = gym_bin
        self.root = root

    # Hydra key path of the vllm_model model-type config.
    _VLLM_MODEL = "policy_model.responses_api_models.vllm_model"

    @staticmethod
    def _hydra_dict(d: dict) -> str:
        """Format a flat dict as a Hydra inline-dict override value (bools lower-cased)."""
        def val(v):
            if isinstance(v, bool):
                return "true" if v else "false"
            return str(v)
        return "{" + ", ".join(f"{k}: {val(v)}" for k, v in d.items()) + "}"

    def build_command(
        self,
        *,
        benchmark: str,
        base_url: str,
        model_name: str,
        output_jsonl: str,
        max_output_tokens: int,
        temperature: Optional[float],
        top_p: Optional[float],
        num_samples: Optional[int],
        num_repeats: Optional[int],
        num_samples_in_parallel: int,
        chat_template_kwargs: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        extra_overrides: Optional[List[str]] = None,
    ) -> List[str]:
        """Assemble the ``gym eval run`` argv (pure; unit-testable).

        ``chat_template_kwargs`` / ``extra_body`` are forwarded to ``/v1/chat/completions``
        via the ``vllm_model`` config; ``extra_overrides`` are extra Hydra ``++key=value``
        strings. The trailing uv overrides are explained inline below.
        """
        argv = [
            self.gym_bin,
            "eval",
            "run",
            "--benchmark",
            benchmark,
            "--split",
            "benchmark",
            "--model-type",
            "vllm_model",
            "--model-url",
            base_url,
            "--model-api-key",
            "dummy_key",
            "--model",
            model_name,
            "--output",
            output_jsonl,
            "--concurrency",
            str(num_samples_in_parallel),
            "--max-output-tokens",
            str(max_output_tokens),
        ]
        if num_samples is not None:
            argv += ["--limit", str(num_samples)]
        if num_repeats is not None:
            argv += ["--num-repeats", str(num_repeats)]
        if temperature is not None:
            argv += ["--temperature", str(temperature)]
        if top_p is not None:
            argv += ["--top-p", str(top_p)]
        if chat_template_kwargs:
            argv.append(f"++{self._VLLM_MODEL}.chat_template_kwargs={self._hydra_dict(chat_template_kwargs)}")
        if extra_body:
            argv.append(f"++{self._VLLM_MODEL}.extra_body={self._hydra_dict(extra_body)}")
        argv += extra_overrides or []
        # Install each server's deps into its own uv venv (not the ambient python), so
        # NeMo-Gym's openai<=2.7.2 pin resolves; reuse/warm those venvs when a dir is set.
        argv.append("++uv_pip_set_python=true")
        uv_venv_dir = os.environ.get("TRTLLM_NEMO_GYM_UV_VENV_DIR")
        if uv_venv_dir:
            argv.append("++skip_venv_if_present=true")
            argv.append(f"++uv_venv_dir={uv_venv_dir}")
        return argv

    def prepare(self, benchmark: str, log_path: str) -> None:
        """Run ``gym eval prepare`` to download + convert the dataset (gpqa needs ``HF_TOKEN``)."""
        argv = [self.gym_bin, "eval", "prepare", "--benchmark", benchmark]
        env = dict(os.environ)
        env.setdefault("UV_LINK_MODE", "copy")
        logger.info(f"NeMo-Gym: preparing `{benchmark}` data (log: {log_path})")
        with open(log_path, "w") as log:
            proc = subprocess.run(argv, cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT)  # nosec B603
        if proc.returncode != 0:
            raise RuntimeError(
                f"`gym eval prepare --benchmark {benchmark}` failed (exit {proc.returncode}); "
                f"see {log_path}. For gpqa, either export HF_TOKEN or stage ns_acc_bench_infra."
            )

    def run(self, argv: List[str], log_path: str) -> None:
        """Run the driver from the NeMo-Gym root (so benchmark paths resolve), tee logs."""
        env = dict(os.environ)
        env.setdefault("UV_LINK_MODE", "copy")  # venv dir / uv cache may be on different filesystems
        uv_cache = os.environ.get("TRTLLM_NEMO_GYM_UV_CACHE")
        if uv_cache:
            env["UV_CACHE_DIR"] = uv_cache
        logger.info(f"NeMo-Gym: launching `gym eval run` (log: {log_path})")
        with open(log_path, "w") as log:
            proc = subprocess.run(  # nosec B603
                argv, cwd=self.root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if proc.returncode != 0:
            tail = _tail(log_path, 40)
            raise RuntimeError(
                f"NeMo-Gym rollout collection failed (exit {proc.returncode}). "
                f"Last lines of {log_path}:\n{tail}"
            )

    @staticmethod
    def parse_aggregate_metrics(metrics_path: str, agent_name: Optional[str] = None) -> dict:
        """Return the ``key_metrics`` dict from a NeMo-Gym ``*_aggregate_metrics.json``.

        The file is a JSON list with one entry per agent; we pick ``agent_name`` if given,
        else the first (single-agent benches have exactly one).
        """
        with open(metrics_path) as f:
            entries = json.load(f)
        if not entries:
            raise RuntimeError(f"NeMo-Gym aggregate metrics file is empty: {metrics_path}")
        entry = entries[0]
        if agent_name is not None:
            for e in entries:
                if (e.get("agent_ref") or {}).get("name") == agent_name:
                    entry = e
                    break
        return entry.get("key_metrics", {}) or {}


def _tail(path: str, n: int) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(log unavailable)"


# --- The evaluator ---------------------------------------------------------
class NemoGymEvaluator(Evaluator):
    """Base class for in-process NeMo-Gym benchmarks.

    Generation + grading happen inside NeMo-Gym; we only host the model and parse the score,
    so ``generate_samples`` / ``compute_score`` are unused and :meth:`evaluate` is overridden.
    """

    BENCHMARK: str = ""            # NeMo-Gym benchmark name (benchmarks/<BENCHMARK>/)
    RESULT_DESC: str = ""          # label used in the result log line
    SERVED_MODEL_NAME: str = "trtllm-model"
    DATA_JSONL: str = ""           # data file relative to the NeMo-Gym root; prepared if absent
    # Request settings forwarded to /v1/chat/completions (None = leave unset).
    ENABLE_THINKING: Optional[bool] = None       # -> chat_template_kwargs.enable_thinking
    SKIP_SPECIAL_TOKENS: Optional[bool] = None    # -> extra_body.skip_special_tokens

    def _chat_template_kwargs(self) -> Optional[dict]:
        return None if self.ENABLE_THINKING is None else {"enable_thinking": self.ENABLE_THINKING}

    def _extra_body(self) -> Optional[dict]:
        return None if self.SKIP_SPECIAL_TOKENS is None else {"skip_special_tokens": self.SKIP_SPECIAL_TOKENS}

    def _extra_overrides(self) -> List[str]:
        """Extra Hydra ``++key=value`` overrides for this benchmark (e.g. an absolute asset
        path). Default: none."""
        return []

    def _ensure_data(self, out_dir: str) -> None:
        """Stage the benchmark data if missing (``gym eval run`` needs it present).

        Default: ``gym eval prepare`` (public download). Subclasses may override to source
        it from ``ns_acc_bench_infra`` instead.
        """
        if self.DATA_JSONL and os.path.isfile(os.path.join(self.runner.root, self.DATA_JSONL)):
            return
        self.runner.prepare(self.BENCHMARK, os.path.join(out_dir, f"{self.BENCHMARK}_prepare.log"))

    def __init__(
        self,
        num_samples: Optional[int] = None,
        num_repeats: Optional[int] = None,
        num_samples_in_parallel: int = 16,
        random_seed: int = 0,
        output_dir: Optional[str] = None,
    ):
        super().__init__(random_seed=random_seed, output_dir=output_dir)
        self.num_samples = num_samples
        self.num_repeats = num_repeats
        self.num_samples_in_parallel = num_samples_in_parallel
        self.runner = NemoGymRunner()

    # Unused: NeMo-Gym owns generation + grading. Present to satisfy the ABC.
    def generate_samples(self) -> Iterable[tuple]:  # pragma: no cover
        raise NotImplementedError("NeMo-Gym benches generate inside NeMo-Gym.")

    def compute_score(self, outputs, references, *auxiliaries) -> float:  # pragma: no cover
        raise NotImplementedError("NeMo-Gym benches grade inside NeMo-Gym.")

    def evaluate(self, llm, sampling_params=None, streaming: bool = False) -> float:
        sampling_params = sampling_params or SamplingParams()
        out_dir = self.output_dir or tempfile.mkdtemp(prefix="trtllm_ng_eval_")
        os.makedirs(out_dir, exist_ok=True)
        output_jsonl = os.path.join(out_dir, f"{self.BENCHMARK}_rollouts.jsonl")
        log_path = os.path.join(out_dir, f"{self.BENCHMARK}_nemo_gym.log")

        self._ensure_data(out_dir)
        with _serve_llm_in_process(llm, self.SERVED_MODEL_NAME) as base_url:
            argv = self.runner.build_command(
                benchmark=self.BENCHMARK,
                base_url=base_url,
                model_name=self.SERVED_MODEL_NAME,
                output_jsonl=output_jsonl,
                max_output_tokens=sampling_params.max_tokens or 8192,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                num_samples=self.num_samples,
                num_repeats=self.num_repeats,
                num_samples_in_parallel=self.num_samples_in_parallel,
                chat_template_kwargs=self._chat_template_kwargs(),
                extra_body=self._extra_body(),
                extra_overrides=self._extra_overrides(),
            )
            self.runner.run(argv, log_path)
        # _serve_llm_in_process exit already shut the LLM down (server lifespan).
        metrics_path = output_jsonl.rsplit(".jsonl", 1)[0] + "_aggregate_metrics.json"
        key_metrics = NemoGymRunner.parse_aggregate_metrics(metrics_path)
        return self._report(key_metrics)

    @staticmethod
    def _metric_by_pattern(
        key_metrics: dict, exact: str, prefix: str, suffix: str
    ) -> Optional[float]:
        """Look up ``exact``, else the first key matching ``prefix``+``suffix``.

        NeMo-Gym bakes the repeat count into some keys (e.g. ``majority@8/accuracy`` when
        ``num_repeats=8``); this finds them regardless of ``K``.
        """
        if exact in key_metrics:
            return key_metrics[exact]
        for k, v in key_metrics.items():
            if k.startswith(prefix) and k.endswith(suffix):
                return v
        return None

    def _report(self, key_metrics: dict) -> float:
        """Log the score (``pass@1/accuracy``) NS-style and return it."""
        accuracy = float(key_metrics.get("pass@1/accuracy", 0.0))
        no_answer = self._metric_by_pattern(key_metrics, "pass@1/no_answer", "pass@1", "/no_answer")
        majority = self._metric_by_pattern(
            key_metrics, "majority@1/accuracy", "majority@", "/accuracy"
        )
        lines = [
            f"NeMo-Gym {self.RESULT_DESC or self.BENCHMARK} results:",
            f"  accuracy        = {accuracy:.2f}%  (pass@1)",
        ]
        if majority is not None:
            lines.append(f"  majority        = {float(majority):.2f}%")
        if no_answer is not None:
            lines.append(f"  no_answer       = {float(no_answer):.2f}%")
        logger.info("\n".join(lines))
        return accuracy

    @classmethod
    def command_harness(cls, ctx, **kwargs) -> None:
        from .. import LLM as PyTorchLLM
        from .._tensorrt_engine import LLM

        llm: Union[LLM, PyTorchLLM] = ctx.obj
        evaluator = cls(
            num_samples=kwargs.pop("num_samples", None),
            num_repeats=kwargs.pop("num_repeats", None),
            num_samples_in_parallel=kwargs.pop("num_samples_in_parallel", 16),
            random_seed=kwargs.pop("random_seed", 0),
            output_dir=kwargs.pop("output_dir", None),
        )
        sp_kwargs: dict[str, Any] = {}
        for key in ("temperature", "top_p"):
            value = kwargs.pop(key, None)
            if value is not None:
                sp_kwargs[key] = float(value)
        top_k = kwargs.pop("top_k", None)
        if top_k is not None:
            sp_kwargs["top_k"] = int(top_k)
        sampling_params = SamplingParams(max_tokens=kwargs.pop("max_output_length"), **sp_kwargs)
        evaluator.evaluate(llm, sampling_params)


def _common_options(default_max_output_length: int):
    """Stack the shared ``trtllm-eval`` options for a NeMo-Gym benchmark."""

    def decorator(func):
        options = [
            click.option(
                "--num_samples",
                type=int,
                default=None,
                help="Number of tasks to evaluate; None means the full split.",
            ),
            click.option(
                "--num_repeats",
                type=int,
                default=None,
                help="Rollouts per task; None uses the benchmark default.",
            ),
            click.option(
                "--num_samples_in_parallel",
                type=int,
                default=16,
                help="Concurrent in-flight rollouts.",
            ),
            click.option(
                "--max_output_length",
                type=int,
                default=default_max_output_length,
                help="Maximum generation length.",
            ),
            click.option("--temperature", type=float, default=None, help="Sampling temperature."),
            click.option("--top_p", type=float, default=None, help="Nucleus top_p."),
            click.option("--top_k", type=int, default=None, help="Top-k sampling."),
            click.option(
                "--random_seed", type=int, default=0, help="Random seed for dataset processing."
            ),
            click.option(
                "--output_dir",
                type=str,
                default=None,
                help="Directory for NeMo-Gym rollouts / metrics / logs.",
            ),
        ]
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


class GPQANemoGym(NemoGymEvaluator):
    """GPQA-diamond via NeMo-Gym (the ``mcqa`` environment, rule-based grader)."""

    BENCHMARK = "gpqa"
    RESULT_DESC = "gpqa (diamond)"
    DATA_JSONL = "benchmarks/gpqa/data/gpqa_diamond_benchmark.jsonl"
    ENABLE_THINKING = True

    def _ensure_data(self, out_dir: str) -> None:
        # gpqa is HF-gated (Idavidrein/gpqa). Prefer the token-free ns_acc_bench_infra copy;
        # fall back to `gym eval prepare` (needs HF_TOKEN) only if the infra is unavailable.
        dst = os.path.join(self.runner.root, self.DATA_JSONL)
        if os.path.isfile(dst) or _stage_gpqa_from_infra(dst):
            return
        self.runner.prepare(self.BENCHMARK, os.path.join(out_dir, "gpqa_prepare.log"))

    @click.command("gpqa_ng")
    @_common_options(default_max_output_length=32768)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        GPQANemoGym.command_harness(ctx, **kwargs)


def _infra_root() -> str:
    """Shared datasets folder: ``NS_ACC_BENCH_INFRA`` or ``<LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra``."""
    env = os.environ.get("NS_ACC_BENCH_INFRA")
    if env:
        return env
    root = os.environ.get("LLM_MODELS_ROOT") or "/code/llm-models"
    return os.path.join(root, "datasets", "ns_acc_bench_infra")


def _resolve_scicode_test_data() -> Optional[str]:
    """Absolute path to SciCode's ``test_data.h5`` (``TRTLLM_NEMO_GYM_SCICODE_H5`` or the infra
    copy), or ``None`` if not found."""
    env = os.environ.get("TRTLLM_NEMO_GYM_SCICODE_H5")
    if env and os.path.isfile(env):
        return os.path.abspath(env)
    cand = os.path.join(_infra_root(), "datasets", "test_data.h5")
    return cand if os.path.isfile(cand) else None


def _stage_gpqa_from_infra(dst_jsonl: str) -> bool:
    """Convert the infra GPQA-diamond dataset to NeMo-Gym's schema at ``dst_jsonl`` (avoids the
    HF-gated download). Returns False if the infra source is absent. The letter->text mapping
    and expected_answer are preserved (no re-shuffle).
    """
    import uuid

    src = os.path.join(_infra_root(), "datasets", "gpqa", "diamond.jsonl")
    if not os.path.isfile(src):
        return False
    letters = ["A", "B", "C", "D"]
    os.makedirs(os.path.dirname(dst_jsonl), exist_ok=True)
    with open(src) as fin, open(dst_jsonl, "w") as fout:
        for line in fin:
            r = json.loads(line)
            question = r["problem"]
            opts = [r[c] for c in letters]  # infra stores per-letter option text
            options_text = "\n".join(f"{c}: {t}" for c, t in zip(letters, opts))
            row = {
                "question": question,
                "options_text": options_text,
                "problem": f"{question}\n{options_text}",
                "options": [{c: t} for c, t in zip(letters, opts)],
                "expected_answer": r["expected_answer"],
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, question)),
            }
            fout.write(json.dumps(row) + "\n")
    logger.info(f"NeMo-Gym: staged GPQA data from infra -> {dst_jsonl} (no HF download)")
    return True


class IFBenchNemoGym(NemoGymEvaluator):
    """AllenAI IFBench (instruction following). The strict grader checks the literal response,
    so ``install_nemo_gym.sh`` patches NeMo-Gym's ifbench server to strip the leading reasoning
    whitespace before grading."""

    BENCHMARK = "ifbench"
    RESULT_DESC = "ifbench (instruction following)"
    DATA_JSONL = "benchmarks/ifbench/data/ifbench_benchmark.jsonl"
    ENABLE_THINKING = True
    SKIP_SPECIAL_TOKENS = True

    def _report(self, key_metrics: dict) -> float:
        # rewards are fractions in [0, 1] -> percent
        instr = float(key_metrics.get("mean/reward", 0.0)) * 100.0
        prompt = key_metrics.get("mean/follow_all_instructions")
        lines = [
            f"NeMo-Gym {self.RESULT_DESC or self.BENCHMARK} results:",
            f"  instruction-level = {instr:.2f}%",
        ]
        if prompt is not None:
            lines.append(f"  prompt-level      = {float(prompt) * 100.0:.2f}%")
        logger.info("\n".join(lines))
        return instr

    @click.command("ifbench_ng")
    @_common_options(default_max_output_length=32768)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        IFBenchNemoGym.command_harness(ctx, **kwargs)


class SciCodeNemoGym(NemoGymEvaluator):
    """SciCode (multi-step scientific code, ``test_aai`` split); code is executed by a Ray
    worker, so it needs ``test_data.h5`` (see ``_resolve_scicode_test_data``)."""

    BENCHMARK = "scicode"
    RESULT_DESC = "scicode"
    DATA_JSONL = "benchmarks/scicode/data/scicode_benchmark.jsonl"
    ENABLE_THINKING = True

    def _extra_overrides(self) -> List[str]:
        h5 = _resolve_scicode_test_data()
        if not h5:
            raise RuntimeError(
                "SciCode needs test_data.h5. Set TRTLLM_NEMO_GYM_SCICODE_H5 to its path, or "
                "stage it under <LLM_MODELS_ROOT>/datasets/ns_acc_bench_infra/datasets/test_data.h5."
            )
        # The scicode resources server runs in its own venv and resolves a relative
        # test_data_fpath against that venv, so it must be absolute.
        return [f"++scicode_benchmark_resources_server.resources_servers.scicode.test_data_fpath={h5}"]

    def _report(self, key_metrics: dict) -> float:
        # rewards are fractions in [0, 1] -> percent
        subtask = float(key_metrics.get("subtask_accuracy", 0.0)) * 100.0
        problem = key_metrics.get("mean/problem_accuracy", key_metrics.get("mean/reward"))
        lines = [
            f"NeMo-Gym {self.RESULT_DESC or self.BENCHMARK} results:",
            f"  subtask_accuracy = {subtask:.2f}%",
        ]
        if problem is not None:
            lines.append(f"  problem_accuracy = {float(problem) * 100.0:.2f}%")
        logger.info("\n".join(lines))
        return subtask

    @click.command("scicode_ng")
    @_common_options(default_max_output_length=32768)
    @click.pass_context
    @staticmethod
    def command(ctx, **kwargs) -> None:
        SciCodeNemoGym.command_harness(ctx, **kwargs)
