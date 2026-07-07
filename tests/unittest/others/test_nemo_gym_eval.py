# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""GPU-free unit tests for the in-process NeMo-Gym eval backend (NeMo-Gym v0.4.0).

These guard the pure-Python surface of :mod:`tensorrt_llm.evaluate.nemo_gym_eval`
(no GPU, no NeMo-Gym install, no model load):

* :meth:`NemoGymRunner.build_command` -- the ``gym eval run`` argv, including optional
  sampling/limit flags, the always-on ``uv_pip_set_python`` fix, and the
  ``skip_venv_if_present`` / ``uv_venv_dir`` offline knobs from
  ``TRTLLM_NEMO_GYM_UV_VENV_DIR``.
* :meth:`NemoGymRunner.parse_aggregate_metrics` -- pulling ``key_metrics`` (and the
  ``pass@1/accuracy`` headline) out of a NeMo-Gym ``*_aggregate_metrics.json``, including
  multi-agent selection by name.
* :meth:`NemoGymEvaluator._report` -- mapping ``key_metrics`` to the returned accuracy,
  including the ``majority@K`` / ``pass@1[avg-of-K]/no_answer`` keys whose ``K`` varies
  with ``num_repeats``.

The end-to-end pipeline (Ray + the 3 NeMo-Gym servers + the vllm_model -> chat-completions
proxy) is covered separately by a live end-to-end trtllm-eval run.
"""

from __future__ import annotations

import json

import pytest

from tensorrt_llm.evaluate import nemo_gym_eval
from tensorrt_llm.evaluate.nemo_gym_eval import (GPQANemoGym, IFBenchNemoGym,
                                                 NemoGymEvaluator, NemoGymRunner,
                                                 SciCodeNemoGym)


def _runner_without_resolution() -> NemoGymRunner:
    """A NemoGymRunner with the gym/root resolution bypassed (no install needed)."""
    return NemoGymRunner(gym_bin="/venv/bin/gym", root="/root")


def _arg_value(argv, flag):
    """Return the value following ``flag`` in an argv list (or None)."""
    return argv[argv.index(flag) + 1] if flag in argv else None


def test_build_command_minimal():
    argv = _runner_without_resolution().build_command(
        benchmark="gpqa",
        base_url="http://127.0.0.1:8123/v1",
        model_name="trtllm-model",
        output_jsonl="/out/gpqa_rollouts.jsonl",
        max_output_tokens=32768,
        temperature=None,
        top_p=None,
        num_samples=None,
        num_repeats=None,
        num_samples_in_parallel=16,
    )
    assert argv[:3] == ["/venv/bin/gym", "eval", "run"]
    assert _arg_value(argv, "--benchmark") == "gpqa"
    assert _arg_value(argv, "--split") == "benchmark"
    assert _arg_value(argv, "--model-type") == "vllm_model"
    assert _arg_value(argv, "--model-url") == "http://127.0.0.1:8123/v1"
    assert _arg_value(argv, "--model") == "trtllm-model"
    assert _arg_value(argv, "--model-api-key") == "dummy_key"
    assert _arg_value(argv, "--output") == "/out/gpqa_rollouts.jsonl"
    assert _arg_value(argv, "--max-output-tokens") == "32768"
    assert _arg_value(argv, "--concurrency") == "16"
    # The uv_pip_set_python fix is always passed (else per-server venvs pick up the
    # ambient openai and conflict with NeMo-Gym's openai<=2.7.2 pin).
    assert "++uv_pip_set_python=true" in argv
    # Optional flags absent when their inputs are None.
    assert "--limit" not in argv
    assert "--num-repeats" not in argv
    assert "--temperature" not in argv


def test_build_command_optional_flags():
    argv = _runner_without_resolution().build_command(
        benchmark="gpqa",
        base_url="http://127.0.0.1:1/v1",
        model_name="m",
        output_jsonl="/o.jsonl",
        max_output_tokens=8192,
        temperature=0.0,
        top_p=0.95,
        num_samples=3,
        num_repeats=8,
        num_samples_in_parallel=4,
    )
    assert _arg_value(argv, "--temperature") == "0.0"
    assert _arg_value(argv, "--top-p") == "0.95"
    assert _arg_value(argv, "--limit") == "3"
    assert _arg_value(argv, "--num-repeats") == "8"
    assert _arg_value(argv, "--concurrency") == "4"


def test_build_command_offline_venv_knobs(monkeypatch):
    monkeypatch.setenv("TRTLLM_NEMO_GYM_UV_VENV_DIR", "/mnt/ng_venvs")
    argv = _runner_without_resolution().build_command(
        benchmark="gpqa",
        base_url="http://127.0.0.1:1/v1",
        model_name="m",
        output_jsonl="/o.jsonl",
        max_output_tokens=8192,
        temperature=None,
        top_p=None,
        num_samples=None,
        num_repeats=None,
        num_samples_in_parallel=16,
    )
    assert "++skip_venv_if_present=true" in argv
    assert "++uv_venv_dir=/mnt/ng_venvs" in argv


def test_build_command_no_offline_knobs_when_unset(monkeypatch):
    monkeypatch.delenv("TRTLLM_NEMO_GYM_UV_VENV_DIR", raising=False)
    argv = _runner_without_resolution().build_command(
        benchmark="gpqa",
        base_url="http://127.0.0.1:1/v1",
        model_name="m",
        output_jsonl="/o.jsonl",
        max_output_tokens=8192,
        temperature=None,
        top_p=None,
        num_samples=None,
        num_repeats=None,
        num_samples_in_parallel=16,
    )
    assert not any(a.startswith("++skip_venv_if_present") for a in argv)
    assert not any(a.startswith("++uv_venv_dir") for a in argv)


# A NeMo-Gym aggregate-metrics file is a JSON list, one entry per agent. Shape mirrors a
# real run (one entry per agent).
_AGG_METRICS = [
    {
        "agent_ref": {"name": "gpqa_mcqa_simple_agent"},
        "agent_metrics": {"mean/reward": 0.71},
        "key_metrics": {
            "mean/reward": 0.71,
            "pass@1/accuracy": 71.21,
            "majority@1/accuracy": 73.0,
            "pass@1/no_answer": 8.08,
        },
        "group_level_metrics": [],
    }
]

# With num_repeats>1 NeMo-Gym bakes the repeat count into the key names.
_AGG_METRICS_AVG_OF_8 = {
    "mean/reward": 0.3333,
    "pass@1[avg-of-8]/accuracy": 33.33,
    "pass@1[avg-of-8]/no_answer": 0.0,
    "pass@8/no_answer": 0.0,
    "majority@8/accuracy": 33.33,
    "pass@1/accuracy": 33.33,
}


def test_parse_aggregate_metrics(tmp_path):
    p = tmp_path / "gpqa_rollouts_aggregate_metrics.json"
    p.write_text(json.dumps(_AGG_METRICS))
    km = NemoGymRunner.parse_aggregate_metrics(str(p))
    assert km["pass@1/accuracy"] == pytest.approx(71.21)
    assert km["pass@1/no_answer"] == pytest.approx(8.08)


def test_parse_aggregate_metrics_selects_agent_by_name(tmp_path):
    entries = [
        {"agent_ref": {"name": "other"}, "key_metrics": {"pass@1/accuracy": 1.0}},
        {"agent_ref": {"name": "want"}, "key_metrics": {"pass@1/accuracy": 99.0}},
    ]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(entries))
    km = NemoGymRunner.parse_aggregate_metrics(str(p), agent_name="want")
    assert km["pass@1/accuracy"] == pytest.approx(99.0)


def test_parse_aggregate_metrics_empty_raises(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text("[]")
    with pytest.raises(RuntimeError):
        NemoGymRunner.parse_aggregate_metrics(str(p))


def test_report_returns_pass_at_1_accuracy():
    evaluator = NemoGymEvaluator.__new__(NemoGymEvaluator)
    evaluator.BENCHMARK = "gpqa"
    evaluator.RESULT_DESC = "gpqa (diamond)"
    score = NemoGymEvaluator._report(evaluator, _AGG_METRICS[0]["key_metrics"])
    assert score == pytest.approx(71.21)


def test_report_handles_avg_of_k_metric_keys():
    """no_answer / majority keys carry the repeat count (K) when num_repeats>1."""
    evaluator = NemoGymEvaluator.__new__(NemoGymEvaluator)
    evaluator.BENCHMARK = "gpqa"
    evaluator.RESULT_DESC = "gpqa (diamond)"
    # Should not raise despite the absence of the bare pass@1/no_answer / majority@1 keys.
    score = NemoGymEvaluator._report(evaluator, _AGG_METRICS_AVG_OF_8)
    assert score == pytest.approx(33.33)
    assert NemoGymEvaluator._metric_by_pattern(
        _AGG_METRICS_AVG_OF_8, "pass@1/no_answer", "pass@1", "/no_answer"
    ) == pytest.approx(0.0)
    assert NemoGymEvaluator._metric_by_pattern(
        _AGG_METRICS_AVG_OF_8, "majority@1/accuracy", "majority@", "/accuracy"
    ) == pytest.approx(33.33)


def test_build_command_model_settings_and_overrides():
    """chat_template_kwargs / extra_body / extra_overrides map to Hydra ++ overrides."""
    argv = _runner_without_resolution().build_command(
        benchmark="ifbench",
        base_url="http://127.0.0.1:1/v1",
        model_name="m",
        output_jsonl="/o.jsonl",
        max_output_tokens=8192,
        temperature=None,
        top_p=None,
        num_samples=None,
        num_repeats=None,
        num_samples_in_parallel=16,
        chat_template_kwargs={"enable_thinking": True},
        extra_body={"skip_special_tokens": True},
        extra_overrides=["++scicode_benchmark_resources_server.resources_servers.scicode.test_data_fpath=/abs/test_data.h5"],
    )
    vm = "policy_model.responses_api_models.vllm_model"
    assert f"++{vm}.chat_template_kwargs={{enable_thinking: true}}" in argv
    assert f"++{vm}.extra_body={{skip_special_tokens: true}}" in argv
    assert "++scicode_benchmark_resources_server.resources_servers.scicode.test_data_fpath=/abs/test_data.h5" in argv


def test_ifbench_report_instruction_and_prompt_level():
    ev = IFBenchNemoGym.__new__(IFBenchNemoGym)
    ev.BENCHMARK = "ifbench"
    ev.RESULT_DESC = "ifbench (instruction following)"
    km = {"mean/reward": 0.748, "mean/follow_all_instructions": 0.72}
    score = IFBenchNemoGym._report(ev, km)
    assert score == pytest.approx(74.8)  # instruction-level = mean/reward * 100


def test_scicode_report_subtask_and_problem():
    ev = SciCodeNemoGym.__new__(SciCodeNemoGym)
    ev.BENCHMARK = "scicode"
    ev.RESULT_DESC = "scicode"
    km = {"subtask_accuracy": 0.4014, "mean/problem_accuracy": 0.179}
    score = SciCodeNemoGym._report(ev, km)
    assert score == pytest.approx(40.14)  # subtask_accuracy * 100


def test_benchmark_request_settings():
    # Per-benchmark request settings match the AA reference config.
    assert GPQANemoGym.ENABLE_THINKING is True and GPQANemoGym.SKIP_SPECIAL_TOKENS is None
    assert IFBenchNemoGym.ENABLE_THINKING is True and IFBenchNemoGym.SKIP_SPECIAL_TOKENS is True
    assert SciCodeNemoGym.ENABLE_THINKING is True


def test_ng_commands_are_registered():
    assert (GPQANemoGym.BENCHMARK, GPQANemoGym.command.name) == ("gpqa", "gpqa_ng")
    assert (IFBenchNemoGym.BENCHMARK, IFBenchNemoGym.command.name) == ("ifbench", "ifbench_ng")
    assert (SciCodeNemoGym.BENCHMARK, SciCodeNemoGym.command.name) == ("scicode", "scicode_ng")


def test_stage_gpqa_from_infra(tmp_path, monkeypatch):
    """gpqa data is sourced from ns_acc_bench_infra (no HF download) in NeMo-Gym's schema."""
    infra = tmp_path / "infra"
    (infra / "datasets" / "gpqa").mkdir(parents=True)
    (infra / "datasets" / "gpqa" / "diamond.jsonl").write_text(json.dumps({
        "problem": "What is 2+2?", "options": "A) 3\nB) 4\nC) 5\nD) 6",
        "A": "3", "B": "4", "C": "5", "D": "6", "expected_answer": "B",
    }) + "\n")
    monkeypatch.setenv("NS_ACC_BENCH_INFRA", str(infra))
    dst = tmp_path / "gpqa_diamond_benchmark.jsonl"
    assert nemo_gym_eval._stage_gpqa_from_infra(str(dst)) is True
    row = json.loads(dst.read_text().splitlines()[0])
    assert sorted(row) == ["expected_answer", "options", "options_text", "problem", "question", "uuid"]
    assert row["expected_answer"] == "B"
    assert row["options_text"] == "A: 3\nB: 4\nC: 5\nD: 6"
    assert row["options"] == [{"A": "3"}, {"B": "4"}, {"C": "5"}, {"D": "6"}]


def test_stage_gpqa_from_infra_missing_returns_false(tmp_path, monkeypatch):
    monkeypatch.setenv("NS_ACC_BENCH_INFRA", str(tmp_path / "nope"))
    assert nemo_gym_eval._stage_gpqa_from_infra(str(tmp_path / "out.jsonl")) is False


def test_data_jsonl_paths():
    assert GPQANemoGym.DATA_JSONL.endswith("gpqa/data/gpqa_diamond_benchmark.jsonl")
    assert IFBenchNemoGym.DATA_JSONL.endswith("ifbench/data/ifbench_benchmark.jsonl")
    assert SciCodeNemoGym.DATA_JSONL.endswith("scicode/data/scicode_benchmark.jsonl")
