#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install NeMo-Gym for the trtllm-eval NeMo-Gym benchmarks (gpqa_ng / ifbench_ng / scicode_ng);
# see README.md. NeMo-Gym is a pure-Python (no torch/cuda), opt-in dependency that trtllm-eval
# imports lazily -- it is not a declared TensorRT-LLM dependency. Requires `uv` on PATH.
#
# INSTALL_MODE:
#   in-container (default): install into the current env with --no-deps + only the deps
#       TensorRT-LLM doesn't already ship, then pin openai==2.7.2. (A plain `pip install
#       nemo-gym` would override TensorRT-LLM pins such as fastapi<=0.121.3.)
#   venv: install into a dedicated `uv` venv (isolated, container untouched); point
#       TRTLLM_NEMO_GYM_VENV at it.
#
# openai is pinned to 2.7.2 because NeMo-Gym caps openai<=2.7.2 and injects the running env's
# openai into its per-server venvs; TensorRT-LLM lists openai unpinned, so this is safe.
set -euo pipefail

NEMO_GYM_VERSION="${NEMO_GYM_VERSION:-0.4.0}"
INSTALL_MODE="${INSTALL_MODE:-in-container}"     # in-container (default) | venv
NEMO_GYM_VENV="${TRTLLM_NEMO_GYM_VENV:-$HOME/.cache/trtllm/nemo_gym_venv}"

command -v uv >/dev/null || { echo "[install] ERROR: 'uv' is required (NeMo-Gym builds a uv venv per server)."; exit 1; }
echo "[install] mode=$INSTALL_MODE  python=$(python3 --version)  uv=$(uv --version)"

# ifbench's strict grader checks the literal response; strip the leading reasoning whitespace
# (a '\n\n' left after the <think> block is split out) before grading. Idempotent.
patch_ifbench_strip() {
  "$1" - <<'PY'
import importlib.util, os, sys
spec = importlib.util.find_spec("nemo_gym")
if not spec or not spec.origin:
    print("[install] nemo_gym not importable; skip ifbench strip patch"); sys.exit(0)
app = os.path.join(os.path.dirname(os.path.dirname(spec.origin)), "resources_servers", "ifbench", "app.py")
if not os.path.isfile(app):
    print("[install] ifbench app.py not found; skip strip patch"); sys.exit(0)
src = open(app).read()
if "_trtllm_ifbench_strip" in src:
    print("[install] ifbench strip patch already applied"); sys.exit(0)
anchor = "                final_response_text = last_output.content[0].text\n"
if anchor not in src:
    print("[install] ifbench anchor not found; skip strip patch (nemo-gym layout changed)"); sys.exit(0)
add = anchor + (
    "                # _trtllm_ifbench_strip: drop leading reasoning remnant + whitespace before\n"
    "                # strict grading (a leading '\\n\\n' otherwise fails exact-format instructions).\n"
    '                if "</think>" in final_response_text:\n'
    '                    final_response_text = final_response_text.rsplit("</think>", 1)[-1]\n'
    "                final_response_text = final_response_text.strip()\n"
)
open(app, "w").write(src.replace(anchor, add, 1))
print("[install] applied ifbench strip patch:", app)
PY
}

if [ "$INSTALL_MODE" = "venv" ]; then
  echo "[install] creating dedicated NeMo-Gym venv at $NEMO_GYM_VENV (container untouched)"
  uv venv --python 3.12 "$NEMO_GYM_VENV"
  # Isolated venv: let uv resolve NeMo-Gym's full dependency tree (incl. openai<=2.7.2).
  uv pip install --python "$NEMO_GYM_VENV/bin/python" "nemo-gym==${NEMO_GYM_VERSION}"
  patch_ifbench_strip "$NEMO_GYM_VENV/bin/python"
  echo
  echo "[install] done. NeMo-Gym in venv: $NEMO_GYM_VENV"
  echo "[install] export it before running trtllm-eval:"
  echo "            export TRTLLM_NEMO_GYM_VENV=$NEMO_GYM_VENV"
  echo "[install] sanity: $NEMO_GYM_VENV/bin/gym list benchmarks | grep gpqa"
else
  # Only the NeMo-Gym deps TensorRT-LLM doesn't already ship (openai/fastapi/pydantic/mcp/
  # omegaconf/uvicorn/datasets/aiohttp/tqdm/rich/... are reused, protecting TRT-LLM's pins).
  CURATED=("anthropic<=0.109.2" devtools uvloop itsdangerous hydra-core orjson
           "ray>=2.55.1" pydot yappi gprof2dot "mlflow-skinny>=3.14.0" wandb)
  echo "[install] nemo-gym==${NEMO_GYM_VERSION} (--no-deps: keep it light, protect TRT-LLM pins)"
  python3 -m pip install --no-deps "nemo-gym==${NEMO_GYM_VERSION}"
  echo "[install] curated light deps (only what TensorRT-LLM doesn't already ship)"
  python3 -m pip install "${CURATED[@]}"
  echo "[install] aligning openai to NeMo-Gym's cap (openai==2.7.2; TensorRT-LLM lists openai unpinned)"
  python3 -m pip install "openai==2.7.2"
  patch_ifbench_strip python3
  echo
  echo "[install] done. NeMo-Gym $(python3 -c 'import importlib.metadata as m;print(m.version("nemo-gym"))') installed in the current environment."
  echo "[install] sanity: gym list benchmarks | grep gpqa"
fi

echo "[install] run:  trtllm-eval --model <X> gpqa_ng|ifbench_ng|scicode_ng --num_samples 20"
echo "[install]       (gpqa/ifbench are HF-gated: export HF_TOKEN; scicode needs TRTLLM_NEMO_GYM_SCICODE_H5)"
