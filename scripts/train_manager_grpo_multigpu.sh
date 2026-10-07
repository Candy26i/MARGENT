#!/usr/bin/env bash
# Full-parameter manager GRPO on GPUs 1-3 (DeepSpeed ZeRO stage 2 by default).
# Advisors are served by a vLLM process on GPU 0 (start_subagent_server.sh).
#
# Prerequisites:
#   1. GPU 0: the vLLM advisor server is running and healthy.
#      bash scripts/start_subagent_server.sh <base_model> <teacher_id>
#   2. Training env has: pip install -r requirements.txt accelerate deepspeed
#
# Usage:
#   bash scripts/train_manager_grpo_multigpu.sh <teacher_id> [extra cli args...]
#
# Environment overrides:
#   VENV_DIR            virtualenv to activate (default: ./.venv if it exists)
#   ACCELERATE_CONFIG   accelerate config (default: configs/accelerate_zero2_ga2.yaml;
#                       use configs/accelerate_zero3.yaml if stage 2 runs out of memory)
#
# Full example (marginal-value SFT -> binary GRPO): see docs/EXPERIMENTS.md, Step 6.

set -e

TEACHER_ID=${1:?"Usage: $0 <teacher_id> [extra cli args...]"}
shift

# Wait for vLLM server to be ready (up to 120s).
SERVER_URL="http://localhost:8000"
echo "[TRAIN] waiting for vLLM server at ${SERVER_URL} ..."
for i in $(seq 1 24); do
    if curl -sf "${SERVER_URL}/health" > /dev/null 2>&1; then
        echo "[TRAIN] vLLM server is ready."
        break
    fi
    if [ "${i}" -eq 24 ]; then
        echo "[TRAIN] ERROR: vLLM server did not become ready within 120s."
        exit 1
    fi
    sleep 5
done

# Defaults below are overridable: argparse takes the LAST occurrence of a flag,
# and "$@" comes after these, so anything you pass wins.
# num_generations=6 exactly divides the global batch (3 GPUs x bs 2); the
# reward mode (--mgr_adc_mode etc.) is intentionally NOT set here — pass it
# per experiment arm (see docs/EXPERIMENTS.md).
VENV_DIR="${VENV_DIR:-.venv}"
if [ -d "${VENV_DIR}" ]; then
    export VIRTUAL_ENV="${VENV_DIR}" PATH="${VENV_DIR}/bin:$PATH"
fi
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_zero2_ga2.yaml}"
CUDA_VISIBLE_DEVICES=1,2,3 \
PYTHONUTF8=1 \
accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    -m src.pipeline.cli train_manager_grpo \
    --teacher_id "${TEACHER_ID}" \
    --subagent_server_url "${SERVER_URL}" \
    --mgr_full_parameter_rl \
    --mgr_bs 2 \
    --mgr_num_generations 6 \
    --mgr_max_completion_length 3072 \
    --mgr_temperature 1.0 \
    --mgr_grpo_beta 0.01 \
    "$@"
