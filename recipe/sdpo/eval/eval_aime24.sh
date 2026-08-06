#!/usr/bin/env bash
# AIME24 pass@k evaluation over merged HuggingFace checkpoints.
#
# Ports self-distillation-analysis/eval/eval.sh with the paper's AIME24 eval recipe:
#   n_sampling=16, k=16, temperature=0.6, top_p=0.95, enable_thinking, max_tokens=38912.
#
# Pre-req: run merge_ckpt.sh first so each step has a `output_hf_model/` directory.
#
# Usage:
#   CKPT_DIR=/path/to/output/<exp> STEPS="10 20 30" bash recipe/sdpo/eval/eval_aime24.sh
set -euo pipefail

# Resolve this script's directory so `utils/` (sibling) is importable, matching the
# source layout (eval.py is run with cwd == eval dir).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Eval-only deps (grader/parser). Eval runs directly with python3 and does not go through ray runtime_env,
# so deps must be installed locally on the eval node; idempotent, near-instant if already installed.
python3 -c "import latex2sympy2, word2number, regex" 2>/dev/null \
    || pip install latex2sympy2 word2number regex

OUTPUT_ROOT=${OUTPUT_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/output"}
DATA_ROOT=${DATA_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data"}
PROJECT_NAME=${PROJECT_NAME:-"verl-sdpo"}
EXPERIMENT=${EXPERIMENT:-"SDPO-Qwen3-8B-math-strict-dapo17k-v1"}
CKPT_DIR=${CKPT_DIR:-"${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT}"}

# AIME24 eval data (verl-native parquet produced by data/prepare_math_data.py).
DATA_PATH=${DATA_PATH:-"${DATA_ROOT}/rl/aime-2024.parquet"}

# Eval recipe (matching the paper's AIME24 main-result settings).
N_SAMPLING=${N_SAMPLING:-16}
K=${K:-16}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
MAX_TOKENS=${MAX_TOKENS:-38912}

# GPUs to use for the vLLM eval (single node).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

STEPS=${STEPS:-"10 20 30 40 50 60 70 80 90 100"}
read -ra step_arr <<< "${STEPS}"

echo "Eval data: ${DATA_PATH}"
echo "Ckpt dir:  ${CKPT_DIR}"
echo "Steps:     ${STEPS}"

cd "${SCRIPT_DIR}"

for step in "${step_arr[@]}"; do
    model_path="${CKPT_DIR}/global_step_${step}/output_hf_model"
    if [ ! -d "${model_path}" ]; then
        echo "[step ${step}] merged model not found (${model_path}); run merge_ckpt.sh first. skip"
        continue
    fi

    echo "==== Evaluating step ${step} (Pass@${K}) ===="
    python3 eval_aime24.py \
        --model_name_or_path "${model_path}" \
        --data_path "${DATA_PATH}" \
        --data_name math \
        --max_tokens ${MAX_TOKENS} \
        --enable_thinking \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --n_sampling ${N_SAMPLING} \
        --k ${K}
done

echo "Eval done. See avg_outputs/ for per-step jsonl results."
