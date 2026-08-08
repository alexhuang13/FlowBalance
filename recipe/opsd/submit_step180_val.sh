#!/usr/bin/env bash
# Submit resumable post-checkpoint validation for an OPSD experiment.
set -euo pipefail
ROOT=${ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang}
RAY_ADDRESS=${RAY_ADDRESS:-http://localhost:8265}
RUNTIME_ENV=${RUNTIME_ENV:-${ROOT}/FlowSD/runtime_env.yaml}
RUN_DIR=${RUN_DIR:?RUN_DIR is required}
EXPERIMENT_NAME=${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}
VAL_STEP=${VAL_STEP:-180}
VAL_OUTPUT_DIR=${VAL_OUTPUT_DIR:-${RUN_DIR}/val/step_${VAL_STEP}_math_benchmarks_5seeds}
VAL_SEEDS=${VAL_SEEDS:-"0 1 2 3 4"}
VAL_TIMEOUT_SECONDS=${VAL_TIMEOUT_SECONDS:-604800}
VAL_KEEP_MERGED_MODEL=${VAL_KEEP_MERGED_MODEL:-0}

args=(
  python3 "${ROOT}/FlowSD/evaluation/math/scripts/ray_flowsd_step180_val.py"
  --run-dir "${RUN_DIR}"
  --experiment-name "${EXPERIMENT_NAME}"
  --algorithm "${VAL_ALGORITHM:-opsd}"
  --model-tag-prefix "${VAL_MODEL_TAG_PREFIX:-opsd}"
  --step "${VAL_STEP}"
  --output-dir "${VAL_OUTPUT_DIR}"
  --timeout-seconds "${VAL_TIMEOUT_SECONDS}"
  --seeds ${VAL_SEEDS}
  --temperature "${VAL_TEMPERATURE:-0.6}"
  --top-p "${VAL_TOP_P:-0.95}"
  --top-k "${VAL_TOP_K:-20}"
  --max-tokens "${VAL_MAX_TOKENS:-38912}"
  --max-model-len "${VAL_MAX_MODEL_LEN:-40960}"
  --eval-batch-size "${VAL_BATCH_SIZE:-16}"
  --gpu-memory-utilization "${VAL_GPU_MEMORY_UTILIZATION:-0.80}"
)
if [[ "${VAL_KEEP_MERGED_MODEL}" == "1" ]]; then args+=(--keep-merged-model); fi
ray job submit --address="${RAY_ADDRESS}" --runtime-env="${RUNTIME_ENV}" --no-wait -- "${args[@]}"
