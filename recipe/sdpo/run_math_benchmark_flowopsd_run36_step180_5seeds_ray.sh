#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy4/share_303378103/user/audenhuang
RUN_DIR=${ROOT}/output/verl-flowopsd/FlowOPSD-Qwen3-8B-math-dapo17k-betaq1-grpo-signadv-frozenref-etaR5-lr1e6-step180-run28fixed-run36
ray job submit \
  --address="${RAY_ADDRESS:-http://localhost:8265}" \
  --runtime-env="${RUNTIME_ENV:-${ROOT}/FlowSD/evaluation/math/runtime_env.yaml}" \
  --no-wait -- \
  python3 ${ROOT}/FlowSD/evaluation/math/scripts/ray_flowopsd_step180_val.py \
    --run-dir "${RUN_DIR}" \
    --experiment-name auden_flowopsd_etaR5_10_30_step180_run28fixed_run36 \
    --step 180 \
    --seeds 0 1 2 3 4 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --max-tokens 38912 \
    --max-model-len 40960 \
    --eval-batch-size 16 \
    --gpu-memory-utilization 0.80
