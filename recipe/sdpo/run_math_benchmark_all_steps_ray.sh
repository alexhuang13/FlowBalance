#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy4/share_303378103/user/audenhuang
RAY_ADDRESS=${RAY_ADDRESS:-http://localhost:8265}
RUNTIME_ENV=${RUNTIME_ENV:-${ROOT}/FlowSD/evaluation/math/runtime_env.yaml}
ray job submit --address="${RAY_ADDRESS}" --runtime-env="${RUNTIME_ENV}" -- \
 python3 ${ROOT}/FlowSD/evaluation/math/scripts/ray_all_steps_entry.py
