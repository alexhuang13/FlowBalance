#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy4/share_303378103/user/audenhuang
ray job submit --address="${RAY_ADDRESS:-http://localhost:8265}" --runtime-env="${RUNTIME_ENV:-${ROOT}/FlowSD/evaluation/math/runtime_env.yaml}" --no-wait -- python3 ${ROOT}/FlowSD/evaluation/math/scripts/ray_selected_steps_entry.py
