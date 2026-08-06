#!/usr/bin/env bash
# Download the data and model required by the SDPO reproduction recipe to a
# shared filesystem. Run this manually on a machine with network access.
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang/data}
MODEL_ROOT=${MODEL_ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang/models}
MODEL_ID=${MODEL_ID:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}
MODEL_DIR=${MODEL_DIR:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-7B}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_ENDPOINT
# Some mirrors do not support the Xet transport used by recent
# huggingface_hub versions. Force regular HTTP downloads for compatibility.
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
mkdir -p "${DATA_ROOT}/rl" "${MODEL_ROOT}"

python3 -m pip install -q "huggingface_hub[cli]" pandas pyarrow datasets
python3 "${SCRIPT_DIR}/prepare_math_data.py" --output_dir "${DATA_ROOT}/rl"
hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}"

echo "Download complete."
echo "Data root: ${DATA_ROOT}/rl"
echo "Model directory: ${MODEL_DIR}"
