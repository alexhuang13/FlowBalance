#!/usr/bin/env bash
set -uxo pipefail

export DATA_HOME="/apdcephfs_gy2/share_302625456/data/rl"
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_HOME}/dapo_raw/dapo-math-17k.parquet"}
export POLARIS_DIR=${POLARIS_DIR:-"${DATA_HOME}/polaris_raw"}
export POLARIS_INT_DIR=${POLARIS_INT_DIR:-"${DATA_HOME}/polaris_int_raw"}
export POLARIS_EXTINT_DIR=${POLARIS_EXTINT_DIR:-"${DATA_HOME}/polaris_extint_raw"}
export DOLCI_RAW_DIR=${DOLCI_RAW_DIR:-"${DATA_HOME}/dolci_raw"}
export DOLCI_INT_RAW_DIR=${DOLCI_INT_RAW_DIR:-"${DATA_HOME}/dolci_int_raw"}
export DOLCI_ZERO_RAW_DIR=${DOLCI_ZERO_RAW_DIR:-"${DATA_HOME}/dolci_zero_raw"}
export TEST_FILE=${TEST_FILE:-"${DATA_HOME}/aime-2024.parquet"}
export OVERWRITE=${OVERWRITE:-0}
export PREPARE_ONLY=${PREPARE_ONLY:-all}

mkdir -p "${DATA_HOME}/dapo_raw"
mkdir -p "${POLARIS_DIR}"
mkdir -p "${POLARIS_INT_DIR}"
mkdir -p "${POLARIS_EXTINT_DIR}"
mkdir -p "${DOLCI_RAW_DIR}"
mkdir -p "${DOLCI_INT_RAW_DIR}"
mkdir -p "${DOLCI_ZERO_RAW_DIR}"

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "dapo" ]; } && \
   { [ ! -f "${TRAIN_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  wget -O "${TRAIN_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"
fi

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "dapo" ]; } && \
   { [ ! -f "${TEST_FILE}" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  wget -O "${TEST_FILE}" "https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet?download=true"
fi

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "polaris" ] || [ "${PREPARE_ONLY}" = "polaris_extint" ]; } && \
   { [ ! -f "${POLARIS_DIR}/.download_done" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  POLARIS_DIR="${POLARIS_DIR}" python3 - << 'PY'
import os
from datasets import load_dataset

save_dir = os.environ["POLARIS_DIR"]
ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K")
ds.save_to_disk(save_dir)
PY
  touch "${POLARIS_DIR}/.download_done"
fi

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "polaris" ]; } && \
   { [ ! -f "${POLARIS_INT_DIR}/.filter_done" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  POLARIS_DIR="${POLARIS_DIR}" POLARIS_INT_DIR="${POLARIS_INT_DIR}" python3 - << 'PY'
import os
import re

from datasets import DatasetDict, load_from_disk

raw_dir = os.environ["POLARIS_DIR"]
save_dir = os.environ["POLARIS_INT_DIR"]
strict_int_pattern = re.compile(r"-?\d+")

dataset = load_from_disk(raw_dir)
train = dataset["train"] if isinstance(dataset, DatasetDict) else dataset

def is_strict_integer_answer(example):
    return strict_int_pattern.fullmatch(str(example.get("answer", "")).strip()) is not None

train = train.filter(is_strict_integer_answer)
train.save_to_disk(save_dir)
PY
  touch "${POLARIS_INT_DIR}/.filter_done"
fi

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "polaris" ] || [ "${PREPARE_ONLY}" = "polaris_extint" ]; } && \
   { [ ! -f "${POLARIS_EXTINT_DIR}/.convert_done" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  python3 recipe/custom/convert_polaris_int_subset.py \
    --input_dir "${POLARIS_DIR}" \
    --output_dir "${POLARIS_EXTINT_DIR}" \
    --split train
  touch "${POLARIS_EXTINT_DIR}/.convert_done"
fi

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "dolci" ]; } && \
   { [ ! -f "${DOLCI_RAW_DIR}/.download_done" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  DOLCI_RAW_DIR="${DOLCI_RAW_DIR}" python3 - << 'PY'
import os
from datasets import load_dataset

save_dir = os.environ["DOLCI_RAW_DIR"]
ds = load_dataset("allenai/Dolci-Think-RL-7B")
ds.save_to_disk(save_dir)
PY
  touch "${DOLCI_RAW_DIR}/.download_done"
fi

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "dolci" ]; } && \
   { [ ! -f "${DOLCI_INT_RAW_DIR}/.filter_done" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  DOLCI_RAW_DIR="${DOLCI_RAW_DIR}" DOLCI_INT_RAW_DIR="${DOLCI_INT_RAW_DIR}" python3 - << 'PY'
import os
import re

from datasets import DatasetDict, load_from_disk

raw_dir = os.environ["DOLCI_RAW_DIR"]
save_dir = os.environ["DOLCI_INT_RAW_DIR"]
strict_int_pattern = re.compile(r"-?\d+")

dataset = load_from_disk(raw_dir)
train = dataset["train"] if isinstance(dataset, DatasetDict) else dataset

def is_strict_integer_ground_truth(example):
    ground_truth = example.get("ground_truth")
    if not isinstance(ground_truth, list) or len(ground_truth) != 1:
        return False
    return strict_int_pattern.fullmatch(str(ground_truth[0]).strip()) is not None

train = train.filter(is_strict_integer_ground_truth)
train.save_to_disk(save_dir)
PY
  touch "${DOLCI_INT_RAW_DIR}/.filter_done"
fi

if { [ "${PREPARE_ONLY}" = "all" ] || [ "${PREPARE_ONLY}" = "dolci_zero" ]; } && \
   { [ ! -f "${DOLCI_ZERO_RAW_DIR}/.download_done" ] || [ "${OVERWRITE}" -eq 1 ]; }; then
  DOLCI_ZERO_RAW_DIR="${DOLCI_ZERO_RAW_DIR}" python3 - << 'PY'
import os
from datasets import load_dataset

save_dir = os.environ["DOLCI_ZERO_RAW_DIR"]
ds = load_dataset("allenai/Dolci-RL-Zero-Math-7B")
ds.save_to_disk(save_dir)
PY
  touch "${DOLCI_ZERO_RAW_DIR}/.download_done"
fi
