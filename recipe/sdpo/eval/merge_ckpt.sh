#!/usr/bin/env bash
# Merge FSDP training checkpoints into HuggingFace format for evaluation.
#
# Ports self-distillation-analysis/eval/merge_models.sh, but env-var driven so it
# points at the Route A output dir by default. Uses verl's built-in model_merger,
# so no extra dependency.
#
# Usage:
#   CKPT_DIR=/path/to/output EXPERIMENT=SDPO-... STEPS="10 20 30" bash recipe/sdpo/eval/merge_ckpt.sh
set -euo pipefail

OUTPUT_ROOT=${OUTPUT_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/output"}
PROJECT_NAME=${PROJECT_NAME:-"verl-sdpo"}
# Directory that contains the global_step_* folders. Defaults to the run's ckpt dir.
EXPERIMENT=${EXPERIMENT:-"SDPO-Qwen3-8B-math-strict-dapo17k-v1"}
CKPT_DIR=${CKPT_DIR:-"${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT}"}

# Steps to merge (space-separated). Override via STEPS env var.
STEPS=${STEPS:-"10 20 30 40 50 60 70 80 90 100"}
# Whether to delete the raw FSDP `actor/` shard after a successful merge (keeps disk
# usage low). The last step is always preserved. Set KEEP_ACTOR=1 to keep all.
KEEP_ACTOR=${KEEP_ACTOR:-0}

read -ra step_arr <<< "${STEPS}"
last_step=${step_arr[-1]}

echo "Merging ckpts under: ${CKPT_DIR}"
echo "Steps: ${STEPS}"

for step in "${step_arr[@]}"; do
    actor_dir="${CKPT_DIR}/global_step_${step}/actor"
    hf_dir="${CKPT_DIR}/global_step_${step}/output_hf_model"

    if [ ! -d "${actor_dir}" ] && [ -d "${hf_dir}" ]; then
        echo "[step ${step}] already merged -> ${hf_dir}, skip"
        continue
    fi
    if [ ! -d "${actor_dir}" ]; then
        echo "[step ${step}] actor dir not found (${actor_dir}), skip"
        continue
    fi

    python3 -m verl.model_merger merge \
        --backend fsdp \
        --local_dir "${actor_dir}" \
        --target_dir "${hf_dir}" \
        --use_cpu_initialization

    if [ $? -eq 0 ] && [ "${step}" != "${last_step}" ] && [ "${KEEP_ACTOR}" != "1" ]; then
        rm -rf "${actor_dir}"
    fi
done

echo "Merge done."
