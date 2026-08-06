#!/usr/bin/env bash
# Strict Run28 eta_R ablation.
#
# Training/model/data/optimizer/rollout/reward/FlowOPSD settings are pinned to
# the effective Run28 values. Across runs this script changes only:
#   1. FLOWOPSD_ETA_R (the ablation variable), and
#   2. experiment/output names (to avoid checkpoint collisions).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/output}
PROJECT_NAME=${PROJECT_NAME:-verl-flowopsd}
ETA_SWEEP_VALUES=${ETA_SWEEP_VALUES:-"5 10 30"}
ETA_SWEEP_RUN_TAG=${ETA_SWEEP_RUN_TAG:-run28-eta-ablation}
ETA_SWEEP_EXP_PREFIX=${ETA_SWEEP_EXP_PREFIX:-FlowOPSD-Qwen3-8B-math-dapo17k}
ETA_SWEEP_POLL_SECONDS=${ETA_SWEEP_POLL_SECONDS:-60}
ETA_SWEEP_TIMEOUT_SECONDS=${ETA_SWEEP_TIMEOUT_SECONDS:-1209600}
VAL_STEP=180

# Run28 identity and assets. Paths may be relocated as a unit, but are identical
# for every eta_R arm.
RUN28_DATA_ROOT=${RUN28_DATA_ROOT:-${ROOT}/data}
RUN28_MODEL_ROOT=${RUN28_MODEL_ROOT:-${ROOT}/models}
RUN28_TRAIN_FILE=${RUN28_TRAIN_FILE:-${RUN28_DATA_ROOT}/rl/train_dapo17k.parquet}
RUN28_TEST_FILE=${RUN28_TEST_FILE:-${RUN28_DATA_ROOT}/rl/aime24_30_boxed.parquet}
RUN28_MODEL_PATH=${RUN28_MODEL_PATH:-${RUN28_MODEL_ROOT}/Qwen3-8B}

if [[ -n "${TOTAL_TRAINING_STEPS:-}" && "${TOTAL_TRAINING_STEPS}" != "${VAL_STEP}" ]]; then
    echo "ERROR: strict Run28 eta ablation pins TOTAL_TRAINING_STEPS=${VAL_STEP}; unset the external override"
    exit 1
fi
if [[ -n "${STEP180_VAL_STEP:-}" && "${STEP180_VAL_STEP}" != "${VAL_STEP}" ]]; then
    echo "ERROR: strict Run28 eta ablation pins STEP180_VAL_STEP=${VAL_STEP}; unset the external override"
    exit 1
fi
if [[ -n "${STEP180_VAL_ENABLE:-}" && "${STEP180_VAL_ENABLE}" != "1" ]]; then
    echo "ERROR: strict Run28 eta ablation pins STEP180_VAL_ENABLE=1; unset the external override"
    exit 1
fi

read -r -a etas <<< "${ETA_SWEEP_VALUES}"
if (( ${#etas[@]} == 0 )); then
    echo "ERROR: ETA_SWEEP_VALUES is empty"
    exit 1
fi
for eta in "${etas[@]}"; do
    if [[ ! "${eta}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "ERROR: invalid eta_R value: ${eta}"
        exit 1
    fi
done

echo "[FlowOPSD eta sweep] strict Run28 ablation; eta_R values: ${etas[*]}"
echo "[FlowOPSD eta sweep] fixed: beta_q=1, LR=1e-6, 4x8 H20 layout, SP=8, batch=256, n=8, lengths=2048/8192/10240"
echo "[FlowOPSD eta sweep] each run trains through step ${VAL_STEP}, then completes the same benchmark validation"

for index in "${!etas[@]}"; do
    eta=${etas[$index]}
    eta_tag=${eta//./p}
    exp_name="${ETA_SWEEP_EXP_PREFIX}-betaq1-grpo-signadv-frozenref-etaR${eta_tag}-lr1e6-${ETA_SWEEP_RUN_TAG}"
    run_dir="${OUTPUT_ROOT}/${PROJECT_NAME}/${exp_name}"
    complete_file="${run_dir}/val/step_${VAL_STEP}_math_benchmarks_5seeds/.complete"

    echo "[FlowOPSD eta sweep] [$((index + 1))/${#etas[@]}] eta_R=${eta}, experiment=${exp_name}"
    if [[ -f "${complete_file}" ]]; then
        echo "[FlowOPSD eta sweep] already complete, skipping: ${complete_file}"
        continue
    fi

    # Explicitly pin every Run28 launcher knob so inherited shell variables cannot
    # silently turn this into a multi-variable ablation.
    PROJECT_NAME="${PROJECT_NAME}" \
    EXP_NAME="${exp_name}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    CKPTS_DIR="${run_dir}" \
    DATA_ROOT="${RUN28_DATA_ROOT}" \
    MODEL_ROOT="${RUN28_MODEL_ROOT}" \
    TRAIN_FILE="${RUN28_TRAIN_FILE}" \
    TEST_FILE="${RUN28_TEST_FILE}" \
    MODEL_PATH="${RUN28_MODEL_PATH}" \
    NNODES=4 \
    N_GPUS_PER_NODE=8 \
    SP_SIZE=8 \
    GEN_TP=1 \
    TRAIN_PROMPT_BSZ=256 \
    PPO_MINI_BATCH_SIZE=128 \
    N_RESP_PER_PROMPT=8 \
    MAX_PROMPT_LENGTH=2048 \
    MAX_RESPONSE_LENGTH=8192 \
    MAX_MODEL_LEN=10240 \
    MAX_REPROMPT_LEN=10240 \
    LR=1e-6 \
    FLOWOPSD_BETA_Q=1 \
    FLOWOPSD_ETA_R="${eta}" \
    FLOWOPSD_BETA_Q_START=null \
    FLOWOPSD_BETA_Q_END=null \
    FLOWOPSD_ETA_R_START=null \
    FLOWOPSD_ETA_R_END=null \
    FLOWOPSD_SCHEDULE_STEPS=0 \
    FLOWOPSD_FLOW_GAP_CLIP_LOW=5.0 \
    FLOWOPSD_FLOW_GAP_CLIP_HIGH=5.0 \
    FLOWOPSD_IMPORTANCE_RATIO_CAP=1.2 \
    FLOWOPSD_RESIDUAL_LENGTH_SCALE=0.0 \
    FLOWOPSD_CLIP_B=4.0 \
    FLOWOPSD_RHO=1.0 \
    FLOWOPSD_LAMBDA_KL=0.0 \
    FLOWOPSD_USE_KL=False \
    FLOWOPSD_GATE_NO_CONTEXT=drop \
    FLOWOPSD_MIN_GROUP_VALID=2 \
    FLOWOPSD_W_CLIP=null \
    FLOWOPSD_W_MIN=0.0 \
    FLOWOPSD_W_MAX=10.0 \
    ENABLE_OVERLONG_BUFFER=False \
    OVERLONG_BUFFER_LEN=0 \
    OVERLONG_PENALTY_FACTOR=0.0 \
    OVERLONG_BUFFER_LOG=False \
    TEST_FREQ=1 \
    LOG_VAL_GENERATIONS=4 \
    SAVE_FREQ=10 \
    RESUME_MODE=disable \
    ROLLOUT_DATA_DIR=null \
    NCCL_TIMEOUT=600 \
    TOTAL_TRAINING_STEPS="${VAL_STEP}" \
    STEP180_VAL_ENABLE=1 \
    STEP180_VAL_STEP="${VAL_STEP}" \
    STEP180_VAL_OUTPUT_DIR="${run_dir}/val/step_${VAL_STEP}_math_benchmarks_5seeds" \
    STEP180_VAL_SEEDS="0 1 2 3 4" \
    STEP180_VAL_TIMEOUT_SECONDS=604800 \
    STEP180_VAL_KEEP_MERGED_MODEL=0 \
    STEP180_VAL_TEMPERATURE=0.6 \
    STEP180_VAL_TOP_P=0.95 \
    STEP180_VAL_TOP_K=20 \
    STEP180_VAL_MAX_TOKENS=38912 \
    STEP180_VAL_MAX_MODEL_LEN=40960 \
    STEP180_VAL_BATCH_SIZE=16 \
    STEP180_VAL_GPU_MEMORY_UTILIZATION=0.80 \
    bash "${SCRIPT_DIR}/run_math_flowopsd.sh"

    echo "[FlowOPSD eta sweep] submitted eta_R=${eta}; waiting for training + benchmark completion"
    deadline=$(( $(date +%s) + ETA_SWEEP_TIMEOUT_SECONDS ))
    while [[ ! -f "${complete_file}" ]]; do
        now=$(date +%s)
        if (( now >= deadline )); then
            echo "ERROR: timed out waiting for eta_R=${eta}: ${complete_file}"
            exit 1
        fi
        checkpoint="${run_dir}/global_step_${VAL_STEP}"
        if [[ -d "${checkpoint}" ]]; then
            state="checkpoint exists; benchmark pending"
        else
            state="training/checkpoint pending"
        fi
        echo "[FlowOPSD eta sweep] eta_R=${eta}: ${state}; waiting ${ETA_SWEEP_POLL_SECONDS}s"
        sleep "${ETA_SWEEP_POLL_SECONDS}"
    done
    echo "[FlowOPSD eta sweep] eta_R=${eta} complete: ${complete_file}"
done

echo "[FlowOPSD eta sweep] all strict Run28 eta_R arms completed: ${etas[*]}"
