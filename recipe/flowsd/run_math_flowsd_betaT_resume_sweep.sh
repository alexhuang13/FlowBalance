#!/usr/bin/env bash
# Serial beta_T (code field: beta_q) sweep resumed from the same Run28 step-170 checkpoint.
set -Eeuo pipefail
trap 'rc=$?; echo "[FlowSD beta_T sweep] failed at line ${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang}
PROJECT_NAME=${PROJECT_NAME:-verl-flowsd}
OUTPUT_ROOT=${OUTPUT_ROOT:-${ROOT}/output}
SOURCE_RUN=${SOURCE_RUN:-${OUTPUT_ROOT}/verl-flowsd/FlowSD-Qwen3-8B-math-dapo17k-betaq1-grpo-signadv-frozenref-etaR15-lr1e6-run28}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${SOURCE_RUN}/global_step_170}
BETA_T_VALUES=${BETA_T_VALUES:-"2 3"}
RUN_TAG=${BETA_T_RUN_TAG:-step170to180-run40}
POLL_SECONDS=${BETA_T_POLL_SECONDS:-60}
TIMEOUT_SECONDS=${BETA_T_TIMEOUT_SECONDS:-1209600}
DRY_RUN=${DRY_RUN:-0}
VAL_STEP=180

DATA_ROOT=${DATA_ROOT:-${ROOT}/data}
MODEL_ROOT=${MODEL_ROOT:-${ROOT}/models}
TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/rl/train_dapo17k.parquet}
TEST_FILE=${TEST_FILE:-${DATA_ROOT}/rl/aime24_30_boxed.parquet}
MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen3-8B}

check_checkpoint() {
  local ckpt=$1 actor=${1}/actor
  [[ -d "$actor" && -f "$ckpt/data.pt" && -f "$actor/fsdp_config.json" ]] || return 1
  local rank0 world count
  rank0=$(find "$actor" -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' -print -quit)
  [[ -n "$rank0" ]] || return 1
  world=$(basename "$rank0" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')
  count=$(find "$actor" -maxdepth 1 -type f -name "model_world_size_${world}_rank_*.pt" | wc -l)
  [[ "$count" -eq "$world" ]] || return 1
  count=$(find "$actor" -maxdepth 1 -type f -name "optim_world_size_${world}_rank_*.pt" | wc -l)
  [[ "$count" -eq "$world" ]] || return 1
}

if ! check_checkpoint "$SOURCE_CHECKPOINT"; then
  echo "ERROR: incomplete source checkpoint: $SOURCE_CHECKPOINT" >&2
  exit 2
fi
for path in "$TRAIN_FILE" "$TEST_FILE"; do [[ -f "$path" ]] || { echo "ERROR: missing $path" >&2; exit 2; }; done
[[ -d "$MODEL_PATH" ]] || { echo "ERROR: missing model $MODEL_PATH" >&2; exit 2; }

read -r -a betas <<< "$BETA_T_VALUES"
[[ ${#betas[@]} -gt 0 ]] || { echo "ERROR: BETA_T_VALUES is empty" >&2; exit 2; }
for beta in "${betas[@]}"; do
  [[ "$beta" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "ERROR: invalid beta_T=$beta" >&2; exit 2; }
done

echo "[FlowSD beta_T sweep] source checkpoint: $SOURCE_CHECKPOINT"
echo "[FlowSD beta_T sweep] beta_T maps to code field flowsd.beta_q"
echo "[FlowSD beta_T sweep] values: ${betas[*]}; serial training + complete benchmark evaluation"

for index in "${!betas[@]}"; do
  beta=${betas[$index]}; beta_tag=${beta//./p}
  exp_name="FlowSD-Qwen3-8B-math-dapo17k-betaT${beta_tag}-etaR15-lr1e6-${RUN_TAG}"
  run_dir="${OUTPUT_ROOT}/${PROJECT_NAME}/${exp_name}"
  complete_file="${run_dir}/val/step_${VAL_STEP}_math_benchmarks_5seeds/.complete"

  echo "[FlowSD beta_T sweep] [$((index+1))/${#betas[@]}] beta_T=${beta}; experiment=${exp_name}"
  if [[ -f "$complete_file" ]]; then
    echo "[FlowSD beta_T sweep] already complete: $complete_file"
    continue
  fi
  if [[ -e "$run_dir" && ! -d "$run_dir" ]]; then
    echo "ERROR: output path exists and is not a directory: $run_dir" >&2; exit 2
  fi

  args=(
    PROJECT_NAME="$PROJECT_NAME" EXP_NAME="$exp_name" OUTPUT_ROOT="$OUTPUT_ROOT" CKPTS_DIR="$run_dir"
    DATA_ROOT="$DATA_ROOT" MODEL_ROOT="$MODEL_ROOT" TRAIN_FILE="$TRAIN_FILE" TEST_FILE="$TEST_FILE" MODEL_PATH="$MODEL_PATH"
    NNODES=4 N_GPUS_PER_NODE=8 SP_SIZE=8 GEN_TP=1 TRAIN_PROMPT_BSZ=256 PPO_MINI_BATCH_SIZE=128 N_RESP_PER_PROMPT=8
    MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=8192 MAX_MODEL_LEN=10240 MAX_REPROMPT_LEN=10240 LR=1e-6
    FLOWSD_BETA_Q="$beta" FLOWSD_ETA_R=15 FLOWSD_BETA_Q_START=null FLOWSD_BETA_Q_END=null
    FLOWSD_ETA_R_START=null FLOWSD_ETA_R_END=null FLOWSD_SCHEDULE_STEPS=0
    FLOWSD_FLOW_GAP_CLIP_LOW=5.0 FLOWSD_FLOW_GAP_CLIP_HIGH=5.0 FLOWSD_IMPORTANCE_RATIO_CAP=1.2
    FLOWSD_RESIDUAL_LENGTH_SCALE=0.0 FLOWSD_CLIP_B=4.0 FLOWSD_RHO=1.0 FLOWSD_LAMBDA_KL=0.0
    FLOWSD_USE_KL=False FLOWSD_GATE_NO_CONTEXT=drop FLOWSD_MIN_GROUP_VALID=2
    FLOWSD_W_CLIP=null FLOWSD_W_MIN=0.0 FLOWSD_W_MAX=10.0
    ENABLE_OVERLONG_BUFFER=False OVERLONG_BUFFER_LEN=0 OVERLONG_PENALTY_FACTOR=0.0 OVERLONG_BUFFER_LOG=False
    TEST_FREQ=0 LOG_VAL_GENERATIONS=0 SAVE_FREQ=180 TOTAL_TRAINING_STEPS=180
    RESUME_MODE=resume_path RESUME_FROM_PATH="$SOURCE_CHECKPOINT" ROLLOUT_DATA_DIR=null NCCL_TIMEOUT=600
    STEP180_VAL_ENABLE=1 STEP180_VAL_STEP=180 STEP180_VAL_OUTPUT_DIR="${run_dir}/val/step_180_math_benchmarks_5seeds"
    STEP180_VAL_SEEDS="0 1 2 3 4" STEP180_VAL_TIMEOUT_SECONDS=604800 STEP180_VAL_KEEP_MERGED_MODEL=0
    STEP180_VAL_TEMPERATURE=0.6 STEP180_VAL_TOP_P=0.95 STEP180_VAL_TOP_K=20
    STEP180_VAL_MAX_TOKENS=38912 STEP180_VAL_MAX_MODEL_LEN=40960 STEP180_VAL_BATCH_SIZE=16
    STEP180_VAL_GPU_MEMORY_UTILIZATION=0.80
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[FlowSD beta_T sweep] command:'; printf ' %q' env "${args[@]}" bash "${SCRIPT_DIR}/run_math_flowsd.sh"; printf '\n'
    continue
  fi

  env "${args[@]}" bash "${SCRIPT_DIR}/run_math_flowsd.sh"
  echo "[FlowSD beta_T sweep] submitted beta_T=${beta}; waiting for step-180 training and all benchmark units"
  deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
  while [[ ! -f "$complete_file" ]]; do
    (( $(date +%s) < deadline )) || { echo "ERROR: timeout waiting for $complete_file" >&2; exit 1; }
    if [[ -d "${run_dir}/global_step_180" ]]; then state='step180 checkpoint ready; evaluation pending';
    else state='resumed training step170->180 pending'; fi
    echo "[FlowSD beta_T sweep] beta_T=${beta}: ${state}; sleeping ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
  echo "[FlowSD beta_T sweep] beta_T=${beta} complete: $complete_file"
done

echo "[FlowSD beta_T sweep] all arms complete: ${betas[*]}"
