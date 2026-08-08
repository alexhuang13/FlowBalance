#!/usr/bin/env bash
# OPSD on the same Ray/FSDP/vLLM math setting as FlowSD, RLSD and GRPO.
# Use DRY_RUN=1 to validate and print the exact Ray command without submitting it.
set -Eeuo pipefail
trap 'rc=$?; echo "[OPSD launcher] failed at line ${LINENO}: ${BASH_COMMAND} (exit=${rc})" >&2; exit ${rc}' ERR

project_name=${PROJECT_NAME:-verl-opsd}
exp_name=${EXP_NAME:-OPSD-Qwen3-8B-math-dapo17k-forwardKL-fixedteacher}
RAY_ADDRESS=${RAY_ADDRESS:-http://localhost:8265}
WORKING_DIR=${WORKING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNTIME_ENV=${RUNTIME_ENV:-${WORKING_DIR}/runtime_env.yaml}
NNODES=${NNODES:-4}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
DATA_ROOT=${DATA_ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang/data}
MODEL_ROOT=${MODEL_ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang/models}
OUTPUT_ROOT=${OUTPUT_ROOT:-/apdcephfs_gy4/share_303378103/user/audenhuang/output}
TRAIN_FILE=${TRAIN_FILE:-${DATA_ROOT}/rl/train_dapo17k.parquet}
TEST_FILE=${TEST_FILE:-${DATA_ROOT}/rl/aime24_30_boxed.parquet}
MODEL_PATH=${MODEL_PATH:-${MODEL_ROOT}/Qwen3-8B}
CKPTS_DIR=${CKPTS_DIR:-${OUTPUT_ROOT}/${project_name}/${exp_name}}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10240}
MAX_REPROMPT_LEN=${MAX_REPROMPT_LEN:-10240}
SP_SIZE=${SP_SIZE:-8}
GEN_TP=${GEN_TP:-1}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-256}
LR=${LR:-1e-6}
OPSD_ALPHA=${OPSD_ALPHA:-0.0}
OPSD_DISTILLATION_TOPK=${OPSD_DISTILLATION_TOPK:-100}
OPSD_TOKEN_LOSS_CLIP=${OPSD_TOKEN_LOSS_CLIP:-0.06}
OPSD_SOLUTION_SOURCE=${OPSD_SOLUTION_SOURCE:-external_first}
OPSD_STUDENT_THINKING=${OPSD_STUDENT_THINKING:-true}
OPSD_TEACHER_THINKING=${OPSD_TEACHER_THINKING:-true}
TEST_FREQ=${TEST_FREQ:-1}
SAVE_FREQ=${SAVE_FREQ:-10}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-4}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
RESUME_MODE=${RESUME_MODE:-disable}
DRY_RUN=${DRY_RUN:-0}
SKIP_PREFLIGHT=${SKIP_PREFLIGHT:-0}
STEP180_VAL_ENABLE=${STEP180_VAL_ENABLE:-0}
STEP180_VAL_STEP=${STEP180_VAL_STEP:-180}
STEP180_VAL_OUTPUT_DIR=${STEP180_VAL_OUTPUT_DIR:-${CKPTS_DIR}/val/step_${STEP180_VAL_STEP}_math_benchmarks_5seeds}

bool_check() { [[ "$2" == "true" || "$2" == "false" ]] || { echo "ERROR: $1 must be true or false, got '$2'" >&2; exit 2; }; }
bool_check OPSD_STUDENT_THINKING "$OPSD_STUDENT_THINKING"
bool_check OPSD_TEACHER_THINKING "$OPSD_TEACHER_THINKING"

if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  python3 "${WORKING_DIR}/recipe/opsd/preflight_opsd_contract.py" \
    --repo-root "$WORKING_DIR" --runtime-env "$RUNTIME_ENV" \
    --train-file "$TRAIN_FILE" --test-file "$TEST_FILE" --model-path "$MODEL_PATH" \
    --nnodes "$NNODES" --gpus-per-node "$N_GPUS_PER_NODE" --sp-size "$SP_SIZE" \
    --train-batch-size "$TRAIN_PROMPT_BSZ" --rollout-n "$N_RESP_PER_PROMPT" \
    --ppo-mini-batch-size "$PPO_MINI_BATCH_SIZE" --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-response-length "$MAX_RESPONSE_LENGTH" --max-model-len "$MAX_MODEL_LEN" \
    --max-reprompt-len "$MAX_REPROMPT_LEN" --alpha "$OPSD_ALPHA" \
    --distillation-topk "$OPSD_DISTILLATION_TOPK" --token-loss-clip "$OPSD_TOKEN_LOSS_CLIP"
fi

hydra_args=(
  "data.train_files=${TRAIN_FILE}" "data.val_files=${TEST_FILE}" "data.prompt_key=prompt"
  "data.truncation=left" "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}" "data.train_batch_size=${TRAIN_PROMPT_BSZ}"
  "data.gen_batch_size=${TRAIN_PROMPT_BSZ}" "data.apply_chat_template_kwargs.enable_thinking=${OPSD_STUDENT_THINKING}"
  "algorithm.adv_estimator=grpo" "algorithm.norm_adv_by_std_in_grpo=True"
  "algorithm.use_kl_in_reward=False" "algorithm.rollout_correction.rollout_is=null"
  "actor_rollout_ref.actor._target_=recipe.opsd.opsd_config.OPSDFSDPActorConfig"
  "actor_rollout_ref.actor.policy_loss.loss_mode=opsd" "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.self_distillation._target_=recipe.opsd.opsd_config.OPSDSelfDistillationConfig"
  "actor_rollout_ref.actor.self_distillation.full_logit_distillation=True"
  "actor_rollout_ref.actor.self_distillation.alpha=${OPSD_ALPHA}"
  "actor_rollout_ref.actor.self_distillation.teacher_regularization=ema"
  "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0"
  "actor_rollout_ref.actor.self_distillation.distillation_topk=${OPSD_DISTILLATION_TOPK}"
  "actor_rollout_ref.actor.self_distillation.distillation_add_tail=True"
  "actor_rollout_ref.actor.self_distillation.token_loss_clip=${OPSD_TOKEN_LOSS_CLIP}"
  "actor_rollout_ref.actor.self_distillation.is_clip=null"
  "actor_rollout_ref.actor.self_distillation.max_reprompt_len=${MAX_REPROMPT_LEN}"
  "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=False"
  "actor_rollout_ref.actor.self_distillation.solution_source=${OPSD_SOLUTION_SOURCE}"
  "actor_rollout_ref.actor.self_distillation.teacher_enable_thinking=${OPSD_TEACHER_THINKING}"
  "actor_rollout_ref.model.path=${MODEL_PATH}" "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True" "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.actor.optim.lr=${LR}" "actor_rollout_ref.actor.optim.lr_warmup_steps=10"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.use_dynamic_bsz=True" "actor_rollout_ref.actor.grad_clip=1.0"
  "actor_rollout_ref.actor.entropy_coeff=0" "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True" "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_MODEL_LEN}"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SP_SIZE}"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN}"
  "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SP_SIZE}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}" "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.calculate_log_probs=True" "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.55"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=True"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.temperature=1.0" "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1" "actor_rollout_ref.rollout.val_kwargs.temperature=0.6"
  "actor_rollout_ref.rollout.val_kwargs.top_p=0.95" "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.n=1" "reward_model.reward_manager=custom_dapo"
  "reward_manager.source=register" "reward_manager.name=custom_dapo"
  "reward_manager.module.path=core/workers/reward_manager/custom_dapo.py"
  "reward_manager.module.name=CustomDAPORewardManager"
  "custom_reward_function.path=core/utils/reward_score/sdpo_math_feedback_score.py"
  "custom_reward_function.name=compute_score" "trainer.project_name=${project_name}"
  "trainer.experiment_name=${exp_name}" "trainer.n_gpus_per_node=${N_GPUS_PER_NODE}"
  "trainer.nnodes=${NNODES}" "trainer.test_freq=${TEST_FREQ}" "trainer.save_freq=${SAVE_FREQ}"
  "trainer.log_val_generations=${LOG_VAL_GENERATIONS}" "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
  "trainer.default_local_dir=${CKPTS_DIR}" "trainer.validation_data_dir=${CKPTS_DIR}/val/"
  "trainer.resume_mode=${RESUME_MODE}"
)
cmd=(ray job submit "--address=${RAY_ADDRESS}" --no-wait "--runtime-env=${RUNTIME_ENV}" -- python3 -m recipe.opsd.main_opsd "${hydra_args[@]}")
printf '[OPSD launcher] cwd=%q\n' "$WORKING_DIR"
printf '[OPSD launcher] command:'; printf ' %q' "${cmd[@]}"; printf '\n'
if [[ "$DRY_RUN" == "1" ]]; then
  if [[ "${STEP180_VAL_ENABLE}" == "1" ]]; then
    printf '[OPSD launcher] validation watcher: run_dir=%q step=%q output=%q\n' "$CKPTS_DIR" "$STEP180_VAL_STEP" "$STEP180_VAL_OUTPUT_DIR"
  fi
  exit 0
fi
cd "$WORKING_DIR"
"${cmd[@]}"
if [[ "${STEP180_VAL_ENABLE}" == "1" ]]; then
  echo "[OPSD launcher] submitting step-${STEP180_VAL_STEP} five-seed benchmark watcher"
  RUN_DIR="$CKPTS_DIR" \
  EXPERIMENT_NAME="$exp_name" \
  VAL_STEP="$STEP180_VAL_STEP" \
  VAL_OUTPUT_DIR="$STEP180_VAL_OUTPUT_DIR" \
  VAL_ALGORITHM="${VAL_ALGORITHM:-opsd}" \
  VAL_MODEL_TAG_PREFIX="${VAL_MODEL_TAG_PREFIX:-opsd}" \
  VAL_SEEDS="${STEP180_VAL_SEEDS:-0 1 2 3 4}" \
  VAL_TIMEOUT_SECONDS="${STEP180_VAL_TIMEOUT_SECONDS:-604800}" \
  VAL_KEEP_MERGED_MODEL="${STEP180_VAL_KEEP_MERGED_MODEL:-0}" \
  VAL_TEMPERATURE="${STEP180_VAL_TEMPERATURE:-0.6}" \
  VAL_TOP_P="${STEP180_VAL_TOP_P:-0.95}" \
  VAL_TOP_K="${STEP180_VAL_TOP_K:-20}" \
  VAL_MAX_TOKENS="${STEP180_VAL_MAX_TOKENS:-38912}" \
  VAL_MAX_MODEL_LEN="${STEP180_VAL_MAX_MODEL_LEN:-40960}" \
  VAL_BATCH_SIZE="${STEP180_VAL_BATCH_SIZE:-16}" \
  VAL_GPU_MEMORY_UTILIZATION="${STEP180_VAL_GPU_MEMORY_UTILIZATION:-0.80}" \
  RAY_ADDRESS="$RAY_ADDRESS" RUNTIME_ENV="$RUNTIME_ENV" \
  bash "${WORKING_DIR}/recipe/opsd/submit_step180_val.sh"
fi
