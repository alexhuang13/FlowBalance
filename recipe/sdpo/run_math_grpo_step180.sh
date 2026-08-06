#!/usr/bin/env bash
# Qwen3 math GRPO terminal-step run: vanilla GRPO, no self-distillation, optional post-step benchmark.
#
# Shares the SDPO overlay entrypoint `recipe.sdpo.main_sdpo`: when loss_mode != "sdpo"
# the SDPO code paths are inert and no teacher module is created (Role.ActorRollout).
# Only the run script + a few overrides differ from the SDPO run.
#
# Usage (from the stable_rl repo root, inside a Ray-connected H20 task):
#   bash recipe/sdpo/run_math_grpo.sh
set -xeuo pipefail

project_name=${PROJECT_NAME:-'verl-grpo'}
exp_name=${EXP_NAME:-'GRPO-Qwen3-4B-math-dapo17k-step180'}

# ----------------------------------------------------------------------------
# Ray / launch
# ----------------------------------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/runtime_env.yaml"}
NNODES=${NNODES:-6}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# ----------------------------------------------------------------------------
# Paths (Route A: local mounted disk). Override to switch to a shared disk.
# ----------------------------------------------------------------------------
DATA_ROOT=${DATA_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data"}
MODEL_ROOT=${MODEL_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/models"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/output"}

TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/rl/train.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_ROOT}/rl/aime-2024.parquet"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-7B"}
CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_ROOT}/${project_name}/${exp_name}"}

# ----------------------------------------------------------------------------
# Lengths (GRPO baseline: shorter budget, match source baseline_grpo config)
# ----------------------------------------------------------------------------
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}

# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------
sp_size=${SP_SIZE:-4}
gen_tp=${GEN_TP:-1}
use_dynamic_bsz=True
offload=True

# ----------------------------------------------------------------------------
# Batch / optim (auto-aligned to allocated GPUs)
# ----------------------------------------------------------------------------
gcd() {
    local a=$1 b=$2 t
    while (( b != 0 )); do
        t=$((a % b))
        a=$b
        b=$t
    done
    echo "$a"
}

lcm() {
    local a=$1 b=$2 g
    g=$(gcd "$a" "$b")
    echo $((a / g * b))
}

round_up_to_multiple() {
    local x=$1 m=$2
    echo $((((x + m - 1) / m) * m))
}

# The rollout batch is train_prompt_bsz * n_resp_per_prompt. verl's
# sequence-length balancing requires it to be divisible by data-parallel size,
# and PPO splitting should also divide it by ppo_mini_batch_size.
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-128}
desired_train_prompt_bsz=${TRAIN_PROMPT_BSZ:-256}
total_gpus=$((NNODES * N_GPUS_PER_NODE))
if (( total_gpus % sp_size != 0 )); then
    echo "ERROR: total_gpus=${total_gpus} must be divisible by SP_SIZE=${sp_size}"
    exit 1
fi
dp_size=$((total_gpus / sp_size))
total_rollout_multiple=$(lcm "${dp_size}" "${train_prompt_mini_bsz}")
prompt_gcd=$(gcd "${total_rollout_multiple}" "${n_resp_per_prompt}")
prompt_bsz_multiple=$((total_rollout_multiple / prompt_gcd))
train_prompt_bsz=$(round_up_to_multiple "${desired_train_prompt_bsz}" "${prompt_bsz_multiple}")
rollout_batch_size=$((train_prompt_bsz * n_resp_per_prompt))
echo "Auto batch sizing: total_gpus=${total_gpus}, sp_size=${sp_size}, dp_size=${dp_size}, ppo_mini_batch=${train_prompt_mini_bsz}, n=${n_resp_per_prompt}, prompt_multiple=${prompt_bsz_multiple}, train_prompt_bsz=${train_prompt_bsz}, rollout_batch=${rollout_batch_size}"
lr=${LR:-1e-6}
loss_mode=vanilla
total_training_steps=${TOTAL_TRAINING_STEPS:-180}
test_freq=${TEST_FREQ:-0}
save_freq=${SAVE_FREQ:-180}
log_val_generations=${LOG_VAL_GENERATIONS:-0}
resume_mode=${RESUME_MODE:-disable}
step180_val_enable=${STEP180_VAL_ENABLE:-1}
step180_val_step=${STEP180_VAL_STEP:-180}
step180_val_output_dir=${STEP180_VAL_OUTPUT_DIR:-${CKPTS_DIR}/val/step_${step180_val_step}_math_benchmarks_5seeds}

if [[ "${step180_val_enable}" == "1" ]]; then
    if (( save_freq <= 0 || step180_val_step % save_freq != 0 )); then
        echo "ERROR: step-${step180_val_step} benchmark requires SAVE_FREQ to divide ${step180_val_step}; got ${save_freq}"
        exit 1
    fi
    if (( total_training_steps < step180_val_step )); then
        echo "ERROR: TOTAL_TRAINING_STEPS=${total_training_steps} < STEP180_VAL_STEP=${step180_val_step}"
        exit 1
    fi
fi

cd "${WORKING_DIR}"
echo "Working dir: ${WORKING_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Train: ${TRAIN_FILE}"
echo "Test:  ${TEST_FILE}"
echo "Ckpts: ${CKPTS_DIR}"

ray job submit --address="${RAY_ADDRESS}" --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.sdpo.main_sdpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${train_prompt_bsz} \
    data.filter_overlong_prompts=True \
    data.shuffle=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT:-600} \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=null \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_model_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_model_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_len} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=custom_dapo \
    reward_model.use_reward_loop=False \
    reward_manager.source=register \
    reward_manager.name=custom_dapo \
    reward_manager.module.path=core/workers/reward_manager/custom_dapo.py \
    reward_manager.module.name=CustomDAPORewardManager \
    custom_reward_function.path=core/utils/reward_score/sdpo_math_feedback_score.py \
    custom_reward_function.name=compute_score \
    ++reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    ++reward_model.reward_kwargs.overlong_buffer_cfg.len=0 \
    ++reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.0 \
    ++reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
    ++reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.log_val_generations=${log_val_generations} \
    trainer.total_epochs=500 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${CKPTS_DIR}/val/" \
    trainer.resume_mode=${resume_mode}

if [[ "${step180_val_enable}" == "1" ]]; then
    echo "Submitting GRPO step-${step180_val_step} benchmark validation watcher"
    RUN_DIR="${CKPTS_DIR}" \
    EXPERIMENT_NAME="${exp_name}" \
    VAL_STEP="${step180_val_step}" \
    VAL_ALGORITHM=grpo \
    VAL_MODEL_TAG_PREFIX=grpo \
    VAL_OUTPUT_DIR="${step180_val_output_dir}" \
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
    RAY_ADDRESS="${RAY_ADDRESS}" \
    RUNTIME_ENV="${RUNTIME_ENV}" \
    bash "${WORKING_DIR}/recipe/flowopsd/submit_step180_val.sh"
fi
