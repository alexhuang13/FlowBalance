#!/usr/bin/env bash
# Canonical RLSD math recipe on the current stable_rl/verl stack.
set -xeuo pipefail
project_name=${PROJECT_NAME:-verl-rlsd}
exp_name=${EXP_NAME:-RLSD-Qwen3-8B-math-dapo17k}
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
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}
max_model_len=${MAX_MODEL_LEN:-10240}
max_reprompt_len=${MAX_REPROMPT_LEN:-10240}
sp_size=${SP_SIZE:-8}
gen_tp=${GEN_TP:-1}
n_resp=${N_RESP_PER_PROMPT:-8}
ppo_mini=${PPO_MINI_BATCH_SIZE:-128}
train_bsz=${TRAIN_PROMPT_BSZ:-256}
lr=${LR:-1e-6}
rlsd_lambda=${RLSD_LAMBDA:-0.5}
rlsd_clip=${RLSD_REWEIGHT_CLIP_RANGE:-0.2}
teacher_sync=${RLSD_TEACHER_SYNC_INTERVAL:-20}
lambda_warmup=${RLSD_LAMBDA_WARMUP_STEPS:-0}
lambda_decay=${RLSD_LAMBDA_DECAY_STEPS:-60}
test_freq=${TEST_FREQ:-1}
save_freq=${SAVE_FREQ:-10}
total_steps=${TOTAL_TRAINING_STEPS:-null}
resume_mode=${RESUME_MODE:-disable}

cd "${WORKING_DIR}"
echo "Working dir: ${WORKING_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Train: ${TRAIN_FILE}"
echo "Test:  ${TEST_FILE}"
echo "Ckpts: ${CKPTS_DIR}"
echo "RLSD: lambda=${rlsd_lambda}, clip=${rlsd_clip}, teacher_sync=${teacher_sync}, warmup=${lambda_warmup}, decay=${lambda_decay}"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
python3 ${WORKING_DIR}/recipe/flowopsd/preflight_math_flowopsd.py \
  --repo-root ${WORKING_DIR} --train-file ${TRAIN_FILE} --test-file ${TEST_FILE} \
  --model-path ${MODEL_PATH} --reward-fn-path core/utils/reward_score/sdpo_math_feedback_score.py \
  --reward-fn-name compute_score --nnodes ${NNODES} --gpus-per-node ${N_GPUS_PER_NODE} \
  --sp-size ${sp_size} --train-batch-size ${train_bsz} --rollout-n ${n_resp} \
  --ppo-mini-batch-size ${ppo_mini} --max-prompt-length ${max_prompt_length} \
  --max-response-length ${max_response_length} --max-model-len ${max_model_len} --max-train-rows 50000
fi

ray job submit --address=${RAY_ADDRESS} --no-wait --runtime-env=${RUNTIME_ENV} -- \
 python3 -m recipe.rlsd.main_rlsd \
 data.train_files=${TRAIN_FILE} data.val_files=${TEST_FILE} data.prompt_key=prompt data.truncation=left \
 data.max_prompt_length=${max_prompt_length} data.max_response_length=${max_response_length} \
 data.train_batch_size=${train_bsz} data.gen_batch_size=${train_bsz} data.filter_overlong_prompts=True data.shuffle=True \
 algorithm.adv_estimator=grpo algorithm.norm_adv_by_std_in_grpo=True algorithm.use_kl_in_reward=False \
 algorithm.rollout_correction.rollout_is=null \
 actor_rollout_ref.actor._target_=recipe.rlsd.rlsd_config.RLSDFSDPActorConfig \
 actor_rollout_ref.actor.policy_loss.loss_mode=rlsd actor_rollout_ref.actor.use_kl_loss=False \
 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.28 \
 actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
 actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
 actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
 actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.0 \
 actor_rollout_ref.actor.self_distillation.max_reprompt_len=${max_reprompt_len} \
 actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
 actor_rollout_ref.actor.self_distillation.distillation_topk=null \
 actor_rollout_ref.actor.rlsd.lambda_=${rlsd_lambda} actor_rollout_ref.actor.rlsd.clip_range=${rlsd_clip} \
 actor_rollout_ref.actor.rlsd.lambda_warmup_steps=${lambda_warmup} \
 actor_rollout_ref.actor.rlsd.lambda_decay_steps=${lambda_decay} \
 actor_rollout_ref.actor.rlsd.teacher_sync_interval=${teacher_sync} \
 actor_rollout_ref.model.path=${MODEL_PATH} actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.model.enable_gradient_checkpointing=True actor_rollout_ref.model.trust_remote_code=True \
 actor_rollout_ref.actor.optim.lr=${lr} actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
 actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini} actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_model_len} \
 actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} actor_rollout_ref.actor.grad_clip=1.0 \
 actor_rollout_ref.actor.entropy_coeff=0 actor_rollout_ref.actor.fsdp_config.param_offload=True \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
 actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_model_len} \
 actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} actor_rollout_ref.ref.fsdp_config.param_offload=True \
 actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=${n_resp} \
 actor_rollout_ref.rollout.calculate_log_probs=True actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_model_len} \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.55 actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
 actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.max_num_batched_tokens=${max_model_len} \
 actor_rollout_ref.rollout.max_model_len=${max_model_len} actor_rollout_ref.rollout.temperature=1.0 \
 actor_rollout_ref.rollout.top_p=1.0 actor_rollout_ref.rollout.top_k=-1 \
 actor_rollout_ref.rollout.val_kwargs.temperature=0.6 actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
 actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.n=1 \
 reward_model.reward_manager=custom_dapo reward_model.use_reward_loop=False \
 reward_manager.source=register reward_manager.name=custom_dapo \
 reward_manager.module.path=core/workers/reward_manager/custom_dapo.py reward_manager.module.name=CustomDAPORewardManager \
 custom_reward_function.path=core/utils/reward_score/sdpo_math_feedback_score.py custom_reward_function.name=compute_score \
 ++reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
 ++reward_model.reward_kwargs.overlong_buffer_cfg.len=0 \
 ++reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=0.0 \
 ++reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
 ++reward_model.reward_kwargs.max_resp_len=${max_response_length} trainer.logger='["console","wandb"]' \
 trainer.project_name=${project_name} trainer.experiment_name=${exp_name} \
 trainer.n_gpus_per_node=${N_GPUS_PER_NODE} trainer.nnodes=${NNODES} trainer.val_before_train=False \
 trainer.test_freq=${test_freq} trainer.log_val_generations=4 trainer.save_freq=${save_freq} \
 trainer.total_epochs=500 trainer.total_training_steps=${total_steps} trainer.default_local_dir=${CKPTS_DIR} \
 trainer.validation_data_dir=${CKPTS_DIR}/val/ trainer.resume_mode=${resume_mode}
