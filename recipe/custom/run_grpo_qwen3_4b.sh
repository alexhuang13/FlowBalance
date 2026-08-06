#!/usr/bin/env bash
set -xeuo pipefail


project_name='GRPO'
exp_name='GRPO-Qwen3-4B-H20'


# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/runtime_env.yaml"}
NNODES=${NNODES:-4}

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data/rl"}
TRAIN_FILE=${TRAIN_FILE:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data/rl/train_dapo17k.parquet"}
TEST_FILE=${TEST_FILE:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data/rl/aime24_30_boxed.parquet"}

MODEL_PATH=${MODEL_PATH:-"/apdcephfs_gy4/share_303378103/user/audenhuang/models/Qwen3-4B"}
CKPTS_DIR=${CKPTS_DIR:-"/apdcephfs_gy4/share_303378103/user/audenhuang/output/${project_name}/${exp_name}"}




echo "Ray address: ${RAY_ADDRESS}"
echo "Working dir: ${WORKING_DIR}"
echo "Runtime env: ${RUNTIME_ENV}"

# -------------------------
# Submit as Ray Job
# (hyperparams identical to your FIRST script)
# -------------------------
ray job submit \
  --address="${RAY_ADDRESS}" \
  --no-wait \
  --runtime-env="${RUNTIME_ENV}" \
  --working-dir="${WORKING_DIR}" \
  -- python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    "$@"
