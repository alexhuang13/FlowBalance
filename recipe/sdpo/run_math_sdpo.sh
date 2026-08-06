#!/usr/bin/env bash
# SDPO (Self-Distillation Policy Optimization) baseline run script.
#
# Mirrors the stable_rl launch convention (recipe/custom/run_dapo_qwen3_4b.sh) but
# targets the SDPO overlay entrypoint `recipe.sdpo.main_sdpo`. All data / model /
# ckpt paths are env-var driven (Route A: local mounted disk by default), so the
# same script can later switch to the shared disk by only changing env vars.
#
# Usage (from the stable_rl repo root, inside a Ray-connected H20 task):
#   bash recipe/sdpo/run_math_sdpo.sh
set -xeuo pipefail

project_name='verl-sdpo'
exp_name=${EXP_NAME:-'SDPO-Qwen3-8B-math-strict-dapo17k-v1'}

# ----------------------------------------------------------------------------
# Ray / launch
# ----------------------------------------------------------------------------
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/runtime_env.yaml"}
# Strict reproduction default: 32 GPUs keeps upstream train_batch_size=256 valid with SP=4.
# Override NNODES=6 to use all 48 GPUs; that will auto-align train_batch_size to 288.
NNODES=${NNODES:-4}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

# ----------------------------------------------------------------------------
# Paths (Route A: local mounted disk). Override to switch to a shared disk.
# ----------------------------------------------------------------------------
DATA_ROOT=${DATA_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data"}
MODEL_ROOT=${MODEL_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/models"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/output"}

TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/rl/train_dapo17k.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_ROOT}/rl/aime24_30_boxed.parquet"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/Qwen3-8B"}
CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_ROOT}/${project_name}/${exp_name}"}

# ----------------------------------------------------------------------------
# Lengths (paper / upstream self-distillation-analysis settings).
# ----------------------------------------------------------------------------
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}       # 2048
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 20))}  # 20480
# Upstream README lists 18944, but this vLLM V1 stack must allow prompt+response
# tokens; otherwise long generation degenerates into token-0 ('!') loops / NaNs.
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}  # 22528
max_reprompt_len=${MAX_REPROMPT_LEN:-22528}

# ----------------------------------------------------------------------------
# Performance
# ----------------------------------------------------------------------------
sp_size=${SP_SIZE:-4}
gen_tp=${GEN_TP:-1}
use_dynamic_bsz=True
offload=True

# ----------------------------------------------------------------------------
# Batch / optim (paper defaults, then auto-aligned to allocated GPUs)
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

# Paper/upstream defaults are train_batch_size=256, rollout.n=8,
# ppo_mini_batch_size=128. With the default 32 GPUs and SP=4, DP=8,
# so rollout_batch=256*8=2048 is divisible by DP and mini-batch size.
# If NNODES=6 is used for 48 GPUs, 256 is auto-rounded to 288.
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
lr=1e-5

# ----------------------------------------------------------------------------
# SDPO self-distillation hyper-parameters (paper / upstream main setting)
# ----------------------------------------------------------------------------
loss_mode=sdpo
alpha=0.5
distillation_topk=100
full_logit_distillation=True
dont_reprompt_on_self_success=True
teacher_regularization=ema
teacher_update_rate=0.0
is_clip=2.0
rollout_is=${ROLLOUT_IS:-token}
rollout_is_threshold=2.0
if [[ "${rollout_is}" == "null" || -z "${rollout_is}" ]]; then
    calculate_rollout_log_probs_default=False
else
    calculate_rollout_log_probs_default=True
fi
calculate_rollout_log_probs=${CALCULATE_ROLLOUT_LOG_PROBS:-${calculate_rollout_log_probs_default}}
total_training_steps=${TOTAL_TRAINING_STEPS:-null}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-10}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-null}
resume_mode=${RESUME_MODE:-disable}
rollout_data_dir=${ROLLOUT_DATA_DIR:-null}

cd "${WORKING_DIR}"
echo "Working dir: ${WORKING_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Train: ${TRAIN_FILE}"
echo "Test:  ${TEST_FILE}"
echo "Ckpts: ${CKPTS_DIR}"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    python3 "${WORKING_DIR}/recipe/sdpo/preflight_math_sdpo.py" \
        --repo-root "${WORKING_DIR}" \
        --train-file "${TRAIN_FILE}" \
        --test-file "${TEST_FILE}" \
        --model-path "${MODEL_PATH}" \
        --reward-fn-path "core/utils/reward_score/sdpo_math_feedback_score.py" \
        --reward-fn-name "compute_score" \
        --nnodes "${NNODES}" \
        --gpus-per-node "${N_GPUS_PER_NODE}" \
        --sp-size "${sp_size}" \
        --train-batch-size "${train_prompt_bsz}" \
        --rollout-n "${n_resp_per_prompt}" \
        --ppo-mini-batch-size "${train_prompt_mini_bsz}" \
        --max-prompt-length "${max_prompt_length}" \
        --max-response-length "${max_response_length}" \
        --max-model-len "${max_model_len}" \
        --max-train-rows 50000
else
    echo "[SDPO preflight] skipped because SKIP_PREFLIGHT=1"
fi

if [[ "${CHECK_RAY_RESOURCES:-1}" != "0" ]]; then
    NNODES="${NNODES}" N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" python3 - <<'PY'
import os
import sys

try:
    import ray
except Exception as exc:
    print(f"[SDPO preflight] ERROR: cannot import ray for resource check: {exc}", file=sys.stderr)
    sys.exit(1)

need = int(os.environ["NNODES"]) * int(os.environ["N_GPUS_PER_NODE"])
try:
    ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
    resources = ray.cluster_resources()
    nodes = ray.nodes()
except Exception as exc:
    print(f"[SDPO preflight] ERROR: cannot connect to local Ray cluster with address='auto': {exc}", file=sys.stderr)
    sys.exit(1)

gpus = int(resources.get("GPU", 0))
alive_nodes = [node for node in nodes if node.get("Alive")]
print(f"[SDPO preflight] Ray resources: GPU={gpus}/{need}, alive_nodes={len(alive_nodes)}, resources={resources}")
for node in alive_nodes:
    node_gpus = node.get("Resources", {}).get("GPU", 0)
    print(f"[SDPO preflight] Ray node {node.get('NodeManagerAddress')} GPU={node_gpus}")
if gpus < need:
    print(
        f"[SDPO preflight] ERROR: Ray registered only {gpus} GPUs, but training asks for {need}. "
        "Restart Ray with --num-gpus on every pod or submit from the Ray head pod.",
        file=sys.stderr,
    )
    sys.exit(1)
PY
else
    echo "[SDPO preflight] Ray resource check skipped because CHECK_RAY_RESOURCES=0"
fi

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
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    actor_rollout_ref.actor.self_distillation.alpha=${alpha} \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
    actor_rollout_ref.actor.self_distillation.distillation_topk=${distillation_topk} \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=${full_logit_distillation} \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${dont_reprompt_on_self_success} \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=${teacher_regularization} \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=${teacher_update_rate} \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${max_reprompt_len} \
    actor_rollout_ref.actor.self_distillation.is_clip=${is_clip} \
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
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${max_model_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_rollout_log_probs} \
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
    reward_model.overlong_buffer.enable=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.save_freq=${save_freq} \
    trainer.max_actor_ckpt_to_keep=${max_actor_ckpt_to_keep} \
    trainer.total_epochs=500 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${CKPTS_DIR}/val/" \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.resume_mode=${resume_mode}
