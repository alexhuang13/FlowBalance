#!/usr/bin/env bash
# Full Anti-SDPO math run script.  This lives under recipe/antisd and does not
# modify existing SDPO/GRPO scripts.
set -xeuo pipefail

project_name=${PROJECT_NAME:-verl-antisd}
exp_name=${EXP_NAME:-AntiSDPO-Qwen3-8B-math-dapo17k}

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
WORKING_DIR=${WORKING_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/runtime_env.yaml"}
NNODES=${NNODES:-4}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}

DATA_ROOT=${DATA_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/data"}
MODEL_ROOT=${MODEL_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/models"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"/apdcephfs_gy4/share_303378103/user/audenhuang/output"}

TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/rl/train_dapo17k.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_ROOT}/rl/aime24_30_boxed.parquet"}
MODEL_PATH=${MODEL_PATH:-"${MODEL_ROOT}/Qwen3-8B"}
CKPTS_DIR=${CKPTS_DIR:-"${OUTPUT_ROOT}/${project_name}/${exp_name}"}

max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 2))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 8))}
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
max_reprompt_len=${MAX_REPROMPT_LEN:-${max_model_len}}

sp_size=${SP_SIZE:-8}
gen_tp=${GEN_TP:-1}
use_dynamic_bsz=True
offload=True

n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-128}
desired_train_prompt_bsz=${TRAIN_PROMPT_BSZ:-256}
lr=${LR:-1e-6}

# AntiSD defaults from AntiSD short launcher.
prm_renyi_sign=${PRM_RENYI_SIGN:--1.0}          # -1 = Anti-SDPO, +1 = SD direction
prm_forward_mode=${PRM_FORWARD_MODE:-jsd_unbiased}
prm_construction=${PRM_CONSTRUCTION:-reverse}
ca_mode=${CA_MODE:-additive}
ca_lambda_mode=${CA_LAMBDA_MODE:-teacher_perp}
ca_lambda=${CA_LAMBDA:-0.1}
lambda_min=${LAMBDA_MIN:--0.02}
lambda_max=${LAMBDA_MAX:-0.5}
warmup_steps=${WARMUP_STEPS:-5}
prm_clip=${PRM_CLIP:-3.0}
log_clip=${LOG_CLIP:-5.0}
k_sigma=${K_SIGMA:-2.0}
sigma_ref_fixed=${SIGMA_REF_FIXED:-0.0}
len_mask=${LEN_MASK:-12000}
tp_target=${TP_TARGET:-0.0}
tp_target_ratio=${TP_TARGET_RATIO:-0.0}
reactivate_ratio=${REACTIVATE_RATIO:-0.0}

solution_mode=${SOLUTION_MODE:-normal}
max_solution_tokens=${MAX_SOLUTION_TOKENS:-3072}

gcd() { local a=$1 b=$2 t; while (( b != 0 )); do t=$((a % b)); a=$b; b=$t; done; echo "$a"; }
lcm() { local a=$1 b=$2 g; g=$(gcd "$a" "$b"); echo $((a / g * b)); }
round_up_to_multiple() { local x=$1 m=$2; echo $((((x + m - 1) / m) * m)); }

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
echo "Auto batch sizing: total_gpus=${total_gpus}, SP=${sp_size}, DP=${dp_size}, train_prompt_bsz=${train_prompt_bsz}, rollout_batch=${rollout_batch_size}"

rollout_is=${ROLLOUT_IS:-token}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
if [[ "${rollout_is}" == "null" || -z "${rollout_is}" ]]; then
    calculate_rollout_log_probs_default=False
else
    calculate_rollout_log_probs_default=True
fi
calculate_rollout_log_probs=${CALCULATE_ROLLOUT_LOG_PROBS:-${calculate_rollout_log_probs_default}}
total_training_steps=${TOTAL_TRAINING_STEPS:-null}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-10}
resume_mode=${RESUME_MODE:-disable}
rollout_data_dir=${ROLLOUT_DATA_DIR:-null}

cd "${WORKING_DIR}"
echo "Working dir: ${WORKING_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Train: ${TRAIN_FILE}"
echo "Test:  ${TEST_FILE}"
echo "Ckpts: ${CKPTS_DIR}"
echo "AntiSDPO: loss_mode=grpo_ca prm_forward_mode=${prm_forward_mode} sign=${prm_renyi_sign} ca_lambda_mode=${ca_lambda_mode}"

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
fi

if [[ "${CHECK_RAY_RESOURCES:-1}" != "0" ]]; then
    NNODES="${NNODES}" N_GPUS_PER_NODE="${N_GPUS_PER_NODE}" python3 - <<'PY'
import os, sys
import ray
need = int(os.environ["NNODES"]) * int(os.environ["N_GPUS_PER_NODE"])
ray.init(address="auto", ignore_reinit_error=True, logging_level="ERROR")
gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"[AntiSDPO preflight] Ray GPU={gpus}/{need}")
if gpus < need:
    sys.exit(f"Ray registered only {gpus} GPUs, need {need}")
PY
fi

ray job submit --address="${RAY_ADDRESS}" --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.antisd.main_antisd \
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
    actor_rollout_ref.actor.policy_loss.loss_mode=grpo_ca \
    actor_rollout_ref.actor.self_distillation.alpha=0.5 \
    actor_rollout_ref.actor.self_distillation.full_logit_distillation=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=null \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=ema \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=1.0 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${max_reprompt_len} \
    actor_rollout_ref.actor.self_distillation.max_solution_tokens=${max_solution_tokens} \
    actor_rollout_ref.actor.self_distillation.solution_selection=random \
    actor_rollout_ref.actor.self_distillation.truncate_solution_at_correct_answer=True \
    actor_rollout_ref.actor.self_distillation.solution_source=group_only \
    actor_rollout_ref.actor.self_distillation.solution_content=full \
    actor_rollout_ref.actor.self_distillation.solution_mode=${solution_mode} \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=True \
    actor_rollout_ref.actor.self_distillation.provide_ground_truth_in_feedback=False \
    actor_rollout_ref.actor.ccir.enabled=False \
    actor_rollout_ref.actor.ccir.ca_mode=${ca_mode} \
    actor_rollout_ref.actor.ccir.ca_lambda=${ca_lambda} \
    actor_rollout_ref.actor.ccir.ca_lambda_mode=${ca_lambda_mode} \
    actor_rollout_ref.actor.ccir.ca_lambda_perp_target=${tp_target} \
    actor_rollout_ref.actor.ccir.ca_lambda_perp_target_ratio=${tp_target_ratio} \
    actor_rollout_ref.actor.ccir.ca_lambda_perp_reactivate_ratio=${reactivate_ratio} \
    actor_rollout_ref.actor.ccir.ca_lambda_tppl_scope=per_seq \
    actor_rollout_ref.actor.ccir.ca_lambda_min=${lambda_min} \
    actor_rollout_ref.actor.ccir.ca_lambda_max=${lambda_max} \
    actor_rollout_ref.actor.ccir.ca_lambda_perp_mask=3.0 \
    actor_rollout_ref.actor.ccir.ca_lambda_warmup_steps=${warmup_steps} \
    actor_rollout_ref.actor.ccir.orm_weight=1.0 \
    actor_rollout_ref.actor.ccir.prm_weight=0.0 \
    actor_rollout_ref.actor.ccir.prm_normalize=True \
    actor_rollout_ref.actor.ccir.prm_normalize_mode=sequence \
    actor_rollout_ref.actor.ccir.prm_entropy_neutral=none \
    actor_rollout_ref.actor.ccir.prm_anchor_to_orm=False \
    actor_rollout_ref.actor.ccir.prm_seq_demean=False \
    actor_rollout_ref.actor.ccir.prm_clip=${prm_clip} \
    actor_rollout_ref.actor.ccir.prm_construction=${prm_construction} \
    actor_rollout_ref.actor.ccir.prm_gamma=1.0 \
    actor_rollout_ref.actor.ccir.prm_forward_mode=${prm_forward_mode} \
    actor_rollout_ref.actor.ccir.prm_renyi_sign=${prm_renyi_sign} \
    actor_rollout_ref.actor.ccir.prm_renyi_virtual_alpha=1.0 \
    actor_rollout_ref.actor.ccir.prm_forward_log_clip=${log_clip} \
    actor_rollout_ref.actor.ccir.prm_u_clip_mode=adaptive \
    actor_rollout_ref.actor.ccir.prm_u_clip_k_sigma=${k_sigma} \
    actor_rollout_ref.actor.ccir.prm_u_clip_sigma_ref_fixed=${sigma_ref_fixed} \
    actor_rollout_ref.actor.ccir.prm_length_mask_threshold=${len_mask} \
    actor_rollout_ref.actor.ccir.si_mode=none \
    actor_rollout_ref.actor.ccir.si_reference=bare \
    actor_rollout_ref.actor.ccir.maxent_coeff=none \
    actor_rollout_ref.actor.ccir.maxent_alpha=0.0 \
    actor_rollout_ref.actor.ccir.ccir_cross_problem=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
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
    trainer.total_epochs=500 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${CKPTS_DIR}/val/" \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.resume_mode=${resume_mode}
