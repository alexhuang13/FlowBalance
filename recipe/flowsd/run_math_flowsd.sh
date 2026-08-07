#!/usr/bin/env bash
# FlowSD math run script. Defaults reproduce the Run28 training setup;
# FLOWSD_ETA_R is the intended ablation knob.
set -xeuo pipefail

project_name=${PROJECT_NAME:-verl-flowsd}
exp_name=${EXP_NAME:-FlowSD-Qwen3-8B-math-dapo17k-betaq1-grpo-signadv-frozenref-etaR15-lr1e6-run28-repro}

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

beta_q=${FLOWSD_BETA_Q:-1}
eta_R=${FLOWSD_ETA_R:-15}
beta_q_start=${FLOWSD_BETA_Q_START:-null}
beta_q_end=${FLOWSD_BETA_Q_END:-null}
eta_R_start=${FLOWSD_ETA_R_START:-null}
eta_R_end=${FLOWSD_ETA_R_END:-null}
flow_schedule_steps=${FLOWSD_SCHEDULE_STEPS:-0}
flow_gap_clip_low=${FLOWSD_FLOW_GAP_CLIP_LOW:-5.0}
flow_gap_clip_high=${FLOWSD_FLOW_GAP_CLIP_HIGH:-5.0}
importance_ratio_cap=${FLOWSD_IMPORTANCE_RATIO_CAP:-1.2}
residual_length_scale=${FLOWSD_RESIDUAL_LENGTH_SCALE:-0.0}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-0}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-0.0}
overlong_buffer_log=${OVERLONG_BUFFER_LOG:-False}
clip_B=${FLOWSD_CLIP_B:-4.0}
rho=${FLOWSD_RHO:-1.0}
lambda_kl=${FLOWSD_LAMBDA_KL:-0.0}
use_flowsd_kl=${FLOWSD_USE_KL:-False}
gate_no_context=${FLOWSD_GATE_NO_CONTEXT:-drop}
min_group_valid=${FLOWSD_MIN_GROUP_VALID:-2}
w_clip=${FLOWSD_W_CLIP:-null}
w_min=${FLOWSD_W_MIN:-0.0}
w_max=${FLOWSD_W_MAX:-10.0}

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

total_training_steps=${TOTAL_TRAINING_STEPS:-null}
test_freq=${TEST_FREQ:-1}
save_freq=${SAVE_FREQ:-10}
resume_mode=${RESUME_MODE:-disable}
resume_from_path=${RESUME_FROM_PATH:-null}
rollout_data_dir=${ROLLOUT_DATA_DIR:-null}
log_val_generations=${LOG_VAL_GENERATIONS:-4}
nccl_timeout=${NCCL_TIMEOUT:-600}

# Full external validation of the saved step-180 policy. This is intentionally
# separate from the lightweight in-training validation: it evaluates seven
# benchmarks at n=1 plus AIME24/25/26 at n=16 over five random seeds.
step180_val_enable=${STEP180_VAL_ENABLE:-1}
step180_val_step=${STEP180_VAL_STEP:-180}
step180_val_output_dir=${STEP180_VAL_OUTPUT_DIR:-${CKPTS_DIR}/val/step_${step180_val_step}_math_benchmarks_5seeds}

cd "${WORKING_DIR}"
echo "Working dir: ${WORKING_DIR}"
echo "Model: ${MODEL_PATH}"
echo "Train: ${TRAIN_FILE}"
echo "Test:  ${TEST_FILE}"
echo "Ckpts: ${CKPTS_DIR}"
echo "FlowSD Run28 setup: beta_q=${beta_q}, eta_R=${eta_R}, lr=${lr}, teacher=frozen_ref, beta_q_start=${beta_q_start}, beta_q_end=${beta_q_end}, eta_R_start=${eta_R_start}, eta_R_end=${eta_R_end}, schedule_steps=${flow_schedule_steps}, flow_gap_clip=[-${flow_gap_clip_low},${flow_gap_clip_high}], importance_ratio_cap=${importance_ratio_cap}, residual_length_scale=${residual_length_scale}, clip_B=${clip_B}, rho=${rho}, test_freq=${test_freq}, log_val_generations=${log_val_generations}, nccl_timeout=${nccl_timeout}s"
echo "Length penalty: enable=${enable_overlong_buffer}, buffer_len=${overlong_buffer_len}, penalty_factor=${overlong_penalty_factor}, max_response_length=${max_response_length}"
echo "Step validation: enable=${step180_val_enable}, step=${step180_val_step}, output=${step180_val_output_dir}"

if [[ "${step180_val_enable}" == "1" ]]; then
    if (( save_freq <= 0 || step180_val_step % save_freq != 0 )); then
        echo "ERROR: step-${step180_val_step} validation requires SAVE_FREQ to be a positive divisor of ${step180_val_step}; got SAVE_FREQ=${save_freq}"
        exit 1
    fi
    if [[ "${total_training_steps}" == "null" ]]; then
        echo "WARNING: STEP180 validation was requested but TOTAL_TRAINING_STEPS is null."
        echo "WARNING: the watcher cannot use GPUs until training releases them; set TOTAL_TRAINING_STEPS=${step180_val_step} for an in-task step-${step180_val_step} result."
    elif (( total_training_steps < step180_val_step )); then
        echo "ERROR: TOTAL_TRAINING_STEPS=${total_training_steps} is smaller than STEP180_VAL_STEP=${step180_val_step}"
        exit 1
    elif (( total_training_steps > step180_val_step )); then
        echo "WARNING: training continues past step ${step180_val_step}; because training occupies all GPUs, full benchmark validation may wait until training exits."
        echo "WARNING: set TOTAL_TRAINING_STEPS=${step180_val_step} when the step-${step180_val_step} benchmark is the desired terminal result."
    fi
fi

if [[ "${enable_overlong_buffer,,}" == "true" ]]; then
    if (( overlong_buffer_len <= 0 || overlong_buffer_len > max_response_length )); then
        echo "ERROR: OVERLONG_BUFFER_LEN=${overlong_buffer_len} must be in [1, ${max_response_length}]"
        exit 1
    fi
fi

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    python3 "${WORKING_DIR}/recipe/flowsd/preflight_math_flowsd.py" \
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

ray job submit --address="${RAY_ADDRESS}" --no-wait --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.flowsd.main_flowsd \
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
    actor_rollout_ref.nccl_timeout=${nccl_timeout} \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=null \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.policy_loss.loss_mode=flowsd \
    actor_rollout_ref.actor.self_distillation.success_reward_threshold=0.5 \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=${max_reprompt_len} \
    actor_rollout_ref.actor.flowsd.beta_q=${beta_q} \
    actor_rollout_ref.actor.flowsd.eta_R=${eta_R} \
    actor_rollout_ref.actor.flowsd.reward_type=grpo_advantage \
    actor_rollout_ref.actor.flowsd.beta_q_start=${beta_q_start} \
    actor_rollout_ref.actor.flowsd.beta_q_end=${beta_q_end} \
    actor_rollout_ref.actor.flowsd.eta_R_start=${eta_R_start} \
    actor_rollout_ref.actor.flowsd.eta_R_end=${eta_R_end} \
    actor_rollout_ref.actor.flowsd.schedule_steps=${flow_schedule_steps} \
    actor_rollout_ref.actor.flowsd.flow_gap_clip_low=${flow_gap_clip_low} \
    actor_rollout_ref.actor.flowsd.flow_gap_clip_high=${flow_gap_clip_high} \
    actor_rollout_ref.actor.flowsd.importance_ratio_cap=${importance_ratio_cap} \
    actor_rollout_ref.actor.flowsd.residual_length_scale=${residual_length_scale} \
    actor_rollout_ref.actor.flowsd.clip_B=${clip_B} \
    actor_rollout_ref.actor.flowsd.rho=${rho} \
    actor_rollout_ref.actor.flowsd.lambda_kl=${lambda_kl} \
    actor_rollout_ref.actor.flowsd.use_flowsd_kl=${use_flowsd_kl} \
    actor_rollout_ref.actor.flowsd.gate_no_context=${gate_no_context} \
    actor_rollout_ref.actor.flowsd.min_group_valid=${min_group_valid} \
    actor_rollout_ref.actor.flowsd.w_clip=${w_clip} \
    actor_rollout_ref.actor.flowsd.w_min=${w_min} \
    actor_rollout_ref.actor.flowsd.w_max=${w_max} \
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
    ++reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    ++reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    ++reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    ++reward_model.reward_kwargs.overlong_buffer_cfg.log=${overlong_buffer_log} \
    ++reward_model.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=${test_freq} \
    trainer.log_val_generations=${log_val_generations} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=500 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.validation_data_dir="${CKPTS_DIR}/val/" \
    trainer.rollout_data_dir=${rollout_data_dir} \
    trainer.resume_mode=${resume_mode} \
    trainer.resume_from_path=${resume_from_path}

if [[ "${step180_val_enable}" == "1" ]]; then
    echo "Submitting FlowSD step-${step180_val_step} benchmark validation watcher"
    RUN_DIR="${CKPTS_DIR}" \
    EXPERIMENT_NAME="${exp_name}" \
    VAL_STEP="${step180_val_step}" \
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
    bash "${WORKING_DIR}/recipe/flowsd/submit_step180_val.sh"
fi
