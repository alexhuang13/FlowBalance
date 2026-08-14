# FlowSD Recipe

This directory contains the canonical implementation of **FlowSD: Trajectory-Balanced On-Policy Self-Distillation**.

## Message

> **FlowSD learns a normalized distribution over complete reasoning trajectories, not only a local update direction.**

Verifier rewards identify successful trajectories, a privileged teacher supplies dense trajectory information, the reference policy preserves pretrained support, and profiled trajectory balance determines the relative probability assigned to each sampled response. Sign gating reinforces teacher-supported positive-advantage responses and reverses teacher pressure on negative-advantage responses.

## Canonical naming

The implementation consistently uses:

| Interface | Name |
|---|---|
| Python package | `recipe.flowsd` |
| Main module | `recipe.flowsd.main_flowsd` |
| Config class | `FlowSDConfig` |
| Actor config | `FlowSDFSDPActorConfig` |
| Hydra config block | `actor_rollout_ref.actor.flowsd` |
| Loss mode | `flowsd` |
| Shell variables | `FLOWSD_*` |
| Metrics | `flowsd/*` |
| Launcher | `run_math_flowsd.sh` |

## Files

```text
recipe/flowsd/
├── README.md
├── config/flowsd_trainer.yaml
├── flowsd_config.py
├── flowsd_core_algos.py
├── flowsd_dp_actor.py
├── flowsd_fsdp_workers.py
├── flowsd_ray_trainer.py
├── main_flowsd.py
├── preflight_math_flowsd.py
├── privileged.md
├── run_math_flowsd.sh
├── run_math_flowsd_eta_sweep.sh
├── run_math_flowsd_betaT_resume_sweep.sh
├── submit_step180_val.sh
└── tests/test_flowsd_p0.py
```

## Objective

For response \(y\), FlowSD computes the privileged gain

\[
G_T(y)=\frac{1}{T^\rho}\sum_t
\mathrm{clip}\left(
\log\pi_T(y_t\mid s_t,c)-\log\pi_{\mathrm{ref}}(y_t\mid s_t),-B,B
\right),
\]

and the sign-gated energy

\[
E(y)=\eta_A A(y)+\beta_TG_T(y)\mathrm{sign}(A(y)).
\]

The target distribution is

\[
p^*(y\mid x,c)\propto
\pi_{\mathrm{ref}}(y\mid x)\exp(E(y)/\tau).
\]

The actor minimizes the squared profiled trajectory-balance residual. All target-side quantities are detached; gradients pass only through current-policy sequence log-probabilities.

Code-field mapping:

| Paper | Config | Default |
|---|---|---:|
| \(\beta_T\) | `flowsd.beta_q` | 1.0 |
| \(\eta_A\) | `flowsd.eta_R` | 15.0 |
| \(B\) | `flowsd.clip_B` | 4.0 |
| \(\rho\) | `flowsd.rho` | 1.0 |

## Privileged context

The student rollout sees only the original prompt. During training, the teacher can score the same sampled tokens under a privileged prompt containing a successful same-group response or configured external solution/feedback.

Default behavior:

```yaml
self_distillation:
  solution_source: group_only
  dont_reprompt_on_self_success: true
  include_environment_feedback: false

flowsd:
  reference_source: frozen_ref
  gate_no_context: drop
  min_group_valid: 2
```

The teacher does not generate a replacement response. It rescales already sampled student tokens under privileged context.

## Default large-scale setup

| Field | Value |
|---|---:|
| Backbones | Qwen3-4B / Qwen3-8B |
| Prompt batch | 256 |
| Responses per prompt | 8 |
| Trajectories per step | 2,048 |
| Actor mini-batch | 128 |
| Prompt length | 2,048 |
| Response length | 8,192 |
| Train context length | 10,240 |
| Learning rate | `1e-6` |
| Sequence parallel size | 8 |
| Rollout backend | vLLM |
| `FLOWSD_BETA_Q` | 1 |
| `FLOWSD_ETA_R` | 15 |
| `FLOWSD_CLIP_B` | 4 |
| `FLOWSD_RHO` | 1 |
| Actor KL | disabled |
| KL in reward | disabled |
| Overlong reward penalty | disabled |

## Preflight

```bash
python3 recipe/flowsd/preflight_math_flowsd.py \
  --repo-root "$PWD" \
  --train-file /path/to/train.parquet \
  --test-file /path/to/validation.parquet \
  --model-path /path/to/Qwen3-8B \
  --reward-fn-path core/utils/reward_score/sdpo_math_feedback_score.py \
  --reward-fn-name compute_score \
  --nnodes 4 \
  --gpus-per-node 8 \
  --sp-size 8 \
  --train-batch-size 256 \
  --rollout-n 8 \
  --ppo-mini-batch-size 128 \
  --max-prompt-length 2048 \
  --max-response-length 8192 \
  --max-model-len 10240
```

## Smoke test

Run on an existing Ray cluster:

```bash
PROJECT_NAME=verl-flowsd-smoke \
EXP_NAME=FlowSD-smoke \
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/train.parquet \
TEST_FILE=/path/to/validation.parquet \
OUTPUT_ROOT=/path/to/output \
NNODES=4 \
N_GPUS_PER_NODE=8 \
SP_SIZE=8 \
TOTAL_TRAINING_STEPS=1 \
TEST_FREQ=1 \
SAVE_FREQ=100000 \
STEP180_VAL_ENABLE=0 \
FLOWSD_BETA_Q=1 \
FLOWSD_ETA_R=15 \
LR=1e-6 \
bash recipe/flowsd/run_math_flowsd.sh
```

## Step-180 run

```bash
PROJECT_NAME=verl-flowsd \
EXP_NAME=FlowSD-Qwen3-8B-math-step180 \
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/train_dapo17k.parquet \
TEST_FILE=/path/to/aime24_30_boxed.parquet \
OUTPUT_ROOT=/path/to/output \
NNODES=4 \
N_GPUS_PER_NODE=8 \
SP_SIZE=8 \
TRAIN_PROMPT_BSZ=256 \
N_RESP_PER_PROMPT=8 \
PPO_MINI_BATCH_SIZE=128 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=8192 \
MAX_MODEL_LEN=10240 \
MAX_REPROMPT_LEN=10240 \
TOTAL_TRAINING_STEPS=180 \
SAVE_FREQ=10 \
FLOWSD_BETA_Q=1 \
FLOWSD_ETA_R=15 \
FLOWSD_CLIP_B=4 \
FLOWSD_RHO=1 \
LR=1e-6 \
STEP180_VAL_ENABLE=1 \
bash recipe/flowsd/run_math_flowsd.sh
```

For `launch/start.sh`:

```bash
TRAIN_RECIPE_SUBDIR=flowsd \
RUN_SCRIPT=run_math_flowsd.sh \
bash launch/start.sh
```

## Evaluation

```bash
RUN_DIR=/path/to/experiment \
EXPERIMENT_NAME=FlowSD-Qwen3-8B-math-step180 \
VAL_STEP=180 \
VAL_SEEDS="0 1 2 3 4" \
VAL_KEEP_MERGED_MODEL=1 \
bash recipe/flowsd/submit_step180_val.sh
```

## Ablations

Verifier coefficient:

```bash
ETA_SWEEP_VALUES="5 10 15" \
bash recipe/flowsd/run_math_flowsd_eta_sweep.sh
```

Teacher coefficient:

```bash
bash recipe/flowsd/run_math_flowsd_betaT_resume_sweep.sh
```

## Monitoring

Target construction:

```text
flowsd/G_q_raw_mean
flowsd/G_q_mean
flowsd/advantage_sign_mean
flowsd/beta_q_G_q_abs_mean
flowsd/eta_R_R_abs_mean
flowsd/logRtildeF_mean
flowsd/logZ_hat_mean
flowsd/valid_sample_fraction
flowsd/degenerate_group_fraction
flowsd/delta_clip_frac
```

Actor optimization:

```text
flowsd/tb_loss
flowsd/residual_abs_mean
flowsd/seq_logp_student_mean
flowsd/target_actor_mean
flowsd/nESS
actor/grad_norm
actor/entropy
```

A near-zero valid-sample fraction usually indicates insufficient privileged demonstrations. A high degenerate-group fraction means too few valid samples share a target-normalization group. A delta-clip fraction near one indicates that most teacher-reference differences saturate at `clip_B`.
