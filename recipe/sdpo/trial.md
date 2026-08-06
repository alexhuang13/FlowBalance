# SDPO / AntiSDPO / FlowOPSD Trial Log

Last updated: 2026-07-20 (Asia/Beijing)

This file records the experiments that can be reconstructed from the Taiji task history, retained launch configurations, git history, and the current code tree. It is intentionally explicit about uncertainty: many early jobs reused the same experiment name while the shared Ceph code changed between launches, so their exact code-level differences cannot always be recovered from the task configuration alone.

## Shared setup used by the later math experiments

- Base model: `Qwen3-8B`
- Training data: `data/rl/train_dapo17k.parquet`
- Validation data: `data/rl/aime24_30_boxed.parquet`
- Later FlowOPSD runs: 4 nodes × 8 H20, sequence parallel size 8
- Maximum response length: 8192
- Maximum model / reprompt length: 10240
- FlowOPSD reward target in the recent runs:
  - `beta_q` controls the privileged-context gain `G_q`.
  - `eta_R` controls verifier reward `R`.
  - `G_q = sum_t clamp(log q(y_t|x,c) - log q(y_t|x), -clip_B, clip_B) / L^rho`.

## Phase A: platform and original SDPO reproduction

### Early platform tests

Taiji tasks:

- `OPD_experiment_test`
- `OPD_experiment_test_2`
- `OPD_experiment_test_3`
- `OPD_experiment_test_4`

These were bring-up jobs before the retained SDPO recipe stabilized. The task history confirms that they ran, but their detailed command-line hyperparameters are no longer present in the current local configuration.

### Original reproduction series

Taiji tasks:

- `OPD_experiment_reproduction_sdpo`
- `OPD_experiment_reproduction_sdpo_1` through `OPD_experiment_reproduction_sdpo_18`

Purpose: reproduce the original SDPO/OPD training path and progressively resolve environment, data, distributed-training, and launch issues. Early checkpoints in git reflect this progression (including early debug and runnable checkpoints). Exact per-run differences are not fully recoverable from the retained simple configs.

### Local SDPO reproduction series

Taiji tasks observed in history:

- `auden_opd_reproduce_2` through `_5`
- `auden_opd_reproduce_7` through `_16`

Recoverable milestones:

- Runs 2–9: repeated launch/debug iterations of `run_math_sdpo.sh`.
- Run 10: switched to the retained DAPO-17K training set and AIME24 boxed validation set.
- Runs 11–12: model/data path corrections, including a temporary model path on another Ceph namespace.
- Runs 13–16: converged to the current local `Qwen3-8B`, DAPO-17K, and AIME24 paths.
- Run 14 explicitly disabled rollout importance sampling with `ROLLOUT_IS=null`.
- The retained SDPO launch uses full-logit/top-k self-distillation and labels the teacher mode `ema`, but its default `teacher_update_rate=0.0`; therefore that configuration is effectively a frozen teacher rather than a moving EMA.

## Phase B: AntiSDPO control

Taiji task:

- `auden_antisdpo_reproduce_16`

Configuration:

- Script: `run_math_antisd.sh`
- Experiment: `AntiSDPO-Qwen3-8B-math-dapo17k`
- Same Qwen3-8B / DAPO-17K / AIME24 data setup as the later SDPO and FlowOPSD runs.

Purpose: a sign/direction control for the self-distillation objective. The task is complete; this file does not claim a quality conclusion because no final metric summary is retained alongside the task config.

## Phase C: FlowOPSD implementation and bring-up

All tasks below used the FlowOPSD recipe unless noted otherwise.

| Run | Created | Recoverable configuration / purpose |
|---|---:|---|
| 1 | 2026-07-08 17:55 | First FlowOPSD Taiji bring-up; launch path used `../flowopsd/run_math_flowopsd.sh`. |
| 2 | 2026-07-09 11:13 | Re-launch after initial integration fixes. |
| 3 | 2026-07-09 12:01 | Corrected launch-script routing to `run_math_flowopsd.sh`. |
| 4 | 2026-07-09 13:03 | Continued distributed/runtime bring-up. |
| 5 | 2026-07-13 08:38 | Continued FlowOPSD implementation validation. |
| 6 | 2026-07-15 01:39 | Continued implementation/debug iteration. |
| 7 | 2026-07-15 10:29 | Long NCCL timeout (`7200s`) for distributed debugging. |
| 8 | 2026-07-15 10:46 | `eta_R=15`, NCCL timeout `7200s`. |
| 9 | 2026-07-16 10:35 | `eta_R=15`, continued runtime/debug fixes. |
| 10 | 2026-07-17 10:28 | `eta_R=15`, continued runtime/debug fixes. |
| 11 | 2026-07-17 14:56 | `eta_R=15`, final job in this high reward-weight group. |

The exact default `beta_q` for the earliest jobs depends on the shared code at their launch time. The Taiji configs did not explicitly export it, so it should not be reconstructed from today's default.

### `G_q` switching control

Taiji task:

- `auden_flowopsd_Gqswitching_1`, created 2026-07-17 17:20

Configuration/name:

- `FlowOPSD-Qwen3-8B-beta1-eta1-gqpos-beta0`
- `beta_q=1`, `eta_R=1`
- Custom script: `run_math_flowopsd_gq_pos_beta0.sh`

Purpose: a sign-dependent `G_q` control in which the positive-`G_q` contribution was switched off. The custom script is no longer present in the current tree, so the task name/config is the authoritative retained description.

## Phase D: FlowOPSD objective and ablations

| Run | Created | Main setting | Notes |
|---|---:|---|---|
| 12 | 2026-07-18 15:51 | `beta_q=1`, `eta_R=1` | Moved from the earlier `eta_R=15` scale to balanced weights. |
| 13 | 2026-07-18 21:31 | `beta_q=1`, `eta_R=1` | NCCL timeout reduced to `1800s`; target-regression/diagnostic work was being integrated. |
| 14 | 2026-07-19 12:11 | `beta_q=1`, `eta_R=1` | Continued balanced FlowOPSD run. |
| 15 | 2026-07-19 12:25 | `beta_q=0`, `eta_R=1` | Removed the privileged-teacher gain; reward-only ablation. |
| 16 | 2026-07-19 13:20 | `beta_q=0`, `eta_R=1` | Reward-only ablation rerun after launch/runtime fixes. |
| 17 | 2026-07-19 13:39 | `beta_q=0`, `eta_R=1` | Named control: `...-betaq0-run17`. |
| 18 | 2026-07-19 13:40 | `beta_q=1`, `eta_R=1` | Paired treatment: `...-betaq1-run18`. |
| 19 | 2026-07-19 15:05 | `beta_q=0`, `eta_R=1` | Paired control with NCCL timeout `600s`; manually stopped on 2026-07-20. |
| 20 | 2026-07-19 15:05 | `beta_q=1`, `eta_R=1` | Paired treatment with NCCL timeout `600s`; manually stopped on 2026-07-20 to release resources for run 22. |
| 21 | 2026-07-20 12:36 | `beta_q=1`, `eta_R=1`, `G_q=min(G_q,0)` | Negative-only `G_q` experiment, named `...-betaq1-gqmin0-run21`; running when this log was written. |

For runs 1–21, the FlowOPSD “teacher” was the co-located reference FSDP module evaluated under a privileged reprompt. Its forward pass and target construction used `torch.no_grad()` / detached targets. Before the EMA change below, FlowOPSD did not call the EMA update path, so the module remained frozen even though the inherited config field said `teacher_regularization=ema`.

## Current prepared trial: real EMA teacher, unclipped-sign `G_q`

Prepared configuration:

- Task: `auden_flowopsd_reproduce_22`
- Experiment: `FlowOPSD-Qwen3-8B-math-dapo17k-betaq1-ema0999-run22`
- `beta_q=1`
- `eta_R=1`
- Removed the run-21 sign truncation; `G_q` may now be positive or negative.
- Teacher mode: EMA with update rate `0.001`, equivalently EMA decay `0.999`:

  `teacher <- 0.999 * teacher + 0.001 * student`

- Teacher inference remains stop-gradient (`torch.no_grad()`). Stop-gradient and EMA are compatible: EMA specifies how weights evolve between optimizer steps, while stop-gradient specifies that teacher outputs do not receive backpropagation gradients.
- In the current memory-efficient wiring, `teacher_module` and `ref_module_fsdp` are the same FSDP module. Consequently, both privileged-context and no-context teacher evaluations follow the EMA weights. The difference producing `G_q` is the input context, not a difference between two model checkpoints.

Run 22 was submitted on 2026-07-20 15:06 (Asia/Beijing), instance `8b1d89d69f6a6326019f7e5904da2308`. It initially waited for resources because the 64-H20 high-priority quota was full. After run 20 was stopped, run 22 entered `TRAINING_RUNNING` at 2026-07-20 15:22:59. The Ray training job failed after global step 2: the main trainer batch already contained `ref_log_prob`, while FlowOPSD recomputed an EMA no-context score under the same key after the teacher had changed, causing `DataProto.union()` to assert that the duplicate tensors differed.

## Interpretation cautions

1. Task creation time and exported environment variables are reliable; exact early code revisions are not, because jobs load shared Ceph code.
2. A config value `teacher_regularization=ema` is not sufficient to prove an EMA was active. A nonzero update rate and an actual call to the update function are both required.
3. “Frozen” and “stop-gradient” are different properties:
   - frozen: teacher weights do not change over training;
   - stop-gradient: teacher forward outputs do not receive gradients;
   - EMA teacher: weights change through an explicit moving-average update, normally still under stop-gradient.
4. This log records configurations and implementation milestones. It does not invent conclusions for jobs whose final validation curves/checkpoints have not been summarized here.


## Run 23: EMA duplicate-key fix

Prepared on 2026-07-20 as `auden_flowopsd_reproduce_23` / `FlowOPSD-Qwen3-8B-math-dapo17k-betaq1-ema0999-run23`. The EMA teacher's no-context score is now named `teacher_ref_log_prob`; the main trainer's existing `ref_log_prob` remains separate. FlowOPSD computes `G_q` from `teacher_log_prob - teacher_ref_log_prob`, so EMA updates no longer create a duplicate-key collision during `DataProto.union()`. All run-22 hyperparameters are otherwise unchanged.
