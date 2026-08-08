# OPSD

This directory integrates **On-Policy Self-Distillation (OPSD)** into the shared
FlowSD `verl + Ray + FSDP + vLLM` stack. A snapshot of the original
TRL/Accelerate implementation is retained under `provenance/opsd_standalone/` for
traceability.

## Objective

The student generates an on-policy response from the original problem context.
A teacher evaluates the same response under a privileged context containing a
reference solution or a successful rollout. The default objective matches a
fixed-teacher token-level forward KL:

```text
alpha = 0.0
full_logit_distillation = true
teacher_update_rate = 0.0
token_loss_clip = 0.06
```

To reduce memory use, the implementation keeps the teacher's top-100
probabilities and an aggregate tail bucket by default. `OPSD_ALPHA=0.5` selects a
generalized JSD objective and `OPSD_ALPHA=1.0` selects reverse KL.

## Default comparable setting

- Qwen3-8B on 4 nodes with 8 GPUs per node.
- DAPO-Math-17K training data and AIME 2024 validation.
- Prompt/response/model lengths: `2048/8192/10240`.
- 256 prompts with 8 responses per prompt.
- PPO mini-batch size 128, sequence parallel size 8, rollout TP 1.
- Rollout sampling: temperature 1.0, top-p 1.0, top-k -1.
- Validation sampling: temperature 0.6, top-p 0.95.
- Learning rate `1e-6`, FSDP parameter/optimizer offload, vLLM memory 0.55.

OPSD does not use the GRPO advantage in its loss, but it retains the same
rollout, reward, and data pipeline for controlled comparisons.

## Privileged solution source

The default `OPSD_SOLUTION_SOURCE=external_first` first looks for a full worked
solution in the dataset. If the dataset contains only a final answer, it falls
back to a successful response from the same rollout group. Use
`OPSD_SOLUTION_SOURCE=group_only` to require same-group demonstrations.

## Run

Validate paths, topology, typed configuration, and the generated Ray command:

```bash
DRY_RUN=1 bash recipe/opsd/run_math_opsd.sh
```

Submit training:

```bash
bash recipe/opsd/run_math_opsd.sh
```

Common overrides:

```bash
OPSD_ALPHA=0 \
OPSD_DISTILLATION_TOPK=100 \
OPSD_TOKEN_LOSS_CLIP=0.06 \
OPSD_STUDENT_THINKING=true \
OPSD_TEACHER_THINKING=true \
bash recipe/opsd/run_math_opsd.sh
```

The launcher uses strict shell error handling, validated booleans, array-based
Hydra overrides, a typed self-distillation config, and preflight checks before
connecting to Ray.
