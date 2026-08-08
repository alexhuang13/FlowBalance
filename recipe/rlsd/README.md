# RLSD

`recipe/rlsd` is the single canonical RLSD implementation in this repository. It
runs on the same vendored `verl` stack, Ray launcher, data format, reward
manager, FSDP worker and vLLM rollout infrastructure as SDPO, AntiSD and
FlowSD.

The former vendored RLSD runtime and the duplicate `recipe/rlsd_native` package
were removed. This avoids maintaining two incompatible verl APIs.

## Algorithm

RLSD first constructs a privileged teacher prompt from a successful response in
the same rollout group. For token `t`, it compares the teacher-conditioned and
student log probabilities and forms

```text
w_t = exp(sign(A_t) * (log p_teacher(y_t | x, c) - log p_student(y_t | x)))
```

The weight is clipped to `[1-clip_range, 1+clip_range]` and mixed with the GRPO
advantage:

```text
A_RLSD,t = A_GRPO,t * ((1-lambda) + lambda * clip(w_t))
```

Samples without an available same-group successful demonstration remain plain
GRPO. `lambda` supports linear warmup and decay. The teacher is represented by
the frozen/synchronized reference module and can be synchronized at a configured
interval.

## Entry points

- `main_rlsd.py`: Hydra/Ray entry point.
- `rlsd_config.py`: RLSD actor and reweight configuration.
- `rlsd_core_algos.py`: pure tensor reweighting and lambda schedule.
- `rlsd_dp_actor.py`: RLSD actor update.
- `rlsd_ray_trainer.py`: privileged prompt construction and metrics.
- `rlsd_fsdp_workers.py`: FSDP/vLLM worker wiring.
- `run_math_rlsd.sh`: canonical Qwen3-8B DAPO-Math launcher.

## Run

From the repository root, on an already-created Ray cluster:

```bash
LR=1e-6 \
RLSD_LAMBDA=0.5 \
RLSD_REWEIGHT_CLIP_RANGE=0.2 \
RLSD_TEACHER_SYNC_INTERVAL=20 \
NNODES=4 N_GPUS_PER_NODE=8 \
bash recipe/rlsd/run_math_rlsd.sh
```

For Taiji, use the common launcher:

```bash
TRAIN_RECIPE_SUBDIR=rlsd RUN_SCRIPT=run_math_rlsd.sh bash launch/start.sh
```

Useful smoke-test overrides:

```bash
TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 SAVE_FREQ=1
```
