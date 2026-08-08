# SDPO

Self-Distillation Policy Optimization (SDPO) is implemented as an extension of
the shared FlowSD PPO stack. It uses a successful same-group response or another
configured solution source to construct a privileged teacher prompt, then trains
the student with token-level distribution matching.

## Components

- `main_sdpo.py`: Hydra and Ray entry point.
- `sdpo_config.py`: typed actor and self-distillation configuration.
- `sdpo_core_algos.py`: forward-KL, reverse-KL, and generalized-JSD helpers.
- `sdpo_dp_actor.py`: actor update and teacher handling.
- `sdpo_fsdp_workers.py`: FSDP actor/rollout/reference wiring.
- `sdpo_ray_trainer.py`: privileged prompt construction and batch injection.
- `config/sdpo_trainer.yaml`: recipe defaults.
- `run_math_sdpo.sh`: canonical Qwen math launcher.

## Run

```bash
NNODES=4 \
N_GPUS_PER_NODE=8 \
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/train.parquet \
TEST_FILE=/path/to/aime24.parquet \
bash recipe/sdpo/run_math_sdpo.sh
```

The launcher validates model paths, Parquet schemas, reward behavior, sequence
lengths, batch divisibility, and optionally the live Ray resources before
submitting the job.

Useful parameters include:

```bash
SDPO_ALPHA=0.5
SDPO_DISTILLATION_TOPK=100
SDPO_TEACHER_REGULARIZATION=ema
SDPO_TEACHER_UPDATE_RATE=0.0
TOTAL_TRAINING_STEPS=180
```

Set `actor_rollout_ref.actor.policy_loss.loss_mode=vanilla` to use the same stack
as a GRPO baseline without constructing a distillation teacher.
