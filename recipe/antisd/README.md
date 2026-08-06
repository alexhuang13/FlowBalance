# Anti-SDPO recipe

This directory adds a standalone Anti-SDPO path without modifying existing `recipe/sdpo` or GRPO files.

## What is implemented

- `loss_mode=grpo_ca`: full Anti-SDPO / GRPO-CA update path.
- Teacher reprompt batch construction for `sdpo` and `grpo_ca`: `antisd_ray_trainer.py`.
- Token-level Anti-SD PRM:
  - default `prm_forward_mode=jsd_unbiased`
  - default `prm_renyi_sign=-1.0` for Anti-SD direction
  - `PRM_RENYI_SIGN=1.0` recovers the ordinary SD direction.
- Sequence PRM normalization, PRM clipping, adaptive `u` clipping, length mask, and teacher-perplexity lambda warmup/controller.
- Independent FSDP worker/actor wrapper that creates and updates a teacher module for `grpo_ca`.

## Key files

- `antisd_config.py`: `AntiSDFSDPActorConfig`, `AntiSDSelfDistillationConfig`, `AntiSDCCIRConfig`.
- `antisd_core_algos.py`: GRPO-CA/JSD PRM advantage construction.
- `antisd_dp_actor.py`: full `grpo_ca` actor update loop.
- `antisd_ray_trainer.py`: teacher/reprompt batch construction for Anti-SDPO.
- `antisd_fsdp_workers.py`: FSDP worker wrapper.
- `main_antisd.py`: Hydra/Ray entry point.
- `config/antisd_trainer.yaml`: default full Anti-SDPO config.
- `run_math_antisd.sh`: Qwen/math launch script.

## Example

```bash
cd /path/to/FlowSD
MODEL_PATH=/path/to/Qwen3-8B \
SP_SIZE=8 \
MAX_RESPONSE_LENGTH=8192 \
MAX_MODEL_LEN=10240 \
MAX_REPROMPT_LEN=10240 \
NNODES=4 \
N_GPUS_PER_NODE=8 \
bash recipe/antisd/run_math_antisd.sh
```

Useful switches:

```bash
PRM_RENYI_SIGN=-1.0      # Anti-SDPO, default
PRM_RENYI_SIGN=1.0       # ordinary SD direction
PRM_FORWARD_MODE=jsd_unbiased
CA_LAMBDA_MODE=teacher_perp
TOTAL_TRAINING_STEPS=1   # smoke test
```
