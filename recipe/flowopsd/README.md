# FlowSD

The implementation is stored under the historical package name `flowopsd` to
preserve configuration and checkpoint compatibility. It augments a GRPO-style
on-policy update with a target energy derived from privileged-context teacher
gain and verifier reward.

## Core idea

For a sampled response `y`, the same reference checkpoint evaluates the response
under the original context `x` and a privileged context `(x, c)`:

```text
G_q(y; x, c) = log pi_teacher(y | x, c) - log pi_ref(y | x)
```

The target distribution is proportional to

```text
pi_ref(y | x) * exp(beta_q * G_q(y; x, c) + eta_R * R(y; x)).
```

The student never observes `c` at inference time. See `privileged.md` for prompt
construction, masking, grouping, and stop-gradient boundaries.

## Implementation

- `main_flowopsd.py`: Hydra/Ray entry point.
- `flowopsd_config.py`: typed FlowSD configuration.
- `flowopsd_core_algos.py`: target construction and diagnostics.
- `flowopsd_dp_actor.py`: actor loss and update path.
- `flowopsd_ray_trainer.py`: privileged batches and training metrics.
- `flowopsd_fsdp_workers.py`: FSDP/vLLM worker integration.
- `config/flowopsd_trainer.yaml`: configuration defaults.

## Comparable math configuration

The canonical launcher defaults to Qwen3-8B, DAPO-Math-17K, AIME 2024
validation, 256 prompts, 8 responses per prompt, 8192 response tokens, sequence
parallel size 8, and four nodes with eight GPUs each. Every parameter can be
overridden through environment variables.

```bash
NNODES=4 \
N_GPUS_PER_NODE=8 \
FLOWOPSD_BETA_Q=1 \
FLOWOPSD_ETA_R=15 \
LR=1e-6 \
bash recipe/flowopsd/run_math_flowopsd.sh
```

For controlled `eta_R` ablations:

```bash
ETA_SWEEP_VALUES="5 10 15 30" \
bash recipe/flowopsd/run_math_flowopsd_eta_sweep.sh
```

Run a short smoke test before a long experiment:

```bash
TOTAL_TRAINING_STEPS=1 TEST_FREQ=1 SAVE_FREQ=1 \
bash recipe/flowopsd/run_math_flowopsd.sh
```
