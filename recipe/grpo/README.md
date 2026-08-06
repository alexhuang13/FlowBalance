# GRPO

This directory provides the canonical GRPO launchers for FlowSD. The algorithm,
trainer, workers, and configuration system are supplied by the vendored `verl`
stack rather than duplicated here.

Relevant implementation paths include:

- `verl/verl/trainer/ppo/core_algos.py`
- `verl/verl/trainer/ppo/ray_trainer.py`
- `verl/verl/trainer/config/`
- `verl/verl/workers/`

Run the general math recipe:

```bash
bash recipe/grpo/run_math_grpo.sh
```

Run the step-180 comparable configuration:

```bash
bash recipe/grpo/run_math_grpo_step180.sh
```

Both launchers use the repository-level `runtime_env.yaml` and accept the common
`DATA_ROOT`, `MODEL_ROOT`, `OUTPUT_ROOT`, `TRAIN_FILE`, `TEST_FILE`, and
`MODEL_PATH` overrides.
