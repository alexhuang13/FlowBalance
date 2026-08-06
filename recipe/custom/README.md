# Additional GRPO and DAPO Recipes

This directory contains alternative launchers and data-preparation utilities for
GRPO- and DAPO-style experiments. They use the shared repository-level
`runtime_env.yaml` and accept local paths through environment variables.

Prepare a compatible prompt-style Parquet dataset with
`preprocess_data.py --help`, then launch a recipe by overriding the common paths:

```bash
DATA_ROOT=/path/to/data \
MODEL_PATH=/path/to/Qwen3-4B \
TRAIN_FILE=/path/to/train.parquet \
TEST_FILE=/path/to/validation.parquet \
OUTPUT_ROOT=/path/to/output \
bash recipe/custom/run_dapo_qwen3_4b.sh
```

The scripts preserve several historical hyperparameter configurations. Review
batch sizes, sequence lengths, tensor/sequence parallelism, and node count before
running them on different hardware.
