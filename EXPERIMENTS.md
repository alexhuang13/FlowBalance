# Experiment Tracking

This repository intentionally does not publish internal scheduler identifiers,
private filesystem paths, credentials, raw W&B metadata, or complete generation
logs. Reproducible experiments should instead record the following public
metadata:

- Git commit and working-tree status;
- model identifier and revision;
- dataset identifier, revision, and preprocessing command;
- canonical recipe and all environment-variable overrides;
- node/GPU topology and software image;
- random seeds and sampling parameters;
- checkpoint selection rule;
- evaluation command and grader revision.

## Recommended run manifest

```json
{
  "algorithm": "flowsd",
  "model": "Qwen3-8B",
  "training_steps": 180,
  "learning_rate": 1e-6,
  "nodes": 4,
  "gpus_per_node": 8,
  "train_file": "train_dapo17k.parquet",
  "validation_file": "aime24_30_boxed.parquet",
  "rollout": {
    "prompts": 256,
    "responses_per_prompt": 8,
    "temperature": 1.0,
    "top_p": 1.0
  },
  "evaluation": {
    "seeds": [0, 1, 2, 3, 4],
    "temperature": 0.6,
    "top_p": 0.95
  }
}
```

Store manifests next to the output directory rather than embedding private
infrastructure paths in source-controlled documentation.
