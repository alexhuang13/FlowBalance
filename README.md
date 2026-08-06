# FlowSD

FlowSD is a unified research codebase for reinforcement-learning and
self-distillation methods for language-model reasoning. The repository combines
algorithm implementations, a shared `verl`/Ray/FSDP/vLLM training stack,
mathematical and code evaluation pipelines, and semantic strategy-diversity
analysis.

## Highlights

- A single runtime environment for GRPO, SDPO, FlowSD, OPSD, RLSD, and Anti-SDPO.
- Qwen3 math-training recipes based on DAPO-Math-17K and AIME validation.
- Multi-node Ray, FSDP, sequence parallelism, and vLLM V1 rollout support.
- Reproducible math and code evaluation utilities.
- LLM-judged semantic strategy-diversity analysis with release-safe example results.
- Vendored `verl` source: the repository does not require a Git submodule.

## Repository layout

```text
FlowSD/
├── analysis/
│   └── diversity/      # Semantic strategy clustering and diversity metrics
├── recipe/
│   ├── flowopsd/       # FlowSD implementation; historical package name: flowopsd
│   ├── opsd/           # On-Policy Self-Distillation
│   ├── rlsd/           # RLSD implementation
│   ├── antisd/         # Anti-SDPO implementation
│   ├── sdpo/           # SDPO implementation and shared distillation utilities
│   ├── grpo/           # Canonical GRPO launchers
│   ├── custom/         # Additional GRPO/DAPO recipes
│   └── custom_sft/     # Supervised fine-tuning recipes
├── core/               # Project trainers, workers, datasets, and reward managers
├── verl/               # Vendored verl training framework
├── evaluation/
│   ├── math/           # Benchmark preparation, inference, grading, and aggregation
│   └── code/           # EvalPlus and training-validation utilities
├── launch/             # Multi-node cluster launcher
├── scripts/            # Environment and release checks
├── runtime_env.yaml    # Shared Ray runtime environment
└── provenance/         # Upstream snapshots retained for traceability
```

## Supported algorithms

| Method | Canonical launcher | Notes |
|---|---|---|
| GRPO | `recipe/grpo/run_math_grpo_step180.sh` | Baseline implemented by the shared PPO stack |
| SDPO | `recipe/sdpo/run_math_sdpo.sh` | Privileged-context self-distillation |
| FlowSD | `recipe/flowopsd/run_math_flowopsd.sh` | Historical internal package name: `flowopsd` |
| OPSD | `recipe/opsd/run_math_opsd.sh` | Fixed-teacher forward-KL objective by default |
| RLSD | `recipe/rlsd/run_math_rlsd.sh` | Advantage reweighting with teacher-conditioned likelihood ratios |
| Anti-SDPO | `recipe/antisd/run_math_antisd.sh` | Anti-self-distillation / GRPO-CA path |
| SFT | `recipe/custom_sft/run_sft_qwen3_8b.sh` | FSDP or Megatron supervised fine-tuning |

## Installation

The H20 training recipes were developed with a CUDA image containing PyTorch,
Ray, vLLM, FlashAttention/FlashInfer, and the corresponding distributed-training
dependencies. Install the vendored `verl` package in editable mode when setting
up a new environment:

```bash
pip install -e ./verl
```

Validate a checkout before allocating an expensive training job:

```bash
python3 scripts/check_environment.py
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for the unified runtime configuration.

## Data, models, and outputs

Large assets are intentionally not stored in the repository. Configure them with
environment variables:

```bash
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/output
export TRAIN_FILE="$DATA_ROOT/rl/train_dapo17k.parquet"
export TEST_FILE="$DATA_ROOT/rl/aime24_30_boxed.parquet"
export MODEL_PATH="$MODEL_ROOT/Qwen3-8B"
```

The standard math recipes expect a prompt-style Parquet dataset compatible with
the `verl` RLHF data loader and a local Hugging Face model directory.

## Quick start

Run a preflight-only OPSD command:

```bash
DRY_RUN=1 bash recipe/opsd/run_math_opsd.sh
```

Launch a one-step smoke test on an existing four-node Ray cluster:

```bash
TOTAL_TRAINING_STEPS=1 \
TEST_FREQ=1 \
SAVE_FREQ=1 \
NNODES=4 \
N_GPUS_PER_NODE=8 \
bash recipe/opsd/run_math_opsd.sh
```

For platform-managed multi-node jobs, `launch/start.sh` starts or joins the Ray
cluster, injects a local W&B credential into an ignored runtime-env copy, runs
preflight checks, and dispatches the selected recipe.

## Evaluation

### Mathematics

```bash
python evaluation/math/scripts/prepare_benchmarks.py --help
python evaluation/math/scripts/eval_benchmarks.py --help
```

The math evaluation package supports checkpoint selection, Hugging Face model
merging, multi-seed vLLM inference, answer normalization, grading, and summary
tables.

### Code

```bash
python evaluation/code/scripts/eval_evalplus.py --help
python evaluation/code/scripts/summarize_training_val.py --help
```

## Semantic strategy diversity

The diversity module clusters independently sampled responses by their core
mathematical strategy and reports strategy count, dominant-cluster ratio,
normalized entropy, Simpson diversity, and correct-only diversity.

```bash
python -m analysis.diversity.evaluate --help
python -m analysis.diversity.aggregate --help
```

Methodology, input schema, and a compact example are documented in
[analysis/diversity/README.md](analysis/diversity/README.md).

## Reproducibility and release notes

- Do not commit model weights, checkpoints, raw evaluation generations, W&B
  directories, secrets, or private runtime-env files.
- The checked-in runtime environment contains a placeholder API key only.
- `provenance/` contains upstream source snapshots for traceability and is not the
  recommended runtime path.
- Historical experiment notes may describe infrastructure that is not available
  outside the original environment. Canonical launchers and the unified runtime
  environment are the maintained public interfaces.

Before publishing a release, run:

```bash
python3 scripts/check_environment.py
python3 scripts/check_release.py
python3 -m pytest -q analysis/diversity/tests recipe/flowopsd/tests recipe/rlsd/tests recipe/opsd/tests
```
