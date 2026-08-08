# Evaluation

FlowSD includes independent pipelines for mathematical reasoning, code
generation, and semantic strategy diversity.

## Mathematics

The math pipeline is located in `evaluation/math/` and supports:

- benchmark preparation and provenance manifests;
- checkpoint discovery and model merging;
- multi-seed vLLM generation;
- answer extraction and normalization;
- pass@1, pass@k, and majority-vote summaries;
- Ray entry points for distributed evaluation.

```bash
python evaluation/math/scripts/prepare_benchmarks.py --help
python evaluation/math/scripts/eval_benchmarks.py --help
```

Evaluation datasets and generated responses may have separate licenses. Review
the source manifest before redistributing data or outputs.

## Code

```bash
python evaluation/code/scripts/eval_evalplus.py --help
python evaluation/code/scripts/summarize_training_val.py --help
```

The EvalPlus dependency and benchmark data should be installed according to
their upstream instructions.

## Semantic diversity

```bash
python -m analysis.diversity.evaluate --help
python -m analysis.diversity.aggregate --help
```

See `analysis/diversity/README.md` for the judge protocol, metrics, and included
release-safe example.

## Environment

Evaluation should run in the same Python/CUDA environment as training whenever
possible. The repository-level `runtime_env.yaml` is intended for Ray training
jobs; evaluation-specific Ray jobs may use `evaluation/math/runtime_env.yaml`.
