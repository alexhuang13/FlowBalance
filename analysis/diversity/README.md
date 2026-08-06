# Semantic Strategy Diversity Analysis

This module measures whether independently sampled mathematical solutions use
meaningfully different strategies rather than superficial wording variations.
It is designed for JSONL outputs produced by the math evaluation pipeline.

## Method

For each problem, an OpenAI-compatible judge receives anonymized and randomly
ordered sampled responses. The judge assigns every identifiable attempt to a
semantic strategy cluster. Clusters should reflect the central representation,
theorem, or decomposition, for example coordinate geometry versus synthetic
geometry or a recurrence versus a modular invariant.

The analysis reports:

- **Strategy count**: number of observed semantic clusters.
- **Dominant ratio**: fraction assigned to the largest cluster; lower is more diverse.
- **Normalized entropy**: entropy divided by `log(n)`.
- **Simpson diversity**: `1 - sum(p_k^2)`; higher is more diverse.
- **Correct-only diversity**: the same metrics after excluding incorrect responses.

Incorrect but coherent attempts remain in the all-response analysis. Empty,
pure-answer, or incoherent outputs may be marked invalid by the judge.

## Input format

Each JSONL row must contain:

```json
{
  "index": "aime24-00",
  "question": "...",
  "responses": ["...", "..."],
  "correctness": [true, false]
}
```

`correctness` is optional, but it is required for correct-only metrics.

## Run the judge

The evaluator supports standard OpenAI-compatible chat-completions endpoints.
Never commit API keys or raw judge request logs.

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=...
# Optional for a compatible provider:
export OPENAI_BASE_URL=https://api.openai.com/v1

python -m analysis.diversity.evaluate \
  --method 'grpo=/path/to/grpo/{dataset}/seed_{seed}/results.jsonl' \
  --method 'flowsd=/path/to/flowsd/{dataset}/seed_{seed}/results.jsonl' \
  --dataset aime24 \
  --seed 0 \
  --output outputs/diversity \
  --concurrency 4
```

Path patterns may use `{dataset}` and `{seed}`. Existing records are skipped, so
the command can be resumed safely.

## Aggregate records

```bash
python -m analysis.diversity.aggregate \
  --input outputs/diversity \
  --output outputs/diversity_summary
```

The aggregator writes per-problem metrics, method-level summaries, and a Markdown
table. Correct-only summaries exclude problems with fewer than two correct
trajectories by default.

## Included example

`example_results/` contains a compact, release-safe example from an AIME 2024,
seed-0, step-180 comparison with 16 samples per problem. It includes only derived
cluster metrics and a figure, not raw model responses or judge traces.

The example found higher correct-only Simpson diversity for FlowSD than for the
GRPO and RLSD runs in that evaluation. This is an empirical result for one model,
checkpoint, sampling configuration, judge, and dataset slice; it should not be
interpreted as a universal ranking.
