# Example: AIME 2024 Seed-0 Semantic Strategy Diversity

- Judge: an OpenAI-compatible language model
- Step: 180
- Samples per problem: 16
- Full responses were used without text compression.
- Primary metric: Simpson strategy diversity, $1-\sum_k p_k^2$.
- Correct-only averages include only problems with at least two correct trajectories.

## Summary

| Method | All-response Simpson ↑ | All strategy count ↑ | Correct-only Simpson ↑ | Correct strategy count ↑ | Correct-only coverage | Mean correct / 16 |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 0.2966 | 2.30 | 0.1017 | 1.44 | 25/30 (83.3%) | 9.57 |
| FlowSD | 0.3156 | 2.43 | 0.2194 | 1.73 | 26/30 (86.7%) | 11.00 |
| RLSD | 0.2255 | 2.10 | 0.1456 | 1.46 | 24/30 (80.0%) | 9.73 |

## Additional metrics

| Method | Mode | Dominant strategy ratio ↓ | Normalized entropy ↑ | Rubric 1–5 ↑ | Problems used |
|---|---|---:|---:|---:|---:|
| GRPO | all | 0.7708 | 0.1850 | 2.000 | 30/30 |
| GRPO | correct_only | 0.9328 | 0.0733 | 1.240 | 25/30 |
| FlowSD | all | 0.7562 | 0.2038 | 1.967 | 30/30 |
| FlowSD | correct_only | 0.8274 | 0.1718 | 1.769 | 26/30 |
| RLSD | all | 0.8292 | 0.1447 | 1.633 | 30/30 |
| RLSD | correct_only | 0.8885 | 0.1278 | 1.417 | 24/30 |

## Paired Simpson comparisons

| Mode | Comparison | Mean difference | Wins–Ties–Losses | Pairs |
|---|---|---:|---:|---:|
| all | flowsd-grpo | 0.0190 | 14–8–8 | 30 |
| all | flowsd-rlsd | 0.0901 | 14–9–7 | 30 |
| all | rlsd-grpo | -0.0711 | 6–11–13 | 30 |
| correct_only | flowsd-grpo | 0.1065 | 10–13–2 | 25 |
| correct_only | flowsd-rlsd | 0.0712 | 9–12–3 | 24 |
| correct_only | rlsd-grpo | 0.0397 | 6–14–4 | 24 |
