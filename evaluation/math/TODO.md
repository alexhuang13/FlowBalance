# GRPO / RLSD / FlowOPSD Math Benchmark Evaluation TODO

Updated: 2026-07-28 (Asia/Beijing)

## Target benchmarks
- AIME 2024
- AIME 2025
- AIME 2026
- HMMT 2025
- Minerva Math
- MATH-500
- Olympiad benchmark (exact HF dataset/config to confirm)

## Target checkpoints
- GRPO checkpoints under `/apdcephfs_gy4/share_303378103/user/audenhuang/output`
- RLSD checkpoints under the same output root
- FlowOPSD checkpoints under the same output root

## Metrics
- pass@1
- pass@16

## Ordered execution plan
- [x] Create evaluation workspace and persistent TODO.
- [x] Inventory existing evaluation scripts and environments.
- [x] Locate authoritative Hugging Face dataset repositories/configs.
- [x] Download each benchmark snapshot and record repo/revision/files/checksums.
- [x] Resolve benchmark ambiguities, especially AIME 2026, HMMT 2025, and "Olympiad".
- [x] Convert datasets to a common verl-compatible parquet schema.
- [x] Validate counts, uniqueness, prompt format, ground truth, and reward compatibility.
- [ ] Inventory GRPO/RLSD/FlowOPSD checkpoints and select checkpoint(s) per method.
- [ ] Lock evaluation semantics: temperature/top-p/max tokens and pass@k estimator.
- [ ] Smoke test one checkpoint on a few AIME24 problems at n=1 and n=16.
- [ ] Run full evaluations in benchmark order.
- [ ] Aggregate accuracy/pass@k, failures, runtime, and reproducible commands.

## Decisions requiring confirmation
1. Which exact checkpoint(s) should represent each method if multiple global steps/runs exist?
2. Does "Olympiad" mean `OlympiadBench`, an olympiad split from another collection, or another dataset?
3. For pass@16, use one batch of 16 sampled completions/problem and report empirical any-success, or generate more than 16 and use the unbiased pass@k estimator?
4. AIME 2026 availability and contamination/evaluation policy must be confirmed because the date is 2026-07-28.

## Reproducibility records
Every downloaded dataset will receive a manifest containing:
- Hugging Face repo and config
- resolved revision/commit
- source filenames
- local output path
- row count and schema
- SHA256
- preprocessing command/version

## Progress log

### 2026-07-28: AIME downloads
- AIME24: `BytedTsinghua-SIA/AIME-2024@aa49075e24ad594b79fdf0bdcefa735c2181be67`; source has 960 rows = 30 questions repeated 32 times. Must deduplicate.
- AIME25 candidate A: `opencompass/AIME2025@a6ad95f611d72cf628a80b58bd0432ef6638f958`; 15 AIME-I + 15 AIME-II questions.
- AIME25 candidate B: `MathArena/aime_2025@c94da77eb22bbd6439e62a323bec18493a421302`; combined 30 questions. Downloaded for cross-check.
- AIME26: `MathArena/aime_2026@d2de22f3c656b4f56cf8981212186377d1e23bc3`; combined 30 questions.
- Raw file checksums: `manifests/aime_raw_sha256.txt`.
- No evaluation jobs submitted yet.

### Current blocker / discussion
- HMMT25 candidates differ: `FlagEval/HMMT_2025` versus `PraMamba/HMMT-202502`. Need inspect scope/count and select benchmark definition.
- Olympiad benchmark definition remains unresolved; likely text-only English math subset of OlympiadBench, but must confirm before downloading/conversion.

### 2026-07-28: processed datasets
- Wrote deterministic converter: `scripts/prepare_benchmarks.py`.
- Processed parquet files: `data/processed/{aime24,aime25,aime26,hmmt25,minerva_math,math500,olympiadbench}.parquet`.
- Counts: 30, 30, 30, 30, 272, 500, 674. All IDs/prompts unique within datasets; no normalized exact prompt overlap across datasets.
- All 1,566 ground truths pass identity checks with the current SDPO grader.
- OlympiadBench: 674 text-only English math problems; 93 multiple-answer rows; one tolerance row. Evaluation grader must preserve these semantics.
- Processed checksums/stats: `manifests/processed_datasets.json`.
- Grader environment lock: `manifests/grader_venv_requirements.txt`.

### 2026-07-28: evaluation job submitted
- Final checkpoints selected: GRPO step 480; RLSD step 190; FlowOPSD step 410.
- Semantics: one n=16 sampling run per problem; pass@1 = mean correctness over 16 samples; pass@16 = fraction of problems with any correct sample. temperature=0.6, top_p=0.95, max_tokens=8192, Qwen3 thinking enabled.
- Independent Taiji task: `auden_math_benchmark_eval_final_ckpts`; instance `8b1d89ec9fa3ebab019fa68cbcd70466`; 1 node x 8 H20.
- Job order: merge three FSDP checkpoints, GRPO/AIME24 2-problem n=1+n=16 smoke tests, then complete 3 models x 7 datasets.

### 2026-07-28: GRPO stopped by user request
- Stopped `auden_grpo_qwen3_8b_wandb_run34` at 2026-07-28 10:38 Asia/Beijing.
- Verified final complete GRPO checkpoint: `global_step_490` (32/32 model shards + data.pt).
- RLSD `auden_rlsd_native_34` remains TRAINING_RUNNING and untouched.
- Evaluation task transitioned from quota waiting to PENDING after GPU release.
- Evaluation script selects GRPO latest complete checkpoint dynamically and labels it `grpo_final`.

### 2026-07-28: switched to every checkpoint
- User requested evaluation of every saved complete checkpoint, not only representative/final checkpoints.
- Discovered complete checkpoints: GRPO 49 (steps 10..490), RLSD 21 (steps 10..210), FlowOPSD 41 (steps 10..410): 111 checkpoints total.
- Each checkpoint will run all 7 benchmarks at n=16, producing 777 checkpoint/benchmark summaries.
- Disk-safe execution: merge one checkpoint, evaluate all benchmarks, persist results/W&B metrics, then delete that temporary merged model.
- Resume granularity is dataset-level plus an all-benchmarks checkpoint completion marker.
- Aggregate output: `results/all_steps_summary.csv`.
