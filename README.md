<div align="center">

# FlowSD: Trajectory-Balanced On-Policy Self-Distillation

**Learn a distribution over correct reasoning trajectories—not just one correct answer.**

[Results](#main-results) · [Diversity](#more-than-accuracy-strategy-diversity) · [Quick Start](#quick-start) · [Evaluation](#evaluation) · [Citation](#citation)

</div>

---

## The message

> **For long-horizon reasoning, the distribution over correct answers matters as much as correctness itself.**

Many reasoning problems admit multiple valid solution strategies. Outcome-only reinforcement learning can discover one successful mode and then concentrate most of the policy mass on it. Local self-distillation provides denser feedback, but token imitation and local policy shaping still do not specify the normalized distribution that the student should represent over complete responses.

**FlowSD fills this gap.** It combines verifier outcomes, privileged teacher feedback, and reference-policy support into an explicit target distribution over complete reasoning trajectories, then fits that target with profiled trajectory balance.

### Why this matters

- **Accuracy:** FlowSD achieves the best five-benchmark average on both Qwen3-4B and Qwen3-8B.
- **Sampling performance:** On Qwen3-8B, FlowSD reaches **89.33% AIME24 Pass@16**.
- **Efficiency:** FlowSD reaches 0.5 AIME24 validation accuracy in about **100 steps**, versus roughly 143 for GRPO—about **1.43× faster**.
- **Stability:** FlowSD remains close to its peak over approximately 400 steps, while GRPO degrades after roughly step 180.
- **Diversity:** FlowSD more than doubles GRPO's correct-only semantic strategy diversity on AIME24.

---

## The central idea

Standard RLVR answers:

> Which sampled responses should receive more probability?

FlowSD asks the stronger question:

> **What normalized distribution should the policy represent over all sampled reasoning trajectories?**

FlowSD assigns four distinct roles:

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Component</th>
      <th align="center">Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center"><strong>Verifier advantage</strong></td>
      <td align="center">Anchors the target to task success or failure</td>
    </tr>
    <tr>
      <td align="center"><strong>Privileged teacher gain</strong></td>
      <td align="center">Supplies dense information along an already sampled trajectory</td>
    </tr>
    <tr>
      <td align="center"><strong>Reference policy</strong></td>
      <td align="center">Preserves support from the pretrained reasoning distribution</td>
    </tr>
    <tr>
      <td align="center"><strong>Trajectory balance</strong></td>
      <td align="center">Converts local signals into normalized relative probabilities over complete responses</td>
    </tr>
  </tbody>
</table>
</div>

The result is not merely another teacher-imitation loss. It is a **probability-conserving reasoning equilibrium**: verified, teacher-supported trajectories gain mass; verifier-rejected trajectories are suppressed through sign gating; and multiple successful solution modes can coexist.

<p align="center">
  <img src="assets/llm_method_overview.png" width="96%" alt="FlowSD method overview for language-model reasoning">
</p>
<p align="center"><sub><b>FlowSD for language-model reasoning.</b> Verifier-calibrated privileged feedback reweights the reference response distribution toward multiple successful reasoning modes.</sub></p>

---

## Main results

All main-table entries are step-180 results reported as mean ± sample standard deviation over five random seeds. AIME24 uses Pass@16; HMMT25, Minerva, MATH500, and OlympiadBench use Pass@1. The average is the unweighted mean of the five benchmark means.

### Qwen3-4B

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Method</th>
      <th align="center">AIME24@16</th>
      <th align="center">HMMT25</th>
      <th align="center">Minerva</th>
      <th align="center">MATH500</th>
      <th align="center">OlympiadBench</th>
      <th align="center"><strong>Avg.</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">GRPO</td>
      <td align="center">78.00 ± 1.83</td>
      <td align="center">26.67 ± 2.36</td>
      <td align="center"><strong>51.18 ± 1.36</strong></td>
      <td align="center">92.04 ± 0.98</td>
      <td align="center">63.68 ± 0.58</td>
      <td align="center">62.31</td>
    </tr>
    <tr>
      <td align="center">RLSD</td>
      <td align="center">73.33 ± 2.36</td>
      <td align="center">21.33 ± 3.80</td>
      <td align="center">50.29 ± 0.88</td>
      <td align="center">91.44 ± 0.52</td>
      <td align="center">61.36 ± 0.66</td>
      <td align="center">59.55</td>
    </tr>
    <tr>
      <td align="center"><strong>FlowSD</strong></td>
      <td align="center"><strong>80.00 ± 0.00</strong></td>
      <td align="center"><strong>32.00 ± 2.98</strong></td>
      <td align="center">50.51 ± 0.56</td>
      <td align="center"><strong>93.28 ± 0.59</strong></td>
      <td align="center"><strong>65.49 ± 0.92</strong></td>
      <td align="center"><strong>64.26</strong></td>
    </tr>
  </tbody>
</table>
</div>

**Takeaway:** FlowSD improves the five-benchmark average by **+1.95 points over GRPO** and **+4.71 over RLSD**, while leading four of the five reported benchmarks.

### Qwen3-8B

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Method</th>
      <th align="center">AIME24@16</th>
      <th align="center">HMMT25</th>
      <th align="center">Minerva</th>
      <th align="center">MATH500</th>
      <th align="center">OlympiadBench</th>
      <th align="center"><strong>Avg.</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">GRPO</td>
      <td align="center">85.33 ± 1.83</td>
      <td align="center">31.33 ± 7.67</td>
      <td align="center">52.87 ± 1.02</td>
      <td align="center">93.16 ± 0.83</td>
      <td align="center">64.78 ± 0.62</td>
      <td align="center">65.49</td>
    </tr>
    <tr>
      <td align="center">RLSD</td>
      <td align="center">82.67 ± 3.65</td>
      <td align="center">28.00 ± 1.83</td>
      <td align="center">52.94 ± 1.38</td>
      <td align="center">93.44 ± 0.17</td>
      <td align="center">63.56 ± 1.19</td>
      <td align="center">64.12</td>
    </tr>
    <tr>
      <td align="center"><strong>FlowSD</strong></td>
      <td align="center"><strong>89.33 ± 1.49</strong></td>
      <td align="center"><strong>34.67 ± 9.89</strong></td>
      <td align="center"><strong>53.68 ± 0.78</strong></td>
      <td align="center"><strong>93.52 ± 0.30</strong></td>
      <td align="center"><strong>66.85 ± 0.46</strong></td>
      <td align="center"><strong>67.61</strong></td>
    </tr>
  </tbody>
</table>
</div>

**Takeaway:** FlowSD obtains the best mean on **every reported benchmark**, improving the aggregate by **+2.12 points over GRPO** and **+3.49 over RLSD**.

### Where the gains are strongest

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Result</th>
      <th align="center">FlowSD</th>
      <th align="center">Best baseline</th>
      <th align="center">Gain</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Qwen3-4B average</td>
      <td align="center"><strong>64.26</strong></td>
      <td align="center">62.31 GRPO</td>
      <td align="center"><strong>+1.95</strong></td>
    </tr>
    <tr>
      <td align="center">Qwen3-8B average</td>
      <td align="center"><strong>67.61</strong></td>
      <td align="center">65.49 GRPO</td>
      <td align="center"><strong>+2.12</strong></td>
    </tr>
    <tr>
      <td align="center">Qwen3-8B AIME24@16</td>
      <td align="center"><strong>89.33</strong></td>
      <td align="center">85.33 GRPO</td>
      <td align="center"><strong>+4.00</strong></td>
    </tr>
    <tr>
      <td align="center">Qwen3-8B OlympiadBench</td>
      <td align="center"><strong>66.85</strong></td>
      <td align="center">64.78 GRPO</td>
      <td align="center"><strong>+2.07</strong></td>
    </tr>
    <tr>
      <td align="center">Qwen3-4B HMMT25</td>
      <td align="center"><strong>32.00</strong></td>
      <td align="center">26.67 GRPO</td>
      <td align="center"><strong>+5.33</strong></td>
    </tr>
  </tbody>
</table>
</div>

These results support the distributional view: the largest gain appears on sampling-heavy AIME24 Pass@16, while improvements on Pass@1 benchmarks show that FlowSD does not trade broad single-sample accuracy for sampling performance.

---

## More than accuracy: strategy diversity

A reasoning model can produce many correct samples that are merely surface-level rewrites of one dominant strategy. The paper therefore evaluates **semantic strategy diversity**, not lexical diversity.

The two-stage LLM-judge protocol:

1. extracts the mathematical representation, tools, theorems, and strategy signature from each complete response;
2. clusters anonymized strategy summaries within each problem and method.

Correct-only Simpson diversity is

```math
D_{\mathrm{Simpson}}=1-\sum_k p_k^2.
```

which is the probability that two sampled correct trajectories use different semantic strategies.

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Method</th>
      <th align="center">Correct-only Simpson diversity</th>
      <th align="center">Relative to GRPO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">GRPO</td>
      <td align="center">0.1017</td>
      <td align="center">1.00×</td>
    </tr>
    <tr>
      <td align="center">RLSD</td>
      <td align="center">0.1456</td>
      <td align="center">1.43×</td>
    </tr>
    <tr>
      <td align="center"><strong>FlowSD</strong></td>
      <td align="center"><strong>0.2194</strong></td>
      <td align="center"><strong>2.16×</strong></td>
    </tr>
  </tbody>
</table>
</div>

<p align="center">
  <img src="assets/llm_strategy_diversity.png" width="58%" alt="LLM-judged semantic strategy diversity on AIME24">
</p>

**FlowSD more than doubles GRPO's judged strategy diversity** in the AIME24 step-180, seed-0 diagnostic. This is the intended behavior of distributional self-distillation: allocate probability to multiple successful reasoning modes instead of collapsing onto the first dominant template.

---

## Canonical implementation

All public interfaces consistently use FlowSD/`flowsd`:

<div align="center">
<table>
  <thead>
    <tr>
      <th align="center">Interface</th>
      <th align="center">Canonical name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Python package</td>
      <td align="center"><code>recipe.flowsd</code></td>
    </tr>
    <tr>
      <td align="center">Main module</td>
      <td align="center"><code>recipe.flowsd.main_flowsd</code></td>
    </tr>
    <tr>
      <td align="center">Config class</td>
      <td align="center"><code>FlowSDConfig</code></td>
    </tr>
    <tr>
      <td align="center">Hydra block</td>
      <td align="center"><code>actor_rollout_ref.actor.flowsd</code></td>
    </tr>
    <tr>
      <td align="center">Loss mode</td>
      <td align="center"><code>flowsd</code></td>
    </tr>
    <tr>
      <td align="center">Environment variables</td>
      <td align="center"><code>FLOWSD_*</code></td>
    </tr>
    <tr>
      <td align="center">Metric prefix</td>
      <td align="center"><code>flowsd/</code></td>
    </tr>
    <tr>
      <td align="center">Launcher</td>
      <td align="center"><code>recipe/flowsd/run_math_flowsd.sh</code></td>
    </tr>
  </tbody>
</table>
</div>

### Repository layout

```text
FlowSD/
├── assets/              # README and paper figures
├── recipe/
│   ├── flowsd/          # Canonical FlowSD implementation
│   ├── grpo/            # GRPO baseline
│   ├── rlsd/            # RLSD baseline
│   ├── opsd/            # Forward-KL OPSD baseline
│   ├── sdpo/            # Shared privileged-distillation utilities
│   └── antisd/          # Anti-self-distillation baseline
├── core/                # Shared trainers, workers, datasets, and rewards
├── evaluation/math/     # Math inference, grading, and aggregation
├── evaluation/code/     # Code evaluation utilities
├── analysis/diversity/  # Semantic strategy-diversity analysis
├── launch/              # Multi-node launcher
├── scripts/             # Environment and release checks
├── verl/                # Vendored verl training framework
└── runtime_env.yaml
```

---

## Installation

The large-scale experiments use PyTorch, Ray, FSDP, vLLM, and the vendored `verl` stack.

```bash
pip install -e ./verl
python3 scripts/check_environment.py
```

See [ENVIRONMENT.md](ENVIRONMENT.md) for runtime details.

### Data and model paths

```bash
export DATA_ROOT=/path/to/data
export MODEL_ROOT=/path/to/models
export OUTPUT_ROOT=/path/to/output
export TRAIN_FILE="$DATA_ROOT/rl/train_dapo17k.parquet"
export TEST_FILE="$DATA_ROOT/rl/aime24_30_boxed.parquet"
export MODEL_PATH="$MODEL_ROOT/Qwen3-8B"
```

The math recipe expects prompt-style Parquet data and verifier-compatible ground-truth answers.

---

## Quick start

### One-step smoke test

Run on an existing Ray cluster:

```bash
STEP180_VAL_ENABLE=0 \
TOTAL_TRAINING_STEPS=1 \
TEST_FREQ=1 \
SAVE_FREQ=100000 \
NNODES=4 \
N_GPUS_PER_NODE=8 \
FLOWSD_BETA_Q=1 \
FLOWSD_ETA_R=15 \
bash recipe/flowsd/run_math_flowsd.sh
```

### Step-180 paper-style run

```bash
PROJECT_NAME=verl-flowsd \
EXP_NAME=FlowSD-Qwen3-8B-math-step180 \
MODEL_PATH=/path/to/Qwen3-8B \
TRAIN_FILE=/path/to/train_dapo17k.parquet \
TEST_FILE=/path/to/aime24_30_boxed.parquet \
OUTPUT_ROOT=/path/to/output \
NNODES=4 \
N_GPUS_PER_NODE=8 \
SP_SIZE=8 \
TRAIN_PROMPT_BSZ=256 \
N_RESP_PER_PROMPT=8 \
PPO_MINI_BATCH_SIZE=128 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=8192 \
MAX_MODEL_LEN=10240 \
MAX_REPROMPT_LEN=10240 \
TOTAL_TRAINING_STEPS=180 \
SAVE_FREQ=10 \
FLOWSD_BETA_Q=1 \
FLOWSD_ETA_R=15 \
FLOWSD_CLIP_B=4 \
FLOWSD_RHO=1 \
LR=1e-6 \
STEP180_VAL_ENABLE=1 \
bash recipe/flowsd/run_math_flowsd.sh
```

For platform-managed jobs:

```bash
TRAIN_RECIPE_SUBDIR=flowsd \
RUN_SCRIPT=run_math_flowsd.sh \
bash launch/start.sh
```

---

## Evaluation

The step-180 evaluator supports five seeds, seven n=1 math benchmarks, and n=16 evaluation on AIME24/25/26.

```bash
RUN_DIR=/path/to/experiment \
EXPERIMENT_NAME=FlowSD-Qwen3-8B-math-step180 \
VAL_STEP=180 \
VAL_SEEDS="0 1 2 3 4" \
VAL_KEEP_MERGED_MODEL=1 \
bash recipe/flowsd/submit_step180_val.sh
```

Expected aggregate outputs:

```text
results_mean_std.md
summary_all_seeds.csv
pass1_n1_mean_std_percent.csv
aime_pass16_n16_mean_std_percent.csv
aime_sample_pass1_n16_mean_std_percent.csv
.complete
```

### Run the paper ablations

```bash
# Verifier coefficient
ETA_SWEEP_VALUES="5 10 15" \
bash recipe/flowsd/run_math_flowsd_eta_sweep.sh

# Teacher coefficient
bash recipe/flowsd/run_math_flowsd_betaT_resume_sweep.sh
```

---

## Reproducibility

A strict reproduction should record:

- paper and code versions;
- container image;
- model and dataset hashes;
- rollout group size and response-length cap;
- verifier implementation;
- `FLOWSD_ETA_R`, `FLOWSD_BETA_Q`, `FLOWSD_CLIP_B`, and `FLOWSD_RHO`;
- reference and teacher refresh semantics;
- checkpoint step and evaluation protocol.

Historical checkpoint paths may retain immutable experiment labels. The maintained source code and public interfaces use only FlowSD/`flowsd`.

---

## Citation

```bibtex
@article{huang2026flowsd,
  title   = {FlowSD: Trajectory-Balanced On-Policy Self-Distillation},
  author  = {Zixun Huang and Kishan Panaganti and Haitao Mi and Leowei Liang},
  year    = {2026}
}
```
