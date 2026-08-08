# Unified training environment

All maintained FlowSD algorithms use the repository-level `runtime_env.yaml` and
must be submitted from the FlowSD repository root. The runtime environment sets
`PYTHONPATH=verl:.`, the H20 compiler wrappers, NCCL socket mode, and vLLM V1.

Supported canonical launchers:

- GRPO: `recipe/grpo/run_math_grpo_step180.sh`
- SDPO: `recipe/sdpo/run_math_sdpo.sh`
- FlowSD: `recipe/flowsd/run_math_flowsd.sh`
- Anti-SDPO: `recipe/antisd/run_math_antisd.sh`
- RLSD: `recipe/rlsd/run_math_rlsd.sh`
- OPSD: `recipe/opsd/run_math_opsd.sh`
- SFT: `recipe/custom_sft/run_sft_qwen3_8b.sh`

`launch/start.sh` defaults to `${CEPH_ROOT}/FlowSD`, creates a credential-injected
`.runtime_env.local.yaml`, starts the Ray cluster, and dispatches the selected
recipe using `TRAIN_RECIPE_SUBDIR` plus `RUN_SCRIPT`.

Before submitting an expensive job:

```bash
cd /path/to/FlowSD
python3 scripts/check_environment.py
DRY_RUN=1 bash recipe/opsd/run_math_opsd.sh
```

The checked-in runtime environment contains only a W&B placeholder. Production
jobs should be started through `launch/start.sh`, or explicitly point
`RUNTIME_ENV` to a local credential-injected copy.
