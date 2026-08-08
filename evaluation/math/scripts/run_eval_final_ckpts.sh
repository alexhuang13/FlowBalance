#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy4/share_303378103/user/audenhuang
WORK=${ROOT}/math_benchmark_eval
STABLE=${ROOT}/stable_rl
export PYTHONPATH=${STABLE}:${STABLE}/verl:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Use the validated recipe toolchain from the successful TP=1 smoke.
TOOLCHAIN=${STABLE}/recipe/sdpo/toolchain
export PATH=${TOOLCHAIN}:${PATH}
export CC=${TOOLCHAIN}/gcc
export CXX=${TOOLCHAIN}/g++
export TRITON_CC=${TOOLCHAIN}/gcc
export CUDAHOSTCXX=${TOOLCHAIN}/g++
export LIBRARY_PATH=/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12${LIBRARY_PATH:+:${LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export COMPILER_PATH=/usr/libexec/gcc/x86_64-TencentOS-linux/12
export LDFLAGS="-L/usr/lib/gcc/x86_64-TencentOS-linux/12 -L/usr/lib64${LDFLAGS:+ ${LDFLAGS}}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
# Same stable NCCL path as launch/start.sh. The platform default Tree algorithm
# cannot serve vLLM's ncclInt8 AllGather on this H20/NCCL build.
export NCCL_ALGO=Ring
export NCCL_IB_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}
rm -rf /root/.cache/flashinfer /root/.cache/torch/inductor /root/.cache/triton
mkdir -p ${WORK}/{logs,results,merged_models}
# Load W&B credentials without echoing the key.
WANDB_ENV_FILE=${ROOT}/.secrets/wandb.env
if [ -f "${WANDB_ENV_FILE}" ]; then
 set +x
 set -a; source "${WANDB_ENV_FILE}"; set +a
 set -x
fi
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-rlsd-math}
export WANDB_DIR=${WANDB_DIR:-${ROOT}/output/wandb}
mkdir -p "${WANDB_DIR}"
LOG=${WORK}/logs/eval_job_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
python3 - <<'PY'
import torch,transformers,vllm
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())
print('transformers',transformers.__version__,'vllm',vllm.__version__)
PY
echo "[preflight] compiler version: $(${CC} --version | head -1)"
echo "[preflight] libgcc: $(${CC} -print-libgcc-file-name)"
# Fail fast on the exact native-link path that broke the previous task.
echo 'int x(void){return 0;}' > /tmp/eval_link_preflight.c
"${CC}" -shared -fPIC /tmp/eval_link_preflight.c -o /tmp/eval_link_preflight.so -lgcc
rm -f /tmp/eval_link_preflight.c /tmp/eval_link_preflight.so
echo "[preflight] compiler/linker can resolve -lgcc via CC=${CC}"
python3 - <<'PY'
from triton.runtime.driver import driver
print('[preflight] triton target:', driver.active.get_current_target())
PY
merge_one () {
 tag=$1; actor=$2; target=${WORK}/merged_models/${tag}
 if [ -f "${target}/model.safetensors.index.json" ] || ls "${target}"/*.safetensors >/dev/null 2>&1; then echo "[merge skip] ${tag}"; return; fi
 mkdir -p "$target"
 cd "$STABLE"
 python3 -m verl.model_merger merge --backend fsdp --local_dir "$actor" --target_dir "$target" --use_cpu_initialization
 test -f "$target/config.json"
}
latest_complete_actor () {
 run_dir=$1
 for step_dir in $(find "$run_dir" -maxdepth 1 -type d -name 'global_step_*' | sort -Vr); do
  actor="$step_dir/actor"
  shards=$(find "$actor" -maxdepth 1 -type f -name 'model_world_size_32_rank_*.pt' 2>/dev/null | wc -l)
  if [ -f "$step_dir/data.pt" ] && [ "$shards" -eq 32 ] && [ -f "$actor/fsdp_config.json" ]; then
   echo "$actor"; return 0
  fi
 done
 return 1
}
GRPO_RUN=${ROOT}/output/verl-grpo/GRPO-Qwen3-8B-math-dapo17k-wandb-run34
GRPO_ACTOR=$(latest_complete_actor "$GRPO_RUN")
# RLSD remains pinned to the user-confirmed current checkpoint while its training continues.
RLSD_ACTOR=${ROOT}/output/verl-rlsd-native/RLSD-Native-Qwen3-8B-math-dapo17k-run34/global_step_190/actor
FLOW_ACTOR=${ROOT}/output/verl-flowsd/FlowSD-Qwen3-8B-math-dapo17k-betaq1-grpo-signadv-frozenref-etaR15-lr1e6-run28/global_step_410/actor
GRPO_STEP=$(basename "$(dirname "$GRPO_ACTOR")" | sed 's/global_step_//')
echo "[selection] GRPO latest complete step=${GRPO_STEP} actor=${GRPO_ACTOR}"
merge_one grpo_final "$GRPO_ACTOR"
merge_one rlsd_step190 "$RLSD_ACTOR"
merge_one flowsd_step410 "$FLOW_ACTOR"
cd "$WORK"
# First: a 2-problem AIME24 n=1/n=16 smoke test on GRPO.
python3 scripts/eval_benchmarks.py --model merged_models/grpo_final --model-tag smoke_grpo_final_n1 --data-root data/processed --output-root results --datasets aime24 --n 1 --start-idx 0 --end-idx 2 --tp 8
python3 scripts/eval_benchmarks.py --model merged_models/grpo_final --model-tag smoke_grpo_final_n16 --data-root data/processed --output-root results --datasets aime24 --n 16 --start-idx 0 --end-idx 2 --tp 8
# Full evaluation, one model process per method; each process loads once and walks datasets in order.
for spec in grpo_final rlsd_step190 flowsd_step410; do
 python3 scripts/eval_benchmarks.py --model merged_models/${spec} --model-tag ${spec} --data-root data/processed --output-root results --datasets aime24 aime25 aime26 hmmt25 minerva_math math500 olympiadbench --n 16 --tp 8
done
python3 - <<'PY'
from pathlib import Path
import json,csv
root=Path('results'); rows=[]
for p in root.glob('*/*/summary_*.json'):
 d=json.loads(p.read_text());
 if not d['model_tag'].startswith('smoke_'): rows.append(d)
out=Path('results/summary.csv')
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['model_tag','dataset','problems','n','pass@1','pass@16','avg_response_tokens','elapsed_seconds','grade_errors'])
 w.writeheader();
 for d in rows: w.writerow({k:d.get(k) for k in w.fieldnames})
print(out)
PY
