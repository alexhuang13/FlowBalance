#!/usr/bin/env bash
set -euo pipefail
ROOT=/apdcephfs_gy4/share_303378103/user/audenhuang
WORK=${ROOT}/math_benchmark_eval
STABLE=${ROOT}/stable_rl
export PYTHONPATH=${STABLE}:${STABLE}/verl:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
TOOLCHAIN=${STABLE}/recipe/sdpo/toolchain
export PATH=${TOOLCHAIN}:${PATH}
export CC=${TOOLCHAIN}/gcc CXX=${TOOLCHAIN}/g++ TRITON_CC=${TOOLCHAIN}/gcc CUDAHOSTCXX=${TOOLCHAIN}/g++
export LIBRARY_PATH=/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12${LIBRARY_PATH:+:${LIBRARY_PATH}}
export LD_LIBRARY_PATH=/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export COMPILER_PATH=/usr/libexec/gcc/x86_64-TencentOS-linux/12
export LDFLAGS="-L/usr/lib/gcc/x86_64-TencentOS-linux/12 -L/usr/lib64${LDFLAGS:+ ${LDFLAGS}}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0}
export NCCL_ALGO=Ring NCCL_IB_DISABLE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}

mkdir -p ${WORK}/{logs,results,merged_models/all_steps,manifests}
WANDB_ENV_FILE=${ROOT}/.secrets/wandb.env
if [ -f "${WANDB_ENV_FILE}" ]; then set -a; source "${WANDB_ENV_FILE}"; set +a; fi
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_PROJECT=${WANDB_PROJECT:-rlsd-math}
export WANDB_DIR=${WANDB_DIR:-${ROOT}/output/wandb}
mkdir -p "${WANDB_DIR}"
LOG=${WORK}/logs/eval_all_steps_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

echo "[plan] every complete checkpoint x all 7 benchmarks; n=16, pass@1/pass@16"
python3 - <<'PY'
import torch,transformers,vllm,wandb
print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.device_count())
print('transformers',transformers.__version__,'vllm',vllm.__version__,'wandb',wandb.__version__)
assert torch.cuda.device_count() == 8, torch.cuda.device_count()
PY
echo "[preflight] compiler: $(${CC} --version | head -1)"
echo 'int x(void){return 0;}' >/tmp/eval_link_preflight.c
"${CC}" -shared -fPIC /tmp/eval_link_preflight.c -o /tmp/eval_link_preflight.so -lgcc
rm -f /tmp/eval_link_preflight.{c,so}
python3 - <<'PY'
from triton.runtime.driver import driver
print('[preflight] triton target:',driver.active.get_current_target())
PY

GRPO_RUN=${ROOT}/output/verl-grpo/GRPO-Qwen3-8B-math-dapo17k-wandb-run34
RLSD_RUN=${ROOT}/output/verl-rlsd-native/RLSD-Native-Qwen3-8B-math-dapo17k-run34
FLOWSD_RUN=${ROOT}/output/verl-flowsd/FlowSD-Qwen3-8B-math-dapo17k-betaq1-grpo-signadv-frozenref-etaR15-lr1e6-run28
DATASETS=(aime24 aime25 aime26 hmmt25 minerva_math math500 olympiadbench)

is_complete_actor() {
 local step_dir=$1
 local actor=${step_dir}/actor
 local shards
 shards=$(find "$actor" -maxdepth 1 -type f -name 'model_world_size_32_rank_*.pt' 2>/dev/null | wc -l)
 [ -f "$step_dir/data.pt" ] && [ "$shards" -eq 32 ] && [ -f "$actor/fsdp_config.json" ]
}

discover_checkpoints() {
 local manifest=${WORK}/manifests/all_step_checkpoints.tsv
 local tmp=${manifest}.tmp
 : >"$tmp"
 local algorithm run step_dir step
 for pair in "grpo:${GRPO_RUN}" "rlsd:${RLSD_RUN}" "flowsd:${FLOWSD_RUN}"; do
  algorithm=${pair%%:*}; run=${pair#*:}
  while IFS= read -r step_dir; do
   is_complete_actor "$step_dir" || continue
   step=${step_dir##*_}
   printf '%s\t%s\t%s\t%s\n' "$algorithm" "$step" "$step_dir/actor" "$run" >>"$tmp"
  done < <(find "$run" -maxdepth 1 -type d -name 'global_step_*' | sort -V)
 done
 mv "$tmp" "$manifest"
 echo "[plan] checkpoint manifest: $manifest"
 awk -F '\t' '{n[$1]++} END{for(k in n) print "[plan]",k,n[k]}' "$manifest" | sort
 echo "[plan] total $(wc -l < "$manifest") checkpoints"
}

merge_checkpoint() {
 local tag=$1 actor=$2 target=$3
 local tmp=${target}.merging
 if [ -f "$target/config.json" ] && { [ -f "$target/model.safetensors.index.json" ] || compgen -G "$target/*.safetensors" >/dev/null; }; then
  echo "[merge resume] $tag -> $target"; return
 fi
 rm -rf "$tmp" "$target"
 mkdir -p "$tmp"
 cd "$STABLE"
 python3 -m verl.model_merger merge --backend fsdp --local_dir "$actor" --target_dir "$tmp" --use_cpu_initialization
 test -f "$tmp/config.json"
 mv "$tmp" "$target"
 echo "[merge done] $tag -> $target"
}

aggregate_results() {
 python3 - <<'PY'
from pathlib import Path
import csv,json,re
root=Path('/apdcephfs_gy4/share_303378103/user/audenhuang/math_benchmark_eval/results')
rows=[]
for p in root.glob('*/*/summary_s0_e-1_n16_seed0.json'):
 d=json.loads(p.read_text())
 m=re.fullmatch(r'(grpo|rlsd|flowsd)_step_(\d+)',d.get('model_tag',''))
 if not m: continue
 d['algorithm']=m.group(1); d['checkpoint_step']=int(m.group(2)); rows.append(d)
rows.sort(key=lambda d:(d['algorithm'],d['checkpoint_step'],d['dataset']))
fields=['algorithm','checkpoint_step','model_tag','dataset','problems','n','pass@1','pass@16','sample_correct','total_samples','avg_response_tokens','elapsed_seconds','grade_errors','result_path']
out=root/'all_steps_summary.csv'; tmp=out.with_suffix('.tmp')
with tmp.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
 for d in rows: w.writerow({k:d.get(k) for k in fields})
tmp.replace(out)
print('[aggregate]',out,'rows=',len(rows))
PY
}

discover_checkpoints
MANIFEST=${WORK}/manifests/all_step_checkpoints.tsv
TOTAL=$(wc -l < "$MANIFEST"); INDEX=0
while IFS=$'\t' read -r algorithm step actor run; do
 INDEX=$((INDEX+1))
 tag=${algorithm}_step_${step}
 target=${WORK}/merged_models/all_steps/${tag}
 done_marker=${WORK}/results/${tag}/.all_benchmarks_complete
 if [ -f "$done_marker" ]; then echo "[checkpoint skip $INDEX/$TOTAL] $tag"; continue; fi
 echo "[checkpoint start $INDEX/$TOTAL] $tag actor=$actor"
 merge_checkpoint "$tag" "$actor" "$target"
 cd "$WORK"
 python3 scripts/eval_benchmarks.py \
  --model "$target" --model-tag "$tag" \
  --data-root data/processed --output-root results \
  --datasets "${DATASETS[@]}" --n 16 --tp 8 \
  --wandb-run-name "math-eval-${tag}" --wandb-group "$algorithm" --checkpoint-step "$step"
 for dataset in "${DATASETS[@]}"; do
  test -f "results/${tag}/${dataset}/summary_s0_e-1_n16_seed0.json"
 done
 touch "$done_marker"
 aggregate_results
 rm -rf "$target"
 echo "[checkpoint done $INDEX/$TOTAL] $tag; temporary merged model removed"
done < "$MANIFEST"
aggregate_results
echo "[done] evaluated every discovered complete checkpoint on all benchmarks"
