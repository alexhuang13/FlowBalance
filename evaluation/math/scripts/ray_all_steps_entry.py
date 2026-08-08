#!/usr/bin/env python3
"""Evaluate complete checkpoints concurrently: one TP=1 vLLM evaluator per GPU."""
from __future__ import annotations
import csv, json, os, re, shutil, subprocess, time
from pathlib import Path
import ray

ROOT=Path('/apdcephfs_gy4/share_303378103/user/audenhuang')
WORK=ROOT/'math_benchmark_eval'
STABLE=ROOT/'stable_rl'
MANIFEST=WORK/'manifests/all_step_checkpoints.tsv'
DATASETS=['aime24','aime25','aime26','hmmt25','minerva_math','math500','olympiadbench']

def complete(tag: str) -> bool:
    return (WORK/'results'/tag/'.all_benchmarks_complete').exists()

@ray.remote(num_gpus=1, num_cpus=8, max_retries=1)
def evaluate_checkpoint(algorithm: str, step: int, actor: str):
    tag=f'{algorithm}_step_{step}'
    target=WORK/'merged_models/all_steps'/tag
    merging=target.with_name(target.name+'.merging')
    marker=WORK/'results'/tag/'.all_benchmarks_complete'
    if marker.exists(): return {'tag':tag,'status':'already_complete'}
    env=os.environ.copy()
    env.update({
      'TOKENIZERS_PARALLELISM':'false','VLLM_USE_V1':'1','VLLM_USE_FLASHINFER_SAMPLER':'0',
      'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','NCCL_ALGO':'Ring','NCCL_IB_DISABLE':'1',
      'TORCH_NCCL_AVOID_RECORD_STREAMS':'1','TORCH_CUDA_ARCH_LIST':'9.0',
      'PYTHONPATH':f'{STABLE}:{STABLE}/verl:'+env.get('PYTHONPATH',''),
      'CC':str(STABLE/'recipe/sdpo/toolchain/gcc'),'CXX':str(STABLE/'recipe/sdpo/toolchain/g++'),
      'TRITON_CC':str(STABLE/'recipe/sdpo/toolchain/gcc'),'CUDAHOSTCXX':str(STABLE/'recipe/sdpo/toolchain/g++'),
    })
    toolchain=str(STABLE/'recipe/sdpo/toolchain')
    env['PATH']=toolchain+':'+env.get('PATH','')
    try:
      if not (target/'config.json').exists():
        shutil.rmtree(merging,ignore_errors=True); shutil.rmtree(target,ignore_errors=True)
        merging.mkdir(parents=True)
        cmd=['python3','-m','verl.model_merger','merge','--backend','fsdp','--local_dir',actor,
             '--target_dir',str(merging),'--use_cpu_initialization']
        subprocess.run(cmd,cwd=STABLE,env=env,check=True)
        if not (merging/'config.json').exists(): raise RuntimeError(f'merge incomplete: {tag}')
        merging.rename(target)
      cmd=['python3',str(WORK/'scripts/eval_benchmarks.py'),
        '--model',str(target),'--model-tag',tag,'--data-root',str(WORK/'data/processed'),
        '--output-root',str(WORK/'results'),'--datasets',*DATASETS,'--n','16','--tp','1',
        '--gpu-memory-utilization','0.80','--wandb-run-name',f'math-eval-{tag}',
        '--wandb-group',algorithm,'--checkpoint-step',str(step)]
      subprocess.run(cmd,cwd=WORK,env=env,check=True)
      for dataset in DATASETS:
        p=WORK/'results'/tag/dataset/'summary_s0_e-1_n16_seed0.json'
        if not p.exists(): raise RuntimeError(f'missing summary: {p}')
      marker.touch()
      return {'tag':tag,'status':'complete'}
    finally:
      shutil.rmtree(merging,ignore_errors=True)
      if marker.exists(): shutil.rmtree(target,ignore_errors=True)

def aggregate():
    rows=[]
    for p in (WORK/'results').glob('*/*/summary_s0_e-1_n16_seed0.json'):
      d=json.loads(p.read_text())
      m=re.fullmatch(r'(grpo|rlsd|flowsd)_step_(\d+)',d.get('model_tag',''))
      if not m: continue
      d['algorithm']=m.group(1); d['checkpoint_step']=int(m.group(2)); rows.append(d)
    rows.sort(key=lambda d:(d['algorithm'],d['checkpoint_step'],d['dataset']))
    fields=['algorithm','checkpoint_step','model_tag','dataset','problems','n','pass@1','pass@16',
      'sample_correct','total_samples','avg_response_tokens','elapsed_seconds','grade_errors','result_path']
    out=WORK/'results/all_steps_summary.csv'; tmp=out.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:
      w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
      for d in rows: w.writerow({k:d.get(k) for k in fields})
    tmp.replace(out); print('[aggregate]',len(rows),'rows',flush=True)

def main():
    ray.init(address='auto',ignore_reinit_error=True)
    print('[cluster]',ray.cluster_resources(),flush=True)
    entries=[]
    for line in MANIFEST.read_text().splitlines():
      algorithm,step,actor,_run=line.split('\t'); entries.append((algorithm,int(step),actor))
    pending=[x for x in entries if not complete(f'{x[0]}_step_{x[1]}')]
    print(f'[plan] total={len(entries)} complete={len(entries)-len(pending)} pending={len(pending)}',flush=True)
    refs={evaluate_checkpoint.remote(*x):x for x in pending}
    done=0
    while refs:
      ready,_=ray.wait(list(refs),num_returns=1,timeout=60)
      if not ready:
        print(f'[progress] done={done} running_or_queued={len(refs)}',flush=True); aggregate(); continue
      ref=ready[0]; spec=refs.pop(ref)
      try: print('[worker result]',ray.get(ref),flush=True)
      except Exception as exc:
        print('[worker failed]',spec,repr(exc),flush=True)
        raise
      done+=1; aggregate()
    aggregate(); print('[done] all checkpoints',flush=True)
if __name__=='__main__': main()
