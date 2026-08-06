#!/usr/bin/env python3
"""Selected experiment: 3 algorithms x steps {150,180}; all benchmarks n=1, AIME n=16."""
from __future__ import annotations
import csv,json,os,re,shutil,subprocess
from pathlib import Path
import ray
ROOT=Path('/apdcephfs_gy4/share_303378103/user/audenhuang'); WORK=ROOT/'math_benchmark_eval'; STABLE=ROOT/'stable_rl'
OUT=WORK/'results_selected_steps_150_180_t06_p095_k20_len38912_b16'; MERGED=WORK/'merged_models/selected_steps_150_180'
ALL=['aime24','aime25','aime26','hmmt25','minerva_math','math500','olympiadbench']; AIME=['aime24','aime25','aime26']
RUNS={
 'grpo':ROOT/'output/verl-grpo/GRPO-Qwen3-8B-math-dapo17k-wandb-run34',
 'rlsd':ROOT/'output/verl-rlsd-native/RLSD-Native-Qwen3-8B-math-dapo17k-run34',
 'flowopsd':ROOT/'output/verl-flowopsd/FlowOPSD-Qwen3-8B-math-dapo17k-betaq1-grpo-signadv-frozenref-etaR15-lr1e6-run28'}

def env():
 e=os.environ.copy(); tc=str(STABLE/'recipe/sdpo/toolchain')
 e.update({'TOKENIZERS_PARALLELISM':'false','VLLM_USE_V1':'1','VLLM_USE_FLASHINFER_SAMPLER':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','NCCL_ALGO':'Ring','NCCL_IB_DISABLE':'1','TORCH_NCCL_AVOID_RECORD_STREAMS':'1','TORCH_CUDA_ARCH_LIST':'9.0','PYTHONPATH':f'{STABLE}:{STABLE}/verl:'+e.get('PYTHONPATH',''),'CC':tc+'/gcc','CXX':tc+'/g++','TRITON_CC':tc+'/gcc','CUDAHOSTCXX':tc+'/g++','PATH':tc+':'+e.get('PATH','')})
 return e

@ray.remote(num_cpus=32)
def merge_one(algorithm,step):
 tag=f'{algorithm}_step_{step}'; target=MERGED/tag; tmp=target.with_name(tag+'.merging'); actor=RUNS[algorithm]/f'global_step_{step}/actor'
 if (target/'config.json').exists(): return str(target)
 shutil.rmtree(tmp,ignore_errors=True); tmp.mkdir(parents=True,exist_ok=True)
 subprocess.run(['python3','-m','verl.model_merger','merge','--backend','fsdp','--local_dir',str(actor),'--target_dir',str(tmp),'--use_cpu_initialization'],cwd=STABLE,env=env(),check=True)
 if not (tmp/'config.json').exists(): raise RuntimeError(f'merge incomplete {tag}')
 shutil.rmtree(target,ignore_errors=True); tmp.rename(target); return str(target)

@ray.remote(num_gpus=1,num_cpus=8,max_retries=1)
def evaluate_unit(algorithm,step,dataset,n,model):
 tag=f'{algorithm}_step_{step}'; setting=f'n{n}'; summary=OUT/tag/setting/dataset/f'summary_s0_e-1_n{n}_seed0.json'
 if summary.exists(): return {'tag':tag,'dataset':dataset,'n':n,'status':'skip'}
 cmd=['python3',str(WORK/'scripts/eval_benchmarks.py'),'--model',model,'--model-tag',tag,
  '--data-root',str(WORK/'data/processed'),'--output-root',str(OUT/setting),'--datasets',dataset,
  '--n',str(n),'--temperature','0.6','--top-p','0.95','--top-k','20','--max-tokens','38912',
  '--max-model-len','40960','--eval-batch-size','16',
  '--tp','1','--gpu-memory-utilization','0.80','--wandb-run-name',f'math-eval-selected-{tag}-{dataset}-n{n}',
  '--wandb-group',f'selected-{algorithm}-step-{step}','--checkpoint-step',str(step)]
 subprocess.run(cmd,cwd=WORK,env=env(),check=True)
 # eval script path is OUT/setting/tag/dataset; normalize to desired OUT/tag/setting/dataset.
 generated=OUT/setting/tag/dataset
 summary_generated=generated/f'summary_s0_e-1_n{n}_seed0.json'
 desired=OUT/tag/setting/dataset; desired.parent.mkdir(parents=True,exist_ok=True)
 if desired.exists(): shutil.rmtree(desired)
 shutil.move(str(generated),str(desired))
 # Remove empty intermediate dirs where possible.
 for p in [OUT/setting/tag,OUT/setting]:
  try:p.rmdir()
  except OSError:pass
 if not (desired/summary_generated.name).exists(): raise RuntimeError(f'missing {desired/summary_generated.name}')
 return {'tag':tag,'dataset':dataset,'n':n,'status':'complete'}

def aggregate():
 rows=[]
 for p in OUT.glob('*/*/*/summary_s0_e-1_n*_seed0.json'):
  # tag / nX / dataset / summary
  tag=p.parents[2].name; setting=p.parents[1].name; dataset=p.parent.name
  m=re.fullmatch(r'(grpo|rlsd|flowopsd)_step_(150|180)',tag)
  if not m:continue
  d=json.loads(p.read_text()); d.update(algorithm=m.group(1),checkpoint_step=int(m.group(2)),setting=setting,dataset=dataset); rows.append(d)
 rows.sort(key=lambda d:(d['algorithm'],d['checkpoint_step'],d['setting'],d['dataset']))
 fields=['algorithm','checkpoint_step','setting','dataset','problems','n','pass@1','pass@16','sample_correct','total_samples','avg_response_tokens','elapsed_seconds','grade_errors','result_path']
 OUT.mkdir(parents=True,exist_ok=True); out=OUT/'summary.csv'; tmp=out.with_suffix('.tmp')
 with tmp.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();[w.writerow({k:d.get(k) for k in fields}) for d in rows]
 tmp.replace(out);print('[aggregate]',len(rows),'of 60',flush=True)

def main():
 ray.init(address='auto',ignore_reinit_error=True); print('[cluster]',ray.cluster_resources(),flush=True)
 specs=[(a,s) for a in RUNS for s in (150,180)]
 merge_refs={x:merge_one.remote(*x) for x in specs}; models={x:ray.get(r) for x,r in merge_refs.items()}
 print('[merge] all six models ready',models,flush=True)
 units=[]
 for a,s in specs:
  units += [(a,s,d,1,models[(a,s)]) for d in ALL]
  units += [(a,s,d,16,models[(a,s)]) for d in AIME]
 print('[plan] units',len(units),'max_parallel_gpu=32',flush=True)
 refs={evaluate_unit.remote(*u):u for u in units};done=0
 while refs:
  ready,_=ray.wait(list(refs),num_returns=1,timeout=60)
  if not ready:print('[progress]',done,'/',len(units),flush=True);aggregate();continue
  ref=ready[0];spec=refs.pop(ref)
  try:print('[result]',ray.get(ref),flush=True)
  except Exception as e:print('[failed]',spec,repr(e),flush=True);raise
  done+=1;aggregate()
 for model in models.values():shutil.rmtree(model,ignore_errors=True)
 aggregate();print('[done] selected experiment',flush=True)
if __name__=='__main__':main()
