#!/usr/bin/env python3
"""Generate n samples/problem with vLLM and report sample accuracy + empirical pass@n.

Outputs are append-safe at dataset granularity: a completed JSONL plus summary JSON
causes that dataset to be skipped on rerun. Raw completions and per-sample correctness
are retained so grading can be audited without regenerating.
"""
from __future__ import annotations
import argparse, gc, json, os, re, sys, time
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Reuse the repository's established math extraction/equivalence grader.
EVAL_UTILS = Path('/apdcephfs_gy4/share_303378103/user/audenhuang/stable_rl/recipe/sdpo/eval')
sys.path.insert(0, str(EVAL_UTILS))
from utils.parser import extract_answer  # noqa: E402
from utils.grader import check_is_correct  # noqa: E402

DATASETS = ['aime24','aime25','aime26','hmmt25','minerva_math','math500','olympiadbench']

def args():
 p=argparse.ArgumentParser()
 p.add_argument('--model',required=True); p.add_argument('--model-tag',required=True)
 p.add_argument('--data-root',required=True); p.add_argument('--output-root',required=True)
 p.add_argument('--datasets',nargs='+',default=DATASETS)
 p.add_argument('--n',type=int,default=16); p.add_argument('--temperature',type=float,default=.6)
 p.add_argument('--top-p',type=float,default=.95); p.add_argument('--top-k',type=int,default=20)
 p.add_argument('--max-tokens',type=int,default=38912)
 p.add_argument('--max-model-len',type=int,default=40960)
 p.add_argument('--eval-batch-size',type=int,default=16)
 p.add_argument('--tp',type=int,default=8); p.add_argument('--seed',type=int,default=0)
 p.add_argument('--start-idx',type=int,default=0); p.add_argument('--end-idx',type=int,default=-1)
 p.add_argument('--gpu-memory-utilization',type=float,default=.92)
 p.add_argument('--enforce-eager',action='store_true')
 p.add_argument('--wandb-project',default=os.environ.get('WANDB_PROJECT','rlsd-math'))
 p.add_argument('--wandb-entity',default=os.environ.get('WANDB_ENTITY') or None)
 p.add_argument('--wandb-run-name',default=None)
 p.add_argument('--wandb-group',default=None)
 p.add_argument('--checkpoint-step',type=int,default=None)
 p.add_argument('--no-wandb',action='store_true')
 a=p.parse_args()
 if a.eval_batch_size < 1: p.error('--eval-batch-size must be >= 1')
 if a.max_model_len <= a.max_tokens: p.error('--max-model-len must be greater than --max-tokens to leave room for the prompt')
 return a

def plain(x):
 if hasattr(x,'tolist'): return x.tolist()
 return x

def grade(response, gt):
 ans=extract_answer(response)
 try: ok=bool(check_is_correct(ans, str(gt)))
 except Exception as e: return ans,False,f'{type(e).__name__}: {e}'
 return ans,ok,None

def main():
 a=args(); outroot=Path(a.output_root)/a.model_tag; outroot.mkdir(parents=True,exist_ok=True)
 wb=None
 if not a.no_wandb and os.environ.get('WANDB_MODE','online') != 'disabled':
  try:
   import wandb
   wb=wandb.init(project=a.wandb_project,entity=a.wandb_entity,
     name=a.wandb_run_name or f'math-eval-{a.model_tag}',group=a.wandb_group,
     job_type='benchmark-eval',dir=os.environ.get('WANDB_DIR'),
     config={'model_tag':a.model_tag,'model_path':a.model,'datasets':a.datasets,'n':a.n,
       'temperature':a.temperature,'top_p':a.top_p,'top_k':a.top_k,'max_tokens':a.max_tokens,
       'max_model_len':a.max_model_len,'eval_batch_size':a.eval_batch_size,'tp':a.tp,'seed':a.seed,'start_idx':a.start_idx,'end_idx':a.end_idx,
       'algorithm':a.wandb_group,'checkpoint_step':a.checkpoint_step},
     tags=['math-benchmark',a.model_tag,f'pass@{a.n}'],reinit=True)
   print('[wandb] run',wb.url,flush=True)
  except Exception as e:
   print(f'[wandb] init failed: {type(e).__name__}: {e}',flush=True)
 tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
 llm=LLM(model=a.model,tensor_parallel_size=a.tp,trust_remote_code=True,
         gpu_memory_utilization=a.gpu_memory_utilization,max_model_len=a.max_model_len,
         enable_prefix_caching=True,seed=a.seed,enforce_eager=a.enforce_eager)
 sp=SamplingParams(n=a.n,temperature=a.temperature,top_p=a.top_p,top_k=a.top_k,
                   max_tokens=a.max_tokens,seed=a.seed)
 for name in a.datasets:
  dsout=outroot/name; dsout.mkdir(parents=True,exist_ok=True)
  suffix=f's{a.start_idx}_e{a.end_idx}_n{a.n}_seed{a.seed}'
  result_path=dsout/f'results_{suffix}.jsonl'; summary_path=dsout/f'summary_{suffix}.json'
  if result_path.exists() and summary_path.exists():
   try:
    previous=json.loads(summary_path.read_text())
    same_config=(previous.get('n') == a.n and previous.get('seed') == a.seed
      and previous.get('temperature') == a.temperature and previous.get('top_p') == a.top_p
      and previous.get('top_k', -1) == a.top_k and previous.get('max_tokens') == a.max_tokens
      and previous.get('max_model_len', 10240) == a.max_model_len
      and previous.get('eval_batch_size') == a.eval_batch_size)
   except Exception:
    same_config=False
   if same_config:
    print('[skip]',name,flush=True); continue
   print('[rerun config changed]',name,flush=True)
  df=pd.read_parquet(Path(a.data_root)/f'{name}.parquet')
  end=len(df) if a.end_idx<0 else min(a.end_idx,len(df)); df=df.iloc[a.start_idx:end].reset_index(drop=True)
  prompts=[]
  for messages in df.prompt:
   messages=plain(messages)
   prompts.append(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True))
  t0=time.time(); outputs=[]
  for batch_start in range(0,len(prompts),a.eval_batch_size):
   outputs.extend(llm.generate(prompts[batch_start:batch_start+a.eval_batch_size],sp))
  elapsed=time.time()-t0
  rows=[]; sample_correct=0; problem_pass=0; token_count=0; grade_errors=0
  for i,(o,(_,r)) in enumerate(zip(outputs,df.iterrows())):
   gt=str(r.reward_model['ground_truth']); responses=[x.text for x in o.outputs]
   answers=[]; correctness=[]; errors=[]; lengths=[]
   for x in o.outputs:
    ans,ok,err=grade(x.text,gt); answers.append(ans); correctness.append(ok); errors.append(err)
    lengths.append(len(x.token_ids)); token_count+=len(x.token_ids); sample_correct+=int(ok); grade_errors+=int(err is not None)
   problem_pass+=int(any(correctness))
   rows.append({'dataset':name,'index':r.extra_info['index'],'question':plain(r.prompt)[0]['content'],
     'gold_answer':gt,'extra_info':plain(r.extra_info),'responses':responses,'extracted_answers':answers,
     'correctness':correctness,'grade_errors':errors,'response_token_lengths':lengths})
  tmp=result_path.with_suffix('.tmp')
  with tmp.open('w') as f:
   for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
  os.replace(tmp,result_path)
  total_samples=len(rows)*a.n
  summary={'model_tag':a.model_tag,'model_path':a.model,'dataset':name,'problems':len(rows),'n':a.n,
   'temperature':a.temperature,'top_p':a.top_p,'top_k':a.top_k,'max_tokens':a.max_tokens,
   'max_model_len':a.max_model_len,'eval_batch_size':a.eval_batch_size,'seed':a.seed,
   'pass@1':sample_correct/total_samples if total_samples else 0.0,
   f'pass@{a.n}':problem_pass/len(rows) if rows else 0.0,
   'sample_correct':sample_correct,'total_samples':total_samples,'problem_any_correct':problem_pass,
   'generated_tokens':token_count,'avg_response_tokens':token_count/total_samples if total_samples else 0.0,
   'elapsed_seconds':elapsed,'grade_errors':grade_errors,'result_path':str(result_path)}
  summary_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
  print('[summary]',json.dumps(summary,ensure_ascii=False),flush=True)
  if wb is not None:
   metrics={f'{name}/pass@1':summary['pass@1'],f'{name}/pass@{a.n}':summary[f'pass@{a.n}'],
     f'{name}/avg_response_tokens':summary['avg_response_tokens'],f'{name}/elapsed_seconds':elapsed,
     f'{name}/grade_errors':grade_errors,'benchmark/problems_completed':len(rows),'checkpoint_step':a.checkpoint_step or 0}
   wb.log(metrics)
   wb.summary[f'{name}/pass@1']=summary['pass@1']; wb.summary[f'{name}/pass@{a.n}']=summary[f'pass@{a.n}']
  gc.collect()
 if wb is not None: wb.finish()
if __name__=='__main__': main()
