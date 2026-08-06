#!/usr/bin/env python3
"""Deterministic, per-dataset EvalPlus evaluation with auditable raw outputs."""
from __future__ import annotations
import argparse,gc,importlib.util,json,os,time
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM,SamplingParams
ROOT=Path('/apdcephfs_gy4/share_303378103/user/audenhuang')

def args():
 p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--model-tag',required=True)
 p.add_argument('--data-root',default=str(ROOT/'data/code/evalplus_v2')); p.add_argument('--output-root',default=str(ROOT/'code_benchmark_eval/results'))
 p.add_argument('--datasets',nargs='+',default=['humanevalplus','mbppplus']); p.add_argument('--tp',type=int,default=1)
 p.add_argument('--seed',type=int,default=0); p.add_argument('--max-tokens',type=int,default=8192); p.add_argument('--gpu-memory-utilization',type=float,default=.8)
 p.add_argument('--temperature',type=float,default=0.0); p.add_argument('--top-p',type=float,default=1.0); p.add_argument('--start-idx',type=int,default=0); p.add_argument('--end-idx',type=int,default=-1)
 return p.parse_args()

def score_module():
 p=ROOT/'stable_rl/core/utils/reward_score/code_flowopsd_score.py'; s=importlib.util.spec_from_file_location('code_score',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def plain(x): return x.tolist() if hasattr(x,'tolist') else x

def main():
 a=args(); score=score_module(); out=Path(a.output_root)/a.model_tag; out.mkdir(parents=True,exist_ok=True)
 tok=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
 llm=LLM(model=a.model,tensor_parallel_size=a.tp,trust_remote_code=True,gpu_memory_utilization=a.gpu_memory_utilization,
  max_model_len=10240,enable_prefix_caching=True,seed=a.seed)
 sp=SamplingParams(n=1,temperature=a.temperature,top_p=a.top_p,max_tokens=a.max_tokens,seed=a.seed)
 all_summaries=[]
 for name in a.datasets:
  dsout=out/name; dsout.mkdir(parents=True,exist_ok=True); result=dsout/f'results_seed{a.seed}.jsonl'; summary=dsout/f'summary_seed{a.seed}.json'
  if result.exists() and summary.exists(): all_summaries.append(json.loads(summary.read_text())); continue
  df=pd.read_parquet(Path(a.data_root)/name/'test.parquet'); end=len(df) if a.end_idx<0 else min(a.end_idx,len(df)); df=df.iloc[a.start_idx:end].reset_index(drop=True)
  prompts=[tok.apply_chat_template(plain(x),tokenize=False,add_generation_prompt=True,enable_thinking=True) for x in df.prompt]
  t=time.time(); outputs=llm.generate(prompts,sp); elapsed=time.time()-t; rows=[]; passed=0; tokens=0
  for o,(_,r) in zip(outputs,df.iterrows()):
   x=o.outputs[0]; z=score.compute_score(x.text,r.reward_model['ground_truth'],plain(r.extra_info),r.data_source); passed+=int(z['acc']); tokens+=len(x.token_ids)
   rows.append({'dataset':name,'index':r.extra_info['index'],'prompt':plain(r.prompt)[-1]['content'],'response':x.text,
    'correct':bool(z['acc']),'score':z['score'],'test_errors':json.loads(z['test_errors']),'response_tokens':len(x.token_ids),
    'extracted_code':score.extract_code(x.text)})
  tmp=result.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)); os.replace(tmp,result)
  s={'model_tag':a.model_tag,'model_path':a.model,'dataset':name,'problems':len(rows),'pass@1':passed/len(rows) if rows else 0,
   'passed':passed,'temperature':a.temperature,'top_p':a.top_p,'do_sample':a.temperature>0,'seed':a.seed,'max_tokens':a.max_tokens,
   'avg_response_tokens':tokens/len(rows) if rows else 0,'elapsed_seconds':elapsed,'result_path':str(result),'protocol':'evalplus_v2_deterministic'}
  summary.write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n'); all_summaries.append(s); print('[summary]',json.dumps(s),flush=True); gc.collect()
 (out/'summary.json').write_text(json.dumps(all_summaries,indent=2,ensure_ascii=False)+'\n')

if __name__=='__main__':main()
