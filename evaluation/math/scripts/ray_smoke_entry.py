#!/usr/bin/env python3
"""Recipe-style one-GPU/TP=1 benchmark smoke test launched as a Ray job."""
import os
import subprocess
from pathlib import Path
import ray

ROOT=Path('/apdcephfs_gy4/share_303378103/user/audenhuang')
WORK=ROOT/'math_benchmark_eval'

@ray.remote(num_gpus=1, num_cpus=4)
def run_smoke():
    print('[ray smoke] CUDA_VISIBLE_DEVICES=',os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
    print('[ray smoke] NCCL_ALGO=',os.environ.get('NCCL_ALGO'),flush=True)
    print('[ray smoke] VLLM_USE_FLASHINFER_SAMPLER=',os.environ.get('VLLM_USE_FLASHINFER_SAMPLER'),flush=True)
    cmd=[
      'python3',str(WORK/'scripts/eval_benchmarks.py'),
      '--model',str(WORK/'merged_models/grpo_final'),
      '--model-tag','recipe_smoke_grpo_final_tp1',
      '--data-root',str(WORK/'data/processed'),
      '--output-root',str(WORK/'results'),
      '--datasets','aime24','--n','1','--start-idx','0','--end-idx','1',
      '--tp','1','--enforce-eager','--gpu-memory-utilization','0.80',
    ]
    print('[ray smoke] exec:', ' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True,cwd=WORK)
    return 'ok'

def main():
    ray.init(address='auto',ignore_reinit_error=True)
    print(ray.get(run_smoke.remote()))
if __name__=='__main__': main()
