#!/usr/bin/env python3
"""Post-checkpoint benchmark validation for one FlowSD experiment.

Wait for a complete FSDP actor checkpoint, merge it to HuggingFace format, then
run the established five-seed validation protocol:
  * n=1 on all seven math benchmarks
  * n=16 on AIME 2024/2025/2026
Results are resumable at dataset/seed granularity and aggregated as mean +/-
sample standard deviation across seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import ray

ROOT = Path('/apdcephfs_gy4/share_303378103/user/audenhuang')
WORK = ROOT / 'math_benchmark_eval'
STABLE = ROOT / 'stable_rl'
ALL = ['aime24', 'aime25', 'aime26', 'hmmt25', 'minerva_math', 'math500', 'olympiadbench']
AIME = ['aime24', 'aime25', 'aime26']
RAW_FIELDS = [
    'algorithm', 'experiment_name', 'checkpoint_step', 'seed', 'setting', 'dataset',
    'problems', 'n', 'pass@1', 'pass@16', 'sample_correct', 'total_samples',
    'avg_response_tokens', 'elapsed_seconds', 'grade_errors', 'result_path',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--experiment-name', required=True)
    p.add_argument('--algorithm', default='flowsd')
    p.add_argument('--model-tag-prefix', default='flowsd')
    p.add_argument('--step', type=int, default=180)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--poll-seconds', type=int, default=60)
    p.add_argument('--timeout-seconds', type=int, default=7 * 24 * 3600)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--temperature', type=float, default=0.6)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument('--max-tokens', type=int, default=38912)
    p.add_argument('--max-model-len', type=int, default=40960)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.80)
    p.add_argument('--keep-merged-model', action='store_true')
    return p.parse_args()


def runtime_env():
    e = os.environ.copy()
    tc = str(STABLE / 'recipe/sdpo/toolchain')
    e.update({
        'TOKENIZERS_PARALLELISM': 'false', 'VLLM_USE_V1': '1',
        'VLLM_USE_FLASHINFER_SAMPLER': '0', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'NCCL_ALGO': 'Ring', 'NCCL_IB_DISABLE': '1',
        'TORCH_NCCL_AVOID_RECORD_STREAMS': '1', 'TORCH_CUDA_ARCH_LIST': '9.0',
        'PYTHONPATH': f'{STABLE}:{STABLE / "verl"}:' + e.get('PYTHONPATH', ''),
        'CC': tc + '/gcc', 'CXX': tc + '/g++', 'TRITON_CC': tc + '/gcc',
        'CUDAHOSTCXX': tc + '/g++', 'PATH': tc + ':' + e.get('PATH', ''),
        'LIBRARY_PATH': '/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12:' + e.get('LIBRARY_PATH', ''),
        'LD_LIBRARY_PATH': '/usr/lib64:/usr/lib/gcc/x86_64-TencentOS-linux/12:' + e.get('LD_LIBRARY_PATH', ''),
        'COMPILER_PATH': '/usr/libexec/gcc/x86_64-TencentOS-linux/12',
    })
    return e


def checkpoint_complete(step_dir: Path) -> tuple[bool, str]:
    actor = step_dir / 'actor'
    if not actor.is_dir():
        return False, 'actor directory missing'
    if not (step_dir / 'data.pt').is_file():
        return False, 'data.pt missing'
    if not (actor / 'fsdp_config.json').is_file():
        return False, 'fsdp_config.json missing'
    rank0 = list(actor.glob('model_world_size_*_rank_0.pt'))
    if len(rank0) != 1:
        return False, f'expected one rank-0 model shard, found {len(rank0)}'
    m = re.fullmatch(r'model_world_size_(\d+)_rank_0\.pt', rank0[0].name)
    if not m:
        return False, f'cannot parse world size from {rank0[0].name}'
    world_size = int(m.group(1))
    shards = list(actor.glob(f'model_world_size_{world_size}_rank_*.pt'))
    if len(shards) != world_size:
        return False, f'model shards {len(shards)}/{world_size}'
    return True, f'complete ({world_size} shards)'


def wait_for_checkpoint(step_dir: Path, poll: int, timeout: int):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        ok, detail = checkpoint_complete(step_dir)
        if ok:
            print(f'[checkpoint ready] {step_dir}: {detail}', flush=True)
            # Avoid racing the final rename/fsync of checkpoint side files.
            time.sleep(min(30, poll))
            return
        if detail != last:
            print(f'[checkpoint wait] {step_dir}: {detail}', flush=True)
            last = detail
        time.sleep(poll)
    raise TimeoutError(f'timed out after {timeout}s waiting for {step_dir}: {last}')


def merge_checkpoint(actor: Path, target: Path):
    if (target / 'config.json').exists() and (list(target.glob('*.safetensors')) or (target / 'model.safetensors.index.json').exists()):
        print(f'[merge resume] {target}', flush=True)
        return
    tmp = target.with_name(target.name + '.merging')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(target, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'python3', '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
        '--local_dir', str(actor), '--target_dir', str(tmp), '--use_cpu_initialization',
    ], cwd=STABLE, env=runtime_env(), check=True)
    if not (tmp / 'config.json').exists():
        raise RuntimeError(f'merge did not produce config.json: {tmp}')
    tmp.rename(target)
    print(f'[merge done] {target}', flush=True)


@ray.remote(num_gpus=1, num_cpus=8, max_retries=1)
def evaluate_unit(spec: dict):
    output_root = Path(spec['output_dir']) / f"seed_{spec['seed']}" / f"n{spec['n']}"
    summary = output_root / spec['model_tag'] / spec['dataset'] / f"summary_s0_e-1_n{spec['n']}_seed{spec['seed']}.json"
    if summary.exists():
        return {'status': 'skip', 'summary': str(summary)}
    cmd = [
        'python3', str(WORK / 'scripts/eval_benchmarks.py'),
        '--model', spec['model'], '--model-tag', spec['model_tag'],
        '--data-root', str(WORK / 'data/processed'), '--output-root', str(output_root),
        '--datasets', spec['dataset'], '--n', str(spec['n']), '--seed', str(spec['seed']),
        '--temperature', str(spec['temperature']), '--top-p', str(spec['top_p']),
        '--top-k', str(spec['top_k']), '--max-tokens', str(spec['max_tokens']),
        '--max-model-len', str(spec['max_model_len']), '--eval-batch-size', str(spec['eval_batch_size']),
        '--tp', '1', '--gpu-memory-utilization', str(spec['gpu_memory_utilization']),
        '--wandb-run-name', f"{spec['experiment_name']}-val-step{spec['step']}-{spec['dataset']}-n{spec['n']}-seed{spec['seed']}",
        '--wandb-group', f"{spec['experiment_name']}-step{spec['step']}-val",
        '--checkpoint-step', str(spec['step']),
    ]
    subprocess.run(cmd, cwd=WORK, env=runtime_env(), check=True)
    if not summary.exists():
        raise RuntimeError(f'missing summary after evaluation: {summary}')
    return {'status': 'complete', 'summary': str(summary)}


def collect_rows(out: Path, model_tag: str, experiment: str, algorithm: str, step: int, seeds: list[int]):
    rows = []
    for p in out.glob('seed_*/*/*/*/summary_s0_e-1_n*_seed*.json'):
        seed_dir, setting, tag, dataset = p.parents[3].name, p.parents[2].name, p.parents[1].name, p.parent.name
        if tag != model_tag:
            continue
        try:
            seed = int(seed_dir.removeprefix('seed_'))
        except ValueError:
            continue
        if seed not in seeds:
            continue
        d = json.loads(p.read_text())
        d.update(algorithm=algorithm, experiment_name=experiment, checkpoint_step=step,
                 seed=seed, setting=setting, dataset=dataset)
        rows.append(d)
    rows.sort(key=lambda d: (d['seed'], d['setting'], d['dataset']))
    return rows


def markdown_table(title: str, table: pd.DataFrame):
    display = table.copy(); display.index.name = 'Model-Step'
    cols = ['Model-Step'] + list(display.columns)
    body = [[str(idx)] + [str(v) for v in row] for idx, row in display.iterrows()]
    widths = [max(len(cols[j]), max((len(r[j]) for r in body), default=0)) for j in range(len(cols))]
    line = lambda row: '| ' + ' | '.join(str(v).ljust(widths[j]) for j, v in enumerate(row)) + ' |'
    text = [f'## {title}', '', line(cols), '| ' + ' | '.join('-' * w for w in widths) + ' |']
    text.extend(line(row) for row in body)
    return '\n'.join(text)


def make_table(df: pd.DataFrame, setting: str, metric: str, datasets: list[str], label: str):
    sub = df[(df.setting == setting) & df.dataset.isin(datasets)].copy()
    stats = sub.groupby('dataset')[metric].agg(['mean', 'std']).reindex(datasets)
    row = {d: f"{100*r['mean']:.2f} +/- {100*r['std']:.2f}" for d, r in stats.iterrows()}
    per_seed_macro = sub.groupby('seed')[metric].mean()
    row['Macro Avg'] = f'{100*per_seed_macro.mean():.2f} +/- {100*per_seed_macro.std(ddof=1):.2f}'
    return pd.DataFrame([row], index=[label])


def aggregate(out: Path, model_tag: str, experiment: str, algorithm: str, step: int, seeds: list[int]):
    rows = collect_rows(out, model_tag, experiment, algorithm, step, seeds)
    raw = out / 'summary_all_seeds.csv'; tmp = raw.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS); w.writeheader()
        for d in rows: w.writerow({k: d.get(k) for k in RAW_FIELDS})
    tmp.replace(raw)
    expected = len(seeds) * (len(ALL) + len(AIME))
    if len(rows) != expected:
        raise RuntimeError(f'incomplete validation rows: {len(rows)}/{expected}')
    df = pd.DataFrame(rows)
    label = f'{algorithm.upper()}-{step}'
    pass1 = make_table(df, 'n1', 'pass@1', ALL, label)
    pass16 = make_table(df, 'n16', 'pass@16', AIME, label)
    sample1 = make_table(df, 'n16', 'pass@1', AIME, label)
    pass1.to_csv(out / 'pass1_n1_mean_std_percent.csv')
    pass16.to_csv(out / 'aime_pass16_n16_mean_std_percent.csv')
    sample1.to_csv(out / 'aime_sample_pass1_n16_mean_std_percent.csv')
    md = [
        f'# {experiment} — Step {step} Validation', '',
        f'Seeds: {", ".join(map(str, seeds))}. Values are mean +/- sample standard deviation across seeds, in percent.', '',
        markdown_table('Pass@1 (n=1, %)', pass1), '',
        markdown_table('AIME Pass@16 (n=16, %)', pass16), '',
        markdown_table('AIME sample-level Pass@1 (n=16, %)', sample1), '',
    ]
    (out / 'results_mean_std.md').write_text('\n'.join(md))
    (out / '.complete').write_text(json.dumps({'experiment': experiment, 'step': step, 'rows': len(rows)}) + '\n')
    print(f'[aggregate done] {out / "results_mean_std.md"}', flush=True)


def main():
    a = parse_args()
    step_dir = a.run_dir / f'global_step_{a.step}'
    out = a.output_dir or (a.run_dir / 'val' / f'step_{a.step}_math_benchmarks_5seeds')
    out.mkdir(parents=True, exist_ok=True)
    model_tag = f'{a.model_tag_prefix}_step_{a.step}'
    merged = out / 'merged_model'
    if (out / '.complete').exists():
        print(f'[already complete] {out}', flush=True)
        return
    wait_for_checkpoint(step_dir, a.poll_seconds, a.timeout_seconds)
    merge_checkpoint(step_dir / 'actor', merged)

    ray.init(address='auto', ignore_reinit_error=True)
    print('[cluster]', ray.cluster_resources(), flush=True)
    base = dict(output_dir=str(out), model=str(merged), model_tag=model_tag,
                experiment_name=a.experiment_name, step=a.step,
                temperature=a.temperature, top_p=a.top_p, top_k=a.top_k,
                max_tokens=a.max_tokens, max_model_len=a.max_model_len,
                eval_batch_size=a.eval_batch_size,
                gpu_memory_utilization=a.gpu_memory_utilization)
    specs = []
    for seed in a.seeds:
        specs.extend(dict(base, seed=seed, n=1, dataset=d) for d in ALL)
        specs.extend(dict(base, seed=seed, n=16, dataset=d) for d in AIME)
    refs = {evaluate_unit.remote(s): s for s in specs}
    done = 0
    while refs:
        ready, _ = ray.wait(list(refs), num_returns=1, timeout=60)
        if not ready:
            print(f'[progress] {done}/{len(specs)}; waiting for GPU resources or evaluation', flush=True)
            continue
        ref = ready[0]; spec = refs.pop(ref)
        try:
            print('[result]', ray.get(ref), flush=True)
        except Exception:
            print('[failed spec]', spec, flush=True)
            raise
        done += 1
    aggregate(out, model_tag, a.experiment_name, a.algorithm, a.step, a.seeds)
    if not a.keep_merged_model:
        shutil.rmtree(merged, ignore_errors=True)
        print(f'[cleanup] removed {merged}', flush=True)


if __name__ == '__main__':
    main()
