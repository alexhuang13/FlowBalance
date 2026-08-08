#!/usr/bin/env python3
"""Selected experiment: 3 algorithms x steps {150,180} x seeds {0..4}.

For every seed, evaluate all seven benchmarks with n=1 and AIME24-26 with
n=16. Aggregate raw results and produce mean +/- sample std tables across seeds.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import ray

ROOT = Path('/apdcephfs_gy4/share_303378103/user/audenhuang')
WORK = ROOT / 'math_benchmark_eval'
STABLE = ROOT / 'stable_rl'
OUT = WORK / 'results_selected_steps_150_180_5seeds_t06_p095_k20_len38912_b16'
MERGED = WORK / 'merged_models/selected_steps_150_180_5seeds'
SEEDS = tuple(range(5))
NEW_SEEDS = SEEDS
ALL = ['aime24', 'aime25', 'aime26', 'hmmt25', 'minerva_math', 'math500', 'olympiadbench']
AIME = ['aime24', 'aime25', 'aime26']
RUNS = {
    'grpo': ROOT / 'output/verl-grpo/GRPO-Qwen3-8B-math-dapo17k-wandb-run34',
    'rlsd': ROOT / 'output/verl-rlsd-native/RLSD-Native-Qwen3-8B-math-dapo17k-run34',
    'flowsd': ROOT / 'output/verl-flowsd/FlowSD-Qwen3-8B-math-dapo17k-betaq1-grpo-signadv-frozenref-etaR15-lr1e6-run28',
}
RAW_FIELDS = [
    'algorithm', 'checkpoint_step', 'seed', 'setting', 'dataset', 'problems', 'n',
    'pass@1', 'pass@16', 'sample_correct', 'total_samples', 'avg_response_tokens',
    'elapsed_seconds', 'grade_errors', 'result_path',
]


def env():
    e = os.environ.copy()
    tc = str(STABLE / 'recipe/sdpo/toolchain')
    e.update({
        'TOKENIZERS_PARALLELISM': 'false', 'VLLM_USE_V1': '1',
        'VLLM_USE_FLASHINFER_SAMPLER': '0', 'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1', 'NCCL_ALGO': 'Ring', 'NCCL_IB_DISABLE': '1',
        'TORCH_NCCL_AVOID_RECORD_STREAMS': '1', 'TORCH_CUDA_ARCH_LIST': '9.0',
        'PYTHONPATH': f'{STABLE}:{STABLE}/verl:' + e.get('PYTHONPATH', ''),
        'CC': tc + '/gcc', 'CXX': tc + '/g++', 'TRITON_CC': tc + '/gcc',
        'CUDAHOSTCXX': tc + '/g++', 'PATH': tc + ':' + e.get('PATH', ''),
    })
    return e


@ray.remote(num_cpus=32)
def merge_one(algorithm, step):
    tag = f'{algorithm}_step_{step}'
    target = MERGED / tag
    tmp = target.with_name(tag + '.merging')
    actor = RUNS[algorithm] / f'global_step_{step}/actor'
    if (target / 'config.json').exists():
        return str(target)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        'python3', '-m', 'verl.model_merger', 'merge', '--backend', 'fsdp',
        '--local_dir', str(actor), '--target_dir', str(tmp), '--use_cpu_initialization',
    ], cwd=STABLE, env=env(), check=True)
    if not (tmp / 'config.json').exists():
        raise RuntimeError(f'merge incomplete {tag}')
    shutil.rmtree(target, ignore_errors=True)
    tmp.rename(target)
    return str(target)


@ray.remote(num_gpus=1, num_cpus=8, max_retries=1)
def evaluate_unit(algorithm, step, dataset, n, seed, model):
    tag = f'{algorithm}_step_{step}'
    setting = f'n{n}'
    output_root = OUT / f'seed_{seed}' / setting
    summary = output_root / tag / dataset / f'summary_s0_e-1_n{n}_seed{seed}.json'
    if summary.exists():
        return {'tag': tag, 'dataset': dataset, 'n': n, 'seed': seed, 'status': 'skip'}
    cmd = [
        'python3', str(WORK / 'scripts/eval_benchmarks.py'), '--model', model,
        '--model-tag', tag, '--data-root', str(WORK / 'data/processed'),
        '--output-root', str(output_root), '--datasets', dataset, '--n', str(n),
        '--seed', str(seed), '--temperature', '0.6', '--top-p', '0.95', '--top-k', '20',
        '--max-tokens', '38912', '--max-model-len', '40960', '--eval-batch-size', '16', '--tp', '1', '--gpu-memory-utilization', '0.80',
        '--wandb-run-name', f'math-eval-selected-5seed-{tag}-{dataset}-n{n}-seed{seed}',
        '--wandb-group', f'selected-5seed-{algorithm}-step-{step}',
        '--checkpoint-step', str(step),
    ]
    subprocess.run(cmd, cwd=WORK, env=env(), check=True)
    if not summary.exists():
        raise RuntimeError(f'missing {summary}')
    return {'tag': tag, 'dataset': dataset, 'n': n, 'seed': seed, 'status': 'complete'}



def collect_rows():
    rows = []
    for p in OUT.glob('seed_*/*/*/*/summary_s0_e-1_n*_seed*.json'):
        # seed_X / nX / tag / dataset / summary.json
        seed_dir, setting, tag, dataset = p.parents[3].name, p.parents[2].name, p.parents[1].name, p.parent.name
        try:
            seed = int(seed_dir.removeprefix('seed_'))
            algorithm, step = tag.rsplit('_step_', 1)
            step = int(step)
        except ValueError:
            continue
        if algorithm not in RUNS or step not in (150, 180) or seed not in SEEDS:
            continue
        d = json.loads(p.read_text())
        d.update(algorithm=algorithm, checkpoint_step=step, seed=seed, setting=setting, dataset=dataset)
        rows.append(d)
    rows.sort(key=lambda d: (d['algorithm'], d['checkpoint_step'], d['seed'], d['setting'], d['dataset']))
    return rows


def write_markdown_table(path, title, table):
    display = table.copy()
    display.index.name = 'Model-Step'
    cols = ['Model-Step'] + list(display.columns)
    body = [[str(idx)] + [str(v) for v in row] for idx, row in display.iterrows()]
    widths = [max(len(cols[j]), max((len(r[j]) for r in body), default=0)) for j in range(len(cols))]
    line = lambda row: '| ' + ' | '.join(str(v).ljust(widths[j]) for j, v in enumerate(row)) + ' |'
    text = [f'## {title}', '', line(cols), '| ' + ' | '.join('-' * w for w in widths) + ' |']
    text.extend(line(row) for row in body)
    return '\n'.join(text)


def formatted_table(stats, setting, metric, datasets):
    x = stats[(stats.setting == setting) & (stats.metric == metric) & stats.dataset.isin(datasets)].copy()
    x['model_step'] = x.algorithm.str.upper() + '-' + x.checkpoint_step.astype(str)
    x['value'] = x.apply(lambda r: f"{100*r['mean']:.2f} +/- {100*r['std']:.2f}", axis=1)
    order = [a.upper() + '-' + str(s) for a in RUNS for s in (150, 180)]
    table = x.pivot(index='model_step', columns='dataset', values='value').reindex(order).reindex(columns=datasets)
    # Macro average is computed independently within each seed, then summarized across five seeds.
    raw = pd.DataFrame(collect_rows())
    metric_col = metric
    sub = raw[(raw.setting == setting) & raw.dataset.isin(datasets)].copy()
    macro = sub.groupby(['algorithm', 'checkpoint_step', 'seed'])[metric_col].mean().reset_index()
    macro = macro.groupby(['algorithm', 'checkpoint_step'])[metric_col].agg(['mean', 'std']).reset_index()
    macro['model_step'] = macro.algorithm.str.upper() + '-' + macro.checkpoint_step.astype(str)
    macro['value'] = macro.apply(lambda r: f"{100*r['mean']:.2f} +/- {100*r['std']:.2f}", axis=1)
    table['Macro Avg'] = macro.set_index('model_step')['value'].reindex(order)
    return table


def aggregate():
    rows = collect_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / 'summary_all_seeds.csv'
    tmp = raw_path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        w.writeheader()
        for d in rows:
            w.writerow({k: d.get(k) for k in RAW_FIELDS})
    tmp.replace(raw_path)
    print('[aggregate]', len(rows), 'of', 300, flush=True)
    if not rows:
        return

    df = pd.DataFrame(rows)
    long = []
    for (algorithm, step, setting, dataset), g in df.groupby(['algorithm', 'checkpoint_step', 'setting', 'dataset']):
        for metric in ('pass@1', 'pass@16'):
            # Some settings do not define every metric (for example, n=1
            # summaries have pass@1 but no pass@16). Treat an absent metric
            # as unavailable for this group instead of aborting aggregation.
            if metric not in g.columns:
                continue
            vals = pd.to_numeric(g[metric], errors='coerce').dropna()
            if len(vals):
                long.append({
                    'algorithm': algorithm, 'checkpoint_step': step, 'setting': setting,
                    'dataset': dataset, 'metric': metric, 'seeds': len(vals),
                    'mean': vals.mean(), 'std': vals.std(ddof=1) if len(vals) > 1 else float('nan'),
                })
    stats = pd.DataFrame(long)
    stats.to_csv(OUT / 'mean_std_long.csv', index=False)

    # Only publish final formatted tables once every cell has five seeds.
    expected_cells = 6 * (7 + 3)
    complete = len(stats.groupby(['algorithm', 'checkpoint_step', 'setting', 'dataset'])) == expected_cells
    complete = complete and (stats.seeds.min() == 5)
    if not complete:
        return
    pass1 = formatted_table(stats, 'n1', 'pass@1', ALL)
    pass16 = formatted_table(stats, 'n16', 'pass@16', AIME)
    sample1 = formatted_table(stats, 'n16', 'pass@1', AIME)
    pass1.to_csv(OUT / 'pass1_n1_mean_std_percent.csv')
    pass16.to_csv(OUT / 'aime_pass16_n16_mean_std_percent.csv')
    sample1.to_csv(OUT / 'aime_sample_pass1_n16_mean_std_percent.csv')
    md = [
        '# Selected Steps 150/180 — Five-Seed Results', '',
        'Seeds: 0, 1, 2, 3, 4. Values are mean +/- sample standard deviation across seeds, in percent.', '',
        write_markdown_table(OUT / 'results_mean_std.md', 'Pass@1 (n=1, %)', pass1), '',
        write_markdown_table(OUT / 'results_mean_std.md', 'AIME Pass@16 (n=16, %)', pass16), '',
        write_markdown_table(OUT / 'results_mean_std.md', 'AIME sample-level Pass@1 (n=16, %)', sample1), '',
    ]
    (OUT / 'results_mean_std.md').write_text('\n'.join(md))


def main():
    ray.init(address='auto', ignore_reinit_error=True)
    print('[cluster]', ray.cluster_resources(), flush=True)
    specs = [(a, s) for a in RUNS for s in (150, 180)]
    merge_refs = {x: merge_one.remote(*x) for x in specs}
    models = {x: ray.get(ref) for x, ref in merge_refs.items()}
    print('[merge] all six models ready', models, flush=True)
    units = []
    for a, s in specs:
        for seed in NEW_SEEDS:
            units.extend((a, s, d, 1, seed, models[(a, s)]) for d in ALL)
            units.extend((a, s, d, 16, seed, models[(a, s)]) for d in AIME)
    print('[plan] new units', len(units), 'seeds', NEW_SEEDS, 'max_parallel_gpu=32', flush=True)
    refs = {evaluate_unit.remote(*u): u for u in units}
    done = 0
    while refs:
        ready, _ = ray.wait(list(refs), num_returns=1, timeout=60)
        if not ready:
            print('[progress]', done, '/', len(units), flush=True)
            continue
        ref = ready[0]
        spec = refs.pop(ref)
        try:
            print('[result]', ray.get(ref), flush=True)
        except Exception as exc:
            print('[failed]', spec, repr(exc), flush=True)
            raise
        done += 1
    print('[evaluation done] raw results are complete; starting aggregation', flush=True)
    aggregate()
    for model in models.values():
        shutil.rmtree(model, ignore_errors=True)
    print('[done] selected five-seed experiment', flush=True)


if __name__ == '__main__':
    main()
