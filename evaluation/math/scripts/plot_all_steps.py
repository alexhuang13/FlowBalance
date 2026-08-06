#!/usr/bin/env python3
"""Plot every-step math benchmark results into one multi-panel figure."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BENCHMARKS = ['aime24','aime25','aime26','hmmt25','minerva_math','math500','olympiadbench']
ALGORITHMS = ['grpo','rlsd','flowopsd']
COLORS = {'grpo':'#1f77b4','rlsd':'#ff7f0e','flowopsd':'#2ca02c'}
LABELS = {'grpo':'GRPO','rlsd':'RLSD','flowopsd':'FlowOPSD'}
TITLES = {
 'aime24':'AIME 2024','aime25':'AIME 2025','aime26':'AIME 2026','hmmt25':'HMMT 2025',
 'minerva_math':'Minerva Math','math500':'MATH-500','olympiadbench':'OlympiadBench',
}

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument('--input',default='/apdcephfs_gy4/share_303378103/user/audenhuang/math_benchmark_eval/results/all_steps_summary.csv')
 p.add_argument('--output',default='/apdcephfs_gy4/share_303378103/user/audenhuang/math_benchmark_eval/results/all_steps_curves.png')
 p.add_argument('--pdf',default=None,help='Optional PDF output path')
 return p.parse_args()

def plot_lines(ax, frame):
 for alg in ALGORITHMS:
  x=frame[frame.algorithm.eq(alg)].sort_values('checkpoint_step')
  if x.empty: continue
  ax.plot(x.checkpoint_step,x['pass@1'],color=COLORS[alg],marker='o',markersize=3,
          linewidth=1.7,label=f'{LABELS[alg]} pass@1')
  ax.plot(x.checkpoint_step,x['pass@16'],color=COLORS[alg],marker='s',markersize=2.5,
          linewidth=1.5,linestyle='--',alpha=.9,label=f'{LABELS[alg]} pass@16')
 ax.set_ylim(-.02,1.02); ax.grid(True,alpha=.25); ax.set_xlabel('Checkpoint step'); ax.set_ylabel('Score')

def main():
 a=parse_args(); inp=Path(a.input); out=Path(a.output)
 if not inp.exists(): raise SystemExit(f'Input not found: {inp}')
 df=pd.read_csv(inp)
 required={'algorithm','checkpoint_step','dataset','pass@1','pass@16'}
 missing=required-set(df.columns)
 if missing: raise SystemExit(f'Missing columns: {sorted(missing)}')
 df=df[df.dataset.isin(BENCHMARKS)&df.algorithm.isin(ALGORITHMS)].copy()
 if df.empty: raise SystemExit('No completed all-step benchmark rows yet.')
 df.checkpoint_step=pd.to_numeric(df.checkpoint_step)
 fig,axes=plt.subplots(2,4,figsize=(21,10.5),sharey=True)
 axes=axes.ravel()
 for ax,name in zip(axes[:7],BENCHMARKS):
  plot_lines(ax,df[df.dataset.eq(name)])
  ax.set_title(TITLES[name],fontweight='bold')
 # Macro average uses only steps with available rows; incomplete steps are still shown transparently.
 macro=(df.groupby(['algorithm','checkpoint_step'],as_index=False)
          .agg(**{'pass@1':('pass@1','mean'),'pass@16':('pass@16','mean'),
                  'benchmarks_completed':('dataset','nunique')}))
 plot_lines(axes[7],macro)
 axes[7].set_title('Macro average (available benchmarks)',fontweight='bold')
 for alg in ALGORITHMS:
  x=macro[macro.algorithm.eq(alg)]
  for _,r in x[x.benchmarks_completed.lt(len(BENCHMARKS))].iterrows():
   axes[7].annotate(f"{int(r.benchmarks_completed)}/7",(r.checkpoint_step,r['pass@16']),
                    fontsize=6,color=COLORS[alg],xytext=(2,2),textcoords='offset points')
 handles=[]
 for alg in ALGORITHMS:
  handles += [
   Line2D([0],[0],color=COLORS[alg],marker='o',lw=1.8,label=f'{LABELS[alg]} pass@1'),
   Line2D([0],[0],color=COLORS[alg],marker='s',lw=1.5,ls='--',label=f'{LABELS[alg]} pass@16')]
 fig.legend(handles=handles,loc='lower center',ncol=6,frameon=False,bbox_to_anchor=(.5,.01))
 completed=df[['algorithm','checkpoint_step']].drop_duplicates().shape[0]
 fig.suptitle(f'Every-checkpoint Math Benchmark Evaluation  |  {len(df)} benchmark results, {completed} checkpoints represented',
              fontsize=16,fontweight='bold')
 fig.tight_layout(rect=(0,.055,1,.95)); out.parent.mkdir(parents=True,exist_ok=True)
 fig.savefig(out,dpi=180,bbox_inches='tight')
 if a.pdf: fig.savefig(a.pdf,bbox_inches='tight')
 print(out)
 if a.pdf: print(a.pdf)

if __name__=='__main__': main()
