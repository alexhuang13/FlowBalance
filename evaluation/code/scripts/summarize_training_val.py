#!/usr/bin/env python3
"""Summarize legacy training validation dumps per benchmark; never mix datasets."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('val_jsonl');a=ap.parse_args();rows=[json.loads(x) for x in Path(a.val_jsonl).open() if x.strip()]
# Current training dataloader preserves validation file order: HE+ 164 then MBPP+ 378.
assert len(rows)==542,len(rows)
for name,rs in [('humanevalplus',rows[:164]),('mbppplus',rows[164:])]:
 print(json.dumps({'dataset':name,'problems':len(rs),'acc':sum(float(x['acc']) for x in rs)/len(rs),
  'score':sum(float(x['score']) for x in rs)/len(rs),'reward':sum(float(x['reward']) for x in rs)/len(rs)},ensure_ascii=False))
