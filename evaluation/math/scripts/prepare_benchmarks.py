#!/usr/bin/env python3
"""Convert the confirmed math benchmarks to a common verl-compatible parquet schema.

This script is deterministic and offline: it only reads pinned raw snapshots already
under math_benchmark_eval/data/raw and writes one row per unique benchmark problem.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MANIFESTS = ROOT / "manifests"

BOXED_INSTRUCTION = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)


def scalar(v: Any) -> Any:
    """Turn numpy/list-like one-element fields into plain Python scalars."""
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, (list, tuple)) and len(v) == 1:
        return scalar(v[0])
    if isinstance(v, np.generic):
        return v.item()
    return v


def clean_answer(v: Any) -> str:
    v = scalar(v)
    if v is None:
        raise ValueError("ground truth is None")
    if isinstance(v, float) and math.isnan(v):
        raise ValueError("ground truth is NaN")
    s = str(v).strip()
    # Dataset answers are sometimes wrapped in display math dollars. Keep LaTeX,
    # but remove only balanced outer dollar delimiters.
    if len(s) >= 2 and s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if not s:
        raise ValueError("empty ground truth")
    return s


def prompt(problem: str) -> list[dict[str, str]]:
    p = str(problem).strip()
    if not p:
        raise ValueError("empty problem")
    # The raw AIME24 prompt has an Answer: wrapper; all processed datasets use one
    # shared instruction so prompting is comparable across benchmarks.
    p = re.sub(
        r"^Solve the following math problem step by step\.\s*"
        r"The last line of your response should be of the form Answer: \$Answer "
        r"\(without quotes\) where \$Answer is the answer to the problem\.\s*",
        "",
        p,
    )
    p = re.sub(r'\s*Remember to put your answer on its own line after "Answer:"\.?\s*$', "", p)
    p = re.sub(
        r"\s*Please reason step by step, and put your final answer within \\boxed\{\}\.\s*$",
        "",
        p,
    )
    return [{"role": "user", "content": p.strip() + BOXED_INSTRUCTION}]


def record(source: str, problem: str, answer: Any, uid: str, **meta: Any) -> dict[str, Any]:
    extra = {"index": str(uid), "split": "test", **meta}
    # Avoid parquet null-only nested fields and normalize numpy values.
    extra = {k: scalar(v) for k, v in extra.items() if v is not None and not (isinstance(v, float) and math.isnan(v))}
    return {
        "data_source": source,
        "prompt": prompt(problem),
        "ability": "MATH",
        "reward_model": {
            "ground_truth": clean_answer(answer),
            "style": "rule-lighteval/MATH_v2",
        },
        "extra_info": extra,
    }


def load_aime24() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = pd.read_parquet(RAW / "aime24/data/aime-2024.parquet")
    d["_index"] = d.extra_info.apply(lambda x: int(x["index"]))
    unique = d.drop_duplicates("_index").sort_values("_index")
    rows = []
    for _, r in unique.iterrows():
        ei = r.extra_info
        rows.append(record("aime24", ei["raw_problem"], r.reward_model["ground_truth"], f"aime24-{int(ei['index']):02d}", competition="AIME 2024"))
    return rows, {"raw_rows": len(d), "dedup_key": "extra_info.index"}


def load_aime25() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for division, fn in [("I", "aime2025-I.jsonl"), ("II", "aime2025-II.jsonl")]:
        d = pd.read_json(RAW / f"aime25_opencompass/{fn}", lines=True)
        for i, r in d.iterrows():
            rows.append(record("aime25", r.question, r.answer, f"aime25-{division}-{i+1:02d}", competition=f"AIME 2025 {division}", problem_number=i+1))
    return rows, {"raw_rows": len(rows), "source_configs": ["AIME2025-I", "AIME2025-II"]}


def load_aime26() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = pd.read_parquet(RAW / "aime26_matharena/data/train-00000-of-00001.parquet")
    rows = []
    # MathArena uses 1..30: 1..15 = I and 16..30 = II.
    for _, r in d.sort_values("problem_idx").iterrows():
        idx = int(r.problem_idx)
        division = "I" if idx <= 15 else "II"
        number = idx if idx <= 15 else idx - 15
        rows.append(record("aime26", r.problem, r.answer, f"aime26-{division}-{number:02d}", competition=f"AIME 2026 {division}", problem_number=number, source_problem_idx=idx))
    return rows, {"raw_rows": len(d), "division_mapping": "problem_idx 1-15 => I; 16-30 => II"}


def load_hmmt25() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = pd.read_json(RAW / "hmmt25_flageval/hmmt_25.jsonl", lines=True)
    rows = []
    for _, r in d.sort_values("id").iterrows():
        rows.append(record("hmmt25", r.question, r.answer, f"hmmt25-{int(r.id):02d}", competition="HMMT February 2025", problem_number=int(r.id)))
    return rows, {"raw_rows": len(d), "cross_checked_against": "PraMamba/HMMT-202502 (30/30 normalized prompts overlap)"}


def load_minerva() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = pd.read_json(RAW / "minerva_mathai/test.jsonl", lines=True)
    rows = []
    for i, r in d.iterrows():
        rows.append(record("minerva_math", r.question, r.answer, f"minerva-math-{i:03d}", problem_number=i))
    return rows, {"raw_rows": len(d), "cross_checked_against": "svc-huggingface/minerva-math (272/272 normalized prompts overlap)"}


def load_math500() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = pd.read_json(RAW / "math500_h4/test.jsonl", lines=True)
    rows = []
    for i, r in d.iterrows():
        rows.append(record("math500", r.problem, r.answer, f"math500-{i:03d}", unique_id=r.unique_id, subject=r.subject, level=int(r.level)))
    return rows, {"raw_rows": len(d)}


def load_olympiad() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    d = pd.read_parquet(RAW / "olympiad_mathai/test.parquet")
    rows = []
    for _, r in d.sort_values("id").iterrows():
        ans = clean_answer(r.final_answer)
        rows.append(record(
            "olympiadbench", r.question, ans, f"olympiadbench-{int(r.id)}",
            source_id=int(r.id), answer_type=str(r.answer_type),
            is_multiple_answer=bool(r.is_multiple_answer),
            tolerance=None if pd.isna(r.error) else str(r.error),
            modality=str(r.modality), language=str(r.language),
        ))
    return rows, {
        "raw_rows": len(d),
        "text_only_rows": int((d.modality == "Text-only").sum()),
        "multiple_answer_rows": int(d.is_multiple_answer.sum()),
        "tolerance_rows": int(d.error.notna().sum()),
        "answer_type_counts": {str(k): int(v) for k, v in d.answer_type.value_counts().items()},
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_problem(row: dict[str, Any]) -> str:
    s = row["prompt"][0]["content"].replace(BOXED_INSTRUCTION, "")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def validate(name: str, rows: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    assert len(rows) == expected, (name, len(rows), expected)
    ids = [r["extra_info"]["index"] for r in rows]
    probs = [normalized_problem(r) for r in rows]
    answers = [r["reward_model"]["ground_truth"] for r in rows]
    assert len(ids) == len(set(ids)), f"{name}: duplicate IDs"
    assert all(probs), f"{name}: empty problem"
    assert all(answers), f"{name}: empty answer"
    duplicate_prompts = len(probs) - len(set(probs))
    return {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "unique_normalized_prompts": len(set(probs)),
        "duplicate_normalized_prompts": duplicate_prompts,
        "empty_answers": sum(not x for x in answers),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    loaders = {
        "aime24": (load_aime24, 30),
        "aime25": (load_aime25, 30),
        "aime26": (load_aime26, 30),
        "hmmt25": (load_hmmt25, 30),
        "minerva_math": (load_minerva, 272),
        "math500": (load_math500, 500),
        "olympiadbench": (load_olympiad, 674),
    }
    manifest: dict[str, Any] = {"schema_version": 1, "prompt_suffix": BOXED_INSTRUCTION, "datasets": {}}
    all_rows = []
    for name, (loader, expected) in loaders.items():
        rows, source_stats = loader()
        checks = validate(name, rows, expected)
        df = pd.DataFrame(rows)
        path = OUT / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["datasets"][name] = {
            "path": str(path), "sha256": sha256(path),
            "source_stats": source_stats, "checks": checks,
        }
        all_rows.extend(rows)
        print(f"[ok] {name}: {len(rows)} rows -> {path}")

    # Cross-dataset exact normalized-prompt overlaps are contamination/reuse signals.
    owners: dict[str, list[str]] = {}
    for r in all_rows:
        owners.setdefault(normalized_problem(r), []).append(r["data_source"] + ":" + r["extra_info"]["index"])
    overlaps = [v for v in owners.values() if len({x.split(':', 1)[0] for x in v}) > 1]
    manifest["cross_dataset_exact_prompt_overlaps"] = overlaps
    out_manifest = MANIFESTS / "processed_datasets.json"
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"[ok] manifest: {out_manifest}")
    print(f"[info] cross-dataset normalized exact overlaps: {len(overlaps)}")


if __name__ == "__main__":
    main()
