#!/usr/bin/env python3
"""Aggregate semantic diversity records into CSV and Markdown summaries."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

METRICS = ("simpson", "k", "dominant_ratio", "normalized_entropy")


def read_records(root: Path) -> list[dict]:
    records = []
    for path in sorted(root.rglob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "all_metrics" in record and "correct_only_metrics" in record:
            records.append(record)
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--correct-only-min", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    records = read_records(args.input)
    if not records:
        raise SystemExit(f"No diversity records found under {args.input}")

    per_problem: list[dict] = []
    for record in records:
        for mode, key in (("all", "all_metrics"), ("correct_only", "correct_only_metrics")):
            metrics = record[key]
            per_problem.append(
                {
                    "method": record["method"],
                    "dataset": record.get("dataset", "unknown"),
                    "seed": record.get("seed", 0),
                    "problem_index": record["problem_index"],
                    "mode": mode,
                    "num_correct": record.get("num_correct", 0),
                    "n": metrics.get("n"),
                    "k": metrics.get("k"),
                    "dominant_ratio": metrics.get("dominant_ratio"),
                    "normalized_entropy": metrics.get("normalized_entropy", metrics.get("entropy_norm")),
                    "simpson": metrics.get("simpson"),
                }
            )
    write_csv(args.output / "per_problem_metrics.csv", per_problem)

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_problem:
        groups[(row["method"], row["dataset"], row["mode"])].append(row)

    summary = []
    for (method, dataset, mode), rows in sorted(groups.items()):
        minimum = args.correct_only_min if mode == "correct_only" else 1
        used = [row for row in rows if (row.get("n") or 0) >= minimum]
        item = {
            "method": method,
            "dataset": dataset,
            "mode": mode,
            "problems_total": len(rows),
            "problems_used": len(used),
            "coverage": len(used) / len(rows),
            "mean_correct_trajectories": mean([row["num_correct"] for row in rows]),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in used if row.get(metric) is not None]
            item[f"{metric}_mean"] = mean(values)
            item[f"{metric}_std_across_problems"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)
    write_csv(args.output / "summary_metrics.csv", summary)

    lines = [
        "# Semantic Strategy Diversity Summary",
        "",
        "Simpson diversity is defined as `1 - sum(p_k^2)` over semantic strategy clusters.",
        f"Correct-only rows require at least {args.correct_only_min} correct trajectories.",
        "",
        "| Method | Dataset | Mode | Simpson | Strategies | Dominant ratio | Entropy | Coverage |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['mode']} | "
            f"{row['simpson_mean']:.4f} | {row['k_mean']:.2f} | "
            f"{row['dominant_ratio_mean']:.4f} | {row['normalized_entropy_mean']:.4f} | "
            f"{row['problems_used']}/{row['problems_total']} ({100 * row['coverage']:.1f}%) |"
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
