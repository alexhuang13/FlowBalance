#!/usr/bin/env python3
"""Cluster sampled reasoning traces with an OpenAI-compatible judge.

Expected input is one JSONL row per problem. Each row must contain ``responses``
and may contain ``question``, ``index``, and ``correctness``. The command writes
one JSON record per problem and method. It is resumable: existing output records
are skipped unless ``--overwrite`` is set.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from urllib import error, request

from analysis.diversity.metrics import diversity_metrics

SYSTEM_PROMPT = """You are a rigorous evaluator of semantic strategy diversity in mathematical reasoning. Cluster attempts by the central mathematical representation, theorem, or decomposition actually used. Merge attempts that differ only in wording, notation, verbosity, arithmetic detail, or the order of equivalent steps. Split attempts when their core tools or representations are substantively different. Do not repair incorrect attempts. An incorrect but coherent attempt still has a strategy. Return valid JSON only."""


def parse_method(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--method must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--method must use NAME=PATH")
    return name, path


def compact(text: str, limit: int) -> str:
    text = text.strip()
    if "</think>" in text:
        visible = text.split("</think>", 1)[1].strip()
        if len(visible) >= 350:
            text = visible
    text = re.sub(r"\n{3,}", "\n\n", text)
    if limit <= 0 or len(text) <= limit:
        return text
    head = max(1, int(limit * 0.7))
    return text[:head] + "\n...[middle truncated]...\n" + text[-(limit - head) :]


def build_prompt(question: str, attempts: list[tuple[int, str]]) -> str:
    blocks = "\n\n".join(f"### Attempt {idx}\n{text}" for idx, text in attempts)
    return f"""Problem:
{question}

Anonymous sampled attempts:

{blocks}

Assign every attempt to exactly one strategy cluster. Mark an attempt invalid only if it is empty, a pure answer with no identifiable method, or incoherent noise. Return exactly one JSON object with this schema:
{{
  "assignments": [{{"attempt_id": 1, "cluster_id": 1, "valid": true}}],
  "clusters": [{{"cluster_id": 1, "name": "short strategy name", "description": "key representation or tools"}}],
  "invalid_attempt_ids": [],
  "notes": "brief note on borderline distinctions"
}}
There must be one assignment for every attempt_id from 1 to {len(attempts)}."""


def parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        candidate = re.sub(r'\\(?![\\"/bfnrtu])', r"\\\\", match.group(0))
        return json.loads(candidate)


def call_judge(prompt: str, args: argparse.Namespace) -> dict:
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    model = args.model or os.environ.get("OPENAI_MODEL")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or pass --api-key")
    if not model:
        raise RuntimeError("Set OPENAI_MODEL or pass --model")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    endpoint = base_url.rstrip("/") + "/chat/completions"
    last_error: Exception | None = None
    for attempt in range(args.retries):
        try:
            req = request.Request(
                endpoint,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with request.urlopen(req, timeout=args.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:1200]
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        if attempt + 1 < args.retries:
            time.sleep(min(2**attempt, 20))
    assert last_error is not None
    raise last_error


def find_jsonl(path_pattern: str, dataset: str, seed: int) -> Path:
    rendered = path_pattern.format(dataset=dataset, seed=seed)
    matches = sorted(Path().glob(rendered) if not Path(rendered).is_absolute() else Path(rendered).parent.glob(Path(rendered).name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one JSONL file for {rendered!r}, found {matches}")
    return matches[0]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_assignments(result: dict, n: int) -> None:
    assignments = result.get("assignments")
    clusters = result.get("clusters")
    if not isinstance(assignments, list) or not isinstance(clusters, list):
        raise ValueError("Judge output is missing assignments or clusters")
    attempt_ids = sorted(int(item["attempt_id"]) for item in assignments)
    if attempt_ids != list(range(1, n + 1)):
        raise ValueError(f"Unexpected attempt IDs: {attempt_ids}")
    cluster_ids = {int(item["cluster_id"]) for item in clusters}
    for item in assignments:
        if bool(item.get("valid", True)) and int(item["cluster_id"]) not in cluster_ids:
            raise ValueError(f"Unknown cluster ID in assignment: {item}")


def evaluate_task(task: tuple, args: argparse.Namespace) -> Path:
    method, pattern, dataset, seed, row_pos = task
    source = find_jsonl(pattern, dataset, seed)
    row = load_rows(source)[row_pos]
    responses = list(row["responses"])
    correctness = list(row.get("correctness", [False] * len(responses)))
    problem_id = str(row.get("index", f"row-{row_pos:04d}"))
    safe_problem_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", problem_id)
    output = args.output / method / dataset / f"seed_{seed}" / f"{safe_problem_id}.json"
    if output.exists() and not args.overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    order_seed = int(hashlib.sha256(f"{method}|{dataset}|{seed}|{problem_id}".encode()).hexdigest()[:8], 16)
    order = list(range(len(responses)))
    random.Random(order_seed).shuffle(order)
    attempts = [(shown, compact(responses[original], args.max_chars)) for shown, original in enumerate(order, 1)]
    started = time.time()
    api_result = call_judge(build_prompt(str(row.get("question", "")), attempts), args)
    parsed = parse_json(api_result["choices"][0]["message"]["content"])
    validate_assignments(parsed, len(responses))

    shown_map = {int(item["attempt_id"]): item for item in parsed["assignments"]}
    labels: list[int | None] = [None] * len(responses)
    valid = [False] * len(responses)
    for shown, original in enumerate(order, 1):
        assignment = shown_map[shown]
        valid[original] = bool(assignment.get("valid", True))
        labels[original] = int(assignment["cluster_id"]) if valid[original] else None
    correct_labels = [label if is_valid and is_correct else None for label, is_valid, is_correct in zip(labels, valid, correctness)]
    record = {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "problem_index": problem_id,
        "source_file": str(source),
        "num_responses": len(responses),
        "num_correct": sum(bool(value) for value in correctness),
        "labels_original_order": labels,
        "valid_original_order": valid,
        "correctness": correctness,
        "all_metrics": diversity_metrics(labels),
        "correct_only_metrics": diversity_metrics(correct_labels),
        "clusters": parsed["clusters"],
        "notes": parsed.get("notes", ""),
        "order_seed": order_seed,
        "judge_model": api_result.get("model", args.model),
        "usage": api_result.get("usage", {}),
        "elapsed_seconds": time.time() - started,
    }
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", required=True, type=parse_method, metavar="NAME=PATH", help="JSONL path pattern; {dataset} and {seed} are supported")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--seed", action="append", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", help=argparse.SUPPRESS)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=0, help="Per-response character limit; 0 keeps full text")
    parser.add_argument("--max-completion-tokens", type=int, default=4000)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-problems", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    tasks = []
    for method, pattern in args.method:
        for dataset in args.dataset:
            for seed in args.seed:
                rows = load_rows(find_jsonl(pattern, dataset, seed))
                limit = min(len(rows), args.limit_problems) if args.limit_problems else len(rows)
                tasks.extend((method, pattern, dataset, seed, row_pos) for row_pos in range(limit))

    failures = args.output / "failures.jsonl"
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(evaluate_task, task, args): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                print(f"OK {future.result()}", flush=True)
                completed += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {task}: {type(exc).__name__}: {exc}", flush=True)
                with failures.open("a") as handle:
                    handle.write(json.dumps({"task": task, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
    print(f"Completed {completed}/{len(tasks)} tasks")
    return 0 if completed == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
