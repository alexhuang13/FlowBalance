#!/usr/bin/env python3
"""Cheap preflight checks before submitting the expensive SDPO Ray job."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _ok(message: str) -> None:
    print(f"[ OK ] {message}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_reward_fn(repo_root: Path, module_path: str, fn_name: str):
    path = Path(module_path)
    if not path.is_absolute():
        path = repo_root / path
    spec = importlib.util.spec_from_file_location("sdpo_preflight_reward_fn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reward function module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, fn_name), path


def _read_parquet_sample(path: Path, columns: list[str], sample_size: int) -> tuple[int, list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyarrow is required for cheap parquet preflight") from exc

    parquet_file = pq.ParquetFile(path)
    if parquet_file.metadata is None:
        raise RuntimeError(f"missing parquet metadata: {path}")
    row_count = parquet_file.metadata.num_rows
    if row_count <= 0:
        return row_count, []

    available = set(parquet_file.schema_arrow.names)
    missing = [col for col in columns if col not in available]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}; available={sorted(available)}")

    try:
        batch = next(parquet_file.iter_batches(batch_size=sample_size, columns=columns))
    except StopIteration:
        return row_count, []
    return row_count, batch.to_pylist()


def _message_texts(prompt: Any) -> list[tuple[str, str]]:
    if prompt is None:
        return []
    messages = prompt.tolist() if hasattr(prompt, "tolist") else prompt
    out: list[tuple[str, str]] = []
    for message in messages:
        if hasattr(message, "as_py"):
            message = message.as_py()
        if not isinstance(message, dict):
            continue
        out.append((str(message.get("role", "")), str(message.get("content", ""))))
    return out


def _check_dataset(path: Path, label: str, sample_size: int, errors: list[str]) -> tuple[int, list[dict[str, Any]]]:
    if not path.is_file():
        _fail(errors, f"{label} parquet not found: {path}")
        return 0, []
    try:
        row_count, rows = _read_parquet_sample(path, ["prompt", "data_source", "reward_model"], sample_size)
    except Exception as exc:
        _fail(errors, f"failed to read {label} parquet {path}: {exc}")
        return 0, []
    _ok(f"{label} parquet readable: rows={row_count}, sample={len(rows)}")
    if row_count <= 0:
        _fail(errors, f"{label} parquet is empty: {path}")
    return row_count, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reward-fn-path", required=True)
    parser.add_argument("--reward-fn-name", default="compute_score")
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--sp-size", type=int, required=True)
    parser.add_argument("--train-batch-size", type=int, required=True)
    parser.add_argument("--rollout-n", type=int, required=True)
    parser.add_argument("--ppo-mini-batch-size", type=int, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-response-length", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-train-rows", type=int, default=50000)
    parser.add_argument("--sample-size", type=int, default=8)
    args = parser.parse_args()

    errors: list[str] = []
    repo_root = Path(args.repo_root).resolve()
    train_file = Path(args.train_file)
    test_file = Path(args.test_file)
    model_path = Path(args.model_path)

    print("[SDPO preflight] starting cheap checks before Ray job submission")

    if not repo_root.is_dir():
        _fail(errors, f"repo root not found: {repo_root}")
    else:
        _ok(f"repo root exists: {repo_root}")

    if not model_path.is_dir():
        _fail(errors, f"model dir not found: {model_path}")
    else:
        _ok(f"model dir exists: {model_path}")
        config_path = model_path / "config.json"
        tokenizer_config_path = model_path / "tokenizer_config.json"
        if not config_path.is_file():
            _fail(errors, f"model config missing: {config_path}")
        else:
            try:
                model_config = _read_json(config_path)
                num_heads = int(model_config.get("num_attention_heads", 0))
                if num_heads <= 0:
                    _warn("model config does not expose num_attention_heads")
                elif num_heads % args.sp_size != 0:
                    _fail(errors, f"num_attention_heads={num_heads} is not divisible by SP_SIZE={args.sp_size}")
                else:
                    _ok(f"SP_SIZE compatible with attention heads: {num_heads} % {args.sp_size} == 0")
            except Exception as exc:
                _fail(errors, f"failed reading model config: {exc}")
        if tokenizer_config_path.is_file():
            _ok("tokenizer_config.json exists")
        else:
            _warn(f"tokenizer_config.json missing: {tokenizer_config_path}")

    total_gpus = args.nnodes * args.gpus_per_node
    if args.sp_size <= 0:
        _fail(errors, f"invalid SP_SIZE={args.sp_size}")
        dp_size = 0
    elif total_gpus % args.sp_size != 0:
        _fail(errors, f"total_gpus={total_gpus} is not divisible by SP_SIZE={args.sp_size}")
        dp_size = 0
    else:
        dp_size = total_gpus // args.sp_size
        _ok(f"parallel sizes: total_gpus={total_gpus}, SP={args.sp_size}, DP={dp_size}")

    rollout_batch = args.train_batch_size * args.rollout_n
    if dp_size and rollout_batch % dp_size != 0:
        _fail(errors, f"rollout_batch={rollout_batch} must be divisible by DP size={dp_size}")
    else:
        _ok(f"rollout_batch divisible by DP: {rollout_batch}")
    if rollout_batch % args.ppo_mini_batch_size != 0:
        _fail(errors, f"rollout_batch={rollout_batch} must be divisible by ppo_mini_batch_size={args.ppo_mini_batch_size}")
    else:
        _ok(f"rollout_batch divisible by ppo_mini_batch_size: {args.ppo_mini_batch_size}")
    required_model_len = args.max_prompt_length + args.max_response_length
    if args.max_model_len < required_model_len:
        _fail(
            errors,
            f"max_model_len={args.max_model_len} is smaller than max_prompt_length+max_response_length={required_model_len}; "
            "this vLLM V1 stack can degenerate into token-0 ('!') loops / NaNs under that setting",
        )

    train_row_count, train_rows = _check_dataset(train_file, "train", args.sample_size, errors)
    _test_row_count, test_rows = _check_dataset(test_file, "test", args.sample_size, errors)
    if train_row_count > args.max_train_rows:
        _fail(
            errors,
            f"train parquet has {train_row_count} rows, larger than strict-reproduction limit {args.max_train_rows}; "
            "check that DAPO-Math-17k was not repeated/expanded",
        )

    train_has_boxed_instruction = False
    train_has_answer_instruction = False
    system_answer_instruction = False
    for row in train_rows:
        for role, content in _message_texts(row.get("prompt")):
            lower = content.lower()
            if role == "system" and ("answer:" in lower or "boxed" in lower or "final answer" in lower):
                system_answer_instruction = True
            if "\\boxed{}" in content or "\\boxed" in content:
                train_has_boxed_instruction = True
            if "answer:" in lower:
                train_has_answer_instruction = True
    if system_answer_instruction:
        _warn("system prompt contains answer-format instructions; verify scorer matches it")
    else:
        _ok("no sampled train system prompt with answer-format instructions")
    if train_has_boxed_instruction:
        _ok("sampled train prompts contain boxed answer instruction")
    elif train_has_answer_instruction:
        _fail(errors, "sampled train prompts contain Answer: instruction, not boxed; upstream SDPO math scorer is boxed-only")
    else:
        _warn("sampled train prompts do not show boxed or Answer: instruction")

    test_has_answer_instruction = any(
        "answer:" in content.lower()
        for row in test_rows
        for _role, content in _message_texts(row.get("prompt"))
    )
    test_has_boxed_instruction = any(
        "\\boxed{}" in content or "\\boxed" in content
        for row in test_rows
        for _role, content in _message_texts(row.get("prompt"))
    )
    if test_has_answer_instruction and train_has_boxed_instruction:
        _warn("train prompts are boxed while sampled eval prompts are Answer:-style; strict SDPO scorer follows upstream boxed-only scoring")
    elif test_has_boxed_instruction:
        _ok("sampled eval prompts contain boxed answer instruction")

    try:
        reward_fn, reward_path = _load_reward_fn(repo_root, args.reward_fn_path, args.reward_fn_name)
        _ok(f"reward function loaded: {reward_path}:{args.reward_fn_name}")
    except Exception as exc:
        _fail(errors, f"failed to load reward function {args.reward_fn_path}:{args.reward_fn_name}: {exc}")
        reward_fn = None

    if reward_fn is not None:
        scorer_cases = [
            ("boxed", "We solve it. Thus \\boxed{34}.", "34", True),
            ("answer_rejected_by_strict_boxed", "Reasoning...\nAnswer: 34", "34", False),
            ("custom_dapo_kwargs", "Thus \\boxed{34}.", "34", True),
        ]
        for case_name, solution, gt, expect_positive in scorer_cases:
            try:
                result = reward_fn(
                    data_source="math_dapo",
                    solution_str=solution,
                    ground_truth=gt,
                    prompt_str="dummy prompt",
                    extra_info={},
                    api_endpoints={},
                )
                score_obj = result.get("score") if isinstance(result, dict) else result
                if score_obj is None:
                    _fail(errors, f"reward scorer case {case_name} returned no score: result={result}")
                    continue
                score = float(score_obj)
                if expect_positive and score <= 0:
                    _fail(errors, f"reward scorer case {case_name} did not score positive: result={result}")
                elif not expect_positive and score > 0:
                    _fail(errors, f"reward scorer case {case_name} should be non-positive for upstream strict boxed scorer: result={result}")
                else:
                    _ok(f"reward scorer case {case_name} passed: result={result}")
            except Exception as exc:
                _fail(errors, f"reward scorer case {case_name} raised {type(exc).__name__}: {exc}")

    if errors:
        print("\n[SDPO preflight] FAILED; refusing to submit expensive Ray job.")
        for idx, error in enumerate(errors, start=1):
            print(f"  {idx}. {error}")
        return 1

    print("\n[SDPO preflight] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
