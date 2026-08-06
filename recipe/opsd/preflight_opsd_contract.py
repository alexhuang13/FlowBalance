#!/usr/bin/env python3
"""Fail-fast validation for the OPSD Ray launch contract.

This intentionally runs before ``ray job submit`` so path, shape, typed config,
and objective errors do not consume a cluster allocation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "verl"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--runtime-env", type=Path, required=True)
    p.add_argument("--train-file", type=Path, required=True)
    p.add_argument("--test-file", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--nnodes", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, required=True)
    p.add_argument("--sp-size", type=int, required=True)
    p.add_argument("--train-batch-size", type=int, required=True)
    p.add_argument("--rollout-n", type=int, required=True)
    p.add_argument("--ppo-mini-batch-size", type=int, required=True)
    p.add_argument("--max-prompt-length", type=int, required=True)
    p.add_argument("--max-response-length", type=int, required=True)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--max-reprompt-len", type=int, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--distillation-topk", type=int, required=True)
    p.add_argument("--token-loss-clip", type=float, required=True)
    args = p.parse_args()

    required = {
        "repo root": args.repo_root,
        "runtime env": args.runtime_env,
        "train file": args.train_file,
        "test file": args.test_file,
        "model path": args.model_path,
        "OPSD config": args.repo_root / "recipe/opsd/config/opsd_trainer.yaml",
        "reward function": args.repo_root / "core/utils/reward_score/sdpo_math_feedback_score.py",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required paths:\n  " + "\n  ".join(missing))

    for name in (
        "nnodes", "gpus_per_node", "sp_size", "train_batch_size", "rollout_n",
        "ppo_mini_batch_size", "max_prompt_length", "max_response_length",
        "max_model_len", "max_reprompt_len", "distillation_topk",
    ):
        positive(name, getattr(args, name))
    total_gpus = args.nnodes * args.gpus_per_node
    if total_gpus % args.sp_size:
        raise ValueError(f"total GPUs {total_gpus} must be divisible by SP size {args.sp_size}")
    dp_size = total_gpus // args.sp_size
    rollout_batch = args.train_batch_size * args.rollout_n
    if rollout_batch % dp_size:
        raise ValueError(f"rollout batch {rollout_batch} must be divisible by DP size {dp_size}")
    if rollout_batch % args.ppo_mini_batch_size:
        raise ValueError(
            f"rollout batch {rollout_batch} must be divisible by PPO mini batch {args.ppo_mini_batch_size}"
        )
    if args.max_prompt_length + args.max_response_length > args.max_model_len:
        raise ValueError("max_prompt_length + max_response_length exceeds max_model_len")
    if args.max_response_length > args.max_reprompt_len:
        raise ValueError("max_response_length exceeds max_reprompt_len")

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {args.alpha}")
    if args.token_loss_clip <= 0:
        raise ValueError(f"token_loss_clip must be > 0, got {args.token_loss_clip}")
    print(
        "OPSD preflight OK: "
        f"gpus={total_gpus}, dp={dp_size}, prompt_batch={args.train_batch_size}, "
        f"rollout_batch={rollout_batch}, alpha={args.alpha}, topk={args.distillation_topk}, "
        f"token_clip={args.token_loss_clip}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
