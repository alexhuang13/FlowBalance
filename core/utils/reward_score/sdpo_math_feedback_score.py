# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# Adapted from self-distillation-analysis/verl/utils/reward_score/feedback/math.py.

import signal
from typing import Any, Optional

try:
    from math_verify import parse as mv_parse, verify as mv_verify
except Exception:
    mv_parse = None
    mv_verify = None


FORMAT_PENALTY = False


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind(r"\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else ""


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    left = r"\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    return ""


class timeout:
    def __init__(self, seconds: int = 1, error_message: str = "Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, value, traceback):
        signal.alarm(0)


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[bool, Optional[str]]:
    """Check correctness using the original SDPO strict boxed-answer rule."""
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None
    return extracted_pred == gt, extracted_pred


def verify(
    solution_str: str, answer: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[bool, str]:
    """Verify the solution with boxed extraction, then math_verify equivalence."""
    correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
    if pred is None:
        pred = ""

    if not correct and pred != "" and mv_parse is not None and mv_verify is not None:
        try:
            with timeout(seconds=5):
                gold_expr = mv_parse(answer)
                pred_expr = mv_parse(pred)
                correct = mv_verify(gold_expr, pred_expr)
        except Exception:
            pass
    return bool(correct), pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    pause_tokens_index: Optional[list[int]] = None,
    format_feedback: bool = True,
    correctness_feedback: bool = False,
    data_source: Optional[str] = None,
    strict_box_verify: bool = True,
    prompt_str: Optional[str] = None,
    api_endpoints: Optional[dict[str, Any]] = None,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Original self-distillation-analysis math reward semantics.

    Correct boxed answers score 1.0; all incorrect or malformed answers score 0.0.
    The expanded signature keeps compatibility with stable_rl's custom_dapo manager.
    """
    del data_source, strict_box_verify, prompt_str, api_endpoints, reward_router_address, reward_model_tokenizer, kwargs

    extra_info = extra_info or {}
    split = extra_info.get("split", "test")
    was_truncated = extra_info.get("truncated", False)

    correct, pred = verify(solution_str, ground_truth, pause_tokens_index)

    reward = 1.0 if correct else 0.0
    score = reward
    incorrect_format = pred is None or pred == ""
    if FORMAT_PENALTY and split == "train" and incorrect_format and not was_truncated:
        score -= 0.5

    feedback = ""
    if incorrect_format and not was_truncated and format_feedback:
        feedback = "Your answer had the wrong format. The solution must be given in the format: \\boxed{your_answer}."
    elif was_truncated and format_feedback:
        feedback = "Your response was truncated because it exceeded the maximum length."
    elif not correct and correctness_feedback:
        feedback = f"Your answer is incorrect. The correct answer is {ground_truth}."

    return {
        "score": score,
        "acc": reward,
        "pred": pred,
        "incorrect_format": 1 if incorrect_format else 0,
        "truncated": 1 if was_truncated else 0,
        "truncated_and_missing_answer": 1 if incorrect_format and was_truncated else 0,
        "feedback": feedback,
    }
