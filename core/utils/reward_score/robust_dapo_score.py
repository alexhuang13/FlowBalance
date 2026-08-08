# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
#
# NOTE (POLARIS tweaks):
# - More aggressive LaTeX/format normalization to reduce false negatives on POLARIS.
#   (\dfrac/\tfrac, \,, \pi vs pi, \frac{a}{b} vs a/b, simple decimals vs rationals, etc.)

import re
from fractions import Fraction
from typing import Any, Optional


# ----------------------------
# Config knobs (safe defaults)
# ----------------------------

# How much of the tail of the solution to look at for extraction/scoring.
# (Keep reasonably small for speed, but >300 helps when answers are longer than MATH-500.)
SOLUTION_TAIL_CHARS = 2000

# If no Answer: line exists, optionally fall back to extracting the last \boxed{...}.
ALLOW_BOXED_FALLBACK = True

# Numeric canonicalization: allow decimal <-> fraction matching and fraction reduction.
ENABLE_NUMERIC_CANONICALIZATION = True
MAX_NUMERIC_DENOMINATOR = 1000  # only canonicalize if reduced denominator <= this

# If True, treat answers like 'f(n)=n' and 'h^2=...' as equivalent to their RHS.
# (Helps on POLARIS where many GTs include an LHS assignment, but models often output only the RHS.)
DROP_EQUATION_LHS = True


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string."""
    idx = string.rfind("\\boxed{")
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

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string."""
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


# ----------------------------
# Normalization utilities
# ----------------------------

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
    # POLARIS / LaTeX variants
    ("\\dfrac", "\\frac"),
    ("\\tfrac", "\\frac"),
    ("\\displaystyle", ""),
    ("\\left", ""),
    ("\\right", ""),
    # Unicode variants
    ("−", "-"),  # U+2212 minus
    ("–", "-"),
    ("—", "-"),
    ("π", "pi"),
    ("\\pi", "pi"),
]

# Expressions that can be safely stripped without changing the math answer (units, spacing tokens, etc.)
REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "m/s",
    "m/s^2",
    "mps",
    "ms^{-1}",
    "ms^-1",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    # LaTeX spacing tokens (POLARIS has \,)
    r"\,",
    r"\;",
    r"\:",
    r"\!",
    r"\quad",
    r"\qquad",
    r",\!",
    "{,}",
    '"',
    "\\dots",
    "~",
]


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        depth = 0
        ok = True
        for i, ch in enumerate(s):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    ok = False
                    break
            if depth < 0:
                ok = False
                break
        if ok and depth == 0:
            s = s[1:-1].strip()
        else:
            break
    return s


def _parse_braced_group(text: str, start: int) -> tuple[Optional[str], int]:
    """Parse a {...} group starting at `start` (which must point to '{')."""
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return None, start


def _replace_latex_fracs_to_slash(s: str) -> str:
    """Convert \frac{a}{b} (and \dfrac/\tfrac) into a/b.

    Also supports shorthand like \frac12 -> 1/2.
    This is a *string* canonicalization (not symbolic equivalence).
    """
    out: list[str] = []
    i = 0
    while i < len(s):
        if s.startswith("\\frac", i) or s.startswith("\\dfrac", i) or s.startswith("\\tfrac", i):
            if s.startswith("\\dfrac", i):
                cmd_len = len("\\dfrac")
            elif s.startswith("\\tfrac", i):
                cmd_len = len("\\tfrac")
            else:
                cmd_len = len("\\frac")

            j = i + cmd_len
            # Skip whitespace
            while j < len(s) and s[j].isspace():
                j += 1

            # Numerator
            if j < len(s) and s[j] == "{":
                num, j2 = _parse_braced_group(s, j)
                if num is None:
                    out.append(s[i])
                    i += 1
                    continue
                j = j2
            elif j < len(s):
                num = s[j]
                j += 1
            else:
                out.append(s[i])
                i += 1
                continue

            while j < len(s) and s[j].isspace():
                j += 1

            # Denominator
            if j < len(s) and s[j] == "{":
                den, j2 = _parse_braced_group(s, j)
                if den is None:
                    out.append(s[i])
                    i += 1
                    continue
                j = j2
            elif j < len(s):
                den = s[j]
                j += 1
            else:
                out.append(s[i])
                i += 1
                continue

            out.append(f"{num}/{den}")
            i = j
            continue

        out.append(s[i])
        i += 1

    return "".join(out)


def _maybe_canonicalize_numeric(s: str) -> str:
    """Optionally canonicalize numeric answers:
    - Reduce fractions (e.g. 4/8 -> 1/2)
    - Convert simple decimals to reduced fractions (e.g. 0.5 -> 1/2, 89.5 -> 179/2)

    Only triggers on *pure numeric* strings to avoid breaking symbolic answers.
    """
    if not ENABLE_NUMERIC_CANONICALIZATION:
        return s
    s0 = s
    s = _strip_outer_parens(s)

    # Pure numeric patterns:
    #   -12, 89.5, 7/8, -1/23, 3.0/4
    if not re.fullmatch(r"[+\-]?\d+(\.\d+)?(/[+\-]?\d+(\.\d+)?)?", s):
        return s0

    try:
        if "/" in s:
            a, b = s.split("/", 1)
            fa = Fraction(a)
            fb = Fraction(b)
            if fb == 0:
                return s0
            f = fa / fb
        else:
            f = Fraction(s)
    except Exception:
        return s0

    if f == 0:
        return "0"
    if f.denominator == 1:
        return str(f.numerator)

    # Keep only "simple" rationals to avoid huge denominators from long decimals.
    if f.denominator <= MAX_NUMERIC_DENOMINATOR:
        return f"{f.numerator}/{f.denominator}"
    return s0


def _normalize_digit_list_if_applicable(s: str) -> str:
    """POLARIS sometimes encodes digit lists without separators (e.g. '024' for {0,2,4}).

    This heuristic only triggers when the answer has a *leading zero* and length>1,
    which is a strong signal it's a concatenation of digits rather than a number.
    """
    if re.fullmatch(r"0\d{1,32}", s):
        return ",".join(list(s))
    return s


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to reduce surface-form mismatches on POLARIS."""
    if final_answer is None:
        return "[INVALID]"

    # Guard: sometimes strings are double-escaped
    # (e.g., literal '\\frac' instead of '\frac')
    final_answer = final_answer.replace("\\\\", "\\").strip()

    # If the answer itself is boxed, extract the content inside the *last* \boxed{...}.
    # IMPORTANT: do this BEFORE splitting on '='; otherwise '\boxed{f(n)=n}' becomes 'n}' (brace leak).
    if "\\boxed{" in final_answer:
        boxed = last_boxed_only_string(final_answer)
        if boxed is not None:
            try:
                final_answer = remove_boxed(boxed).strip()
            except Exception:
                # Fallback: leave as-is and let regex-based cleanup handle it later.
                pass


    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\mathrm\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> \sqrt{a}b
    final_answer = re.sub(r"(frac)([^\{])(.)", r"frac{\2}{\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^\{])", r"sqrt{\2}", final_answer)

    # Drop $ delimiters
    final_answer = final_answer.replace("$", "")

    # Optional: treat equations like "x=1/2" as equivalent to "1/2"
    if DROP_EQUATION_LHS and "=" in final_answer:
        # Split on the last "=" only; this is a lightweight heuristic used in many MATH verifiers.
        final_answer = final_answer.split("=")[-1]

    # POLARIS: unify \pi vs pi (also handled in SUBSTITUTIONS, but keep it here as a safety net)
    final_answer = final_answer.replace("\\pi", "pi").replace("π", "pi")

    # POLARIS: unify \dfrac/\tfrac -> \frac (also handled above)
    final_answer = final_answer.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # POLARIS: normalize \frac{a}{b} <-> a/b
    final_answer = _replace_latex_fracs_to_slash(final_answer)

    # POLARIS: digit-list encoding like "024"
    final_answer = _normalize_digit_list_if_applicable(final_answer)

    # Strip common trailing punctuation that often sneaks into "Answer: ..."
    final_answer = final_answer.strip().strip(" .,:;")

    # Normalize numbers:
    # - remove commas in thousands: "1,234" -> "1234" (also supports decimals: "1,234.5")
    if re.fullmatch(r"[+\-]?\d{1,3}(,\d{3})+(\.\d+)?", final_answer):
        final_answer = final_answer.replace(",", "")

    # Optional: numeric canonicalization (reduce fractions + simple decimals -> fractions)
    if ENABLE_NUMERIC_CANONICALIZATION:
        final_answer = _maybe_canonicalize_numeric(final_answer)

    return final_answer.strip()


# ----------------------------
# Answer extraction + scoring
# ----------------------------

# Detect presence of a valid Answer line.
# We consider it valid if we can extract a non-empty payload from the last "Answer:" marker,
# even if the payload is on the next line (i.e., "Answer:\\n<ans>").


def _extract_answer_from_answer_line(solution_str: str) -> Optional[str]:
    """Extract the answer payload from the *last* Answer: line.

    Handles both:
      - Answer: <ans>
      - Answer:\n<ans>
    """
    # Find last occurrence of 'Answer:' (case-insensitive, multiline)
    matches = list(re.finditer(r"(?im)^\s*Answer\s*:\s*", solution_str))
    if not matches:
        return None
    start = matches[-1].end()
    tail = solution_str[start:].lstrip()
    if not tail:
        return ""
    return tail.splitlines()[0].strip()


def _has_valid_answer_line(solution_str: str) -> bool:
    """Return True if the solution contains a usable Answer: marker.

    We treat it as valid if we can extract a non-empty payload from the *last*
    Answer: marker, even if the payload is on the next line.
    """
    ans = _extract_answer_from_answer_line(solution_str)
    return ans is not None and ans.strip() != ""


def _extract_answer(solution_str: str) -> tuple[str, bool]:
    """Return (extracted_answer, used_answer_line)."""
    ans = _extract_answer_from_answer_line(solution_str)
    if ans is not None:
        return ans, True

    if ALLOW_BOXED_FALLBACK:
        boxed = last_boxed_only_string(solution_str)
        if boxed is not None:
            return remove_boxed(boxed), False

    return "[INVALID]", False


def is_correct_minerva(solution_str: str, gt: str, gt_need_extract: bool = False) -> tuple[bool, str]:
    """String-based correctness check after normalization."""
    extracted_answer, _used_answer_line = _extract_answer(solution_str)
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        boxed_gt = last_boxed_only_string(gt)
        gt = normalize_final_answer(remove_boxed(boxed_gt)) if boxed_gt is not None else normalize_final_answer(gt)
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(
    pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None
) -> tuple[int, Optional[str]]:
    """Strict boxed answer equality (legacy)."""
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    boxed_pred = last_boxed_only_string(pred)
    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None
    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(
    solution_str: str,
    answer: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> tuple[bool, str]:
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[dict[str, Any]] = None,
    reward_router_address: Optional[str] = None,
    reward_model_tokenizer: Any = None,
    prompt_str: Optional[str] = None,
    api_endpoints: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict:
    """Compute reward for a solution.

    Base reward:
      +1.0 if correct else -1.0
    """
    # Keep full string for format detection, but use only tail for scoring to stay fast.
    full_solution = solution_str if solution_str is not None else ""
    has_answer_line = _has_valid_answer_line(full_solution)

    # Scoring: use the full string so we don't miss an earlier Answer: line
    # (e.g., if the model continues generating after the answer).
    correct, pred = verify(full_solution, ground_truth, strict_box_verify, pause_tokens_index)

    reward = 1.0 if correct else -1.0

    return {
        "score": reward,
        "acc": bool(correct),
        "pred": pred,
        "has_answer_line": has_answer_line,
    }
