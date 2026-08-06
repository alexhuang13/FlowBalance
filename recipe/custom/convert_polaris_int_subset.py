#!/usr/bin/env python3
import argparse
import re
from fractions import Fraction

from datasets import Dataset, DatasetDict, load_from_disk


RE_INT = re.compile(r"^-?\d+$")
RE_FRAC_LATEX = re.compile(r"^\\frac\{(-?\d+)\}\{(\d+)\}$")
RE_FRAC_SLASH = re.compile(r"^(-?\d+)/(\d+)$")
RE_DECIMAL = re.compile(r"^-?\d+\.\d+$")
RE_LIST_COMMA = re.compile(r"^-?\d+(,-?\d+)+$")
RE_LIST_BRACES = re.compile(r"^\{(-?\d+(,-?\d+)*)\}$")
RE_LIST_DIGITS = re.compile(r"^\d{2,}$")

FRAC_PROMPT_SUFFIX = "\n\nIf your answer can be expressed as a reduced fraction a/b, output a+b."
LIST_PROMPT_SUFFIX = "\n\nIf the solutions are integers, output the sum of all solutions."
LARGE_INT_PROMPT_SUFFIX = (
    "\n\nIf your final numeric answer is an integer with more than 5 digits, "
    "output the sum of digits of its absolute value."
)


def _strip_outer_boxed(answer: str) -> str:
    prefix = r"\boxed{"
    if not answer.startswith(prefix) or not answer.endswith("}"):
        return answer
    inner = answer[len(prefix) : -1]
    if inner.count("{") != inner.count("}"):
        return answer
    return inner


def normalize_answer(answer: str) -> str:
    text = answer.strip().replace("$", "")
    text = _strip_outer_boxed(text)
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\s+", "", text)
    return text


def parse_fraction(text: str) -> tuple[int, int] | None:
    match = RE_FRAC_LATEX.fullmatch(text)
    if match is not None:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator != 0:
            return numerator, denominator

    match = RE_FRAC_SLASH.fullmatch(text)
    if match is not None:
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if denominator != 0:
            return numerator, denominator
    return None


def parse_integer_list(text: str) -> list[int] | None:
    if RE_LIST_BRACES.fullmatch(text):
        text = text[1:-1]
    if RE_LIST_COMMA.fullmatch(text):
        return [int(x) for x in text.split(",")]
    if RE_LIST_DIGITS.fullmatch(text) and text.startswith("0"):
        return [int(ch) for ch in text]
    return None


def maybe_shrink_large_integer(problem: str, answer: str, transform_tag: str) -> tuple[str, str, str]:
    if RE_INT.fullmatch(answer) is None:
        return problem, answer, transform_tag
    digit_text = answer.lstrip("-")
    if len(digit_text) <= 5:
        return problem, answer, transform_tag
    shrunk = str(sum(int(ch) for ch in digit_text))
    return problem, shrunk, f"{transform_tag}_largeint_digit_sum"


def apply_ordered_prompt_suffix(problem: str, transform_tag: str, answer: str) -> str:
    base = problem.rstrip()
    if transform_tag.startswith("frac_num_den_sum") or transform_tag.startswith("dec_to_frac_num_den_sum"):
        return base + FRAC_PROMPT_SUFFIX
    if transform_tag.startswith("int_list_sum"):
        return base + LIST_PROMPT_SUFFIX
    if RE_INT.fullmatch(answer) and len(answer.lstrip("-")) > 5:
        return base + LARGE_INT_PROMPT_SUFFIX
    return base


def transform_example(problem: str, answer: str) -> tuple[str, str, str] | None:
    normalized = normalize_answer(answer)

    integer_list = parse_integer_list(normalized)
    if integer_list is not None:
        new_problem = problem
        new_answer = str(sum(integer_list))
        new_problem, new_answer, transform_tag = maybe_shrink_large_integer(new_problem, new_answer, "int_list_sum")
        new_problem = apply_ordered_prompt_suffix(new_problem, transform_tag, new_answer)
        return new_problem, new_answer, transform_tag

    if RE_INT.fullmatch(normalized):
        new_problem, new_answer, transform_tag = maybe_shrink_large_integer(problem, str(int(normalized)), "int_identity")
        new_problem = apply_ordered_prompt_suffix(new_problem, transform_tag, new_answer)
        return new_problem, new_answer, transform_tag

    fraction = parse_fraction(normalized)
    if fraction is not None:
        reduced = Fraction(fraction[0], fraction[1])
        new_problem = problem
        new_answer = str(reduced.numerator + reduced.denominator)
        new_problem, new_answer, transform_tag = maybe_shrink_large_integer(new_problem, new_answer, "frac_num_den_sum")
        new_problem = apply_ordered_prompt_suffix(new_problem, transform_tag, new_answer)
        return new_problem, new_answer, transform_tag

    if RE_DECIMAL.fullmatch(normalized):
        reduced = Fraction(normalized)
        new_problem = problem
        new_answer = str(reduced.numerator + reduced.denominator)
        new_problem, new_answer, transform_tag = maybe_shrink_large_integer(new_problem, new_answer, "dec_to_frac_num_den_sum")
        new_problem = apply_ordered_prompt_suffix(new_problem, transform_tag, new_answer)
        return new_problem, new_answer, transform_tag

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    loaded = load_from_disk(args.input_dir)
    source = loaded[args.split] if isinstance(loaded, DatasetDict) else loaded

    converted_rows = []
    skipped = 0
    for row in source:
        result = transform_example(str(row.get("problem", "")), str(row.get("answer", "")))
        if result is None:
            skipped += 1
            continue

        new_problem, new_answer, transform_tag = result
        out = dict(row)
        out["problem"] = new_problem
        out["answer"] = new_answer
        out["transform"] = transform_tag
        out["orig_answer"] = str(row.get("answer", ""))
        converted_rows.append(out)

    converted = Dataset.from_list(converted_rows)
    converted.save_to_disk(args.output_dir)
    print(f"kept={len(converted_rows)} skipped={skipped}")


if __name__ == "__main__":
    main()
