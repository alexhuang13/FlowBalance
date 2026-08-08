#!/usr/bin/env python3
"""Run lightweight checks for a public FlowSD source release."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRST_PARTY_DOCS = [
    ROOT / "README.md",
    ROOT / "ENVIRONMENT.md",
    ROOT / "EVALUATION.md",
    ROOT / "EXPERIMENTS.md",
    ROOT / "RELEASE.md",
    ROOT / "analysis/diversity/README.md",
    *sorted((ROOT / "recipe").glob("*/README.md")),
]
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SECRET_PATTERNS = [
    re.compile(r"wandb_[A-Za-z0-9_-]{20,}"),
    re.compile(r'"(?:Token|secret|passwd|password)"\s*:'),
]
SKIP_PARTS = {".git", "provenance", "verl", "__pycache__", ".pytest_cache", "raw", "example_results"}


def iter_release_files():
    paths = list(FIRST_PARTY_DOCS)
    paths.extend(
        [
            ROOT / "runtime_env.yaml",
            ROOT / "launch/start.sh",
            ROOT / "analysis/diversity/evaluate.py",
            ROOT / "analysis/diversity/aggregate.py",
            ROOT / "analysis/diversity/metrics.py",
        ]
    )
    paths.extend(sorted((ROOT / "recipe").glob("*/run_*.sh")))
    seen = set()
    for path in paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        yield path


def main() -> int:
    errors: list[str] = []
    for path in FIRST_PARTY_DOCS:
        if not path.is_file():
            errors.append(f"missing first-party document: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(errors="replace")
        if CJK.search(text):
            errors.append(f"non-English CJK text in first-party document: {path.relative_to(ROOT)}")

    for path in iter_release_files():
        text = path.read_text(errors="replace")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential in {path.relative_to(ROOT)}")
                break

    runtime_env = (ROOT / "runtime_env.yaml").read_text()
    if 'WANDB_API_KEY: "your key"' not in runtime_env:
        errors.append("runtime_env.yaml must contain only the W&B placeholder")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("FlowSD release check PASSED")
    if not (ROOT / "LICENSE").exists():
        print("[WARN] No root LICENSE file is present; choose a license before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
