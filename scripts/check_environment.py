#!/usr/bin/env python3
"""Validate that a FlowSD checkout can be used as a Ray working directory."""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = (
    "runtime_env.yaml",
    "verl/verl/models/registry.py",
    "verl/verl/workers/config/model.py",
    "core/trainer/main_ppo.py",
    "recipe/sdpo/main_sdpo.py",
    "recipe/flowopsd/main_flowopsd.py",
    "recipe/antisd/main_antisd.py",
    "recipe/rlsd/main_rlsd.py",
    "recipe/opsd/main_opsd.py",
)
IMPORTS = (
    "verl.models.registry",
    "core.trainer.main_ppo",
    "recipe.sdpo.main_sdpo",
    "recipe.flowopsd.main_flowopsd",
    "recipe.antisd.main_antisd",
    "recipe.rlsd.main_rlsd",
    "recipe.opsd.main_opsd",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-imports", action="store_true")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    errors: list[str] = []

    for relative in REQUIRED_FILES:
        path = root / relative
        if path.is_file():
            print(f"[ OK ] required file: {relative}")
        else:
            errors.append(f"missing required file: {relative}")

    models_file = "verl/verl/models/registry.py"
    ignored = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", models_file], check=False
    ).returncode == 0
    if ignored:
        errors.append(f"Ray-critical package is ignored by git rules: {models_file}")
    else:
        print(f"[ OK ] Ray packaging includes: {models_file}")

    toolchain = root / "recipe/sdpo/toolchain"
    for compiler in ("gcc", "g++"):
        path = toolchain / compiler
        if path.is_file() and path.stat().st_mode & 0o111:
            print(f"[ OK ] toolchain wrapper: {path.relative_to(root)}")
        else:
            errors.append(f"missing/non-executable toolchain wrapper: {path}")

    if not args.skip_imports:
        sys.path[:0] = [str(root / "verl"), str(root)]
        for name in IMPORTS:
            try:
                module = importlib.import_module(name)
                print(f"[ OK ] import {name}: {module.__file__}")
            except Exception as exc:
                errors.append(f"cannot import {name}: {type(exc).__name__}: {exc}")

    if errors:
        for error in errors:
            print(f"[FAIL] {error}", file=sys.stderr)
        return 1
    print("FlowSD environment check PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
