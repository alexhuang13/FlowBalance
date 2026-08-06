"""Dependency-free source checks for lightweight repository environments."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_opsd_python_sources_parse():
    for path in (ROOT / "recipe/opsd").glob("*.py"):
        ast.parse(path.read_text(), filename=str(path))


def test_launcher_has_fail_fast_contract():
    text = (ROOT / "recipe/opsd/run_math_opsd.sh").read_text()
    for required in ("set -Eeuo pipefail", "DRY_RUN", "preflight_opsd_contract.py", "hydra_args=("):
        assert required in text
