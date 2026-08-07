#!/usr/bin/env python3
"""FlowSD preflight checks.

FlowSD uses the same data, model, reward, length, and batch-alignment checks as
SDPO. The training objective differs after rollout, so no extra pre-submit check
is needed here.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recipe.sdpo.preflight_math_sdpo import main


if __name__ == "__main__":
    sys.exit(main())
