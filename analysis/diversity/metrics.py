"""Metrics for semantic strategy clusters."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Hashable


def diversity_metrics(labels: Iterable[Hashable | None]) -> dict[str, object]:
    """Compute diversity metrics after dropping ``None`` labels.

    Simpson diversity is ``1 - sum(p_k ** 2)``. Normalized entropy uses
    ``log(n)`` as the denominator so that it remains defined when the number of
    observed clusters varies across problems.
    """
    values = [label for label in labels if label is not None]
    n = len(values)
    if not n:
        return {
            "n": 0,
            "k": None,
            "dominant_ratio": None,
            "normalized_entropy": None,
            "simpson": None,
            "cluster_sizes": {},
        }
    counts = Counter(values)
    probabilities = [count / n for count in counts.values()]
    entropy = -sum(p * math.log(p) for p in probabilities)
    return {
        "n": n,
        "k": len(counts),
        "dominant_ratio": max(probabilities),
        "normalized_entropy": entropy / math.log(n) if n > 1 else 0.0,
        "simpson": 1.0 - sum(p * p for p in probabilities),
        "cluster_sizes": dict(counts),
    }
