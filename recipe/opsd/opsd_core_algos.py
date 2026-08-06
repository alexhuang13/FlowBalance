"""Pure OPSD loss helpers without Ray/verl runtime dependencies."""

from __future__ import annotations

import torch


def clip_pointwise_divergence(
    per_token_divergence: torch.Tensor,
    token_loss_clip: float | None,
) -> tuple[torch.Tensor, float]:
    """Clip non-negative per-token divergence and return clipped fraction."""
    if token_loss_clip is None:
        return per_token_divergence, 0.0
    clip = float(token_loss_clip)
    if clip <= 0:
        raise ValueError(f"token_loss_clip must be positive or None, got {clip}")
    fraction = (
        int((per_token_divergence >= clip).sum().item()) / per_token_divergence.numel()
        if per_token_divergence.numel()
        else 0.0
    )
    return per_token_divergence.clamp(max=clip), fraction
