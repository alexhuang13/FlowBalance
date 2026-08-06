"""Pure tensor algorithms for RLSD."""
from __future__ import annotations

from typing import Optional
import torch
import verl.utils.torch_functional as verl_F


def resolve_lambda(target: float, step: int, warmup_steps: int, decay_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return target * float(step) / float(warmup_steps)
    if decay_steps > 0 and step >= warmup_steps:
        progress = float(step - warmup_steps) / float(decay_steps)
        return target * max(1.0 - progress, 0.0)
    return target


def build_rlsd_advantages(
    *,
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_mask: torch.Tensor,
    lam: float,
    clip_range: Optional[float],
    negative_only: bool = False,
    thought_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply RLSD token reweighting to a standard GRPO advantage tensor."""
    mask = response_mask.to(dtype=student_log_probs.dtype)
    sample_mask = self_distillation_mask.to(mask.device, mask.dtype).unsqueeze(-1)
    sign_adv = torch.sign(advantages)
    delta = (teacher_log_probs.detach() - student_log_probs.detach()) * mask
    raw_weight = torch.exp((sign_adv * delta).clamp(min=-20.0, max=20.0))
    if clip_range is None:
        clipped_weight = raw_weight
        low_ratio = high_ratio = 0.0
    else:
        low, high = 1.0 - float(clip_range), 1.0 + float(clip_range)
        clipped_weight = raw_weight.clamp(min=low, max=high)
        low_ratio = verl_F.masked_mean((raw_weight < low).float(), mask).item()
        high_ratio = verl_F.masked_mean((raw_weight > high).float(), mask).item()

    mixed = (1.0 - lam) + lam * clipped_weight
    # Samples without a same-group successful demonstration remain plain GRPO.
    reweight = sample_mask * mixed + (1.0 - sample_mask)
    if negative_only:
        seq_negative = (advantages.sum(dim=-1, keepdim=True) < 0).to(mask.dtype)
        reweight = seq_negative * reweight + (1.0 - seq_negative)
    if thought_mask is not None:
        reweight = thought_mask * reweight + (1.0 - thought_mask)
    reweight = reweight * mask
    out = advantages * reweight.detach()

    metrics = {
        "rlsd/lambda": float(lam),
        "rlsd/delta_mean": verl_F.masked_mean(delta, mask).item(),
        "rlsd/raw_weight_mean": verl_F.masked_mean(raw_weight, mask).item(),
        "rlsd/weight_mean": verl_F.masked_mean(reweight, mask).item(),
        "rlsd/clip_low_ratio": low_ratio,
        "rlsd/clip_high_ratio": high_ratio,
        "rlsd/reweighted_adv_abs_mean": verl_F.masked_mean(out.abs(), mask).item(),
        "rlsd/teacher_sample_fraction": self_distillation_mask.float().mean().item(),
    }
    return out, metrics
