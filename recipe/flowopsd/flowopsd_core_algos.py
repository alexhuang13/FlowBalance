# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Core FlowOPSD target and diagnostic computations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

__all__ = ["compute_flowopsd_target", "resolve_flowopsd_coefficients"]


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return float(value)


def _linear_schedule(base: float, start: Any, end: Any, schedule_steps: int, step: int | None) -> tuple[float, float]:
    start_value = _optional_float(start)
    end_value = _optional_float(end)
    if step is None or schedule_steps <= 0 or start_value is None or end_value is None:
        return float(base), 0.0
    progress = min(max(float(step) / float(schedule_steps), 0.0), 1.0)
    value = start_value + progress * (end_value - start_value)
    return float(value), progress



def resolve_flowopsd_coefficients(
    flow_opsd_config: Any, global_step: int | None = None
) -> tuple[float, float, float, float]:
    """Resolve scheduled beta_q/eta_R values and their schedule progress."""
    schedule_steps = int(_cfg_get(flow_opsd_config, "schedule_steps", 0))
    beta_q, beta_progress = _linear_schedule(
        float(_cfg_get(flow_opsd_config, "beta_q", 1.0)),
        _cfg_get(flow_opsd_config, "beta_q_start", None),
        _cfg_get(flow_opsd_config, "beta_q_end", None),
        schedule_steps,
        global_step,
    )
    eta_R, eta_progress = _linear_schedule(
        float(_cfg_get(flow_opsd_config, "eta_R", 15.0)),
        _cfg_get(flow_opsd_config, "eta_R_start", None),
        _cfg_get(flow_opsd_config, "eta_R_end", None),
        schedule_steps,
        global_step,
    )
    return beta_q, eta_R, beta_progress, eta_progress


def _uids_to_list(uids: Any) -> list[Any]:
    if isinstance(uids, np.ndarray):
        return uids.tolist()
    if hasattr(uids, "tolist"):
        return uids.tolist()
    return list(uids)


@torch.no_grad()
def compute_flowopsd_target(
    teacher_log_prob: torch.Tensor | None,
    ref_log_prob: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    reward_signal: torch.Tensor,
    advantage_signal: torch.Tensor,
    self_distillation_mask: torch.Tensor | None,
    uids: Any,
    flow_opsd_config: Any,
    global_step: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute detached FlowOPSD length-normalized targets and diagnostics.

    Returns:
        flowopsd_target: shape [B], length-normalized sequence log-prob target
        flowopsd_mask: shape [B], float
        metrics: scalar diagnostics
    """
    device = ref_log_prob.device
    dtype = ref_log_prob.dtype
    response_mask = response_mask.to(device=device, dtype=dtype)
    ref_log_prob = ref_log_prob.to(device=device, dtype=dtype)
    old_log_probs = old_log_probs.to(device=device, dtype=dtype)
    reward_signal = reward_signal.to(device=device, dtype=dtype)
    advantage_signal = advantage_signal.to(device=device, dtype=dtype)
    if self_distillation_mask is None:
        self_distillation_mask = torch.ones(response_mask.shape[0], device=device, dtype=dtype)
    else:
        self_distillation_mask = self_distillation_mask.to(device=device, dtype=dtype)

    batch_size = response_mask.shape[0]
    uid_list = _uids_to_list(uids)
    if len(uid_list) != batch_size:
        raise ValueError(f"uid length {len(uid_list)} does not match batch size {batch_size}")

    beta_q, eta_R, beta_q_schedule_progress, eta_R_schedule_progress = resolve_flowopsd_coefficients(
        flow_opsd_config, global_step
    )
    beta_zero = abs(beta_q) <= 1e-12
    clip_B = float(_cfg_get(flow_opsd_config, "clip_B", 4.0))
    rho = float(_cfg_get(flow_opsd_config, "rho", 1.0))
    gate_no_context = _cfg_get(flow_opsd_config, "gate_no_context", "drop")
    min_group_valid = int(_cfg_get(flow_opsd_config, "min_group_valid", 2))
    reference_source = _cfg_get(flow_opsd_config, "reference_source", "frozen_ref")

    raw_lengths = response_mask.sum(dim=-1)
    lengths = raw_lengths.clamp(min=1.0)
    length_norm = lengths.pow(rho)
    seq_logp_ref_raw = (ref_log_prob * response_mask).sum(dim=-1)
    seq_logp_old_raw = (old_log_probs * response_mask).sum(dim=-1)
    if teacher_log_prob is None:
        teacher_log_prob = ref_log_prob
    else:
        teacher_log_prob = teacher_log_prob.to(device=device, dtype=dtype)
    seq_logp_teacher_raw = (teacher_log_prob * response_mask).sum(dim=-1)
    seq_logp_ref = seq_logp_ref_raw / length_norm
    seq_logp_old = seq_logp_old_raw / length_norm
    seq_logp_teacher = seq_logp_teacher_raw / length_norm

    if reference_source == "old_log_probs":
        ref_for_gain = old_log_probs
        seq_logp_ref_for_reward = seq_logp_old
    else:
        ref_for_gain = ref_log_prob
        seq_logp_ref_for_reward = seq_logp_ref

    delta_raw = (teacher_log_prob - ref_for_gain) * response_mask
    delta = delta_raw.clamp(min=-clip_B, max=clip_B) * response_mask
    if gate_no_context == "keep":
        gain_gate = self_distillation_mask.unsqueeze(-1)
        delta = delta * gain_gate
    G_q_raw = delta.sum(dim=-1) / length_norm
    sequence_advantage = (advantage_signal * response_mask).sum(dim=-1) / lengths
    advantage_sign = torch.sign(sequence_advantage)
    reward_type = _cfg_get(flow_opsd_config, "reward_type", "grpo_advantage")
    if reward_type == "grpo_advantage":
        R = sequence_advantage
    elif reward_type == "raw_score":
        R = reward_signal.sum(dim=-1)
    else:
        raise ValueError(f"Unsupported flow_opsd.reward_type={reward_type!r}")
    G_q = G_q_raw * advantage_sign
    logRtildeF = seq_logp_ref_for_reward + beta_q * G_q + eta_R * R
    b_old = seq_logp_old - logRtildeF

    # beta_q=0 is the FlowRL-compatible P0 path: the privileged gain is absent,
    # so every non-empty rollout participates regardless of reprompt availability.
    if beta_zero or gate_no_context == "keep":
        valid = raw_lengths > 0
    else:
        valid = (self_distillation_mask > 0.5) & (raw_lengths > 0)

    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uid_list):
        groups[uid].append(idx)

    flowopsd_mask = valid.to(dtype=dtype)
    baseline = torch.zeros(batch_size, device=device, dtype=dtype)
    degenerate_groups = 0
    for indices in groups.values():
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_valid = valid[idx_tensor]
        required_group_valid = 1 if beta_zero else min_group_valid
        if int(group_valid.sum().item()) < required_group_valid:
            degenerate_groups += 1
            flowopsd_mask[idx_tensor] = 0.0
            continue
        valid_indices = idx_tensor[group_valid]
        group_baseline = b_old[valid_indices].mean()
        baseline[idx_tensor] = group_baseline

    flowopsd_target = (logRtildeF + baseline).detach()
    flowopsd_mask = flowopsd_mask.detach()
    valid_final = flowopsd_mask > 0.5

    metrics: dict[str, float] = {
        "flowopsd/valid_sample_fraction": flowopsd_mask.mean().item() if batch_size else 0.0,
        "flowopsd/degenerate_group_fraction": degenerate_groups / max(len(groups), 1),
        "flowopsd/empty_target_batch": float(not bool(valid_final.any().item())),
        "flowopsd/beta_q": beta_q,
        "flowopsd/eta_R": eta_R,
        "flowopsd/beta_q_schedule_progress": beta_q_schedule_progress,
        "flowopsd/eta_R_schedule_progress": eta_R_schedule_progress,
        "flowopsd/beta_zero_flowrl_path": float(beta_zero),
        "flowopsd/reward_is_grpo_advantage": float(reward_type == "grpo_advantage"),
    }

    token_valid = response_mask.bool()
    if token_valid.any():
        clipped = (delta_raw.abs() >= clip_B) & token_valid
        metrics["flowopsd/delta_clip_frac"] = clipped.float().mean().item()
    else:
        metrics["flowopsd/delta_clip_frac"] = 0.0

    if valid_final.any():
        vf = valid_final
        gq_raw_v = G_q_raw[vf]
        gq_v = G_q[vf]
        r_v = R[vf]
        beta_gq_v = beta_q * gq_v
        eta_r_v = eta_R * r_v
        gq_abs_mean = gq_v.abs().mean()
        r_abs_mean = r_v.abs().mean()
        beta_gq_abs_mean = beta_gq_v.abs().mean()
        eta_r_abs_mean = eta_r_v.abs().mean()
        metrics.update(
            {
                "flowopsd/G_q_raw_mean": gq_raw_v.mean().item(),
                "flowopsd/G_q_raw_abs_mean": gq_raw_v.abs().mean().item(),
                "flowopsd/advantage_sign_mean": advantage_sign[vf].mean().item(),
                "flowopsd/advantage_zero_fraction": int((advantage_sign[vf] == 0).sum().item()) / int(vf.sum().item()),
                "flowopsd/G_q_mean": gq_v.mean().item(),
                "flowopsd/G_q_abs_mean": gq_abs_mean.item(),
                "flowopsd/G_q_std": gq_v.std(unbiased=False).item(),
                "flowopsd/G_q_min": gq_v.min().item(),
                "flowopsd/G_q_max": gq_v.max().item(),
                "flowopsd/beta_q_G_q_mean": beta_gq_v.mean().item(),
                "flowopsd/beta_q_G_q_abs_mean": beta_gq_abs_mean.item(),
                "flowopsd/R_mean": r_v.mean().item(),
                "flowopsd/R_abs_mean": r_abs_mean.item(),
                "flowopsd/R_std": r_v.std(unbiased=False).item(),
                "flowopsd/R_min": r_v.min().item(),
                "flowopsd/R_max": r_v.max().item(),
                "flowopsd/eta_R_R_mean": eta_r_v.mean().item(),
                "flowopsd/eta_R_R_abs_mean": eta_r_abs_mean.item(),
                "flowopsd/R_to_G_q_abs_ratio": (r_abs_mean / gq_abs_mean.clamp(min=1e-8)).item(),
                "flowopsd/weighted_R_to_G_q_abs_ratio": (
                    eta_r_abs_mean / beta_gq_abs_mean.clamp(min=1e-8)
                ).item(),
                "flowopsd/logRtildeF_mean": logRtildeF[vf].mean().item(),
                "flowopsd/logZ_hat_mean": (-baseline[vf]).mean().item(),
                "flowopsd/target_mean": flowopsd_target[vf].mean().item(),
                "flowopsd/length_norm_mean": length_norm[vf].mean().item(),
                "flowopsd/seq_logp_ref_mean": seq_logp_ref[vf].mean().item(),
                "flowopsd/seq_logp_teacher_mean": seq_logp_teacher[vf].mean().item(),
                "flowopsd/seq_logp_old_mean": seq_logp_old[vf].mean().item(),
                "flowopsd/seq_logp_ref_raw_mean": seq_logp_ref_raw[vf].mean().item(),
                "flowopsd/seq_logp_teacher_raw_mean": seq_logp_teacher_raw[vf].mean().item(),
                "flowopsd/seq_logp_old_raw_mean": seq_logp_old_raw[vf].mean().item(),
            }
        )
    else:
        metrics.update(
            {
                "flowopsd/G_q_raw_mean": 0.0,
                "flowopsd/G_q_raw_abs_mean": 0.0,
                "flowopsd/advantage_sign_mean": 0.0,
                "flowopsd/advantage_zero_fraction": 0.0,
                "flowopsd/G_q_mean": 0.0,
                "flowopsd/G_q_abs_mean": 0.0,
                "flowopsd/G_q_std": 0.0,
                "flowopsd/G_q_min": 0.0,
                "flowopsd/G_q_max": 0.0,
                "flowopsd/beta_q_G_q_mean": 0.0,
                "flowopsd/beta_q_G_q_abs_mean": 0.0,
                "flowopsd/R_mean": 0.0,
                "flowopsd/R_abs_mean": 0.0,
                "flowopsd/R_std": 0.0,
                "flowopsd/R_min": 0.0,
                "flowopsd/R_max": 0.0,
                "flowopsd/eta_R_R_mean": 0.0,
                "flowopsd/eta_R_R_abs_mean": 0.0,
                "flowopsd/R_to_G_q_abs_ratio": 0.0,
                "flowopsd/weighted_R_to_G_q_abs_ratio": 0.0,
                "flowopsd/logRtildeF_mean": 0.0,
                "flowopsd/logZ_hat_mean": 0.0,
                "flowopsd/target_mean": 0.0,
                "flowopsd/length_norm_mean": 0.0,
                "flowopsd/seq_logp_ref_mean": 0.0,
                "flowopsd/seq_logp_teacher_mean": 0.0,
                "flowopsd/seq_logp_old_mean": 0.0,
                "flowopsd/seq_logp_ref_raw_mean": 0.0,
                "flowopsd/seq_logp_teacher_raw_mean": 0.0,
                "flowopsd/seq_logp_old_raw_mean": 0.0,
            }
        )

    return flowopsd_target, flowopsd_mask, metrics
