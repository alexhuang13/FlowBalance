# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Standalone Anti-SDPO algorithm utilities.

The implementation follows AntiSD's GRPO-CA formulation:

    A'_t = ORM_WEIGHT * A_GRPO + lambda * PRM_t

where ``PRM_t`` is a token-level student/teacher signal.  The default PRM is the
bounded unbiased JSD signal with ``prm_renyi_sign=-1`` (anti-distillation).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from recipe.sdpo.sdpo_core_algos import compute_self_distillation_loss

__all__ = ["compute_anti_self_distillation_loss", "compute_anti_sdpo_refined_advantages"]


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _valid_mask(response_mask: torch.Tensor, self_distillation_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    valid = response_mask.bool()
    if self_distillation_mask is not None and self_distillation_mask.numel() > 0:
        sd_valid = self_distillation_mask.bool().unsqueeze(-1)
        masked = valid & sd_valid
        if masked.any():
            return masked
    return valid


def _seq_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
    return (x * mask).sum(dim=-1, keepdim=True) / denom


def _normalize_prm(prm: torch.Tensor, response_mask: torch.Tensor, valid: torch.Tensor, mode: str) -> torch.Tensor:
    mask = response_mask.float()
    if mode == "sequence":
        mean = _seq_mean(prm, mask)
        centered = (prm - mean) * mask
        var = _seq_mean(centered.square(), mask)
        return centered / var.sqrt().clamp(min=1e-6) * mask
    if mode == "sequence_demean":
        return (prm - _seq_mean(prm, mask)) * mask
    if mode == "batch":
        if valid.any():
            vals = prm[valid]
            return ((prm - vals.mean()) / vals.std().clamp(min=1e-6)) * mask
        return torch.zeros_like(prm)
    return prm * mask


def _maybe_adaptive_u_clip(
    actor_state: Any,
    base_gap: torch.Tensor,
    response_mask: torch.Tensor,
    ccir_config: Any,
    current_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics = {
        "grpo_ca/u_clip_active": 0.0,
        "grpo_ca/u_clip_sigma_ref": 0.0,
        "grpo_ca/u_clip_range": 0.0,
        "grpo_ca/u_clip_frac": 0.0,
        "grpo_ca/u_std_ratio": 1.0,
    }
    if _cfg_get(ccir_config, "prm_u_clip_mode", "none") != "adaptive":
        return base_gap, metrics

    valid = response_mask.bool()
    if not valid.any():
        return base_gap, metrics

    warmup_steps = int(_cfg_get(ccir_config, "ca_lambda_warmup_steps", 5))
    fixed_sigma = float(_cfg_get(ccir_config, "prm_u_clip_sigma_ref_fixed", 0.0))
    k_sigma = float(_cfg_get(ccir_config, "prm_u_clip_k_sigma", 2.0))
    batch_std = base_gap[valid].std().item() if base_gap[valid].numel() > 1 else 0.0

    if actor_state is None:
        sigma_ref = fixed_sigma if fixed_sigma > 0 else max(batch_std, 1e-6)
    else:
        if fixed_sigma > 0 and not hasattr(actor_state, "_antisd_u_clip_sigma_ref"):
            actor_state._antisd_u_clip_sigma_ref = fixed_sigma
        if fixed_sigma <= 0 and current_step < warmup_steps:
            if not hasattr(actor_state, "_antisd_u_clip_warmup_stds"):
                actor_state._antisd_u_clip_warmup_stds = []
            actor_state._antisd_u_clip_warmup_stds.append(batch_std)
            return base_gap, metrics
        if not hasattr(actor_state, "_antisd_u_clip_sigma_ref"):
            vals = getattr(actor_state, "_antisd_u_clip_warmup_stds", [])
            sigma_ref_local = sum(vals) / len(vals) if vals else max(batch_std, 1e-6)
            sigma_tensor = torch.tensor([sigma_ref_local], device=base_gap.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(sigma_tensor, op=torch.distributed.ReduceOp.AVG)
            actor_state._antisd_u_clip_sigma_ref = max(sigma_tensor.item(), 1e-6)
        sigma_ref = actor_state._antisd_u_clip_sigma_ref

    clip_range = k_sigma * max(float(sigma_ref), 1e-6)
    safe = base_gap.abs() <= clip_range
    clipped_gap = torch.where(safe, base_gap, torch.zeros_like(base_gap))
    metrics.update(
        {
            "grpo_ca/u_clip_active": 1.0,
            "grpo_ca/u_clip_sigma_ref": float(sigma_ref),
            "grpo_ca/u_clip_range": float(clip_range),
            "grpo_ca/u_clip_frac": (~safe[valid]).float().mean().item(),
            "grpo_ca/u_std_ratio": float(batch_std / max(float(sigma_ref), 1e-6)),
        }
    )
    return clipped_gap, metrics


def _compute_lambda(
    actor_state: Any,
    teacher_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    base_kl: torch.Tensor,
    response_mask: torch.Tensor,
    ccir_config: Any,
    current_step: int,
) -> tuple[float, Optional[torch.Tensor], dict[str, float]]:
    mode = _cfg_get(ccir_config, "ca_lambda_mode", "fixed")
    base_lambda = float(_cfg_get(ccir_config, "ca_lambda", 0.1))
    lam_min = float(_cfg_get(ccir_config, "ca_lambda_min", -0.02))
    lam_max = float(_cfg_get(ccir_config, "ca_lambda_max", 0.5))
    metrics: dict[str, float] = {
        "grpo_ca/warmup_active": 0.0,
        "grpo_ca/perp_target": 0.0,
        "grpo_ca/perp_masked_frac": 0.0,
        "grpo_ca/teacher_perplexity": 0.0,
        "grpo_ca/student_perplexity": 0.0,
    }

    if mode == "teacher_perp":
        lengths = response_mask.sum(dim=-1).clamp(min=1.0)
        seq_t_logp = (teacher_log_prob.detach() * response_mask).sum(dim=-1) / lengths
        seq_s_logp = (old_log_prob.detach() * response_mask).sum(dim=-1) / lengths
        seq_t_perp = torch.exp(-seq_t_logp)
        seq_s_perp = torch.exp(-seq_s_logp)
        teacher_perp = seq_t_perp.mean().item()
        metrics["grpo_ca/teacher_perplexity"] = teacher_perp
        metrics["grpo_ca/student_perplexity"] = seq_s_perp.mean().item()

        target = float(_cfg_get(ccir_config, "ca_lambda_perp_target", 0.0))
        warmup_steps = int(_cfg_get(ccir_config, "ca_lambda_warmup_steps", 5))
        if target <= 0 and current_step < warmup_steps:
            if actor_state is not None:
                if not hasattr(actor_state, "_antisd_warmup_perp_values"):
                    actor_state._antisd_warmup_perp_values = []
                if teacher_perp < 10.0:
                    actor_state._antisd_warmup_perp_values.append(teacher_perp)
            metrics["grpo_ca/warmup_active"] = 1.0
            return 0.0, None, metrics

        if target <= 0:
            if actor_state is not None and not hasattr(actor_state, "_antisd_perp_target"):
                vals = sorted(getattr(actor_state, "_antisd_warmup_perp_values", []))
                median = vals[len(vals) // 2] if vals else teacher_perp
                median_tensor = torch.tensor([median], device=teacher_log_prob.device)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(median_tensor, op=torch.distributed.ReduceOp.AVG)
                median = median_tensor.item()
                ratio = float(_cfg_get(ccir_config, "ca_lambda_perp_target_ratio", 0.0))
                delta = float(_cfg_get(ccir_config, "ca_lambda_perp_delta", 0.10))
                actor_state._antisd_perp_target = max(median * ratio if ratio > 0 else median - delta, 1.01)
            target = float(getattr(actor_state, "_antisd_perp_target", max(teacher_perp - 0.10, 1.01)))

        perp_alpha = float(_cfg_get(ccir_config, "ca_lambda_perp_alpha", 2.0))
        log_ratio = torch.log((seq_t_perp / max(target, 1e-6)).clamp(min=0.5, max=3.0))
        scope = _cfg_get(ccir_config, "ca_lambda_tppl_scope", "per_seq")
        if scope == "batch_mean":
            lam_seq = (base_lambda * perp_alpha * log_ratio.mean()).clamp(min=lam_min, max=lam_max).expand_as(seq_t_perp)
        else:
            lam_seq = (base_lambda * perp_alpha * log_ratio).clamp(min=lam_min, max=lam_max)
        mask_threshold = float(_cfg_get(ccir_config, "ca_lambda_perp_mask", 3.0))
        perp_masked = seq_t_perp > mask_threshold
        lam_seq = lam_seq.clone()
        lam_seq[perp_masked] = 0.0
        if bool(_cfg_get(ccir_config, "ca_lambda_mean_shift_always", False)) or (
            bool(_cfg_get(ccir_config, "ca_lambda_mean_shift", False)) and lam_seq.mean().item() < 0
        ):
            lam_seq = lam_seq - lam_seq.mean()
        metrics.update(
            {
                "grpo_ca/perp_target": float(target),
                "grpo_ca/perp_masked_frac": perp_masked.float().mean().item(),
                "grpo_ca/ca_lambda_std": lam_seq.std().item() if lam_seq.numel() > 1 else 0.0,
                "grpo_ca/ca_lambda_min_seq": lam_seq.min().item(),
                "grpo_ca/ca_lambda_max_seq": lam_seq.max().item(),
                "grpo_ca/ca_lambda_pos_frac": (lam_seq > 0).float().mean().item(),
            }
        )
        return lam_seq.mean().item(), lam_seq.unsqueeze(-1), metrics

    if mode == "prm_strength":
        lengths = response_mask.sum(dim=-1).clamp(min=1.0)
        seq_strength = (base_kl.detach().abs() * response_mask).sum(dim=-1) / lengths
        target = float(_cfg_get(ccir_config, "ca_lambda_perp_target", 0.0))
        warmup_steps = int(_cfg_get(ccir_config, "ca_lambda_warmup_steps", 5))
        if target <= 0 and current_step < warmup_steps:
            if actor_state is not None:
                if not hasattr(actor_state, "_antisd_prm_strength_values"):
                    actor_state._antisd_prm_strength_values = []
                actor_state._antisd_prm_strength_values.append(seq_strength.mean().item())
            metrics["grpo_ca/warmup_active"] = 1.0
            return 0.0, None, metrics
        if target <= 0:
            if actor_state is not None and not hasattr(actor_state, "_antisd_prm_strength_target"):
                vals = sorted(getattr(actor_state, "_antisd_prm_strength_values", []))
                median = vals[len(vals) // 2] if vals else seq_strength.mean().item()
                delta = float(_cfg_get(ccir_config, "ca_lambda_perp_delta", 0.10))
                actor_state._antisd_prm_strength_target = max(median * (1.0 - delta), 1e-6)
            target = float(getattr(actor_state, "_antisd_prm_strength_target", seq_strength.mean().item()))
        alpha = float(_cfg_get(ccir_config, "ca_lambda_perp_alpha", 2.0))
        lam_seq = (base_lambda * alpha * torch.log((seq_strength / max(target, 1e-6)).clamp(0.5, 3.0))).clamp(lam_min, lam_max)
        metrics.update(
            {
                "grpo_ca/prm_strength_target": float(target),
                "grpo_ca/ca_lambda_std": lam_seq.std().item() if lam_seq.numel() > 1 else 0.0,
                "grpo_ca/ca_lambda_min_seq": lam_seq.min().item(),
                "grpo_ca/ca_lambda_max_seq": lam_seq.max().item(),
                "grpo_ca/ca_lambda_pos_frac": (lam_seq > 0).float().mean().item(),
            }
        )
        return lam_seq.mean().item(), lam_seq.unsqueeze(-1), metrics

    if int(_cfg_get(ccir_config, "ca_lambda_step_cutoff", -1)) > 0 and current_step >= int(_cfg_get(ccir_config, "ca_lambda_step_cutoff", -1)):
        base_lambda = 0.0
    metrics.update(
        {
            "grpo_ca/ca_lambda_std": 0.0,
            "grpo_ca/ca_lambda_min_seq": base_lambda,
            "grpo_ca/ca_lambda_max_seq": base_lambda,
            "grpo_ca/ca_lambda_pos_frac": 1.0 if base_lambda > 0 else 0.0,
        }
    )
    return base_lambda, None, metrics


def compute_anti_sdpo_refined_advantages(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ccir_config: Any,
    self_distillation_mask: Optional[torch.Tensor] = None,
    actor_state: Any = None,
    current_step: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build GRPO-CA Anti-SDPO token advantages.

    ``log_prob`` keeps gradient for the PPO loss; all PRM signals are detached and
    used only as token-level advantages, matching AntiSD's implementation.
    """
    metrics: dict[str, float] = {}
    response_mask_f = response_mask.float()
    valid = _valid_mask(response_mask_f, self_distillation_mask)

    prm_construction = _cfg_get(ccir_config, "prm_construction", "reverse")
    gamma = float(_cfg_get(ccir_config, "prm_gamma", 1.0))
    if prm_construction == "reverse":
        base_kl = (log_prob - gamma * teacher_log_prob).detach() * response_mask_f
    elif prm_construction == "raw":
        base_kl = (teacher_log_prob - log_prob).detach() * response_mask_f
    elif prm_construction == "self_reward":
        base_kl = log_prob.detach() * response_mask_f
    else:
        # Keep unsupported ablation names runnable by falling back to the paper default.
        base_kl = (log_prob - gamma * teacher_log_prob).detach() * response_mask_f

    base_kl, uclip_metrics = _maybe_adaptive_u_clip(actor_state, base_kl, response_mask_f, ccir_config, current_step)
    metrics.update(uclip_metrics)

    prm_forward_mode = _cfg_get(ccir_config, "prm_forward_mode", "jsd_unbiased")
    forward_gap = (teacher_log_prob - log_prob).detach() * response_mask_f  # u = t - s
    if prm_forward_mode in {"renyi_unbiased", "jsd_unbiased"}:
        forward_gap, fclip_metrics = _maybe_adaptive_u_clip(actor_state, forward_gap, response_mask_f, ccir_config, current_step)
        metrics.update({k: v for k, v in fclip_metrics.items() if k not in metrics or v})

    if prm_forward_mode == "token_is":
        beta = float(_cfg_get(ccir_config, "prm_forward_beta", 1.0))
        clip = float(_cfg_get(ccir_config, "prm_forward_log_clip", 5.0))
        weights = torch.exp((beta * forward_gap).clamp(-clip, clip)) * response_mask_f
        PRM_t = base_kl * weights
        metrics["grpo_ca/forward_token_active"] = 1.0
        metrics["grpo_ca/forward_token_weight_mean"] = weights[valid].mean().item() if valid.any() else 1.0
    elif prm_forward_mode == "renyi_unbiased":
        alpha = float(_cfg_get(ccir_config, "prm_renyi_alpha", 1.0))
        sign = float(_cfg_get(ccir_config, "prm_renyi_sign", -1.0))
        clip = float(_cfg_get(ccir_config, "prm_forward_log_clip", 5.0))
        renyi_log = (alpha * forward_gap).clamp(-clip, clip)
        PRM_t = sign * (torch.exp(renyi_log) - 1.0) * response_mask_f
        metrics.update({"grpo_ca/renyi_active": 1.0, "grpo_ca/renyi_sign": sign})
    elif prm_forward_mode == "jsd_unbiased":
        sign = float(_cfg_get(ccir_config, "prm_renyi_sign", -1.0))
        clip = float(_cfg_get(ccir_config, "prm_forward_log_clip", 5.0))
        gap = forward_gap.clamp(-clip, clip)
        # sign=-1 -> 0.5 * (log(2) - softplus(t-s)), bounded max-JSD direction.
        PRM_t = 0.5 * sign * (torch.nn.functional.softplus(gap) - math.log(2.0)) * response_mask_f
        metrics.update(
            {
                "grpo_ca/jsd_active": 1.0,
                "grpo_ca/jsd_sign": sign,
                "grpo_ca/jsd_gap_mean": forward_gap[valid].mean().item() if valid.any() else 0.0,
                "grpo_ca/jsd_gap_abs_mean": forward_gap[valid].abs().mean().item() if valid.any() else 0.0,
            }
        )
    else:
        PRM_t = base_kl

    tau = _cfg_get(ccir_config, "prm_tanh_tau", None)
    if tau is not None and tau > 0:
        PRM_t = torch.tanh(PRM_t / float(tau)) * float(tau)

    PRM_processed = _normalize_prm(PRM_t, response_mask_f, valid, _cfg_get(ccir_config, "prm_normalize_mode", "sequence"))
    prm_clip = _cfg_get(ccir_config, "prm_clip", None)
    if prm_clip is not None:
        PRM_processed = PRM_processed.clamp(-float(prm_clip), float(prm_clip))

    len_threshold = int(_cfg_get(ccir_config, "prm_length_mask_threshold", 0))
    prm_length_masked_frac = 0.0
    if len_threshold > 0:
        long_seq = response_mask_f.sum(dim=-1) >= len_threshold
        prm_length_masked_frac = long_seq.float().mean().item()
        PRM_processed = PRM_processed * (~long_seq).float().unsqueeze(-1)

    max_pos = int(_cfg_get(ccir_config, "prm_max_position", 0))
    if max_pos > 0:
        pos = torch.cumsum(response_mask_f, dim=-1)
        PRM_processed = PRM_processed * (pos <= max_pos).float() * response_mask_f

    ca_lambda, ca_lambda_broadcast, lambda_metrics = _compute_lambda(
        actor_state=actor_state,
        teacher_log_prob=teacher_log_prob,
        old_log_prob=old_log_prob,
        base_kl=base_kl,
        response_mask=response_mask_f,
        ccir_config=ccir_config,
        current_step=current_step,
    )
    metrics.update(lambda_metrics)

    ca_mode = _cfg_get(ccir_config, "ca_mode", "additive")
    orm_weight = float(_cfg_get(ccir_config, "orm_weight", 1.0))
    if ca_mode == "multiplicative":
        lam = ca_lambda_broadcast if ca_lambda_broadcast is not None else ca_lambda
        refined = advantages * (1.0 + lam * PRM_processed)
    elif ca_mode == "rlsd":
        eps = float(_cfg_get(ccir_config, "rlsd_eps_w", 0.2))
        rlsd_lam = float(_cfg_get(ccir_config, "rlsd_lambda", 1.0))
        w = torch.exp(torch.sign(advantages).detach() * (teacher_log_prob - log_prob).detach())
        w = w.clamp(1.0 - eps, 1.0 + eps)
        refined = advantages * ((1.0 - rlsd_lam) + rlsd_lam * w)
    else:
        if bool(_cfg_get(ccir_config, "prm_anchor_to_orm", False)) and valid.any():
            orm_scale = (orm_weight * advantages[valid]).abs().mean().clamp(min=1e-8)
            prm_scale = PRM_processed[valid].abs().mean().clamp(min=1e-8)
            PRM_processed = PRM_processed * (orm_scale / prm_scale)
        lam = ca_lambda_broadcast if ca_lambda_broadcast is not None else ca_lambda
        refined = orm_weight * advantages + lam * PRM_processed

    if float(_cfg_get(ccir_config, "length_penalty_alpha", 0.0)) > 0:
        alpha = float(_cfg_get(ccir_config, "length_penalty_alpha", 0.0))
        target = float(_cfg_get(ccir_config, "length_penalty_target", 10000.0))
        overshoot = ((response_mask_f.sum(dim=-1, keepdim=True) - target) / max(target, 1.0)).clamp(min=0.0)
        refined = refined - alpha * overshoot * response_mask_f

    if valid.any():
        metrics.update(
            {
                "grpo_ca/base_kl_mean": base_kl[valid].mean().item(),
                "grpo_ca/base_kl_abs_mean": base_kl[valid].abs().mean().item(),
                "grpo_ca/PRM_raw_mean": PRM_t[valid].mean().item(),
                "grpo_ca/PRM_raw_std": PRM_t[valid].std().item() if PRM_t[valid].numel() > 1 else 0.0,
                "grpo_ca/PRM_processed_mean": PRM_processed[valid].mean().item(),
                "grpo_ca/PRM_processed_abs_mean": PRM_processed[valid].abs().mean().item(),
                "grpo_ca/A_seq_mean": advantages[valid].mean().item(),
                "grpo_ca/A_seq_abs_mean": advantages[valid].abs().mean().item(),
                "grpo_ca/refined_adv_mean": refined[valid].mean().item(),
                "grpo_ca/refined_adv_abs_mean": refined[valid].abs().mean().item(),
                "grpo_ca/s_minus_t_mean": (log_prob.detach() - teacher_log_prob.detach())[valid].mean().item(),
                "grpo_ca/t_minus_s_mean": (teacher_log_prob.detach() - log_prob.detach())[valid].mean().item(),
                "grpo_ca/ca_lambda": ca_lambda,
                "grpo_ca/prm_length_masked_frac": prm_length_masked_frac,
                "grpo_ca/orm_weight": orm_weight,
                "grpo_ca/prm_weight": float(_cfg_get(ccir_config, "prm_weight", 0.0)),
                "grpo_ca/ca_mode": {"additive": 0.0, "multiplicative": 1.0, "rlsd": 4.0}.get(ca_mode, -1.0),
                "grpo_ca/prm_normalize_mode": {"none": 0.0, "batch": 1.0, "sequence": 2.0, "sequence_demean": 3.0}.get(
                    _cfg_get(ccir_config, "prm_normalize_mode", "sequence"), -1.0
                ),
            }
        )
        if ca_mode == "additive":
            orm_term = orm_weight * advantages[valid]
            prm_term = (ca_lambda * PRM_processed[valid])
            metrics["grpo_ca/orm_term_abs_mean"] = orm_term.abs().mean().item()
            metrics["grpo_ca/prm_term_abs_mean"] = prm_term.abs().mean().item()
            metrics["grpo_ca/prm_dominance"] = (prm_term.abs().mean() / (orm_term.abs().mean() + prm_term.abs().mean() + 1e-8)).item()

    return refined.detach() * response_mask_f, metrics


def compute_anti_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compatibility path: signed SDPO KL/JSD loss.

    Full Anti-SDPO uses ``loss_mode=grpo_ca`` and
    :func:`compute_anti_sdpo_refined_advantages`; this function keeps the older
    ``loss_mode=sdpo`` overlay runnable.
    """
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=self_distillation_config,
        old_log_probs=old_log_probs,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
        self_distillation_mask=self_distillation_mask,
        loss_agg_mode=loss_agg_mode,
        rollout_is_weights=rollout_is_weights,
    )
    anti_cfg = _cfg_get(self_distillation_config, "anti_sd", {})
    enabled = bool(_cfg_get(anti_cfg, "enabled", True))
    sign = float(_cfg_get(anti_cfg, "distillation_sign", -1.0 if enabled else 1.0))
    if not enabled:
        sign = 1.0
    metrics["anti_sd/enabled"] = enabled
    metrics["anti_sd/distillation_sign"] = sign
    metrics["anti_sd/sd_loss_before_sign"] = loss.detach().item()
    return loss * sign, metrics
