# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Standalone Anti-SDPO config dataclasses.

This package is intentionally separate from ``recipe.sdpo``.  The SDPO recipe is
left untouched; Anti-SDPO adds its own actor config and its own CCIR / PRM block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from recipe.sdpo.sdpo_config import SDPOSelfDistillationConfig
from verl.base_config import BaseConfig
from verl.workers.config import FSDPActorConfig

__all__ = [
    "AntiSDOptions",
    "AntiSDCCIRConfig",
    "AntiSDSelfDistillationConfig",
    "AntiSDFSDPActorConfig",
]


@dataclass
class AntiSDOptions(BaseConfig):
    """Compatibility switch for pure SDPO-style signed distillation."""

    enabled: bool = True
    distillation_sign: float = -1.0


@dataclass
class AntiSDCCIRConfig(BaseConfig):
    """Anti-SDPO / GRPO-CA pointwise reward-model configuration.

    The default matches the AntiSD short-context launcher: GRPO + token-level
    JSD PRM, sequence normalization, teacher-perplexity lambda controller, and
    anti-distillation sign ``-1``.
    """

    enabled: bool = False
    num_contrastive: int = 1
    prm_weight: float = 0.0
    kl_coeff: float = 0.0

    # GRPO-CA composition
    ca_mode: str = "additive"  # additive, multiplicative, rlsd
    ca_lambda: float = 0.1
    ca_lambda_mode: str = "teacher_perp"  # fixed, teacher_perp, prm_strength
    ca_lambda_tppl_scope: str = "per_seq"  # per_seq, batch_mean
    ca_lambda_min: float = -0.02
    ca_lambda_max: float = 0.5
    ca_lambda_perp_target: float = 0.0
    ca_lambda_perp_target_ratio: float = 0.0
    ca_lambda_perp_delta: float = 0.10
    ca_lambda_perp_reactivate_target: float = 0.0
    ca_lambda_perp_reactivate_ratio: float = 0.0
    ca_lambda_perp_alpha: float = 2.0
    ca_lambda_perp_mask: float = 3.0
    ca_lambda_warmup_steps: int = 5
    ca_lambda_step_cutoff: int = -1
    ca_lambda_mean_shift: bool = False
    ca_lambda_mean_shift_always: bool = False
    orm_weight: float = 1.0

    # PRM construction and shaping
    prm_construction: str = "reverse"
    prm_gamma: float = 1.0
    prm_forward_mode: str = "jsd_unbiased"  # none, token_is, renyi_unbiased, jsd_unbiased
    prm_forward_beta: float = 1.0
    prm_forward_log_clip: float = 5.0
    prm_renyi_alpha: float = 1.0
    prm_renyi_sign: float = -1.0
    prm_renyi_virtual_alpha: float = 1.0
    prm_u_clip_mode: str = "adaptive"  # none, adaptive
    prm_u_clip_k_sigma: float = 2.0
    prm_u_clip_sigma_ref_fixed: float = 0.0
    prm_normalize: bool = True
    prm_normalize_mode: str = "sequence"  # none, batch, sequence, sequence_demean
    prm_clip: Optional[float] = 3.0
    prm_seq_demean: bool = False
    prm_anchor_to_orm: bool = False
    prm_tanh_tau: Optional[float] = None
    prm_length_mask_threshold: int = 12000
    prm_max_position: int = 0

    # Optional stabilizers kept for parity / ablations.
    prm_entropy_neutral: str = "none"
    entropy_gate_mode: str = "none"
    entropy_gate_h_ratio_low: float = 0.45
    entropy_gate_h_ratio_high: float = 0.60
    maxent_coeff: str = "none"
    maxent_alpha: float = 0.0
    maxent_lr: float = 0.01
    maxent_h_target: Optional[float] = None
    maxent_position: str = "advantage"
    maxent_conditional: bool = False
    maxent_entropy_gate: bool = False
    length_penalty_alpha: float = 0.0
    length_penalty_target: float = 10000.0

    # RLSD ablation.
    rlsd_eps_w: float = 0.2
    rlsd_lambda: float = 1.0

    # Future/contrastive hooks are present so Hydra overrides from AntiSD do not fail.
    si_mode: str = "none"
    si_reference: str = "bare"
    si_model: str = "ema"
    si_lambda: float = 1.0
    si_wrong_label: str = "same"
    contrastive_brake_beta: float = 0.0
    contrastive_brake_adaptive: bool = False
    contrastive_brake_length_target: float = 10000.0
    position_debias_mode: str = "none"
    future_conf_gamma: float = 0.0
    kl_ref_beta: float = 0.0
    ccir_cross_problem: bool = False
    ccir_cross_problem_mode: str = "full"
    ccir_cross_problem_beta: float = 1.0
    ccir_cross_problem_alpha: float = 0.5

    def __post_init__(self):
        if self.num_contrastive < 1:
            raise ValueError(f"ccir.num_contrastive must be >= 1, got {self.num_contrastive}")
        if self.ca_mode not in {"additive", "multiplicative", "rlsd"}:
            raise ValueError(f"unsupported ccir.ca_mode={self.ca_mode}")
        if self.ca_lambda_mode not in {"fixed", "teacher_perp", "prm_strength"}:
            raise ValueError(f"unsupported ccir.ca_lambda_mode={self.ca_lambda_mode}")
        if self.prm_forward_mode not in {"none", "token_is", "renyi_unbiased", "jsd_unbiased"}:
            raise ValueError(f"unsupported ccir.prm_forward_mode={self.prm_forward_mode}")
        if self.prm_normalize_mode not in {"none", "batch", "sequence", "sequence_demean"}:
            raise ValueError(f"unsupported ccir.prm_normalize_mode={self.prm_normalize_mode}")
        if self.prm_renyi_sign not in (-1.0, 1.0):
            raise ValueError(f"ccir.prm_renyi_sign must be -1.0 or +1.0, got {self.prm_renyi_sign}")
        if self.prm_forward_log_clip <= 0:
            raise ValueError("ccir.prm_forward_log_clip must be > 0")
        if self.prm_u_clip_mode not in {"none", "adaptive"}:
            raise ValueError(f"unsupported ccir.prm_u_clip_mode={self.prm_u_clip_mode}")


@dataclass
class AntiSDSelfDistillationConfig(SDPOSelfDistillationConfig):
    """Self-distillation prompt config plus AntiSD-specific prompt knobs."""

    max_solution_tokens: Optional[int] = 3072
    solution_selection: str = "random"  # random, prefer_short
    truncate_solution_at_correct_answer: bool = True
    solution_source: str = "group_only"  # group_first, external_first, external_only, group_only
    solution_content: str = "full"  # full, feedback_only
    solution_mode: str = "normal"
    reprompt_style: str = "suffix"
    provide_ground_truth_in_feedback: bool = False
    require_solution_for_distillation: bool = False
    remove_answer_from_solution: bool = False
    teacher_prompt_suffix: str = ""
    reprompt_template_feedback_only: str = (
        "{prompt}{feedback}\n\nBased on the feedback above, please rethink the problem carefully and try to solve it again.\n"
    )
    reprompt_system_prefix_template: str = "Here is a previous attempt at the problem that follows:{solution}{feedback}"
    wrong_solution_template: str = "\nHere is an incorrect attempt:\n\n{successful_previous_attempt}\n\n"
    anti_sd: AntiSDOptions = field(default_factory=AntiSDOptions)


@dataclass
class AntiSDFSDPActorConfig(FSDPActorConfig):
    self_distillation: AntiSDSelfDistillationConfig = field(default_factory=AntiSDSelfDistillationConfig)
    ccir: AntiSDCCIRConfig = field(default_factory=AntiSDCCIRConfig)
