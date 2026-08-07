# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""FlowSD config dataclasses.

This recipe keeps SDPO files unchanged and adds a standalone actor config with a
`flowsd` block. The self-distillation prompt config is reused from SDPO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from recipe.sdpo.sdpo_config import SDPOSelfDistillationConfig
from verl.base_config import BaseConfig
from verl.workers.config import FSDPActorConfig

__all__ = ["FlowSDConfig", "FlowSDFSDPActorConfig"]


@dataclass
class FlowSDConfig(BaseConfig):
    """Sequence-level FlowSD trajectory-balance configuration."""

    beta_q: float = 1.0
    eta_R: float = 15.0
    beta_q_start: Optional[float] = None
    beta_q_end: Optional[float] = None
    eta_R_start: Optional[float] = None
    eta_R_end: Optional[float] = None
    schedule_steps: int = 0
    flow_gap_clip_low: float = 5.0
    flow_gap_clip_high: float = 5.0
    importance_ratio_cap: Optional[float] = 1.2
    residual_length_scale: float = 0.0
    clip_B: float = 4.0
    rho: float = 1.0
    lambda_kl: float = 0.0
    use_flowsd_kl: bool = False
    z_estimator: str = "vargrad_group"
    reference_source: str = "frozen_ref"
    objective: str = "tb"
    reward_type: str = "grpo_advantage"
    gate_no_context: str = "drop"
    min_group_valid: int = 2
    w_clip: Optional[float] = None
    w_min: float = 0.0
    w_max: float = 10.0

    def __post_init__(self):
        if self.beta_q < 0:
            raise ValueError(f"flowsd.beta_q must be >= 0, got {self.beta_q}")
        if self.eta_R < 0:
            raise ValueError(f"flowsd.eta_R must be >= 0, got {self.eta_R}")
        for name in ("beta_q_start", "beta_q_end", "eta_R_start", "eta_R_end"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"flowsd.{name} must be >= 0 or None, got {value}")
        if self.schedule_steps < 0:
            raise ValueError(f"flowsd.schedule_steps must be >= 0, got {self.schedule_steps}")
        if self.flow_gap_clip_low < 0:
            raise ValueError(f"flowsd.flow_gap_clip_low must be >= 0, got {self.flow_gap_clip_low}")
        if self.flow_gap_clip_high < 0:
            raise ValueError(f"flowsd.flow_gap_clip_high must be >= 0, got {self.flow_gap_clip_high}")
        if self.importance_ratio_cap is not None and self.importance_ratio_cap <= 0:
            raise ValueError(
                f"flowsd.importance_ratio_cap must be positive or None, got {self.importance_ratio_cap}"
            )
        if self.residual_length_scale < 0:
            raise ValueError(f"flowsd.residual_length_scale must be >= 0, got {self.residual_length_scale}")
        if self.clip_B <= 0:
            raise ValueError(f"flowsd.clip_B must be > 0, got {self.clip_B}")
        if not 0 < self.rho <= 1:
            raise ValueError(f"flowsd.rho must be in (0, 1], got {self.rho}")
        if self.lambda_kl < 0:
            raise ValueError(f"flowsd.lambda_kl must be >= 0, got {self.lambda_kl}")
        if self.z_estimator != "vargrad_group":
            raise ValueError(f"flowsd.z_estimator only supports 'vargrad_group', got {self.z_estimator}")
        if self.reference_source not in {"frozen_ref", "old_log_probs"}:
            raise ValueError(
                "flowsd.reference_source must be 'frozen_ref' or 'old_log_probs', "
                f"got {self.reference_source}"
            )
        if self.objective != "tb":
            raise ValueError(f"flowsd.objective only supports 'tb', got {self.objective}")
        if self.reward_type not in {"raw_score", "grpo_advantage"}:
            raise ValueError(
                "flowsd.reward_type must be 'raw_score' or 'grpo_advantage', "
                f"got {self.reward_type}"
            )
        if self.gate_no_context not in {"drop", "keep"}:
            raise ValueError(f"flowsd.gate_no_context must be 'drop' or 'keep', got {self.gate_no_context}")
        if self.min_group_valid < 1:
            raise ValueError(f"flowsd.min_group_valid must be >= 1, got {self.min_group_valid}")
        if self.w_clip is not None and self.w_clip <= 0:
            raise ValueError(f"flowsd.w_clip must be positive or None, got {self.w_clip}")
        if self.w_min < 0:
            raise ValueError(f"flowsd.w_min must be >= 0, got {self.w_min}")
        if self.w_max <= 0:
            raise ValueError(f"flowsd.w_max must be > 0, got {self.w_max}")
        if self.w_min > self.w_max:
            raise ValueError(f"flowsd.w_min must be <= w_max, got {self.w_min}>{self.w_max}")


@dataclass
class FlowSDFSDPActorConfig(FSDPActorConfig):
    """FSDP actor config extended with SDPO reprompt config and FlowSD config."""

    self_distillation: SDPOSelfDistillationConfig = field(default_factory=SDPOSelfDistillationConfig)
    flowsd: FlowSDConfig = field(default_factory=FlowSDConfig)
