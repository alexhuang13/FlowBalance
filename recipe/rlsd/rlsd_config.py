"""RLSD configuration overlays for stable_rl/verl 0.7."""
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from recipe.sdpo.sdpo_config import SDPOFSDPActorConfig


@dataclass
class RLSDReweightConfig(BaseConfig):
    lambda_: float = 0.5
    clip_range: Optional[float] = 0.2
    negative_only: bool = False
    thought_only: bool = False
    lambda_warmup_steps: int = 0
    lambda_decay_steps: int = 60
    teacher_sync_interval: int = 20

    def __post_init__(self):
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"rlsd.lambda_ must be in [0,1], got {self.lambda_}")
        if self.clip_range is not None and self.clip_range < 0:
            raise ValueError(f"rlsd.clip_range must be >=0 or None, got {self.clip_range}")
        if self.lambda_warmup_steps < 0 or self.lambda_decay_steps < 0:
            raise ValueError("RLSD warmup/decay steps must be non-negative")
        if self.teacher_sync_interval < 0:
            raise ValueError("RLSD teacher_sync_interval must be non-negative")


@dataclass
class RLSDFSDPActorConfig(SDPOFSDPActorConfig):
    rlsd: RLSDReweightConfig = field(default_factory=RLSDReweightConfig)
