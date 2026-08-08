"""Typed configuration for the OPSD recipe.

OPSD uses the same FSDP actor transport and privileged-context batch format as
SDPO, while selecting a forward-KL/JSD objective and optional point-wise token
clipping.
"""

from dataclasses import dataclass, field

from recipe.sdpo.sdpo_config import SDPOSelfDistillationConfig
from verl.workers.config import FSDPActorConfig


@dataclass
class OPSDSelfDistillationConfig(SDPOSelfDistillationConfig):
    """OPSD defaults matching the released fixed-teacher objective."""

    full_logit_distillation: bool = True
    alpha: float = 0.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.0
    distillation_topk: int | None = 100
    distillation_add_tail: bool = True
    token_loss_clip: float | None = 0.06
    solution_source: str = "external_first"
    dont_reprompt_on_self_success: bool = False


@dataclass
class OPSDFSDPActorConfig(FSDPActorConfig):
    """FSDP actor config extended with the OPSD self-distillation block."""

    self_distillation: OPSDSelfDistillationConfig = field(default_factory=OPSDSelfDistillationConfig)
