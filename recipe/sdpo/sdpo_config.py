# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SDPO config dataclasses (overlay).

These extend the clean verl 0.7.0 ``FSDPActorConfig`` with a ``self_distillation``
block, without modifying the verl submodule. The yaml at
``recipe/sdpo/config/sdpo_trainer.yaml`` points ``actor_rollout_ref.actor._target_``
at :class:`SDPOFSDPActorConfig` so that ``omega_conf_to_dataclass`` (hydra
``instantiate``) builds this subclass instead of the upstream one.
"""

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.workers.config import FSDPActorConfig

__all__ = ["SDPOSelfDistillationConfig", "SDPOFSDPActorConfig"]


@dataclass
class SDPOSelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Distillation is enabled when ``policy_loss.loss_mode == "sdpo"``.

    Args:
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL,
            in-between=JSD.
        success_reward_threshold (float): Minimum sequence reward to be considered
            successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema",
            "trust-region".
        teacher_update_rate (float): EMA update rate for teacher weights, or
            trust-region mixing coefficient.
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k
            distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt
            (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think>
            tags from successful demonstrations before reprompting.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables
            IS weighting.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution},
            {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses
            {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses
            {feedback_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in
            reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback
            when no solution is available (ignore feedback when solution exists).
        solution_source (str): Source and fallback policy for successful solution
            demonstrations. One of "group_only", "group_first", "external_only",
            "external_first", or "none".
    """

    full_logit_distillation: bool = True
    alpha: float = 0.5
    success_reward_threshold: float = 0.5
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.0
    distillation_topk: Optional[int] = 100
    distillation_add_tail: bool = True
    max_reprompt_len: int = 22528
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = True
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    token_loss_clip: Optional[float] = None
    reprompt_template: str = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
    solution_template: str = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
    feedback_template: str = (
        "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    solution_source: str = "group_only"
    teacher_enable_thinking: Optional[bool] = None

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        valid_teacher_regularization = ["ema", "trust-region"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.token_loss_clip is not None and self.token_loss_clip <= 0:
            raise ValueError(
                f"self_distillation.token_loss_clip must be positive or null, got {self.token_loss_clip}"
            )
        valid_solution_sources = {
            "group_only",
            "group_first",
            "external_only",
            "external_first",
            "none",
        }
        if self.solution_source not in valid_solution_sources:
            raise ValueError(
                "self_distillation.solution_source must be one of "
                f"{sorted(valid_solution_sources)}, got {self.solution_source!r}"
            )


@dataclass
class SDPOFSDPActorConfig(FSDPActorConfig):
    """FSDP actor config extended with a ``self_distillation`` block.

    Everything else is inherited verbatim from the clean verl ``FSDPActorConfig``.
    """

    self_distillation: SDPOSelfDistillationConfig = field(default_factory=SDPOSelfDistillationConfig)
