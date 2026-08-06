# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""FSDP worker wrapper for standalone Anti-SDPO."""

from __future__ import annotations

import logging
import os

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass

from recipe.antisd.antisd_dp_actor import AntiSDDataParallelPPOActor
from recipe.sdpo.sdpo_fsdp_workers import SDPOAsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["AntiSDAsyncActorRolloutRefWorker"]


class AntiSDAsyncActorRolloutRefWorker(SDPOAsyncActorRolloutRefWorker):
    """Actor-rollout-ref worker whose actor uses Anti-SDPO logic."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return

        teacher_module = getattr(self.actor, "teacher_module", None)
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = AntiSDDataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )

        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        sd_cfg = self.config.actor.get("self_distillation", None)
        if teacher_module is None and loss_mode in {"sdpo", "grpo_ca"} and sd_cfg is not None:
            teacher_regularization = sd_cfg.get("teacher_regularization", "ema")
            if teacher_regularization == "trust-region":
                teacher_module = TrustRegionTeacher(
                    ref_module=self.ref_module_fsdp,
                    student_module=self.actor_module_fsdp,
                    mix_coef=sd_cfg.get("teacher_update_rate", 0.0),
                )
            else:
                teacher_module = self.ref_module_fsdp
        self.actor.teacher_module = teacher_module

        if self.rank == 0:
            loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
            ccir_cfg = self.config.actor.get("ccir", {})
            logger.warning(
                "[AntiSDPO] actor wrapped; loss_mode=%s prm_forward_mode=%s prm_sign=%s ca_lambda_mode=%s",
                loss_mode,
                ccir_cfg.get("prm_forward_mode", "jsd_unbiased"),
                ccir_cfg.get("prm_renyi_sign", -1.0),
                ccir_cfg.get("ca_lambda_mode", "teacher_perp"),
            )
