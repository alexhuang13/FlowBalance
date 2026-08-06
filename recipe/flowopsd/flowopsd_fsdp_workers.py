# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""FSDP worker wrapper for FlowOPSD."""

from __future__ import annotations

import logging
import os

import torch

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch

from recipe.flowopsd.flowopsd_dp_actor import FlowOPSDDataParallelPPOActor, TrustRegionTeacher
from recipe.sdpo.sdpo_fsdp_workers import SDPOAsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["FlowOPSDAsyncActorRolloutRefWorker"]


class FlowOPSDAsyncActorRolloutRefWorker(SDPOAsyncActorRolloutRefWorker):
    """Actor-rollout-ref worker with frozen-reference FlowOPSD log-prob APIs."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_actor:
            return

        teacher_module = getattr(self.actor, "teacher_module", None)
        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = FlowOPSDDataParallelPPOActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )

        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        sd_cfg = self.config.actor.get("self_distillation", None)
        if teacher_module is None and loss_mode == "sdpo" and sd_cfg is not None:
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
            logger.warning("[FlowOPSD] actor wrapped; loss_mode=%s", loss_mode)

    def _compute_ref_module_log_prob(
        self,
        data: DataProto,
        input_key: str,
        attention_key: str,
        position_key: str,
        output_key: str,
        module=None,
    ) -> DataProto:
        select_keys = ["responses", input_key, attention_key, position_key]
        selected = data.select(batch_keys=select_keys, non_tensor_batch_keys=[])
        tensors = {
            "responses": selected.batch["responses"],
            "input_ids": selected.batch[input_key],
            "attention_mask": selected.batch[attention_key],
            "position_ids": selected.batch[position_key],
        }
        selected = DataProto.from_dict(tensors=tensors, meta_info=dict(data.meta_info))

        use_dynamic_bsz = self.config.ref.get("log_prob_use_dynamic_bsz", False)
        micro_batch_size = self.config.ref.get("log_prob_micro_batch_size_per_gpu", 1)
        max_token_len = self.config.ref.get("log_prob_max_token_len_per_gpu", None)
        if max_token_len is None:
            max_token_len = self.config.actor.ppo_max_token_len_per_gpu
        temperature = self.config.rollout.temperature

        if use_dynamic_bsz:
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                selected,
                max_token_len=max_token_len * self.actor.ulysses_sequence_parallel_size,
            )
        else:
            micro_batches = selected.split(micro_batch_size)
            batch_idx_list = None

        log_probs = []
        # FlowOPSD evaluates both privileged and original contexts with the same
        # frozen initial reference module. ``module`` remains overridable for the
        # inherited SDPO path, but FlowOPSD never updates the reference weights.
        module = self.ref_module_fsdp if module is None else module
        with self.ulysses_sharding_manager:
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                with torch.no_grad():
                    output = self.actor._forward_micro_batch_distill(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=False,
                        return_all_logps=False,
                        distill_topk=None,
                        module=module,
                    )
                log_probs.append(output["log_probs"])

        output_log_probs = torch.cat(log_probs, dim=0)
        if use_dynamic_bsz:
            output_log_probs = restore_dynamic_batch(output_log_probs, batch_idx_list)
        return DataProto.from_dict(tensors={output_key: output_log_probs}).to("cpu")

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_flowopsd_ref_log_probs(self, data: DataProto) -> DataProto:
        """Return only frozen-reference log-probs for the beta_q=0 path."""
        assert self._is_actor
        return self._compute_ref_module_log_prob(
            data,
            input_key="input_ids",
            attention_key="attention_mask",
            position_key="position_ids",
            output_key="teacher_ref_log_prob",
            module=self.ref_module_fsdp,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_flowopsd_log_probs(self, data: DataProto) -> DataProto:
        """Return frozen-reference log-probs under privileged and original contexts."""
        assert self._is_actor
        teacher = self._compute_ref_module_log_prob(
            data,
            input_key="teacher_input_ids",
            attention_key="teacher_attention_mask",
            position_key="teacher_position_ids",
            output_key="teacher_log_prob",
            module=self.ref_module_fsdp,
        )
        teacher_ref = self._compute_ref_module_log_prob(
            data,
            input_key="input_ids",
            attention_key="attention_mask",
            position_key="position_ids",
            output_key="teacher_ref_log_prob",
            module=self.ref_module_fsdp,
        )
        return teacher.union(teacher_ref)
