# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Standalone Anti-SDPO actor.

``loss_mode=grpo_ca`` is the full Anti-SDPO path.  It keeps the normal GRPO
objective, builds a token-level Anti-SD PRM from student/teacher log-probs, and
feeds the refined token advantages to the standard PPO clipped loss.
"""

from __future__ import annotations

import logging
import os

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.seqlen_balancing import prepare_dynamic_batch

from recipe.antisd.antisd_core_algos import (
    compute_anti_sdpo_refined_advantages,
    compute_anti_self_distillation_loss,
)
from recipe.sdpo.sdpo_dp_actor import SDPODataParallelPPOActor, TrustRegionTeacher

__all__ = ["AntiSDDataParallelPPOActor", "TrustRegionTeacher"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _append_to_metrics(metrics: dict, new_metrics: dict) -> None:
    from verl.utils.py_functional import append_to_dict

    append_to_dict(metrics, new_metrics)


class AntiSDDataParallelPPOActor(SDPODataParallelPPOActor):
    """SDPO-compatible actor with a full Anti-SDPO ``grpo_ca`` branch."""

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self._global_training_steps = 0

    @GPUMemoryLogger(role="antisd dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode == "sdpo":
            import recipe.sdpo.sdpo_dp_actor as sdpo_actor_module

            original_loss_fn = sdpo_actor_module.compute_self_distillation_loss
            sdpo_actor_module.compute_self_distillation_loss = compute_anti_self_distillation_loss
            try:
                return super().update_policy(data)
            finally:
                sdpo_actor_module.compute_self_distillation_loss = original_loss_fn
        if loss_mode != "grpo_ca":
            return super().update_policy(data)

        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        ccir_cfg = getattr(self.config, "ccir", None)
        assert self_distillation_cfg is not None, "loss_mode=grpo_ca requires actor.self_distillation config."
        assert ccir_cfg is not None, "loss_mode=grpo_ca requires actor.ccir config."

        required_keys = {
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
            "self_distillation_mask",
        }
        missing = required_keys - set(data.batch.keys())
        assert not missing, f"Missing required Anti-SDPO keys: {missing}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            *required_keys,
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=[])
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
        if teacher_regularization == "trust-region" and self.use_fused_kernels:
            raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")

        metrics = {"actor/pg_loss": 0.0, "actor/kl_loss": 0.0}
        did_update = False

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                did_backward = False

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    self_distillation_mask = model_inputs.get("self_distillation_mask")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or entropy_coeff != 0

                    loss_scale_factor = (
                        response_mask.shape[0] / self.config.ppo_mini_batch_size
                        if self.config.use_dynamic_bsz
                        else 1 / self.gradient_accumulation
                    )

                    outputs = self._forward_micro_batch_distill(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=False,
                        distill_topk=None,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs.get("entropys") if calculate_entropy else None

                    if getattr(self.config, "use_rollout_log_probs", False):
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()

                    teacher_inputs = {
                        "responses": model_inputs["responses"],
                        "input_ids": model_inputs["teacher_input_ids"],
                        "attention_mask": model_inputs["teacher_attention_mask"],
                        "position_ids": model_inputs["teacher_position_ids"],
                    }
                    teacher_model = self.teacher_module or self.actor_module
                    if teacher_regularization == "trust-region" and (
                        self.teacher_module is None or self.teacher_module is self.actor_module
                    ):
                        raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                    with torch.no_grad():
                        teacher_outputs = self._forward_micro_batch_distill(
                            teacher_inputs,
                            temperature=temperature,
                            calculate_entropy=False,
                            return_all_logps=False,
                            distill_topk=None,
                            module=teacher_model,
                        )
                    teacher_log_prob = teacher_outputs["log_probs"]

                    refined_advantages, ca_metrics = compute_anti_sdpo_refined_advantages(
                        log_prob=log_prob,
                        old_log_prob=old_log_prob,
                        teacher_log_prob=teacher_log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        ccir_config=ccir_cfg,
                        self_distillation_mask=self_distillation_mask,
                        actor_state=self,
                        current_step=self._global_training_steps,
                    )

                    policy_loss_fn = get_policy_loss_fn("vanilla")
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=refined_advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    pg_metrics.update(ca_metrics)
                    pg_metrics["grpo_ca/pg_loss"] = pg_loss.detach().item()
                    pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        pg_metrics.update(
                            compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                        )

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        pg_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss = policy_loss - entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        pg_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    did_backward = True

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    _append_to_metrics(metrics, pg_metrics)

                if did_backward:
                    grad_norm = self._optimizer_step()
                    if torch.isfinite(grad_norm).item():
                        did_update = True
                else:
                    grad_norm = torch.zeros((), device=response_mask.device)
                _append_to_metrics(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
            self._global_training_steps += 1
        return metrics
