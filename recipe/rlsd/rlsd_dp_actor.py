"""RLSD actor: standard clipped GRPO with teacher-conditioned advantage weights."""
from __future__ import annotations

from collections import defaultdict
import logging
import os
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import POLICY_LOSS_REGISTRY, agg_loss, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.py_functional import append_to_dict

from recipe.sdpo.sdpo_dp_actor import SDPODataParallelPPOActor
from recipe.rlsd.rlsd_core_algos import build_rlsd_advantages

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RLSDDataParallelPPOActor(SDPODataParallelPPOActor):
    @GPUMemoryLogger(role="rlsd dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        if self.config.policy_loss.get("loss_mode", "vanilla") != "rlsd":
            return super().update_policy(data)

        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        rlsd_cfg = self.config.rlsd
        required = {
            "responses", "response_mask", "input_ids", "attention_mask", "position_ids",
            "old_log_probs", "advantages", "teacher_input_ids", "teacher_attention_mask",
            "teacher_position_ids", "self_distillation_mask",
        }
        missing = required - set(data.batch.keys())
        assert not missing, f"Missing RLSD keys: {missing}"
        keys = list(required)
        if self.config.use_kl_loss:
            keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch:
            keys.append("rollout_is_weights")
        data = data.select(batch_keys=keys, non_tensor_batch_keys=[])
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        metrics = defaultdict(list)
        did_update = False

        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                self.actor_optimizer.zero_grad()
                did_backward = False
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    calculate_entropy = self.config.calculate_entropy or self.config.entropy_coeff != 0
                    outputs = self._forward_micro_batch_distill(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs.get("entropys")
                    old_log_prob = log_prob.detach() if on_policy and not getattr(self.config, "use_rollout_log_probs", False) else model_inputs["old_log_probs"]
                    teacher_inputs = {
                        "responses": model_inputs["responses"],
                        "input_ids": model_inputs["teacher_input_ids"],
                        "attention_mask": model_inputs["teacher_attention_mask"],
                        "position_ids": model_inputs["teacher_position_ids"],
                    }
                    with torch.no_grad():
                        teacher_lp = self._forward_micro_batch_distill(
                            teacher_inputs,
                            temperature=temperature,
                            module=self.teacher_module or self.actor_module,
                        )["log_probs"]
                    adv, rlsd_metrics = build_rlsd_advantages(
                        teacher_log_probs=teacher_lp,
                        student_log_probs=log_prob,
                        advantages=model_inputs["advantages"],
                        response_mask=response_mask,
                        self_distillation_mask=model_inputs["self_distillation_mask"],
                        lam=float(data.meta_info.get("rlsd_lambda", rlsd_cfg.lambda_)),
                        clip_range=rlsd_cfg.clip_range,
                        negative_only=rlsd_cfg.negative_only,
                    )
                    loss_fn = POLICY_LOSS_REGISTRY["vanilla"]
                    pg_loss, pg_metrics = loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=adv,
                        response_mask=response_mask,
                        loss_agg_mode=self.config.loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=model_inputs.get("rollout_is_weights"),
                    )
                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None and self.config.entropy_coeff != 0:
                        entropy_loss = agg_loss(entropy, response_mask, self.config.loss_agg_mode)
                        policy_loss = policy_loss - self.config.entropy_coeff * entropy_loss
                    if self.config.use_kl_loss:
                        kld = kl_penalty(logprob=log_prob, ref_logprob=model_inputs["ref_log_prob"], kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(kld, response_mask, self.config.loss_agg_mode)
                        policy_loss = policy_loss + self.config.kl_loss_coef * kl_loss
                        append_to_dict(metrics, {"actor/kl_loss": kl_loss.detach().item()})
                    scale = response_mask.shape[0] / self.config.ppo_mini_batch_size if self.config.use_dynamic_bsz else 1 / self.gradient_accumulation
                    loss = policy_loss * scale
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    did_backward = True
                    append_to_dict(metrics, {"actor/pg_loss": pg_loss.detach().item(), **pg_metrics, **rlsd_metrics})
                if did_backward:
                    grad_norm = self._optimizer_step()
                    did_update = did_update or bool(torch.isfinite(grad_norm).item())
                else:
                    grad_norm = torch.zeros((), device=get_device_id())
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        if did_update:
            interval = int(rlsd_cfg.teacher_sync_interval)
            step = int(data.meta_info.get("global_step", 0))
            if interval > 0 and step > 0 and step % interval == 0:
                if self.teacher_module is not None and self.teacher_module is not self.actor_module:
                    with torch.no_grad():
                        for teacher, student in zip(self.teacher_module.parameters(), self.actor_module.parameters()):
                            teacher.data.copy_(student.data.to(teacher.device))
                append_to_dict(metrics, {"rlsd/teacher_synced": 1.0})
            else:
                append_to_dict(metrics, {"rlsd/teacher_synced": 0.0})
        return metrics
