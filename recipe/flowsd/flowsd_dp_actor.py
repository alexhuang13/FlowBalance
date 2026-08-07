# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""FlowSD data-parallel actor."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.seqlen_balancing import prepare_dynamic_batch

from recipe.sdpo.sdpo_dp_actor import SDPODataParallelPPOActor, TrustRegionTeacher

__all__ = ["FlowSDDataParallelPPOActor", "TrustRegionTeacher"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _append_to_metrics(metrics: dict, values: dict) -> None:
    from verl.utils.py_functional import append_to_dict

    append_to_dict(metrics, values)


class FlowSDDataParallelPPOActor(SDPODataParallelPPOActor):
    """SDPO-compatible actor with a length-normalized sequence-level FlowSD branch."""

    @GPUMemoryLogger(role="flowsd dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "flowsd":
            return super().update_policy(data)
        return self._update_policy_flowsd(data)

    def _update_policy_flowsd(self, data: DataProto):
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        flow_cfg = getattr(self.config, "flowsd", None)
        if flow_cfg is None:
            raise ValueError("loss_mode=flowsd requires actor.flowsd config.")

        required_keys = {
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "flowsd_target",
            "flowsd_mask",
        }
        missing = required_keys - set(data.batch.keys())
        assert not missing, f"Missing required FlowSD keys: {missing}"

        select_keys = list(required_keys)
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if _cfg_get(flow_cfg, "use_flowsd_kl", False) or _cfg_get(flow_cfg, "lambda_kl", 0.0) > 0:
            if "ref_log_prob" not in data.batch.keys():
                raise ValueError("FlowSD KL requires ref_log_prob in batch.")
            select_keys.append("ref_log_prob")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=[])
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {"actor/pg_loss": 0.0, "actor/kl_loss": 0.0}
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                local_valid = mini_batch.batch["flowsd_mask"].float().sum()
                mini_valid = local_valid.clamp(min=1.0)

                # FSDP collectives must be entered in exactly the same order on every
                # rank. A FlowSD shard can have no valid privileged-context samples
                # even when another DP shard does, so a local ``if local_valid > 0``
                # must never decide whether this rank participates in backward. Use a
                # globally synchronized gate only for the all-empty case; otherwise
                # every rank runs backward, and locally empty ranks contribute a
                # graph-connected zero loss.
                global_valid = local_valid.to(device=get_device_id(), dtype=torch.float32)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(global_valid, op=torch.distributed.ReduceOp.SUM)
                global_has_valid = bool(global_valid.item() > 0)

                if not global_has_valid:
                    self.actor_optimizer.zero_grad()
                    zero_grad_norm = torch.zeros((), device=get_device_id())
                    _append_to_metrics(
                        metrics,
                        {
                            "flowsd/global_empty_mini_batch": 1.0,
                            "actor/grad_norm": zero_grad_norm.item(),
                        },
                    )
                    continue

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    flowsd_mask = model_inputs["flowsd_mask"].float()
                    flowsd_target = model_inputs["flowsd_target"].float()
                    old_log_prob = model_inputs["old_log_probs"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or entropy_coeff != 0

                    outputs = self._forward_micro_batch_distill(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=False,
                        distill_topk=None,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs.get("entropys") if calculate_entropy else None
                    rho = float(_cfg_get(flow_cfg, "rho", 1.0))
                    lengths = response_mask.float().sum(dim=-1).clamp(min=1.0)
                    length_norm = lengths.pow(rho)
                    seq_logp_raw = (log_prob * response_mask).sum(dim=-1)
                    old_seq_logp_raw = (old_log_prob * response_mask).sum(dim=-1)
                    seq_logp = seq_logp_raw / length_norm
                    old_seq_logp = old_seq_logp_raw / length_norm
                    residual = seq_logp - flowsd_target

                    raw_weight = torch.exp((seq_logp.detach() - old_seq_logp).clamp(min=-20.0, max=20.0))
                    w_clip = _cfg_get(flow_cfg, "w_clip", None)
                    if w_clip is None:
                        weights = torch.ones_like(raw_weight)
                    else:
                        w_min = float(_cfg_get(flow_cfg, "w_min", 0.0))
                        w_max = min(float(_cfg_get(flow_cfg, "w_max", 10.0)), float(w_clip))
                        weights = raw_weight.clamp(min=w_min, max=w_max)

                    tb_per_sample = weights.detach() * residual.pow(2)
                    pg_loss = (tb_per_sample * flowsd_mask).sum() / mini_valid.to(tb_per_sample.device)
                    policy_loss = pg_loss
                    micro_metrics = {
                        "flowsd/tb_loss": pg_loss.detach().item(),
                        "flowsd/residual_abs_mean": (
                            residual.detach().abs()[flowsd_mask > 0.5].mean().item()
                            if (flowsd_mask > 0.5).any()
                            else 0.0
                        ),
                        "flowsd/seq_logp_student_mean": (
                            seq_logp.detach()[flowsd_mask > 0.5].mean().item()
                            if (flowsd_mask > 0.5).any()
                            else 0.0
                        ),
                        "flowsd/seq_logp_student_raw_mean": (
                            seq_logp_raw.detach()[flowsd_mask > 0.5].mean().item()
                            if (flowsd_mask > 0.5).any()
                            else 0.0
                        ),
                        "flowsd/target_actor_mean": (
                            flowsd_target.detach()[flowsd_mask > 0.5].mean().item()
                            if (flowsd_mask > 0.5).any()
                            else 0.0
                        ),
                        "flowsd/micro_valid": flowsd_mask.sum().item(),
                    }
                    if raw_weight.numel() > 0:
                        denom = weights.pow(2).sum().clamp(min=1e-8)
                        micro_metrics["flowsd/nESS"] = (weights.sum().pow(2) / (weights.numel() * denom)).item()
                        micro_metrics["flowsd/is_weight_raw_mean"] = raw_weight.mean().item()
                        micro_metrics["flowsd/is_weight_raw_max"] = raw_weight.max().item()
                        micro_metrics["flowsd/is_weight_mean"] = weights.mean().item()
                        micro_metrics["flowsd/is_weight_max"] = weights.max().item()

                    lambda_kl = float(_cfg_get(flow_cfg, "lambda_kl", 0.0))
                    if _cfg_get(flow_cfg, "use_flowsd_kl", False) and lambda_kl > 0:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type)
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + lambda_kl * kl_loss
                        micro_metrics["flowsd/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_loss"] += kl_loss.detach().item()

                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss = policy_loss - entropy_agg * entropy_coeff

                    # Always participate in backward when the global mini-batch
                    # contains any valid sample. On a locally empty rank, ``pg_loss``
                    # is zero but remains connected to ``log_prob``, producing zero
                    # gradients while preserving the FSDP collective sequence.
                    if self.scaler is not None:
                        self.scaler.scale(policy_loss).backward()
                    else:
                        policy_loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item()
                    _append_to_metrics(metrics, micro_metrics)

                # All ranks that entered backward must also enter gradient clipping
                # and optimizer step, because FSDP clip_grad_norm_ performs its own
                # collective synchronization.
                grad_norm = self._optimizer_step()
                _append_to_metrics(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics
