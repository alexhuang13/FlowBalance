# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""SDPO data-parallel actor (overlay).

Subclasses stable_rl's ``CustomDataParallelPPOActor`` and adds:
  * a ``teacher_module`` (EMA copy of the student, or a :class:`TrustRegionTeacher`),
  * ``_update_teacher`` (EMA update),
  * ``_forward_micro_batch_distill`` (a dict-returning forward that additionally
    exposes full / top-k log-probs and accepts an explicit ``module``), and
  * an ``update_policy`` ``sdpo`` branch that runs the teacher forward and
    :func:`compute_self_distillation_loss`.

The inherited tuple-returning ``_forward_micro_batch`` / ``compute_log_prob`` are
left untouched so the worker's ``output, entropys = actor.compute_log_prob(...)``
contract still holds. For non-sdpo ``loss_mode`` we fully delegate to the parent
``update_policy`` (i.e. the GRPO baseline is unchanged).
"""

import logging
import os
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    slice_input_tensor,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
)

from core.workers.actor.dp_actor import CustomDataParallelPPOActor
from recipe.sdpo.sdpo_core_algos import compute_self_distillation_loss

__all__ = ["TrustRegionTeacher", "SDPODataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _has_real_multi_modal_inputs(multi_modal_inputs) -> bool:
    """Return True only when the non-tensor field contains actual modality tensors."""
    if multi_modal_inputs is None:
        return False
    for inputs in multi_modal_inputs:
        inputs = inputs.data if hasattr(inputs, "data") else inputs
        if inputs is None:
            continue
        if isinstance(inputs, dict):
            if any(value is not None for value in inputs.values()):
                return True
            continue
        return True
    return False


class TrustRegionTeacher(nn.Module):
    """Teacher whose logits are a linear interpolation between a frozen reference
    model and the live student: ``logits = lerp(ref, student, mix_coef)``."""

    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class SDPODataParallelPPOActor(CustomDataParallelPPOActor):
    """SDPO actor: GRPO-compatible actor plus a self-distillation update path."""

    def __init__(self, config, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        super().__init__(config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.teacher_module: Optional[nn.Module] = None

    # ------------------------------------------------------------------ teacher

    def _update_teacher(self) -> None:
        """EMA update of the teacher weights towards the student."""
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode not in {"sdpo", "flowsd", "opsd"}:
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if teacher_regularization != "ema":
            return
        update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
        if update_rate == 0.0:
            return
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("EMA teacher requires a separate teacher_module in the actor worker.")
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_module.parameters(),
                self.actor_module.parameters(),
            ):
                student_data = student_param.data.to(device=teacher_param.device)
                teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)

    # ------------------------------------------------------- distillation forward

    def _forward_micro_batch_distill(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """Dict-returning forward that additionally exposes full / top-k log-probs.

        Returns a dict with ``log_probs`` and optionally ``entropys`` / ``all_logps``
        / ``topk_logps`` / ``topk_indices``.
        """
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(getattr(model, "module", model).config, "vision_config")
                    if is_vlm_model:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)
                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    all_logps_rmpad = torch.log_softmax(logits_rmpad, dim=-1) if compute_all_logps else None

                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    if calculate_entropy:
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(logits_rmpad, topk, dim=-1)
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(logits_rmpad, dim=-1, index=topk_indices_rmpad)
                        logsumexp_rmpad = torch.logsumexp(logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                if self.use_ulysses_sp:
                    log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                            )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if compute_all_logps:
                        all_logps_rmpad = all_logps_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                    )
                if compute_all_logps:
                    full_all_logps = pad_input(
                        hidden_states=all_logps_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad, indices=indices, batch=batch_size, seqlen=seqlen
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )

                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]
                if compute_all_logps:
                    all_logps = full_all_logps[:, -response_length - 1 : -1, :]
                if use_topk:
                    topk_logps = full_topk_logps[:, -response_length - 1 : -1, :]
                    if return_topk_indices:
                        topk_indices = full_topk_indices[:, -response_length - 1 : -1, :]

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]
                else:
                    logits = output.logits
                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, logits.size(-1))
                            topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    # --------------------------------------------------------------- update loop

    @GPUMemoryLogger(role="sdpo dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if loss_mode not in {"sdpo", "opsd"}:
            # GRPO / other baselines are unchanged.
            return super().update_policy(data)

        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        assert self_distillation_cfg is not None, "loss_mode=sdpo/opsd requires actor.self_distillation config."

        self_distillation_required_keys = {
            "teacher_input_ids",
            "teacher_attention_mask",
            "teacher_position_ids",
            "self_distillation_mask",
        }
        missing = self_distillation_required_keys - set(data.batch.keys())
        assert not missing, f"Missing required SDPO keys: {missing}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        select_keys.extend(list(self_distillation_required_keys))
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        multi_modal_inputs = data.non_tensor_batch.get("multi_modal_inputs", None)
        has_multi_modal_inputs = _has_real_multi_modal_inputs(multi_modal_inputs)
        assert not has_multi_modal_inputs, "Multi-modal inputs are not supported for SDPO distillation."

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=[])

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
        if teacher_regularization == "trust-region" and self.use_fused_kernels:
            raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
        return_all_logps = self_distillation_cfg.full_logit_distillation and not self_distillation_cfg.distillation_topk
        distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None

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
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode
                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask")
                    if self_distillation_mask is not None and self_distillation_mask.sum().item() == 0:
                        from verl.utils.py_functional import append_to_dict

                        append_to_dict(metrics, {"self_distillation/empty_target_batch": True})
                        continue

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    outputs = self._forward_micro_batch_distill(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    if getattr(self.config, "use_rollout_log_probs", False):
                        old_log_prob = model_inputs["old_log_probs"]
                    elif on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]

                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

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
                            return_all_logps=return_all_logps,
                            distill_topk=distill_topk,
                            topk_indices=student_topk_indices,
                            module=teacher_model,
                        )
                    teacher_log_prob = teacher_outputs["log_probs"]
                    teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                    teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None

                    pg_loss, pg_metrics = compute_self_distillation_loss(
                        student_log_probs=log_prob,
                        teacher_log_probs=teacher_log_prob,
                        response_mask=response_mask,
                        self_distillation_config=self_distillation_cfg,
                        old_log_probs=old_log_prob,
                        student_all_log_probs=student_all_logps,
                        teacher_all_log_probs=teacher_all_logps,
                        student_topk_log_probs=student_topk_logps,
                        teacher_topk_log_probs=teacher_topk_logps,
                        self_distillation_mask=self_distillation_mask,
                        loss_agg_mode=loss_agg_mode,
                        rollout_is_weights=rollout_is_weights,
                    )
                    pg_metrics["self_distillation/empty_target_batch"] = self_distillation_mask.sum().item() == 0
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss = policy_loss - entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    did_backward = True

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    from verl.utils.py_functional import append_to_dict

                    append_to_dict(metrics, micro_batch_metrics)

                if did_backward:
                    grad_norm = self._optimizer_step()
                    if torch.isfinite(grad_norm).item():
                        did_update = True
                else:
                    grad_norm = torch.zeros((), device=response_mask.device)
                from verl.utils.py_functional import append_to_dict

                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        self.actor_optimizer.zero_grad()
        if did_update:
            self._update_teacher()
        return metrics
