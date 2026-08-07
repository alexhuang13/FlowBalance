# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""FlowSD Ray trainer hook."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from recipe.flowsd.flowsd_core_algos import compute_flowsd_target, resolve_flowsd_coefficients
from recipe.sdpo.sdpo_ray_trainer import SDPORayPPOTrainer

__all__ = ["FlowSDRayPPOTrainer"]


class FlowSDRayPPOTrainer(SDPORayPPOTrainer):
    """Adds FlowSD target construction while preserving SDPO and vanilla paths."""

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Print and log validation samples with a step-dependent selection."""
        generations_to_log = int(self.config.trainer.get("log_val_generations", 0))
        if generations_to_log <= 0:
            return

        import numpy as np

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42 + int(self.global_steps))
        rng.shuffle(samples)
        samples = samples[:generations_to_log]

        def shorten(text: str, limit: int) -> str:
            text = str(text)
            return text if len(text) <= limit else text[:limit] + "\n...[truncated]..."

        print(f"[FlowSD validation] step={self.global_steps}, showing {len(samples)} samples")
        for idx, (sample_input, sample_output, sample_score) in enumerate(samples, start=1):
            print(f"[FlowSD validation][{idx}] score={sample_score}")
            print(f"[FlowSD validation][{idx}] input:\n{shorten(sample_input, 2000)}")
            print(f"[FlowSD validation][{idx}] output:\n{shorten(sample_output, 4000)}")

        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _update_actor(self, batch: DataProto) -> DataProto:
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "flowsd":
            return super()._update_actor(batch)

        flow_cfg = self.config.actor_rollout_ref.actor.get("flowsd", None)
        beta_q, _, _, _ = resolve_flowsd_coefficients(flow_cfg, int(self.global_steps))
        beta_zero = abs(beta_q) <= 1e-12
        reward_type = flow_cfg.get("reward_type", "grpo_advantage")
        if reward_type == "grpo_advantage":
            if "advantages" not in batch.batch:
                raise ValueError("flowsd.reward_type=grpo_advantage requires batch advantages")
            reward_signal = batch.batch["advantages"]
        elif reward_type == "raw_score":
            reward_signal = batch.batch["token_level_scores"]
        else:
            raise ValueError(f"Unsupported flowsd.reward_type={reward_type!r}")

        flow_metrics: dict[str, float] = {}
        if beta_zero:
            # FlowRL-compatible P0 path: no privileged gain means no reprompt, no
            # contextual teacher forward, no context gate, and grouping by uid.
            flow_log_probs = self.actor_rollout_wg.compute_flowsd_ref_log_probs(batch)
            batch = batch.union(flow_log_probs)
            teacher_log_prob = None
            self_distillation_mask = None
            group_keys = batch.non_tensor_batch["uid"]
            flow_metrics["flowsd/skipped_privileged_teacher"] = 1.0
        else:
            sd_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
            if sd_cfg is None:
                raise ValueError("loss_mode=flowsd with beta_q>0 requires actor.self_distillation config")
            raw_reward = batch.batch["token_level_scores"]
            reward_extra_infos_dict = self._reconstruct_reward_extra_infos(batch, sd_cfg)
            sd_data = self._maybe_build_flowsd_reprompt_batch(batch, raw_reward, reward_extra_infos_dict)
            if sd_data is not None:
                sd_batch, sd_metrics = sd_data
                batch = batch.union(sd_batch)
                flow_metrics.update(sd_metrics)
            flow_log_probs = self.actor_rollout_wg.compute_flowsd_log_probs(batch)
            batch = batch.union(flow_log_probs)
            teacher_log_prob = batch.batch["teacher_log_prob"]
            self_distillation_mask = batch.batch["self_distillation_mask"]
            group_keys = batch.non_tensor_batch.get("flowsd_group_key", batch.non_tensor_batch["uid"])
            flow_metrics["flowsd/skipped_privileged_teacher"] = 0.0

        target, mask, target_metrics = compute_flowsd_target(
            teacher_log_prob=teacher_log_prob,
            ref_log_prob=batch.batch["teacher_ref_log_prob"],
            old_log_probs=batch.batch["old_log_probs"],
            response_mask=batch.batch["response_mask"],
            reward_signal=reward_signal,
            advantage_signal=batch.batch["advantages"],
            self_distillation_mask=self_distillation_mask,
            uids=group_keys,
            flowsd_config=flow_cfg,
            global_step=int(self.global_steps),
        )
        batch = batch.union(DataProto.from_dict(tensors={"flowsd_target": target, "flowsd_mask": mask}))
        flow_metrics.update(target_metrics)

        actor_output = super(SDPORayPPOTrainer, self)._update_actor(batch)
        if flow_metrics:
            metrics_dict = actor_output.meta_info.setdefault("metrics", {})
            for key, value in flow_metrics.items():
                metrics_dict[key] = [value]
        return actor_output

    def _maybe_build_flowsd_reprompt_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        if cfg is None:
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]

        feedback_list = self._collect_feedback(
            include_environment_feedback=cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )
        success_by_uid = self._collect_solutions_by_uid(
            batch, reward_tensor, success_reward_threshold=cfg.success_reward_threshold
        )
        def _get_group_solutions():
            return [
                self._get_solution(
                    i,
                    success_by_uid,
                    batch.non_tensor_batch["uid"],
                    response_texts,
                    cfg.dont_reprompt_on_self_success,
                    cfg.get("remove_thinking_from_demonstration", False),
                )
                for i in range(batch_size)
            ]

        extra_infos = batch.non_tensor_batch.get("extra_info", np.array([None] * batch_size))
        def _get_external_solution(i: int):
            ei = extra_infos[i] if i < len(extra_infos) else None
            sol = ei.get("solution") if isinstance(ei, dict) else None
            return sol if sol else None

        solution_source = cfg.get("solution_source", "group_only")
        num_source_group = 0
        num_source_external = 0
        if solution_source in ("group_first", "group_only"):
            solution_strs = _get_group_solutions()
            num_source_group = sum(s is not None for s in solution_strs)
            if solution_source == "group_first":
                for i in range(batch_size):
                    if solution_strs[i] is None:
                        ext = _get_external_solution(i)
                        if ext is not None:
                            solution_strs[i] = ext
                            num_source_external += 1
        elif solution_source in ("external_first", "external_only"):
            solution_strs = [_get_external_solution(i) for i in range(batch_size)]
            num_source_external = sum(s is not None for s in solution_strs)
            if solution_source == "external_first":
                grouped = _get_group_solutions()
                for i in range(batch_size):
                    if solution_strs[i] is None and grouped[i] is not None:
                        solution_strs[i] = grouped[i]
                        num_source_group += 1
        elif solution_source == "none":
            solution_strs = [None] * batch_size
        else:
            raise ValueError(f"Unsupported self_distillation.solution_source={solution_source!r}")

        def build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            feedback_only_without_solution = cfg.get("environment_feedback_only_without_solution", False)
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            solution_template = cfg.get("solution_template", "\nCorrect solution:\n\n{successful_previous_attempt}\n\n")
            feedback_template = cfg.get(
                "feedback_template",
                "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n",
            )
            reprompt_template = cfg.get(
                "reprompt_template", "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
            )

            solution_section = ""
            if has_solution:
                solution_section = solution_template.format(successful_previous_attempt=solution_strs[i])
            feedback_section = ""
            if use_feedback:
                feedback_section = feedback_template.format(feedback_raw=feedback_list[i])

            if has_solution or use_feedback:
                reprompt_text = reprompt_template.format(
                    prompt=prompt_texts[i], solution=solution_section, feedback=feedback_section
                )
            else:
                reprompt_text = prompt_texts[i]
            return system_messages + [{"role": "user", "content": reprompt_text}]

        messages = [build_teacher_message(i) for i in range(batch_size)]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", None)
        enable_thinking = apply_chat_template_kwargs.get("enable_thinking", True) if apply_chat_template_kwargs else True
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        feedback_only_without_solution = cfg.get("environment_feedback_only_without_solution", False)
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]
        self_distillation_mask = torch.tensor(
            [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
            dtype=torch.float32,
            device=device,
        )
        group_keys = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            group_keys[i] = (
                str(batch.non_tensor_batch["uid"][i]),
                str(solution_strs[i]) if solution_strs[i] is not None else "",
                str(feedback_list[i]) if feedback_used[i] else "",
            )

        uids = set(batch.non_tensor_batch["uid"])
        num_with_feedback_available = sum(1 for f in feedback_list if f is not None)
        num_with_feedback_used = sum(1 for f in feedback_used if f)
        num_with_solution = sum(1 for s in solution_strs if s is not None)
        metrics = {
            "self_distillation/success_group_fraction": len(
                [uid for uid in uids if len(success_by_uid[uid]) > 0]
            )
            / len(uids),
            "self_distillation/success_sample_fraction": num_with_solution / batch_size,
            "self_distillation/feedback_available_fraction": num_with_feedback_available / batch_size,
            "self_distillation/feedback_used_fraction": num_with_feedback_used / batch_size,
            "self_distillation/reprompt_sample_fraction": self_distillation_mask.float().mean().item(),
            "self_distillation/source_group_fraction": num_source_group / batch_size,
            "self_distillation/source_external_fraction": num_source_external / batch_size,
        }
        return (
            DataProto.from_dict(
                tensors={
                    "teacher_input_ids": teacher_input_ids,
                    "teacher_attention_mask": teacher_attention_mask,
                    "teacher_position_ids": teacher_position_ids,
                    "self_distillation_mask": self_distillation_mask,
                },
                non_tensors={"flowsd_group_key": group_keys},
            ),
            metrics,
        )
