# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.
"""Standalone Anti-SDPO trainer hook.

This subclasses the existing SDPO trainer but widens the teacher-batch hook to
``loss_mode=grpo_ca`` and adds the prompt options used by AntiSD.  No file under
``recipe/sdpo`` is modified.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from recipe.sdpo.sdpo_ray_trainer import SDPORayPPOTrainer

__all__ = ["AntiSDRayPPOTrainer"]


class AntiSDRayPPOTrainer(SDPORayPPOTrainer):
    """Builds teacher prompts for both signed-SDPO and full GRPO-CA Anti-SDPO."""

    def _update_actor(self, batch: DataProto) -> DataProto:
        sd_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        sd_metrics: dict[str, float] = {}

        if sd_cfg is not None and loss_mode in ("sdpo", "grpo_ca"):
            reward_tensor = batch.batch["token_level_scores"]
            reward_extra_infos_dict = self._reconstruct_reward_extra_infos(batch, sd_cfg)
            sd_data = self._maybe_build_self_distillation_batch(batch, reward_tensor, reward_extra_infos_dict)
            if sd_data is not None:
                sd_batch, sd_metrics = sd_data
                batch = batch.union(sd_batch)

        actor_output = super(SDPORayPPOTrainer, self)._update_actor(batch)
        if sd_metrics:
            metrics_dict = actor_output.meta_info.setdefault("metrics", {})
            for key, value in sd_metrics.items():
                metrics_dict[key] = [value]
        return actor_output

    @staticmethod
    def _head_tail_truncate_tokenized(tokenized_batch: dict, tokenizer, max_len: int, head_ratio: float = 1 / 3) -> dict:
        input_ids = tokenized_batch["input_ids"]
        attention_mask = tokenized_batch["attention_mask"]
        if input_ids.shape[1] <= max_len:
            return tokenized_batch
        marker_ids = tokenizer.encode(" ... ", add_special_tokens=False)
        marker = torch.tensor(marker_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        marker_mask = torch.ones(input_ids.shape[0], len(marker_ids), dtype=attention_mask.dtype, device=attention_mask.device)
        budget = max(max_len - len(marker_ids), 1)
        head_len = int(budget * head_ratio)
        tail_len = budget - head_len
        tokenized_batch["input_ids"] = torch.cat([input_ids[:, :head_len], marker, input_ids[:, -tail_len:]], dim=1)
        tokenized_batch["attention_mask"] = torch.cat(
            [attention_mask[:, :head_len], marker_mask, attention_mask[:, -tail_len:]], dim=1
        )
        return tokenized_batch

    @staticmethod
    def _find_last_boxed_end(text: str) -> Optional[int]:
        start = text.rfind(r"\boxed{")
        if start < 0:
            return None
        depth = 0
        i = start + len(r"\boxed{")
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                if depth == 0:
                    return i + 1
                depth -= 1
            i += 1
        return None

    def _truncate_solution_to_budget(
        self,
        solution_str: str,
        prompt_text: str,
        feedback_text: Optional[str],
        max_reprompt_len: int,
        max_solution_tokens: Optional[int],
        chat_overhead: int = 300,
    ) -> str:
        prompt_len = len(self.tokenizer.encode(prompt_text, add_special_tokens=False))
        feedback_len = len(self.tokenizer.encode(feedback_text, add_special_tokens=False)) if feedback_text else 0
        available = max(max_reprompt_len - prompt_len - feedback_len - chat_overhead, 128)
        if max_solution_tokens is not None:
            available = min(available, int(max_solution_tokens))
        tokens = self.tokenizer.encode(solution_str, add_special_tokens=False)
        if len(tokens) <= available:
            return solution_str
        head_budget = max(1, available // 4)
        tail_budget = max(1, available - head_budget)
        return (
            self.tokenizer.decode(tokens[:head_budget], skip_special_tokens=False)
            + "\n...\n"
            + self.tokenizer.decode(tokens[-tail_budget:], skip_special_tokens=False)
        )

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if cfg is None or loss_mode not in ("sdpo", "grpo_ca"):
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
        success_by_uid = self._collect_solutions_by_uid(batch, reward_tensor, cfg.success_reward_threshold)

        reward_models = batch.non_tensor_batch.get("reward_model", [{} for _ in range(batch_size)])
        extra_infos = batch.non_tensor_batch.get("extra_info", [None for _ in range(batch_size)])
        solution_source = cfg.get("solution_source", "group_only")
        solution_selection = cfg.get("solution_selection", "random")

        def external_solution(i: int) -> Optional[str]:
            info = extra_infos[i] if i < len(extra_infos) else None
            return info.get("solution") if isinstance(info, dict) and info.get("solution") else None

        def group_solution(i: int) -> Optional[str]:
            gt = reward_models[i].get("ground_truth") if isinstance(reward_models[i], dict) else None
            sol = self._get_solution(
                i,
                success_by_uid,
                batch.non_tensor_batch["uid"],
                response_texts,
                cfg.dont_reprompt_on_self_success,
                cfg.get("remove_thinking_from_demonstration", False),
            )
            if sol and cfg.get("truncate_solution_at_correct_answer", False) and gt:
                end = self._find_last_boxed_end(sol)
                if end is not None:
                    sol = sol[:end]
            return sol

        group_solutions = [group_solution(i) for i in range(batch_size)]
        solution_strs: list[Optional[str]] = [None] * batch_size
        num_source_group = 0
        num_source_external = 0
        for i in range(batch_size):
            g = group_solutions[i]
            e = external_solution(i)
            if solution_source == "group_only":
                solution_strs[i] = g
            elif solution_source == "external_only":
                solution_strs[i] = e
            elif solution_source == "external_first":
                solution_strs[i] = e or g
            else:
                solution_strs[i] = g or e
            if solution_strs[i] is not None:
                if solution_strs[i] == g:
                    num_source_group += 1
                else:
                    num_source_external += 1

        if cfg.get("solution_content", "full") == "feedback_only":
            solution_strs = [None for _ in solution_strs]

        max_reprompt_len = int(cfg.max_reprompt_len)
        max_solution_tokens = cfg.get("max_solution_tokens", None)
        feedback_only_without_solution = cfg.get("environment_feedback_only_without_solution", False)
        for i, sol in enumerate(solution_strs):
            if sol is not None:
                effective_feedback = None if feedback_only_without_solution else feedback_list[i]
                solution_strs[i] = self._truncate_solution_to_budget(
                    sol, prompt_texts[i], effective_feedback, max_reprompt_len, max_solution_tokens
                )

        reprompt_style = cfg.get("reprompt_style", "suffix")

        def build_message(i: int) -> list[dict[str, str]]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            solution_section = ""
            if has_solution:
                solution_section = cfg.get(
                    "solution_template", "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
                ).format(successful_previous_attempt=solution_strs[i])

            feedback_section = ""
            if use_feedback:
                feedback_raw = feedback_list[i]
                if not has_solution and cfg.get("provide_ground_truth_in_feedback", False):
                    gt = reward_models[i].get("ground_truth") if isinstance(reward_models[i], dict) else None
                    if gt:
                        feedback_raw = f"{feedback_raw}\nThe correct answer is \\boxed{{{gt}}}."
                feedback_section = cfg.get(
                    "feedback_template",
                    "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n",
                ).format(feedback_raw=feedback_raw)

            if reprompt_style == "multi_turn" and (has_solution or use_feedback):
                retry = "Please try again." if not use_feedback else f"{feedback_list[i]} Please try again."
                return system_messages + [
                    {"role": "user", "content": prompt_texts[i]},
                    {"role": "assistant", "content": solution_strs[i] or ""},
                    {"role": "user", "content": retry},
                ]

            if has_solution or use_feedback:
                template = cfg.get(
                    "reprompt_template", "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
                )
                content = template.format(prompt=prompt_texts[i], solution=solution_section, feedback=feedback_section)
            else:
                suffix = cfg.get("teacher_prompt_suffix", "")
                content = prompt_texts[i] + suffix if suffix else prompt_texts[i]
            return system_messages + [{"role": "user", "content": content}]

        messages = [build_message(i) for i in range(batch_size)]
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
            padding=True,
            truncation=False,
        )
        teacher_prompt = self._head_tail_truncate_tokenized(teacher_prompt, self.tokenizer, max_reprompt_len)
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]
        if cfg.get("require_solution_for_distillation", False):
            mask_values = [solution_strs[i] is not None for i in range(batch_size)]
        else:
            mask_values = [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)]
        self_distillation_mask = torch.tensor(mask_values, dtype=torch.float32, device=device)

        uids = set(batch.non_tensor_batch["uid"])
        metrics = {
            "self_distillation/success_group_fraction": len([uid for uid in uids if len(success_by_uid[uid]) > 0]) / len(uids),
            "self_distillation/success_sample_fraction": sum(s is not None for s in solution_strs) / batch_size,
            "self_distillation/feedback_available_fraction": sum(f is not None for f in feedback_list) / batch_size,
            "self_distillation/feedback_used_fraction": sum(feedback_used) / batch_size,
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
                }
            ),
            metrics,
        )
