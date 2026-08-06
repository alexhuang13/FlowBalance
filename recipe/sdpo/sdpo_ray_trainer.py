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
"""SDPO Ray PPO trainer (overlay).

Subclasses FlowSD's ``CustomRayPPOTrainer`` and injects the self-distillation
("reprompt") batch construction *without* copying the ~380-line ``fit()`` body.

The hook is placed in :meth:`_update_actor`: at that point the batch already
contains everything the source injects in ``fit()`` right after
``batch.batch["token_level_scores"] = reward_tensor`` -- i.e. the per-token
reward (``token_level_scores``), the rollout ``responses`` / ``response_mask``,
the ``uid`` grouping and the original ``raw_prompt`` messages. The teacher
tensors only feed the actor's SDPO loss, so building them just before
``super()._update_actor`` is functionally equivalent to the upstream placement
(advantage computation never reads them).

The reprompt / teacher-batch methods (``_remove_thinking_trace``,
``_get_solution``, ``_collect_solutions_by_uid``, ``_collect_feedback`` and
``_maybe_build_self_distillation_batch``) are ported verbatim from the
self-distillation-analysis fork's ``verl/trainer/ppo/ray_trainer.py``.
"""

import re
from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask

from core.trainer.ppo.ray_trainer import CustomRayPPOTrainer

__all__ = ["SDPORayPPOTrainer"]


class SDPORayPPOTrainer(CustomRayPPOTrainer):
    """GRPO-compatible trainer plus SDPO reprompt/teacher-batch construction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The reprompt tokenization truncates from this side (left/right/error).
        sd_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        if sd_cfg is not None:
            self.tokenizer.truncation_side = sd_cfg.get("reprompt_truncation", "error")

    # --------------------------------------------------------------- main hook

    def _update_actor(self, batch: DataProto) -> DataProto:
        """Build & union the self-distillation batch, then run the actor update.

        For non-sdpo ``loss_mode`` (e.g. GRPO baseline) this is a no-op wrapper
        around the parent implementation.
        """
        sd_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        sd_metrics: dict[str, float] = {}

        if sd_cfg is not None and loss_mode in {"sdpo", "opsd"}:
            # `token_level_scores` is the per-token reward tensor set in fit()
            # right before advantage computation; its row-sum is the seq score.
            reward_tensor = batch.batch["token_level_scores"]
            reward_extra_infos_dict = self._reconstruct_reward_extra_infos(batch, sd_cfg)
            sd_data = self._maybe_build_self_distillation_batch(
                batch, reward_tensor, reward_extra_infos_dict
            )
            if sd_data is not None:
                sd_batch, sd_metrics = sd_data
                batch = batch.union(sd_batch)

        actor_output = super()._update_actor(batch)

        # Surface the trainer-side diagnostic metrics through the actor output's
        # reduce_metrics path (single-element lists -> mean == value).
        if sd_metrics:
            metrics_dict = actor_output.meta_info.setdefault("metrics", {})
            for key, value in sd_metrics.items():
                metrics_dict[key] = [value]

        return actor_output

    def _reconstruct_reward_extra_infos(
        self, batch: DataProto, sd_cfg: Any
    ) -> Optional[dict[str, list]]:
        """Rebuild the slice of ``reward_extra_infos_dict`` that the reprompt
        path needs. Only ``feedback`` is consumed, and only when environment
        feedback is enabled -- otherwise return ``None``."""
        if not sd_cfg.get("include_environment_feedback", False):
            return None
        if "feedback" not in batch.non_tensor_batch:
            return None
        return {"feedback": list(batch.non_tensor_batch["feedback"])}

    # ----------------------------------------------- ported reprompt utilities

    @staticmethod
    def _collect_feedback(
        include_environment_feedback: bool,
        reward_extra_infos_dict: Optional[dict[str, Any]],
        batch_size: int,
    ) -> list[Any]:
        """Collect environment feedback from ``reward_extra_infos_dict``."""
        feedback_list: list[Any] = [None] * batch_size
        if include_environment_feedback and reward_extra_infos_dict is not None:
            raw_feedback = reward_extra_infos_dict.get("feedback", [])
            for i in range(min(len(raw_feedback), batch_size)):
                if raw_feedback[i] and isinstance(raw_feedback[i], str) and raw_feedback[i].strip():
                    feedback_list[i] = raw_feedback[i]
        return feedback_list

    def _collect_solutions_by_uid(
        self, batch: DataProto, reward_tensor: torch.Tensor, success_reward_threshold: float
    ) -> dict[Any, list[int]]:
        seq_scores = reward_tensor.sum(dim=-1).detach().cpu().numpy()
        uids = batch.non_tensor_batch["uid"]
        success_by_uid: dict[Any, list[int]] = defaultdict(list)
        for idx, uid in enumerate(uids):
            if seq_scores[idx] >= success_reward_threshold:
                success_by_uid[uid].append(idx)
        return success_by_uid

    @staticmethod
    def _remove_thinking_trace(text: str) -> str:
        """Remove ``<think>...</think>`` content (handles a missing opening tag,
        e.g. DeepSeek-R1-Distill-Qwen)."""
        # 1) Normal case: remove matched <think>...</think> pairs.
        result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        # 2) Edge case: no opening <think> but a stray </think> exists.
        result = re.sub(r"^.*?</think>\s*", "", result, flags=re.DOTALL)
        return result

    def _get_solution(
        self,
        idx: int,
        success_by_uid: dict[Any, list[int]],
        uids: list[Any],
        response_texts: list[str],
        dont_reprompt_on_self_success: bool = False,
        remove_thinking_from_demonstration: bool = False,
    ) -> Optional[str]:
        uid = uids[idx]
        solution_idxs = success_by_uid[uid]
        if dont_reprompt_on_self_success:
            solution_idxs = [j for j in solution_idxs if j != idx]
        if len(solution_idxs) == 0:
            return None
        # Taking the first successful demonstration effectively selects a random one.
        solution_idx = solution_idxs[0]
        solution_str = response_texts[solution_idx]
        if remove_thinking_from_demonstration:
            solution_str = self._remove_thinking_trace(solution_str)
        return solution_str

    def _maybe_build_self_distillation_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: Optional[dict[str, list]] = None,
    ) -> Optional[tuple[DataProto, dict[str, float]]]:
        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if self_distillation_cfg is None or loss_mode not in {"sdpo", "rlsd", "opsd"}:
            return None

        device = batch.batch["input_ids"].device
        response_mask = batch.batch["response_mask"]
        responses = batch.batch["responses"]
        response_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in responses]
        prompt_texts = [msgs[-1]["content"] for msgs in batch.non_tensor_batch["raw_prompt"]]
        batch_size = batch.batch.batch_size[0]

        # Extract feedback if available and include_environment_feedback is enabled.
        feedback_list = self._collect_feedback(
            include_environment_feedback=self_distillation_cfg.include_environment_feedback,
            reward_extra_infos_dict=reward_extra_infos_dict,
            batch_size=batch_size,
        )

        success_by_uid = self._collect_solutions_by_uid(
            batch, reward_tensor, success_reward_threshold=self_distillation_cfg.success_reward_threshold
        )
        def _group_solutions() -> list[Optional[str]]:
            return [
                self._get_solution(
                    i,
                    success_by_uid,
                    batch.non_tensor_batch["uid"],
                    response_texts,
                    self_distillation_cfg.dont_reprompt_on_self_success,
                    self_distillation_cfg.get("remove_thinking_from_demonstration", False),
                )
                for i in range(batch_size)
            ]

        def _external_solution(i: int) -> Optional[str]:
            # OPSD datasets normally expose a full worked solution.  The common
            # verl math schema may instead expose only reward_model.ground_truth;
            # accept both so the recipe can share the exact FlowSD/RLSD dataset.
            extra_infos = batch.non_tensor_batch.get("extra_info", None)
            if extra_infos is not None and i < len(extra_infos):
                info = extra_infos[i]
                if isinstance(info, dict):
                    for key in ("solution", "reference_solution", "ground_truth_solution"):
                        value = info.get(key)
                        if value is not None and str(value).strip():
                            return str(value)
            reward_models = batch.non_tensor_batch.get("reward_model", None)
            if reward_models is not None and i < len(reward_models):
                reward_model = reward_models[i]
                if isinstance(reward_model, dict):
                    for key in ("solution", "reference_solution"):
                        value = reward_model.get(key)
                        if value is not None and str(value).strip():
                            return str(value)
            return None

        solution_source = self_distillation_cfg.get("solution_source", "group_only")
        grouped = _group_solutions() if solution_source != "external_only" else [None] * batch_size
        external = (
            [_external_solution(i) for i in range(batch_size)]
            if solution_source in {"external_first", "external_only", "group_first"}
            else [None] * batch_size
        )
        if solution_source == "group_only":
            solution_strs = grouped
        elif solution_source == "group_first":
            solution_strs = [grouped[i] or external[i] for i in range(batch_size)]
        elif solution_source == "external_only":
            solution_strs = external
        elif solution_source == "external_first":
            solution_strs = [external[i] or grouped[i] for i in range(batch_size)]
        elif solution_source == "none":
            solution_strs = [None] * batch_size
        else:
            raise ValueError(f"Unsupported self_distillation.solution_source={solution_source!r}")

        def _build_teacher_message(i: int) -> list[dict]:
            system_messages = batch.non_tensor_batch["raw_prompt"][i][:-1]
            has_solution = solution_strs[i] is not None
            has_feedback = feedback_list[i] is not None
            feedback_only_without_solution = self_distillation_cfg.get(
                "environment_feedback_only_without_solution", False
            )

            # If feedback_only_without_solution is True, only use feedback when no solution exists.
            use_feedback = has_feedback and (not feedback_only_without_solution or not has_solution)

            solution_template = self_distillation_cfg.get(
                "solution_template", "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
            )
            feedback_template = self_distillation_cfg.get(
                "feedback_template",
                "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n",
            )
            reprompt_template = self_distillation_cfg.get(
                "reprompt_template", "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
            )

            solution_section = ""
            if has_solution:
                solution_section = solution_template.format(successful_previous_attempt=solution_strs[i])

            feedback_section = ""
            if use_feedback:
                feedback_section = feedback_template.format(feedback_raw=feedback_list[i])

            if use_feedback or has_solution:
                reprompt_text = reprompt_template.format(
                    prompt=prompt_texts[i],
                    solution=solution_section,
                    feedback=feedback_section,
                )
            else:
                reprompt_text = prompt_texts[i]

            return system_messages + [{"role": "user", "content": reprompt_text}]

        messages = [_build_teacher_message(i) for i in range(batch_size)]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", None)
        student_enable_thinking = (
            apply_chat_template_kwargs.get("enable_thinking", True) if apply_chat_template_kwargs else True
        )
        configured_teacher_thinking = self_distillation_cfg.get("teacher_enable_thinking", None)
        enable_thinking = (
            student_enable_thinking if configured_teacher_thinking is None else bool(configured_teacher_thinking)
        )
        teacher_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            continue_final_message=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            max_length=self_distillation_cfg.max_reprompt_len,
            padding=True,
            truncation=True,
        )
        teacher_input_ids = torch.cat([teacher_prompt["input_ids"].to(device), responses], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt["attention_mask"].to(device), response_mask], dim=1)
        teacher_position_ids = compute_position_id_with_mask(teacher_attention_mask)

        # Compute which samples actually use feedback.
        feedback_only_without_solution = self_distillation_cfg.get(
            "environment_feedback_only_without_solution", False
        )
        feedback_used = [
            feedback_list[i] is not None and (not feedback_only_without_solution or solution_strs[i] is None)
            for i in range(batch_size)
        ]

        # self_distillation_mask is True if the sample will get a reprompted message.
        self_distillation_mask = torch.tensor(
            [solution_strs[i] is not None or feedback_used[i] for i in range(batch_size)],
            dtype=torch.float32,
            device=device,
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
