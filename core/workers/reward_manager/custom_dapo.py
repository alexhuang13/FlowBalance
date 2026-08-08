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

from collections import defaultdict

import torch
import inspect

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.utils.reward_score import default_compute_score

from verl.experimental.reward_loop.reward_manager import register as rm_base_register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from core.workers.reward_manager.api_calls import call_api


def compute_overlong_reward(valid_response_length, max_resp_len, overlong_buffer_cfg):
    """Return the DAPO linear overlong-buffer penalty as a Python float."""
    if overlong_buffer_cfg is None or not overlong_buffer_cfg.enable:
        return 0.0
    buffer_len = int(overlong_buffer_cfg.len)
    if buffer_len <= 0:
        raise ValueError(f"overlong_buffer_cfg.len must be > 0, got {buffer_len}")
    if max_resp_len is None or max_resp_len < buffer_len:
        raise ValueError(
            f"max_resp_len must be provided and >= overlong_buffer_cfg.len, got {max_resp_len=}, {buffer_len=}"
        )
    response_len = float(valid_response_length)
    expected_len = max_resp_len - buffer_len
    exceed_fraction = (response_len - expected_len) / buffer_len
    return min(-exceed_fraction * float(overlong_buffer_cfg.penalty_factor), 0.0)

def custom_default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    prompt_str=None,
    api_endpoints=None,
    **kwargs,
):
    try:
        res = default_compute_score(data_source=data_source,
                                    solution_str=solution_str,
                                    ground_truth=ground_truth,
                                    extra_info=extra_info,
                                    sandbox_fusion_url=sandbox_fusion_url,
                                    concurrent_semaphore=concurrent_semaphore,
                                    memory_limit_mb=memory_limit_mb,
                                    **kwargs)
    except Exception as e:

        if data_source in ["acereason_math", "deepscaler", "numinamath_1.5", "openmathreasoning"]:
            from verl.utils.reward_score import math_dapo
            res = math_dapo.compute_score(solution_str, ground_truth)
        elif api_endpoints is not None and data_source in api_endpoints:
            res = call_api(api_endpoint=api_endpoints[data_source],
                           domain=data_source, prompt=prompt_str, response=solution_str, label=ground_truth, **kwargs)
        else:
            raise NotImplementedError(f"Reward function is not implemented for {data_source=}")
    return res

@register("custom_dapo")
class CustomDAPORewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        api_endpoints=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or custom_default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.api_endpoints = api_endpoints

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})

            extra_info["rollout_reward_scores"] = rollout_reward_scores

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                prompt_str=prompt_str,
                extra_info=extra_info,
                api_endpoints=self.api_endpoints,
            )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                # NOTE(haitaomi): there are some mismatches in size when we use multiple data sources.
                # verl/protocol.py", line 486, in check_consistency
                #    assert val.shape[0] == batch_size, (
                # AssertionError: key acc length 672 is not equal to batch size 6144
                # Keep benchmark accuracy distinct from shaped reward/continuous score.
                reward_extra_info["score"].append(score)
                reward_extra_info["acc"].append(float(result.get("acc", score)))

                # skip the following keys
                # for key, value in result.items():
                #    reward_extra_info[key].append(value)
            else:
                score = result
                reward_extra_info["score"].append(score)
                reward_extra_info["acc"].append(score)

            reward = score

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_reward = compute_overlong_reward(
                    valid_response_length, self.max_resp_len, self.overlong_buffer_cfg
                )
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


@rm_base_register("custom_dapo")
class CustomDAPORewardManagerBase(RewardManagerBase):
    """DAPO Reward Manager."""

    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        overlong_buffer_cfg = config.reward_model.get("reward_kwargs", {}).get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = config.reward_model.get("reward_kwargs", {}).get("max_resp_len", None)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_reward = compute_overlong_reward(
                valid_response_length, self.max_resp_len, self.overlong_buffer_cfg
            )
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}
