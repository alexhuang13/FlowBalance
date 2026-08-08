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
"""Rewrite DAPO-Math prompts into the ``\\boxed{}`` answer format.

Ported from ``self-distillation-analysis/experiments/math/change_math_prompts.py``.
The DAPO-Math-17k prompts ship with an ``Answer: $Answer`` instruction; DeepSeek-R1
distilled models (and the math_verify reward) expect a ``\\boxed{}`` final answer,
so we strip the original instruction wrapper and append the boxed instruction.

The prompt column is a list of chat messages: ``[{"role": "user", "content": ...}]``.
"""

import re

__all__ = ["transform_prompt", "BOXED_INSTRUCTION"]

BOXED_INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."

_LEADING_INSTRUCTION = re.compile(
    r"^Solve the following math problem step by step\.\s*"
    r"The last line of your response should be of the form Answer: \$Answer "
    r"\(without quotes\) where \$Answer is the answer to the problem\.\s*\n*"
)
_TRAILING_INSTRUCTION = re.compile(
    r'\s*\n*Remember to put your answer on its own line after "Answer:"\.?\s*$'
)
_BOXED_INSTRUCTION_RE = re.compile(
    r"\s*Please reason step by step, and put your final answer within \\boxed\{\}\.\s*"
)


def transform_prompt(prompt_array):
    """Rewrite a single ``prompt`` cell (list of chat messages) in-place style.

    Returns a new ``[{"role": "user", "content": ...}]`` list with the original
    ``Answer:`` instruction replaced by the boxed instruction.
    """
    content = prompt_array[0]["content"]
    content = _LEADING_INSTRUCTION.sub("", content)
    content = _TRAILING_INSTRUCTION.sub("", content)
    content = _BOXED_INSTRUCTION_RE.sub("\n", content)
    new_content = content.strip() + BOXED_INSTRUCTION
    return [{"content": new_content, "role": "user"}]
