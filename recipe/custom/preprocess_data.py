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
"""Preprocess datasets to DAPO-compatible multiturn format."""

import argparse
import json
import os
import json
import uuid
import random

import datasets

MATH_DAPO_COMPLIANT_INSTRUCTION = (
    "\nPlease reason step by step.\n"
    "For the FINAL answer presentation format, ignore any earlier conflicting formatting instructions "
    "(e.g., requests to use \\boxed{...}).\n"
    'IMPORTANT: The last line of your response should be EXACTLY "Answer: <final-answer>" '
    "(without quotes) where <final-answer> is the answer to the problem. "
    "Use a single math expression for <final-answer> (fractions/roots OK). "
    "Do not include text after the Answer line."
)


def parse_kv(arg):
    if "=" not in arg:
        raise argparse.ArgumentTypeError("Expected key=value")
    return arg.split("=", 1)


def load_dataset_with_local_fallback(local_dataset_path, data_path):
    normalized_local_path = (local_dataset_path or "").strip()
    if normalized_local_path:
        expanded_path = os.path.expanduser(normalized_local_path)
        if os.path.isdir(expanded_path) and (
            os.path.exists(os.path.join(expanded_path, "dataset_dict.json"))
            or os.path.exists(os.path.join(expanded_path, "state.json"))
        ):
            return datasets.load_from_disk(expanded_path)

        if os.path.isfile(expanded_path) and expanded_path.endswith(".parquet"):
            return datasets.load_dataset("parquet", data_files=expanded_path)

        return datasets.load_dataset(normalized_local_path)

    return datasets.load_dataset(data_path)


def get_default_data_path(task):
    if task in {"polaris", "polaris_int", "polaris_extint"}:
        return "POLARIS-Project/Polaris-Dataset-53K"
    if task == "dolci_int":
        return "allenai/Dolci-Think-RL-7B"
    if task == "dolci_zero":
        return "allenai/Dolci-RL-Zero-Math-7B"
    return "BytedTsinghua-SIA/DAPO-Math-17k"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/retool_dapo", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dapo",
        choices=[
            "dapo",
            "olmo2dapo",
            "nemotron2dapo",
            "nemotronif",
            "nemotronswe",
            "polaris",
            "polaris_int",
            "polaris_extint",
            "dolci_int",
            "dolci_zero",
        ],
        help="Preprocessing task type.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dapo",
        choices=["dapo", "olmo2dapo", "nemotron2dapo", "nemotronif", "nemotronswe"],
        help="Tasks: dapo: only add extra_info, olmo2dapo: change olmo prompt from str to dict",
    )
    parser.add_argument(
        "--data_path", default=None, help="Override dataset path when --local_dataset_path is not provided."
    )
    parser.add_argument("--split", default="train", help="Dataset split to preprocess.")
    parser.add_argument(
        "--keys",
        nargs="*",
        choices=["prompt", "extra_info", "reward_model", "data_source", "ability"],
        default=["prompt", "extra_info", "reward_model", "data_source", "ability"],
        help="Keys to keep in the final data.",
    )
    parser.add_argument(
        "--map",
        type=parse_kv,
        nargs="+",
        help="Map pairs like target=source or target=source:list (pick first element).",
    )
    args = parser.parse_args()

    mapping = dict(args.map) if args.map else {}

    data_path = args.data_path or get_default_data_path(args.task)
    dataset = load_dataset_with_local_fallback(args.local_dataset_path, data_path)
    train_dataset = dataset[args.split] if isinstance(dataset, datasets.DatasetDict) else dataset

    tot_print = 0
    def make_map_fn():
        def process_fn(example, idx):
            del idx

            for target, source in mapping.items():
                source_key = source
                source_type = None
                if ":" in source:
                    source_key, source_type = source.split(":", 1)
                value = example.pop(source_key)
                if source_type == "list":
                    value = value[0]
                example[target] = value

            if args.task in {"polaris", "polaris_int", "polaris_extint"}:
                ground_truth = str(example.get("answer", "")).strip()
                problem = str(example.get("problem", "")).rstrip()
                example["data_source"] = "math_dapo"
                example["prompt"] = [{"role": "user", "content": f"{problem}\n\n{MATH_DAPO_COMPLIANT_INSTRUCTION}"}]
                example["ability"] = str(example.get("difficulty", "MATH"))
                example["reward_model"] = {"ground_truth": ground_truth}
            elif args.task in {"dolci_int", "dolci_zero"}:
                raw_ground_truth = example.get("ground_truth", "")
                if isinstance(raw_ground_truth, list):
                    ground_truth = str(raw_ground_truth[0]).strip() if raw_ground_truth else ""
                else:
                    ground_truth = str(raw_ground_truth).strip()

                prompt_text = str(example.get("prompt", example.get("problem", ""))).strip()
                if not prompt_text and isinstance(example.get("messages"), list):
                    for message in example["messages"]:
                        if message.get("role") == "user":
                            prompt_text = str(message.get("content", "")).strip()
                            break
                if prompt_text.startswith("user:"):
                    prompt_text = prompt_text[len("user:") :].lstrip()

                dataset_field = example.get("dataset", "math")
                if isinstance(dataset_field, list):
                    ability = str(dataset_field[0]) if dataset_field else "math"
                else:
                    ability = str(dataset_field)

                example["data_source"] = "math_dapo"
                example["prompt"] = [{"role": "user", "content": f"{prompt_text}\n\n{MATH_DAPO_COMPLIANT_INSTRUCTION}"}]
                example["ability"] = ability
                example["reward_model"] = {"ground_truth": ground_truth}
            elif args.task in {"olmo2dapo", "nemotron2dapo", "nemotronif", "nemotronswe"}:
                if args.task == "nemotronif":
                    example["data_source"] = "ifeval"  # create ifeval for nemotronif
                    instruction_id_list = example.pop("instruction_id_list")
                    kwargs = example.pop("kwargs")
                    org_ground_truth = [
                        {
                            "instruction_id": instruction_id_list,
                            "kwargs": json.dumps(kwargs, ensure_ascii=True),
                        }
                    ]  # NOTE: we has issues of kwargs `Couldn't cast array`, so we use json.dumps.
                elif args.task == "nemotronswe":
                    example["data_source"] = "swe_unidiff"
                    prompt = example.pop("prompt")
                    example["prompt"] = [{"content": prompt, "role": "user"}]
                    oracle_patches = example.pop("golden_patch")
                    org_ground_truth = json.dumps({"oracle_patches": [oracle_patches]}, ensure_ascii=True)

                else:
                    org_prompt = example.pop("prompt").lstrip("user: ")
                    example["prompt"] = [{"content": org_prompt, "role": "user"}]
                    if args.task == "olmo2dapo":
                        org_ground_truth = example["ground_truth"][0]  # Notes: only pick the first result
                    else:
                        org_ground_truth = example["answer"]
                example["reward_model"] = {"ground_truth": org_ground_truth}
                default_id = str(uuid.uuid4())
                if "extra_info" in example:
                    if "index" not in example["extra_info"]:
                        example["extra_info"]["index"] = example.get("index", default_id)
                else:
                    example["extra_info"] = {"index": example.get("custom_id", default_id)}

            else:
                ground_truth = example["reward_model"]["ground_truth"]

            global tot_print
            if tot_print < 50 and random.random() > 0.95:
                print(json.dumps(dict(example), ensure_ascii=True, indent=2))
                tot_print += 1

            extra_info = example.get("extra_info", {})
            if not isinstance(extra_info, dict):
                extra_info = {}
            extra_info["need_tools_kwargs"] = True
            extra_info["tools_kwargs"] = {
                "code_interpreter": {
                    "create_kwargs": {
                        "ground_truth": ground_truth,
                    }
                }
            }
            example["extra_info"] = extra_info

            for key in list(example.keys()):
                if key not in args.keys:
                    example.pop(key)

            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn(), with_indices=True)

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(save_dir, "train.parquet"))
