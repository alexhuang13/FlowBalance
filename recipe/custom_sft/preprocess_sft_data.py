# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import argparse
import os

import datasets
from recipe.custom.preprocess_data import parse_kv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", help="The local path to the raw dataset.")
    parser.add_argument(
        "--local_save_dir",
        default="~/data/open_math_reasoning",
        help="The save directory for the preprocessed dataset.",
    )
    parser.add_argument("--split", type=str, default="cot", choices=["cot", "train"],
                        help="Split: cot, train")

    parser.add_argument("--task", type=str, default="openmath", choices=["openmath", "nemotron"],
                        help="Tasks: openmath: add messages and filter, nemotron: simply merge all into a single file")
    parser.add_argument(
        "--names",
        nargs='*',
        help="All sub-names in data"
    )


    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    if args.names is not None:
        dataframes = []
        for name in args.names:
            dataframes.append(datasets.load_dataset(local_dataset_path, name, split=args.split))
        dataset = datasets.concatenate_datasets(dataframes)
    else:
        dataset = datasets.load_dataset(local_dataset_path, split=args.split)

    def make_map_fn():
        def process_fn(example, idx):
            if args.task == "openmath":
                question = example.pop("problem")
                solution = example.pop("generated_solution")

                extra_info = {}
                for key, value in example.items():
                    extra_info[key] = value
                example.clear()

                data = {
                    "prompt": question,
                    "response": solution,
                    "messages": [
                        {"role": "user", "content": question, "loss_mask": 0},
                        {"role": "assistant", "content": solution, "loss_mask": 1},
                    ],
                    "extra_info": extra_info,
                }
            elif args.task == "nemotron":
                # nothing to do for now.
                messages = example.pop("messages")
                response = messages[-1]["content"]
                response_role = messages[-1]["role"]
                assert response_role == "assistant", f"Wrong data format: {messages=}"

                # remove response
                messages = messages[:-1]
                assert len(messages) > 0, f"Empty messages found!"

                extra_info = {}
                for key, value in example.items():
                    extra_info[key] = value
                example.clear()

                data = {
                    "prompt": messages,
                    "response": response,
                    "extra_info": extra_info,
                }


            return data

        return process_fn
    if args.task == "openmath":
        # filter out data where the problem_type is not has_answer_extracted
        dataset = dataset.filter(lambda example: example["problem_type"] == "has_answer_extracted")

    dataset = dataset.map(function=make_map_fn(), with_indices=True)

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
