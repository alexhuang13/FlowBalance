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
"""Prepare the DeepSeek math RL data for the SDPO overlay.

DAPO-Math-17k (train) and AIME-2024 (eval) are already in verl's native parquet
schema (``prompt`` / ``data_source`` / ``reward_model`` / ``extra_info`` / ...), so
this script only:

  1. downloads the two parquet files from HuggingFace (honours ``HF_ENDPOINT`` so a
     mirror such as ``https://hf-mirror.com`` can be used), and
  2. rewrites the train prompts into the ``\\boxed{}`` answer format expected by the
     DeepSeek-R1-Distill model and the math_verify reward.

Output (under ``$DATA_ROOT/rl`` by default)::

    train.parquet         # DAPO-Math-17k, boxed-rewritten prompts
    aime-2024.parquet     # AIME-2024 eval set

All paths are environment-driven so the data root can later be switched to a shared
disk without code changes.

Usage::

    DATA_ROOT=/apdcephfs/share_300719894/user/audenhuang/data \
        python3 -m recipe.sdpo.data.prepare_math_data
"""

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

from recipe.sdpo.data.change_math_prompts import transform_prompt

TRAIN_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
TRAIN_FILENAME = "data/dapo-math-17k.parquet"
TEST_REPO = "BytedTsinghua-SIA/AIME-2024"
TEST_FILENAME = "data/aime-2024.parquet"


def _default_data_root() -> str:
    return os.environ.get(
        "DATA_ROOT", "/apdcephfs_gy4/share_303378103/user/audenhuang/data"
    )


def _download(repo_id: str, filename: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] already exists: {dest}")
        return
    print(f"[download] {repo_id}:{filename} (HF_ENDPOINT={os.environ.get('HF_ENDPOINT', 'default')})")
    cached = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    df = pd.read_parquet(cached)
    df.to_parquet(dest, index=False)
    print(f"[done] {dest} ({len(df)} rows, columns={df.columns.tolist()})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=_default_data_root())
    parser.add_argument(
        "--no_rewrite_prompts",
        action="store_true",
        help="Skip the \\boxed{} prompt rewrite on the train set.",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.data_root, "rl")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.parquet")
    test_path = os.path.join(out_dir, "aime-2024.parquet")

    _download(TRAIN_REPO, TRAIN_FILENAME, train_path)
    _download(TEST_REPO, TEST_FILENAME, test_path)

    if not args.no_rewrite_prompts:
        print(f"[rewrite] boxed-format prompts -> {train_path}")
        df = pd.read_parquet(train_path)
        df["prompt"] = df["prompt"].apply(transform_prompt)
        for i in range(min(2, len(df))):
            print(f"=== train row {i} ===")
            print(df["prompt"].iloc[i][0]["content"][:600])
        df.to_parquet(train_path, index=False)
        print("[done] prompt rewrite")

    print("\nData ready:")
    print(f"  TRAIN_FILE={train_path}")
    print(f"  TEST_FILE={test_path}")


if __name__ == "__main__":
    main()
