
# Custom Data Set


from typing import List

import datasets
import numpy as np
import os

from verl.utils.dataset.rl_dataset import RLHFDataset
from transformers import PreTrainedTokenizer
from omegaconf import DictConfig


class JsonlRLHFDataset(RLHFDataset):
    """
    Custom RLHF dataset that loads JSONL files instead of Parquet/JSON.
    Each line in the file must be a valid JSON object.

    Inherits from verl.utils.dataset.rl_dataset.RLHFDataset and
    overrides _read_files_and_tokenize to support JSONL.
    """

    def __init__(
        self,
        data_files: str | List[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        **kwargs,
    ):
        super().__init__(data_files, tokenizer, config, **kwargs)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            if not os.path.isfile(parquet_file):
                raise FileNotFoundError(f"JSONL file not found: {parquet_file}")
            # read files and cache
            dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
