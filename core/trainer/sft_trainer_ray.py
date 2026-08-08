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


from verl.utils.device import auto_set_device
from core.utils.dataset.sft_dataset import CustomSFTDataset

import ray
import hydra
import logging
import os

from verl.trainer.sft_trainer_ray import SFTTrainer

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class CustomSFTTrainer(SFTTrainer):
    def __init__(
        self,
        config,
    ):
        super().__init__(config=config)

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor=processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor=processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset


def run_sft(config):
    ray.init()
    trainer = CustomSFTTrainer(config=config)
    trainer.fit()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_object

        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to CustomSFTDataset
        # prompt: [{"role": "system", "content": "xxx"}, {"role": "user", "content": "xxx"}]
        # response: yyy
        dataset_cls = CustomSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config
    )
    return dataset


if __name__ == "__main__":
    main()
