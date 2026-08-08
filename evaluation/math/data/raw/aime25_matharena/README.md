---
dataset_info:
  features:
  - name: problem_idx
    dtype: int64
  - name: problem
    dtype: string
  - name: answer
    dtype: int64
  - name: problem_type
    sequence: string
  splits:
  - name: train
    num_bytes: 16990
    num_examples: 30
  download_size: 14313
  dataset_size: 16990
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
license: cc-by-nc-sa-4.0
language:
- en
pretty_name: AIME 2025
size_categories:
- n<1K
---

### Homepage and repository

- **Homepage:** [https://matharena.ai/](https://matharena.ai/)
- **Repository:** [https://github.com/eth-sri/matharena](https://github.com/eth-sri/matharena)

### Dataset Summary

This dataset contains the questions from AIME 2025 used for the MathArena Leaderboard

### Data Fields


The dataset contains the following fields:

- `problem_idx` (`int64`): Problem index within the corresponding MathArena benchmark.
- `problem` (`string`): Problem statement, usually stored as LaTeX source.
- `answer` (`int64`): Gold final answer.
- `problem_type` (`list[string]`): Problem type/category labels.

### Source Data

The original questions were sourced from the AIME 2025 competition. Questions were extracted, converted to LaTeX and verified.

### Licensing Information

This dataset is licensed under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). Please abide by the license when using the provided data.

### Citation Information

```
@article{dekoninck2026matharena,
      title={Beyond Benchmarks: MathArena as an Evaluation Platform for Mathematics with LLMs}, 
      author={Jasper Dekoninck and Nikola Jovanović and Tim Gehrunger and Kári Rögnvaldsson and Ivo Petrov and Chenhao Sun and Martin Vechev},
      year={2026},
      eprint={2605.00674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.00674}, 
}
```
