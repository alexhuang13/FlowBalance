---
license: mit
task_categories:
- question-answering
language:
- en
size_categories:
- n<1K
configs:
  - config_name: AIME2025-I
    data_files:
      - split: test
        path: aime2025-I.jsonl
  - config_name: AIME2025-II
    data_files:
      - split: test
        path: aime2025-II.jsonl
---

# AIME 2025 Dataset
## Dataset Description
This dataset contains problems from the American Invitational Mathematics Examination (AIME) 2025-I & II.