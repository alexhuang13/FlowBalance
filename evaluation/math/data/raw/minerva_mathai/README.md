---
language:
- en
dataset_info:
  features:
  - name: question
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: test
    num_examples: 272
configs:
- config_name: default
  data_files:
  - split: test
    path: test*
---
