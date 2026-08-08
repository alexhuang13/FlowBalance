# Source state

整理日期：2026-08-03（Asia/Beijing）。

## 当前统一训练代码

- 来源目录：`../stable_rl/`
- Git HEAD：`171b77c9157f166ff0c71f9f3fdcf88b80de2327`
- 策略：复制当时工作树的实际文件状态，包括未提交的有效代码改动；排除 `.git`、缓存和运行产物。

## verl

- 来源目录：`../stable_rl/verl/`
- Git HEAD：`7522bef0eb5c5761500fa8652e7ed45936f5323d`
- 策略：将源码完整展开到本仓库，不保留嵌套 Git 仓库依赖。

## 独立 RLSD 快照

- 来源目录：`../RLSD/`
- Git HEAD：`aca313b978de4fb3a509ddf1831bf61fe597a2eb`
- 目标目录：`provenance/rlsd_standalone/`
- 用途：保留历史独立实现以便追溯。当前推荐实现是 `recipe/rlsd/`。

## Evaluation

- 数学评测来源：`../math_benchmark_eval/`
- 代码评测来源：`../code_benchmark_eval/`
- 包含：源码、脚本、manifest、benchmark 小型数据和 toolchain wrapper。
- 排除：日志、模型合并产物和历史结果目录。

## 未复制内容

模型、训练数据主目录、checkpoint、output、W&B、日志、缓存、`.git` 元数据和评测历史结果。这些不是代码仓库内容，并可能体积巨大或含环境专用信息。

## 独立 OPSD 快照

- 来源目录：`../OPSD/`
- Git HEAD：`7448751f307a9cdbcc1246dd1565a1a605b443df`
- 目标目录：`provenance/opsd_standalone/`
- 用途：保留原 TRL/Accelerate 实现。统一 Ray/FSDP/vLLM 实现在 `recipe/opsd/`。
