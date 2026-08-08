#!/bin/bash

BASE_MODEL="/data0/shared/Qwen3-4B"

# Evaluate the both-nonthink 4B model (student & teacher both non-thinking during training)
# at checkpoint-100 on AIME24, in non-thinking inference mode.
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4 \
    --no_thinking \
    --checkpoint_dir /data0/siyanz/opsd/qwen34b_gen1024_both_nonthink_fixteacher_temp11_forwardbeta0_clip1e-6/checkpoint-100
wait
