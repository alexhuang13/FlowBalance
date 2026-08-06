#!/bin/bash

BASE_MODEL="/data0/shared/Qwen3-1.7B"
EXP_DIR="/data1/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"

# evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size 4
wait 

# after trained, evaluate the performance of the trained model. 
for step in 25 50 75 100; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size 4 \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step"
done
