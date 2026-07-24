#!/bin/bash

# V3 training bash script
# Make sure to download the raft large checkpoint from
# https://github.com/alebeck/jax-raft/releases/download/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.msgpack
# and place it into ./checkpoints

# Pre-training phase
python -u run_train.py \
    --data-dir /home2/s3591077/scratch/datasets \
    --train-set scisr/train \
    --val-set scisr/test/GT \
    --val-every 2500 \
    --accu-steps 1 \
    --num-blocks 2 4 2 \
    --embed-dims 90 90 90 \
    --attention-heads 12 \
    --deformable-groups 12 \
    --freeze-flow-first 2_200_000 \
    --flow-grad-multiplier 0.1 \
    --max-grad-norm 1.0 \
    --n-iter 5000 \
    --num-workers 4 \
    --patch-size 50 \
    --seq-len 8 \
    --local-batch-size 1 \
    --tag v3-pre \
    --thera_dim 512

# # Post-training phase with lower temporal freq. initialization;
# # omitted default arguments for brevity
# python -u run_train.py \
#     --data-dir /home2/s3591077/scratch/datasets \
#     --train-set scisr99/train \
#     --val-set scisr99/test \
#     --freeze-flow-first 500_000 \
#     --flow-grad-multiplier 1.0 \
#     --freeze-encoder-first 100_000 \
#     --encoder-grad-multiplier 0.1 \
#     --max-grad-norm 10.0 \
#     --t-init-scale 0.33 \
#     --lr 1e-3 \
#     --n-iter 500_000 \
#     --tag v3
