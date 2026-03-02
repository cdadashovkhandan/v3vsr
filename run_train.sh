#!/bin/bash

# V3 training bash script
# Make sure to download the raft large checkpoint from
# https://github.com/alebeck/jax-raft/releases/download/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.msgpack
# and place it into ./checkpoints

# Pre-training phase
python -u run_train.py \
    --data-dir ./data/ \
    --train-set Adobe240/train \
    --val-set Adobe240/valid \
    --val-every 5_000 \
    --accu-steps 1 \
    --pretrained-raft ./checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.msgpack \
    --num-blocks 2 4 2 \
    --embed-dims 90 90 90 \
    --thera-dim 512 \
    --attention-heads 12 \
    --deformable-groups 12 \
    --freeze-flow-first 2_200_000 \
    --flow-grad-multiplier 0.1 \
    --max-grad-norm 1.0 \
    --n-iter 2_500_000 \
    --num-workers 30 \
    --patch-size 80 \
    --seq-len 14 \
    --local-batch-size 1 \
    --tag v3-pre

# Post-training phase with lower temporal freq. initialization;
# omitted default arguments for brevity
python -u run_train.py \
    --data-dir ./data/ \
    --train-set Adobe240/train \
    --val-set Adobe240/valid \
    --pretrained-raft ./checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.msgpack \
    --pretrained-path ./logs/params_latest-v3-pre.pkl \
    --freeze-flow-first 500_000 \
    --flow-grad-multiplier 1.0 \
    --freeze-encoder-first 100_000 \
    --encoder-grad-multiplier 0.1 \
    --max-grad-norm 10.0 \
    --t-init-scale 0.33 \
    --lr 1e-3 \
    --n-iter 500_000 \
    --tag v3
