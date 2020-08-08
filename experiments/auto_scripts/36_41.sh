#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 36 --batch_size 512 --conv_width 128 --dropout 0.0 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 37 --batch_size 256 --conv_width 128 --dropout 0.0 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 38 --batch_size 128 --conv_width 128 --dropout 0.0 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 39 --batch_size 64 --conv_width 128 --dropout 0.0 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 40 --batch_size 32 --conv_width 128 --dropout 0.0 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 41 --batch_size 8 --conv_width 128 --dropout 0.0 --num_block 4 --num_dense 1 --weight_decay 0.0
echo "set completed..."
