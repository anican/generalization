#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 54 --batch_size 512 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 55 --batch_size 256 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 56 --batch_size 128 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 57 --batch_size 64 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 58 --batch_size 32 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 59 --batch_size 8 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
echo "set completed..."
