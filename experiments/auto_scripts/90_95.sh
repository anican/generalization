#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 90 --batch_size 512 --conv_width 128 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 91 --batch_size 256 --conv_width 128 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 92 --batch_size 128 --conv_width 128 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 93 --batch_size 64 --conv_width 128 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 94 --batch_size 32 --conv_width 128 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 95 --batch_size 8 --conv_width 128 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
echo "set completed..."
