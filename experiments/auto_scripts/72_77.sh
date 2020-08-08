#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 72 --batch_size 512 --conv_width 128 --dropout 0.0 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 73 --batch_size 256 --conv_width 128 --dropout 0.0 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 74 --batch_size 128 --conv_width 128 --dropout 0.0 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 75 --batch_size 64 --conv_width 128 --dropout 0.0 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 76 --batch_size 32 --conv_width 128 --dropout 0.0 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 77 --batch_size 8 --conv_width 128 --dropout 0.0 --num_block 2 --num_dense 2 --weight_decay 0.0
echo "set completed..."
