#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 96 --batch_size 512 --conv_width 256 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 97 --batch_size 256 --conv_width 256 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 98 --batch_size 128 --conv_width 256 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 99 --batch_size 64 --conv_width 256 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 100 --batch_size 32 --conv_width 256 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 101 --batch_size 8 --conv_width 256 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
echo "set completed..."
