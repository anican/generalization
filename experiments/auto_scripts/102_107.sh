#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 102 --batch_size 512 --conv_width 512 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 103 --batch_size 256 --conv_width 512 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 104 --batch_size 128 --conv_width 512 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 105 --batch_size 64 --conv_width 512 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 106 --batch_size 32 --conv_width 512 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 107 --batch_size 8 --conv_width 512 --dropout 0.5 --num_block 2 --num_dense 2 --weight_decay 0.0
echo "set completed..."
