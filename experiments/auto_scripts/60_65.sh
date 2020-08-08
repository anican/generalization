#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 60 --batch_size 512 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 61 --batch_size 256 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 62 --batch_size 128 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 63 --batch_size 64 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 64 --batch_size 32 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 65 --batch_size 8 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 1 --weight_decay 0.0
echo "set completed..."
