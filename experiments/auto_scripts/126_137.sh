#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 126 --batch_size 512 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 127  --batch_size 256 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 128 --batch_size 128 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 129 --batch_size 64 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 130 --batch_size 32 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 131 --batch_size 8 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
echo "First Model Set Completed..."

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 132 --batch_size 512 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 133  --batch_size 256 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 134 --batch_size 128 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 135 --batch_size 64 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 136 --batch_size 32 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 137 --batch_size 8 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.0
echo "First Model Set Completed..."
