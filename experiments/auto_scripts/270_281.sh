#!/usr/bin/env bash

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 270 --batch_size 512 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 270"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 271 --batch_size 256 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 271"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 272 --batch_size 128 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 272"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 273 --batch_size 64 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 273"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 274 --batch_size 32 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 274"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 275 --batch_size 8 --conv_width 128 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 275"
echo "Model Set Completed..."

echo "Using GPU device CUDA:$1"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 276 --batch_size 512 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 276"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 277 --batch_size 256 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 277"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 278 --batch_size 128 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 278"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 279 --batch_size 64 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 279"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 280 --batch_size 32 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 280"
CUDA_VISIBLE_DEVICES=$1 python main.py --model_num 281 --batch_size 8 --conv_width 256 --dropout 0.5 --num_block 4 --num_dense 2 --weight_decay 0.001
echo "Finished Model 281"
echo "Models 276 -- 281 completed"
