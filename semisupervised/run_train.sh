#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python main.py --method $2 --dataset multi_all --source real --target sketch --net $3 --save_check
