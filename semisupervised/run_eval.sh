#!/bin/sh
CUDA_VISIBLE_DEVICES=$1 python eval.py --method $2 --dataset multi_all --source real --target sketch --net $3 --step $4
