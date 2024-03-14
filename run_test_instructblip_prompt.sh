#!/bin/sh
CUDA_VISIBLE_DEVICES=0 python /CODEDIR/test_instructblip_prompt.py --num_workers 8 --batch_size 1 --use_opt False --use_retrieval True --data_type nytimes

CUDA_VISIBLE_DEVICES=0 python /CODEDIR/test_instructblip_prompt.py --num_workers 8 --batch_size 1 --use_opt False --use_retrieval True --data_type goodnews