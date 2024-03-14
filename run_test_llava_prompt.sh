#!/bin/sh
CUDA_VISIBLE_DEVICES=2 python /CODEDIR/test_llava_prompt.py --num_workers 8 --batch_size 16 --use_opt False --use_retrieval True --data_type goodnews

CUDA_VISIBLE_DEVICES=2 python /CODEDIR/test_llava_prompt.py --num_workers 8 --batch_size 16 --use_opt False --use_retrieval True --data_type nytimes