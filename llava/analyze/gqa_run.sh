#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_NAME="Video-LLaVA-7B"
# CKPT="checkpoints/${CKPT_NAME}"
CKPT="LanguageBind/${CKPT_NAME}"
SPLIT="llava_gqa_testdev_balanced"
EVAL="/home/akane38/Video-LLaVA/llava/eval"
ANALYZE="/home/akane38/Video-LLaVA/llava/analyze"
GQADIR="${EVAL}/gqa/data"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 -m llava.analyze.model_vqa_loader \
        --model-path ${CKPT} \
        --question-file ${EVAL}/gqa/$SPLIT.jsonl \
        --image-folder ${EVAL}/gqa/data/images \
        --answers-file ${ANALYZE}/gqa/answers/$SPLIT/${CKPT_NAME}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done