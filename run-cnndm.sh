#!/bin/bash

#SBATCH --job-name=lit-llama
#SBATCH --partition=RTX4090
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks-per-node=2

DATASET=cnn-dm-heading96
OUTPUT_DIR=out/lora/$DATASET/r16-lr6e-5-1

mkdir -p $OUTPUT_DIR

cp finetune/lora.py $OUTPUT_DIR/lora.py
cp run-cnndm.sh $OUTPUT_DIR/run.sh

NCCL_P2P_DISABLE=1 CUDA_LAUNCH_BLOCKING=1 python finetune/lora.py \
--out_dir $OUTPUT_DIR \
--data_dir data/$DATASET \
--resume_path /home/jskai/workspace/lit-llama/out/lora/cnn-dm-heading96/r16-lr6e-5-1/iter-281599-ckpt.pth \
| tee -a $OUTPUT_DIR/train.log