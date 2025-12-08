#!/usr/bin/bash

set -x

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

run_name=DIVA_for_CLIP_init_config_openai
num_steps=4600
# 4 gpus
# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
#     --nproc_per_node=4 \
#     --nnodes=1 \
#     --node_rank=0 \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     run_DIVA_with_OpenAICLIP.py \
#     --clip_image_size 224 \
#     --visual_pattern None \
#     --train_steps 2 \
#     --image_size 512 \
#     --fixed_image_size False \
#     --dataset_path datasets/cc3m/ \
#     --output_dir ./outputs/$run_name \
#     --overwrite_output_dir True \
#     --remove_unused_columns False \
#     --do_train \
#     --ddp_find_unused_parameters True \
#     --dataloader_num_workers 8 \
#     --learning_rate 1e-4 \
#     --bf16 True \
#     --tf32 True \
#     --warmup_ratio 0.10 \
#     --weight_decay 0 \
#     --max_steps $num_steps \
#     --per_device_train_batch_size 32 \
#     --logging_strategy steps \
#     --logging_steps 50 \
#     --gradient_accumulation_steps 5 \
#     --save_strategy steps \
#     --save_steps $num_steps \
#     --save_total_limit 1 \
#     --ddp_backend nccl \
#     --report_to wandb \
#     --run_name $run_name \
#     --enable_flash True \
#     --lr_scheduler_type cosine \
#     --seed 42 \
#     --accelerator_config accelerator.json > ./logs/debug_$run_name.log 
torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    run_DIVA_with_OpenAICLIP.py \
    --clip_image_size 224 \
    --visual_pattern None \
    --train_steps 2 \
    --image_size 512 \
    --fixed_image_size False \
    --dataset_path datasets/cc3m/ \
    --output_dir ./outputs/$run_name \
    --overwrite_output_dir True \
    --remove_unused_columns False \
    --do_train \
    --ddp_find_unused_parameters True \
    --dataloader_num_workers 8 \
    --learning_rate 1e-4 \
    --bf16 True \
    --tf32 True \
    --warmup_ratio 0.005 \
    --weight_decay 0 \
    --max_steps $num_steps \
    --per_device_train_batch_size 16 \
    --logging_strategy steps \
    --logging_steps 50 \
    --gradient_accumulation_steps 5 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --ddp_backend nccl \
    --report_to wandb \
    --run_name $run_name \
    --enable_flash True \
    --lr_scheduler_type cosine \
    --seed 42 \
    --accelerator_config accelerator.json > ./logs/debug_$run_name.log 