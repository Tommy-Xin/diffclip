#!/usr/bin/bash

set -x

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

run_name=diff_clip_8gpus_seed99999_diffusionclip2
num_steps=4600
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    DIFF_CLIPFATR.py \
    --clip_image_size 224 \
    --visual_pattern None \
    --train_steps 2 \
    --image_size 512 \
    --fixed_image_size False \
    --dataset_path datasets/cc3m/ \
    --output_dir ./outputs/100%/$run_name \
    --overwrite_output_dir False \
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
    --seed 9999\
    --accelerator_config accelerator.json > ./logs/100%/debug_$run_name.log 