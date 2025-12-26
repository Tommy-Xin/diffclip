#!/usr/bin/bash

set -x

export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1

MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

run_name=diff_clip_wtoken_reduction_2gpus_seed9999_diffusionclip2

# Base training steps for 100% dataset
base_steps=4600
# Dataset ratio (0.25 = 25%, 0.5 = 50%, 0.75 = 75%, 1.0 = 100%)
dataset_ratio=0.5

# Calculate actual training steps based on dataset ratio
# This ensures training time scales with dataset size
num_steps=$(python3 -c "import sys; print(int($base_steps * $dataset_ratio))")

echo "Base steps: $base_steps"
echo "Dataset ratio: $dataset_ratio"
echo "Calculated steps: $num_steps"

CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
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
    --dataset_ratio $dataset_ratio \
    --output_dir ./outputs/50%/$run_name \
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
    --gradient_accumulation_steps 20 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --ddp_backend nccl \
    --report_to wandb \
    --run_name $run_name \
    --enable_flash True \
    --lr_scheduler_type cosine \
    --seed 9999\
    --accelerator_config accelerator.json > ./logs/50%/debug_$run_name.log 