# #!/usr/bin/bash

# set -x

# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
# export OMP_NUM_THREADS=4
# export NCCL_P2P_DISABLE=1
# MASTER_ADDR=$(hostname -I | awk '{print $1}')
# MASTER_PORT=12345

# run_name=DIVA_for_DFN
# num_steps=4600
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 \
# 	--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --use_env \
#     run_DIVA_with_DFN.py \
#     --clip_image_size 224 \
#     --visual_pattern None \
#     --train_steps 2 \
#     --image_size 512 \
#     --fixed_image_size False \
#     --dataset_path dataset/cc3m/\*.tar \
#     --output_dir ./outputs/$run_name \
#     --remove_unused_columns False \
#     --do_train \
#     --ddp_find_unused_parameters True \
#     --dataloader_num_workers 8 \
#     --learning_rate 1e-4 \
#     --bf16 True \
#     --tf32 True \
#     --warmup_ratio 0.005 \
#     --weight_decay 0 \
#     --max_steps $num_steps \
#     --per_device_train_batch_size 16 \
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
#     --lr_scheduler_type "cosine" \
#     --seed 42 \
#     --accelerator_config accelerator.json > ./logs/debug_$run_name.log
#!/usr/bin/bash
#bash DIVA_for_DFN.sh
set -x

# 环境变量配置
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=1

# 主节点信息
MASTER_ADDR=$(hostname -I | awk '{print $1}')
MASTER_PORT=12345

# 训练配置
run_name=DIVA_for_DFN
num_steps=4600

# torchrun 启动分布式训练
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    run_DIVA_with_DFN.py \
    --clip_image_size 224 \
    --visual_pattern None \
    --train_steps 2 \
    --image_size 512 \
    --fixed_image_size False \
    --dataset_path "datasets/cc3m/*.tar" \
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
    --per_device_train_batch_size 32 \
    --logging_strategy steps \
    --logging_steps 50 \
    --gradient_accumulation_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --ddp_backend nccl \
    --report_to wandb \
    --run_name $run_name \
    --enable_flash True \
    --lr_scheduler_type "cosine" \
    --seed 42 \
    > ./logs/debug_$run_name.log
