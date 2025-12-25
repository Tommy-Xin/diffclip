#!/bin/bash
# Example script showing how to train with a subset of CC3M dataset using take() method

# Method 1: Use max_train_samples to read a fixed number of samples
python DIFF_CLIPFATR.py \
    --dataset_path datasets/cc3m/ \
    --max_train_samples 100000 \
    --seed 3407 \
    --output_dir outputs/subset_100k \
    # ... other training arguments

# Method 2: Use sample_ratio to read a percentage of the dataset
# Note: This uses an estimated total of 3M samples for CC3M
python DIFF_CLIPFATR.py \
    --dataset_path datasets/cc3m/ \
    --sample_ratio 0.25 \
    --seed 3407 \
    --output_dir outputs/subset_25pct \
    # ... other training arguments

# Method 3: Combine with other training arguments
python DIFF_CLIPFATR.py \
    --dataset_path datasets/cc3m/ \
    --max_train_samples 50000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --learning_rate 1e-4 \
    --seed 3407 \
    --output_dir outputs/subset_50k_3epochs

