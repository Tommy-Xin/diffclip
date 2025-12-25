#!/usr/bin/env python3
"""
Script to read a subset of CC3M dataset from DiffusionCLIP/datasets/cc3m
Usage:
    python scripts/read_cc3m_subset.py --num_samples 1000
    python scripts/read_cc3m_subset.py --sample_ratio 0.1
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset


def read_cc3m_subset(dataset_path, num_samples=None, sample_ratio=None, seed=42):
    """
    Read a subset of CC3M dataset using take() method.
    
    Args:
        dataset_path: Path to the CC3M dataset directory
        num_samples: Number of samples to read (if specified, takes priority)
        sample_ratio: Ratio of samples to read (0.0 to 1.0)
        seed: Random seed for shuffling
    """
    # Load dataset in streaming mode
    print(f"Loading dataset from: {dataset_path}")
    data = load_dataset(
        "webdataset", 
        data_dir=dataset_path, 
        split="train", 
        streaming=True
    )
    
    # Shuffle the data
    print(f"Shuffling data with seed={seed}...")
    data = data.shuffle(buffer_size=2_000, seed=seed)
    
    # Determine how many samples to take
    if num_samples is not None:
        samples_to_take = num_samples
        print(f"Taking {samples_to_take} samples (fixed number)")
    elif sample_ratio is not None:
        # Estimate total dataset size (CC3M has ~3M samples)
        # Note: This is an approximation
        estimated_total = 3_000_000
        samples_to_take = int(estimated_total * sample_ratio)
        print(f"Taking {samples_to_take} samples ({sample_ratio*100:.1f}% of estimated {estimated_total:,} total)")
    else:
        print("No limit specified, reading all available samples")
        samples_to_take = None
    
    # Use take() to get subset
    if samples_to_take is not None:
        data = data.take(samples_to_take)
    
    # Iterate through the data
    print("\nReading samples...")
    count = 0
    for sample in data:
        count += 1
        
        # Print sample info every 100 samples
        if count % 100 == 0:
            print(f"Processed {count} samples...")
        
        # Example: print first sample details
        if count == 1:
            print(f"\nFirst sample keys: {list(sample.keys())}")
            if 'jpg' in sample:
                print(f"  Image shape/type: {type(sample['jpg'])}")
            if 'txt' in sample:
                print(f"  Text preview: {str(sample['txt'])[:100]}...")
        
        # You can add your processing logic here
        # For example, save images, process text, etc.
    
    print(f"\nTotal samples read: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Read a subset of CC3M dataset")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/cc3m",
        help="Path to CC3M dataset directory (default: datasets/cc3m)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to read (takes priority over sample_ratio)"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=None,
        help="Ratio of samples to read (0.0 to 1.0, e.g., 0.25 for 25%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Resolve dataset path
    script_dir = Path(__file__).parent.parent
    dataset_path = script_dir / args.dataset_path
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    # Validate arguments
    if args.num_samples is None and args.sample_ratio is None:
        print("Warning: No limit specified. Reading all available samples.")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    if args.sample_ratio is not None:
        if not (0.0 < args.sample_ratio <= 1.0):
            print(f"Error: sample_ratio must be between 0.0 and 1.0, got {args.sample_ratio}")
            sys.exit(1)
    
    # Read subset
    read_cc3m_subset(
        dataset_path=str(dataset_path),
        num_samples=args.num_samples,
        sample_ratio=args.sample_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

