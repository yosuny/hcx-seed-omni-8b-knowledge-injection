import random
import os
import argparse

def split_data(input_file, output_dir, split_ratio=0.9):
    print(f"Reading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    random.seed(42)
    random.shuffle(lines)
    
    split_idx = int(len(lines) * split_ratio)
    train_data = lines[:split_idx]
    valid_data = lines[split_idx:]
    
    # Ensure invalid/valid splits exist (handles edge case of very small files)
    if not valid_data and len(lines) > 1:
        valid_data = [lines[-1]]
        train_data = lines[:-1]
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")
    
    with open(train_path, "w", encoding='utf-8') as f:
        f.writelines(train_data)
        
    with open(valid_path, "w", encoding='utf-8') as f:
        f.writelines(valid_data)
        
    print(f"Data split complete.")
    print(f"Train: {len(train_data)} lines -> {train_path}")
    print(f"Valid: {len(valid_data)} lines -> {valid_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_dir", help="Directory to save train.jsonl and valid.jsonl")
    args = parser.parse_args()
    
    split_data(args.input_file, args.output_dir)
