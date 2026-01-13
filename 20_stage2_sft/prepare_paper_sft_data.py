import argparse
import random
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Input QA JSONL file")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split-ratio", type=float, default=0.9, help="Train split ratio")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    random.shuffle(lines)
    
    split_idx = int(len(lines) * args.split_ratio)
    train_lines = lines[:split_idx]
    valid_lines = lines[split_idx:]
    
    # Ensure at least one validation sample
    if not valid_lines and train_lines:
        valid_lines = [train_lines.pop()]

    with open(os.path.join(args.output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for line in train_lines:
            data = json.loads(line)
            # Transform to text format (bypass chat template requirement)
            text = f"User: {data['question']}\n\nAssistant: {data['answer']}"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            
    with open(os.path.join(args.output_dir, "valid.jsonl"), 'w', encoding='utf-8') as f:
        for line in valid_lines:
            data = json.loads(line)
            text = f"User: {data['question']}\n\nAssistant: {data['answer']}"
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Split {len(lines)} samples into {len(train_lines)} train and {len(valid_lines)} valid.")

if __name__ == "__main__":
    main()
