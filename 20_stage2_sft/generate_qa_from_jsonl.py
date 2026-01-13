import mlx.core as mx
from mlx_lm import load, generate
import json
import os
import argparse
from tqdm import tqdm

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Generate SFT Q&A dataset from text chunks")
    parser.add_argument("--input-file", type=str, required=True, help="Input JSONL file with 'text' field")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file for SFT")
    parser.add_argument("--model-path", type=str, default="models/HyperCLOVAX-SEED-Omni-8B-Text-4bit", help="Path to base model")
    parser.add_argument("--adapter-path", type=str, default="adapters_omni_8b_paper_cpt", help="Path to adapter (optional)")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    return parser

def format_prompt(text):
    # Prompt engineering for Q&A generation
    return f"""Based on the following text, generate a single relevant Question and a detailed Answer.
    
Text:
{text}

Format your response exactly as:
Question: [Your Question]
Answer: [Your Answer]

Response:
"""

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

    print(f"Loading model from {args.model_path} with adapter {args.adapter_path}...")
    # Fix regex issue implicitly
    model, tokenizer = load(args.model_path, adapter_path=args.adapter_path, tokenizer_config={"fix_mistral_regex": True})

    results = []
    
    with open(args.input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if args.num_samples > 0:
        lines = lines[:args.num_samples]

    print(f"Processing {len(lines)} contexts...")

    for line in tqdm(lines):
        try:
            data = json.loads(line)
            text = data.get("text", "").strip()
            if len(text) < 100: # Skip too short contexts
                continue
                
            prompt = format_prompt(text)
            
            # Generate
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=512, 
                verbose=False
            )
            
            # Simple parsing
            q_marker = "Question:"
            a_marker = "Answer:"
            
            if q_marker in response and a_marker in response:
                q_part = response.split(q_marker)[1].split(a_marker)[0].strip()
                a_part = response.split(a_marker)[1].strip()
                
                qa_entry = {
                    "text": f"User: {q_part}\nAssistant: {a_part}"  # SFT format usually needs prompt/completion or messages
                    # For MLX SFT example, it usually expects 'text' field with full conversation
                    # or 'messages' field. We will save as 'messages' format for flexibility?
                    # MLX lora example usually takes 'text'. Let's stick to 'text' complying with chat template.
                }
                
                # Let's save both raw Q/A and formatted text
                qa_entry_structured = {
                    "question": q_part,
                    "answer": a_part,
                    "local_source": text[:50] + "..."
                }
                
                # Write immediately to line
                with open(args.output_file, 'a', encoding='utf-8') as out_f:
                    json.dump(qa_entry_structured, out_f, ensure_ascii=False)
                    out_f.write('\n')
                    
        except Exception as e:
            print(f"Error processing line: {e}")

    print(f"Done. Saved to {args.output_file}")

if __name__ == "__main__":
    main()
