import mlx.core as mx
from mlx_lm import load, generate
import time
import os

# Configuration
MODEL_PATH = "models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"
CPT_ADAPTER = "adapters_omni_8b_paper_cpt"
SFT_ADAPTER = "adapters_omni_8b_paper_sft"

TEST_QUERIES = [
    "What is CLaRa?",
    "What is SCP in CLaRa?",
    "Explain the joint training in CLaRa."
]

def measure_inference(model, tokenizer, prompt, name):
    print(f"--- {name} Inference ---")
    start_time = time.time()
    
    # Simple User/Assistant format for uniformity, though Base might prefer raw
    full_prompt = f"User: {prompt}\n\nAssistant:"
    
    response = generate(
        model, 
        tokenizer, 
        prompt=full_prompt, 
        max_tokens=100, 
        verbose=False
    )
    
    end_time = time.time()
    duration = end_time - start_time
    tokens = len(tokenizer.encode(response))
    tps = tokens / duration if duration > 0 else 0
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response.strip().replace(chr(10), ' ')}") # Flatten for log
    print(f"Stats: {duration:.2f}s, {tps:.2f} t/s")
    return {
        "model": name,
        "prompt": prompt,
        "response": response.strip(),
        "tps": tps,
        "duration": duration
    }

def main():
    results = []

    # 1. Base Model
    print("\n[1/3] Loading Base Model...")
    mx.clear_cache()
    try:
        model, tokenizer = load(MODEL_PATH, tokenizer_config={"fix_mistral_regex": True})
        for q in TEST_QUERIES:
            results.append(measure_inference(model, tokenizer, q, "Base Model"))
        del model
    except Exception as e:
        print(f"Base Model Failed: {e}")

    # 2. CPT Model
    print("\n[2/3] Loading CPT Model...")
    mx.clear_cache()
    try:
        model, tokenizer = load(MODEL_PATH, adapter_path=CPT_ADAPTER, tokenizer_config={"fix_mistral_regex": True})
        for q in TEST_QUERIES:
            results.append(measure_inference(model, tokenizer, q, "CPT Model"))
        del model
    except Exception as e:
        print(f"CPT Model Failed: {e}")

    # 3. SFT Model
    print("\n[3/3] Loading SFT Model...")
    mx.clear_cache()
    try:
        model, tokenizer = load(MODEL_PATH, adapter_path=SFT_ADAPTER, tokenizer_config={"fix_mistral_regex": True})
        for q in TEST_QUERIES:
            results.append(measure_inference(model, tokenizer, q, "SFT Model"))
        del model
    except Exception as e:
        print(f"SFT Model Failed: {e}")

    # Summary
    print("\n\n=== Benchmark Summary ===")
    print(f"{'Model':<15} | {'Query':<30} | {'Tokens/s':<10} | {'Response Start'}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<15} | {r['prompt'][:30]:<30} | {r['tps']:<10.2f} | {r['response'][:40]}...")

if __name__ == "__main__":
    main()
