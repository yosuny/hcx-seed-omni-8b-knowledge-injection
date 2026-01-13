import argparse
import time
import mlx.core as mx
from mlx_lm import load, generate

def benchmark(model_path, prompt="안녕하세요, 자기소개 좀 해주세요.", max_tokens=200):
    print(f"Benchmarking model: {model_path}")
    
    # Measure Load Time
    start_time = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start_time
    print(f"Load Time: {load_time:.4f} sec")
    
    # Measure Peak Memory (After Load)
    mx.metal.clear_cache()
    # Note: MLX memory reporting might need system check or mx.metal.get_active_memory()
    # Simple peak memory check using mlx 
    mem_load = mx.metal.get_active_memory() / 1024**3
    print(f"Memory (Load): {mem_load:.4f} GB")

    # Warmup
    print("Warming up...")
    generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)
    
    # Measure Inference
    print(f"Generating {max_tokens} tokens...")
    start_time = time.time()
    
    # Use generate wrapper to get text, but for speed measurement we can inspect internal if needed.
    # For now, end-to-end time is sufficient proxy.
    text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Approximate tokens generated (output length)
    # Re-tokenize output to count tokens
    output_tokens = len(tokenizer.encode(text))
    tps = output_tokens / total_time
    
    mem_peak = mx.metal.get_peak_memory() / 1024**3
    
    print(f"Total Time: {total_time:.4f} sec")
    print(f"Output Tokens: {output_tokens}")
    print(f"Speed: {tps:.2f} tokens/sec")
    print(f"Peak Memory (Session): {mem_peak:.4f} GB")
    
    return {
        "load_time": load_time,
        "tps": tps,
        "peak_memory": mem_peak
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MLX Model")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--prompt", type=str, default="안녕하세요, 자기소개 좀 해주세요.", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    benchmark(args.model, args.prompt, args.max_tokens)
