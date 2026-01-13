import mlx.core as mx
from mlx_lm import load, generate

MODEL_PATH = "models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"

TARGET_TERMS = [
    "CLaRa", 
    "SCP (Salient Compressor Pretraining)",
    "Joint Training in CLaRa"
]

def probe_baseline():
    print(f"Loading Base Model: {MODEL_PATH}")
    model, tokenizer = load(MODEL_PATH)
    
    print("\n" + "="*50)
    print(" [ Baseline Probing Result ]")
    print("="*50)
    
    for term in TARGET_TERMS:
        prompt = f"User: What is {term}?\n\nAssistant:"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100, verbose=False)
        print(f"\n[Query]: {term}")
        print(f"[Response]: {response.strip().replace(chr(10), ' ')}")

if __name__ == "__main__":
    probe_baseline()
