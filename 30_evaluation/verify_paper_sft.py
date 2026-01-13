import mlx.core as mx
from mlx_lm import load, generate

model_path = "models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"
adapter_path = "adapters_omni_8b_paper_sft"

print(f"Loading model from {model_path} with adapter {adapter_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path, tokenizer_config={"fix_mistral_regex": True})

prompts = [
    "What is CLaRa in the context of retrieval-generation?",
    "Explain SCP and its purpose.",
    "Describe the joint training process in CLaRa."
]

print("\n--- Starting SFT Inference ---\n")

for i, text in enumerate(prompts):
    print(f"Q{i+1}: {text}")
    
    # Use the format trained in SFT
    prompt = f"User: {text}\n\nAssistant:"
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=200, 
        verbose=True
    )
    print("\n----------------\n")
