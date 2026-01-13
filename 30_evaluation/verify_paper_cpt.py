import mlx.core as mx
from mlx_lm import load, generate

model_path = "models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"
adapter_path = "adapters_omni_8b_paper_cpt"

print(f"Loading model from {model_path} with adapter {adapter_path}...")
model, tokenizer = load(model_path, adapter_path=adapter_path, tokenizer_config={"fix_mistral_regex": True})

prompts = [
    "What is CLaRa in the context of retrieval-generation?",
    "Explain SCP (Salient Compressor Pretraining).",
    "How does CLaRa perform end-to-end joint training?"
]

print("\n--- Starting Inference ---\n")

for i, prompt_text in enumerate(prompts):
    print(f"Q{i+1}: {prompt_text}")
    
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        # Simple fallback if no template
        prompt = f"User: {prompt_text}\nAssistant:"
        
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=200, 
        verbose=True
    )
    print("\n----------------\n")
