import mlx.core as mx
from mlx_lm import load, generate
import os

# Model ID on Hugging Face
# 사용자가 정확한 모델 ID를 입력해야 할 수 있음. (예: naver-clova/HyperCLOVAX-SEED-Omni-8B)
# 만약 로컬에 다운로드 받았다면 로컬 경로를 입력해도 됩니다.
model_path = "models/HyperCLOVAX-SEED-Omni-8B-Text-8bit"

print(f"Loading model {model_path}...")

# Load model and tokenizer
# trust_remote_code=True는 일부 커스텀 모델 아키텍처에 필요할 수 있음
model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})

# Test Prompt
user_input = "안녕하세요, 자기소개 좀 해주세요."
messages = [{"role": "user", "content": user_input}]

# Apply Chat Template if available
if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    # tokenize=False로 하여 string prompt를 얻음
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
else:
    # Fallback template (ChatML style usually works for many modern models, or simple User/Assistant)
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

print(f"\n[Prompt]:\n{prompt}\n")
print(f"[Generation]:")

# Generate
# temp=0.0 (greedy) or higher for sampling
response = generate(
    model, 
    tokenizer, 
    prompt=prompt, 
    max_tokens=512, 
    verbose=True
)
