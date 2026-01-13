import streamlit as st
import os
import json
import time
import csv
import re
import pandas as pd
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from datetime import datetime
from typing import List, Tuple, Optional

# --- Configuration & Constants ---
class Config:
    PAGE_TITLE = "HCX Omni 8B Knowledge Injection by AX Lab"
    PAGE_ICON = "🧠"
    MODELS_DIR = "models"
    LOG_FILE = "inference_logs.csv"
    DEFAULT_SFT_ADAPTER = "adapters_omni_8b_paper_sft"
    IGNORE_FILES = {".DS_Store"}
    EXCLUDED_MODELS = {"HyperCLOVAX-SEED-Omni-8B", "HyperCLOVAX-SEED-Omni-8B-Text"}

# --- Utility Functions ---

def clean_think_tags(text: str) -> str:
    """Removes <think>...</think> blocks and tokens from the text."""
    if not text:
        return ""
    # Remove standard <think> tags (include newlines)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove standalone tags/tokens
    for token in ["<think>", "</think>", "<|thinking|>", "<|/thinking|>"]:
        cleaned = cleaned.replace(token, "")
    return cleaned.strip()

# --- Model & Adapter Management ---

class ModelManager:
    @staticmethod
    def get_available_models() -> List[str]:
        """Scans MODELS_DIR for valid model directories."""
        if not os.path.exists(Config.MODELS_DIR):
            return []
        
        models = []
        try:
            for d in os.listdir(Config.MODELS_DIR):
                if d in Config.IGNORE_FILES or d in Config.EXCLUDED_MODELS:
                    continue
                
                model_path = os.path.join(Config.MODELS_DIR, d)
                if os.path.isdir(model_path):
                    # Verify it has a config.json just to be sure
                    if os.path.exists(os.path.join(model_path, "config.json")):
                        models.append(d)
        except Exception as e:
            st.error(f"Error scanning models: {e}")
            
        return sorted(models)

    @staticmethod
    def get_available_adapters() -> List[str]:
        """Scans current directory for adapter folders."""
        adapters = ["None"]
        try:
            for d in os.listdir("."):
                if d in Config.IGNORE_FILES:
                    continue
                if os.path.isdir(d) and d.startswith("adapters_"):
                    adapters.append(d)
        except Exception as e:
            st.error(f"Error scanning adapters: {e}")
            
        return sorted(adapters)

@st.cache_resource
def load_model_resource(model_name: str, adapter_name: str):
    """
    Loads and caches the model/tokenizer. 
    Wrapped in a standalone function for Streamlit caching.
    """
    model_path = os.path.join(Config.MODELS_DIR, model_name)
    adapter_path = adapter_name if adapter_name != "None" else None
    
    print(f"Loading model: {model_path} | Adapter: {adapter_path}")
    
    # Implicitly set fix_mistral_regex=True for HCX Omni compatibility
    model, tokenizer = load(
        model_path, 
        adapter_path=adapter_path, 
        tokenizer_config={"fix_mistral_regex": True}
    )
    return model, tokenizer

# --- Logging ---

class InferenceLogger:
    @staticmethod
    def log(model_name: str, adapter_name: str, prompt: str, response: str, 
            duration: float, tps: float, peak_mem: float):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = os.path.isfile(Config.LOG_FILE)
        
        try:
            with open(Config.LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "Model", "Adapter", "Prompt", "Response", "Time(s)", "Speed(TPS)", "Memory(GB)"])
                writer.writerow([timestamp, model_name, adapter_name, prompt, response, f"{duration:.2f}", f"{tps:.2f}", f"{peak_mem:.2f}"])
        except Exception as e:
            st.warning(f"Failed to write log: {e}")

    @staticmethod
    def show_history():
        if os.path.exists(Config.LOG_FILE):
            try:
                # on_bad_lines='skip' ensures robustness if column counts change
                df = pd.read_csv(Config.LOG_FILE, on_bad_lines='skip')
                st.write(f"Total Logs: {len(df)}")
                if st.checkbox("Show History"):
                    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
                    
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Logs CSV",
                        data=csv_data,
                        file_name="inference_logs.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.warning(f"Could not load log file (ParserError?): {e}")

# --- Main Application ---

def main():
    st.set_page_config(page_title=Config.PAGE_TITLE, page_icon=Config.PAGE_ICON, layout="wide")
    st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        
        # 1. Model Selection
        models = ModelManager.get_available_models()
        if not models:
            st.error(f"No models found in '{Config.MODELS_DIR}'!")
            st.stop()
        
        selected_index = 0
        for i, m in enumerate(models):
            if "4bit" in m:
                selected_index = i
                break
        
        selected_model = st.selectbox("Select Model", models, index=selected_index)

        # 2. Adapter Selection
        adapters = ModelManager.get_available_adapters()
        
        # Default logic: Select SFT adapter if present and no user choice yet (stateless, so we try always or index)
        # To strictly better UX, we just default the index.
        default_index = 0
        if Config.DEFAULT_SFT_ADAPTER in adapters:
            default_index = adapters.index(Config.DEFAULT_SFT_ADAPTER)
            
        selected_adapter = st.selectbox("Select Adapter", adapters, index=default_index)
        
        # 3. Parameters
        # Intuitive naming: "Enable Creativity" (Do Sample)
        enable_sampling = st.checkbox("Enable Creativity (Randomness)", value=True, help="Check to generic diverse responses. Uncheck for deterministic (fact-based) responses.")
        
        # Adjust slider behavior based on sampling
        if enable_sampling:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1, help="Higher values = more creative/random. Lower values = more focused.")
        else:
            # If disabled, force 0.0 and disable input
            temperature = st.slider("Temperature (Fixed)", 0.0, 2.0, 0.0, 0.1, disabled=True, help="Temperature is set to 0 in deterministic mode.")

        max_tokens = st.slider("Max Tokens", 64, 2048, 512, 64)

        
        # 4. Utilities
        if st.button("Reload Model (Clear Cache)"):
            st.cache_resource.clear()
            st.success("Cache cleared!")

        # 5. Conversation Settings
        st.markdown("---")
        st.markdown("### Conversation")
        enable_multiturn = st.checkbox("Enable Multi-turn Context", value=True, help="If checked, the model sees the entire conversation history. If unchecked, it only sees the current question.")

        st.markdown("---")
        st.markdown("### Logging")
        InferenceLogger.show_history()

    # --- Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Inference
        try:
            with st.spinner(f"Generating with {selected_model} + {selected_adapter}..."):
                # Load
                model, tokenizer = load_model_resource(selected_model, selected_adapter)
                
                # Prepare Messages: Context vs Single Turn
                if enable_multiturn:
                    # Pass full history
                    messages_to_process = st.session_state.messages
                else:
                    # Pass only the last message (current prompt)
                    # We create a temporary list with just the user's latest input
                    messages_to_process = [{"role": "user", "content": prompt}]

                # Format Prompt
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages_to_process, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                else:
                    # Fallback (User/Assistant style)
                    formatted_prompt = ""
                    for msg in messages_to_process:
                        role = msg["role"]
                        content = msg["content"]
                        tag = "user" if role == "user" else "assistant"
                        formatted_prompt += f"<|im_start|>{tag}\n{content}<|im_end|>\n"
                    formatted_prompt += "<|im_start|>assistant\n"

                # Generate
                mx.metal.clear_cache()
                start_time = time.time()
                
                # Create sampler with user parameters
                # If enable_sampling is False, force temperature to 0 (Greedy)
                # (Though the slider is already 0, we ensure logic consistency)
                actual_temp = temperature if enable_sampling else 0.0
                sampler = make_sampler(temp=actual_temp)

                raw_response = generate(
                    model, 
                    tokenizer, 
                    prompt=formatted_prompt, 
                    max_tokens=max_tokens, 
                    verbose=False,
                    sampler=sampler
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Post-processing
                clean_response = clean_think_tags(raw_response)
                
                # Stats
                output_tokens = len(tokenizer.encode(raw_response))
                tps = output_tokens / duration if duration > 0 else 0
                peak_mem = mx.metal.get_peak_memory() / 1024**3

                # Display
                with st.chat_message("assistant"):
                    st.markdown(clean_response)
                    st.caption(f"⏱️ {duration:.2f}s | ⚡ {tps:.1f} t/s | 💾 {peak_mem:.2f} GB")
                
                st.session_state.messages.append({"role": "assistant", "content": clean_response})
                
                # Log
                InferenceLogger.log(selected_model, selected_adapter, prompt, clean_response, duration, tps, peak_mem)

        except Exception as e:
            st.error(f"Inference Error: {e}")

if __name__ == "__main__":
    main()
