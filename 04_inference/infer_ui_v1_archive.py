import streamlit as st
import os
import glob
import pandas as pd
import json
import time
import csv
import re
from datetime import datetime
import mlx.core as mx
from mlx_lm import load, generate

# --- Constants & Settings ---
MODELS_DIR = "models"
LOG_FILE = "inference_logs.csv"
PAGE_TITLE = "HCX Omni 8B Knowledge Injection"
PAGE_ICON = "🧠"

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- Helper Functions ---

def clean_think_tags(text):
    """Removes <think>...</think> blocks from the text."""
    if not text:
        return ""
    # Remove standard <think> tags (dotall to handle newlines)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove standalone tags if any remain
    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    # Remove other known thinking tokens if they appear
    cleaned = cleaned.replace("<|thinking|>", "").replace("<|/thinking|>", "")
    return cleaned.strip()

def get_available_models():
    """Scans the MODELS_DIR for subdirectories (assumed to be models) and filters supported ones."""
    if not os.path.exists(MODELS_DIR):
        return []
    
    models = []
    try:
        for d in os.listdir(MODELS_DIR):
            if d == ".DS_Store":
                continue
                
            model_path = os.path.join(MODELS_DIR, d)
            if os.path.isdir(model_path):
                # Check config.json to confirm it's a model
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r") as f:
                            json.load(f)
                    except:
                        pass
            
            # Filter specifically for 4-bit and 8-bit Omni models as requested
            # Or simplified: accept if it looks like a model directory
            if os.path.isdir(model_path):
                 models.append(d)
                 
    except Exception as e:
        print(f"Error scanning models: {e}")
            
    models.sort()
    return models

def get_available_adapters():
    """Scans for adapter directories starting with 'adapters_'."""
    adapters = ["None"]
    try:
        for d in os.listdir("."):
            if d == ".DS_Store":
                continue
                
            if os.path.isdir(d) and d.startswith("adapters_"):
                adapters.append(d)
    except Exception as e:
        print(f"Error scanning adapters: {e}")
        
    adapters.sort()
    return adapters

@st.cache_resource
def load_model_cached(model_name, adapter_name):
    """Loads the model and tokenizer, cached by Streamlit."""
    model_path = os.path.join(MODELS_DIR, model_name)
    print(f"Loading model from {model_path}...")
    
    adapter_path = None
    if adapter_name and adapter_name != "None":
        adapter_path = adapter_name
        print(f"Loading adapter from {adapter_path}...")

    # Set fix_mistral_regex=True implicitly for safety based on recent issues
    model, tokenizer = load(model_path, adapter_path=adapter_path, tokenizer_config={"fix_mistral_regex": True})
    return model, tokenizer

def append_to_log(model_name, adapter_name, prompt, response, time_taken, tps, peak_mem):
    """Appends the inference result to the CSV log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Check if file exists to write header
    file_exists = os.path.isfile(LOG_FILE)
    
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Model", "Adapter", "Prompt", "Response", "Time(s)", "Speed(TPS)", "Memory(GB)"])
        writer.writerow([timestamp, model_name, adapter_name, prompt, response, f"{time_taken:.2f}", f"{tps:.2f}", f"{peak_mem:.2f}"])

# --- UI Layout ---

st.title(f"{PAGE_ICON} {PAGE_TITLE}")

# Sidebar: Controls
with st.sidebar:
    st.header("Settings")
    
    # Model Selection
    available_models = get_available_models()
    if not available_models:
        st.error(f"No models found in `{MODELS_DIR}` directory!")
        st.stop()
        
    selected_model_name = st.selectbox("Select Model", available_models, index=0)

    # Adapter Selection
    available_adapters = get_available_adapters()
    
    # Try to set default to SFT adapter
    default_adapter_index = 0
    target_adapter = "adapters_omni_8b_paper_sft"
    if target_adapter in available_adapters:
        default_adapter_index = available_adapters.index(target_adapter)
        
    selected_adapter_name = st.selectbox(
        "Select Adapter (Optional)", 
        available_adapters, 
        index=default_adapter_index
    )
    
    # Parameters
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=64, max_value=2048, value=512, step=64)
    do_sample = st.checkbox("Do Sample", value=True)
    
    # Reload Button
    if st.button("Reload Model"):
        st.cache_resource.clear()
        st.success("Cache cleared. Model will reload on next inference.")

    st.markdown("---")
    st.markdown("### Logging")
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE, on_bad_lines='skip') # Robustly read even if cols mismatch
            st.write(f"Total Logs: {len(df)}")
            if st.checkbox("Show History"):
                st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
                
                # Download button
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Logs CSV",
                    data=csv_data,
                    file_name="inference_logs.csv",
                    mime="text/csv",
                )
        except Exception as e:
             st.warning(f"Could not load log file: {e}")

# Main Area: Chat Interface

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("메시지를 입력하세요..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Model Inference
    try:
        with st.spinner(f"Generating with {selected_model_name} + {selected_adapter_name}..."):
            # Load model
            model, tokenizer = load_model_cached(selected_model_name, selected_adapter_name)
            
            # Prepare Prompt (Apply template if available)
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                formatted_prompt = tokenizer.apply_chat_template(st.session_state.messages, add_generation_prompt=True, tokenize=False)
            else:
                # Simple fallback
                formatted_prompt = ""
                for msg in st.session_state.messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "user":
                        formatted_prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        formatted_prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                formatted_prompt += "<|im_start|>assistant\n"

            # Generation
            start_time = time.time()
            mx.metal.clear_cache() # Optional cleanup
            
            # Generate response
            raw_response_text = generate(
                model, 
                tokenizer, 
                prompt=formatted_prompt, 
                max_tokens=max_tokens, 
                verbose=False
            )
            
            # Clean response for display
            response_text = clean_think_tags(raw_response_text)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Calculate stats
            output_tokens = len(tokenizer.encode(raw_response_text)) # Use raw for correct TPS
            tps = output_tokens / duration if duration > 0 else 0
            peak_mem = mx.metal.get_peak_memory() / 1024**3

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response_text)
                st.caption(f"⏱️ {duration:.2f}s | ⚡ {tps:.1f} t/s | 💾 {peak_mem:.2f} GB")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Log to CSV
            append_to_log(selected_model_name, selected_adapter_name, prompt, response_text, duration, tps, peak_mem)

    except Exception as e:
        st.error(f"Error during generation: {e}")
