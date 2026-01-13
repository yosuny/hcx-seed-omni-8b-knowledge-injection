import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
import json
import glob
import os
import shutil
from huggingface_hub import snapshot_download

# Settings
MODEL_ID = "naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B"
SRC_PATH = "models/HyperCLOVAX-SEED-Omni-8B"
DEST_PATH = "models/HyperCLOVAX-SEED-Omni-8B-Text"

def download_model():
    if not os.path.exists(SRC_PATH):
        print(f"Downloading {MODEL_ID} to {SRC_PATH}...")
        snapshot_download(repo_id=MODEL_ID, local_dir=SRC_PATH)
    else:
        print(f"Model already exists at {SRC_PATH}")

def extract_text_backbone():
    print("Loading config...")
    config_path = os.path.join(SRC_PATH, "config.json")
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Extract text config
    if "text_config" in full_config:
        text_config = full_config["text_config"]
        # Ensure model_type is set for MLX (usually 'llama' for this arch)
        # text_config['model_type'] is already 'llama' in the provided json
        print("Found text_config, using it as main config.")
    else:
        print("Error: 'text_config' not found in config.json")
        return

    # Create destination directory
    os.makedirs(DEST_PATH, exist_ok=True)

    # Save new config
    # We might need to adjust some params if necessary, but usually raw extraction is fine.
    # MLX Llama model expects standard LlamaConfig keys.
    # checking text_config content from previous grep: it has 'model_type': 'llama', 'vocab_size' etc.
    # It should work directly with mlx_lm.
    
    with open(os.path.join(DEST_PATH, "config.json"), "w") as f:
        json.dump(text_config, f, indent=2)
    print(f"Saved new config to {os.path.join(DEST_PATH, 'config.json')}")

    # Process weights
    print("Processing weights...")
    weight_files = glob.glob(os.path.join(SRC_PATH, "*.safetensors"))
    weight_files.sort()
    
    # We will iterate through files, load, filter, and save.
    # Since we are creating a split model, we can try to save one file per original file to keep it simple,
    # or merge them. Given 8B is small enough, we might merge or keep properly.
    # Safer to just process and save.
    
    for wf in weight_files:
        basename = os.path.basename(wf)
        print(f"Processing {basename}...")
        
        weights = mx.load(wf)
        new_weights = {}
        
        for k, v in weights.items():
            # Check for language model keys
            if k.startswith("model.language_model."):
                # Remove prefix
                # Map: model.language_model.model.layers... -> model.layers...
                # Map: model.language_model.lm_head -> lm_head
                
                if k.startswith("model.language_model.model."):
                    new_k = k.replace("model.language_model.model.", "model.")
                    new_weights[new_k] = v
                elif k.startswith("model.language_model.lm_head."):
                    new_k = k.replace("model.language_model.lm_head.", "lm_head.")
                    new_weights[new_k] = v
                else:
                    # Potential other keys like norms if they are not inside .model
                    # Looking at index file: model.language_model.model.norm.weight might exist
                    # Let's handle generic replacement just in case
                    new_k = k.replace("model.language_model.", "")
                    # But wait, if we have model.language_model.model.norm, removing prefix makes model.norm
                    # If we have model.language_model.lm_head, removing prefix makes lm_head
                    # Seems correct?
                    # WARNING: 'model.language_model.model' -> 'model'
                    # But 'model.language_model.lm_head' -> 'lm_head'
                    
                    # Implementation detail:
                    # k = "model.language_model.model.layers.0..."
                    # replace("model.language_model.", "") -> "model.layers.0..." -> CORRECT
                    
                    # k = "model.language_model.lm_head.weight"
                    # replace("model.language_model.", "") -> "lm_head.weight" -> CORRECT
                    
                    # So simple replacement of prefix should work?
                    new_weights[new_k] = v
                    
        if new_weights:
            print(f"  -> Extracted {len(new_weights)} tensors.")
            save_path = os.path.join(DEST_PATH, basename)
            mx.save_safetensors(save_path, new_weights)
        else:
            print("  -> No text model weights found in this file (likely audio/vision only). Skipping.")

    # Copy Tokenizer files
    print("Copying tokenizer files...")
    for f in glob.glob(os.path.join(SRC_PATH, "tokenizer*")):
        shutil.copy(f, DEST_PATH)
    
    # Copy specialized token configs/dicts if any
    for f in glob.glob(os.path.join(SRC_PATH, "*.txt")): # like merges.txt
        shutil.copy(f, DEST_PATH)
    
    for f in glob.glob(os.path.join(SRC_PATH, "*.json")):
        if "tokenizer" in f or "special_tokens" in f or "map" in f:
             shutil.copy(f, DEST_PATH)

    print("Extraction complete!")
    print(f"You can now run inference using: mlx_lm.generate --model {DEST_PATH}")

if __name__ == "__main__":
    download_model()
    extract_text_backbone()
