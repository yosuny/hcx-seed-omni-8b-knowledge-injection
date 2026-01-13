import subprocess
import os
import sys

# Settings
INPUT_MODEL = "models/HyperCLOVAX-SEED-Omni-8B-Text"
OUTPUT_MODEL = "models/HyperCLOVAX-SEED-Omni-8B-Text-8bit"

def quantize_model():
    print(f"Quantizing {INPUT_MODEL} to {OUTPUT_MODEL} (8-bit)...")
    
    # Check if input exists
    if not os.path.exists(INPUT_MODEL):
        print(f"Error: Input model {INPUT_MODEL} not found.")
        return

    # Run mlx_lm.convert
    # -q: quantize
    # --q-bits 8: 8-bit quantization
    # --mlx-path: output path
    command = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", INPUT_MODEL,
        "--mlx-path", OUTPUT_MODEL,
        "-q",
        "--q-bits", "8"
    ]
    
    try:
        subprocess.run(command, check=True)
        print("Quantization complete!")
        print(f"Saved to {OUTPUT_MODEL}")
    except subprocess.CalledProcessError as e:
        print(f"Error during quantization: {e}")

if __name__ == "__main__":
    quantize_model()
