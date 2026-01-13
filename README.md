# HyperCLOVA X Seed (Omni 8B) Knowledge Injection Project

This project simulates the full lifecycle of adapting a Large Language Model (LLM) to a specific domain on Apple Silicon (M-series chips). We use **HyperCLOVA X Seed Omni 8B** as the base model and inject knowledge from the **CLaRa** paper (Continuous Latent Reasoning) using MLX-LM.

## Project Overview

The goal is to enable the model to correctly answer expert-level questions about the "CLaRa" framework and "SCP" (Salient Compressor Pretraining) without hallucinating common acronyms (e.g., confusing SCP with Secure Copy Protocol).

**Key Methodologies:**
1.  **4-bit Quantization**: Compressing the model to ~5.1GB to run efficiently on local hardware.
2.  **CPT (Continuous Pre-Training)**: Ingesting raw paper text (`.jsonl`) to build a knowledge base.
3.  **SFT (Supervised Fine-Tuning) with Golden Data**: Using synthesized Q&A and manually curated "Golden Data" to teach the model *how* to use that knowledge and fix specific hallucinations.

## Directory Structure

| Directory | Description |
| :--- | :--- |
| `models/` | Contains the 4-bit quantized base model (`HyperCLOVAX-SEED-Omni-8B-Text-4bit`). |
| `adapters_omni_8b_paper_cpt/` | LoRA adapters from Stage 1 (CPT). Contains knowledge weights. |
| `adapters_omni_8b_paper_sft/` | **Final LoRA adapters** from Stage 2 (SFT). Use this for inference. |
| `03_quantization/` | Scripts for converting and quantizing the original model. |
| `04_inference/` | **Streamlit UI** (`infer_ui.py`) for interactive testing. |
| `10_stage1_cpt/` | Training scripts and data gen for Continuous Pre-Training. |
| `20_stage2_sft/` | Training scripts (`train_omni_8b_paper_sft.sh`) and Golden Data injection. |
| `30_evaluation/` | Benchmark (`compare_hcx_stages.py`) and verification scripts. |
| `reference/` | Legacy files (32B model, SolverX experiments) archive. |

## Workflow & Results

### Stage 1: CPT (Knowledge Injection)
- **Objective**: Teach the model the raw content of the CLaRa paper.
- **Process**: Trained on 36 text chunks from the PDF.
- **Outcome**: The model learned the text patterns but struggled to retrieve definitions in a Q&A format (e.g., defining "SCP" as "Self-optimizing Presenter").

### Stage 2: SFT (Instruction Tuning & Correction)
- **Objective**: Fix hallucinations and enable natural Q&A.
- **Process**:
    1.  Generated synthetic Q&A pairs from the CPT data.
    2.  **Injected "Golden Data"**: Manually added precise definitions for *CLaRa*, *SCP*, and *Joint Training*.
    3.  Retrained using the CPT adapter as a starting point (Continual Learning).
- **Outcome**: The model now correctly identifies "SCP" as **"Salient Compressor Pretraining"** and "CLaRa" as **"Continuous Latent Reasoning"**.

### Benchmark Results
We compared the three stages of the model:

| Model Stage | CLaRa Definition | SCP Definition | Inference Speed |
| :--- | :--- | :--- | :--- |
| **Base (4-bit)** | Wrong (Hallucination) | Wrong (Secure Copy Protocol) | ~15.9 t/s |
| **CPT** | Unstable | Wrong (Hallucination) | ~15.8 t/s |
| **SFT (Final)** | **Correct** | **Correct** | **~15.7 t/s** |

## How to Run

### 1. Interactive Inference (UI)
The easiest way to test the model is via the Streamlit UI.

```bash
# Activate virtual environment
source .venv/bin/activate

# Run UI
streamlit run 04_inference/infer_ui.py
```
- **Select Model**: `HyperCLOVAX-SEED-Omni-8B-Text-4bit`
- **Select Adapter**: `adapters_omni_8b_paper_sft` (For best results)

### 2. Verify via CLI
Run the verification script to see the model answer the key benchmark questions.

```bash
python 30_evaluation/verify_paper_sft.py
```

### 3. Run Benchmark Comparison
To reproduce the table above:

```bash
python 30_evaluation/compare_hcx_stages.py
```

## Paper for knowledge injection
- **Title**: [CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning](https://arxiv.org/abs/2511.18659)
- **Source**: arXiv:2511.18659
- **License**: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)

## Acknowledgements
This project was forked and spun-off from [hcx-seed-think-32b-knowledge-injection](https://github.com/shezgone/hcx-seed-think-32b-knowledge-injection). Special thanks to the original authors for the foundational work.

