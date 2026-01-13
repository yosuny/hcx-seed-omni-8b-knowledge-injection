#!/bin/bash

# Training script for CPT on 2511.18659v2 paper data
# Model: 4-bit Quantized HyperCLOVAX-SEED-Omni-8B

.venv/bin/python 99_utils/train_with_early_stopping.py \
    --model models/HyperCLOVAX-SEED-Omni-8B-Text-4bit \
    --train \
    --data data_paper_cpt \
    --batch-size 1 \
    --num-layers 2 \
    --iters 200 \
    --val-batches 1 \
    --steps-per-eval 10 \
    --patience 3 \
    --adapter-path adapters_omni_8b_paper_cpt \
    --learning-rate 2e-5 \
    --seed 42
