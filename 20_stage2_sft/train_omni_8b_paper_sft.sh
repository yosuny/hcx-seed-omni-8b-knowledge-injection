#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

# Configuration
MODEL_PATH="models/HyperCLOVAX-SEED-Omni-8B-Text-4bit"
ADAPTER_PATH="adapters_omni_8b_paper_sft"
DATA_DIR="data_paper_sft"

echo "Starting SFT on $DATA_DIR with adapter $ADAPTER_PATH..."

# Ensure adapter config exists (it should from cp)
if [ ! -d "$ADAPTER_PATH" ]; then
    echo "Adapter path $ADAPTER_PATH does not exist!"
    exit 1
fi

.venv/bin/python 99_utils/train_with_early_stopping.py \
    --model $MODEL_PATH \
    --adapter-path $ADAPTER_PATH \
    --data $DATA_DIR \
    --batch-size 1 \
    --num-layers 2 \
    --iters 100 \
    --learning-rate 1e-5 \
    --steps-per-eval 10 \
    --save-every 20 \
    --patience 5 \
    --train
