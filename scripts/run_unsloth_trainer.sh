#!/bin/bash

# Script to run unsloth-trainer.py with arguments from the notebook

python3 unsloth-trainer.py \
  --model_id "unsloth/Llama-3.2-3B-bnb-4bit" \
  --tokenizer_id "Naod-Demissie/Llama-3.2-3B-bnb-4bit-amh-128k" \
  --data_path "data/LMTextData_normalized_unique_rm_unkfidels_rm_leng1_2.txt" \
  --output_dir "outputs" \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --num_epochs 5 \
  --run_name "local-llm-unsloth-mode-llama-3.2-3B-bnb-4bit-128k"
