#!/bin/bash

# Script to run inference_module.py with arguments

python3 src/inference_module.py \
  --model_id "unsloth/Llama-3.2-3B-bnb-4bit" \
  --tokenizer_id "Naod-Demissie/Llama-3.2-3B-bnb-4bit-amh-128k" \
  --lora_repo "your-huggingface-username/lora-adapters-repo" \
  --prompt "አዲስ አበባ የኢትዮጵያ " \
  --max_new_tokens 64
