import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from peft import PeftModel


# Load environment variables from .env file
load_dotenv()


class HFInference:
    def __init__(self, model_id, tokenizer_id, lora_repo, max_seq_length=2048):
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.lora_repo = lora_repo
        self.max_seq_length = max_seq_length

    def load_model_and_tokenizer(self):
        # Load base model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=os.getenv("HF_TOKEN"),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            trust_remote_code=True,
            use_auth_token=os.getenv("HF_TOKEN"),
        )

        # Load LoRA adapters from Hugging Face Hub and merge them into the model
        model = PeftModel.from_pretrained(
            model,
            self.lora_repo,
            use_auth_token=os.getenv("HF_TOKEN"),
        )
        model = model.merge_and_unload()

        return model, tokenizer

    def generate_completion(self, prompt, max_new_tokens=64):
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()

        # Tokenize input
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generate text
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, use_cache=True
        )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inference using a fine-tuned model with LoRA adapters"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, help="Base model identifier"
    )
    parser.add_argument(
        "--tokenizer_id", type=str, required=True, help="Tokenizer identifier"
    )
    parser.add_argument(
        "--lora_repo",
        type=str,
        required=True,
        help="Hugging Face repository for the LoRA adapters",
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt for text generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate",
    )

    args = parser.parse_args()

    inference = HFInference(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        lora_repo=args.lora_repo,
    )
    result = inference.generate_completion(
        prompt=args.prompt, max_new_tokens=args.max_new_tokens
    )
    print("Generated Text:")
    print(result[0])
