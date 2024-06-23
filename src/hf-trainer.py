import os
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)


# Load environment variables from .env file
load_dotenv()


class HFTrainer:
    def __init__(self, model_id, tokenizer_id, data_path, max_seq_length=2048):
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.data_path = data_path
        self.max_seq_length = max_seq_length

    def preprocess_model_and_tokenizer(self):
        # Load model and tokenizer
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            use_auth_token=os.getenv("HF_TOKEN"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            trust_remote_code=True,
            use_auth_token=os.getenv("HF_TOKEN"),
        )

        # Update special tokens
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

        # Resize token embeddings
        model.resize_token_embeddings(len(tokenizer))

        return model, tokenizer

    def prepare_dataset(self):
        # Load dataset
        dataset = load_dataset("text", data_files=self.data_path, split="train")

        # Add EOS token to each example
        eos_token = AutoTokenizer.from_pretrained(self.tokenizer_id).eos_token

        def formatting_prompts_func(examples):
            return {"text": [example + eos_token for example in examples["text"]]}

        dataset = dataset.map(formatting_prompts_func, batched=True)
        return dataset

    def train_model(
        self,
        output_dir,
        batch_size=2,
        gradient_accumulation_steps=16,
        num_epochs=5,
        run_name="default-run",
    ):
        # Preprocess model and tokenizer
        model, tokenizer = self.preprocess_model_and_tokenizer()

        # Prepare dataset
        dataset = self.prepare_dataset()

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.1,
            num_train_epochs=num_epochs,
            learning_rate=5e-4,
            fp16=torch.cuda.is_available(),
            logging_steps=1,
            optim="adamw_torch",
            weight_decay=0.00,
            lr_scheduler_type="cosine",
            seed=3407,
            report_to="wandb",
            save_steps=50000,
            run_name=run_name,
        )

        # Initialize trainer
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            args=training_args,
        )

        # Train the model
        trainer.train()

        # Save the model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model using HFTrainer")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier")
    parser.add_argument(
        "--tokenizer_id", type=str, required=True, help="Tokenizer identifier"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the training data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--run_name", type=str, default="default-run", help="Name of the training run"
    )

    args = parser.parse_args()

    trainer = HFTrainer(
        model_id=args.model_id,
        tokenizer_id=args.tokenizer_id,
        data_path=args.data_path,
    )
    trainer.train_model(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        run_name=args.run_name,
    )
