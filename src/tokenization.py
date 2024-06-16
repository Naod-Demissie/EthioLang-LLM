import os
from datasets import load_dataset
from transformers import AutoTokenizer

def train_tokenizer(data_path: str, model_id: str, vocab_size: int, output_dir: str):
    """Trains a new tokenizer from the given dataset."""
    data = load_dataset("text", data_files=data_path, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    new_tokenizer = tokenizer.train_new_from_iterator(data["text"], vocab_size=vocab_size)
    new_tokenizer.save_pretrained(output_dir)
    return new_tokenizer

def merge_tokenizers(base_tokenizer_path: str, new_tokens: list, output_dir: str):
    """Merges new tokens into an existing tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_path)
    tokenizer.add_tokens(new_tokens)
    tokenizer.save_pretrained(output_dir)
    return tokenizer

def push_tokenizer_to_hub(tokenizer_path: str, hub_repo: str, private: bool = True):
    """Pushes a tokenizer to the Hugging Face Hub."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.push_to_hub(hub_repo, private=private)

def save_and_push_tokenizer(tokenizer, output_dir: str, hub_repo: str, private: bool = True):
    """Saves a tokenizer locally and pushes it to the Hugging Face Hub."""
    tokenizer.save_pretrained(output_dir)
    tokenizer.push_to_hub(hub_repo, private=private)
