from .data import get_dataset
from .mistral_prompt import prompt

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def run():
    dataset = get_dataset("en-fi-fr-sv", "en-fi-fr-sv", "all")

    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, model_max_length=512, padding_side="left", add_eos_token=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(prompt):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        return tokenize(prompt(data_point))

    tokenized_train_dataset = dataset["train"].map(generate_and_tokenize_prompt)
    tokenized_val_dataset = dataset["test"].map(generate_and_tokenize_prompt)

    print(tokenized_train_dataset)
