from .data import get_dataset
from .mistral_prompt import prompt

dataset = get_dataset("en-fi-fr-sv", "en-fi-fr-sv", "all")

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def run():
    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir="cache",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 2048  # This was an appropriate max length for my dataset

    def generate_and_tokenize_prompt(example):
        result = tokenizer(
            prompt(example),
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = dataset["train"].map(generate_and_tokenize_prompt)
    tokenized_val_dataset = dataset["dev"].map(generate_and_tokenize_prompt)
    """
    import matplotlib.pyplot as plt

    def plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset):
        lengths = [len(x["input_ids"]) for x in tokenized_train_dataset]
        lengths += [len(x["input_ids"]) for x in tokenized_val_dataset]
        print(len(lengths))

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(lengths, bins=20, alpha=0.7, color="blue")
        plt.xlabel("Length of input_ids")
        plt.ylabel("Frequency")
        plt.title("Distribution of Lengths of input_ids")
        plt.show()

    plot_data_lengths(tokenized_train_dataset, tokenized_val_dataset)
    """

    eval_prompt = prompt(dataset["train"][0], labels=False)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
    )

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    model.eval()
    with torch.no_grad():
        print(
            tokenizer.decode(
                model.generate(
                    **model_input, max_new_tokens=256, repetition_penalty=1.15
                )[0],
                skip_special_tokens=True,
            )
        )
