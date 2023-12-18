import os

from dotenv import load_dotenv

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from .data import get_dataset
from .mistral_prompt import prompt


def run():
    load_dotenv()
    wandb_project_name = f"mistral_prompt"

    os.environ["WANDB_PROJECT"] = wandb_project_name
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
    os.environ["WANDB_WATCH"] = "all"

    import wandb

    wandb.login()

    print("Using wandb")

    dataset = get_dataset("en-fi-fr-sv", "en-fi-fr-sv", "all", few_shot=10)

    print(dataset)
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

    tokenized_train_dataset = (
        dataset["train"]
        .map(generate_and_tokenize_prompt)
        .remove_columns(
            ["label", "label_text", "language", "text", "id", "split", "length"]
        )
    )
    tokenized_val_dataset = (
        dataset["dev"]
        .map(generate_and_tokenize_prompt)
        .remove_columns(
            ["label", "label_text", "language", "text", "id", "split", "length"]
        )
    )
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

    from peft import prepare_model_for_kbit_training

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    print(model)

    import transformers
    from datetime import datetime

    project = "register_classes"
    base_model_name = "mistral"
    run_name = base_model_name + "-" + project
    output_dir = "./" + run_name

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            learning_rate=1e-5,  # Want a small lr for finetuning
            bf16=True,
            optim="paged_adamw_8bit",
            num_train_epochs=1,
            save_total_limit=2,
            logging_steps=50,  # When to start reporting loss
            logging_dir="./logs",  # Directory for storing logs
            save_strategy="steps",  # Save the model checkpoint every logging step
            save_steps=50,  # Save checkpoints every 50 steps
            evaluation_strategy="steps",  # Evaluate the model every logging step
            eval_steps=50,  # Evaluate and save checkpoints every 50 steps
            do_eval=True,  # Perform evaluation at the end of training
            report_to="wandb",  # Comment this out if you don't want to use weights & baises
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name of the W&B run (optional)
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )

    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()
