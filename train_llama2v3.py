import os

os.environ["TRANSFORMERS_CACHE"] = ".hf/transformers_cache"
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)

import bitsandbytes as bnb

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    AutoPeftModelForCausalLM,
)

seed = 42
max_length = 4096
output_dir = "./llama_results"

from huggingface_hub import login

login()

data_path = "data/en/train_instruction_inst.tsv"
model_name = "meta-llama/Llama-2-7b-hf"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto",
    max_memory={0: "32000MB"},
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load dataset
dataset = load_dataset("csv", data_files=data_path, split="train", sep="\t")

print(f"Number of prompts: {len(dataset)}")


def tokenize(example):
    return tokenizer(
        example["text"],
        # truncation=True,
        # max_length=max_length,
        padding=True,
    )


dataset = dataset.map(tokenize)

# Filter out samples that have "input_ids" exceeding "max_length"
dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

# Shuffle dataset
dataset = dataset.shuffle(seed=seed)


print(f"Filtered examples: {len(dataset)}")
print(f"Example 1: {dataset[0]}")

"""Initializing QLoRA and TrainingArguments parameters below for training."""

# Enable gradient checkpointing to reduce memory usage during fine-tuning
model.gradient_checkpointing_enable()

# Prepare the model for training
model = prepare_model_for_kbit_training(model)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)


model = get_peft_model(
    model,
    LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    ),
)

# Training parameters
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=2,
        max_steps=500,
        learning_rate=5e-6,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

model.config.use_cache = False

print("Training...")

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
print(metrics)

# Save model
print("Saving last checkpoint of the model...")
os.makedirs(output_dir, exist_ok=True)
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del model
del trainer
torch.cuda.empty_cache()

# Load fine-tuned weights
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir, device_map="auto", torch_dtype=torch.bfloat16
)
# Merge the LoRA layers with the base model
model = model.merge_and_unload()

# Save fine-tuned model at a new location
output_merged_dir = (
    f"{output_dir}/register_classification_llama2_7b/final_merged_checkpoint"
)
os.makedirs(output_merged_dir, exist_ok=True)
model.save_pretrained(output_merged_dir, safe_serialization=True)

# Save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(output_merged_dir)

# Fine-tuned model name on Hugging Face Hub
new_model = "erikhenriksson/register-classification-25-llama-2-7b"

# Push fine-tuned model and tokenizer to Hugging Face Hub
model.push_to_hub(new_model, use_auth_token=True)
tokenizer.push_to_hub(new_model, use_auth_token=True)
