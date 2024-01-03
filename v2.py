from argparse import ArgumentParser
from datetime import datetime
import os
import sys
import tempfile

project = "/scratch/project_2009199"

tempfile.tempdir = f"{project}/tmp"

_print = print


# Print with datetime
def print(*args, **kw):
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _print(f"[{formatted_now}]", *args, **kw)


os.environ["TRANSFORMERS_CACHE"] = ".hf/transformers_cache"
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"


# Get CLI options

parser = ArgumentParser()

# Mode of operation

parser.add_argument("--mode", "-m", default="train")

# Data and model

parser.add_argument("--train", "-t", required=True)
parser.add_argument("--test")
parser.add_argument("--labels", default="all")
parser.add_argument("--model_name", default="xlm-roberta-base")
parser.add_argument("--data_path", default="data")
parser.add_argument("--output_path", default="output")
parser.add_argument("--transformer_model", default="AutoModelForSequenceClassification")
parser.add_argument("--seed", "-s", type=int, default=42)
parser.add_argument("--torch_dtype")
parser.add_argument("--accelerate", action="store_true")
parser.add_argument("--wandb_project", "-w")
parser.add_argument("--cwd")
parser.add_argument("--quit", "-q", action="store_true")
parser.add_argument("--ignore_mismatched_sizes", action="store_true")

# Tokenizer

parser.add_argument("--add_prefix_space", action="store_true")  # For peft
parser.add_argument("--low_cpu_mem_usage", action="store_true")
parser.add_argument("--custom_tokenizer")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--return_tensors")
parser.add_argument("--use_slow", action="store_true")
parser.add_argument("--not_legacy", action="store_true")
parser.add_argument("--add_special_tokens", action="store_true")

# Visualization

parser.add_argument("--plot", "-p")

# Hyperparameter search

parser.add_argument("--hp_search_lib", default="ray")
parser.add_argument("--ray_log_path", default=f"{project}/log")
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--max_lr", type=float, default=1e-4)

# Training arguments

parser.add_argument("--lr", "-lr", type=float, default=1e-5)
parser.add_argument("--train_batch_size", "-b", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--weight_decay", "-wd", type=float, default=0.05)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--warmup_ratio", type=float, default=0.01)
parser.add_argument("--metric_for_best_model", type=str, default="eval_f1")
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--iter_strategy", type=str, default="epoch")
parser.add_argument("--eval_steps", type=int, default=None)
parser.add_argument("--logging_steps", type=int, default=None)
parser.add_argument("--save_steps", type=int, default=None)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_total_limit", type=int, default=2)
parser.add_argument("--optim", default="adamw_torch")
parser.add_argument("--lr_scheduler_type", default="linear")
parser.add_argument("--max_grad_norm", type=float, default=1)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--device_map", default="auto")
parser.add_argument("--infer_device_map", action="store_true"),
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--tf32", action="store_true")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--num_gpus", "-g", type=int)

# Balanced sampling

parser.add_argument("--balance", action="store_true")

# Loss function, uses BCEFocalLoss by default

parser.add_argument("--loss", "-l", default="BCEFocalLoss")
parser.add_argument("--loss_alpha", type=float, default=0.85)
parser.add_argument("--loss_gamma", type=float, default=3.00)
parser.add_argument("--loss_penalty", type=float, default=10.00)

# (Q)lora / peft

parser.add_argument("--use_flash_attention_2", action="store_true")
parser.add_argument("--add_classification_head", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--kbit", action="store_true")
parser.add_argument("--peft", action="store_true")
parser.add_argument("--peft_modules")
parser.add_argument("--set_pad_id", action="store_true")
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=float, default=1)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_bias", default="none")

# LLM specific settings

parser.add_argument("--few_shot", type=int, default=0)
parser.add_argument("--llm", action="store_true")
parser.add_argument("--left_padding", action="store_true")


options = parser.parse_args()

print(f"Args: {' '.join(sys.argv)}")
print(f"Settings: {options}")

if options.cwd:
    os.chdir(options.cwd)

if options.quit:
    print(os.getcwd())
    exit()

if options.num_gpus:
    os.environ[
        "CUDA_VISIBLE_DEVICES"
    ] = f"{','.join([str(x) for x in range(0, options.num_gpus)])}"

    print(f'Chosen GPUs: [{os.environ["CUDA_VISIBLE_DEVICES"]}]')

from torch import cuda

print(f"Cuda has {cuda.device_count()} GPUs")

if not cuda.device_count():
    print("No GPUs!")
    exit()

from v2 import main

main.run(options)
