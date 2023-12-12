from argparse import ArgumentParser
from datetime import datetime
import os
import sys

from v2 import main


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

parser.add_argument("--mode", default="train")

# Data and model

parser.add_argument("--train", required=True)
parser.add_argument("--test")
parser.add_argument("--labels", default="all")
parser.add_argument("--model_name", default="xlm-roberta-base")
parser.add_argument("--data_path", default="data")
parser.add_argument("--output_path", default="output")
parser.add_argument("--transformer_model", default="AutoModelForSequenceClassification")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--torch_dtype")

# Tokenizer

parser.add_argument("--add_prefix_space", action="store_true")  # For peft
parser.add_argument("--low_cpu_mem_usage", action="store_true")
parser.add_argument("--custom_tokenizer")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--return_tensors")

# Visualization

parser.add_argument("--plot")

# Hyperparameter search

parser.add_argument("--hp_search_lib")

# Training arguments

parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--warmup_ratio", type=float, default=0.01)
parser.add_argument("--metric_for_best_model", type=str, default="eval_f1")
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--iter_strategy", type=str, default="epoch")
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_total_limit", type=int, default=2)
parser.add_argument("--optim", default="adamw_torch")
parser.add_argument("--lr_scheduler_type", default="linear")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--max_grad_norm", type=float, default=1)
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--device_map", default="auto")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--resume", action="store_true")

# Balanced sampling

parser.add_argument("--balance")

# Loss function

parser.add_argument("--loss")  # BCEFocalLoss
parser.add_argument("--loss_alpha", type=float, default=0.85)
parser.add_argument("--loss_gamma", type=float, default=3.00)

# (Q)lora / peft

parser.add_argument("--use_flash_attention_2", action="store_true")
parser.add_argument("--add_classification_head", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--peft", action="store_true")
parser.add_argument("--peft_modules")
parser.add_argument("--set_pad_id", action="store_true")
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=float, default=1)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_bias", default="none")

options = parser.parse_args()

print(f"Args: {' '.join(sys.argv)}")
print(f"Settings: {options}")

main.run(options)