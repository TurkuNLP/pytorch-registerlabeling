INSTRUCTION = """
Categorize the following web-scraped text into one or more of the following categories:

MT, LY, SP, ID, NA, HI, IN, OP, IP

Give the label(s) based on the following instructions:

MT: The web page is machine translated or generated from a template.
LY: The web page is lyrical, such as songs or poems.
SP: The web page is originally spoken (e.g. interview).
ID: The web page is an interactive discussion written by multiple participants in a discussion format (e.g. discussion or Q&A forum).
NA: The purpose of the document is to narrate or report on events (e.g. news report, sports, report, narrative blog).
HI: The purpose of the document is to explain how-to or instructions (e.g. recipe or typically other step step-by-step, objective instructions on how to do something).
IN: The purpose of the document is to describe or explain information (e.g. encyclopedia article, research article, description of a thing or person, FAQ, Legal terms and conditions, course materials and blogs for informing the reader)
OP: The purpose of the document to express opinions (review, opinion blog typically written by an amateur writer, such as a politician, to express their opinion, denominational religious blog / sermon, advice).
IP: The purpose of the document is to describe or explain facts with intent to persuade or market (e.g. description with intent to sell a product or service, a news & opinion blog or editorial typically written by a professional writer on a news-related topic with well-structured argumentation, or other descriptive texts that e.g. sell or promote a service, product or upcoming event, such as a hotel, a smartphone or a football game).

Give the label(s) as a space-separated list. If you are unsure, output "None". 
"""
import csv
import glob
import json
import os
import random
import gzip
import shutil
from pydoc import locate

from datasets import Dataset, DatasetDict, concatenate_datasets
import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

from .data import balanced_dataloader, get_dataset
from .labels import (
    binarize_labels,
    normalize_labels,
    decode_binary_labels,
    label_schemes,
    subcategory_to_parent_index,
    map_to_xgenre_binary,
    upper_all_indexes,
)

small_languages = [
    "ar",
    "ca",
    "es",
    "fa",
    "hi",
    "id",
    "jp",
    "no",
    "pt",
    "ur",
    "zh",
]


def get_linear_modules(model):

    linear_modules = set()

    for name, module in model.named_modules():
        name = name.lower()
        if "attention" in name and "self" in name and "Linear" in str(type(module)):
            linear_modules.add(name.split(".")[-1])

    print(f"\nFound linear modules: {linear_modules}")
    return list(linear_modules)


# Data loading

file_opener = lambda file_path, use_gz: (
    gzip.open(file_path + ".gz", "rt", encoding="utf-8")
    if use_gz
    else open(file_path, "r")
)


def format_prompt(input, output):
    return f"### Instruction: {INSTRUCTION} \n### Output: {output} \n### Input text: {input}"


def gen(languages, split, label_scheme, use_gz):
    for l in languages:

        with file_opener(
            f"data/{l}/{split if l not in small_languages else l}.tsv", use_gz
        ) as c:
            re = csv.reader(c, delimiter="\t")
            for ro in re:
                if not (ro[0] and ro[1]):
                    continue
                normalized_labels = normalize_labels(ro[0], label_scheme)
                if not normalized_labels:
                    continue

                yield {"text": format_prompt(ro[1], " ".join(normalized_labels))}


def get_dataset(cfg, tokenizer):
    def process(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    generate = lambda split: Dataset.from_generator(
        gen,
        gen_kwargs={
            "languages": dict(cfg)[split].split("-"),
            "split": split,
            "label_scheme": cfg.labels,
            "use_gz": cfg.use_gz,
        },
    )
    splits = {}
    include_splits = ["train", "dev", "test"] if not cfg.just_evaluate else ["test"]
    for s in include_splits:
        split = generate(s)
        if cfg.sample:
            split = split.select(range(min(len(split), cfg.sample)))
        splits[s] = split

    dataset = DatasetDict(splits).shuffle(seed=cfg.seed)

    return dataset.map(
        process,
        batched=True,
    )


def run(cfg):

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    base_model_id = "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = get_dataset(cfg, tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, resume_download=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    eval_prompt = format_prompt(
        "The quick brown fox jumps over the lazy dog.",
        "",
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
