import csv
import glob
import json
import os
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import get_dataset, balanced_dataloader
from .labels import decode_binary_labels, label_schemes


def run(cfg):
    dir_structure = f"{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}"
    model_output_dir = f"{cfg.model_output}/{dir_structure}"
    label_scheme = label_schemes[cfg.labels]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = get_dataset(cfg, tokenizer)
    dataset = dataset.remove_columns(["label", "language"])
    dataset.set_format("torch", device=cfg.device)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir, num_labels=len(label_scheme)
    )
    with torch.no_grad():
        output = model(**dataset["test"][0:10])

    print(output)
