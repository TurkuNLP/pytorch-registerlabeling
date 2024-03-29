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
    BitsAndBytesConfig,
)

from .data import get_dataset, balanced_dataloader
from .labels import decode_binary_labels, label_schemes


def get_linear_modules(model):

    linear_modules = set()

    for name, module in model.named_modules():
        name = name.lower()
        if "attention" in name and "self" in name and "Linear" in str(type(module)):
            linear_modules.add(name.split(".")[-1])

    print(f"\nFound linear modules: {linear_modules}")
    return list(linear_modules)


def run(cfg):

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_language = ""  # Used when predicting
    label_scheme = label_schemes[cfg.labels]
    dir_structure = f"{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}{('/fold_'+str(cfg.use_fold)) if cfg.use_fold else ''}"
    model_output_dir = f"{cfg.model_output}/{dir_structure}"
    results_output_dir = f"results/{dir_structure}"

    print(f"This run saves models to {model_output_dir}")
    print(f"Results are logged to {results_output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if "mixtral" in cfg.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
    dataset = get_dataset(cfg, tokenizer)

    class MultiLabelTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super(MultiLabelTrainer, self).__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            BCE_loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(), reduction="none"
            )
            pt = torch.exp(-BCE_loss)
            loss = cfg.loss_alpha * (1 - pt) ** cfg.loss_gamma * BCE_loss

            # Class balancing
            loss = loss * (
                labels * cfg.loss_alpha + (1 - labels) * (1 - cfg.loss_alpha)
            )
            loss = loss.mean()

            return (loss, outputs) if return_outputs else loss

        if len(cfg.train.split("-")) > 1 and cfg.method != "test":

            def get_train_dataloader(self):
                return balanced_dataloader(self, "train", cfg.train_batch_size)

        if len(cfg.dev.split("-")) > 1 and cfg.method != "test":

            def get_eval_dataloader(self, eval_dataset=None):
                return balanced_dataloader(self, "eval", cfg.eval_batch_size)

    def compute_metrics(p):
        labels = p.label_ids
        predictions = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        predictions = sigmoid(predictions)

        best_threshold, best_f1 = 0, 0
        for threshold in np.arange(0.3, 0.7, 0.05):
            binary_predictions = predictions > threshold

            f1 = f1_score(labels, binary_predictions, average="micro")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        binary_predictions = predictions > best_threshold

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_predictions, average="micro"
        )
        accuracy = accuracy_score(labels, binary_predictions)
        pr_auc = average_precision_score(labels, predictions, average="micro")

        metrics = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "pr_auc": pr_auc,
            "threshold": best_threshold,
        }

        if cfg.method == "test":

            cl_report_dict = classification_report(
                labels,
                binary_predictions,
                target_names=label_scheme,
                digits=4,
                output_dict=True,
            )
            metrics["label_scores"] = {
                key: val for key, val in cl_report_dict.items() if key in label_scheme
            }

            true_labels_str = decode_binary_labels(labels, cfg.labels)
            predicted_labels_str = decode_binary_labels(binary_predictions, cfg.labels)

            data = list(zip(true_labels_str, predicted_labels_str))

            os.makedirs(results_output_dir, exist_ok=True)

            with open(
                f"{results_output_dir}/predictions_{test_language}.tsv", "w", newline=""
            ) as csvfile:
                csv_writer = csv.writer(csvfile, delimiter="\t")
                csv_writer.writerows(data)

            with open(f"{results_output_dir}/metrics_{test_language}.json", "w") as f:
                json.dump(metrics, f)

            print(metrics)

        return metrics

    base_model_path = (
        model_output_dir if cfg.method != "train" and not cfg.peft else cfg.model_name
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=len(label_scheme),
        torch_dtype=torch.bfloat16 if not cfg.no_bf16 else torch.float,
        use_flash_attention_2=cfg.fa2,
        load_in_4bit=cfg.4bit
    )

    if cfg.peft:
        if cfg.method == "train":
            print("Using LoRa")
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules=get_linear_modules(model),
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                ),
            )
        else:
            model = PeftModel.from_pretrained(model, model_output_dir)

    trainer = MultiLabelTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=30,
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            warmup_ratio=0.05,
            weight_decay=0.01,
            learning_rate=cfg.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            gradient_accumulation_steps=cfg.grad_acc_steps,
            eval_accumulation_steps=8,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            save_total_limit=2,
            tf32=True,
            group_by_length=True,
        ),
        train_dataset=dataset.get("train", []),
        eval_dataset=dataset.get("dev", []),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if cfg.method == "train":
        trainer.train()
        for dir_path in glob.glob(f"{model_output_dir}/checkpoint*"):
            shutil.rmtree(dir_path, ignore_errors=True)
        trainer.save_model()
        shutil.rmtree(f"{model_output_dir}/runs", ignore_errors=True)

    print("Predicting..")
    cfg.method = "test"

    for language in cfg.test.split("-"):
        print(f"-- {language} --")
        test_language = language
        trainer.predict(
            dataset["test"].filter(lambda example: example["language"] == language)
        )
