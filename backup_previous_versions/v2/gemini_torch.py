import torch

# from torch.utils.data import DataLoader, Dataset
from datasets import Dataset
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import pandas as pd
import ast
from torch.nn import BCEWithLogitsLoss
from torch import Tensor
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
)

import torch.nn as nn
import torch.nn.functional as F

from .model import GeminiModel
from .loss import BCEFocalLoss
from .data import get_gemini_data


def run(options):
    dataset = get_gemini_data(options.language)
    model = GeminiModel()

    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss_fct = BCEFocalLoss()

            loss = loss_fct(
                logits.view(-1, self.model.num_labels),
                labels.float().view(-1, self.model.num_labels),
            )

            return (loss, outputs) if return_outputs else loss

    def optimize_threshold(predictions, labels):
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(Tensor(predictions))
        best_f1 = 0
        best_f1_threshold = 0.5
        y_true = labels
        for th in np.arange(0.3, 0.7, 0.05):
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= th)] = 1
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = th

        return best_f1_threshold

    # Calculate metrics

    def compute_metrics_fn(labels, predictions, return_preds=False):
        threshold = optimize_threshold(predictions, labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_th05 = np.zeros(probs.shape)
        y_th05[np.where(probs >= 0.5)] = 1
        try:
            roc_auc = roc_auc_score(labels, y_pred, average="micro")
        except:
            roc_auc = 0

        # Compute precision and recall
        precision = precision_score(labels, y_pred, average="micro")
        recall = recall_score(labels, y_pred, average="micro")

        # Compute PR AUC
        try:
            pr_auc = average_precision_score(labels, probs, average="micro")
        except:
            pr_auc = 0
        accuracy = accuracy_score(labels, y_pred)
        metrics = {
            "f1": f1_score(y_true=labels, y_pred=y_pred, average="micro"),
            "f1_th05": f1_score(y_true=labels, y_pred=y_th05, average="micro"),
            "precision": precision,
            "recall": recall,
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "threshold": threshold,
        }
        if not return_preds:
            return metrics
        else:
            return metrics, y_pred

    def compute_metrics(p):
        predictions, labels = p

        return compute_metrics_fn(labels, predictions)

    trainer = CustomTrainer(
        model=model,
        args=TrainingArguments(
            output_dir="./results",
            num_train_epochs=30,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.05,
            warmup_steps=0,
            warmup_ratio=0.01,
            save_total_limit=2,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=0.001,
            greater_is_better=True,
            metric_for_best_model="eval_f1",
            load_best_model_at_end=True,
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Train the model
    trainer.train()

    print("Evaluating with test set...")
    p = trainer.predict(dataset["test"])
    predictions = (
        p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    )
    metrics, preds = compute_metrics_fn(p.label_ids, predictions, return_preds=True)
    print(metrics)
