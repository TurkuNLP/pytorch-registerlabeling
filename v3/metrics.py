import torch
import torch.nn
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    classification_report,
)

sigmoid = torch.nn.Sigmoid()


def optimize_threshold(logits, labels):
    probs = sigmoid(logits)
    best_f1 = 0
    best_f1_threshold = 0.5
    for th in np.arange(0.3, 0.7, 0.05):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= th)] = 1
        f1 = f1_score(y_true=labels, y_pred=y_pred, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th

    return best_f1_threshold


def compute_metrics(logits, labels, label_scheme=None):
    threshold = optimize_threshold(logits, labels)
    probs = sigmoid(logits)
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

    if label_scheme:
        print(
            classification_report(labels, y_pred, target_names=label_scheme, digits=4)
        )

    return metrics
