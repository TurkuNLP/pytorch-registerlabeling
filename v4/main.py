import numpy as np
import random
import torch
import torch.nn.functional as F
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
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

from .data import get_dataset
from .dataloader import balanced_dataloader
from .labels import label_schemes


def run(cfg):

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    labels = label_schemes[cfg.labels]
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

        if len(cfg.train.split("-")) > 1:

            def get_train_dataloader(self):
                return balanced_dataloader(
                    self, "train", self.cfg.dataloader.dev_batch_size
                )

        if len(cfg.dev.split("-")) > 1:

            def get_eval_dataloader(self, eval_dataset=None):
                return balanced_dataloader(
                    self, "eval", self.cfg.dataloader.dev_batch_size
                )

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

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "best_threshold": best_threshold,
        }

    trainer = MultiLabelTrainer(
        model=AutoModelForSequenceClassification.from_pretrained(
            cfg.model, num_labels=len(labels), torch_dtype=torch.float16
        ),
        args=TrainingArguments(
            output_dir="./hf_results",
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
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            save_total_limit=2,
        ),
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    print("Evaluating on dev set...")
    print(trainer.evaluate(dataset["dev"]))
    print("Evaluating on test set...")
    print(trainer.evaluate(dataset["test"]))
