from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from datasets import Dataset
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"


def few_shot(dataset, num):
    def sample_group(group, random_state=42):
        n = min(len(group), num)
        return group.sample(n, random_state=random_state)

    pd_dataset = (
        pd.DataFrame(dataset)
        .groupby(["language", "label_text"])
        .apply(sample_group)
        .reset_index(drop=True)
    )

    print(pd_dataset)

    dataset = Dataset.from_pandas(pd_dataset)

    return dataset


def setfit_train(dataset, label_scheme):

    def compute_metrics(y_pred, labels):
        try:
            roc_auc = roc_auc_score(labels, y_pred, average="micro")
        except:
            roc_auc = 0

        precision = precision_score(labels, y_pred, average="micro")
        recall = recall_score(labels, y_pred, average="micro")

        try:
            pr_auc = average_precision_score(labels, y_pred, average="micro")
        except:
            pr_auc = 0
        accuracy = accuracy_score(labels, y_pred)
        print(
            classification_report(labels, y_pred, target_names=label_scheme, digits=4)
        )
        return {
            f"f1": f1_score(y_true=labels, y_pred=y_pred, average="micro"),
            f"precision": precision,
            f"recall": recall,
            f"pr_auc": pr_auc,
            f"roc_auc": roc_auc,
            f"accuracy": accuracy,
        }

    train_dataset = dataset["train"].rename_column("labels", "label")
    dev_dataset = dataset["dev"].rename_column("labels", "label")
    test_dataset = dataset["test"].rename_column("labels", "label")

    train_dataset = few_shot(train_dataset, 8)
    dev_dataset = dev_dataset.select(range(100))
    test_dataset = test_dataset

    model = SetFitModel.from_pretrained(model_id, multi_target_strategy="multi-output")

    args = TrainingArguments(
        batch_size=16,
        evaluation_strategy="steps",
        eval_steps=500,
        num_epochs=1,
        save_strategy="epoch",
        # load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=args,
        metric="accuracy",
        column_mapping={"text": "text", "label": "label"},
    )

    trainer.train()

    metrics = trainer.evaluate(test_dataset)

    print(metrics)
