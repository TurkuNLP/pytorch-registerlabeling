from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

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

    train_dataset = sample_dataset(
        dataset["train"], label_column="label_text", num_samples=8
    )
    dev_dataset = sample_dataset(
        dataset["dev"], label_column="label_text", num_samples=100
    )

    test_dataset = sample_dataset(
        dataset["dev"], label_column="label_text", num_samples=100
    )

    model = SetFitModel.from_pretrained(model_id, multi_target_strategy="multi-output")

    args = TrainingArguments(
        batch_size=64,
        evaluation_strategy="epoch",
        # eval_steps=1,
        # max_steps=1,
        num_epochs=1,
        # evaluation_strategy="steps",
        save_strategy="epoch",
        # load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=args,
        metric=compute_metrics,
        column_mapping={"text": "text", "labels": "label"},
    )

    trainer.train()

    metrics = trainer.evaluate(test_dataset)

    print(metrics)
