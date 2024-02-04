from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset

model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"


def setfit_train(dataset):

    train_dataset = sample_dataset(
        dataset["train"], label_column="label_text", num_samples=1
    )
    dev_dataset = sample_dataset(
        dataset["dev"], label_column="label_text", num_samples=1
    )
    test_dataset = sample_dataset(
        dataset["test"], label_column="label_text", num_samples=1
    )

    model = SetFitModel.from_pretrained(model_id, multi_target_strategy="multi-output")

    args = TrainingArguments(
        batch_size=64,
        num_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=args,
        metric="accuracy",
        column_mapping={"text": "text", "labels": "label"},
    )

    trainer.train()

    metrics = trainer.evaluate(test_dataset)

    print(metrics)
