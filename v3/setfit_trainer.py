from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer, TrainingArguments

model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"


def setfit_train(dataset):
    model = SetFitModel.from_pretrained(model_id, multi_target_strategy="one-vs-rest")

    args = TrainingArguments(
        batch_size=16,
        num_epochs=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        args=args,
        metric="accuracy",
        column_mapping={"text": "text", "labels": "label"},
    )

    trainer.train()

    metrics = trainer.evaluate()

    print(metrics)
