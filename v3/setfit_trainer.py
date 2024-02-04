from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, Trainer

model_id = "sentence-transformers/distiluse-base-multilingual-cased-v2"


def setfit_train(dataset):
    model = SetFitModel.from_pretrained(model_id, multi_target_strategy="one-vs-rest")

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        loss_class=CosineSimilarityLoss,
        num_iterations=20,
        column_mapping={"text": "text", "labels": "label"},
    )

    trainer.train()

    metrics = trainer.evaluate()

    print(metrics)
