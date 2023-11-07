from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import sys
from pprint import PrettyPrinter

import numpy as np

import transformers
import datasets
import torch

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

logging.disable(logging.INFO)
pprint = PrettyPrinter(compact=True).pprint

LEARNING_RATE = 1e-5
BATCH_SIZE = 8
TRAIN_EPOCHS = 15
MODEL_NAME = "xlm-roberta-base"
PATIENCE = 5
WORKING_DIR = "/scratch/project_2005092/register-models"

labels_full = [
    "HI",
    "ID",
    "IN",
    "IP",
    "LY",
    "MT",
    "NA",
    "OP",
    "SP",
    "av",
    "ds",
    "dtp",
    "ed",
    "en",
    "fi",
    "it",
    "lt",
    "nb",
    "ne",
    "ob",
    "ra",
    "re",
    "rs",
    "rv",
    "sr",
]
labels_upper = ["HI", "ID", "IN", "IP", "LY", "MT", "NA", "OP", "SP"]


# Register scheme mapping
sub_register_map = {
    "NA": "NA",
    "NE": "ne",
    "SR": "sr",
    "PB": "nb",
    "HA": "NA",
    "FC": "NA",
    "TB": "nb",
    "CB": "nb",
    "OA": "NA",
    "OP": "OP",
    "OB": "ob",
    "RV": "rv",
    "RS": "rs",
    "AV": "av",
    "IN": "IN",
    "JD": "IN",
    "FA": "fi",
    "DT": "dtp",
    "IB": "IN",
    "DP": "dtp",
    "RA": "ra",
    "LT": "lt",
    "CM": "IN",
    "EN": "en",
    "RP": "IN",
    "ID": "ID",
    "DF": "ID",
    "QA": "ID",
    "HI": "HI",
    "RE": "re",
    "IP": "IP",
    "DS": "ds",
    "EB": "ed",
    "ED": "ed",
    "LY": "LY",
    "PO": "LY",
    "SO": "LY",
    "SP": "SP",
    "IT": "it",
    "FS": "SP",
    "TV": "SP",
    "OS": "OS",
    "IG": "IP",
    "MT": "MT",
    "HT": "HI",
    "FI": "fi",
    "OI": "IN",
    "TR": "IN",
    "AD": "OP",
    "LE": "OP",
    "OO": "OP",
    "MA": "NA",
    "ON": "NA",
    "SS": "NA",
    "OE": "IP",
    "PA": "IP",
    "OF": "ID",
    "RR": "ID",
    "FH": "HI",
    "OH": "HI",
    "TS": "HI",
    "OL": "LY",
    "PR": "LY",
    "SL": "LY",
    "TA": "SP",
    "OTHER": "OS",
    "": "",
}

# Only train and test for these languages
small_languages = [
    "ar",
    "ca",
    "es",
    "fa",
    "hi",
    "id",
    "jp",
    "no",
    "pt",
    "tr",
    "ur",
    "zh",
]

# Data column structures
cols = {
    "fr": ["a", "b", "label", "text", "c"],
    "fi": ["label", "text", "a", "b", "c"],
    "sv": ["a", "b", "label", "text", "c"],
}


def argparser():
    ap = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model_name", default=MODEL_NAME, help="Pretrained model name")
    ap.add_argument("--train", required=True, help="Path to training data")
    ap.add_argument("--test", required=True, help="Path to test data")
    ap.add_argument(
        "--batch_size",
        metavar="INT",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for training",
    )
    ap.add_argument(
        "--epochs",
        metavar="INT",
        type=int,
        default=TRAIN_EPOCHS,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--learning_rate",
        metavar="FLOAT",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate",
    )
    ap.add_argument(
        "--patience",
        metavar="INT",
        type=int,
        default=PATIENCE,
        help="Early stopping patience",
    )
    ap.add_argument("--save_model", default=True, type=bool, help="Save model to file")
    ap.add_argument(
        "--threshold",
        default=None,
        metavar="FLOAT",
        type=float,
        help="threshold for calculating f-score",
    )
    ap.add_argument("--labels", choices=["full", "upper"], default="full")
    ap.add_argument(
        "--load_model", default=None, metavar="FILE", help="Load existing model"
    )
    ap.add_argument("--class_weights", default=False, type=bool)
    ap.add_argument("--working_dir", default=WORKING_DIR, help="Working directory")
    ap.add_argument("--tune", default=False, type=bool, help="Tune hyperparameters")

    return ap


options = argparser().parse_args(sys.argv[1:])
working_dir = f"{options.working_dir}/{options.train}_{options.test}"
model_name = options.model_name

labels = labels_full if options.labels == "full" else labels_upper
num_labels = len(labels)
print(f"Number of labels: {num_labels}")


# Data preprocessing
def preprocess_data(example):
    text = example["text"] or ""
    encoding = tokenizer(text, padding=True, truncation=True, max_length=512)
    mapped_labels = set(
        [
            sub_register_map[l] if l not in labels else l
            for l in (example["label"] or "NA").split()
        ]
    )
    encoding["label"] = [1 if l in mapped_labels else 0 for l in labels]
    return encoding


def get_data():
    data_files = {"train": [], "dev": [], "test": []}

    for l in options.train.split("-"):
        data_files["train"].append(f"data/{l}/train.tsv")
        if not (l in small_languages):
            data_files["dev"].append(f"data/{l}/dev.tsv")
        else:
            # Small languages use test as dev
            data_files["dev"].append(f"data/{l}/test.tsv")

    for l in options.test.split("-"):
        # check if zero-shot for small languages, if yes then test with full data
        if l in small_languages and not (l in options.train.split("-")):
            data_files["test"].append(f"data/{l}/{l}.tsv")
        else:
            data_files["test"].append(f"data/{l}/test.tsv")

    return data_files


data_files = get_data()

print("data files", data_files)

dataset = datasets.load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t",
    column_names=cols.get(options.train, ["label", "text"]),
    features=datasets.Features(
        {
            "text": datasets.Value("string"),
            "label": datasets.Value("string"),
        }
    ),
    cache_dir=f"{working_dir}/dataset_cache",
)
dataset = dataset.shuffle(seed=42)


tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
dataset = dataset.map(preprocess_data)

print("dataset pre-processed")

# Evaluate only
if options.load_model is not None:
    model = torch.load(options.load_model)
    trues = dataset["test"]["label"]
    inputs = dataset["test"]["text"]
    pred_labels = []
    for index, i in enumerate(inputs):
        tok = tokenizer(i, truncation=True, max_length=512, return_tensors="pt")
        pred = model(**tok)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(pred.logits.detach().numpy()))
        preds = np.zeros(probs.shape)
        preds[np.where(probs >= options.threshold)] = 1
        pred_labels.extend(preds)
    print("F1-score", f1_score(y_true=trues, y_pred=pred_labels, average="micro"))
    print(classification_report(trues, pred_labels, target_names=labels))
    exit()


def compute_class_weights(dataset):
    y = [
        i
        for example in dataset["train"]
        for i, val in enumerate(example["label"])
        if val
    ]

    weights = len(dataset["train"]) / (len(labels) * np.bincount(y))

    class_weights = torch.FloatTensor(weights)
    print(f"class weights: {class_weights}")
    return y


def model_init():
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=f"{working_dir}/model_cache",
    )


if options.class_weights is True:
    print("Using class weights")
    class_weights = compute_class_weights(dataset)


class MultilabelTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if options.class_weights:
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


trainer_args = transformers.TrainingArguments(
    f"{working_dir}/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=True,
    eval_steps=100,
    logging_steps=100,
    learning_rate=options.learning_rate,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    per_device_train_batch_size=options.batch_size,
    per_device_eval_batch_size=32,
    num_train_epochs=options.epochs,
    report_to="wandb" if options.tune else None,
)


# in case a threshold was not given, choose the one that works best with the evaluated data
def optimize_threshold(predictions, labels):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    best_f1 = 0
    best_f1_threshold = 0.5  # use 0.5 as a default threshold
    y_true = labels
    for th in np.arange(0.3, 0.7, 0.05):
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= th)] = 1
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = th
    return best_f1_threshold


def multi_label_metrics(predictions, labels, threshold):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_th05 = np.zeros(probs.shape)
    y_th05[np.where(probs >= 0.5)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {
        "f1": f1_micro_average,
        "f1_th05": f1_score(y_true=y_true, y_pred=y_th05, average="micro"),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "threshold": threshold,
    }
    return metrics


def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    if options.threshold == None:
        best_f1_th = optimize_threshold(preds, p.label_ids)
        threshold = best_f1_th
    result = multi_label_metrics(
        predictions=preds, labels=p.label_ids, threshold=threshold
    )
    return result


trainer = MultilabelTrainer(
    model=None,
    model_init=model_init,
    args=trainer_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[
        transformers.EarlyStoppingCallback(early_stopping_patience=options.patience)
    ],
)


if options.tune:
    from ray import tune

    asha_scheduler = tune.schedulers.ASHAScheduler(
        metric="eval_f1",
        mode="max",
    )

    tune_config = {
        "learning_rate": tune.grid_search([5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]),
        "per_device_train_batch_size": tune.grid_search([6, 8, 12, 16]),
    }

    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        scheduler=asha_scheduler,
        direction="maximize",
    )
else:
    print("Training...")
    trainer.train()

print("Evaluating with test set...")
eval_results = trainer.evaluate(dataset["test"])

pprint(eval_results)

test_pred = trainer.predict(dataset["test"])
trues = test_pred.label_ids
predictions = test_pred.predictions

if options.threshold == None:
    threshold = optimize_threshold(predictions, trues)
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(predictions))

preds = np.zeros(probs.shape)
preds[np.where(probs >= threshold)] = 1

print(classification_report(trues, preds, target_names=labels))

if options.save_model:
    torch.save(
        trainer.model,
        f"{working_dir}/saved_model",
    )
