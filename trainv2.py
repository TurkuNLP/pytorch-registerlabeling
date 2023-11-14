import os

os.environ["TRANSFORMERS_CACHE"] = ".hf/transformers_cache"
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
from dotenv import load_dotenv

from pydoc import locate
from argparse import ArgumentParser
import re
from ray.tune.schedulers import ASHAScheduler
from ray.tune import grid_search, CLIReporter, loguniform, choice
from ray.tune.search.hyperopt import HyperOptSearch
import ray

from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

ray.init(ignore_reinit_error=True, num_cpus=1)
import numpy as np
from transformers import (
    AutoTokenizer,
    Trainer,
    AutoModelForSequenceClassification,
    MistralForSequenceClassification,
    LlamaForSequenceClassification,
    MistralForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset, Features, Value
from torch.nn import BCEWithLogitsLoss, Sigmoid, Linear
from torch import Tensor, FloatTensor, bfloat16

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_int8_training



# Get CLI options

parser = ArgumentParser()

# Model and data

parser.add_argument("--model_name", type=str, default="xlm-roberta-base")
parser.add_argument("--labels", choices=["full", "upper"], default="full")
parser.add_argument("--custom_tokenizer", type=str, default=None)
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument(
    "--output_path", type=str, default="/scratch/project_2005092/register-models"
)
parser.add_argument(
    "--transformer_model", type=str, default="AutoModelForSequenceClassification"
)
parser.add_argument("--seed", type=str, default=42)
parser.add_argument("--hp_search", action="store_true")
parser.add_argument("--evaluate_only", action="store_true")

# Training arguments

parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=15)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--warmup_ratio", type=float, default=0)
parser.add_argument("--metric_for_best_model", type=str, default="loss")
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--iter_strategy", type=str, default="epoch")
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--optimizer", type=str, default="adamw_torch")
parser.add_argument("--lr_scheduler_type", type=str, default="linear")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--report_to", type=str, default="wandb")
parser.add_argument("--class_weights", action="store_true")
parser.add_argument("--threshold", type=float, default=None)

# (Q)lora / peft related options

parser.add_argument("--add_prefix_space", action="store_true")
parser.add_argument("--use_flash_attention_2", action="store_true")
parser.add_argument("--add_classification_head", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--peft", action="store_true")
parser.add_argument("--peft_modules", type=str, default=None)
parser.add_argument("--set_pad_id", action="store_true")
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=float, default=1)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_bias", type=str, default="none")

options = parser.parse_args()

print(f"Settings: {options}")


# Register config


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

# Common variables

labels = labels_full if options.labels == "full" else labels_upper
model_name = options.model_name
working_dir = f"{options.output_path}/{options.train}_{options.test}{'_tuning' if options.hp_search else ''}/{model_name.replace('/', '_')}"
peft_modules = options.peft_modules.split(",") if options.peft_modules else None

# Wandb setup

if options.report_to == "wandb":
    print("Using wandb")
    os.environ[
        "WANDB_PROJECT"
    ] = f"register-labeling_{options.train}_{options.test}{'_tuning' if options.hp_search else ''}_{model_name.replace('/', '_')}"

    load_dotenv()
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
    import wandb
    wandb.login()


# Data preprocessing


def preprocess_data(example):
    text = example["text"] or ""
    encoding = tokenizer(
        text, padding="max_length", truncation=True, max_length=options.max_length
    )
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


print("Getting data...")

data_files = get_data()

print("data files", data_files)

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t",
    column_names=cols.get(options.train, ["label", "text"]),
    features=Features(
        {
            "text": Value("string"),
            "label": Value("string"),
        }
    ),
    cache_dir=f"{working_dir}/dataset_cache",
)
dataset = dataset.shuffle(seed=options.seed)


tokenizer = AutoTokenizer.from_pretrained(
    model_name if not options.custom_tokenizer else options.custom_tokenizer,
    add_prefix_space=options.add_prefix_space,
)

if options.set_pad_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

print("Preprocessing...")

dataset = dataset.map(preprocess_data)

print("Got preprocessed dataset and tokenizer")


# Start modeling

if options.class_weights:
    y = [
        i
        for example in dataset["train"]
        for i, val in enumerate(example["label"])
        if val
    ]

    weights = len(dataset["train"]) / (len(labels) * np.bincount(y))
    class_weights = FloatTensor(weights)

    print(f"using class weights: {class_weights}")


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if options.class_weights:
            loss_fct = BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


def optimize_threshold(predictions, labels):
    sigmoid = Sigmoid()
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


def compute_metrics(p):
    _, labels = p
    predictions = (
        p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    )
    threshold = (
        options.threshold
        if options.threshold
        else optimize_threshold(predictions, labels)
    )
    sigmoid = Sigmoid()
    probs = sigmoid(Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_th05 = np.zeros(probs.shape)
    y_th05[np.where(probs >= 0.5)] = 1
    roc_auc = roc_auc_score(labels, y_pred, average="micro")
    accuracy = accuracy_score(labels, y_pred)
    metrics = {
        "f1": f1_score(y_true=labels, y_pred=y_pred, average="micro"),
        "f1_th05": f1_score(y_true=labels, y_pred=y_th05, average="micro"),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "threshold": threshold,
    }
    return metrics


def model_init():
    if options.quantize:
        print("Using quantization")
    model_type = locate(f"transformers.{options.transformer_model}")
    model = model_type.from_pretrained(
        model_name,
        num_labels=len(labels),
        cache_dir=f"{working_dir}/model_cache",
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_flash_attention_2=options.use_flash_attention_2,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=bfloat16,
        )
        if options.quantize
        else None,
    )

    if options.set_pad_id:
        model.config.pad_token_id = model.config.eos_token_id

    if options.peft:
        print("Using PEFT")
        # Get module names

        model.config.pretraining_tp = 1  # Set max linear layers

        model_modules = str(model.modules)
        pattern = r"\((\w+)\): Linear"
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        # Print the names of the Linear layers
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))

        # Define LoRA Config
        lora_config = LoraConfig(
            r=options.lora_rank,
            lora_alpha=options.lora_alpha,
            target_modules=target_modules if not peft_modules else peft_modules,
            lora_dropout=options.lora_dropout,
            bias=options.lora_bias,
            task_type=TaskType.SEQ_CLS,
            # inference_mode=True,
        )

        # add LoRA adaptor
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model = prepare_model_for_int8_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if options.add_classification_head:
        # Add a classification head on top of the model
        model.resize_token_embeddings(len(tokenizer))
        model.classifier = Linear(model.config.hidden_size, len(labels))

    return model


trainer = MultilabelTrainer(
    model=None,
    model_init=model_init,
    args=TrainingArguments(
        f"{working_dir}/checkpoints",
        overwrite_output_dir=True if options.overwrite else False,
        evaluation_strategy=options.iter_strategy,
        save_strategy=options.iter_strategy,
        logging_strategy=options.iter_strategy,
        load_best_model_at_end=True,
        eval_steps=options.eval_steps,
        logging_steps=options.logging_steps,
        save_steps=options.save_steps,
        weight_decay=options.weight_decay,
        warmup_steps=options.warmup_steps,
        warmup_ratio=options.warmup_ratio,
        learning_rate=options.lr,
        lr_scheduler_type=options.lr_scheduler_type,
        metric_for_best_model=options.metric_for_best_model,
        greater_is_better=False if "loss" in options.metric_for_best_model else True,
        per_device_train_batch_size=options.train_batch_size,
        per_device_eval_batch_size=options.eval_batch_size,
        num_train_epochs=options.epochs,
        gradient_checkpointing=True,
        gradient_accumulation_steps=options.gradient_steps,
        report_to=options.report_to,
        optim=options.optimizer,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=options.patience)],
)

if not options.evaluate_only:
    if not options.hp_search:
        print("Training...")
        trainer.train()

    else:
        asha_scheduler = ASHAScheduler(
            metric="eval_f1",
            mode="max",
        )

        tune_config = {
            # "learning_rate": grid_search([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]),
            # "per_device_train_batch_size": grid_search([6, 8, 12]),
            "learning_rate": loguniform(1e-6, 1e-3),
            "learning_rate": choice([6, 8, 12, 16]),
            # "per_device_train_batch_size": grid_search([6, 8, 12]),
        }

        reporter = CLIReporter(
            parameter_columns={
                "learning_rate": "learning_rate",
                "per_device_train_batch_size": "train_bs/gpu",
                "num_train_epochs": "num_epochs",
            },
            metric_columns=[
                "eval_f1",
                "eval_f1_th05",
                "eval_threshold",
                "training_iteration",
            ],
        )

        best_model = trainer.hyperparameter_search(
            hp_space=lambda _: tune_config,
            backend="ray",
            # scheduler=asha_scheduler,
            scheduler=ASHAScheduler(metric="eval_f1", mode="max"),
            search_alg=HyperOptSearch(metric="eval_f1", mode="max"),
            progress_reporter=reporter,
            direction="maximize",
            local_dir=f"{working_dir}/ray",
            log_to_file=True,
        )
        print("Best model according to Ray:")
        print(best_model)

print("Evaluating with dev set...")
print(trainer.evaluate(dataset["dev"]))

print("Evaluating with test set...")
print(trainer.evaluate(dataset["test"]))

test_pred = trainer.predict(dataset["test"])
trues = test_pred.label_ids
predictions = test_pred.predictions
threshold = (
    options.threshold if options.threshold else optimize_threshold(predictions, trues)
)
sigmoid = Sigmoid()
probs = sigmoid(Tensor(predictions))
preds = np.zeros(probs.shape)
preds[np.where(probs >= threshold)] = 1

print(classification_report(trues, preds, target_names=labels))

if not options.evaluate_only and options.save_model:
    trainer.model.save_pretrained(f"{working_dir}/saved_model")
