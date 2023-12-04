from datetime import datetime

_print = print


# Print with datetime
def print(*args, **kw):
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _print(f"[{formatted_now}]", *args, **kw)


import os

os.environ["TRANSFORMERS_CACHE"] = ".hf/transformers_cache"
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"

from argparse import ArgumentParser

# Get CLI options

parser = ArgumentParser()

# Mode of operation

parser.add_argument(
    "--mode",
    choices=["train", "evaluate", "extract_docembeddings", "stats"],
    default="train",
)

# Model and data

parser.add_argument("--train", required=True)
parser.add_argument("--test")
parser.add_argument("--model_name", default="xlm-roberta-base")
parser.add_argument("--custom_tokenizer")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--data_path", default="data")
parser.add_argument("--output_path", default="output")
parser.add_argument("--transformer_model", default="AutoModelForSequenceClassification")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--low_cpu_mem_usage", action="store_true")
parser.add_argument("--torch_dtype")
parser.add_argument("--log_to_file", action="store_true")
parser.add_argument("--labels", default="all")
parser.add_argument("--downsample", action="store_true")
parser.add_argument("--hp_search")

# Loss

parser.add_argument("--loss")
parser.add_argument("--loss_alpha", type=float, default=1.0)
parser.add_argument("--loss_gamma", type=float, default=1.0)

# Tokenizer

parser.add_argument("--return_tensors")

# Training arguments

parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=8)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--warmup_steps", type=int, default=0)
parser.add_argument("--warmup_ratio", type=float, default=0)
parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gradient_steps", type=int, default=1)
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--iter_strategy", type=str, default="epoch")
parser.add_argument("--eval_steps", type=int, default=100)
parser.add_argument("--logging_steps", type=int, default=100)
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--save_model", action="store_true")
parser.add_argument("--save_total_limit", type=int, default=2)
parser.add_argument("--optim", default="adamw_torch")
parser.add_argument("--lr_scheduler_type", default="linear")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--max_grad_norm", type=float, default=1)
parser.add_argument("--class_weights", action="store_true")
parser.add_argument("--threshold", type=float, default=None)
parser.add_argument("--device_map", default="auto")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--resume", action="store_true")

# (Q)lora / peft related options

parser.add_argument("--add_prefix_space", action="store_true")
parser.add_argument("--use_flash_attention_2", action="store_true")
parser.add_argument("--add_classification_head", action="store_true")
parser.add_argument("--quantize", action="store_true")
parser.add_argument("--peft", action="store_true")
parser.add_argument("--peft_modules")
parser.add_argument("--set_pad_id", action="store_true")
parser.add_argument("--lora_rank", type=int, default=16)
parser.add_argument("--lora_alpha", type=float, default=1)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--lora_bias", default="none")

options = parser.parse_args()

import csv
import sys

if options.log_to_file:
    file_name_opts = [
        datetime.now(),
        options.train,
        options.test,
        options.model_name.replace("/", "_"),
        options.lr,
        options.train_batch_size,
        options.hp_search,
    ]
    file_name = "_".join(map(str, filter(None, file_name_opts)))
    log = open(
        f"logs/{file_name}",
        "a",
    )
    sys.stdout = log

print(f"Args: {' '.join(sys.argv)}")
print(f"Settings: {options}")

from pydoc import locate
import re

from tqdm import tqdm

import numpy as np

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)

from torch.nn import BCEWithLogitsLoss, Sigmoid, Linear
from torch import Tensor, FloatTensor, cuda, float16, float32, bfloat16
from accelerate import Accelerator

from labels import get_label_scheme
from resources import get_dataset, get_statistics

# Common variables

options.test = options.train if not options.test else options.test
model_name = options.model_name
working_dir = f"{options.output_path}/{options.train}_{options.test}{'_'+options.hp_search if options.hp_search else ''}/{model_name.replace('/', '_')}"
peft_modules = options.peft_modules.split(",") if options.peft_modules else None
accelerator = Accelerator()

# Labels
label_scheme = get_label_scheme(options.labels)

print(f"Predicting {len(label_scheme)} labels")

# Torch dtypes

torch_dtype_map = {
    None: None,
    "torch.float16": float16,
    "torch.float32": float32,
    "torch.bfloat16": bfloat16,
    "auto": "auto",
}

torch_dtype = torch_dtype_map[options.torch_dtype]


def log_gpu_memory():
    for gpu in range(cuda.device_count()):
        allocated_memory = cuda.memory_allocated(gpu) / (1024**3)  # Convert to GB
        max_allocated_memory = cuda.max_memory_allocated(gpu) / (1024**3)
        print(
            f"GPU {gpu}: Current Memory Allocated: {allocated_memory:.2f} GB, Max Memory Allocated: {max_allocated_memory:.2f} GB"
        )


# Imports based on options

if options.use_flash_attention_2:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func

if options.hp_search:
    if options.hp_search == "ray":
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.schedulers import PopulationBasedTraining
        from ray.tune import grid_search, CLIReporter, loguniform, choice, uniform
        from ray.tune.search.hyperopt import HyperOptSearch
        from ray import init as ray_init

if options.peft:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
    )

# Wandb setup

try:
    from dotenv import load_dotenv

    load_dotenv()
    wandb_project_name = f"{options.train}_{options.test}{'_tuning' if options.hp_search else ''}_{model_name.replace('/', '_')}"

    os.environ["WANDB_PROJECT"] = wandb_project_name
    os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
    os.environ["WANDB_WATCH"] = "all"

    import wandb

    wandb.login()
    print("Using wandb")
except:
    print("No wandb!")
    pass

print(f"Imports finished")

# Data processing

# Init tokenizer

tokenizer = AutoTokenizer.from_pretrained(
    model_name if not options.custom_tokenizer else options.custom_tokenizer,
    add_prefix_space=options.add_prefix_space,
    low_cpu_mem_usage=options.low_cpu_mem_usage,
    torch_dtype=torch_dtype,
    cache_dir=f"{working_dir}/tokenizer_cache",
)

# Some LLM's require a pad id
if options.set_pad_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token


def encode_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        return_tensors=options.return_tensors,
    )


# Get data

dataset = get_dataset(options.train, options.test, options.downsample, options.labels)

# If stats mode, stop here

if options.mode == "stats":
    get_statistics(dataset)
    exit()

# Shuffle data and tokenize

dataset = dataset.shuffle(seed=options.seed)
dataset = dataset.map(encode_data, batched=True)

print(dataset)

print("Data tokenized!")


# Model

if options.class_weights:
    y = [
        i
        for example in dataset["train"]
        for i, val in enumerate(example["label"])
        if val
    ]

    weights = len(dataset["train"]) / (len(label_scheme) * np.bincount(y))
    class_weights = FloatTensor(weights).to("cuda")

    print(f"Using class weights: {class_weights}")


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if options.loss:
            loss_cls = locate(f"loss.{options.loss}")

            loss_fct = loss_cls(alpha=options.loss_alpha, gamma=options.loss_gamma)
        else:
            loss_fct = BCEWithLogitsLoss(
                pos_weight=class_weights if options.class_weights else None
            )
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
    try:
        roc_auc = roc_auc_score(labels, y_pred, average="micro")
    except:
        roc_auc = 0
    accuracy = accuracy_score(labels, y_pred)
    metrics = {
        "f1": f1_score(y_true=labels, y_pred=y_pred, average="micro"),
        "f1_th05": f1_score(y_true=labels, y_pred=y_th05, average="micro"),
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "threshold": threshold,
    }
    return metrics


# Model initialization, used by trainer
def model_init():
    model_type = locate(f"transformers.{options.transformer_model}")
    model = model_type.from_pretrained(
        model_name,
        num_labels=len(label_scheme),
        cache_dir=f"{options.output_path}/model_cache",
        trust_remote_code=True,
        device_map=options.device_map or None,
        offload_folder="offload",
        low_cpu_mem_usage=True,
        use_flash_attention_2=options.use_flash_attention_2,
        torch_dtype=torch_dtype,
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

        # model.config.pretraining_tp = 1  # Set max linear layers (llama2)

        model_modules = str(model.modules)
        pattern = r"\((\w+)\): Linear"
        linear_layer_names = re.findall(pattern, model_modules)

        names = []
        # Print the names of the Linear layers
        for name in linear_layer_names:
            names.append(name)
        target_modules = list(set(names))

        print(f"Linear layers:\n {target_modules}")

        # Define LoRA Config
        lora_config = LoraConfig(
            r=options.lora_rank,
            lora_alpha=options.lora_alpha,
            target_modules=target_modules if not peft_modules else peft_modules,
            lora_dropout=options.lora_dropout,
            bias=options.lora_bias,
            task_type=TaskType.SEQ_CLS,
        )

        # add LoRA adaptor
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if options.add_classification_head:
        # Add a classification head on top of the model
        model.resize_token_embeddings(len(tokenizer))
        model.classifier = Linear(model.config.hidden_size, len(label_scheme))

    print("Model initialized")

    return model


trainer = MultilabelTrainer(
    model=None if options.hp_search else model_init(),
    model_init=model_init if options.hp_search else None,
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
        save_total_limit=options.save_total_limit,
        weight_decay=options.weight_decay,
        warmup_steps=options.warmup_steps,
        warmup_ratio=options.warmup_ratio,
        learning_rate=options.lr,
        max_grad_norm=options.max_grad_norm,
        lr_scheduler_type=options.lr_scheduler_type,
        metric_for_best_model=options.metric_for_best_model,
        greater_is_better=False if "loss" in options.metric_for_best_model else True,
        per_device_train_batch_size=options.train_batch_size,
        per_device_eval_batch_size=options.eval_batch_size,
        num_train_epochs=options.epochs,
        gradient_checkpointing=options.gradient_checkpointing,
        gradient_accumulation_steps=options.gradient_steps,
        optim=options.optim,
        fp16=options.fp16,
        bf16=options.bf16,
        group_by_length=True,
        resume_from_checkpoint=True if options.resume else None,
    ),
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=options.max_length
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=options.patience)],
)

trainer = accelerator.prepare(trainer)

print("Trainer prepared!")

# Start mode

if options.mode == "extract_embeddings":
    model = model_init()
    print("Extracting document embeddings...")
    dataset.set_format(type="torch")
    model = model.to("cpu")

    with open(f"{working_dir}/doc_embeddings2.tsv", "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for d in tqdm(dataset["train"]):
            label_text = d.pop("label_text")
            d.pop("label")
            d.pop("text")
            language = d.pop("language")

            outputs = model(**d, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            # embeddings = last_hidden_states[0, 0, :]
            doc_embeddings = last_hidden_states[0][0, :].detach().numpy()

            writer.writerow(
                [
                    language,
                    label_text,
                    " ".join([str(x) for x in doc_embeddings.tolist()]),
                ]
            )


def pool_embeddings_for_words(token_embeddings, tokens):
    # Initialize a dictionary to hold the pooled embeddings for each full word
    word_embeddings = {}
    current_word_embeddings = []
    current_word = ""

    for idx, token in enumerate(tokens):
        # Skip special tokens like <s>, </s>, etc.
        if token in tokenizer.all_special_tokens:
            continue
        # New word starts with _
        if token.startswith("▁"):
            if current_word_embeddings:
                # Pool the embeddings for the previous word and add to the dictionary
                pooled_embedding = np.mean(current_word_embeddings, axis=0).tolist()
                word_embeddings[current_word] = pooled_embedding
                current_word_embeddings = []

            # Remove the underscore from the token to get the word
            current_word = token[1:]
        else:
            # For tokens that are not the start of a new word, append them to the current word
            current_word += token

        # Add the current subword embedding
        current_word_embeddings.append(token_embeddings[idx])

    # Pool and add the last word
    if current_word_embeddings:
        pooled_embedding = np.mean(current_word_embeddings, axis=0).tolist()
        word_embeddings[current_word] = pooled_embedding

    return word_embeddings


def compute_cosine_similarity(doc_embedding, word_embeddings):
    # Convert the document embedding to a 2D array
    doc_embedding_2d = np.array(doc_embedding).reshape(1, -1)

    # Initialize a dictionary to hold cosine similarities
    cosine_similarities = {}

    for word, word_embedding in word_embeddings.items():
        # Convert the word embedding to a 2D array
        word_embedding_2d = np.array(word_embedding).reshape(1, -1)

        # Compute the cosine similarity
        similarity = cosine_similarity(doc_embedding_2d, word_embedding_2d)[0][0]

        # Store the similarity
        cosine_similarities[word] = similarity

    # Sort the dictionary by similarity in descending order
    sorted_cosine_similarities = dict(
        sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_cosine_similarities


if options.mode == "extract_keywords":
    from sklearn.metrics.pairwise import cosine_similarity

    model = model_init()
    print("Extracting keywords...")
    dataset.set_format(type="torch")
    model = model.to("cpu")

    category_doc_word_similarities = {}
    import json
    from numpy import dot
    from numpy.linalg import norm

    cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))

    with open(f"{working_dir}/keywords2.tsv", "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for d in tqdm(dataset["train"]):
            word_ids = d["input_ids"].tolist()[0]
            tokens = tokenizer.convert_ids_to_tokens(word_ids)
            label_text = d.pop("label_text")
            label_one_hot = d.pop("label")
            text = d.pop("text")
            language = d.pop("language")

            outputs = model(**d, output_hidden_states=True)

            text_embedding = outputs.hidden_states[-1][0][0, :].detach().numpy()
            token_embeddings = outputs.hidden_states[-1][0].detach().numpy()

            word_embeddings = pool_embeddings_for_words(token_embeddings, tokens)

            cosine_similarities = compute_cosine_similarity(
                text_embedding,
                word_embeddings,
            )

            writer.writerow(
                [
                    language,
                    label_text,
                    " ".join([str(x) for x in text_embedding.tolist()]),
                    # json.dumps(str(cosine_similarities)),
                    json.dumps(str(word_embeddings)),
                ]
            )

            continue

            if label_text not in category_doc_word_similarities:
                category_doc_word_similarities[label_text] = {}

            for k, v in cosine_similarities.items():
                if k not in category_doc_word_similarities[label_text]:
                    category_doc_word_similarities[label_text][k] = []
                category_doc_word_similarities[label_text][k].append(v)

        # for k, v in category_doc_word_similarities.items():
        #    print(f"CATEGORY {k}")
        #    print(v)
        #    exit()


if options.mode == "train":
    if not options.hp_search:
        print("Training...")
        log_gpu_memory()
        trainer.train()

    else:
        hp_config = {
            "direction": "maximize",
            "backend": options.hp_search,
            "local_dir": f"{working_dir}/{options.hp_search}",
            "hp_space": {},
        }

        if options.hp_search == "ray":
            ray_init(ignore_reinit_error=True, num_cpus=1)
            """
            hp_config["scheduler"] = ASHAScheduler(metric="eval_f1", mode="max")
            hp_config["search_alg"] = HyperOptSearch(metric="eval_f1", mode="max")
            hp_config["hp_space"] = {
                "learning_rate": loguniform(1e-6, 1e-3),
                "per_device_train_batch_size": choice([6, 8, 12, 16]),
            }
            """
            hp_config["scheduler"] = PopulationBasedTraining(
                metric="eval_f1",
                mode="max",
                perturbation_interval=1,
                hyperparam_mutations={
                    # "learning_rate": uniform(1e-5, 5e-5),
                    "learning_rate": [1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                    # "per_device_train_batch_size": [6, 8, 10],
                },
            )
            hp_config["hp_space"] = lambda _: {}

        elif options.hp_search == "wandb":
            hp_config["hp_space"] = lambda _: {
                "method": "bayes",
                "name": wandb_project_name,
                "metric": {"goal": "maximize", "name": "eval_f1"},
                "parameters": {
                    "per_device_train_batch_size": {"values": [8, 10, 12]},
                    "learning_rate": {"values": [5e-6, 1e-5, 5e-5, 1e-4]},
                },
            }

        best_model = trainer.hyperparameter_search(**hp_config)

        print(f"Best model according to {options.hp_search}:")
        print(best_model)
        exit()

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

print(classification_report(trues, preds, target_names=label_scheme))
print(f"Test eval threshold: {threshold}")

if options.save_model:
    trainer.model.save_pretrained(f"{working_dir}/saved_model")
