import csv
import glob
import json
import os
import random
import shutil
from pydoc import locate

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import balanced_dataloader, get_dataset
from .labels import (
    decode_binary_labels,
    label_schemes,
    subcategory_to_parent_index,
    map_to_xgenre_binary,
    upper_all_indexes,
    upper_all_indexes_en,
    get_binary_representations,
)


def get_linear_modules(model):

    linear_modules = set()

    for name, module in model.named_modules():
        name = name.lower()
        if "attention" in name and "self" in name and "Linear" in str(type(module)):
            linear_modules.add(name.split(".")[-1])

    print(f"\nFound linear modules: {linear_modules}")
    return list(linear_modules)


def run(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_language = ""  # Used when predicting
    test_dataset = []  # Used when predicting
    multilabel_exclusion_stats = {"excluded": 0, "included": 0}

    # CUDA events for timing
    if device == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    label_scheme = label_schemes[cfg.labels]
    prediction_label_scheme = label_schemes[cfg.predict_labels]
    print(f"Predicting {len(label_scheme)} labels")
    predict_upper_using_full = cfg.labels == "all" and cfg.predict_labels == "upper"
    predict_upper_en_using_full_en = (
        cfg.labels == "en_all" and cfg.predict_labels == "en_upper"
    )
    predict_xgenre_using_full = cfg.labels == "all" and cfg.predict_labels == "xgenre"
    model_output_dir = f"{cfg.model_output}/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}{('/fold_'+str(cfg.use_fold)) if cfg.use_fold else ''}"
    results_output_dir = f"{cfg.predictions_output}/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}{('/fold_'+str(cfg.use_fold)) if cfg.use_fold else ''}"
    print(
        f"This run {'saves models to' if not cfg.just_evaluate else 'uses model from'} {model_output_dir}"
    )
    print(f"Results are logged to {results_output_dir}")
    torch_dtype = locate(f"torch.{cfg.torch_dtype}")
    if not torch.cuda.is_available():
        torch_dtype = torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if "mixtral" in cfg.model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
    dataset = get_dataset(cfg, tokenizer)

    base_model_path = (
        model_output_dir if cfg.just_evaluate and not cfg.peft else cfg.model_name
    )

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=len(label_scheme),
        torch_dtype=torch_dtype,
        use_flash_attention_2=cfg.fa2,
        quantization_config=nf4_config if cfg.nf4 else None,
        device_map="auto" if "mixtral" in cfg.model_name.lower() else None,
    )

    label2id = {label: id for id, label in enumerate(label_scheme)}
    id2label = {id: label for label, id in label2id.items()}

    model.config.label2id = label2id
    model.config.id2label = id2label

    if cfg.peft:
        if cfg.just_evaluate:
            model = PeftModel.from_pretrained(model, model_output_dir)
        else:
            print("Using LoRa")
            model = get_peft_model(
                model,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    target_modules=(
                        get_linear_modules(model)
                        if not cfg.target_modules
                        else cfg.target_modules.split(",")
                    ),
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                ),
            )

    class CustomEarlyStoppingCallback(EarlyStoppingCallback):
        def __init__(
            self,
            early_stopping_patience: int = 1,
            early_stopping_threshold: float = 0.0,
        ):
            super().__init__(early_stopping_patience, early_stopping_threshold)
            self.best_epoch = 0

        def check_metric_value(self, args, state, control, metric_value):
            # best_metric is set by code for load_best_model
            operator = np.greater if args.greater_is_better else np.less
            if state.best_metric is None or (
                operator(metric_value, state.best_metric)
                and abs(metric_value - state.best_metric)
                > self.early_stopping_threshold
            ):
                self.early_stopping_patience_counter = 0
                self.best_epoch = state.global_step  # Update the best epoch
            else:
                self.early_stopping_patience_counter += 1

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

        if len(cfg.train.split("-")) > 1 and not cfg.just_evaluate:

            def get_train_dataloader(self):
                return balanced_dataloader(self, "train", cfg.train_batch_size)

        if len(cfg.dev.split("-")) > 1 and not cfg.just_evaluate:

            def get_eval_dataloader(self, eval_dataset=None):
                return balanced_dataloader(self, "eval", cfg.eval_batch_size)

    def compute_metrics(p):
        true_labels = p.label_ids
        predictions = sigmoid(
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        if cfg.labels in ["all", "all_mix"]:
            # Ensure that subcategory has corresponding parent category
            for i in range(predictions.shape[0]):
                for (
                    subcategory_index,
                    parent_index,
                ) in subcategory_to_parent_index.items():
                    if predictions[i, parent_index] < predictions[i, subcategory_index]:
                        predictions[i, parent_index] = predictions[i, subcategory_index]

        if predict_upper_using_full:
            true_labels = true_labels[:, upper_all_indexes]
            predictions = predictions[:, upper_all_indexes]
        elif predict_upper_en_using_full_en:
            true_labels = true_labels[:, upper_all_indexes_en]
            predictions = predictions[:, upper_all_indexes_en]

        best_threshold, best_f1 = 0, 0
        for threshold in np.arange(0.3, 0.7, 0.05):
            binary_predictions = predictions > threshold
            f1 = f1_score(true_labels, binary_predictions, average="micro")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        binary_predictions = predictions > best_threshold

        if predict_xgenre_using_full:
            true_labels, binary_predictions = map_to_xgenre_binary(
                true_labels, binary_predictions
            )

        if cfg.multilabel_eval:
            # Get row indices for binary representations of multilabel predictions
            binary_representations = get_binary_representations(cfg.predict_labels)
            multilabel_prediction_indexes = []

            for i, example in enumerate(binary_predictions):
                if cfg.multilabel_eval == "exclude_multilabel":
                    condition = [int(val) for val in example] in binary_representations
                elif cfg.multilabel_eval == "exclude_singlelabel":
                    condition = [
                        int(val) for val in example
                    ] not in binary_representations

                if condition:
                    multilabel_prediction_indexes.append(i)

            # Filter predictions and true_labels
            binary_predictions = binary_predictions[multilabel_prediction_indexes]
            true_labels = true_labels[multilabel_prediction_indexes]

            multilabel_exclusion_stats["included"] += len(multilabel_prediction_indexes)
            multilabel_exclusion_stats["excluded"] += len(predictions) - len(
                multilabel_prediction_indexes
            )

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, binary_predictions, average="micro"
        )
        accuracy = accuracy_score(true_labels, binary_predictions)
        # pr_auc = average_precision_score(true_labels, predictions, average="micro")

        metrics = {
            "f1": f1,
            "f1_macro": f1_score(
                true_labels, binary_predictions, average="macro", zero_division=np.nan
            ),
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            # "pr_auc": pr_auc,
            "threshold": best_threshold,
        }

        if cfg.just_evaluate:

            cl_report_dict = classification_report(
                true_labels,
                binary_predictions,
                target_names=label_schemes[cfg.predict_labels],
                digits=4,
                output_dict=True,
            )
            metrics["label_scores"] = {
                key: val
                for key, val in cl_report_dict.items()
                if key in prediction_label_scheme
            }

            true_labels_str = decode_binary_labels(true_labels, cfg.labels)
            predicted_labels_str = decode_binary_labels(binary_predictions, cfg.labels)
            example_indices = [x["row"] for x in test_dataset]
            data = list(zip(true_labels_str, predicted_labels_str, example_indices))
            trues_and_probs = list(zip(true_labels, np.round(predictions, 4)))
            if cfg.save_predictions:
                os.makedirs(results_output_dir, exist_ok=True)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}.tsv",
                    "w",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter="\t")
                    csv_writer.writerows(data)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}_probs_{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}.tsv",
                    "w",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter="\t")
                    csv_writer.writerows(trues_and_probs)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}_metrics.json",
                    "w",
                ) as f:
                    json.dump(metrics, f)

            print(metrics)

        return metrics

    early_stopping_callback = CustomEarlyStoppingCallback(
        early_stopping_patience=cfg.patience
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=model_output_dir,
            overwrite_output_dir=True,
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
            eval_accumulation_steps=8,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            save_total_limit=2,
            tf32=True if torch.cuda.is_available() else False,
            group_by_length=True,
            report_to=None,
        ),
        train_dataset=dataset.get("train", []),
        eval_dataset=dataset.get("dev", []),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    if not cfg.just_evaluate:
        trainer.train()
        trainer.save_model()
        for dir_path in glob.glob(f"{model_output_dir}/checkpoint*"):
            shutil.rmtree(dir_path, ignore_errors=True)
        shutil.rmtree(f"{model_output_dir}/runs", ignore_errors=True)

        # Gather training parameters and metadata
        training_metadata = {
            "batch_size": trainer.args.per_device_train_batch_size,
            "learning_rate": trainer.args.learning_rate,
            "warmup_ratio": trainer.args.warmup_ratio,
            "stopped_epoch": early_stopping_callback.best_epoch,
            "total_epochs": trainer.state.epoch,
        }

        with open(f"{model_output_dir}/training_metadata.json", "w") as f:
            json.dump(training_metadata, f, indent=4)

    print("Predicting..")
    cfg.just_evaluate = True
    for language in cfg.test.split("-"):
        print(f"-- {language} --")
        test_language = language
        test_dataset = dataset["test"].filter(
            lambda example: example["language"] == language
        )

        if cfg.sample:

            test_dataset = test_dataset.select(range(cfg.sample))

        if device == "cuda":

            start_event.record()
            trainer.predict(test_dataset)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start_event.elapsed_time(end_event)

            total_samples = len(test_dataset)
            latency = (
                elapsed_time_ms / total_samples
            )  # Latency per sample in milliseconds
            throughput = total_samples / (
                elapsed_time_ms / 1000
            )  # Throughput in samples per second

            print(f"Latency per sample: {latency} ms")
            print(f"Throughput: {throughput} samples/sec")

        else:
            trainer.predict(test_dataset)

        if cfg.multilabel_eval:
            print(
                f"Excluded {multilabel_exclusion_stats['excluded']} examples and kept {multilabel_exclusion_stats['included']} examples"
            )
