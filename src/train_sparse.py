import csv
import glob
import json
import os
import random
import shutil
from collections import namedtuple
from pydoc import locate

from transformers.modeling_outputs import ModelOutput

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.special import expit as sigmoid
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    RobertaForSequenceClassification,
    XLMRobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)


from .data import balanced_dataloader, get_dataset
from .labels import (
    decode_binary_labels,
    get_binary_representations,
    label_schemes,
    map_to_xgenre_binary,
    subcategory_to_parent_index,
    upper_all_indexes,
    upper_all_indexes_en,
)


@dataclass
class SequenceClassifierOutput(ModelOutput):
    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    encoded: torch.FloatTensor = None
    decoded: torch.FloatTensor = None


class SparseXLMRobertaForSequenceClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.encoder2 = nn.Linear(config.hidden_size, 512)
        self.decoder2 = nn.Linear(512, config.hidden_size)
        self.classifier = PoolingXLMRobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        encoded_output = torch.relu(self.encoder2(sequence_output))
        decoded_output = self.decoder2(encoded_output)
        logits = self.classifier(encoded_output)

        loss = None
        if labels is not None:
            print("Error! labels")
            exit()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=outputs.hidden_states[-1],
            attentions=outputs.attentions,
            encoded=encoded_output,
            decoded=decoded_output,
        )


class PoolingXLMRobertaClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(512, 512)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(512, config.num_labels)

    def forward(self, features, **kwargs):
        x = features.mean(dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def run(cfg):

    test_language = ""  # Used when predicting
    test_dataset = []  # Used when predicting
    pred_suffix = ("_" + cfg.test) if "multi" in cfg.test else ""

    # Make process deterministic
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    label_scheme = label_schemes[cfg.labels]
    prediction_label_scheme = label_schemes[cfg.predict_labels]
    print(f"Predicting {len(label_scheme)} labels")
    predict_upper_using_full = cfg.labels == "all" and cfg.predict_labels == "upper"
    predict_upper_en_using_full_en = (
        cfg.labels == "en_all" and cfg.predict_labels == "en_upper"
    )
    predict_xgenre_using_full = cfg.labels == "all" and cfg.predict_labels == "xgenre"
    print(f"Predicting {len(label_scheme)} labels")

    model_output_dir = f"{cfg.model_output}/{cfg.model_name}_sparse{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}{('/fold_'+str(cfg.use_fold)) if cfg.use_fold else ''}{('/subset_'+str(cfg.sample_subset)) if cfg.sample_subset else ''}"
    results_output_dir = f"{cfg.predictions_output}/{cfg.model_name}_sparse{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}{('/fold_'+str(cfg.use_fold)) if cfg.use_fold else ''}{('/subset_'+str(cfg.sample_subset)) if cfg.sample_subset else ''}"
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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = get_dataset(cfg, tokenizer)

    model = SparseXLMRobertaForSequenceClassification.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        num_labels=len(label_scheme),
    )

    print(model)

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
            logits, encoded, decoded = outputs.logits, outputs.encoded, outputs.decoded

            # Classification loss
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

            # Reconstruction loss
            reconstruction_loss = torch.mean((outputs.last_hidden_state - decoded) ** 2)

            # Sparsity loss
            sparsity_loss = torch.mean(torch.abs(encoded))

            # Combine losses
            total_loss = loss + 0 * (reconstruction_loss + sparsity_loss)

            return (total_loss, outputs) if return_outputs else total_loss

        if (
            len(cfg.train.split("-")) > 1 or cfg.balanced_dataloader
        ) and not cfg.just_evaluate:

            def get_train_dataloader(self):
                return balanced_dataloader(self, "train", cfg.train_batch_size)

        if (
            len(cfg.dev.split("-")) > 1 or cfg.balanced_dataloader
        ) and not cfg.just_evaluate:

            def get_eval_dataloader(self, eval_dataset=None):
                return balanced_dataloader(self, "eval", cfg.eval_batch_size)

    def compute_metrics(p):
        true_labels = p.label_ids
        predictions = sigmoid(
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )

        print(predictions)
        print(predictions.shape)
        print(true_labels)
        print(true_labels.shape)

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

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, binary_predictions, average="micro"
        )
        accuracy = accuracy_score(true_labels, binary_predictions)

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
            trues_and_probs = list(
                zip(true_labels, np.round(predictions, 4), example_indices)
            )
            if cfg.save_predictions:
                os.makedirs(results_output_dir, exist_ok=True)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{pred_suffix}{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}.tsv",
                    "w",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter="\t")
                    csv_writer.writerows(data)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{pred_suffix}_probs{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}.tsv",
                    "w",
                    newline="",
                ) as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter="\t")
                    csv_writer.writerows(trues_and_probs)

                with open(
                    f"{results_output_dir}/{cfg.labels}_{cfg.predict_labels}_{test_language}{pred_suffix}{('_'+cfg.multilabel_eval) if cfg.multilabel_eval else ''}_metrics.json",
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
            per_device_eval_batch_size=2,
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
        if not cfg.no_save:
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

        test_language = language

        print(f"-- {language} --")

        test_dataset = dataset["test"].filter(
            lambda example: example["language"] == language,
        )

        trainer.predict(test_dataset)
