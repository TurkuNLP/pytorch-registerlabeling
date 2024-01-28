import os
import random
import shutil
from pprint import pprint
import json
import csv
import tempfile
import math

import numpy as np

import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from accelerate import Accelerator
from accelerate.state import AcceleratorState

from torch.optim import AdamW
from tqdm import tqdm, trange
import torch
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR

from peft import get_peft_model, LoraConfig, TaskType

from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch
from ray import init as ray_init
from ray.train import RunConfig

from .labels import get_label_scheme, decode_binary_labels
from .data import get_dataset, preprocess_data
from .dataloader import init_dataloaders
from .utils import get_torch_dtype, get_linear_modules, log_gpu_memory
from .embeddings import extract_doc_embeddings
from .metrics import compute_metrics
from .scheduler import linear_warmup_decay
from .loss import BCEFocalLoss
from .optimizer import create_optimizer
from .model import PooledRobertaForSequenceClassification


class Main:
    def __init__(self, cfg):
        cfg.torch_dtype = get_torch_dtype(cfg.torch_dtype)
        cfg.label_scheme = get_label_scheme(cfg.data.labels)
        cfg.num_labels = len(cfg.label_scheme)
        cfg.device = torch.device(cfg.device)
        cfg.working_dir = "/".join(
            [
                cfg.data.output_path,
                cfg.model.name,
                f"labels_{cfg.data.labels}",
                "_".join([cfg.data.train or "", cfg.data.dev or ""]),
                f"seed_{cfg.seed}",
            ]
        )
        cfg.wandb_project = cfg.working_dir.split("/", 1)[1].replace("/", ",")
        print(f"Working directory: {cfg.working_dir}")
        self.cfg = cfg
        if self.cfg.accelerate:
            self.accelerator = Accelerator()
            self.state = AcceleratorState()

            print(
                f"Accelerate is using {self.state.num_processes} GPUs using {self.state.distributed_type}."
            )

            num_gpus = torch.cuda.device_count()
            print(f"torch.cuda.device_count() returned {num_gpus}.")

            if self.cfg.accelerate_bf16:
                self.accelerator.bf16_weights = True

        # Tf32
        if not self.cfg.no_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True

        # Make process deterministic
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Prepare dataset
        dataset = get_dataset(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name, torch_dtype=cfg.torch_dtype
        )
        self.dataset = preprocess_data(
            dataset,
            self.tokenizer,
            cfg.seed,
            cfg.data.max_length,
            cfg.data.remove_unused_cols,
            cfg.data.no_dynamic_padding,
        )

        # Get dataloaders
        self.dataloaders = init_dataloaders(
            self.dataset,
            cfg.dataloader,
            self.tokenizer.pad_token_id,
        )

        # Run
        getattr(self, cfg.method)()

    def _evaluate(self, split="dev"):
        self.model.eval()
        batch_logits = []
        batch_labels = []
        batch_losses = []

        progress_bar = trange(len(self.dataloaders[split]))

        progress_bar.set_description(f"Testing {split}")
        for batch in self.dataloaders[split]:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.no_grad():
                outputs = self.model(**batch)

            loss = BCEFocalLoss(
                outputs,
                labels,
                self.cfg.trainer.loss_gamma,
                self.cfg.trainer.loss_alpha,
            )
            batch_logits.append(outputs.logits)
            batch_labels.append(labels)
            batch_losses.append(loss.item())

            progress_bar.update(1)

        metrics = compute_metrics(
            torch.cat(batch_logits, dim=0),
            torch.cat(batch_labels, dim=0),
            split,
            self.cfg.label_scheme if split == "test" else None,
            True if split == "test" else False,
        )
        if split == "dev":
            metrics["dev/loss"] = sum(batch_losses) / len(batch_losses)
            return metrics

        elif split == "test":
            self._save_predictions(*metrics[1])
            return metrics[0]

    def _wrap_peft(self):
        print("Wrapping PEFT model")

        if self.cfg.peft.target_modules == "linear":
            target_modules = get_linear_modules(self.model)
        else:
            target_modules = self.cfg.peft.target_modules.split(",")

        self.lora_config = LoraConfig(
            r=self.cfg.peft.lora_rank,
            lora_alpha=self.cfg.peft.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def _save_checkpoint(self, optimizer, lr_scheduler, dev_metrics):
        checkpoint_dir = f"{self.cfg.working_dir}/best_checkpoint"
        os.makedirs(self.cfg.working_dir, exist_ok=True)
        if self.cfg.gpus > 1:
            self.model.module.save_pretrained(checkpoint_dir)
        else:
            self.model.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), f"{checkpoint_dir}/optimizer_state.pth")
        torch.save(
            lr_scheduler.state_dict(), f"{checkpoint_dir}/lr_scheduler_state.pth"
        )

        with open(f"{checkpoint_dir}/dev_metrics.json", "w") as f:
            json.dump(dev_metrics, f)

    def _save_model(self):
        shutil.rmtree(f"{self.cfg.working_dir}/best_model", ignore_errors=True)
        shutil.copytree(
            f"{self.cfg.working_dir}/best_checkpoint",
            f"{self.cfg.working_dir}/best_model",
        )

    def _save_predictions(self, trues, preds):
        true_labels_str = decode_binary_labels(trues, self.cfg.label_scheme)
        predicted_labels_str = decode_binary_labels(preds, self.cfg.label_scheme)

        data = list(zip(true_labels_str, predicted_labels_str))
        out_file = f"{self.cfg.working_dir}/test_{self.cfg.data.test or self.cfg.data.dev or self.cfg.data.train}.csv"

        with open(out_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter="\t")
            csv_writer.writerows(data)

        print(f"Predictions saved to {out_file}")

    def _init_model(self, model_path=None):
        model_cls = AutoModelForSequenceClassification
        if self.cfg.model.roberta_pooled:
            model_cls = PooledRobertaForSequenceClassification

        model_params = {
            "num_labels": self.cfg.num_labels,
        }
        if self.cfg.model.low_cpu_mem_usage:
            model_params["low_cpu_mem_usage"] = True
        if self.cfg.model.quantize:
            model_params["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True
            )
        if self.cfg.model.roberta_pooled:
            model_params["pooling"] = self.cfg.model.roberta_pooled

        model = model_cls.from_pretrained(
            self.cfg.model.name if not model_path else model_path, **model_params
        )

        if self.cfg.gpus > 1 and not self.cfg.accelerate:
            model = DataParallel(model, device_ids=list(range(self.cfg.gpus)))

        if not self.cfg.model.quantize and not self.cfg.accelerate:
            model = model.to(self.cfg.device, dtype=self.cfg.torch_dtype)

        if self.cfg.model.compile:
            model = torch.compile(model)

        self.model = model

    def extract_doc_embeddings(self):
        path = "/".join(self.cfg.working_dir.split("/")[:-1]) + "/embeddings"
        self._init_model()
        os.makedirs(path, exist_ok=True)
        extract_doc_embeddings(
            self.model, self.dataset, path, self.cfg.device, self.cfg.embeddings
        )

    def predict(self, from_checkpoint=False):
        model_path = f"{self.cfg.working_dir}/best_{'checkpoint' if from_checkpoint else 'model'}"

        if self.cfg.peft.enable:
            self._init_model()
            self.model.load_adapter(model_path)
        else:
            self._init_model(model_path)

        if self.cfg.data.dev or self.cfg.method == "finetune":
            print("Final dev set evaluation")
            print(self._evaluate())

        print("Test set evaluation")
        print(self._evaluate("test"))

    def _train_epoch(self, optimizer, lr_scheduler, epoch, progress_bar, patience):
        self.model.train()
        batch_losses = []
        for batch_i, batch in enumerate(self.dataloaders["train"]):
            if not self.cfg.accelerate:
                batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = self.model(**batch)

            loss = BCEFocalLoss(
                outputs,
                labels,
                self.cfg.trainer.loss_gamma,
                self.cfg.trainer.loss_alpha,
            )

            batch_losses.append(loss.item())
            loss = loss / self.cfg.trainer.gradient_accumulation_steps
            if self.cfg.accelerate:
                self.accelerator.backward(loss)
            else:
                loss.backward()
            if (batch_i + 1) % self.cfg.trainer.gradient_accumulation_steps == 0:
                if self.cfg.trainer.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.trainer.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_description(
                    f"E-{epoch}:{int((batch_i/len(self.dataloaders['train'])* 100))}% ({patience}), loss: {(sum(batch_losses) / len(batch_losses)):4f}",
                    refresh=False,
                )

        log_gpu_memory()

        return {
            "train/loss": sum(batch_losses) / len(batch_losses),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/epoch": epoch,
        }

    def _condition(self, patience_metric, best_score):
        if "loss" in self.cfg.trainer.best_model_metric:
            return patience_metric < best_score
        return patience_metric > best_score

    def _train(self, config={}):
        # TODO: Not working with ray!
        wandb.login()
        wandb.init(
            project=f"{self.cfg.method}_{self.cfg.wandb_project}",
            config=self.cfg,
        )
        self._init_model(
            self.cfg.resume if (self.cfg.resume and not self.cfg.peft.enable) else None
        )

        if self.cfg.peft.enable:
            if not self.cfg.resume:
                self._wrap_peft()
            else:
                self.model.load_adapter(self.cfg.resume)

        num_training_steps = int(
            self.cfg.trainer.epochs
            * len(self.dataloaders["train"])
            / self.cfg.trainer.gradient_accumulation_steps
        )

        optimizer = create_optimizer(
            self.model,
            {
                "lr": config.get("learning_rate", self.cfg.trainer.learning_rate),
                "weight_decay": config.get(
                    "weight_decay", self.cfg.trainer.weight_decay
                ),
            },
        )

        if self.cfg.resume:
            optimizer.load_state_dict(
                torch.load(f"{self.cfg.resume}/optimizer_state.pth")
            )

        lr_scheduler = LambdaLR(
            optimizer,
            linear_warmup_decay(
                math.ceil(num_training_steps * self.cfg.trainer.warmup_ratio),
                num_training_steps,
            ),
        )
        if self.cfg.accelerate:
            (
                self.model,
                optimizer,
                self.dataloaders["train"],
                lr_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                optimizer,
                self.dataloaders["train"],
                lr_scheduler,
            )

        best_starting_score = False

        if self.cfg.resume:
            lr_scheduler.load_state_dict(
                torch.load(f"{self.cfg.resume}/lr_scheduler_state.pth")
            )
            with open(f"{self.cfg.resume}/model_state.json", "r") as f:
                loaded_data = json.load(f)
                best_starting_score = loaded_data[self.cfg.trainer.best_model_metric]
                print(
                    f"Previous best {self.cfg.trainer.best_model_metric} was {best_score}"
                )
        progress_bar = trange(num_training_steps, mininterval=self.cfg.tqdm_mininterval)
        best_epoch = 0
        best_score = best_starting_score
        remaining_patience = ""

        for epoch in range(self.cfg.trainer.epochs):
            train_metrics = self._train_epoch(
                optimizer, lr_scheduler, epoch + 1, progress_bar, remaining_patience
            )
            pprint(train_metrics)
            dev_metrics = self._evaluate()
            pprint(dev_metrics)
            wandb.log({**dev_metrics, **train_metrics})
            patience_metric = dev_metrics[self.cfg.trainer.best_model_metric]
            if best_score is False or self._condition(patience_metric, best_score):
                best_score = patience_metric
                best_epoch = epoch
                self._save_checkpoint(optimizer, lr_scheduler, dev_metrics)

            elif epoch - best_epoch > self.cfg.trainer.patience:
                print("Early stopped training at epoch %d" % epoch)
                break

            if self.cfg.method == "ray_tune":
                with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                    path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
                    torch.save((self.model.state_dict(), optimizer.state_dict()), path)
                    checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)
                    train.report(
                        {"loss": dev_metrics["dev/loss"], "f1": dev_metrics["dev/f1"]},
                        checkpoint=checkpoint,
                    )

            remaining_patience = f"{self.cfg.trainer.patience - (epoch - best_epoch)}/{self.cfg.trainer.patience}"

        if (
            self.cfg.model.save
            and best_score is not False
            and (
                not self.cfg.resume or self._condition(best_score, best_starting_score)
            )
        ):
            self._save_model()

        if self.cfg.predict:
            self.predict(from_checkpoint=True)

    def finetune(self):
        print("Fine-tuning")
        self._train()

    def ray_tune(self):
        self.cfg.tqdm_mininterval = 10
        self.cfg.model.save = False
        self.cfg.predict = False

        config = {
            "learning_rate": tune.quniform(
                *self.cfg.ray.learning_rate, self.cfg.ray.learning_rate[0]
            ),
        }
        scheduler = tune.schedulers.ASHAScheduler()

        ray_init(
            ignore_reinit_error=True, num_cpus=1, _temp_dir=self.cfg.root_path + "/tmp"
        )
        ray_dir = f"{self.cfg.root_path}/tmp/ray/{self.cfg.wandb_project}"
        shutil.rmtree(ray_dir, ignore_errors=True)
        os.makedirs(ray_dir, exist_ok=True)

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self._train),
                resources={"cpu": 1, "gpu": self.cfg.ray.gpus_per_trial},
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=20,
                search_alg=HyperOptSearch(metric="loss", mode="min"),
            ),
            run_config=RunConfig(
                name=self.cfg.wandb_project,
                storage_path=ray_dir,
                local_dir=ray_dir,
            ),
            param_space=config,
        )
        results = tuner.fit()

        best_result = results.get_best_result("loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print(
            "Best trial final validation loss: {}".format(best_result.metrics["loss"])
        )
        print("Best trial final validation f1: {}".format(best_result.metrics["f1"]))

    def hf_finetune(self):
        from transformers import (
            Trainer,
            TrainingArguments,
            EarlyStoppingCallback,
            DataCollatorWithPadding,
        )

        self._init_model()

        loss_gamma = self.trainer.loss_gamma
        loss_alpha = self.trainer.loss_alpha

        def compute_metrics_fn(p):
            _, labels = p
            predictions = (
                p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            )
            return compute_metrics(predictions, labels)

        class MultiLabelTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super(MultiLabelTrainer, self).__init__(*args, **kwargs)

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                loss = BCEFocalLoss(outputs, labels, loss_gamma, loss_alpha)

                return (loss, outputs) if return_outputs else loss

        trainer = MultiLabelTrainer(
            model=self.model,
            args=TrainingArguments(
                f"{self.cfg.working_dir}/hf_checkpoints",
                overwrite_output_dir=not self.cfg.resume,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                logging_strategy="epoch",
                load_best_model_at_end=self.cfg.model.save,
                save_total_limit=2,
                weight_decay=self.cfg.trainer.weight_decay,
                warmup_ratio=self.cfg.trainer.warmup_ratio,
                learning_rate=self.cfg.trainer.learning_rate,
                max_grad_norm=self.cfg.trainer.max_grad_norm,
                lr_scheduler_type="linear",
                metric_for_best_model=self.cfg.trainer.best_model_metric,
                greater_is_better=not "loss" in self.cfg.trainer.best_model_metric,
                per_device_train_batch_size=self.cfg.dataloader.train_batch_size,
                per_device_eval_batch_size=self.cfg.dataloader.dev_batch_size,
                num_train_epochs=self.cfg.trainer.epochs,
                gradient_checkpointing=False,
                gradient_accumulation_steps=self.cfg.trainer.gradient_accumulation_steps,
                optim="adamw_torch",
                bf16=self.cfg.bf16,
                tf32=not self.cfg.no_tf32,
                resume_from_checkpoint=self.cfg.resume,
            ),
            train_dataset=self.dataset.get("train", []),
            eval_dataset=self.dataset.get("dev", []),
            compute_metrics=compute_metrics_fn,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                padding="longest",
                max_length=self.cfg.data.max_length,
            ),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.cfg.trainer.patience)
            ],
        )

        trainer.train()
