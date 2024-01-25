import os
import random
import shutil
from pprint import pprint
import json
import csv
import tempfile
from functools import partial

import numpy as np

import wandb

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from torch.optim import AdamW
from tqdm.auto import tqdm

tqdm = partial(tqdm, position=0, leave=True)
import torch
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR

from peft import get_peft_model, LoraConfig, TaskType

from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import setup_wandb
from ray import init as ray_init

from .labels import get_label_scheme, decode_binary_labels
from .data import get_dataset, preprocess_data
from .dataloader import init_dataloaders
from .utils import get_torch_dtype, get_linear_modules, log_gpu_memory
from .embeddings import extract_doc_embeddings
from .metrics import compute_metrics
from .scheduler import linear_warmup_decay
from .loss import BCEFocalLoss


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

        # Tf32
        if not self.cfg.no_tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.allow_tf32

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

    def _train_epoch(self, optimizer, lr_scheduler, epoch, progress_bar, patience):
        self.model.train()
        batch_losses = []
        log_gpu_memory()
        for batch_i, batch in enumerate(self.dataloaders["train"]):
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
                    f"E-{epoch}:{int((batch_i/len(self.dataloaders['train'])* 100))}% ({patience}), loss: {(sum(batch_losses) / len(batch_losses)):4f}"
                )
        return {
            "train/loss": sum(batch_losses) / len(batch_losses),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/epoch": epoch,
        }

    def _evaluate(self, split="dev"):
        self.model.eval()
        batch_logits = []
        batch_labels = []
        batch_losses = []
        progress_bar = tqdm(range(len(self.dataloaders[split])))
        progress_bar.set_description(f"testing {split}")
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
        out_file = f"{self.cfg.working_dir}/test_{self.cfg.data.test}.csv"

        with open(out_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter="\t")
            csv_writer.writerows(data)

        print(f"Predictions saved to {out_file}")

    def _init_model(self, model_path=None):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model.name if not model_path else model_path,
            num_labels=self.cfg.num_labels,
            low_cpu_mem_usage=self.cfg.model.low_cpu_mem_usage,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True
            )
            if self.cfg.model.quantize
            else None,
        )

        if self.cfg.gpus > 1:
            model = DataParallel(model, device_ids=list(range(self.cfg.gpus)))

        if not self.cfg.model.quantize:
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

    def _condition(self, patience_metric, best_score):
        if "loss" in self.cfg.trainer.best_model_metric:
            return patience_metric < best_score
        return patience_metric > best_score

    def _train(self, config):
        wandb.login()
        wandb.init(
            project=f"{self.cfg.method}_self.cfg.wandb_project",
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

        optimizer = AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=self.cfg.trainer.weight_decay,
        )

        if self.cfg.resume:
            optimizer.load_state_dict(
                torch.load(f"{self.cfg.resume}/optimizer_state.pth")
            )

        lr_scheduler = LambdaLR(
            optimizer,
            linear_warmup_decay(
                self.cfg.trainer.warmup_ratio * num_training_steps,
                num_training_steps,
            ),
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

        progress_bar = tqdm(range(num_training_steps))
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

        return best_starting_score, best_score

    def finetune(self):
        print("Fine-tuning")

        config = {"learning_rater": self.cfg.trainer.learning_rate}

        best_starting_score, best_score = self._train(config)

        if (
            self.cfg.model.save
            and best_score is not False
            and (
                not self.cfg.resume or self._condition(best_score, best_starting_score)
            )
        ):
            self._save_model()

        self.predict(from_checkpoint=True)

    def ray_tune(self):
        config = {
            "learning_rate": tune.quniform(
                *self.cfg.ray.learning_rate, self.cfg.ray.learning_rate[0]
            ),
        }
        scheduler = tune.schedulers.ASHAScheduler()

        ray_init(
            ignore_reinit_error=True, num_cpus=1, _temp_dir=self.cfg.root_path + "/tmp"
        )

        os.makedirs(f"{self.cfg.working_dir}/ray", exist_ok=True)

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
            run_config=train.RunConfig(
                name=self.cfg.wandb_project,
                storage_path=f"{self.cfg.working_dir}/ray",
            ),
            param_space=config,
        )
        results = tuner.fit()

        best_result = results.get_best_result("loss", "min")

        print("Best trial config: {}".format(best_result.config))
        print(
            "Best trial final validation loss: {}".format(best_result.metrics["loss"])
        )
        print(
            "Best trial final validation accuracy: {}".format(
                best_result.metrics["accuracy"]
            )
        )
