import os
import random
import shutil
from pprint import pprint

import numpy as np

import wandb

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR

from peft import get_peft_model, LoraConfig, TaskType

from .labels import get_label_scheme
from .data import get_dataset, preprocess_data
from .dataloader import init_dataloaders
from .utils import get_torch_dtype, get_linear_modules
from .metrics import compute_metrics
from .scheduler import linear_warmup_decay
from .loss import BCEFocalLoss

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32


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
        print(f"Working directory: {cfg.working_dir}")
        self.cfg = cfg

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
            dataset, self.tokenizer, cfg.seed, cfg.data.max_length
        )

        # Get dataloaders
        self.dataloaders = init_dataloaders(
            self.dataset,
            cfg.dataloader,
            self.tokenizer.pad_token_id,
        )

        # Run
        getattr(self, cfg.method)()

    def _train(self, optimizer, lr_scheduler, epoch, progress_bar):
        self.model.train()
        batch_losses = []
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
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                progress_bar.set_description(
                    f"Epoch {epoch} ({int((batch_i/len(self.dataloaders['train'])* 100))}%)"
                )
        return {
            "train/loss": sum(batch_losses) / len(batch_losses),
            "train/learning_rate": optimizer.param_groups[0]["lr"],
            "train/epoch": epoch,
        }

    def _evaluate(self, split="dev", cl_report=False):
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
            self.cfg.label_scheme if cl_report else None,
        )
        if split == "dev":
            metrics["dev/loss"] = sum(batch_losses) / len(batch_losses)

        return metrics

    def _wrap_peft(self):
        print("Wrapping PEFT model")

        target_modules = self.cfg.peft.target_modules
        if self.cfg.peft.target_modules == "linear":
            target_modules = get_linear_modules(self.model)

        self.lora_config = LoraConfig(
            r=self.cfg.peft.lora_rank,
            lora_alpha=self.cfg.peft.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            task_type=TaskType.SEQ_CLS,
        )

        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def _save_checkpoint(self):
        os.makedirs(self.cfg.working_dir, exist_ok=True)
        self.model.save_pretrained(
            f"{self.cfg.working_dir}/best_checkpoint",
        )

    def _save_model(self):
        shutil.copytree(
            f"{self.cfg.working_dir}/best_checkpoint",
            f"{self.cfg.working_dir}/best_model",
        )

    def _init_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model.name, num_labels=self.cfg.num_labels
        ).to(self.cfg.device, dtype=self.cfg.torch_dtype)

        if self.cfg.model.compile:
            model = torch.compile(model)

        self.model = model

    def predict(self, from_checkpoint=False):
        print("Test evaluation")

        model_path = f"{self.cfg.working_dir}/best_{'checkpoint' if from_checkpoint else 'model'}"

        if self.cfg.peft.enable:
            self.model = self._init_model()
            self.model.load_adapter(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        print(self._evaluate("test", cl_report=True))

    def finetune(self):
        print("Fine-tuning")
        # Wandb

        wandb.login()
        wandb.init(
            project=self.cfg.working_dir.split("/", 1)[1].replace("/", ","),
            config=self.cfg,
        )

        # Init model
        self._init_model()

        if self.cfg.peft.enable:
            self._wrap_peft()

        num_training_steps = int(
            self.cfg.trainer.epochs
            * len(self.dataloaders["train"])
            / self.cfg.trainer.gradient_accumulation_steps
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.trainer.learning_rate,
            weight_decay=self.cfg.trainer.weight_decay,
        )

        lr_scheduler = LambdaLR(
            optimizer,
            linear_warmup_decay(
                self.cfg.trainer.warmup_ratio * num_training_steps,
                num_training_steps,
            ),
        )

        progress_bar = tqdm(range(num_training_steps))
        best_score = -1
        best_epoch = -1
        for epoch in range(self.cfg.trainer.epochs):
            train_metrics = self._train(
                optimizer, lr_scheduler, epoch + 1, progress_bar
            )
            pprint(train_metrics)
            dev_metrics = self._evaluate()
            pprint(dev_metrics)
            wandb.log({**dev_metrics, **train_metrics})
            patience_metric = dev_metrics[self.cfg.trainer.best_model_metric]
            if patience_metric > best_score:
                best_score = patience_metric
                best_epoch = epoch
                self._save_checkpoint()
            elif epoch - best_epoch > self.cfg.trainer.patience:
                print("Early stopped training at epoch %d" % epoch)
                break

        if best_score > -1 and self.cfg.model.save:
            self._save_model()

        self.predict(from_checkpoint=True)
