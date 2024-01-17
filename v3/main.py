import os
import random

import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm.auto import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR

from .labels import get_label_scheme
from .data import get_dataset, preprocess_data
from .dataloader import init_dataloaders
from .utils import get_torch_dtype
from .metrics import compute_metrics
from .scheduler import linear_warmup_decay
from .loss import BCEFocalLoss

torch.set_float32_matmul_precision("high")


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
                "_".join(
                    [cfg.data.train or "", cfg.data.dev or "", cfg.data.test or ""]
                ),
            ]
        )
        self.cfg = cfg

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        # for cuda
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        # Prepare dataset
        dataset = get_dataset(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.name, torch_dtype=self.cfg.torch_dtype
        )
        self.dataset = preprocess_data(
            dataset, self.tokenizer, self.cfg.seed, self.cfg.data.max_length
        )

        # Get dataloaders
        self.dataloaders = init_dataloaders(
            self.dataset, self.cfg.dataloader, self.tokenizer.pad_token_id
        )

        # Init model
        self._init_model()

        # Run
        getattr(self, cfg.method)()

    def _checkpoint(self):
        os.makedirs(self.cfg.working_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"{self.cfg.working_dir}/best_model.pth",
        )

    def _resume(self):
        self.model.load_state_dict(torch.load(f"{self.cfg.working_dir}/best_model.pth"))

    def _train(self, optimizer, lr_scheduler, epoch, progress_bar):
        self.model.train()
        batch_i = 0
        for batch in self.dataloaders["train"]:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = self.model(**batch)

            loss = BCEFocalLoss(
                outputs,
                labels,
                self.cfg.trainer.loss_gamma,
                self.cfg.trainer.loss_alpha,
            )

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            batch_i += 1
            progress_bar.set_description(
                f"Epoch {epoch} ({int((batch_i/len(self.dataloaders['train'])* 100))}%)"
            )

    def _evaluate(self, split="dev", cl_report=False):
        self.model.eval()
        all_logits = []
        all_labels = []
        progress_bar = tqdm(range(len(self.dataloaders[split])))
        progress_bar.set_description(f"testing {split}")
        for batch in self.dataloaders[split]:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.no_grad():
                outputs = self.model(**batch)

            all_logits.append(outputs.logits)
            all_labels.append(labels)

            progress_bar.update(1)

        return compute_metrics(
            torch.cat(all_logits, dim=0),
            torch.cat(all_labels, dim=0),
            self.cfg.label_scheme if cl_report else None,
        )

    def _init_model(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model.name, num_labels=self.cfg.num_labels
        ).to(self.cfg.device, dtype=self.cfg.torch_dtype)

        if self.cfg.model.compile:
            self.model = torch.compile(self.model)

    def predict(self):
        print("Test evaluation")
        self._resume()
        print(self._evaluate("test", True))

    def finetune(self):
        print("Fine-tuning")

        num_training_steps = self.cfg.trainer.epochs * len(self.dataloaders["train"])

        optimizer = AdamW(self.model.parameters(), lr=self.cfg.trainer.learning_rate)

        lr_scheduler = LambdaLR(
            optimizer,
            linear_warmup_decay(
                self.cfg.trainer.warmup_ratio * num_training_steps, num_training_steps
            ),
        )

        progress_bar = tqdm(range(num_training_steps))
        best_score = -1
        best_epoch = -1
        for epoch in range(self.cfg.trainer.epochs):
            self._train(optimizer, lr_scheduler, epoch + 1, progress_bar)
            metrics = self._evaluate()
            print(metrics)
            patience_metric = metrics[self.cfg.trainer.best_model_metric]
            if patience_metric > best_score:
                best_score = patience_metric
                best_epoch = epoch
                self._checkpoint()
            elif epoch - best_epoch > self.cfg.trainer.patience:
                print("Early stopped training at epoch %d" % epoch)
                break

        self.predict()
