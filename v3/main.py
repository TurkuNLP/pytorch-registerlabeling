from .labels import get_label_scheme
from .data import get_dataset, preprocess_data
from .utils import get_torch_dtype
from .metrics import compute_metrics
from .scheduler import linear_warmup_decay
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn.functional as F
import numpy as np


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

        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # Prepare dataset
        dataset = get_dataset(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model.name, torch_dtype=self.cfg.torch_dtype
        )
        self.dataset = preprocess_data(
            dataset, self.tokenizer, self.cfg.seed, self.cfg.data.max_length
        )

        # Get dataloaders
        self.dataloaders = {k: self._init_dataloader(k) for k in self.dataset.keys()}

        # Run
        getattr(self, cfg.method)()

    def _init_dataloader(self, split):
        def collate_fn(batch):
            max_length = max(len(example["input_ids"]) for example in batch)
            # Pad sequences dynamically to the maximum length in the batch
            for example in batch:
                pad_length = max_length - len(example["input_ids"])
                for key in example:
                    if key == "input_ids":
                        # Use tokenizer.pad_token_id as the padding value for input_ids
                        example[key] = torch.nn.functional.pad(
                            example[key],
                            (0, pad_length),
                            value=self.tokenizer.pad_token_id,
                        )
                    elif key == "attention_mask":
                        # Use 0 as the padding value for attention_mask
                        example[key] = torch.nn.functional.pad(
                            example[key], (0, pad_length), value=0
                        )

            return {
                key: torch.stack([example[key] for example in batch])
                for key in batch[0]
            }

        dataloader = DataLoader(
            self.dataset[split],
            shuffle=True,
            batch_size=self.cfg.dataloader[f"{split}_batch_size"],
            collate_fn=collate_fn,
        )
        print(f"{split} dataloader size: {len(dataloader)}")

        return dataloader

    def _checkpoint(self, model):
        torch.save(
            model.state_dict(),
            f"{self.working_dir}/best_model.pth",
        )

    def _resume(self, model):
        model.load_state_dict(torch.load(f"{self.working_dir}/best_model.pth"))

    def _train(self, model, optimizer, lr_scheduler, epoch, progress_bar):
        model.train()
        batch_i = 0
        for batch in self.dataloaders["train"]:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)

            # BCE Focal Loss
            BCE_loss = F.binary_cross_entropy_with_logits(
                outputs.logits, labels.float(), reduction="none"
            )
            pt = torch.exp(-BCE_loss)
            loss = (
                self.cfg.trainer.loss_alpha
                * (1 - pt) ** self.cfg.trainer.loss_gamma
                * BCE_loss
            )

            # Class balancing
            loss = loss * (
                labels * self.cfg.trainer.loss_alpha
                + (1 - labels) * (1 - self.cfg.trainer.loss_alpha)
            )
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            batch_i += 1
            progress_bar.set_description(
                f"Epoch {epoch} ({int((batch_i/len(self.dataloaders['train'])* 100))}%)"
            )

    def _evaluate(self, model, split="dev", cl_report=False):
        model.eval()
        all_logits = []
        all_labels = []
        progress_bar = tqdm(range(len(self.dataloaders[split])))
        progress_bar.set_description(f"testing {split}")
        for batch in self.dataloaders[split]:
            batch = {k: v.to(self.cfg.device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.no_grad():
                outputs = model(**batch)

            all_logits.append(outputs.logits)
            all_labels.append(labels)

            progress_bar.update(1)

        return compute_metrics(
            torch.cat(all_logits, dim=0),
            torch.cat(all_labels, dim=0),
            self.cfg.label_scheme if cl_report else None,
        )

    def finetune(self):
        print("Fine-tuning")

        num_training_steps = self.cfg.trainer.epochs * len(self.dataloaders["train"])

        model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model.name, num_labels=self.cfg.num_labels
        ).to(self.cfg.device)

        optimizer = AdamW(model.parameters(), lr=self.cfg.trainer.learning_rate)

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
            self._train(model, optimizer, lr_scheduler, epoch + 1, progress_bar)
            metrics = self._evaluate(model)
            print(metrics)
            patience_metric = metrics[self.cfg.trainer.best_model_metric]
            if patience_metric > best_score:
                best_score = patience_metric
                best_epoch = epoch
                self._checkpoint(model)
            elif epoch - best_epoch > self.cfg.trainer.patience:
                print("Early stopped training at epoch %d" % epoch)
                break

        print("Testing")
        self._resume(model)
        print(self._evaluate(model, "test", True))
