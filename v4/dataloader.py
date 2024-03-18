from torch.utils.data import DataLoader
from transformers.trainer_utils import seed_worker

from .sampler import BalancedLanguageSampler


def balanced_dataloader(self, split, b) -> DataLoader:
    dataset = self.train_dataset if split == "train" else self.eval_dataset
    language_data = [sample["language"] for sample in dataset]
    dataset = self._remove_unused_columns(
        dataset,
        description=split,
    )

    sampler = BalancedLanguageSampler

    batch_size = self._train_batch_size if split == "train" else b
    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": self.data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "sampler": sampler(language_data),
        "drop_last": self.args.dataloader_drop_last,
        "worker_init_fn": seed_worker,
    }

    return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))
