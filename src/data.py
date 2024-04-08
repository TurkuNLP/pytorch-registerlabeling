import csv
import sys
import gzip

csv.field_size_limit(sys.maxsize)
from itertools import cycle

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import DataLoader, Sampler
from transformers.trainer_utils import seed_worker

from .labels import binarize_labels, normalize_labels


class BalancedLanguageSampler(Sampler):
    def __init__(self, language_data, size="smallest", lang_cycle="random"):
        self.language_data = language_data
        self.indices_per_language = self._create_indices_per_language()
        language_indice_lengths = [
            len(indices) for indices in self.indices_per_language.values()
        ]
        if size == "smallest":
            dataset_size = min(language_indice_lengths)
        elif size == "mean":
            dataset_size = int(np.mean(language_indice_lengths))

        self.epoch_size = dataset_size * len(self.indices_per_language)
        self.cycle = lang_cycle

    def _create_indices_per_language(self):
        indices_per_language = {lang: [] for lang in set(self.language_data)}
        for idx, lang in enumerate(self.language_data):
            indices_per_language[lang].append(idx)
        return indices_per_language

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        language_cycle = cycle(self.indices_per_language.keys())
        for _ in range(self.epoch_size):
            if self.cycle == "random":
                language = np.random.choice(list(self.indices_per_language.keys()))
            elif self.cycle == "cycle":
                language = next(language_cycle)

            # Replenish the indices for the language if necessary
            if not self.indices_per_language[language]:
                self.indices_per_language[language] = [
                    idx
                    for idx, lang in enumerate(self.language_data)
                    if lang == language
                ]

            idx = np.random.choice(self.indices_per_language[language], replace=False)
            self.indices_per_language[language].remove(idx)
            yield idx


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
    "ur",
    "zh",
]


def gen(languages, split, label_scheme):
    for l in languages:
        file_path = f"data/{l}/{split if l not in small_languages else l}.tsv.gz"
        with gzip.open(file_path, "rt", encoding="utf-8") as c:
            re = csv.reader(c, delimiter="\t")
            for ro in re:
                if not (ro[0] and ro[1]):
                    continue
                normalized_labels = normalize_labels(ro[0], label_scheme)
                if not normalized_labels:
                    continue

                yield {
                    "label": binarize_labels(normalized_labels, label_scheme),
                    "text": ro[1],
                    "language": l,
                }


def get_dataset(cfg, tokenizer):
    generate = lambda split: Dataset.from_generator(
        gen,
        gen_kwargs={
            "languages": dict(cfg)[split].split("-"),
            "split": split,
            "label_scheme": cfg.labels,
        },
    )
    splits = {}

    if cfg.use_fold:
        data_to_be_folded = list(
            concatenate_datasets([generate("train"), generate("dev")]).shuffle(
                seed=cfg.seed
            )
        )

        y = np.array([x["label"] for x in data_to_be_folded])
        k_fold_fn = IterativeStratification(n_splits=cfg.num_folds, order=1)

        train_fold, dev_fold = list(k_fold_fn.split(list(range(len(y))), y))[
            cfg.use_fold - 1
        ]

        splits["train"] = Dataset.from_list(
            [data_to_be_folded[int(i)] for i in train_fold]
        )
        splits["dev"] = Dataset.from_list([data_to_be_folded[int(i)] for i in dev_fold])
        splits["test"] = generate("test")

    else:
        include_splits = ["train", "dev", "test"] if cfg.method == "train" else ["test"]
        for s in include_splits:
            splits[s] = generate(s)

    dataset = DatasetDict(splits).shuffle(seed=cfg.seed)

    return dataset.map(
        lambda example: tokenizer(
            example["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        ),
        remove_columns=(["text"]),
        batched=True,
    )
