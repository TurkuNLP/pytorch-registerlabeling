import csv
import string
import sys
import gzip
import seaborn as sns

csv.field_size_limit(sys.maxsize)
from itertools import cycle
import random
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data import DataLoader, Sampler
from transformers.trainer_utils import seed_worker

from .labels import binarize_labels, normalize_labels

language_names = {
    "en": "English",
    "fi": "Finnish",
    "fr": "French",
    "sv": "Swedish",
    "tr": "Turkish",
    "ar": "Arabic",
    "ca": "Catalan",
    "es": "Spanish",
    "fa": "Persian",
    "hi": "Hindi",
    "id": "Indonesian",
    "jp": "Japanese",
    "no": "Norwegian",
    "pt": "Portuguese",
    "ur": "Urdu",
    "zh": "Chinese",
}


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

file_opener = lambda file_path, use_gz: (
    gzip.open(file_path + ".gz", "rt", encoding="utf-8")
    if use_gz
    else open(file_path, "r")
)


def gen(languages, split, label_scheme, use_gz):
    for l in languages:

        with file_opener(
            f"data/{l}/{split if l not in small_languages else l}.tsv", use_gz
        ) as c:
            idx = -1
            row = -1
            re = csv.reader(c, delimiter="\t")
            for ro in re:
                row += 1
                if not (ro[0] and ro[1]):
                    continue
                normalized_labels = normalize_labels(ro[0], label_scheme)

                if not normalized_labels:
                    continue

                idx += 1
                yield {
                    "idx": idx,
                    "row": row,
                    "label": binarize_labels(normalized_labels, label_scheme),
                    "label_text": " ".join(normalized_labels),
                    "text": ro[1],
                    "language": l,
                }


def get_dataset(cfg, tokenizer=None):
    def generate(split):
        kwargs = {
            "gen_kwargs": {
                "languages": dict(cfg)[split].split("-"),
                "split": split,
                "label_scheme": cfg.labels,
                "use_gz": cfg.use_gz,
            }
        }

        if hasattr(cfg, "cachedir") and cfg.cachedir:
            kwargs["cache_dir"] = cfg.cachedir

        return Dataset.from_generator(
            gen,
            **kwargs,
        )

    splits = {}

    cfg.train = "-".join([s for s in cfg.train.split("-") if s not in small_languages])
    cfg.dev = "-".join([s for s in cfg.dev.split("-") if s not in small_languages])

    if cfg.sample_subset:

        data_to_be_folded = list(
            generate("train").shuffle(
                seed=cfg.seed
            )
        )

        y = np.array([x["label"] for x in data_to_be_folded])

        # We take cfg.sample_subset samples per run
        n_splits = len(data_to_be_folded) // (cfg.sample_subset)

        k_fold_fn = IterativeStratification(n_splits=n_splits, order=1)

        folds = list(k_fold_fn.split(list(range(len(y))), y))
        print(len(folds))
        print(len(folds[0]))
        exit()

        if not cfg.just_evaluate:
            splits["train"] = Dataset.from_list(
                [data_to_be_folded[int(i)] for i in train_fold]
            )
            splits["dev"] = generate("dev")
        splits["test"] = generate("test")

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
        if not cfg.just_evaluate:
            splits["train"] = Dataset.from_list(
                [data_to_be_folded[int(i)] for i in train_fold]
            )
            splits["dev"] = Dataset.from_list(
                [data_to_be_folded[int(i)] for i in dev_fold]
            )
        splits["test"] = generate("test")

    else:
        include_splits = ["train", "dev", "test"] if not cfg.just_evaluate else ["test"]
        for s in include_splits:
            splits[s] = generate(s)

    dataset = DatasetDict(splits).shuffle(seed=cfg.seed)

    if hasattr(cfg, "skip_tokenize") and cfg.skip_tokenize:
        return dataset

    def process(examples):
        tokenized_texts = tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

        if cfg.mask_alphabets:
            # Retrieve the token IDs for unknown and punctuation tokens
            unk_token_id = tokenizer.unk_token_id

            # Initialize new_input_ids to hold the processed batches
            new_input_ids = []

            # Iterate over each set of input_ids in the batch
            for input_ids in tokenized_texts["input_ids"]:
                # Convert input_ids to tokens
                tokens = tokenizer.convert_ids_to_tokens(input_ids)

                # Replace non-punctuation tokens with the <unk> token ID
                processed_ids = [
                    (
                        unk_token_id
                        if not (
                            token in string.punctuation
                            or token in tokenizer.all_special_tokens
                        )
                        else token_id
                    )
                    for token, token_id in zip(tokens, input_ids)
                ]

                # Append the processed IDs to the new_input_ids list
                new_input_ids.append(processed_ids)

            # Update the input_ids in tokenized_batch with the new IDs
            tokenized_texts["input_ids"] = new_input_ids

            tokens = tokenizer.convert_ids_to_tokens(tokenized_texts["input_ids"][0])

        return tokenized_texts

    return dataset.map(
        process,
        remove_columns=(
            cfg.remove_columns.split(",")
            if hasattr(cfg, "remove_columns") and cfg.remove_columns
            else None
        ),
        batched=True,
    )
