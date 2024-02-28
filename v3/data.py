import csv
import sys

import numpy as np

from datasets import Dataset

csv.field_size_limit(sys.maxsize)

from datasets import Dataset, DatasetDict, concatenate_datasets
from skmultilearn.model_selection import IterativeStratification

from .labels import binarize_labels, normalize_labels

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

language_names = {
    "ar": "Arabic",
    "ca": "Catalan",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "jp": "Japanese",
    "no": "Norwegian",
    "pt": "Portuguese",
    "tr": "Turkish",
    "ur": "Urdu",
    "zh": "Chinese",
}


def split_gen(splits, languages, label_cfg, concat_small, prefix=""):
    row_id = 0
    for l in languages.split("-"):
        for split in splits:
            concat = concat_small and l in small_languages
            with open(
                f"data/{l}/{l if concat else (split if l not in small_languages else l)}.tsv",
                "r",
            ) as c:
                re = csv.reader(c, delimiter="\t")
                for ro in re:
                    if ro[0] and ro[1]:
                        normalized_labels = normalize_labels(ro[0], label_cfg)
                        text = ro[1]
                        label = binarize_labels(normalized_labels, label_cfg)
                        label_text = " ".join(normalized_labels)

                        if label_text:
                            yield {
                                "label": label,
                                "label_text": label_text,
                                "language": "small" if concat else l,
                                "text": prefix + text,
                                "id": str(row_id),
                                "split": split,
                                "length": len(text),
                            }
                            row_id += 1


def get_dataset(cfg):
    train, dev, test = cfg.data.train, cfg.data.dev, cfg.data.test
    if cfg.method == "predict":
        if train and not test:
            test = train
        train = None
    else:
        if not dev:
            dev = train
        if not test:
            test = dev

    make_generator = lambda splits, target: Dataset.from_generator(
        split_gen,
        gen_kwargs={
            "splits": splits,
            "languages": target,
            "label_cfg": cfg.data.labels,
            "concat_small": cfg.data.concat_small,
            "prefix": cfg.data.text_prefix,
        },
        cache_dir=cfg.working_dir_root + "/tokens_cache",
    )

    splits = {}

    if cfg.data.use_fold:
        data_to_be_folded = list(
            make_generator(["train", "dev"], train).shuffle(seed=cfg.seed)
        )
        y = np.array([x["label"] for x in data_to_be_folded])
        X = np.array(list(range(len(y))))

        k_fold = IterativeStratification(n_splits=10, order=1)
        folds = [fold for _, fold in k_fold.split(X, y)]
        dev_fold_n = cfg.data.use_fold - 1

        dev_idx = folds.pop(dev_fold_n)
        train_idx = [item for sublist in folds for item in sublist]

        splits["dev"] = Dataset.from_list([data_to_be_folded[int(i)] for i in dev_idx])
        splits["train"] = Dataset.from_list(
            [data_to_be_folded[int(i)] for i in train_idx]
        )
        splits["test"] = make_generator(["test"], test)

    else:
        if train:
            splits["train"] = make_generator(["train"], train)
        if dev:
            splits["dev"] = make_generator(["dev"], dev)
        splits["test"] = make_generator(["test"], test)

        if cfg.data.test_all_data:
            train_to_test = make_generator(["train"], test)
            dev_to_test = make_generator(["dev"], test)

            splits["test"] = concatenate_datasets(
                [splits["test"], train_to_test, dev_to_test]
            )

        if cfg.data.use_augmented_data:
            augmented_to_train = make_generator(["train_aug"], train)
            splits["train"] = concatenate_datasets(
                [splits["train"], augmented_to_train]
            )

    return DatasetDict(splits)


def preprocess_data(dataset, tokenizer, cfg):
    dataset = dataset.shuffle(seed=cfg.seed)
    if cfg.data.use_inference_time_test_data:
        dataset["test"] = dataset["test"].select(range(1000))
    if not cfg.model.sentence_transformer:
        dataset = dataset.map(
            lambda example: tokenizer(
                example["text"],
                truncation=True,
                max_length=cfg.data.max_length,
                padding="max_length" if cfg.data.no_dynamic_padding else False,
            ),
            batched=True,
        )
    if cfg.data.remove_unused_cols:
        dataset = dataset.remove_columns(
            ["label_text", "text", "id", "split", "length"]
        )
    dataset = dataset.rename_column("label", "labels")
    if not "setfit" in cfg.method:
        dataset.set_format("torch", device=cfg.device)
    return dataset
