import csv
import sys

csv.field_size_limit(sys.maxsize)
from datasets import Dataset, DatasetDict

from .labels import normalize_labels, binarize_labels

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


def gen(languages, split, label_scheme, text_prefix):
    for l in languages:
        with open(
            f"data/{l}/{split if l not in small_languages else l}.tsv",
            "r",
        ) as c:
            re = csv.reader(c, delimiter="\t")
            for ro in re:
                if not (ro[0] and ro[1]):
                    continue
                normalized_labels = normalize_labels(ro[0], label_scheme)
                if not normalized_labels:
                    continue

                yield {
                    "label": binarize_labels(normalized_labels, label_scheme),
                    "text": text_prefix + ro[1],
                    "language": l,
                }


def get_dataset(cfg, tokenizer):
    generate = lambda split: Dataset.from_generator(
        gen,
        cache_dir="./hf_results/tokens_cache",
        gen_kwargs={
            "languages": dict(cfg)[split].split("-"),
            "split": split,
            "label_scheme": cfg.labels,
            "text_prefix": cfg.text_prefix,
        },
    )
    splits = {}
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
