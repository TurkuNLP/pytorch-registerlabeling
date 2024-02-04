import csv
import sys

from datasets import Dataset

csv.field_size_limit(sys.maxsize)

from datasets import Dataset, DatasetDict

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
    "tr",
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


def split_gen(split, languages, label_cfg, concat_small):
    row_id = 0
    for l in languages.split("-"):
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
                            "text": text,
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

    make_generator = lambda split, target: Dataset.from_generator(
        split_gen,
        gen_kwargs={
            "split": split,
            "languages": target,
            "label_cfg": cfg.data.labels,
            "concat_small": cfg.data.concat_small,
        },
        cache_dir=cfg.working_dir_root + "/tokens_cache",
    )

    splits = {}

    if train:
        splits["train"] = make_generator("train", train)
    if dev:
        splits["dev"] = make_generator("dev", dev)
    splits["test"] = make_generator("test", test)

    return DatasetDict(splits)


def preprocess_data(dataset, tokenizer, cfg):
    dataset = dataset.shuffle(seed=cfg.seed)
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
    dataset.set_format("torch")
    return dataset
