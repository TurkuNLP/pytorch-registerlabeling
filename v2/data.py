import csv
import sys
import ast

import pandas as pd

csv.field_size_limit(sys.maxsize)

from datasets import Dataset, DatasetDict

from .labels import (
    binarize_labels,
    normalize_labels,
)

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


def get_dataset(train, test, label_config, few_shot=False):
    def data_gen(ls, split):
        row_id = 0
        for l in ls.split("-"):
            use_split = split
            if l in small_languages:
                if split == "dev":
                    use_split = "test"
                elif not (l in train.split("-")):
                    use_split = l

            with open(f"data/{l}/{use_split}.tsv", "r") as c:
                re = csv.reader(c, delimiter="\t")
                for ro in re:
                    if ro[0] and ro[1]:
                        normalized_labels = normalize_labels(ro[0], label_config)
                        text = ro[1]
                        label = binarize_labels(normalized_labels, label_config)
                        label_text = " ".join(normalized_labels)

                        if label_text:
                            yield {
                                "label": label,
                                "label_text": label_text,
                                "language": l,
                                "text": text,
                                "id": str(row_id),
                                "split": split,
                                "length": len(text),
                            }
                            row_id += 1

    dataset = DatasetDict(
        {
            "train": Dataset.from_generator(
                data_gen, gen_kwargs={"ls": train, "split": "train"}
            ),
            "dev": Dataset.from_generator(
                data_gen, gen_kwargs={"ls": train, "split": "dev"}
            ),
            "test": Dataset.from_generator(
                data_gen, gen_kwargs={"ls": test, "split": "test"}
            ),
        }
    )

    if few_shot:

        def sample_group(group, random_state=42):
            n = min(len(group), few_shot)
            return group.sample(n, random_state=random_state)

        for split in ["train", "dev", "test"]:
            dataset[split] = Dataset.from_pandas(
                pd.DataFrame(dataset[split])
                .groupby(["language", "label_text"])
                .apply(sample_group)
                .reset_index(drop=True)
            )

    return dataset


def get_gemini_data(lang):
    def convert_to_list(text):
        l = ast.literal_eval(text)
        if len(l) not in [25, 768]:
            print("er")
        return l

    df_dev = pd.read_csv(
        f"google_embeddings/{lang}_dev.csv",
    ).dropna(subset=["Embeddings", "label"])
    df_test = pd.read_csv(
        f"google_embeddings/{lang}_test.csv",
    ).dropna(subset=["Embeddings", "label"])
    df_train = pd.read_csv(
        f"google_embeddings/{lang}_train.csv",
    ).dropna(subset=["Embeddings", "label"])

    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "input_ids": df_train["Embeddings"].apply(convert_to_list),
                    "label": df_train["label"].apply(convert_to_list),
                }
            ),
            "dev": Dataset.from_dict(
                {
                    "input_ids": df_dev["Embeddings"].apply(convert_to_list),
                    "label": df_dev["label"].apply(convert_to_list),
                }
            ),
            "test": Dataset.from_dict(
                {
                    "input_ids": df_test["Embeddings"].apply(convert_to_list),
                    "label": df_test["label"].apply(convert_to_list),
                }
            ),
        }
    )
