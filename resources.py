import csv
import sys

import pandas as pd

csv.field_size_limit(sys.maxsize)

from datasets import Dataset, DatasetDict

from labels import (
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


def downsample_most_frequent_language(df):
    # Group by label_text and language and count the occurrences
    counts = df.groupby(["label_text", "language"]).size().reset_index(name="count")

    # Initialize an empty DataFrame to store the adjusted data
    adjusted_df = pd.DataFrame()

    for label in df["label_text"].unique():
        # Filter counts for the current label
        label_counts = counts[counts["label_text"] == label]

        # Sort by count and get the top 2 languages
        top_languages = label_counts.sort_values(by="count", ascending=False).head(2)

        if len(top_languages) > 1:
            # Get the count for the second most frequent language
            second_most_count = top_languages.iloc[1]["count"]

            # Downsample the most frequent language
            most_freq_lang = top_languages.iloc[0]["language"]
            most_freq_lang_rows = df[
                (df["label_text"] == label) & (df["language"] == most_freq_lang)
            ]
            downsampled_rows = most_freq_lang_rows.sample(
                n=second_most_count, random_state=0
            )

            # Add the downsampled rows and all other rows for the label to the adjusted DataFrame
            other_rows = df[
                (df["label_text"] == label) & (df["language"] != most_freq_lang)
            ]
            adjusted_df = pd.concat([adjusted_df, downsampled_rows, other_rows])
        else:
            # If there's only one language for the label, just add it to the adjusted DataFrame
            adjusted_df = pd.concat([adjusted_df, df[df["label_text"] == label]])

    merged_df = (
        df.groupby("id")
        .agg(
            {
                "text": "first",
                "label": "first",
                "language": "first",
                "label_text": " ".join,
            }
        )
        .reset_index()
    )

    return adjusted_df


def get_dataset(train, test, downsample, label_config):
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
                    if ro[1]:
                        normalized_labels = normalize_labels(ro[0], label_config)
                        text = ro[1]
                        label = binarize_labels(normalized_labels, label_config)
                        label_text = " ".join(normalized_labels)
                        yield {
                            "label": label,
                            "label_text": label_text,
                            "language": l,
                            "text": text,
                            "id": str(row_id),
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

    if downsample:
        print("Downsampling...")
        for split in ["train", "dev", "test"]:
            len_before = len(dataset[split])

            dataset[split] = Dataset.from_pandas(
                downsample_most_frequent_language(pd.DataFrame(dataset[split]))
            )
            len_after = len(dataset[split])

            print(
                f"{split}: {len_before} -> {len_after} ({round(len_after/len_before, 2)*100}%)"
            )

    return dataset
