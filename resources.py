import csv
import sys

import numpy as np

import pandas as pd

csv.field_size_limit(sys.maxsize)

from datasets import Dataset, DatasetDict
import matplotlib.pyplot as plt

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
        adjusted_df.groupby("id")
        .agg(
            {
                "text": "first",
                "label": "first",
                "language": "first",
                "length": "first",
                "split": "first",
                "label_text": " ".join,
            }
        )
        .reset_index()
    )

    return merged_df


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


def get_statistics(dataset):
    df = pd.concat(
        [pd.DataFrame(dataset[split]) for split in dataset], ignore_index=True
    )[["label_text", "language", "length"]]

    print(df.head(5))
    df["label_text"] = df["label_text"].str.split(" ")
    df = df.explode("label_text")

    # Parent and label mappings
    parent_categories = {
        "SP": ["it", "os"],
        "NA": ["ne", "sr", "nb", "on"],
        "HI": ["re", "oh"],
        "IN": ["en", "ra", "dtp", "fi", "lt", "oi"],
        "OP": ["rv", "ob", "rs", "av", "oo"],
        "IP": ["ds", "ed", "oe"],
    }
    label_mapping = {
        "MT": "Machine tr.",
        "LY": "Lyrical",
        "it": "Interview",
        "os": "Other",
        "ID": "Discussion",
        "ne": "News report",
        "sr": "Sports report",
        "nb": "Narrative blog",
        "on": "Other",
        "re": "Recipe",
        "oh": "Other",
        "en": "Encyclopedia article",
        "ra": "Research article",
        "dtp": "Description",
        "fi": "FAQ",
        "lt": "Legal",
        "oi": "Other",
        "rv": "Review",
        "ob": "Opinion blog",
        "rs": "Religious",
        "av": "Advice",
        "oo": "Other",
        "ds": "Sales promotion",
        "ed": "Informed persuasion",
        "oe": "Other",
    }

    # Function to apply the mapping
    def map_label(label):
        return label_mapping[label]

    # Sample DataFrame
    # df = ...

    # Filter out parent categories
    df = df[~df["label_text"].isin(parent_categories.keys())]

    # Group and plot data
    plot_data = df.groupby(["label_text", "language"]).size().unstack(fill_value=0)
    plot_data = plot_data.reindex(label_mapping.keys())

    # Plot and get the axes object
    ax = plot_data.plot(kind="bar", stacked=True)

    # Set custom x-tick labels
    ax.set_xticklabels(
        [map_label(label.get_text()) for label in ax.get_xticklabels()],
        rotation=25,
        ha="right",
    )

    # Function to calculate middle index of each parent category group
    def calculate_mid_index(categories):
        indices = [plot_data.index.get_loc(cat) for cat in categories]
        return np.mean(indices)

    plt.subplots_adjust(bottom=2)

    # Annotate plot with parent categories
    for parent, children in parent_categories.items():
        mid_index = calculate_mid_index(children)
        # Place parent category label at the bottom of the plot
        plt.text(
            mid_index, -20, parent, ha="center", va="bottom", transform=ax.transData
        )

        # Draw a horizontal line to group children categories (optional)
        start = plot_data.index.get_loc(children[0])
        end = plot_data.index.get_loc(children[-1]) + 1
        plt.hlines(
            -20,
            start - 0.5,
            end - 0.5,
            colors="gray",
            linestyles="dashed",
            transform=ax.transData,
        )

    plt.xlabel("Label Text")
    plt.ylabel("Count")
    plt.title("Register labels")
    plt.show()

    exit()

    parent_categories = {
        "SP": ["it", "os"],
        "NA": ["ne", "sr", "nb", "on"],
        "HI": ["re", "oh"],
        "IN": ["en", "ra", "dtp", "fi", "lt", "oi"],
        "OP": ["rv", "ob", "rs", "av", "oo"],
        "IP": ["ds", "ed", "oe"],
    }
    label_mapping = {
        "MT": "Machine tr.",
        "LY": "Lyrical",
        "it": "Interview",
        "os": "Other",
        "ID": "Discussion",
        "ne": "News report",
        "sr": "Sports report",
        "nb": "Narrative blog",
        "on": "Other",
        "re": "Recipe",
        "oh": "Other",
        "en": "Encyclopedia article",
        "ra": "Research article",
        "dtp": "Description",
        "fi": "FAQ",
        "lt": "Legal",
        "oi": "Other",
        "rv": "Review",
        "ob": "Opinion blog",
        "rs": "Religious",
        "av": "Advice",
        "oo": "Other",
        "ds": "Sales promotion",
        "ed": "Informed persuasion",
        "oe": "Other",
    }

    # Function to apply the mapping
    def map_label(label):
        return label_mapping[label]

    df = df[~df["label_text"].isin(parent_categories.keys())]

    plot_data = df.groupby(["label_text", "language"]).size().unstack(fill_value=0)
    plot_data = plot_data.reindex(label_mapping.keys())
    print(plot_data)

    # Plot and get the axes object
    ax = plot_data.plot(kind="bar", stacked=True)

    plt.xlabel("Label Text")
    plt.ylabel("Count")
    plt.title("Register labels")

    # Apply the mapping to x-tick labels
    ax.set_xticklabels(
        [map_label(label.get_text()) for label in ax.get_xticklabels()],
        rotation=25,
        ha="right",
    )
    plt.show()
