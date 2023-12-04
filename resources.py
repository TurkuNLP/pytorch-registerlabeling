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


def balance_languages(df):
    adjusted_df = pd.DataFrame()

    for label in df["label_text"].unique():
        label_df = df[df["label_text"] == label]

        # Count the occurrences for each language for the current label
        lang_counts = label_df["language"].value_counts()

        # Skip if there's only one language for this label
        if len(lang_counts) <= 1:
            adjusted_df = pd.concat([adjusted_df, label_df])
            continue

        # Find the most frequent language and its count
        most_freq_lang = lang_counts.idxmax()
        most_freq_count = lang_counts.max()

        # Calculate the mean count excluding the most frequent language
        mean_count_excluding_most = lang_counts[lang_counts != most_freq_count].mean()

        # Downsample the most frequent language if it exceeds 1.5 times the mean
        if most_freq_count > 2 * mean_count_excluding_most:
            downsample_count = int(2 * mean_count_excluding_most)
            most_freq_lang_rows = label_df[label_df["language"] == most_freq_lang]
            downsampled_rows = most_freq_lang_rows.sample(
                n=downsample_count, random_state=0
            )
            adjusted_df = pd.concat(
                [
                    adjusted_df,
                    downsampled_rows,
                    label_df[label_df["language"] != most_freq_lang],
                ]
            )
        else:
            adjusted_df = pd.concat([adjusted_df, label_df])

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
                balance_languages(pd.DataFrame(dataset[split]))
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

    plt.subplots_adjust(bottom=0.2)

    # Annotate plot with parent categories
    for parent, children in parent_categories.items():
        mid_index = calculate_mid_index(children)
        # Place parent category label at the bottom of the plot
        plt.text(mid_index, 0, parent, ha="center", va="bottom", transform=ax.transData)

        # Draw a horizontal line to group children categories (optional)
        start = plot_data.index.get_loc(children[0])
        end = plot_data.index.get_loc(children[-1]) + 1
        plt.hlines(
            0,
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


from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader
from torch.utils.data import Sampler


class CustomBalancedLanguageSampler(Sampler):
    def __init__(self, dataset, batch_size, language_info):
        self.dataset = dataset
        self.batch_size = batch_size
        self.language_info = language_info  # language of each sample
        self.indices_per_language = self._create_indices_per_language()
        self.smallest_set_size = (
            min(len(indices) for indices in self.indices_per_language.values())
            // batch_size
        )
        self.num_languages = len(self.indices_per_language)
        self.total_batches = self.smallest_set_size * self.num_languages

    def _create_indices_per_language(self):
        # Map each language to the indices of samples in that language.
        indices_per_language = {}
        for idx, language in enumerate(self.language_info):
            if language not in indices_per_language:
                indices_per_language[language] = []
            indices_per_language[language].append(idx)
        return indices_per_language


def __iter__(self):
    language_order = list(self.indices_per_language.keys())

    for _ in range(self.smallest_set_size):
        np.random.shuffle(language_order)
        for language in language_order:
            batch_size_for_language = min(
                len(self.indices_per_language[language]), self.batch_size
            )

            if batch_size_for_language > 0:
                language_indices = np.random.choice(
                    self.indices_per_language[language],
                    size=batch_size_for_language,
                    replace=False,
                )
                yield language_indices

            # Reshuffle and replenish if necessary
            if len(self.indices_per_language[language]) < self.batch_size:
                self.indices_per_language[
                    language
                ] = self._create_indices_per_language()[language]
                np.random.shuffle(self.indices_per_language[language])

    def __len__(self):
        return self.total_batches


def custom_train_dataloader(self) -> DataLoader:
    language_info = [sample["language"] for sample in self.train_dataset]
    train_dataset = self._remove_unused_columns(
        self.train_dataset, description="training"
    )

    batch_size = self._train_batch_size
    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": self.data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "sampler": CustomBalancedLanguageSampler(
            train_dataset, batch_size, language_info
        ),
        "drop_last": self.args.dataloader_drop_last,
        "worker_init_fn": seed_worker,
    }

    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
