from datasets import concatenate_datasets
import pandas as pd
from ..data import get_dataset


def run(cfg):
    dataset = get_dataset(cfg)
    dataset = concatenate_datasets([dataset["train"], dataset["dev"], dataset["test"]])

    df = pd.DataFrame(dataset)

    # Calculate the total number of texts in each language
    language_counts = df["language"].value_counts()

    # Print the counts
    print(language_counts)
    # Get the total count of all texts
    total_count = language_counts.sum()

    # Print the total count
    print("\nTotal count of texts:")
    print(total_count)
