from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
import json

# Arguments
parser = ArgumentParser()
parser.add_argument("--file", type=str, default="")
options = parser.parse_args()


# Get data
df = pd.read_csv(
    options.file, sep="\t", names=["language", "labels", "embeddings", "words"]
)

# Preprocess
df["embeddings"] = df["embeddings"].apply(
    lambda x: np.array([float(y) for y in x.split()])
)
# df["words"] = df["words"].apply(lambda x: json.loads(x))
df["label"] = df["labels"].str.split(" ")
df = df.explode("label")
df["language_label"] = df["language"] + "_" + df["label"]
df = df.sort_values(["language", "label"])
import ast

print(df.head(5))
label_words = {}
for name, group in df.groupby("label"):
    if name not in label_words:
        label_words[name] = {}
        for ind in group.index:
            words = ast.literal_eval(json.loads(group["words"][ind]))

            for k, v in words.items():
                if k not in label_words[name]:
                    label_words[name][k] = []
                label_words[name][k].append(v)


for k, v in label_words.items():
    print(k)


# Initialize a dictionary to store the best words for each dictionary
best_data = {}

# Iterate through each dictionary
for label, word_list in label_words.items():
    best_words = {}

    # Iterate through the words and scores in the current dictionary
    for word, score in word_list.items():
        is_best = True  # Assume the word is the best until proven otherwise
        mean_score = np.mean(score)

        if len(score) < 5:
            continue

        # Compare the current word's score with other dictionaries
        for other_label, other_word_list in label_words.items():
            if (
                other_label != label
                and word in other_word_list
                and np.mean(other_word_list[word]) >= mean_score
            ):
                is_best = False  # There is a word with a higher or equal score in another dictionary
                break

        if is_best:
            best_words[word] = score

    best_data[label] = best_words

# Print the resulting best data
# for key, sub_dict in best_data.items():
#    print(f"{key}: {sub_dict}")


"""
# groups = df.groupby(["language_label"])["embeddings"].mean()

import csv

with open(
    f"output/fi_fi/_Users_erikhenriksson_temp/class_embeddings.tsv", "w", newline=""
) as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
    for name, group in df.groupby("language_label"):
        m = df["embeddings"].mean()
        writer.writerow(
            [
                name,
                " ".join([str(x) for x in m.tolist()]),
            ]
        )
"""

d = best_data["ra"]
# d = label_words["nb"]

sorted_best_data = {
    key: dict(sorted(val.items(), key=lambda item: np.mean(item[1]), reverse=True)[:20])
    for key, val in best_data.items()
}

for label, words in sorted_best_data.items():
    print(f"Register: {label}")
    print(", ".join(words.keys()))
    print()


from collections import Counter

### STATS

# Convert the dictionary values (lists) to NumPy arrays for easier computation
data_arrays = {word: np.array(values) for word, values in d.items()}

# Calculate the maximum number of occurrences
max_occurrences = max(len(values) for values in data_arrays.values())

# Calculate the mean length of the lists
mean_length = np.mean([len(values) for values in data_arrays.values()])

# Calculate the mean values for each word
mean_values = {word: np.mean(values) for word, values in data_arrays.items()}

# Print the results
print(f"Maximum number of occurrences: {max_occurrences}")
print(f"Mean length of the lists: {mean_length}")


# d = {k: v for k, v in d.items() if len(v) >= 10}

# Sort the dictionary by mean in descending order and get the first 10 items
sorted_d = dict(sorted(d.items(), key=lambda item: np.mean(item[1]), reverse=True)[:20])

# Plotting
plt.figure(figsize=(20, 10))
plt.boxplot(sorted_d.values(), labels=sorted_d.keys())
plt.title("Boxplot of Top 10 Lists by Mean Value")
plt.ylabel("Values")
plt.show()


"""
d = label_words["HI"]
d = {k: v for k, v in d.items() if len(v) >= 10}

# Calculate the mean of each list
means = {key: np.mean(value) for key, value in d.items()}

# Sort the dictionary by mean in descending order and get the first 10 items
sorted_d = dict(
    sorted(d.items(), key=lambda item: np.mean(item[1]), reverse=True)[:300]
)

# Plotting
plt.figure(figsize=(20, 10))
plt.boxplot(sorted_d.values(), labels=sorted_d.keys())
plt.title("Boxplot of Top 10 Lists by Mean Value")
plt.ylabel("Values")
plt.show()
"""
