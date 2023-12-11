from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

# Arguments
parser = ArgumentParser()
parser.add_argument("--file", type=str, default="")
options = parser.parse_args()

# Common variables
reducer = umap.UMAP()
color_palette = sns.color_palette("Paired", 1000)

# Get data
df = pd.read_csv(
    options.file, sep="\t", names=["language", "labels", "embeddings", "words"]
)

# Preprocess
df["embeddings"] = df["embeddings"].apply(
    lambda x: np.array([float(y) for y in x.split()])
)
df["label"] = df["labels"].str.split(" ")
df = df.explode("label")
df["language_label"] = df["language"] + "_" + df["label"]
df = df.sort_values(["language", "label"])

# Get UMAP embeddings

scaled_embeddings = StandardScaler().fit_transform(df["embeddings"].tolist())
red_embedding = reducer.fit_transform(scaled_embeddings)

# Add embeddings to dataframe
df["x"] = red_embedding[:, 0]
df["y"] = red_embedding[:, 1]

label_i = 0
last_lan = ""

for language_label, group in df.groupby("language_label"):
    language = language_label[:2]
    if language != last_language:
        label_i = 0
    last_language = language_label[:2]
    marker = "*" if language == "en" else "."
    plt.scatter(
        group.x,
        group.y,
        s=30,
        marker=marker,
        label=language_label,
        edgecolor="none",
        c=[color_palette[label_i]],
    )
    label_i += 1

lgnd = plt.legend()
for handle in lgnd.legend_handles:
    handle._sizes = [60]

plt.grid(True)
plt.show()
