import csv
import labels
import sys

import pandas as pd
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

languages = "fi-fr-en-sv"


def data_gen(ls, split):
    for l in ls.split("-"):
        use_split = split

        with open(f"data/{l}/{use_split}.tsv", "r") as c:
            re = csv.reader(c, delimiter="\t")
            for ro in re:
                if ro[1]:
                    normalized_labels = labels.normalize_labels(ro[0], "xgenre")
                    text = ro[1]
                    label = labels.binarize_labels(normalized_labels, "xgenre")
                    label_text = " ".join(normalized_labels)
                    yield {
                        "label": label,
                        "label_text": label_text,
                        "language": l,
                        "text": text,
                    }


data = data_gen("en-fi-fr-sv", "train")

data = list(data)

print(data[0])

# Convert to DataFrame
df = pd.DataFrame(data)
df["label_text"] = df["label_text"].str.split(" ")
df = df.explode("label_text")
# Prepare data for plotting
plot_data = df.groupby(["label_text", "language"]).size().unstack(fill_value=0)

# Plot
plot_data.plot(kind="bar", stacked=True)
plt.xlabel("Label Text")
plt.ylabel("Count")
plt.title("Histogram of Label Texts by Language")
plt.show()
