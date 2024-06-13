from skmultilearn.model_selection import (
    iterative_train_test_split,
    IterativeStratification,
)

import sys
import csv

csv.field_size_limit(sys.maxsize)

import numpy as np
import os

import random
from src import labels

# Make process deterministic
np.random.seed(42)
random.seed(42)

lang = "multi"
directory = f"data/{lang}_cleaned"
file_name = f"{lang}.tsv"

# Initialize a list to hold all rows from all files
all_rows = []
example_indices = []
data = []
data_i = 0

file_path = os.path.join(directory, file_name)
with open(file_path, "r") as file:
    rows = list(csv.reader(file, delimiter="\t"))

    # Step 5: Concatenate the lists
    for i, row in enumerate(rows):

        if len(row) > 1:
            normalized_labels = labels.normalize_labels(row[0], "all")
            label = labels.binarize_labels(normalized_labels, "all")
            all_rows.append(label)
            example_indices.append(data_i)
            datum = [row[0], row[1]]
            if len(row) > 2:
                datum.append(row[2])
            data.append(datum)
            data_i += 1



# Assuming X is your features and Y is your binary encoded labels
X = np.array(example_indices)
Y = np.array(all_rows)

stratifier_initial = IterativeStratification(n_splits=10, order=1)
splits = list(stratifier_initial.split(X, Y))

train_indexes = []
test_indexes = []
dev_indexes = []

for i, split in enumerate(splits):
    if i in [0, 1, 2, 3, 4, 5, 6]:
        train_indexes += list(split[1])
    elif i in [7, 8]:
        test_indexes += list(split[1])
    elif i in [9]:
        dev_indexes += list(split[1])

X_test, Y_test = X[test_indexes], Y[test_indexes]
X_train, Y_train = X[train_indexes], Y[train_indexes]
X_dev, Y_dev = X[dev_indexes], Y[dev_indexes]

print(f"Train idx: {len(X_train)}")
print(f"Test idx: {len(X_test)}")
print(f"Dev idx: {len(X_dev)}")
print(any([x in X_train for x in X_test]))
print(any([x in X_train for x in X_dev]))
print(any([x in X_test for x in X_dev]))
print(any([x in X_test for x in X_train]))
print(any([x in X_dev for x in X_train]))
print(any([x in X_dev for x in X_test]))

train_data = [data[i] for i in X_train]
test_data = [data[i] for i in X_test]
dev_data = [data[i] for i in X_dev]

with open(f"{directory}/train.tsv", "w", encoding="utf-8") as outfile:
    for d in train_data:
        outfile.write(f"{d[0]}\t{d[1]}\t{d[2]}\n")

with open(f"{directory}/test.tsv", "w", encoding="utf-8") as outfile:
    for d in test_data:
        outfile.write(f"{d[0]}\t{d[1]}\t{d[2]}\n")

with open(f"{directory}/dev.tsv", "w", encoding="utf-8") as outfile:
    for d in dev_data:
        outfile.write(f"{d[0]}\t{d[1]}\t{d[2]}\n")
