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

source_directory = f"data/multi_all_test_bad"
target_directory = f"data/multi_all_test_bad"
source_file_name = f"train_dev.tsv"
target1_file_name = f"dev.tsv"
target2_file_name = f"train.tsv"

# Configuration for test data from cleaned, train + dev from others
n_splits = 2
target1_splits = [0]
target2_splits = [1]


# Initialize a list to hold all rows from all files
all_rows = []
example_indices = []
data = []
data_i = 0
# distribution = (1 - 0.2637, 0.2637)
# distribution = (1 - (0.828), 0.828)
distribution = (0.828, 1 - (0.828))

file_path = os.path.join(source_directory, source_file_name)
with open(file_path, "r") as file:
    rows = list(csv.reader(file, delimiter="\t"))
    random.shuffle(rows)

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

stratifier_initial = IterativeStratification(
    n_splits=n_splits,
    order=1,
    sample_distribution_per_fold=distribution or (1 / n_splits, 1 - (1 / n_splits)),
)
splits = list(stratifier_initial.split(X, Y))

print(f"Splits: {len(splits)}")
print(f"Split 0-0{len(splits[0][0])}")
print(f"Split 0-1{len(splits[0][1])}")
print(f"Split 1-0{len(splits[1][0])}")
print(f"Split 1-1{len(splits[1][1])}")


target1_indexes = []
target2_indexes = []

for i, split in enumerate(splits):
    if i in target1_splits:
        target1_indexes += list(split[1])
    elif i in target2_splits:
        target2_indexes += list(split[1])


X_1 = X[target1_indexes]
X_2 = X[target2_indexes]

print(f"Target1 idx: {len(X_1)}")
print(f"Target2 idx: {len(X_2)}")
print(any([x in X_1 for x in X_2]))
print(any([x in X_2 for x in X_1]))

target1_data = [data[i] for i in X_1]
target2_data = [data[i] for i in X_2]

with open(f"{target_directory}/{target1_file_name}", "w", encoding="utf-8") as outfile:
    for d in target1_data:
        outfile.write(f"{d[0]}\t{d[1]}\t{d[2]}\n")


with open(f"{target_directory}/{target2_file_name}", "w", encoding="utf-8") as outfile:
    for d in target2_data:
        outfile.write(f"{d[0]}\t{d[1]}\t{d[2]}\n")

print("Done")
