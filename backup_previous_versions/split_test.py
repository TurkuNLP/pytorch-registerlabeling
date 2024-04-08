from skmultilearn.model_selection import (
    iterative_train_test_split,
    IterativeStratification,
)

import numpy as np
import os

from v3 import labels

lang = "en"

directory = f"data/{lang}"
file_name = f"{lang}.tsv"

# Step 1 & 2: Check if the file exists and create the list accordingly
if os.path.exists(os.path.join(directory, file_name)):
    files_list = [file_name]
else:
    files_list = ["train.tsv", "test.tsv", "dev.tsv"]

# Initialize a list to hold all rows from all files
all_rows = []
example_indices = []
data = []

data_i = 0
# Step 3 & 4: Iterate over the files list and parse the tsv files
for file in files_list:
    file_path = os.path.join(directory, file)
    with open(file_path, "r") as f:
        # Parse the tsv file into a list of lists
        rows = [line.strip().split("\t") for line in f]
        # Step 5: Concatenate the lists
        for i, row in enumerate(rows):

            if len(row) > 1:
                normalized_labels = labels.normalize_labels(row[0], "all")
                label = labels.binarize_labels(normalized_labels, "all")
                all_rows.append(label)
                example_indices.append(data_i)
                data.append([row[0], row[1]])
                data_i += 1


# Assuming X is your features and Y is your binary encoded labels
X = np.array(example_indices)
Y = np.array(all_rows)

# Calculate split indexes for initial split: 70% train, 30% for temp (test+dev)
stratifier_initial = IterativeStratification(
    n_splits=2, order=1, sample_distribution_per_fold=[0.7, 0.3]
)

temp_indexes, train_indexes = next(stratifier_initial.split(X, Y))

print(f"Train idx: {len(train_indexes)}")

X_train, Y_train = X[train_indexes], Y[train_indexes]
X_temp, Y_temp = X[temp_indexes], Y[temp_indexes]

stratifier_second = IterativeStratification(
    n_splits=2, order=1, sample_distribution_per_fold=[2 / 3, 1 / 3]
)
dev_indexes, test_indexes = next(stratifier_second.split(X_temp, Y_temp))

X_test, Y_test = X_temp[test_indexes], Y_temp[test_indexes]
X_dev, Y_dev = X_temp[dev_indexes], Y_temp[dev_indexes]


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

with open(f"data/{lang}_new/train.tsv", "w", encoding="utf-8") as outfile:
    for d in train_data:
        outfile.write(f"{d[0]}\t{d[1]}\n")

with open(f"data/{lang}_new/test.tsv", "w", encoding="utf-8") as outfile:
    for d in test_data:
        outfile.write(f"{d[0]}\t{d[1]}\n")

with open(f"data/{lang}_new/dev.tsv", "w", encoding="utf-8") as outfile:
    for d in dev_data:
        outfile.write(f"{d[0]}\t{d[1]}\n")
