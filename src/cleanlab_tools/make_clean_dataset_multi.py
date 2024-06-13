import csv
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)

from cleanlab.multilabel_classification.dataset import (
    rank_classes_by_multilabel_quality,
)
from cleanlab.multilabel_classification.filter import find_label_issues

from ..labels import labels_all


def parse_array(array_str, dtype):
    # Parse data from a TSV file
    array_str = array_str.strip("[]").replace("\n", " ")
    array = np.array(array_str.split(), dtype=float)  # Convert to float first
    if dtype == int:
        return array.astype(int)  # Cast to int if needed
    return array


def read_tsv(file_path, row_offset):
    data = []
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            # Convert each row
            col1 = parse_array(row[0], int)  # First column: list of ints
            indices = np.where(col1 == 1)[0]
            col2 = parse_array(row[1], float)  # Second column: list of floats
            col3 = int(row[2]) + row_offset  # Third column: row ID
            data.append([indices, col2, col3])
    return data


target = "multi"
langs = ["en", "fi", "fr", "sv", "tr"]
output_file_path = "data/multi_cleaned/multi.tsv"
bad_output_file_path = "data/multi_cleaned/multi_bad.tsv"
index_offset = 0

if __name__ == "__main__":

    label_data = []
    text_data = []

    # Process the 10 splits we have made for cross-validation
    for i in range(10):

        split = f"{target}_{i+1}"
        print(f"Processing {split}")

        # Get text data from this split
        with open(
            f"data/{split}/test.tsv", "r", newline="", encoding="utf-8"
        ) as data_file:
            split_text_data = list(csv.reader(data_file, delimiter="\t"))

        # Add to full text data
        text_data += split_text_data
        split_label_data = []
        for lang in langs:
            split_label_lang_data = read_tsv(
                f"predictions/xlm-roberta-large/{split}_{split}/seed_42/all_all_{lang}_probs.tsv",
                index_offset,
            )
            split_label_data += split_label_lang_data

        label_data += split_label_data
        index_offset += len(split_text_data)

        print(len(split_text_data))
        print(len(split_label_data))

    # Make the final data
    labels = [list(item[0]) for item in label_data]
    pred_probs = [item[1] for item in label_data]

    print("got data")

    # df = common_multilabel_issues(labels=labels, pred_probs=np.array(pred_probs))
    df = rank_classes_by_multilabel_quality(
        labels=labels, pred_probs=np.array(pred_probs), class_names=labels_all
    )
    # Convert DataFrame to a tab-separated string
    tsv_output = df.to_csv(sep="\t", index=False)

    # Print the TSV output
    print(tsv_output)

    exit()  # Remove this to continue writing data

    issues = find_label_issues(
        labels=labels,
        pred_probs=np.array(pred_probs),
        return_indices_ranked_by="self_confidence",
    )

    bad_row_idx = {}
    for idx, row_id in enumerate(issues):
        bad_row_idx[int(label_data[row_id][2])] = idx

    print(
        f"Found {len(bad_row_idx)} bad examples. ({(len(bad_row_idx) / len(text_data)):.4f}) of all"
    )

    # Open the output file in append mode
    with open(output_file_path, "a", newline="", encoding="utf-8") as out_file:
        tsv_writer = csv.writer(
            out_file, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        with open(
            bad_output_file_path, "a", newline="", encoding="utf-8"
        ) as bad_out_file:
            tsv_bad_writer = csv.writer(
                bad_out_file, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"
            )

            # Iterate over text_data and check if the row index is not in bad_rows
            for index, row in enumerate(text_data):
                if index not in bad_row_idx:
                    tsv_writer.writerow(row)
                else:
                    row.append(bad_row_idx[index])
                    tsv_bad_writer.writerow(row)
