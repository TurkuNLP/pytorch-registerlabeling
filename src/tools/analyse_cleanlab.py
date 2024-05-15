import csv
import sys

from tqdm import tqdm
import numpy as np

csv.field_size_limit(sys.maxsize)

from cleanlab import Datalab
from cleanlab.internal.multilabel_utils import int2onehot, onehot2int
from cleanlab.multilabel_classification.filter import find_label_issues
from ..data import small_languages
from ..labels import labels_structure


def parse_array(array_str, dtype):
    # Remove brackets and split by whitespace
    array_str = array_str.strip("[]").replace("\n", " ")
    array = np.array(array_str.split(), dtype=float)  # Convert to float first
    if dtype == int:
        return array.astype(int)  # Cast to int if needed
    return array


def read_tsv(file_path):
    data = []
    with open(file_path, "r") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            # Convert each row
            col1 = parse_array(row[0], int)  # First column: list of ints
            indices = np.where(col1 == 1)[0]
            col2 = parse_array(row[1], float)  # Second column: list of floats
            col3 = int(row[2])  # Third column: int

            data.append([indices, col2, col3])
    return data


def run(cfg):

    target = cfg.test if cfg.test in small_languages else "test"

    # Path to the data file
    text_data_path = f"data/{cfg.test}/{target}.tsv"

    with open(text_data_path, "r", newline="", encoding="utf-8") as data_file:
        text_data = list(csv.reader(data_file, delimiter="\t"))

    data = read_tsv(
        f"predictions/xlm-roberta-large/{cfg.train}_{cfg.train}/seed_42/all_all_{cfg.test}_probs.tsv"
    )

    labels = [list(item[0]) for item in data]
    pred_probs = [item[1] for item in data]

    issues = find_label_issues(
        labels=labels,
        pred_probs=np.array(pred_probs),
        return_indices_ranked_by="self_confidence",
    )

    print(list(issues))
    print(len(list(issues)))
    exit()

    labels = [[0], [0, 2], [0]]
    pred_probs = [
        [
            1.0,
            0.0,
            0.0,
        ],
        [0.96, 0.09, 0.88],
        [1.0, 0.01, 0.22],
    ]

    for row in pred_probs:
        print(type(row))

    num_to_display = 3  # increase this to see more examples

    print(f"labels for first {num_to_display} examples in format expected by cleanlab:")
    print(labels[:num_to_display])
    print(
        f"pred_probs for first {num_to_display} examples in format expected by cleanlab:"
    )
    print(pred_probs[:num_to_display])

    lab = Datalab(
        data={"labels": labels},
        label_name="labels",
        task="multilabel",
    )

    lab.find_issues(pred_probs=pred_probs, issue_types={"label": {}})

    label_issues = lab.get_issues("label")

    issues = (
        label_issues.query("is_label_issue").sort_values("label_score").index.values
    )

    print(f"Indices of examples with label issues:\n{issues}")
