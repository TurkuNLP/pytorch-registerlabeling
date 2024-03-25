import csv
import glob

from sklearn.metrics import average_precision_score, f1_score

import numpy as np

from .labels import binarize_labels, label_schemes, normalize_labels


def run(cfg):
    label_scheme = label_schemes[cfg.labels]
    dir_structure = f"{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}"
    results_output_dir = f"results/{dir_structure}"  # Save results in the repo

    for file_path in glob.glob(f"{results_output_dir}/predictions*.tsv"):
        print(file_path)

        with open(file_path, "r", encoding="utf-8") as file:
            labels, predictions = zip(
                *[
                    [
                        binarize_labels(normalize_labels(y, cfg.labels), cfg.labels)
                        for y in x
                    ]
                    for x in list(csv.reader(file, delimiter="\t"))
                ]
            )

        labels = np.array(labels)
        predictions = np.array(predictions)

        micro_f1 = f1_score(labels, predictions, average="micro")
        micro_pr_auc = average_precision_score(labels, predictions, average="micro")

        print(f"Micro-averaged F1 Score: {micro_f1}")
        print(f"Micro-averaged PR AUC Score: {micro_pr_auc}")
