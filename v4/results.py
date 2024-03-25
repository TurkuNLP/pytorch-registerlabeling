import csv
import glob

from sklearn.metrics import average_precision_score, f1_score

import numpy as np

from .labels import binarize_labels, label_schemes, normalize_labels


def run(cfg):

    results_path = f"results/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}"

    """

    label_scheme = label_schemes[cfg.labels]
    results_path = f"results/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}"
    results = {}
    for seed in [42, 43, 44]:
        for file_path in glob.glob(f"{results_path}/seed_{seed}/predictions*.tsv"):
            language = file_path.split("_")[-1].split(".tsv")[0]
            if language not in results:
                results[language] = []

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

            results[language].append([micro_f1, micro_pr_auc])

    main_languages = ["en", "fi", "fr", "sv", "tr"]

    print(results)

    for lang in main_languages:
        print(f"--- {lang} ---")
        data = results[lang]
        # Convert data to a NumPy array
        data_array = np.array(data)

        # Compute column-wise averages and standard deviations
        columnwise_averages = np.mean(data_array, axis=0)
        columnwise_std = np.std(data_array, axis=0)

        print("Column-wise averages:", columnwise_averages)
        print("Column-wise standard deviations:", columnwise_std)
    """