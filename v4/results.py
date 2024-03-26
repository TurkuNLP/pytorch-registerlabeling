import csv
import glob
import json

from sklearn.metrics import average_precision_score, f1_score

import numpy as np

from .labels import binarize_labels, label_schemes, normalize_labels
from .data import small_languages

s = lambda x: "{\scriptsize" + f"({x:.2f})" + "}"


def run(cfg):

    results_path = f"results/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}"
    results = {}
    for seed in [42, 43, 44]:
        for file_path in glob.glob(f"{results_path}/seed_{seed}/metrics*.json"):
            language = file_path.split("_")[-1].split(".json")[0]
            if language not in results:
                results[language] = {"f1": [], "pr_auc": []}

            data = json.load(open(file_path, "r", encoding="utf-8"))
            results[language]["f1"].append(data["f1"] * 100)
            results[language]["pr_auc"].append(data["pr_auc"] * 100)

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
    """
    main_languages = ["en", "fi", "fr", "sv", "tr"]

    for lang_group in [main_languages, small_languages]:
        main_lang_data = {"f1": [], "f1_std": [], "pr_auc": [], "pr_auc_std": []}
        for lang in lang_group:
            print(lang)
            f1s = np.array(results[lang]["f1"])
            pr_aucs = np.array(results[lang]["pr_auc"])
            f1_avg = np.mean(f1s)
            f1_std = np.std(f1s)
            pr_auc_avg = np.mean(pr_aucs)
            pr_auc_std = np.std(pr_aucs)
            print(f"\t& {f1_avg:.2f} {s(f1_std)} & {pr_auc_avg:.2f} {s(pr_auc_std)}")

            main_lang_data["f1"].append(f1_avg)
            main_lang_data["f1_std"].append(f1_std)
            main_lang_data["pr_auc"].append(pr_auc_avg)
            main_lang_data["pr_auc_std"].append(pr_auc_std)

        all_f1_avg = np.mean(np.array(main_lang_data["f1"]))
        all_f1_std = np.mean(np.array(main_lang_data["f1_std"]))
        all_pr_auc_avg = np.mean(np.array(main_lang_data["pr_auc"]))
        all_pr_auc_std = np.mean(np.array(main_lang_data["pr_auc_std"]))
        print("avg")
        print(
            f"\t& {all_f1_avg:.2f} {s(all_f1_std)} & {all_pr_auc_avg:.2f} {s(all_pr_auc_std)}"
        )
        print("---")
