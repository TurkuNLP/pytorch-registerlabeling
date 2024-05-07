import glob
import json

from sklearn.metrics import average_precision_score, f1_score

import numpy as np

from .labels import binarize_labels, label_schemes, normalize_labels
from .data import small_languages

s = lambda x: "\\tiny{" + f"({x:.2f})" + "}"

main_languages = ["en", "fi", "fr", "sv", "tr"]
small_languages = [
    "ar",
    "ca",
    "es",
    "fa",
    "hi",
    "id",
    "jp",
    "no",
    "pt",
    "ur",
    "zh",
]

multi_str = "en-fi-fr-sv-tr"

seeds = [42, 43, 44]


def run(cfg):

    average = "" if cfg.average == "micro" else "_macro"

    result_str = ""
    all_f1s = []
    all_f1_stds = []
    for lang in main_languages if cfg.target == "main" else small_languages:
        f1s = []
        for seed in seeds:
            if cfg.test == "mono":
                lang_path = lang
            elif cfg.test == "zero":
                lang_path = "-".join([x for x in multi_str.split("-") if x != lang])
            elif cfg.test == "multi":
                lang_path = multi_str
            data = json.load(
                open(
                    f"predictions/{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/{lang_path}_{lang_path}/seed_{seed}/{cfg.labels}_{cfg.predict_labels}_{lang}_metrics.json",
                    "r",
                    encoding="utf-8",
                )
            )
            f1s.append(data[f"f1{average}"] * 100)

        f1s = np.array(f1s)
        mean = np.mean(f1s)
        std = np.std(f1s)
        all_f1s.append(mean)
        all_f1_stds.append(std)
        print(lang)
        print(f1s)

        result_str += f"& {np.mean(f1s):.0f} {s(np.std(f1s))} "
    result_str += f"& {np.mean(all_f1s):.0f} {s(np.mean(all_f1_stds))} "
    print(result_str)
