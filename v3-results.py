import ast
import re
import sys

import numpy as np

s = lambda x: "{\scriptsize" + f"({x:.2f})" + "}"

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

include_data_symbol = "="

model = sys.argv[1]
labels = sys.argv[2]
targets = sys.argv[3]

if len(sys.argv) > 4:
    if sys.argv[4] == "z":
        include_data_symbol = "z"
    elif sys.argv[4] == "x":
        include_data_symbol = "x"
        if len(sys.argv) > 5:
            include_data_symbol += f"-{sys.argv[5]}"
    elif sys.argv[4] == "m":
        include_data_symbol = "m"

print_avgs = "-" in targets


def get_scores(target_lines_symbol):

    all_f1s = []
    all_f1_stds = []
    all_pr_aucs = []
    all_pr_auc_stds = []

    scores = {}

    for target in targets.split(","):
        small_prefix = (
            f"{target}/" if target not in small_languages else "small_languages/"
        )
        pr_aucs = []
        f1s = []
        with open(
            f"v3/configs/{model}/labels_{labels}/{small_prefix}{target}.yaml"
        ) as f:
            for line in f:
                if line.startswith(f"#{target_lines_symbol}"):

                    dict_str = re.search(r"\{.*\}", line).group()
                    result_dict = ast.literal_eval(dict_str)

                    f1 = result_dict["f1"]
                    pr_auc = result_dict["pr_auc"]

                    f1s.append(f1)
                    pr_aucs.append(pr_auc)

        f1_avg = np.mean(f1s) * 100
        f1_std = np.std(f1s) * 100

        pr_auc_avg = np.mean(pr_aucs) * 100
        pr_auc_std = np.std(pr_aucs) * 100

        all_f1s.append(np.mean(f1s))
        all_f1_stds.append(np.std(f1s))

        all_pr_aucs.append(np.mean(pr_aucs))
        all_pr_auc_stds.append(np.std(pr_aucs))

        scores[target] = {
            "f1_avg": f1_avg,
            "f1_std": f1_std,
            "pr_auc_avg": pr_auc_avg,
            "pr_auc_std": pr_auc_std,
        }

        # print(f"{target}\n===========")
        # print(f"& {f1_avg:.2f} {s(f1_std)} & {pr_auc_avg:.2f} {s(pr_auc_std)}")

    all_f1_avg = np.mean(all_f1s) * 100
    all_f1_std = np.mean(all_f1_stds) * 100

    all_pr_auc_avg = np.mean(all_pr_aucs) * 100
    all_pr_auc_std = np.mean(all_pr_auc_stds) * 100

    # print(f"Averages for {targets}\n===========")
    # print(f"& {all_f1_avg:.2f} {s(all_f1_std)} & {all_pr_auc_avg:.2f} {s(all_pr_auc_std)}")

    scores["avg"] = {
        "f1_avg": all_f1_avg,
        "f1_std": all_f1_std,
        "pr_auc_avg": all_pr_auc_avg,
        "pr_auc_std": all_pr_auc_std,
    }
    return scores


scores = get_scores(include_data_symbol)

for target in scores.keys():
    print(f"{target}\n===========")
    print(
        f"& {scores[target]['f1_avg']:.2f} {s(scores[target]['f1_std'])} & {scores[target]['pr_auc_avg']:.2f} {s(scores[target]['pr_auc_std'])}"
    )

if "-" in targets:
    print(f"Averages for {targets}\n===========")
    print(
        f"& {scores['avg']['f1_avg']:.2f} {s(scores['avg']['f1_std'])} & {scores['avg']['pr_auc_avg']:.2f} {s(scores['avg']['pr_auc_std'])}"
    )


if len(sys.argv) > 5:
    full_zs_scores = get_scores("z")

    for target in scores.keys():
        print(f"{target} full zero-shot difference\n===========")
        print(
            f"& {(scores[target]['f1_avg'] - full_zs_scores[target]['f1_avg']):.2f} {s(scores[target]['f1_std'] - full_zs_scores[target]['f1_std'])} & {(scores[target]['pr_auc_avg'] - full_zs_scores[target]['pr_auc_avg']):.2f} {s(scores[target]['pr_auc_std'] - full_zs_scores[target]['pr_auc_std'])}"
            # f"& {(scores[target]['f1_avg'] - full_zs_scores[target]['f1_avg']):.2f} {s(scores[target]['f1_std'] - full_zs_scores[target]['f1_std'])}"
        )
