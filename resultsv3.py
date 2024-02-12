import ast
import re
import sys

import numpy as np

f1s = []
pr_aucs = []

model = sys.argv[1]
labels = sys.argv[2]
targets = sys.argv[3]

if targets == "4":


with open(f"v3/configs/{model}/labels_{labels}/") as f:
    for line in f:
        if line.startswith("#="):

            dict_str = re.search(r"\{.*\}", line).group()
            result_dict = ast.literal_eval(dict_str)

            f1s.append(result_dict["f1"])
            pr_aucs.append(result_dict["pr_auc"])

f1_avg = np.mean(f1s) * 100
f1_std = np.std(f1s) * 100

pr_auc_avg = np.mean(pr_aucs) * 100
pr_auc_std = np.std(pr_aucs) * 100

s = lambda x: "{\scriptsize" + f"({x:.2f})" + "}"


print(f"& {f1_avg:.2f} {s(f1_std)} & {pr_auc_avg:.2f} {s(pr_auc_std)}")
