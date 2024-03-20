import csv
import json

from .labels import decode_binary_labels


def get_linear_modules(model):
    print("Getting linear module names")
    print(model)

    linear_modules = set()

    for name, module in model.named_modules():
        name = name.lower()
        if "attention" in name and "self" in name and "Linear" in str(type(module)):
            linear_modules.add(name.split(".")[-1])

    print(f"Found linear modules: {linear_modules}")
    return list(linear_modules)


def save_predictions(trues, preds, metrics, cfg):
    true_labels_str = decode_binary_labels(trues, cfg.label_scheme)
    predicted_labels_str = decode_binary_labels(preds, cfg.label_scheme)

    data = list(zip(true_labels_str, predicted_labels_str))
    out_file = f"{cfg.working_dir}/test_predictions_{cfg.data.test or cfg.data.dev or cfg.data.train}_{cfg.trainer.learning_rate}.csv"
    out_file_metrics = f"{cfg.working_dir}/test_metrics_{cfg.data.test or cfg.data.dev or cfg.data.train}_{cfg.trainer.learning_rate}.json"

    with open(out_file, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter="\t")
        csv_writer.writerows(data)

    with open(out_file_metrics, "w") as f:
        json.dump(metrics, f)

    print(f"Predictions saved to {out_file}")
    print(f"Metrics saved to {out_file_metrics}")
