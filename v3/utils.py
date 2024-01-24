from datetime import datetime
import csv
from pydoc import locate
import torch
from tqdm import tqdm

_print = print


# Print with datetime
def print(*args, **kw):
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _print(f"[{formatted_now}]", *args, **kw)


def get_torch_dtype(torch_dtype_str):
    return (
        locate(f"torch.{torch_dtype_str}")
        if torch_dtype_str not in [None, "auto"]
        else torch_dtype_str
    )


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


def model_output_embeddings(batch_data, model, output_path, device):
    batch = {
        "input_ids": torch.stack([x for x in batch_data["input_ids"]]),
        "attention_mask": torch.stack([x for x in batch_data["attention_mask"]]),
    }
    batch = {k: v.to(device) for k, v in batch_data.items()}
    outputs = model(
        input_ids=batch.pop("input_ids"),
        attention_mask=batch.pop("attention_mask"),
        output_hidden_states=True,
    )
    batch_data["embedding"] = outputs.hidden_states[-1][:, 0, :].detach().numpy()

    with open(f"{output_path}/doc_embeddings.tsv", "a", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for b in range(len(batch_data["embedding"])):
            writer.writerow(
                [
                    batch_data["language"][b],
                    batch_data["label"][b],
                    " ".join([str(x) for x in batch_data["embedding"][b].tolist()]),
                ]
            )


init_batch_data = lambda: {
    "input_ids": [],
    "attention_mask": [],
    "language": [],
    "label": [],
}


def extract_doc_embeddings(model, dataset, output_path, device):
    dataset.set_format(type="torch")

    for split, data in dataset.items():
        print(f"Extracting from {split}")
        print(f"Writing to {output_path}/doc_embeddings.tsv")
        batch_size = 16
        batch_data = init_batch_data()
        for d in tqdm(data):
            batch_data["input_ids"].append(d["input_ids"])
            batch_data["attention_mask"].append(d["attention_mask"])
            batch_data["language"].append(d["language"])
            batch_data["label"].append(d["label_text"])

            if len(batch_data["input_ids"]) == batch_size:
                model_output_embeddings(batch_data, model, output_path, device)
                batch_data = init_batch_data()

        if len(batch_data["input_ids"]):
            model_output_embeddings(batch_data, model, output_path, device)
            batch_data = init_batch_data()
