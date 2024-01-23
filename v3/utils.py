from datetime import datetime
import csv
from pydoc import locate

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


def extract_doc_embeddings(model, dataset, output_path):
    dataset.set_format(type="torch")
    model = model.to("cpu")

    for split, data in dataset.items():
        print(f"Extracting from {split}")
        print(f"Writing to {output_path}/doc_embeddings.tsv")
        for d in tqdm(data):
            label_text = d.pop("label_text")
            d.pop("labels")
            d.pop("text")
            d.pop("id")
            d.pop("split")
            d.pop("length")
            language = d.pop("language")

            d["input_ids"] = d["input_ids"].unsqueeze(0)
            d["attention_mask"] = d["attention_mask"].unsqueeze(0)

            outputs = model(**d, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            doc_embeddings = last_hidden_states[0][0, :].detach().numpy()
            with open(f"{output_path}/doc_embeddings.tsv", "a", newline="") as tsvfile:
                writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    [
                        language,
                        label_text,
                        " ".join([str(x) for x in doc_embeddings.tolist()]),
                    ]
                )
