import csv
import os

import numpy as np
import torch
from tqdm import tqdm

init_batch_data = lambda: {
    "input_ids": [],
    "attention_mask": [],
    "language": [],
    "label": [],
}


def extract_doc_embeddings(model, dataset, output_path, device, method):
    def model_output_embeddings(batch_data):
        batch = {
            "input_ids": torch.stack([x for x in batch_data["input_ids"]]),
            "attention_mask": torch.stack([x for x in batch_data["attention_mask"]]),
        }
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch.pop("input_ids"),
            attention_mask=batch.pop("attention_mask"),
            output_hidden_states=True,
        )
        if method == "document":
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().detach().numpy()
        elif method == "tokens_mean":
            embeddings = [
                np.mean(x, axis=0)
                for x in outputs.hidden_states[-1].cpu().detach().numpy()
            ]

        elif method == "tokens_max":
            embeddings = [
                np.max(x, axis=0)
                for x in outputs.hidden_states[-1].cpu().detach().numpy()
            ]

        batch_data["embeddings"] = embeddings

        with open(f"{output_path}/{method}_embeddings.tsv", "a", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
            for b in range(len(batch_data["embeddings"])):
                writer.writerow(
                    [
                        batch_data["language"][b],
                        batch_data["label"][b],
                        " ".join(
                            [str(x) for x in batch_data["embeddings"][b].tolist()]
                        ),
                    ]
                )

    os.makedirs(output_path, exist_ok=True)
    dataset.set_format(type="torch")

    for split, data in dataset.items():
        print(f"Extracting from {split}")
        print(f"Writing to {output_path}/{method}_embeddings.tsv")
        batch_size = 2
        batch_data = init_batch_data()
        for d in tqdm(data):
            batch_data["input_ids"].append(d["input_ids"])
            batch_data["attention_mask"].append(d["attention_mask"])
            batch_data["language"].append(d["language"])
            batch_data["label"].append(d["label_text"])

            if len(batch_data["input_ids"]) == batch_size:
                model_output_embeddings(batch_data)
                batch_data = init_batch_data()

        if len(batch_data["input_ids"]):
            model_output_embeddings(batch_data)
            batch_data = init_batch_data()


def extract_st_doc_embeddings(model, dataset, output_path):
    for split, data in dataset.items():
        print(f"Extracting from {split}")
        for d in tqdm(data):
            with open(f"{output_path}/st_embeddings.tsv", "a", newline="") as tsvfile:
                writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    [
                        d["language"],
                        d["label_text"],
                        " ".join([str(x) for x in model.encode(d["text"]).tolist()]),
                    ]
                )


def extract_e5_doc_embeddings(model, dataset):
    def average_pool(
        last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    for d in dataset["train"]:
        outputs = model(**[d])
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:2] @ embeddings[2:].T) * 100
        print(scores.tolist())
