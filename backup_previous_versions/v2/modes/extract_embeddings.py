import csv

import tqdm


def extract_doc_embeddings(model, dataset, working_dir):
    dataset.set_format(type="torch")
    model = model.to("cpu")

    with open(f"{working_dir}/doc_embeddings.tsv", "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for d in tqdm(dataset["train"]):
            label_text = d.pop("label_text")
            d.pop("label")
            d.pop("text")
            language = d.pop("language")

            outputs = model(**d, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            doc_embeddings = last_hidden_states[0][0, :].detach().numpy()

            writer.writerow(
                [
                    language,
                    label_text,
                    " ".join([str(x) for x in doc_embeddings.tolist()]),
                ]
            )
