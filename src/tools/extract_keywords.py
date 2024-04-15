import os
import csv
import json
import sys
import string

import numpy as np
import torch
from tqdm import tqdm

from datasets import concatenate_datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import get_dataset

csv.field_size_limit(sys.maxsize)

init_batch_data = lambda: {
    "input_ids": [],
    "attention_mask": [],
    "language": [],
    "label": [],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pool_embeddings_for_words(token_embeddings, tokens, special_tokens, method="mean"):
    pool = np.mean if method == "mean" else np.max
    # Initialize a dictionary to hold the pooled embeddings for each full word
    word_embeddings = []
    current_word_embeddings = []
    current_word = ""

    # preprocessed_tokens = [preprocess_token(token, special_tokens) for token in tokens]

    for idx, token in enumerate(tokens):
        # Skip special tokens like <s>, </s>, etc.
        if token in special_tokens:
            continue
        # New word starts with _
        if token.startswith("▁"):
            if current_word_embeddings:
                # Pool the embeddings for the previous word and add to the dictionary
                pooled_embedding = pool(current_word_embeddings, axis=0).tolist()
                word_embeddings.append((current_word, pooled_embedding))
                current_word_embeddings = []

            # Remove the underscore from the token to get the word
            current_word = token[1:]
        else:
            # For tokens that are not the start of a new word, append them to the current word
            current_word += token

        # Add the current subword embedding
        current_word_embeddings.append(token_embeddings[idx])

    # Pool and add the last word
    if current_word_embeddings:
        pooled_embedding = pool(current_word_embeddings, axis=0).tolist()
        # word_embeddings[current_word] = pooled_embedding
        word_embeddings.append((current_word, pooled_embedding))

    return word_embeddings


def generate_ngrams_with_embeddings(
    word_embedding_tuples, method="mean", ngram_range=(1, 2)
):
    pool = np.mean if method == "mean" else np.max
    ngrams_with_embeddings = []

    # Iterate through the list to generate n-grams within the specified range
    for n in range(ngram_range[0], ngram_range[1] + 1):
        for i in range(len(word_embedding_tuples) - n + 1):
            # Extract the current slice for n-gram creation
            current_slice = word_embedding_tuples[i : i + n]
            if ngram_range[1] > 1:
                if any(
                    word[0].endswith(tuple(string.punctuation))
                    for word in current_slice[:-1]
                ):
                    continue

            # Construct the n-gram phrase by joining the words
            ngram_phrase = " ".join(
                word.strip(string.punctuation).lower()
                for word, _ in current_slice
                if word.strip(string.punctuation).lower()
            )

            # Calculate the average embedding for the current n-gram
            embeddings = [embedding for _, embedding in current_slice]
            averaged_embedding = pool(embeddings, axis=0)

            ngrams_with_embeddings.append((ngram_phrase, averaged_embedding))

    return ngrams_with_embeddings


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity


def compute_cosine_similarities(batch_data, i, special_tokens):

    word_embeddings = pool_embeddings_for_words(
        batch_data["token_embeddings"][i],
        batch_data["tokens"][i],
        special_tokens,
        "mean",
    )

    ngram_embeddings = generate_ngrams_with_embeddings(word_embeddings, "mean")

    cosine_similarities = []

    for ngram, embedding in ngram_embeddings:
        # Compute cosine similarity - choose the manual or sklearn function based on preference
        similarity = cosine_similarity(embedding, batch_data["doc_embeddings"][i])
        # similarity = cosine_similarity_sklearn(np.array(embedding), np.array(document_embedding))

        cosine_similarities.append((ngram, similarity))

    ordered_cosine_similarities = sorted(
        cosine_similarities, key=lambda x: x[1], reverse=True
    )
    return ordered_cosine_similarities


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_batch_embeddings(batch_data, model, tokenizer, output_path):

    batch = {
        "input_ids": torch.stack([x for x in batch_data["input_ids"]]),
        "attention_mask": torch.stack([x for x in batch_data["attention_mask"]]),
    }
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        output_hidden_states=True,
    )

    doc_embeddings = (
        average_pool(outputs.hidden_states[-1], batch["attention_mask"])
        .cpu()
        .detach()
        .numpy()
    )

    batch_data["doc_embeddings"] = doc_embeddings

    # Extract token IDs for each document in the batch
    batch_data["word_ids"] = [x.tolist() for x in batch["input_ids"]]

    # Convert token IDs to tokens
    batch_data["tokens"] = [
        [x.lower() for x in tokenizer.convert_ids_to_tokens(ids)]
        for ids in batch_data["word_ids"]
    ]

    # Extract and store token embeddings for each document
    token_embeddings = outputs.hidden_states[-1].cpu().detach().numpy()
    batch_data["token_embeddings"] = token_embeddings

    for i in range(len(batch_data["tokens"])):
        cosine_similarities = compute_cosine_similarities(
            batch_data, i, tokenizer.all_special_tokens
        )
        with open(f"{output_path}/keywords.tsv", "a", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    batch_data["language"][i],
                    batch_data["label"][i],
                    batch_data["doc_embeddings"][i],
                    json.dumps(cosine_similarities),
                ]
            )


def run(cfg):
    path = "output/keywords"
    os.makedirs(path, exist_ok=True)
    # Init model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model.eval()

    # Get the original model's name and init tokenizer
    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    dataset = get_dataset(cfg, tokenizer)
    dataset.set_format(type="torch")
    print(dataset)
    if cfg.sample:
        dataset["train"] = dataset["train"].select(range(cfg.sample))
        dataset["test"] = dataset["test"].select(range(cfg.sample))
        dataset["dev"] = dataset["dev"].select(range(cfg.sample))

    data = concatenate_datasets([dataset["train"], dataset["dev"], dataset["test"]])
    batch_size = 16
    batch_data = init_batch_data()
    for d in tqdm(data):
        batch_data["input_ids"].append(d["input_ids"])
        batch_data["attention_mask"].append(d["attention_mask"])
        batch_data["language"].append(d["language"])
        batch_data["label"].append(d["label_text"])

        if len(batch_data["input_ids"]) == batch_size:
            get_batch_embeddings(batch_data, model, tokenizer, path)
            batch_data = init_batch_data()

    if len(batch_data["input_ids"]):
        get_batch_embeddings(batch_data, model, tokenizer, path)
