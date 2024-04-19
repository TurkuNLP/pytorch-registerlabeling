import os
import csv
import json
import sys


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
from tqdm import tqdm

from datasets import concatenate_datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import get_dataset

csv.field_size_limit(sys.maxsize)


init_batch_data = lambda: {
    "input_ids": [],
    "attention_mask": [],
    "label": [],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to calculate average pooling of embeddings using the attention mask
def average_pool(hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
    sum_mask = mask_expanded.sum(1)
    return sum_embeddings / sum_mask.clamp(min=1e-9)


# Function to compute cosine similarity for each token in a single document, excluding special tokens
def compute_cosine_similarity_for_document(
    token_embeddings, doc_embedding, tokens, special_tokens
):
    # Normalize the document embedding to have unit norm
    doc_embedding = normalize(doc_embedding.reshape(1, -1))
    # Filter out special tokens and their embeddings using list comprehension
    filtered_token_embeddings, filtered_tokens = zip(
        *[
            (embed, token)
            for embed, token in zip(token_embeddings, tokens)
            if token not in special_tokens
        ]
    )

    if filtered_token_embeddings:  # Ensure there are tokens left after filtering
        filtered_token_embeddings = normalize(
            np.array(filtered_token_embeddings)
        )  # Normalize the filtered embeddings
        # Calculate cosine similarity between the normalized document embedding and each normalized token's embedding
        similarities = cosine_similarity(
            filtered_token_embeddings, doc_embedding
        ).flatten()
    else:
        similarities = []

    return filtered_tokens, similarities


# Function to scale similarities based on their min and max values
def scale_similarities(similarities):
    # Check if the array is not empty
    if similarities.size > 0:
        min_val = np.min(similarities)
        max_val = np.max(similarities)
        if max_val != min_val:
            # Scale the values to be between 0 and 1
            similarities = (similarities - min_val) / (max_val - min_val)
        else:
            # Avoid division by zero if all similarities are the same
            similarities = np.zeros_like(similarities)
    else:
        similarities = np.array(
            []
        )  # Return an empty array if there are no similarities to scale
    return similarities


# Function to transform and then re-scale similarities
def transform_and_rescale(similarities):
    # Inverse transformation: make differences near 1 more significant
    transformed = 1 / (
        1 - similarities + 0.01
    )  # Adding a small constant to avoid division by zero

    # Re-scale to [0, 1]
    min_val = np.min(transformed)
    max_val = np.max(transformed)
    if max_val != min_val:
        scaled = (transformed - min_val) / (max_val - min_val)
    else:
        scaled = np.zeros_like(transformed)

    return scaled


def logarithmic_scale(similarities):
    # Assuming similarities are already scaled between 0 and 1
    epsilon = 1e-5  # small number to avoid log(0)
    # Shift similarities by a small epsilon away from 0 to apply logarithm
    similarities = np.maximum(similarities, epsilon)
    # Apply a negative logarithm to make differences at the high end more pronounced
    return -np.log(similarities)


# Function to generate simple HTML for document visualization
def generate_html_for_document(tokens, similarities, doc_index):
    html_content = "<html><head><title>Document Visualization</title></head><body><p>"
    for token, similarity in zip(tokens, similarities):
        opacity = similarity
        html_content += f'<span style="background-color: rgba(0, 0, 255, {opacity}); margin: 2px; padding: 2px;">{token}</span>'
    html_content += "</p></body></html>"
    with open(f"document_{doc_index}.html", "w") as file:
        file.write(html_content)


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

    doc_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().detach().numpy()

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

    special_tokens = (
        tokenizer.all_special_tokens
    )  # Get all special tokens from the tokenizer
    for i in range(len(batch_data["doc_embeddings"])):
        token_embeds = batch_data["token_embeddings"][
            i
        ]  # Embeddings of tokens for the i-th document
        doc_embed = batch_data["doc_embeddings"][i]  # Embedding of the i-th document
        tokens = batch_data["tokens"][i]  # Tokens of the i-th document

        # Compute cosine similarities for the current document, excluding special tokens
        filtered_tokens, similarities = compute_cosine_similarity_for_document(
            token_embeds, doc_embed, tokens, special_tokens
        )

        # Transform and scale the similarities
        transformed_and_scaled_similarities = logarithmic_scale(np.array(similarities))

        # Generate and save HTML for visualization
        generate_html_for_document(
            filtered_tokens, transformed_and_scaled_similarities, i + 1
        )

        # Print or store the cosine similarities along with their corresponding tokens
        print(f"Document {i + 1}:")
        for token, similarity in zip(
            filtered_tokens, transformed_and_scaled_similarities
        ):
            print(f"  Token: {token}, Cosine Similarity: {similarity:.4f}")

    exit()


def run(cfg):
    if not cfg.train == cfg.dev == cfg.test:
        print("This script only works with the same dataset for train, dev and test")
        exit()
    path = "output/keywords"
    os.makedirs(path, exist_ok=True)
    path = f"{path}/{cfg.train}.csv"
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
    dataset["train"] = dataset["train"].filter(
        lambda example: "LY" in example["label_text"]
    )
    dataset["test"] = dataset["test"].filter(
        lambda example: "LY" in example["label_text"]
    )
    dataset["dev"] = dataset["dev"].filter(
        lambda example: "LY" in example["label_text"]
    )

    if cfg.sample:
        dataset["train"] = dataset["train"].select(
            range(min(cfg.sample, len(dataset["train"])))
        )
        dataset["test"] = dataset["test"].select(
            range(min(cfg.sample, len(dataset["test"])))
        )
        dataset["dev"] = dataset["dev"].select(
            range(min(cfg.sample, len(dataset["dev"])))
        )

    print(dataset["train"])

    data = concatenate_datasets([dataset["train"], dataset["dev"], dataset["test"]])
    batch_size = 16
    batch_data = init_batch_data()
    for d in tqdm(data):
        batch_data["input_ids"].append(d["input_ids"])
        batch_data["attention_mask"].append(d["attention_mask"])
        batch_data["label"].append(d["label_text"])

        if len(batch_data["input_ids"]) == batch_size:
            get_batch_embeddings(batch_data, model, tokenizer, path)
            batch_data = init_batch_data()

    if len(batch_data["input_ids"]):
        get_batch_embeddings(batch_data, model, tokenizer, path)
