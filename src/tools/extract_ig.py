import csv
import json
import os
import re

import captum
import numpy as np
import torch
from captum.attr import LayerIntegratedGradients
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import get_dataset
from ..labels import decode_binary_labels, label_schemes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_scheme = label_schemes["upper"]

init_batch_data = lambda: {"texts": [], "labels": [], "idx": []}


def print_aggregated(target, txt, real_label):
    """ "
    This requires one target and one agg vector at a time
    Shows agg scores as colors
    """
    print("<html><body>")

    x = captum.attr.visualization.format_word_importances(
        [t for t, a in txt], [a for t, a in txt]
    )
    print(f"<b>prediction: {target}, real label: {real_label}</b>")
    print(f"""<table style="border:solid;">{x}</table>""")
    print("</body></html>")


def mean_pool_ngrams(word_scores, k, n):
    # Check for valid n
    if n < 1:
        raise ValueError("n must be a positive integer")

    ngrams = []

    # Create n-grams for 1 up to n
    for size in range(k, n + 1):
        for i in range(len(word_scores) - size + 1):
            ngram_words = [word_scores[j][0] for j in range(i, i + size)]
            ngram_scores = [word_scores[j][1] for j in range(i, i + size)]
            mean_score = sum(ngram_scores) / len(
                ngram_scores
            )  # Calculate mean score for the n-gram
            ngrams.append(
                (" ".join(ngram_words), mean_score)
            )  # Append the n-gram and its mean score

    return ngrams


def aggregate(scores, tokens, special_tokens):
    scores = scores.cpu().tolist()

    # Initialize variables
    current_word = None
    max_abs_score = float("-inf")
    max_score = None
    word_scores = []

    # Process each token and corresponding score
    for score, token in zip(scores, tokens):
        if token in special_tokens:
            continue

        if token.startswith("▁"):  # This token is the start of a new word
            if current_word is not None:  # Save the max score of the previous word
                word_scores.append((current_word, max_score))
            current_word = token[1:]  # Start a new word (omit the initial "▁")
            max_score = score  # Reset the max score for the new word
            max_abs_score = abs(score)  # Reset the max absolute score for the new word
        else:
            if current_word is not None:
                current_word += token  # Append token to the current word
                if (
                    abs(score) > max_abs_score
                ):  # Check if the absolute value of the score is greater
                    max_score = score  # Update max score
                    max_abs_score = abs(score)  # Update max absolute score

    # Don't forget to save the last word's score
    if current_word is not None:
        word_scores.append((current_word, max_score))

    return word_scores


def blank_reference_input(tokenized_input, blank_token_id):
    blank_input_ids = tokenized_input.input_ids.clone().detach()
    # Blank out tokens that are not special, for each item in batch
    blank_input_ids[tokenized_input.special_tokens_mask == 0] = blank_token_id
    return blank_input_ids, tokenized_input.attention_mask


def process_batch(batch, model, tokenizer, threshold, path):

    texts = batch["texts"]
    labels = batch["labels"]
    idxs = batch["idx"]

    # Normalize spacing for punctuation
    texts = [
        re.sub("(?<! )(?=[:.,!?()])|(?<=[:.,!?()])(?! )", r" ", text) for text in texts
    ]

    # Tokenize batch
    inp = tokenizer(
        texts,
        return_tensors="pt",
        return_special_tokens_mask=True,
        truncation=True,
        padding="max_length",
        max_length=512,
    ).to(model.device)

    blank_input_ids = inp.input_ids.clone().detach()
    blank_input_ids[inp.special_tokens_mask == 0] = tokenizer.pad_token_id

    def predict_f(inputs, attention_mask=None):
        return model(inputs, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(predict_f, model.roberta.embeddings)

    with torch.no_grad():
        logits = model(inp.input_ids, attention_mask=inp.attention_mask).logits

    probs = torch.sigmoid(logits)
    bin_predictions = probs > threshold

    predicted_labels = decode_binary_labels(bin_predictions, "upper")

    # Calculate Integrated Gradients for each label in each text
    for i in range(len(texts)):
        predicted_label = predicted_labels[i]
        true_label = labels[i]
        tokens = tokenizer.convert_ids_to_tokens(inp.input_ids[i : i + 1][0])
        idx = idxs[i]
        for label_idx in range(len(label_scheme)):
            attrs = lig.attribute(
                inputs=(
                    inp.input_ids[i : i + 1],
                    inp.attention_mask[i : i + 1],
                ),  # Single example in batch
                baselines=(blank_input_ids[i : i + 1], inp.attention_mask[i : i + 1]),
                target=label_idx,
                internal_batch_size=10,
                n_steps=50,
            )

            attrs_sum = attrs.sum(dim=-1).squeeze(0)
            attrs_sum = attrs_sum / torch.norm(attrs_sum)
            aggregated_tg = aggregate(attrs_sum, tokens, tokenizer.all_special_tokens)
            test_label = label_scheme[label_idx]

            with open(f"{path}", "a", newline="") as tsvfile:
                writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
                writer.writerow(
                    [
                        idx.item(),
                        probs[i].cpu().tolist(),
                        true_label if type(true_label) == str else " ".join(true_label),
                        predicted_label,
                        test_label,
                        json.dumps(aggregated_tg, ensure_ascii=False),
                    ]
                )


def run(cfg):
    if not cfg.train == cfg.dev == cfg.test:
        print("This script only works with the same dataset for train, dev and test")
        exit()
    path = f"output/keywords_ig"
    if cfg.save_path_suffix:
        path += "/" + cfg.save_path_suffix
    os.makedirs(path, exist_ok=True)
    path = f"{path}/{cfg.train}.csv"
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )

    threshold = cfg.threshold

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

    # data = concatenate_datasets([dataset["train"], dataset["dev"], dataset["test"]])
    data = dataset["test"]
    batch_size = 8
    batch_data = init_batch_data()

    for d in tqdm(data):
        batch_data["texts"].append(d["text"])
        batch_data["labels"].append(d["label_text"])
        batch_data["idx"].append(d["idx"])

        if len(batch_data["texts"]) == batch_size:
            process_batch(batch_data, model, tokenizer, threshold, path)
            batch_data = init_batch_data()

    if len(batch_data["texts"]):
        process_batch(batch_data, model, tokenizer, threshold, path)
