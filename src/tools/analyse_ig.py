import json
import re
from captum.attr import LayerIntegratedGradients
import captum
import numpy as np
import torch
import string

from transformers import AutoModelForSequenceClassification, AutoTokenizer


SPECIAL_TOKENS = ["<s>", "</s>"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###### BELOW IS CONTRUBITOR STUFF

import numpy as np


def aggregate_scores(texts):
    scores_by_word = {}

    # Collect all scores for each word
    for text in texts:
        for word, score in text:
            if word not in scores_by_word:
                scores_by_word[word] = []
            scores_by_word[word].append(score)

    return scores_by_word


def compute_statistics(scores_by_word):
    stats_by_word = {}

    # Compute statistics for each word
    for word, scores in scores_by_word.items():
        mean_score = np.mean(scores)
        std_dev = np.std(scores)
        stats_by_word[word] = {"mean": mean_score, "std_dev": std_dev, "scores": scores}

    return stats_by_word


def find_top_contributors(stats_by_word, top_n=20):
    # Sort words by mean score for positive and negative contributions
    sorted_by_mean_positive = sorted(
        stats_by_word.items(), key=lambda x: x[1]["mean"], reverse=True
    )
    sorted_by_mean_negative = sorted(stats_by_word.items(), key=lambda x: x[1]["mean"])

    top_positive = sorted_by_mean_positive[:top_n]
    top_negative = sorted_by_mean_negative[:top_n]

    return top_positive, top_negative


#### END CONTRIBUTOR STUFF


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


def aggregate(scores, tokens, special_tokens, prepare_keywords=False):
    scores = scores.cpu().tolist()

    # Initialize variables
    current_word = None
    max_abs_score = float("-inf")
    max_score = None
    word_scores = []

    # Process each token and corresponding score
    for score, token in zip(scores, tokens):
        print(token)
        if prepare_keywords:
            token = token.lower()
            if not token.strip(string.punctuation).replace("▁", "").strip():
                continue
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


def run(cfg):
    # model = torch.load(f"{cfg.model_path}", map_location=torch.device(device))
    # model = model.to(device)
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )

    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    texts = [
        "Full service design and build contractor providing custom outdoor kitchens , outdoor fireplaces , patios , arbors , pergolas , decks , outdoor fire pits , swimming pools , custom stonework and hardscapes , fences , water features , retaining walls and a variety of remodeling , renovations and home repairs . To serve you better , we reserve the right to make improvements to the products and services seen on our website . Therefore , some may change without notice ."
    ]

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
        padding=True,
    ).to(model.device)

    blank_input_ids = inp.input_ids.clone().detach()
    blank_input_ids[inp.special_tokens_mask == 0] = tokenizer.pad_token_id

    def predict_f(inputs, attention_mask=None):
        return model(inputs, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(predict_f, model.roberta.embeddings)

    with torch.no_grad():
        logits = model(inp.input_ids, attention_mask=inp.attention_mask).logits

    bin_predictions = torch.sigmoid(logits) > 0.45

    print(bin_predictions)

    # Calculate Integrated Gradients for each label in each text
    for i in range(len(texts)):  # Loop through batch
        text_labels = bin_predictions[i]
        label_indices = text_labels.nonzero(as_tuple=True)[0]
        for label_idx in label_indices:
            attrs = lig.attribute(
                inputs=inp.input_ids[i : i + 1],
                baselines=blank_input_ids[i : i + 1],
                target=label_idx.item(),
            )
            tokens = tokenizer.convert_ids_to_tokens(inp.input_ids[i : i + 1][0])
            attrs_sum = attrs.sum(dim=-1).squeeze(0)
            attrs_sum = attrs_sum / torch.norm(attrs_sum)
            aggregated_tg = aggregate(
                attrs_sum, tokens, tokenizer.all_special_tokens, prepare_keywords=False
            )
            print_aggregated(label_idx, aggregated_tg, "SOME LABEL")

    exit()
    for tg in targets:

        attrs = lig.attribute(
            inputs=(inp.input_ids, inp.attention_mask),
            baselines=(b_input_ids, b_attention_mask),
            target=tuple(tg),
            internal_batch_size=10,
            n_steps=50,
        )
        # append the calculated and normalized scores to aggregated
        attrs_sum = attrs.sum(dim=-1).squeeze(0)
        attrs_sum = attrs_sum / torch.norm(attrs_sum)

        print(attrs_sum)

        aggregated_tg = aggregate(
            attrs_sum, tokens, tokenizer.all_special_tokens, prepare_keywords=True
        )

        print(aggregated_tg)

        result = mean_pool_ngrams(aggregated_tg, 1, 1)

        print(result)

        # Compute statistics for each word
        stats_by_word = compute_statistics(result)

        # Find the top 20 positive and negative contributing words
        top_positive, top_negative = find_top_contributors(stats_by_word)

        print(
            "Top Positive Keywords:",
            [(word, data["mean"], data["std_dev"]) for word, data in top_positive],
        )
        print(
            "Top Negative Keywords:",
            [(word, data["mean"], data["std_dev"]) for word, data in top_negative],
        )

        print_aggregated(tg, result, "SOME LABEL")

        exit()
