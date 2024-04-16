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


def aggregate_subword_attentions(tokens, attention_scores):
    word_attentions = []
    current_word = ""
    current_score = 0.0
    subword_count = 0  # To track the number of subwords in the current word

    for token, score in zip(tokens, attention_scores):
        if token.startswith("▁"):
            if current_word:  # Check if there's a current word being built
                average_score = (
                    current_score / subword_count
                    if subword_count > 0
                    else current_score
                )
                word_attentions.append((current_word, average_score))
            current_word = token[1:]  # Start a new word, stripping the separator
            current_score = score
            subword_count = 1  # Reset subword count for the new word
        else:
            if not current_word:
                # If it's the first token and doesn't start with "▁", start a new word
                current_word = token
                subword_count = 1
            else:
                # Append the subword to the current word
                current_word += token
                subword_count += 1
            current_score += score  # Accumulate the score for subwords

    # Append the last word if exists
    if current_word:
        average_score = (
            current_score / subword_count if subword_count > 0 else current_score
        )
        word_attentions.append((current_word, average_score))

    return word_attentions


word_attention_aggregate = {}
doc_plaintexts = []


def get_batch_embeddings(batch_data, model, tokenizer, output_path):

    batch = {
        "input_ids": torch.stack([x for x in batch_data["input_ids"]]),
        "attention_mask": torch.stack([x for x in batch_data["attention_mask"]]),
    }
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_attentions=True,
        )

    attentions = outputs.attentions

    # Extract the last layer's attention (as previously described)
    last_layer_attention = attentions[
        -1
    ]  # Shape: [batch_size, num_heads, seq_length, seq_length]

    # Average across the attention heads
    avg_attention = torch.mean(
        last_layer_attention, dim=1
    )  # Shape: [batch_size, seq_length, seq_length]

    all_layers_attention = torch.stack(
        attentions
    )  # Stack: [num_layers, batch_size, num_heads, seq_length, seq_length]
    avg_attention_ALL = torch.mean(all_layers_attention, dim=0)

    avg_attention_ALL = torch.mean(avg_attention_ALL, dim=1)

    avg_attention = avg_attention_ALL if True else avg_attention

    # Extract the attention each token pays to the CLS token (index 0)
    cls_attention_scores = avg_attention[:, :, 0]  # Shape: [batch_size, seq_length]
    token_attention_scores = torch.sum(
        avg_attention, dim=2
    )  # Summing across columns, shape: [batch_size, seq_length]

    target = cls_attention_scores if True else token_attention_scores

    batch_size = cls_attention_scores.shape[0]
    for i in range(batch_size):
        # Get the sequence length to avoid padding tokens
        seq_length = (batch["attention_mask"][i] == 1).sum()

        # Get the token representations and corresponding attention scores
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i][:seq_length])
        attention_scores = target[i, :seq_length].tolist()

        # Aggregate subword attentions into full word attentions
        word_attentions = aggregate_subword_attentions(tokens, attention_scores)
        doc_plaintexts.append([])
        # MAKE THE DICT
        for word, att in word_attentions:
            word = word.lower()
            if word not in word_attention_aggregate:
                word_attention_aggregate[word] = []
            word_attention_aggregate[word].append(att)
            doc_plaintexts[-1].append(word)

        doc_plaintexts[-1] = " ".join(doc_plaintexts[-1])

        # Sort and find the top-k words with the highest accumulated attention scores
        word_attentions.sort(
            key=lambda x: x[1], reverse=True
        )  # Sort by attention score
        most_influential_words = word_attentions[:10]  # Adjust k as needed

        print(f"Most influential words for item {i}:", most_influential_words)

    return

    ### BELOW IS OLD CODE

    last_layer_attention = attentions[-1]
    avg_attention = torch.mean(last_layer_attention, dim=1)
    batch_size = avg_attention.shape[0]
    full_words_lists = []

    for i in range(batch_size):

        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])
        seq_length = (batch["attention_mask"][i] == 1).sum()
        full_words = []
        current_word = ""
        indices_to_words = [
            0
        ] * seq_length  # Initialize mapping of tokens to word indices
        word_index = -1  # Start with -1 to increment at the first word

        # Build list of full words from tokens and map tokens to words
        for idx, token in enumerate(tokens[:seq_length]):
            if token.startswith("▁"):
                if current_word:  # If there's a current word built up, append it
                    full_words.append(current_word)
                    word_index += 1
                current_word = token[1:]  # Start a new word, strip the prefix
            else:
                current_word += token  # Continue building the current word
            indices_to_words[idx] = word_index  # Map token index to current word index

        # Append the last word if it's not empty
        if current_word:
            full_words.append(current_word)
            word_index += 1

        # Initialize word-level attention matrix
        num_words = len(full_words)
        word_attention = torch.zeros(
            num_words, num_words, dtype=torch.float, device=avg_attention.device
        )

        # Aggregate attention scores for words
        for start in range(seq_length):
            for end in range(seq_length):
                start_word = indices_to_words[start]
                end_word = indices_to_words[end]
                addition = avg_attention[i, start, end]
                # print(
                #    f"Adding {addition:.10f} to word indices {indices_to_words[start]} -> {indices_to_words[end]}"
                # )
                word_attention[start_word, end_word] += addition

        print(word_attention)
        print(torch.sum(word_attention, dim=1))

        word_attention_scores = torch.sum(
            word_attention, dim=1
        )  # Summing across columns

        values, indices = word_attention_scores.topk(k=10)  # Get top 3 words
        most_influential_words = [full_words[idx] for idx in indices]

        print(
            f"Most influential words for item {i}:",
            list(zip(most_influential_words, values)),
        )

        values, indices = word_attention_scores.topk(
            k=10, largest=False
        )  # Get top 3 words
        least_influential_words = [full_words[idx] for idx in indices]
        print(
            f"Least influential words for item {i}:",
            list(zip(least_influential_words, values)),
        )
        print("---")

    return

    last_layer_attention = attentions[-1]
    avg_attention = torch.mean(last_layer_attention, dim=1)
    batch_size = avg_attention.shape[0]
    for i in range(batch_size):
        input_ids = batch["input_ids"][i]

        # Decode each subtoken individually
        decoded_tokens = [
            tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids.tolist()
        ]

        # Print the entire decoded sequence for the current batch item
        print(f"Decoded sequence for item {i}:")
        for idx, token in enumerate(decoded_tokens):
            print(
                f"Index {idx}: Token ID {input_ids[idx].item()}, Decoded Token '{token}'"
            )

        token_attention_scores = torch.sum(avg_attention, dim=2)

        batch_size = token_attention_scores.shape[0]
        for i in range(batch_size):
            # Get the sequence length to avoid padding tokens
            seq_length = (batch["attention_mask"][i] == 1).sum()

            # Get the token representations for clearer debugging
            tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][i])

            # Retrieve the attention scores for valid tokens only
            valid_attention_scores = token_attention_scores[i, :seq_length]

            # Get the indices of the tokens with the highest attention scores
            _, indices = valid_attention_scores.topk(k=10)  # Adjust k as needed

            # Decode the most influential tokens
            most_influential_tokens = [tokens[idx] for idx in indices]

            print(f"Most influential tokens for item {i}:", most_influential_tokens)
    exit()
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
        with open(f"{output_path}", "a", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    batch_data["label"][i],
                    batch_data["doc_embeddings"][i],
                    json.dumps(cosine_similarities, ensure_ascii=False),
                ]
            )


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
        lambda example: "lt" in example["label_text"]
    )
    dataset["test"] = dataset["test"].filter(
        lambda example: "lt" in example["label_text"]
    )
    dataset["dev"] = dataset["dev"].filter(
        lambda example: "" in example["label_text"]
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
    print(len(data))
    batch_size = 8
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

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: text.split(), stop_words=None, use_idf=True, norm=None
    )
    tfidf_matrix = vectorizer.fit_transform(doc_plaintexts)
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary to store IDF for each word
    word_idf = {word: idf for word, idf in zip(feature_names, vectorizer.idf_)}

    import math

    # Function to adjust attention scores by IDF and calculate mean
    def adjust_scores_by_idf(word_scores, word_idf):
        adjusted_scores = {
            word: np.mean(scores) * word_idf.get(word, 1)
            for word, scores in word_scores.items()
        }
        return adjusted_scores

    # Adjust the scores
    adjusted_scores = adjust_scores_by_idf(word_attention_aggregate, word_idf)

    # Sort words by the adjusted scores
    sorted_words_by_idf = sorted(
        adjusted_scores.items(), key=lambda item: item[1], reverse=True
    )

    # Display the top 10 words
    top_10_words_by_idf = sorted_words_by_idf[:20]
    print("Top 10 words adjusted by TF-IDF and their scores:")
    for word, score in top_10_words_by_idf:
        print(f"{word}: {score:.4f}")

    bottom_10_words_by_idf = sorted_words_by_idf[-10:]

    # Display the bottom 10 words and their scores
    print("Bottom 10 words adjusted by TF-IDF and their scores:")
    for word, score in reversed(bottom_10_words_by_idf):
        print(f"{word}: {score:.4f}")
    exit()

    # Function to calculate weighted mean of attention scores
    def weighted_mean(values, frequency):
        if frequency > 1:
            return sum(values) / len(values)  # * math.log(
            # frequency + 1
            # )  # Log to dampen the impact of very high frequencies
        return sum(values) / len(values)

    # Assuming word_attention_aggregate is a dictionary where:
    # key = word, value = list of attention scores
    # We also need a dictionary to count frequencies of each word
    word_frequencies = {
        word: len(scores) for word, scores in word_attention_aggregate.items()
    }

    # Compute the mean attention for each word and sort words by this new weighted metric
    sorted_words = sorted(
        word_attention_aggregate.items(),
        key=lambda item: weighted_mean(item[1], word_frequencies[item[0]]),
        reverse=True,
    )

    sorted_words_2 = sorted(
        word_attention_aggregate.items(),
        key=lambda item: weighted_mean(item[1], word_frequencies[item[0]]),
        reverse=False,
    )

    # Display the top 10 words
    top_10_words = sorted_words[:20]
    print("Top 10 weighted most attended words and their scores:")
    for word, scores in top_10_words:
        print(f"{word}: {weighted_mean(scores, word_frequencies[word]):.4f}")

    # Display the bottom 10 words
    top_10_words = sorted_words_2[:20]
    print("Bottom 10 weighted most attended words and their scores:")
    for word, scores in top_10_words:
        print(f"{word}: {weighted_mean(scores, word_frequencies[word]):.4f}")
