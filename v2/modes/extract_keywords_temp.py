import csv

import tqdm

from sklearn.metrics.pairwise import cosine_similarity


def pool_embeddings_for_words(token_embeddings, tokens, tokenizer):
    # Initialize a dictionary to hold the pooled embeddings for each full word
    word_embeddings = {}
    current_word_embeddings = []
    current_word = ""

    for idx, token in enumerate(tokens):
        # Skip special tokens like <s>, </s>, etc.
        if token in tokenizer.all_special_tokens:
            continue
        # New word starts with _
        if token.startswith("▁"):
            if current_word_embeddings:
                # Pool the embeddings for the previous word and add to the dictionary
                pooled_embedding = np.mean(current_word_embeddings, axis=0).tolist()
                word_embeddings[current_word] = pooled_embedding
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
        pooled_embedding = np.mean(current_word_embeddings, axis=0).tolist()
        word_embeddings[current_word] = pooled_embedding

    return word_embeddings


def compute_cosine_similarity(doc_embedding, word_embeddings):
    # Convert the document embedding to a 2D array
    doc_embedding_2d = np.array(doc_embedding).reshape(1, -1)

    # Initialize a dictionary to hold cosine similarities
    cosine_similarities = {}

    for word, word_embedding in word_embeddings.items():
        # Convert the word embedding to a 2D array
        word_embedding_2d = np.array(word_embedding).reshape(1, -1)

        # Compute the cosine similarity
        similarity = cosine_similarity(doc_embedding_2d, word_embedding_2d)[0][0]

        # Store the similarity
        cosine_similarities[word] = similarity

    # Sort the dictionary by similarity in descending order
    sorted_cosine_similarities = dict(
        sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_cosine_similarities


import json


def extract_doc_keywords(model, dataset, tokenizer, output_file):
    category_doc_word_similarities = {}

    with open(f"{output_file}", "w", newline="") as tsvfile:
        writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
        for d in tqdm(dataset["train"]):
            word_ids = d["input_ids"].tolist()[0]
            tokens = tokenizer.convert_ids_to_tokens(word_ids)
            label_text = d.pop("label_text")
            label_one_hot = d.pop("label")
            text = d.pop("text")
            language = d.pop("language")

            outputs = model(**d, output_hidden_states=True)

            text_embedding = outputs.hidden_states[-1][0][0, :].detach().numpy()
            token_embeddings = outputs.hidden_states[-1][0].detach().numpy()

            word_embeddings = pool_embeddings_for_words(
                token_embeddings, tokens, tokenizer
            )

            cosine_similarities = compute_cosine_similarity(
                text_embedding,
                word_embeddings,
            )

            writer.writerow(
                [
                    language,
                    label_text,
                    " ".join([str(x) for x in text_embedding.tolist()]),
                    # json.dumps(str(cosine_similarities)),
                    json.dumps(str(word_embeddings)),
                ]
            )

            continue

            if label_text not in category_doc_word_similarities:
                category_doc_word_similarities[label_text] = {}

            for k, v in cosine_similarities.items():
                if k not in category_doc_word_similarities[label_text]:
                    category_doc_word_similarities[label_text][k] = []
                category_doc_word_similarities[label_text][k].append(v)

        # for k, v in category_doc_word_similarities.items():
        #    print(f"CATEGORY {k}")
        #    print(v)
        #    exit()
