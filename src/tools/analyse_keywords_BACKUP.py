import pandas as pd

import json
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..labels import flatten_labels


def sort_and_extract_top_words(filtered_tf_idf_scores, top_n=20):
    """
    Sort the filtered TF-IDF scores for each category and extract the top N words.

    :param filtered_tf_idf_scores: Dict of filtered TF-IDF scores for each term in each category.
    :param top_n: Number of top terms to extract for each category.
    :return: Dict of sorted and top N words for each category.
    """
    top_words_per_category = {}

    for category, terms_scores in filtered_tf_idf_scores.items():
        # Sort terms by their TF-IDF scores in descending order
        sorted_terms = sorted(
            terms_scores.items(), key=lambda item: item[1], reverse=True
        )

        # Extract the top N words
        top_words = sorted_terms[:top_n]

        # Store the top words and their scores for the category
        top_words_per_category[category] = top_words

    return top_words_per_category


def calculate_tf_idf_with_cosine_similarity(keywords, method="cosine_sim"):
    """
    Calculate TF-IDF scores using cosine similarity values for each term in each category.

    :param keywords: Dict of categories, each with a dict of words and their cosine similarity scores.
    :return: Dict of TF-IDF scores for terms within each category.
    """
    # Calculate document frequency for each term across all categories
    doc_count = len(keywords)
    term_doc_frequency = {}
    for category in keywords:
        for term, scores in keywords[category].items():
            if term.strip():
                term_doc_frequency[term] = term_doc_frequency.get(term, 0) + (
                    np.mean(scores) if method == "cosine_sim" else 1
                )

    # Calculate IDF scores
    idf_scores = {
        term: math.log(doc_count / df) for term, df in term_doc_frequency.items()
    }

    # Calculate TF (average cosine similarity) and then TF-IDF scores
    tf_idf_scores = {category: {} for category in keywords}
    for category in keywords:
        category_words = np.sum(
            np.sum(y) if method == "cosine_sim" else len(y)
            for _, y in keywords[category].items()
        )
        for term, scores in keywords[category].items():
            if term.strip() and term in idf_scores:
                tf = (
                    (math.log(len(scores)) * np.mean(scores))
                    if method == "cosine_sim"
                    else len(scores)
                ) / category_words
                tf_idf_scores[category][term] = tf * idf_scores[term]

    return tf_idf_scores


def run(cfg):
    path = cfg.source_data_path
    df = pd.read_csv(
        f"{path}/keywords.tsv",
        sep="\t",
        header=None,
        names=["language", "label", "doc_embedding", "words"],
        na_values="",  # Don't interpret NA as NaN!
        keep_default_na=False,
    )

    print(df.head(10))
    wi = 0

    keywords = {}

    df["label_flat"] = df["label"].apply(flatten_labels)
    df = df.explode("label_flat")

    # Group the DataFrame by language
    language_groups = df.groupby("language")

    # Iterate over each group
    for language, language_group in language_groups:
        keywords[language] = {}
        # print(f"=============== {language} ===============")
        # Group the DataFrame by language
        label_groups = language_group.groupby("label_flat")
        for category, label_group in label_groups:
            # print(f"Label: {category}\n==========")
            keywords[language][category] = {}
            cat_wi = 0
            for _, row in label_group.iterrows():

                seen_words = set()
                for word in json.loads(row["words"]):
                    # if word[0] in seen_words:
                    #    continue
                    seen_words.add(word[0])
                    wi += 1
                    cat_wi += 1

                    # if " " in word[0]:
                    #    continue

                    if word[0] in keywords[language][category]:
                        keywords[language][category][word[0]].append(word[1])
                    else:
                        keywords[language][category][word[0]] = [word[1]]

        # Calculate TF-IDF scores using cosine similarity values
        tf_idf_scores = calculate_tf_idf_with_cosine_similarity(keywords[language])
        top_n = 20  # Number of top terms to extract
        top_words_per_category = sort_and_extract_top_words(tf_idf_scores, top_n)

        # Display the top words for each category (optional, for verification)
        for category, top_words in top_words_per_category.items():
            print(f"Category: {category}\n==========")
            for word, score in top_words:
                print(f"    {word}: {score}")
            print("\n")

    print(wi)
