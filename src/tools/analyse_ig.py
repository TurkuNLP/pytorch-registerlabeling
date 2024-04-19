import pandas as pd
import json
import numpy as np
from ..labels import decode_binary_labels, label_schemes
import ast
from heapq import nlargest, nsmallest


def run(cfg):

    classes = label_schemes["upper"]

    def compute_predicted_label(probs):
        probs = np.array(ast.literal_eval(probs))
        binary_vector = [(probs >= cfg.threshold).astype(int).tolist()]

        return decode_binary_labels(binary_vector, "upper")[0]

    # Define the column names
    columns = [
        "text_id",
        "predicted_probs",
        "true_label",
        "pred_label",
        "test_label",
        "token_contributions",
    ]

    # Load the TSV file
    df = pd.read_csv(
        "output/keywords_ig/fi.csv",
        delimiter="\t",
        header=None,
        names=columns,
        na_values="",  # Don't interpret NA as NaN!
        keep_default_na=False,
    )

    # df["predicted_probs"] = df["predicted_probs"].apply(ast.literal_eval)

    df["pred_label"] = df["predicted_probs"].apply(compute_predicted_label)

    # Remove rows with NaN in the predicted_label column
    df = df.dropna(subset=["pred_label"])

    # Further filter to exclude rows where predicted_label or true_label contains spaces
    df = df[~df["pred_label"].str.contains(" ")]
    df = df[~df["true_label"].str.contains(" ")]

    # Filter for test_label matching predicted_label
    df = df[df["test_label"] == df["pred_label"]]

    # Check the first few rows to understand the data structure
    print(df.head(10))

    df["token_contributions"] = df["token_contributions"].apply(json.loads)

    # Initialize nested dictionaries for each class
    contributions = {cls: {"CP": {}, "CN": {}, "IP": {}, "IN": {}} for cls in classes}
    global_doc_freq = {}

    def update_contributions(row):
        contributions_list = row["token_contributions"]
        seen_tokens = set()  # Set to track unique tokens in this document

        # Collect unique tokens from this document
        for token, _ in contributions_list:
            seen_tokens.add(token)

        # Update global document frequency for each unique token
        for token in seen_tokens:
            global_doc_freq[token] = global_doc_freq.get(token, 0) + 1
        # Remove least frequent tokens

    df.apply(update_contributions, axis=1)

    # Filter tokens appearing in less than 5 documents
    tokens_to_keep = {
        token.lower() for token, count in global_doc_freq.items() if count >= 3
    }

    print(tokens_to_keep)

    def update_class_contributions(row, n=5):
        true_label = row["true_label"]
        pred_label = row["pred_label"]
        contributions_list = row["token_contributions"]

        contributions_list = [
            (token.lower(), contrib)
            for token, contrib in contributions_list
            if any(char.isalpha() for char in token) and token.lower() in tokens_to_keep
        ]

        # Separate contributions into positive and negative
        positive_contribs = [
            (token, contrib) for token, contrib in contributions_list if contrib > 0
        ]
        negative_contribs = [
            (token, contrib) for token, contrib in contributions_list if contrib <= 0
        ]

        # Get top n positive and negative contributions
        top_pos = nlargest(n, positive_contribs, key=lambda x: x[1])
        top_neg = nsmallest(n, negative_contribs, key=lambda x: x[1])

        # Determine correct or incorrect prediction context
        pos_key = "CP" if true_label == pred_label else "IP"
        neg_key = "CN" if true_label == pred_label else "IN"

        # Update dictionaries for positive contributions
        for token, contrib in top_pos:
            contributions[true_label][pos_key][token] = (
                contributions[true_label][pos_key].get(token, 0) + contrib
            )

        # Update dictionaries for negative contributions
        for token, contrib in top_neg:
            contributions[true_label][neg_key][token] = (
                contributions[true_label][neg_key].get(token, 0) + contrib
            )

    # Apply the function to each row
    df.apply(update_class_contributions, axis=1)

    # Create separate DataFrames for the top 20 positive and negative tokens for each class
    top_tokens_by_class = {}
    for cls in classes:
        dfs = {}
        for ctype in ["CP", "CN", "IP", "IN"]:
            df_temp = pd.DataFrame(
                list(contributions[cls][ctype].items()),
                columns=["Token", f"Total_{ctype}_Contrib_{cls}"],
            )
            # Sort and get top 20
            df_sorted = df_temp.sort_values(
                by=f"Total_{ctype}_Contrib_{cls}", ascending=(ctype in ["CN", "IN"])
            )
            top20 = df_sorted.head(20)
            dfs[ctype] = top20
        top_tokens_by_class[cls] = dfs

    test_class = "IP"

    # Example: Access the DataFrames for class 'MT'
    print("Top 20 Positive Contributions in Correct Predictions for Class MT:")
    print(top_tokens_by_class[test_class]["CP"])
    print("Top 20 Negative Contributions in Correct Predictions for Class MT:")
    print(top_tokens_by_class[test_class]["CN"])
    print("Top 20 Positive Contributions in Incorrect Predictions for Class MT:")
    print(top_tokens_by_class[test_class]["IP"])
    print("Top 20 Negative Contributions in Incorrect Predictions for Class MT:")
    print(top_tokens_by_class[test_class]["IN"])

    exit()

    # PREVIOUS VERSION:

    # Initialize dictionaries to store the aggregated contributions
    CP = {}  # Correct Positive
    CN = {}  # Correct Negative
    IP = {}  # Incorrect Positive
    IN = {}  # Incorrect Negative

    # Function to update the dictionaries
    def update_contributions(row):
        true = row["true_label"]
        pred = row["pred_label"]
        contributions = row["token_contributions"]

        for token, contrib in contributions:
            if true == pred:  # Correct predictions
                if contrib > 0:
                    CP[token] = CP.get(token, 0) + contrib
                else:
                    CN[token] = CN.get(token, 0) + contrib
            else:  # Incorrect predictions
                if contrib > 0:
                    IP[token] = IP.get(token, 0) + contrib
                else:
                    IN[token] = IN.get(token, 0) + contrib

    # Apply the function to each row in the DataFrame
    df.apply(update_contributions, axis=1)

    # Create DataFrames for each dictionary
    df_CP = pd.DataFrame(
        list(CP.items()), columns=["Token", "Total_Pos_Contrib_Correct"]
    )
    df_CN = pd.DataFrame(
        list(CN.items()), columns=["Token", "Total_Neg_Contrib_Correct"]
    )
    df_IP = pd.DataFrame(
        list(IP.items()), columns=["Token", "Total_Pos_Contrib_Incorrect"]
    )
    df_IN = pd.DataFrame(
        list(IN.items()), columns=["Token", "Total_Neg_Contrib_Incorrect"]
    )

    print(df_CP)
    top_cp = df_CP.sort_values(by="Total_Pos_Contrib_Correct", ascending=False).head(10)

    print(top_cp)
    exit()
    # Optionally, merge these DataFrames into a single DataFrame for easier analysis
    dfs = [df_CP, df_CN, df_IP, df_IN]
    from functools import reduce

    df_final = reduce(
        lambda left, right: pd.merge(left, right, on="Token", how="outer"), dfs
    ).fillna(0)

    # View the final DataFrame
    print(df_final.head())
