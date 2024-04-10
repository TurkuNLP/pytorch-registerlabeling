import json

import matplotlib.pyplot as plt

import numpy as np
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

from .data import get_dataset
from .labels import label_schemes, decode_binary_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
entropy = lambda probs: -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))


def predict(dataset, model, tokenizer, cfg):
    model.eval()  # Ensure the model is in evaluation mode
    logits_list = []
    texts = []
    model = model.to(device)
    decoded_texts = []
    with torch.no_grad():
        for i in tqdm(
            range(0, len(dataset["input_ids"]), cfg.batch_size), desc="Processing"
        ):
            batch = {
                k: torch.tensor(v[i : i + cfg.batch_size])
                for k, v in dataset.items()
                if k in ["input_ids", "attention_mask"]
            }
            batch_token_ids = dataset["input_ids"][i : i + cfg.batch_size]
            for token_ids in batch_token_ids:
                decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
                decoded_texts.append(decoded_text)

            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            batch_logits = outputs.logits.detach().cpu().tolist()
            logits_list.extend(batch_logits)

    probs = sigmoid(np.array(logits_list))

    labels = np.array(dataset["label"])

    best_threshold, best_f1 = 0, 0
    for threshold in np.arange(0.3, 0.7, 0.05):
        binary_predictions = probs > threshold

        f1 = f1_score(labels, binary_predictions, average="micro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return {
        "f1": best_f1,
        "probs": probs,
        "preds": probs > best_threshold,
        "threshold": best_threshold,
        "texts": decoded_texts,
        "true_labels": decode_binary_labels(labels, cfg.labels),
        "pred_labels": decode_binary_labels(binary_predictions, cfg.labels),
    }


def run(cfg):
    label_scheme = label_schemes[cfg.labels]

    # Init model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model.eval()

    # Get the original model's name and init tokenizer
    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    dataset = get_dataset(cfg, tokenizer)["test"]
    if cfg.sample:
        dataset = dataset.select(range(cfg.sample))

    dataset = dataset[:]

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path, num_labels=len(label_scheme)
    )

    result = predict(dataset, model, tokenizer, cfg)
    entropies = entropy(result["probs"])

    # Calculate mean, median, and max entropy for each example
    mean_entropies = [np.mean(e) for e in entropies]
    median_entropies = [np.median(e) for e in entropies]
    max_entropies = [np.max(e) for e in entropies]

    # Define a list of tuples containing the data and file names
    entropy_data = [
        (
            mean_entropies,
            "mean_entropies_distribution.png",
            "skyblue",
            "Distribution of Mean Entropies",
        ),
        (
            median_entropies,
            "median_entropies_distribution.png",
            "lightgreen",
            "Distribution of Median Entropies",
        ),
        (
            max_entropies,
            "max_entropies_distribution.png",
            "salmon",
            "Distribution of Max Entropies",
        ),
    ]

    # Directory where plots will be saved
    save_dir = "output/"

    # Loop through each data set and save the histogram
    for data, filename, color, title in entropy_data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=10, color=color)
        ax.set_title(title)
        plt.tight_layout()
        full_path = f"{save_dir}{filename}"
        plt.savefig(full_path, dpi=200)  # Save the figure
        plt.close(fig)  # Close the figure

        print(f"Saved: {full_path}")

    exit()

    for method in ["mean", "median", "max"]:

        avg_entropies = methods[method](label_entropies, axis=1)

        mean_entropies = [
            np.mean(example_entropy) for example_entropy in label_entropies
        ]
        median_entropies = [
            np.median(example_entropy) for example_entropy in label_entropies
        ]
        max_entropies = [np.max(example_entropy) for example_entropy in label_entropies]

        print(label_entropies)

        low_percentile, high_percentile = np.percentile(mean_entropies, [10, 90])

        # Pair texts with their entropies and sort by entropy
        text_entropy_pairs = sorted(
            zip(result["texts"], result["true_labels"], mean_entropies),
            key=lambda x: x[1],
        )

        # Filter based on calculated percentiles
        low_entropy_texts = [
            (text, label)
            for text, label, entropy in text_entropy_pairs
            if entropy <= low_percentile
        ]
        high_entropy_texts = [
            (text, label)
            for text, label, entropy in text_entropy_pairs
            if entropy >= high_percentile
        ]

        print("Low Entropy Texts:", low_entropy_texts)
        print("High Entropy Texts:", high_entropy_texts)

        exit()

    # To get an uncertainty measure per example, you can take the mean entropy across all labels for each example
    example_uncertainty = np.mean(label_entropies, axis=1)

    print(example_uncertainty)

    mean_uncertainty = np.mean(example_uncertainty)
    std_uncertainty = np.std(example_uncertainty)

    method = "std"

    if method == "std":

        # You can adjust the multiplier for the standard deviation based on how conservative you want to be
        # A larger multiplier will result in discarding fewer examples, being more conservative
        factor = 1  # This can be adjusted
        uncertainty_threshold = mean_uncertainty + factor * std_uncertainty

        # Determine which examples exceed the uncertainty threshold
        high_uncertainty_indices = example_uncertainty > uncertainty_threshold

        # Print out how many examples are considered too uncertain
        print(
            f"Number of examples considered too uncertain: {np.sum(high_uncertainty_indices)}"
        )
    elif method == "percentile":
        percentile = 75  # This means we discard the top 25% most uncertain examples
        uncertainty_threshold_percentile = np.percentile(
            example_uncertainty, percentile
        )

        # Determine which examples exceed the uncertainty percentile threshold
        high_uncertainty_indices = (
            example_uncertainty > uncertainty_threshold_percentile
        )

        print(
            f"Number of examples considered too uncertain (percentile approach): {np.sum(high_uncertainty_indices)}"
        )

    indices_to_keep = ~high_uncertainty_indices

    # Convert the boolean array to actual indices
    selected_indices = np.where(indices_to_keep)[0]

    # Use the `select()` function to keep only the desired examples in the dataset
    filtered_dataset = Dataset.from_dict(dataset).select(selected_indices.tolist())[:]

    result = predict(filtered_dataset, model, cfg)

    print(result["f1"])

    exit()
