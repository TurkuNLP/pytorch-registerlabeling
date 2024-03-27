import numpy as np
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

from .data import balanced_dataloader, get_dataset
from .labels import decode_binary_labels, label_schemes


def predict(dataset, model):
    model.eval()  # Ensure the model is in evaluation mode

    batch_size = 32

    # Initialize an empty list to collect logits for all examples
    logits_list = []

    with torch.no_grad():
        # Iterate over the dataset in batches
        for i in tqdm(
            range(0, len(dataset["input_ids"]), batch_size), desc="Processing"
        ):
            # Create a batch by slicing each key in the dataset
            batch = {
                k: torch.tensor(v[i : i + batch_size])
                for k, v in dataset.items()
                if k in ["input_ids", "attention_mask"]
            }

            # Ensure batch tensors are on the right device (e.g., CPU or GPU)
            batch = {
                k: v.to("cpu") for k, v in batch.items()
            }  # Replace 'cpu' with 'cuda' if using a GPU

            # Forward pass through the model
            outputs = model(**batch)

            # Detach logits from the current batch and convert to list, then extend the logits_list
            batch_logits = outputs.logits.detach().cpu().tolist()
            logits_list.extend(batch_logits)

    # Assuming `logits_list` is your list of logits for each example and `labels` is the true labels matrix
    probs = sigmoid(np.array(logits_list))  # Convert logits to probabilities

    labels = np.array(dataset["label"])

    best_threshold, best_f1 = 0, 0
    for threshold in np.arange(0.3, 0.7, 0.05):
        binary_predictions = probs > threshold

        f1 = f1_score(labels[:100], binary_predictions, average="micro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f1)

    return probs, probs > best_threshold, best_threshold


def run(cfg):
    dir_structure = f"{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}"
    model_output_dir = f"{cfg.model_output}/{dir_structure}"
    label_scheme = label_schemes[cfg.labels]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    dataset = get_dataset(cfg, tokenizer)["test"][:50]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir, num_labels=len(label_scheme)
    )

    probs, binary_predictions, best_threshold = predict(dataset, model)

    # Calculate uncertainty measure for each label based on the best threshold
    uncertainty_measure = np.abs(probs - best_threshold)

    # Now, `uncertainty_measure` contains the uncertainty for each label of each example,
    # with smaller values indicating higher uncertainty.

    # If you want to aggregate this to get a single uncertainty value per example, you could take the mean or max, etc.
    # Here's an example of taking the mean uncertainty across all labels for each example:
    example_uncertainty = np.mean(uncertainty_measure, axis=1)

    print(example_uncertainty)

    mean_uncertainty = np.mean(example_uncertainty)
    std_uncertainty = np.std(example_uncertainty)

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

    indices_to_keep = ~high_uncertainty_indices

    # Convert the boolean array to actual indices
    selected_indices = np.where(indices_to_keep)[0]

    # Use the `select()` function to keep only the desired examples in the dataset
    filtered_dataset = Dataset.from_dict(dataset).select(selected_indices.tolist())

    probs, binary_predictions, best_threshold = predict(filtered_dataset, model)

    exit()

    """

    num_labels = 25  # Assuming there are 25 labels
    best_thresholds = np.zeros(num_labels)
    best_f1s = np.zeros(num_labels)

    # Iterate over each label
    for label_idx in range(num_labels):
        # Initialize the best F1 score and threshold for the current label
        best_f1 = 0
        best_threshold = 0

        # Try different thresholds for the current label
        for threshold in np.arange(0.3, 0.7, 0.05):
            # Apply threshold to get binary predictions for the current label
            binary_predictions = probs[:, label_idx] > threshold

            # Compute the F1 score for the current label and threshold
            f1 = f1_score(labels[:100, label_idx], binary_predictions, average="binary")

            # Update the best F1 score and threshold if the current F1 is better
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Store the best F1 score and threshold for the current label
        best_f1s[label_idx] = best_f1
        best_thresholds[label_idx] = best_threshold

    print(best_f1s)
    # Convert probabilities to binary predictions using the optimized thresholds for each label
    binary_predictions_optimized = np.zeros_like(probs)
    for label_idx in range(num_labels):
        binary_predictions_optimized[:, label_idx] = (
            probs[:, label_idx] > best_thresholds[label_idx]
        )

    # Now `binary_predictions_optimized` contains the binary predictions for each label using the optimized thresholds

    # Compute the overall F1 score for the whole dataset using micro-averaging
    overall_f1 = f1_score(labels[:100], binary_predictions_optimized, average="micro")

    print(f"Overall F1 Score (Micro-Averaged) with Optimized Thresholds: {overall_f1}")

    exit()
    probs = sigmoid(logits_list)
    """
