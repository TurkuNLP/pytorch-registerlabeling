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

    model = model.to("cuda")

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
                k: v.to("cuda") for k, v in batch.items()
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

        f1 = f1_score(labels, binary_predictions, average="micro")

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(best_f1)

    return probs, probs > best_threshold, best_threshold


def run(cfg):
    # sample_size = 1000
    dir_structure = f"{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}"
    model_output_dir = f"{cfg.model_output}/{dir_structure}"
    label_scheme = label_schemes[cfg.labels]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    dataset = get_dataset(cfg, tokenizer)["test"][:]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir, num_labels=len(label_scheme)
    )

    probs, binary_predictions, best_threshold = predict(dataset, model)

    method = "entropy"

    if method == "threshold":

        # Calculate uncertainty measure for each label based on the best threshold
        uncertainty_measure = np.abs(probs - best_threshold)

        # Now, `uncertainty_measure` contains the uncertainty for each label of each example,
        # with smaller values indicating higher uncertainty.

        # If you want to aggregate this to get a single uncertainty value per example, you could take the mean or max, etc.
        # Here's an example of taking the mean uncertainty across all labels for each example:
        example_uncertainty = np.mean(uncertainty_measure, axis=1)

        print(example_uncertainty)

    elif method == "entropy":

        # Assuming `probs` is your array of label probabilities
        def binary_entropy(probs):
            # Ensure no log(0) issue; clip probabilities to avoid log(0). Adjust the epsilon if needed.
            epsilon = 1e-9
            probs = np.clip(probs, epsilon, 1 - epsilon)

            # Calculate the entropy for each label
            entropy = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))

            return entropy

        # Calculate the entropy for each label in your predictions
        label_entropies = binary_entropy(probs)

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

    probs, binary_predictions, best_threshold = predict(filtered_dataset, model)

    exit()
