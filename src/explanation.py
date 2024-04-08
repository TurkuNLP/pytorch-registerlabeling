import numpy as np
import torch
from scipy.special import expit as sigmoid
from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

from .data import get_dataset
from .labels import label_schemes


def predict(dataset, model, cfg):
    model.eval()  # Ensure the model is in evaluation mode
    batch_size = 32
    logits_list = []
    model = model.to(cfg.device)

    with torch.no_grad():
        for i in tqdm(
            range(0, len(dataset["input_ids"]), batch_size), desc="Processing"
        ):
            batch = {
                k: torch.tensor(v[i : i + batch_size])
                for k, v in dataset.items()
                if k in ["input_ids", "attention_mask"]
            }

            batch = {k: v.to(cfg.device) for k, v in batch.items()}

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
    }


def run(cfg):
    dir_structure = f"{cfg.model_name}{('_'+cfg.path_suffix) if cfg.path_suffix else ''}/labels_{cfg.labels}/{cfg.train}_{cfg.dev}/seed_{cfg.seed}"
    model_output_dir = f"{cfg.model_output}/{dir_structure}"
    label_scheme = label_schemes[cfg.labels]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    dataset = get_dataset(cfg, tokenizer)["test"][:]
    if cfg.sample:
        dataset = dataset[: cfg.sample]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_output_dir, num_labels=len(label_scheme)
    )

    result = predict(dataset, model, cfg)

    probs = result["probs"]
    best_threshold = result["best_threshold"]

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

    result = predict(filtered_dataset, model, cfg)

    print(result["f1"])

    exit()
