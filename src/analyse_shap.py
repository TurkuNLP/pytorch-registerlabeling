import json
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .data import get_dataset
from .labels import label_schemes
import scipy as sp
import shap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(cfg):
    label_scheme = label_schemes[cfg.labels]

    # Load the model with the correct number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_path, num_labels=len(label_scheme)
    ).to(device)
    model.eval()

    # Get the original model's name and init tokenizer
    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

       # Prepare dataset
    dataset = get_dataset(cfg, tokenizer)["test"]
    if cfg.sample:
        dataset = dataset.select(range(cfg.sample))

    # Function to process texts into model outputs
    def f(texts):
        # Encode texts
        encodings = tokenizer(texts, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        encodings = {key: value.to(device) for key, value in encodings.items()}
        # Model forward pass
        with torch.no_grad():
            outputs = model(**encodings)
        # Convert logits to probabilities
        probabilities = torch.sigmoid(outputs.logits).cpu().numpy()
        # Convert probabilities to log-odds (more suitable for SHAP)
        log_odds = sp.special.logit(probabilities)
        return log_odds

    # Tokenize background data
    background_texts = dataset["text"][:100]  # Select background examples
    background_data = [tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt") for text in background_texts]

    # Creating a SHAP explainer with a background dataset
    explainer = shap.Explainer(f, background_data)

    # Calculating SHAP values for a subset of data
    shap_values = explainer(dataset["text"][0:5])

    # Plot the SHAP values for a label 'NA'
    shap.plots.bar(shap_values[..., label_scheme.index("NA")].mean(0))