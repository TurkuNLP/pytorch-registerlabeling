import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
from .data import get_dataset
from .labels import label_schemes

import scipy as sp

import shap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # define a prediction function

    def f(x):
        tv = torch.tensor(
            [
                tokenizer.encode(
                    v, padding="max_length", max_length=512, truncation=True
                )
                for v in x
            ]
        ).to(device)
        outputs = model(tv)[0].detach().cpu().numpy()
        probabilities = torch.sigmoid(
            torch.from_numpy(outputs)
        ).numpy()  # sigmoid activation
        # print("a")
        # print(probabilities)
        return probabilities[:, 1]

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)

    shap_values = explainer(dataset["text"])

    fig = plt.figure()

    # Generate the plot
    shap.plots.bar(shap_values.abs.max(0), show=False)

    plt.savefig("shap_plot2.png")
