import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
from ..data import get_dataset
from ..labels import label_schemes

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
                    v, padding="max_length", max_length=256, truncation=True
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
        return probabilities[:, 6]

    # build an explainer using a token masker
    explainer = shap.Explainer(f, tokenizer)

    text = "TEXT"

    shap_values = explainer({"text": [text]})

    html_content = shap.plots.text(shap_values[0], display=False)

    # Write the HTML string to a file
    with open("visualization.html", "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
