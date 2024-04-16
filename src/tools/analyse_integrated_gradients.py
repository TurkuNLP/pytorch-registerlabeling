import os
import csv
import json
import sys
import string
import re
from captum.attr import LayerIntegratedGradients
import numpy as np
import torch
from tqdm import tqdm

from datasets import concatenate_datasets

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import get_dataset

csv.field_size_limit(sys.maxsize)

init_batch_data = lambda: {
    "input_ids": [],
    "attention_mask": [],
    "label": [],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def analyse(d, model, tokenizer):

    # Define model output
    def model_output(inputs):
        logits = model(inputs)[0]
        print(logits)
        return logits

    # Define model input
    model_input = model.roberta.embeddings

    lig = LayerIntegratedGradients(model_output, model_input)

    def construct_input_and_baseline(text):

        max_length = 510
        baseline_token_id = tokenizer.pad_token_id
        sep_token_id = tokenizer.sep_token_id
        cls_token_id = tokenizer.cls_token_id

        text_ids = tokenizer.encode(
            text, max_length=max_length, truncation=True, add_special_tokens=False
        )

        test = tokenizer(
            text, return_tensors="pt", return_special_tokens_mask=True, truncation=True
        )

        input_ids = [cls_token_id] + text_ids + [sep_token_id]
        token_list = tokenizer.convert_ids_to_tokens(input_ids)

        # print(input_ids)

        # print(test["input_ids"][0].tolist())

        # exit()

        baseline_input_ids = (
            [cls_token_id] + [baseline_token_id] * len(text_ids) + [sep_token_id]
        )
        return (
            torch.tensor([input_ids], device="cpu"),
            torch.tensor([baseline_input_ids], device="cpu"),
            token_list,
        )

    text = d["text"]
    text = "With the Biden administration proposing a variety of new taxes , it is worth revisiting the literature on how taxes impact economic growth . In 2012 , we published a review of the evidence , noting that most studies find negative impacts . However , many papers have been written since , some using more sophisticated empirical methods to identify a causal impact of taxes on economic growth . Below we review this new evidence , again confirming our original findings : Taxes , particularly on corporate and individual income , harm economic growth . The economic impacts of tax changes on economic growth , measured as a change in real GDP or the components of GDP such as consumption and investment , are difficult to measure . Some tax changes occur as a response to economic growth , and looking at a tax cut at a certain point in time could lead to the mistaken conclusion that tax cuts are bad for growth , since tax cuts are often enacted during economic downturns . For this reason , most of the literature in recent years , and reviewed below , has followed the methodology developed in Romer and Romer ( 2010 ) : Looking at unanticipated changes in tax policy , which economists refer to as â€œexogenous shocks . â€ There are other methodological challenges as well . Failure to control for other factors that impact economic growth , such as government spending and monetary policy , could understate or overstate the impact of taxes on growth . Some tax changes in particular may have stronger long-run impacts relative to the short run , such as corporate tax changes , and a study with only a limited time series would miss this effect . Finally , tax reforms involve many moving parts : Certain taxes may go up , while others may drop . This can make it difficult to characterize certain reforms as net tax increases or decreases , leading to mistaken interpretations of how taxes impact growth . We investigate papers in top economics journals and National Bureau of Economic Research ( NBER ) working papers over the past few years , considering both U . S . and international evidence . This research covers a wide variety of taxes , including income , consumption , and corporate taxation . All seven paper"
    text = re.sub("(?<! )(?=[:.,!?()])|(?<=[:.,!?()])(?! )", r" ", text)
    print(text)

    input_ids, baseline_input_ids, all_tokens = construct_input_and_baseline(text)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,
        return_convergence_delta=True,
        internal_batch_size=1,
        target=tuple([np.array(6)]),
    )
    print(attributions.size())

    def summarize_attributions(attributions):

        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        return attributions

    attributions_sum = summarize_attributions(attributions)
    print(attributions_sum.size())

    from captum.attr import visualization as viz

    score_vis = viz.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=torch.max(model(input_ids)[0]),
        pred_class=torch.argmax(model(input_ids)[0]).numpy(),
        true_class=1,
        attr_class=text,
        attr_score=attributions_sum.sum(),
        raw_input_ids=all_tokens,
        convergence_score=delta,
    )

    from IPython.core.display import display, HTML
    import IPython.display as ipd

    # Create the visualization (this would display in a Jupyter notebook)
    visualization = viz.visualize_text([score_vis])

    # Capturing the output in an HTML format (appropriate for a script)
    html_output = ipd.HTML(data=visualization.data)
    html_content = html_output.data  # This extracts the HTML string

    # Write the HTML string to a file
    with open("visualization.html", "w", encoding="utf-8") as html_file:
        html_file.write(html_content)

    print("Visualization saved to visualization.html")

    exit()


def run(cfg):
    if not cfg.train == cfg.dev == cfg.test:
        print("This script only works with the same dataset for train, dev and test")
        exit()
    path = "output/keywords"
    os.makedirs(path, exist_ok=True)
    path = f"{path}/{cfg.train}.csv"
    # Init model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model.eval()

    # Get the original model's name and init tokenizer
    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    dataset = get_dataset(cfg, tokenizer)
    dataset.set_format(type="torch")
    dataset["train"] = dataset["train"].filter(
        lambda example: "LY" in example["label_text"]
    )
    dataset["test"] = dataset["test"].filter(
        lambda example: "LY" in example["label_text"]
    )
    dataset["dev"] = dataset["dev"].filter(
        lambda example: "LY" in example["label_text"]
    )

    if cfg.sample:
        dataset["train"] = dataset["train"].select(
            range(min(cfg.sample, len(dataset["train"])))
        )
        dataset["test"] = dataset["test"].select(
            range(min(cfg.sample, len(dataset["test"])))
        )
        dataset["dev"] = dataset["dev"].select(
            range(min(cfg.sample, len(dataset["dev"])))
        )

    print(dataset["train"])

    data = concatenate_datasets([dataset["train"], dataset["dev"], dataset["test"]])
    batch_size = 4
    batch_data = init_batch_data()
    for d in tqdm(data):
        analyse(d, model, tokenizer)
