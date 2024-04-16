import os
import csv
import json
import sys
import string
import re
from captum.attr import LayerIntegratedGradients
import captum
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

SPECIAL_TOKENS = ["<s>", "</s>"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_aggregated(target, txt, real_label):
    """ "
    This requires one target and one agg vector at a time
    Shows agg scores as colors
    """
    print("<html><body>")

    x = captum.attr.visualization.format_word_importances(
        [t for t, a in txt], [a for t, a in txt]
    )
    print(f"<b>prediction: {target}, real label: {real_label}</b>")
    print(f"""<table style="border:solid;">{x}</table>""")
    print("</body></html>")


def aggregate(inp, attrs, tokenizer):
    """Detokenize and merge attributions. This works for languages that use white spaces between the words."""
    tokens = tokenizer.convert_ids_to_tokens(inp.input_ids[0])
    attrs = attrs.cpu().tolist()
    res = []
    for token, a_val in zip(tokens, attrs):
        if token in SPECIAL_TOKENS:  # special tokens
            res.append((token, a_val))
        elif token.startswith("▁"):
            # This NOT is a continuation. A NEW word.
            res.append((token[1:], a_val))
            # print(res)
        else:  # we're continuing a word and need to choose the larger abs value of the two
            last_a_val = res[-1][1]
            # print("last val", last_a_val)
            if abs(a_val) < abs(last_a_val):  # past value bigger
                res[-1] = (res[-1][0] + token, last_a_val)
            else:  # new value bigger
                res[-1] = (res[-1][0] + token, a_val)

    return res


# # Forward on the model -> data in, prediction out, nothing fancy really
def predict(model, inputs, int_bs=None, attention_mask=None):
    pred = model(inputs, attention_mask=attention_mask)  # TODO: batch_size?
    return pred.logits  # return the output of the classification layer


def blank_reference_input(
    tokenized_input, blank_token_id
):  # b_encoding is the output of HFace tokenizer
    """
    Makes a tuple of blank (input_ids, token_type_ids, attention_mask)
    right now position_ids, and attention_mask simply point to tokenized_input
    """

    blank_input_ids = tokenized_input.input_ids.clone().detach()
    blank_input_ids[tokenized_input.special_tokens_mask == 0] = (
        blank_token_id  # blank out everything which is not special token
    )
    return blank_input_ids, tokenized_input.attention_mask


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

    text = "With the Biden administration proposing a variety of new taxes , it is worth revisiting the literature on how taxes impact economic growth . In 2012 , we published a review of the evidence , noting that most studies find negative impacts . However , many papers have been written since , some using more sophisticated empirical methods to identify a causal impact of taxes on economic growth . Below we review this new evidence , again confirming our original findings : Taxes , particularly on corporate and individual income , harm economic growth . The economic impacts of tax changes on economic growth , measured as a change in real GDP or the components of GDP such as consumption and investment , are difficult to measure . Some tax changes occur as a response to economic growth , and looking at a tax cut at a certain point in time could lead to the mistaken conclusion that tax cuts are bad for growth , since tax cuts are often enacted during economic downturns . For this reason , most of the literature in recent years , and reviewed below , has followed the methodology developed in Romer and Romer ( 2010 ) : Looking at unanticipated changes in tax policy , which economists refer to as â€œexogenous shocks . â€ There are other methodological challenges as well . Failure to control for other factors that impact economic growth , such as government spending and monetary policy , could understate or overstate the impact of taxes on growth . Some tax changes in particular may have stronger long-run impacts relative to the short run , such as corporate tax changes , and a study with only a limited time series would miss this effect . Finally , tax reforms involve many moving parts : Certain taxes may go up , while others may drop . This can make it difficult to characterize certain reforms as net tax increases or decreases , leading to mistaken interpretations of how taxes impact growth . We investigate papers in top economics journals and National Bureau of Economic Research ( NBER ) working papers over the past few years , considering both U . S . and international evidence . This research covers a wide variety of taxes , including income , consumption , and corporate taxation . All seven paper"

    # white space inbetween punctuation => for standard tokenisation
    text = re.sub("(?<! )(?=[:.,!?()])|(?<=[:.,!?()])(?! )", r" ", text)
    # Tokenize and make the blank reference input
    inp = tokenizer(
        text, return_tensors="pt", return_special_tokens_mask=True, truncation=True
    ).to(model.device)
    b_input_ids, b_attention_mask = blank_reference_input(
        inp, tokenizer.convert_tokens_to_ids("-")
    )

    def predict_f(inputs, attention_mask=None):
        return predict(model, inputs, attention_mask=attention_mask)

    # Here's where the magic happens
    lig = LayerIntegratedGradients(predict_f, model.roberta.embeddings)

    # make a prediction
    prediction = predict(model, inp.input_ids, attention_mask=inp.attention_mask)
    # get the classification layer outputs
    logits = prediction.cpu().detach().numpy()[0]
    # calculate sigmoid for each
    sigm = 1.0 / (1.0 + np.exp(-logits))
    # make the classification, threshold = 0.5
    target = np.array([pl > 0.5 for pl in sigm]).astype(int)
    # get the classifications' indices
    target = np.where(target == 1)
    # return None if no classification was done
    if len(target[0]) == 0:  # escape early if no prediction
        return None, None, sigm

    # loop over the targets => "[0]" to flatten the extra dimension, actually looping over all targets
    for tg in target[0]:
        attrs, delta = lig.attribute(
            inputs=(inp.input_ids, inp.attention_mask),
            baselines=(b_input_ids, b_attention_mask),
            return_convergence_delta=True,
            target=tuple([np.array(tg)]),
            internal_batch_size=10,
            n_steps=50,
        )
        # append the calculated and normalized scores to aggregated
        attrs_sum = attrs.sum(dim=-1).squeeze(0)
        attrs_sum = attrs_sum / torch.norm(attrs_sum)

        print(attrs_sum)

        aggregated_tg = aggregate(inp, attrs_sum, tokenizer)

        print_aggregated(tg, aggregated_tg, "SOME LABEL")

        exit()
