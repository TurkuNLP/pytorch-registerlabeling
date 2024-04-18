import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from captum.attr import visualization as viz
from captum.attr import (
    IntegratedGradients,
    LayerConductance,
    LayerIntegratedGradients,
    LayerActivation,
)
from captum.attr import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# replace <PATH-TO-SAVED-MODEL> with the real path of the saved model
model_path = "/Users/erikhenriksson/Downloads/fold_7"

# load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, output_attentions=True
)
model.to(device)
model.eval()
model.zero_grad()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)


def predict(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    output = model(
        inputs,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    return output.start_logits, output.end_logits, output.attentions


def squad_pos_forward_func(
    inputs, token_type_ids=None, position_ids=None, attention_mask=None, position=0
):
    pred = model(
        inputs_embeds=inputs,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    pred = pred[position]
    return pred.max(1).values


cls_token_id = tokenizer.cls_token_id
ref_token_id = tokenizer.pad_token_id
sep_token_id = tokenizer.sep_token_id

text = "I am dog"

text_ids = tokenizer.encode(text, add_special_tokens=False)

# construct input token ids
input_ids = [cls_token_id] + text_ids + [sep_token_id]
ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]

seq_length = input_ids.size(1)
position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
ref_position_ids = torch.zeros(seq_length, dtype=torch.long, device=device)

position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
ref_position_ids = ref_position_ids.unsqueeze(0).expand_as(input_ids)


token_type_ids = torch.tensor(
    [[0 if i <= sep_ind else 1 for i in range(seq_length)]], device=device
)
ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)  # * -1

start_scores, end_scores, output_attentions = predict(
    input_ids,
    token_type_ids=token_type_ids,
    position_ids=position_ids,
    attention_mask=attention_mask,
)
