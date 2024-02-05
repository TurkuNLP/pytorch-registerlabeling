import json

import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification

from .utils import DotDict


class PooledRobertaForSequenceClassification(RobertaForSequenceClassification):
    def __init__(self, config, pooling):
        super().__init__(config)
        # You can add any additional customization if needed

        self.classifier = PooledRobertaClassificationHead(config, pooling)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.classifier(outputs[0])

        return DotDict(
            {
                "logits": logits,
                "hidden_states": outputs.hidden_states,
                "attentions": outputs.attentions,
            }
        )


class PooledRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, pooling):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        print(f"Classifier dropout: {classifier_dropout}")
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.pooling = pooling

    def forward(self, features, **kwargs):
        if self.pooling == "max":
            x, _ = features.max(dim=1)  # max pooling across the sequence dimension
        else:
            x = features.mean(dim=1)  # mean pooling across the sequence dimension
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        return logits


class AttDict(dict):
    __getattr__ = dict.__getitem__


class Cnf:
    def __init__(self, output_size):
        self.num_labels = output_size

    def to_dict(self):
        return vars(self)

    def to_json_string(self):
        return json.dumps(vars.self)


class ClassificationModel(nn.Module):
    def __init__(self, input_size=768, num_labels=25):
        super(ClassificationModel, self).__init__()
        self.config = Cnf(num_labels)
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_ids, labels=None):
        x = self.fc1(input_ids)
        x = self.bn1(x)  # Applying batch normalization
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)  # Applying batch normalization
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return {"logits": x}
