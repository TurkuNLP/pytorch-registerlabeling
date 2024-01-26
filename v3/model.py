from transformers import RobertaForSequenceClassification
import torch
import torch.nn as nn


class DummyDotDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]


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
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        logits = self.classifier(outputs[0])

        return DummyDotDict(
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
