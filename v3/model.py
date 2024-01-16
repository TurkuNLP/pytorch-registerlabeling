from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import torch
from transformers import AutoModel, AutoConfig


class RegisterModel(torch.nn.Module):
    def __init__(self, base_model_name, NUM_CLASSES):
        super().__init__()
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name, config=self.config)
        self.lin_layer = torch.nn.Linear(self.config.hidden_size, NUM_CLASSES)

    def forward(self, inputs):
        features = self.base_model(inputs)
        outs = self.lin_layer(features)
        return outs


"""

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=25
)


optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
"""
