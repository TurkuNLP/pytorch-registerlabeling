import json
import torch.nn as nn


class Cnf:
    def __init__(self, output_size):
        self.num_labels = output_size

    def to_dict(self):
        return vars(self)

    def to_json_string(self):
        return json.dumps(vars.self)


class AttDict(dict):
    __getattr__ = dict.__getitem__


class GeminiModel(nn.Module):
    def __init__(self, input_size=768, output_size=25):
        super(GeminiModel, self).__init__()
        self.config = Cnf(output_size)
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, output_size)
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
