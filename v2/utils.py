from torch import cuda


def log_gpu_memory():
    for gpu in range(cuda.device_count()):
        allocated_memory = cuda.memory_allocated(gpu) / (1024**3)  # Convert to GB
        max_allocated_memory = cuda.max_memory_allocated(gpu) / (1024**3)
        print(
            f"GPU {gpu}: Current Memory Allocated: {allocated_memory:.2f} GB, Max Memory Allocated: {max_allocated_memory:.2f} GB"
        )


def infer_device_map():
    num_available_gpus = cuda.device_count()
    if num_available_gpus > 1:
        device_map = {0: [i for i in range(num_available_gpus)]}
    else:
        device_map = {}
    return device_map


from transformers import Trainer, TrainingArguments
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels
