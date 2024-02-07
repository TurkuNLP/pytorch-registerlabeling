import math
from datetime import datetime
from pydoc import locate

import numpy as np
import torch
from torch import cuda

_print = print


class DotDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]


def format_working_dir(model_name, data, seed):
    return "/".join(
        [
            data.output_path,
            model_name,
            f"labels_{data.labels}",
            "_".join([data.train or "", data.dev or data.train]),
            f"seed_{seed}",
        ]
    )


# Print with datetime
def print(*args, **kw):
    formatted_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _print(f"[{formatted_now}]", *args, **kw)


def get_torch_dtype(torch_dtype_str):
    return (
        locate(f"torch.{torch_dtype_str}")
        if torch_dtype_str not in [None, "auto"]
        else torch_dtype_str
    )


def get_linear_modules(model):
    print("Getting linear module names")
    print(model)

    linear_modules = set()

    for name, module in model.named_modules():
        name = name.lower()
        if "attention" in name and "self" in name and "Linear" in str(type(module)):
            linear_modules.add(name.split(".")[-1])

    print(f"Found linear modules: {linear_modules}")
    return list(linear_modules)


def model_has_improved(metric, patience_metric, best_score):
    if "loss" in metric:
        return patience_metric < best_score
    return patience_metric > best_score


def model_save_condition(cfg, best_score, best_starting_score):
    if cfg.model.save:
        if not cfg.resume:
            return True
        if (
            best_score is not False
            and best_starting_score is not False
            and model_has_improved(
                cfg.trainer.best_model_metric, best_score, best_starting_score
            )
        ):
            return True

    return False


def get_eval_step(data_len, eval_step):
    return (
        data_len
        if eval_step == 0
        else (eval_step if eval_step >= 1 else math.ceil(eval_step * data_len))
    )


def log_gpu_memory():
    for gpu in range(cuda.device_count()):
        allocated_memory = cuda.memory_allocated(gpu) / (1024**3)  # Convert to GB
        max_allocated_memory = cuda.max_memory_allocated(gpu) / (1024**3)
        print(f"[GPU-{gpu}]: {allocated_memory:.2f} ({max_allocated_memory:.2f}) GB")


def log_model_mean_weights(model):
    # Access the weights of the model
    weights = model.parameters()

    # Convert weights to NumPy array
    weights_np = []
    for weight in weights:
        weights_np.append(weight.detach().numpy())

    # Calculate mean and std
    mean_weights = np.array([np.mean(weight_tensor) for weight_tensor in weights_np])
    std_weights = np.array([np.std(weight_tensor) for weight_tensor in weights_np])

    print(f"Mean weights {mean_weights:.4f} ({std_weights:.4f})")


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def convert_embeddings_to_input(outputs, batch, sentence_transformer, device):
    embeddings = (
        average_pool(outputs.last_hidden_state, batch["attention_mask"])
        if not sentence_transformer
        else torch.Tensor(outputs).to(device)
    )

    return {"input_ids": embeddings}
