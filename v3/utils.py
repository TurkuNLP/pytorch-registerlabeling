from datetime import datetime
from pydoc import locate

from torch import cuda

from tqdm import tqdm

_print = print


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


def log_gpu_memory():
    for gpu in range(cuda.device_count()):
        allocated_memory = cuda.memory_allocated(gpu) / (1024**3)  # Convert to GB
        max_allocated_memory = cuda.max_memory_allocated(gpu) / (1024**3)
        print(f"[GPU-{gpu}]: {allocated_memory:.2f} ({max_allocated_memory:.2f}) GB")
