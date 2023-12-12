from torch import cuda


def log_gpu_memory():
    for gpu in range(cuda.device_count()):
        allocated_memory = cuda.memory_allocated(gpu) / (1024**3)  # Convert to GB
        max_allocated_memory = cuda.max_memory_allocated(gpu) / (1024**3)
        print(
            f"GPU {gpu}: Current Memory Allocated: {allocated_memory:.2f} GB, Max Memory Allocated: {max_allocated_memory:.2f} GB"
        )
