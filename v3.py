from dataclasses import dataclass
from jsonargparse import ArgumentParser, ActionConfigFile
import os

os.environ["TRANSFORMERS_CACHE"] = ".hf/transformers_cache"
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"


@dataclass
class Data:
    train: str = None
    dev: str = None
    test: str = None
    labels: str = "all"
    output_path: str = "output"
    source_path: str = "data"
    max_length: int = 512
    return_tensors: str = "pt"


@dataclass
class Model:
    name: str = "xlm-roberta-base"


@dataclass
class Trainer:
    epochs: int = 30
    wandb_project: str = "unnamed_project"
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.01
    loss_alpha: float = 0.75
    loss_gamma: float = 2
    dev_classification_report: bool = False
    patience: int = 5
    best_model_metric: str = "f1"


@dataclass
class Dataloader:
    train_batch_size: int = 8
    dev_batch_size: int = 8
    test_batch_size: int = 8


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--method", "-m", type=str, default="finetune")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data", type=Data, default=Data())
    parser.add_argument("--model", type=Model, default=Model())
    parser.add_argument("--trainer", type=Trainer, default=Trainer())
    parser.add_argument("--dataloader", type=Dataloader, default=Dataloader())
    parser.add_argument("--config", "-c", action=ActionConfigFile)

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.main import Main
    from v3.explore import explore

    if "explore" in cfg.method:
        explore(cfg)
    else:
        Main(cfg)
