from dataclasses import dataclass
from jsonargparse import ArgumentParser, ActionConfigFile
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["WANDB_WATCH"] = "all"


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
    concat_small: bool = False


@dataclass
class Model:
    name: str = "xlm-roberta-base"
    compile: bool = False
    save: bool = True
    resume: str = None


@dataclass
class Trainer:
    epochs: int = 30
    learning_rate: float = 1e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.01
    loss_alpha: float = 0.75
    loss_gamma: float = 2
    patience: int = 5
    best_model_metric: str = "dev/f1"
    gradient_accumulation_steps: int = 1


@dataclass
class Dataloader:
    train_batch_size: int = 8
    dev_batch_size: int = 8
    test_batch_size: int = 8
    balancing_sampler: bool = False


@dataclass
class Peft:
    enable: bool = False
    lora_rank: int = 128
    lora_alpha: int = 256
    target_modules: str = "linear"


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
    parser.add_argument("--peft", type=Peft, default=Peft())
    parser.add_argument("--config", "-c", action=ActionConfigFile)

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.main import Main
    from v3.explore import explore

    if "explore" in cfg.method:
        explore(cfg)
    else:
        Main(cfg)
