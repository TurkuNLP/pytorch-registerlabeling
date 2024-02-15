import sys

sys.path.append("venv/lib/python/python3.9/site-packages")

from dataclasses import dataclass, field
import tempfile

from jsonargparse import ArgumentParser, ActionConfigFile
import os

from dotenv import load_dotenv

load_dotenv()

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY", "")
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_DATA_DIR"] = ".wandb"


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
    remove_unused_cols: bool = True
    no_dynamic_padding: bool = False
    text_prefix: str = ""
    test_all_data: bool = False


@dataclass
class Model:
    name: str = "xlm-roberta-base"
    save: bool = True
    low_cpu_mem_usage: bool = False
    quantize: bool = False
    roberta_pooled: str = None
    sentence_transformer: bool = False


@dataclass
class Trainer:
    epochs: int = 30
    eval_step: float = 0
    learning_rate: float = 1e-5
    weight_decay: float = 0.05
    warmup_ratio: float = 0.01
    loss_alpha: float = 0.75
    loss_gamma: float = 2
    patience: int = 5
    best_model_metric: str = "eval_loss"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0


@dataclass
class Dataloader:
    train_batch_size: int = 8
    dev_batch_size: int = 128
    test_batch_size: int = 128
    balancing_sampler: bool = False
    pad_direction: str = "right"


@dataclass
class Peft:
    enable: bool = False
    lora_rank: int = 128
    lora_alpha: int = 256
    target_modules: str = "linear"


@dataclass
class Ray:
    learning_rate: list = field(default_factory=lambda: [1e-5, 1e-3])
    loss_alpha: list = field(default_factory=lambda: [0.15, 0.95])
    loss_gamma: list = field(default_factory=lambda: [1.5, 3.5])
    gpus_per_trial: int = 1


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument(
        "--method",
        "-m",
        default="finetune",
        choices=[
            "finetune",
            "hf_finetune",
            "predict",
            "extract_doc_embeddings",
            "extract_st_doc_embeddings",
            "ray_tune",
            "setfit_train",
        ],
    )
    parser.add_argument(
        "--embeddings",
        default="document",
        choices=["document", "tokens_mean", "tokens_max"],
    )
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--set_pad_token", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_fa2", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--no_tf32", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--train_using_embeddings", type=int, default=0)
    parser.add_argument("--root_path", default="/scratch/project_2009199")
    parser.add_argument("--tqdm_mininterval", type=float, default=0.5)
    parser.add_argument("--predict", type=bool, default=True)
    parser.add_argument("--predict_from_checkpoint", action="store_true")
    parser.add_argument("--data", type=Data, default=Data())
    parser.add_argument("--model", type=Model, default=Model())
    parser.add_argument("--trainer", type=Trainer, default=Trainer())
    parser.add_argument("--dataloader", type=Dataloader, default=Dataloader())
    parser.add_argument("--peft", type=Peft, default=Peft())
    parser.add_argument("--ray", type=Ray, default=Ray())
    parser.add_argument("--config", "-c", action=ActionConfigFile)
    parser.add_argument("--temp_test", action="store_true")

    cfg = parser.parse_args()

    print(parser.dump(cfg, format="yaml", skip_default=True))

    tempfile.tempdir = cfg.root_path + "/tmp"

    # Explicitly choose n GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in range(cfg.gpus))

    from v3.main import Main

    Main(cfg)
