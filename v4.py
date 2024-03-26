import os
from pydoc import locate

from jsonargparse import ActionConfigFile, ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    parser = ArgumentParser()
    # Main args
    parser.add_argument("--model_name", "-m", default="xlm-roberta-large")
    parser.add_argument("--method", "-me", default="train")
    parser.add_argument("--model_output", "-o", default="models")
    parser.add_argument("--path_suffix", default="")
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--config", "-c", action=ActionConfigFile)

    # Data
    parser.add_argument("--train", "-t", default="")
    parser.add_argument("--dev", "-d", default="")
    parser.add_argument("--test", "-te", default="")
    parser.add_argument("--text_prefix", default="")
    parser.add_argument("--labels", default="all")

    # Trainer
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", "-bt", type=int, default=8)
    parser.add_argument("--eval_batch_size", "-bd", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--loss_alpha", type=float, default=0.5)
    parser.add_argument("--loss_gamma", type=int, default=1)
    parser.add_argument("--device", default="cuda")

    # Peft
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    cfg = parser.parse_args()

    if not cfg.train:
        print("--train is required.")
        exit()
    cfg.dev = cfg.train if not cfg.dev else cfg.dev
    cfg.test = cfg.dev if not cfg.test else cfg.test

    print(parser.dump(cfg))

    locate(f"v4.{cfg.method}").run(cfg)
