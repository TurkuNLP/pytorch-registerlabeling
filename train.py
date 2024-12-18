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
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--method", default="train")
    parser.add_argument("--model_name", default="xlm-roberta-large")
    parser.add_argument("--model_output", default="models")
    parser.add_argument("--predictions_output", default="predictions")
    parser.add_argument("--path_suffix", default="")
    parser.add_argument("--config", "-c", action=ActionConfigFile)
    parser.add_argument("--just_evaluate", action="store_true")
    parser.add_argument("--speedtest", action="store_true")
    parser.add_argument("--no_save", action="store_true")

    # Data
    parser.add_argument("--train", "-t", default="en-fi-fr-sv-tr")
    parser.add_argument("--dev", "-d", default="")
    parser.add_argument(
        "--test", "-te", default="en-fi-fr-sv-tr-ar-ca-es-fa-hi-id-jp-no-pt-ur-zh"
    )
    parser.add_argument("--labels", default="all")
    parser.add_argument("--predict_labels", default="")
    parser.add_argument("--use_fold", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=10)
    parser.add_argument("--use_gz", action="store_true")
    parser.add_argument("--remove_columns", default="")
    parser.add_argument("--mask_alphabets", type=bool, default=False)
    parser.add_argument("--cachedir", default="")
    parser.add_argument("--save_predictions", type=bool, default=True)
    parser.add_argument("--multilabel_eval", default="")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--sample_subset", type=int, default=0)

    # Trainer
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train_batch_size", "-bt", type=int, default=8)
    parser.add_argument("--eval_batch_size", "-bd", type=int, default=8)
    parser.add_argument("--patience", "-p", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--loss_alpha", type=float, default=1)
    parser.add_argument("--loss_gamma", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--fa2", action="store_true")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--balanced_dataloader", action="store_true")
    parser.add_argument("--mean_pooling", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)

    # Peft
    parser.add_argument("--peft", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--target_modules", default="")
    cfg = parser.parse_args()

    if not cfg.train:
        print("--train is required.")
        exit()
    cfg.dev = cfg.train if not cfg.dev else cfg.dev
    cfg.test = cfg.dev if not cfg.test else cfg.test
    cfg.predict_labels = cfg.labels if not cfg.predict_labels else cfg.predict_labels

    print(parser.dump(cfg))
    locate(f"src.{cfg.method}").run(cfg)
