import os

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_DISABLED"] = "true"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", "-ro", default="/scratch/project_2009056")
    parser.add_argument("--path_suffix", "-ps", default="")
    parser.add_argument("--peft", "-pe", action="store_true")
    parser.add_argument("--method", "-me", default="train")
    parser.add_argument("--labels", "-l", default="all")
    parser.add_argument("--train", "-tr", default="")
    parser.add_argument("--dev", "-de", default="")
    parser.add_argument("--test", "-te", default="")
    parser.add_argument("--model_name", "-mn", default="BAAI/bge-m3-retromae")
    parser.add_argument("--max_length", "-ml", type=int, default=512)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", "-tb", type=int, default=8)
    parser.add_argument("--eval_batch_size", "-eb", type=int, default=8)
    parser.add_argument("--seed", "-se", type=int, default=42)
    parser.add_argument("--grad_acc_steps", "-ga", type=int, default=1)
    parser.add_argument("--loss_alpha", "-la", type=float, default=0.5)
    parser.add_argument("--loss_gamma", "-lg", type=int, default=1)

    cfg = parser.parse_args()

    if not cfg.train:
        print("--train missing.")
        exit()
    cfg.dev = cfg.train if not cfg.dev else cfg.dev
    cfg.test = cfg.dev if not cfg.test else cfg.test

    print(parser.dump(cfg))

    from v4 import main

    main.run(cfg)
