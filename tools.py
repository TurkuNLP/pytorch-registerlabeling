import os
import sys
from pydoc import locate

sys.path.append(
    f"venv/lib/python{'.'.join(map(str, sys.version_info[:3]))}/site-packages"
)

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="analyse_entropy")
    parser.add_argument("--labels", default="all")
    parser.add_argument("--train", default="en")
    parser.add_argument("--dev", default="")
    parser.add_argument("--test", default="")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_path", default="xlm-roberta-large")
    parser.add_argument("--source_data_path", default="")
    parser.add_argument("--just_evaluate", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_fold", type=int, default=0)
    parser.add_argument("--use_gz", action="store_true")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--keep_columns", type=bool, default=True)

    cfg = parser.parse_args()

    cfg.dev = cfg.train if not cfg.dev else cfg.dev
    cfg.test = cfg.dev if not cfg.test else cfg.test

    print(parser.dump(cfg))
    locate(f"src.tools.{cfg.task}").run(cfg)
