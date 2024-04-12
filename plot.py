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
    parser.add_argument("--what", default="entropy")
    parser.add_argument("--labels", default="all")
    parser.add_argument("--train", "-t", default="en-fi-fr-sv-tr")
    parser.add_argument("--dev", "-d", default="")
    parser.add_argument("--test", "-te", default="en-fi-fr-sv-tr")
    parser.add_argument("--model_name", default="xlm-roberta-large")
    parser.add_argument("--just_evaluate", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_fold", type=int, default=0)
    parser.add_argument("--use_gz", action="store_true")
    parser.add_argument("--keep_columns", type=bool, default=True)

    cfg = parser.parse_args()

    cfg.dev = cfg.train if not cfg.dev else cfg.dev
    cfg.test = cfg.dev if not cfg.test else cfg.test

    print(parser.dump(cfg))
    locate(f"src.plot_{cfg.what}").run(cfg)
