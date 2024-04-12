import os
from pydoc import locate

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--model_name", default="xlm-roberta-large")
    parser.add_argument("--path_suffix", default="")
    parser.add_argument("--test", default="mono")
    parser.add_argument("--target", default="main")
    parser.add_argument("--labels", default="all")
    parser.add_argument("--predict_labels", default="")
    cfg = parser.parse_args()

    cfg.predict_labels = cfg.labels if not cfg.predict_labels else cfg.predict_labels

    print(parser.dump(cfg))
    locate(f"src.results").run(cfg)
