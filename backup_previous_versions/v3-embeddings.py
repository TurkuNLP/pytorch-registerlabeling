import os

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", "-m")
    parser.add_argument("--model_path", "-p")
    parser.add_argument("--labels", "-l", default="all")
    parser.add_argument("--language", "-d", default="en-fi-fr-sv-tr")
    parser.add_argument("--output", "-o", default="data/embeddings")

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.embeddings import Embeddings

    Embeddings(cfg)
