import os
from pydoc import locate

from jsonargparse import ActionConfigFile, ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", "-s", type=int, default=42)
    parser.add_argument("--train_labels", default="all")
    parser.add_argument("--predict_labels", default="upper")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_batches", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model_path", default="xlm-roberta-large")
    parser.add_argument("--output_file", default="output/out.jsonl")
    parser.add_argument("--local_data", "-t", default="test_data")
    parser.add_argument("--stream_data", default="")

    cfg = parser.parse_args()

    print(parser.dump(cfg))
    locate(f"src.predict").run(cfg)
