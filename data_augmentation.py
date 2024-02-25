from dataclasses import dataclass
from dotenv import load_dotenv
import os

from jsonargparse import ArgumentParser

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
load_dotenv()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", "-m")
    parser.add_argument("--source", "-s")
    parser.add_argument("--target", "-t")
    parser.add_argument(
        "--deepl_auth_key", "-d", default=os.getenv("DEEPL_API_KEY", "")
    )

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.data_augmentation import Augment

    Augment(cfg)
