import os
from argparse import ArgumentParser
from v2 import mistral, mistral_predict

os.environ["TRANSFORMERS_CACHE"] = ".hf/transformers_cache"
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"

parser = ArgumentParser()
parser.add_argument("--mode", "-m", default="train")
options = parser.parse_args()

if options.mode == "train":
    mistral.run()

elif options.mode == "predict":
    mistral_predict.run()
