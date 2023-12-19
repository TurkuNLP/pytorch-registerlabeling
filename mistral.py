import os
from argparse import ArgumentParser
from v2 import mistral_train, mistral_predict

os.environ["TRANSFORMERS_CACHE"] = ".hf/transformers_cache"
os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"

parser = ArgumentParser()
parser.add_argument("--mode", "-m", default="train")
parser.add_argument("--base_model", "-b", default="mistralai/Mistral-7B-Instruct-v0.2")
parser.add_argument("--new_model", "-n", default="mistralai-finetuned")
parser.add_argument("--peft_model_path", "-p", default="")
options = parser.parse_args()

if options.mode == "train":
    mistral_train.run(options.base_model, options.new_model)

elif options.mode == "predict":
    mistral_predict.run(options.peft_model_path)
