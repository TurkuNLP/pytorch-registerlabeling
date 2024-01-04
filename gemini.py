import os
from argparse import ArgumentParser
from v2 import gemini, gemini_torch


parser = ArgumentParser()
parser.add_argument("--language", "-l", default="sv")
parser.add_argument("--create_embeddings", "-c", action="store_true")
options = parser.parse_args()

if options.create_embeddings:
    gemini.run(options)

else:
    gemini_torch.run(options)
