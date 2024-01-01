import os
from argparse import ArgumentParser
from v2 import gemini


parser = ArgumentParser()
parser.add_argument("--language", "-l", default="sv")
options = parser.parse_args()

gemini.run(options)
