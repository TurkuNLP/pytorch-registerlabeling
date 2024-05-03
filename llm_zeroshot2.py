import os

import torch

os.environ["HF_HOME"] = ".hf/hf_home"

import logging
import warnings

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN", ""))

PROMPT = """You will be given a text randomly scraped from the web. Your task is to classify it into one or more of MAIN classes, and optionally, into subclasses.Here are the MAIN categories:

MT: machine translated or generated from a template
LY: lyrical, such as songs or poems
ID: interactive discussion written by multiple participants in a discussion format
SP: any clearly spoken content
NA: narratives and reports
HI: how-to and instructions
IN: informational description (wikis, articles, FAQs, etc)
OP: opinionated text
IP: persuasion, such as marketing or other persuasive writing
Other: none of the above

After classifying the text into one of the above, optionally choose from the following subclasses for each MAIN class:

it: interview (subclass of SP)
ne: news report (subclass of NA)
sr: sports report (subclass of NA)
nb: narrative blog (subclass of NA)
re: recipe (subclass of HI)
en: encyclopedia article (subclass of IN)
ra: research article (subclass of IN)
dtp: description of a thing or person (subclass of IN)
fi: FAQ (subclass of IN)
lt: legal terms and conditions (subclass of IN)
rv: review (subclass of OP)
ob: opinion blog (subclass of OP)
rs: denominational religious text (subclass of OP)
av: advice (subclass of OP)
ds: marketing description (subclass of IP)
ed: persuasive facts (subclass of IP)

Importantly, only choose a subclass for a MAIN class you have already chosen. A subclass can only be given if its MAIN class is also given.

Strictly output only the class abbreviations as a space-separated (" ") list. Do not explain your decision.

Here is the text (enclosed within ``` and ```)

```
"""

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


def generate_label(text):

    text = text[:3000]

    messages = [
        {"role": "user", "content": PROMPT + text + "\n```"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=32,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
    )
    response = outputs[0][input_ids.shape[-1] :]
    result = tokenizer.decode(response, skip_special_tokens=True)
    print(f'"{text[:100]}..." -> "{result}"')
    return result


# Define the path to the directories containing the files
base_path = "data/en/"
file_names = ["dev.tsv"]


# Process each file
for file_name in file_names:
    # Construct the file path
    file_path = base_path + file_name

    # Read the TSV file into a DataFrame
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["true_labels", "text"],
        na_values="",  # Don't interpret NA as NaN!
        keep_default_na=False,
    )

    # Strip whitespace from strings in the DataFrame
    df["true_labels"] = df["true_labels"].str.strip()
    df["text"] = df["text"].str.strip()

    df.dropna(inplace=True)

    # Filter out rows where either 'true_labels' or 'text' are empty
    df = df[(df["true_labels"] != "") & (df["text"] != "")]

    # Apply the generate_label function to the 'text' column and create a new column with progress display
    df["new_labels"] = df["text"].progress_apply(generate_label)

    # Reorder the columns to the specified order
    df = df[["true_labels", "new_labels", "text"]]

    # Construct the output file path
    output_file_path = base_path + file_name.replace(
        ".tsv", "_full_gen_new_prompt.tsv"
    )

    # Save the new DataFrame to a TSV file
    df.to_csv(output_file_path, sep="\t", index=False, header=False)
