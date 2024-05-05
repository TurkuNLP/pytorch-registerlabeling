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

LY: lyrical, such as songs or poems
ID: online forum discussion
SP: originally spoken content (more than 50% spoken)
NA: narratives and reports
HI: how-to and instructions
IN: informational description (wikis, articles, FAQs, etc)
OP: opinionated text
IP: persuasion, such as marketing or other persuasive writing
Other: none of the above

Optionally, also choose one of the subclasses for the MAIN class (in parentheses):

it: interview (SP)
ne: news report (NA)
sr: sports report (NA)
nb: narrative blog (NA)
re: recipe (HI)
en: encyclopedia article (IN)
ra: research article (IN)
dtp: description of a thing or person (IN)
fi: FAQ (IN)
lt: legal terms and conditions (IN)
rv: review (OP)
ob: opinion blog (OP)
rs: denominational religious text (OP)
av: advice (OP)
ds: marketing description (IP)
ed: persuasive facts (IP)

Very importantly, only choose a subclass if if you already chose its MAIN class (in parentheses). 

Only choose multiple classes if the text clearly belongs to multiple classes. Otherwise, choose a single class (and an optional subclass).

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
