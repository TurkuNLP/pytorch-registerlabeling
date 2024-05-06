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

PROMPT = """
You are an expert in categorizing web-scraped text into predefined linguistic registers (Biber and Egbert 2018). 

The register taxonomy is hierarchical, with the following MAIN registers. First, choose the most appropriate MAIN register for the text:

LYRICAL: song lyrics or poem
FORUM: online forum discussion
SPOKEN: originally spoken content (more than half spoken)
NARRATIVE: any narrative or report
HOWTO: how-to and instructions
INFORMATIONAL: informational description (wikis, articles, FAQs, etc.)
OPINION: opinionated text
PERSUASION: persuasion, such as marketing or other persuasive writing
OTHER: none of the above

Optionally, also choose one of the sub-registers for the MAIN class (given in parentheses):

interview: interview (SPOKEN)
news: news report (NARRATIVE)
sports: sports report (NARRATIVE)
blog: narrative blog (NARRATIVE)
recipe: recipe (HOWTO)
encyclopedia: encyclopedia article (INFORMATIONAL)
research: research article (INFORMATIONAL)
description: description of a thing or person (INFORMATIONAL)
faq: FAQ (INFORMATIONAL)
legal: legal terms and conditions (INFORMATIONAL)
review: review (OPINION)
opinion-blog: opinion blog (OPINION)
religious: denominational religious text (OPINION)
advice: advice (OPINION)
marketing: marketing description (PERSUASION)
persuasive-facts: persuasive facts (PERSUASION)

Very importantly, only choose a sub-register if if you already chose its corresponding MAIN register!

A text should be given multiple registers if and only if it features characteristics of more than one register.

Strictly output only the register names as a space-separated (" ") list. Do not explain your decision.

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
