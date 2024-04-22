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

SYSTEM_PROMPT = """
You are a classifier for web-scraped texts written in different languages. Given a text, you must label it as one or more of the following categories:

MT, LY, SP, ID, NA, HI, IN, OP, IP

Give the label(s) based on the following instructions:

MT: The text is machine translated or generated from a template.
LY: The text is lyrical (e.g. songs or poems).
SP: The text is originally spoken (e.g. interview).
ID: The text is an interactive discussion written by multiple participants in a discussion format (e.g. discussion or Q&A forum).
NA: The text narrates or reports on events (e.g. news report, sports, report, narrative blog).
HI: The text explains how-to or gives instructions (e.g. recipe or other step step-by-step, objective instructions on how to do something).
IN: The text describes or explains information (e.g. encyclopedia article, research article, description of a thing or person, FAQ, Legal terms and conditions, course materials and blogs for informing the reader)
OP: The text expresses opinions (e.g. review, opinion blog typically written by an amateur writer, such as a politician, to express their opinion, denominational religious blog or sermon, advice).
IP: The text describes or explains facts with intent to persuade or market (e.g. description with intent to sell a product or service, a news & opinion blog or editorial typically written by a professional writer on a news-related topic with well-structured argumentation).
    
Just output the label(s). If there are many labels, separate them with a single space (" "). ONLY output the label(s), nothing else. If you are unsure, output "None". Prefer single labels over multiple labels.
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
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {"role": "user", "content": text},
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
    print(f'"{text[:10]}..." -> "{result}"')
    return result


# Define the path to the directories containing the files
base_path = "data/en/"
file_names = ["dev.tsv", "train.tsv", "test.tsv"]


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
    output_file_path = base_path + file_name.replace(".tsv", "_gen.tsv")

    # Save the new DataFrame to a TSV file
    df.to_csv(output_file_path, sep="\t", index=False, header=False)
