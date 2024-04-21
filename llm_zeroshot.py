import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import logging
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN", ""))

SYSTEM_PROMPT = """
You are a classifier for web-scraped texts written in different languages. Given a text, you must label it as one or more of the following categories:

MT, LY, SP, ID, NA, HI, IN, OP, IP

Give the label(s) based on the following instructions:

MT: The web page is machine translated or generated from a template.
LY: The web page is lyrical, such as songs or poems.
SP: The web page is originally spoken (e.g. interview).
ID: The web page is an interactive discussion written by multiple participants in a discussion format (e.g. discussion or Q&A forum).
NA: The purpose of the document is to narrate or report on events (e.g. news report, sports, report, narrative blog).
HI: The purpose of the document is to explain how-to or instructions (e.g. recipe or typically other step step-by-step, objective instructions on how to do something).
IN: The purpose of the document is to describe or explain information (e.g. encyclopedia article, research article, description of a thing or person, FAQ, Legal terms and conditions, course materials and blogs for informing the reader)
OP: The purpose of the document to express opinions (review, opinion blog typically written by an amateur writer, such as a politician, to express their opinion, denominational religious blog / sermon, advice).
IP: The purpose of the document is to describe or explain facts with intent to persuade or market (e.g. description with intent to sell a product or service, a news & opinion blog or editorial typically written by a professional writer on a news-related topic with well-structured argumentation).
    
Just give the label(s) as a space-separated list. ONLY output the label(s), nothing else. If you are unsure, output "None".
"""

test_input = """
The quick brown fox did a news report.
"""

model = "meta-llama/Meta-LLAMA-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, load_in_4bit=True)
model.eval()


def generate_response(text):
    """Generate a response given a user input."""
    torch.cuda.empty_cache()

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

    input_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=20,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0,
    )
    response = outputs[0][input_ids.shape[-1] :]
    output = tokenizer.decode(response, skip_special_tokens=True)

    return output


label = generate_response(test_input)
print(label)
