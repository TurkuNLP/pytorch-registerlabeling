INSTRUCTION = """
Categorize the following web-scraped text into one or more of the following categories:

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
IP: The purpose of the document is to describe or explain facts with intent to persuade or market (e.g. description with intent to sell a product or service, a news & opinion blog or editorial typically written by a professional writer on a news-related topic with well-structured argumentation, or other descriptive texts that e.g. sell or promote a service, product or upcoming event, such as a hotel, a smartphone or a football game).

Just output the appropriate label(s) as a space-separated list. if you are unsure, leave blank.

"""

INPUT_TEXT = "The quick brown fox did a news report."

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_4bit=True, device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": f"### Instruction:\n{INSTRUCTION}\n### Input text:\n{INPUT_TEXT}",
    }
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
