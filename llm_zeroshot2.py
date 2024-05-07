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
You are an expert in categorizing web text into linguistic registers.

Choose strictly from the following pre-defined linguistic registers:

LY: song lyric or poem
ID: online forum discussion
SP: any clearly spoken content
SP-it: spoken interview
NA: any narrative or report
NA-ne: news report
NA-sr: sports report
NA-nb: narrative blog
HI: how-to and instructions
HI-re: recipe
IN: any informational description
IN-en: encyclopedia article
IN-ra: research article
IN-dtp: description of a thing or person
IN-fi: FAQ
IN-lt: legal terms and conditions
OP: any opinionated text
OP-rv: review
OP-ob: opinion blog
OP-rs: religious text
OP-av: advice
IP: any persuasive writing
IP-ds: marketing 
IP-ed: persuasive facts
OTHER: rejected text (no label)

A text can be given multiple registers if it fits more than one. Otherwise, use only one register.

Below are more detailed instructions on how to label the text for register:

### LY: Lyrical ###
- Most often song lyrics or poems
- Typically, the texts are originally written by professional songwriters and poets, but they are posted online by fans and online contributors

### ID: Interactive discussion ###
- Interactive forum discussions with discussion participants and possibly other readers
- Question-answer forums, where one person asks a question and one or several answer it
- If the text only consists of comments that belong/are related to a blog post or an article, but the blog post or article is missing, then these comments should be annotated as Interactive discussion. However, comments following a visible blog post or article are not separately labelled as Interactive discussion but according to the main body of text.
- Text that consist of a single comment can also be annotated as Interactive discussion (when it’s clear that it is a comment)
- Originally written!

### SP-it: Interview (Spoken) ###
- Typically one interviewer and one interviewee
- Participants a radio show host / journalist and a famous person or an invited expert
- Most interviews are dialogic and have a question-answer format

### SP: Other Spoken ###
- Texts other than Interviews that are composed of more than 50% spoken material
- For example formal speeches, such as ones held by politicians
- TV/movie transcripts or Youtube video transcripts

### NA-ne: News report (Narrative) ###
- News reports written by journalists and published by news outlets
- Releases and newsletters published by sports associations, companies, etc.
- Weather forecasts
- Text purpose is to report on recent events
- Typically professionally written and time-sensitive - published and read as fast as possible

### NA-sr: Sports report (Narrative) ###
- Text purpose is to report on a recent sports event
- Typically written by professional journalists, but can also be published by amateur writers, for instance on sports club home pages
- Time-sensitive – published and read as fast as possible
- Note that not all texts on the topic of sport are automatically annotated as Sports report. If the purpose of the text is not to report on a sports event, other register classes should be considered. For example, an article about politics in sports could be annotated as News report (NA-nr).

### NA-nb: Blog (Narrative) ###
- Personal blogs, travel blogs, lifestyle blogs, blogs written by communities
- Purpose to narrate / comment about events experienced by the writer(s)
- Typically amateur writers
- Can include interactive aspects, such as comments following the blog post

### NA: Other Narrative ###
- Narrative texts that are not News reports (NA-nr), Sports reports (NA-sr) or Narrative blogs (NA-nb)
- Text purpose is to narrate or report on an event
- Focus on objective, factual, neutral content
- Texts such as short stories, fiction, magazine articles, other online articles

### HI-re: Recipe (How-to / instructional)
- Step-by-step instructions on how to prepare or cook something, typically food
- Should include at least the ingredients and/or the actual instructions

### HI: Other How-to / instructional
- How-to instructions that are not Recipes (HI-re)
- Objective instructions on how to perform a task, often step-by-step
- E.g., rules of a game, tutorials, instructions on how to fill a form
- Subjective instructions should be annotated as Advice (OP-av)
- Can be written on a personal, commercial or institutional website

### IN-en: Encyclopedia article (Informational description) ###
- Texts that describe or explain a topic
- Objective is to synthesize the current state of knowledge from all available studies
- Typically written by a collaborative group of co-authors
- A “wiki” platform; either Wikipedia or a similar one
- A dictionary entry
- Many of the articles are biographical describing a person’s life and accomplishments

### IN-ra: Research article (Informational description) ###
- Describes a research study, including the motivation for the study, the methods used, and the major research findings
- Written either by an individual or a collaborative group of co-authors, associated with an academic institution
- Target audience specialists
- Also dissertations and theses are included in this group

### IN-dtp: Description of a thing or person (Informational description ###
- Texts describing a thing or a person (excluding Encyclopedia articles (IN-en))
- A variety of documents ranging from administrative websites describing taxation to health care officials describing illnessess and associations describing their activities
- This category includes also job descriptions and notices of open tender### IN-fi: Frequently Asked Questions (Informational description) ###
- Documents sructured as questions-and-answers
- Text purpose to provide specific information about something
- Websites with procedural information often have special pages with FAQs, anticipating questions that end-users may have
- The author is usually associated with an institutional or commercial site

### IN-lt: Legal terms and conditions (Informational description ###
- Any document describing legislation
- Texts belonging to Legal terms and conditions are official by nature
- E.g., privacy policies, long cookie descriptions, texts describing terms and conditions, bills, rules of an association
- For rules of a game, see Other how-to (HI-oh)

### IN: Other Informational description ###
- Texts that describe or explain something but are not Encyclopedia articles (IN-en), Research articles (IN-ra), Descriptions of a thing or a person (IN-dtp), FAQs (IN-fi), or Legal terms or conditions (IN-lt).
- For instance, course materials, test papers, meeting minutes, and descriptive reports
- Also informational blogs informing the reader
- Presented as objective information rather than personal opinion

### OP-rv: Review (Opinion)
- Texts evaluating the quality of a product or a service
- Can be written on a personal, institutional, or commercial website

### OP-ob: Opinion blog (Opinion) ###
- Blogs written to express the writer’s / writers’ opinion
- Typically written by an amateur writer, such as a politician
- Typical topics include politics, governmental policies and social issues
- The author does not need to have any special expertise or credentials
- Focus on present time orientation
- Expressions of evaluation and stance, overt argumentation

### OP-rs: Denominational religious blog / sermon (Opinion) ###
- Denominational religious blog, sermon or basically any other denominational religious text
- Focus on denominational: texts describing a religion can be e.g., Description of a thing or a person (IN-dtp)

### OP-av: Advice (opinion) ###
- Based on a personal opinion
- Purpose to offer advice that leads to suggested actions and solves a particular problem
- Objective instructions should be annotated as Other how-to (HI-oh)
- Often associated with an institutional or commercial site
- Differs from Opinion blogs (OP-ob) and other Opinions in being directive and suggesting actions for the reader
- Typical topics include healthcare, finding a job, parenting, training for a sport

### OP-oo: Other Opinion ###
- Text expressing the writer’s or writers’ opinion that are not Reviews (OP-rv), Opinion blogs (OP-ob), Denominational religious blogs / sermons (OP-rs), or Advice (OP-av)
- For example an opinion piece
- Compared to News & opinion blog or editorial (IP-ed), Other opinion features less solid argumentation

### IP-ds: Description with intent to sell (Informational persuasion) ###
- Texts describing something with the purpose of selling
- Overtly marketing, but money does not need to be mentioned
- E.g., book blurbs (including library recommendations), product descriptions, marketing a service

### IP-ed: News & opinion blog or editorial (Informational persuasion) ###
- Purpose to persuade the reader by using information and facts
- Typically written by a professional on a news-related topic
- Can be associated with a newspaper or magazine

### IP-oe: Other Informational persuasion ###
- Texts with intent to persuade that are not Description with intent to sell (IP-ds) or News & opinion blog or editorial (IP-ed)
- Unlike Description with intent to sell (IP-ds), these texts are not overtly marketing a product, service or similar
- For instance, persuasive and argumentative essays, texts on public health promoting healthy lifestyles, texts that advertise an upcoming event, market an enterprise without overtly selling (e.g. description of an enterprise) or product placement, etc.

### Other: Rejected text (no label) ###
- the text consists of short list items (e.g. product listings on a web shop page, lists of links)
- the main body of text are short photo captions
- the sentences don’t form a coherent text
- there are only individual lines of actual text
- the amount of coherent text is very small compared to the ‘junk’ text (otherwise focus on annotating the actual text and ignore the junk text)
- there are no complete sentences, or fewer than two (e.g. lists of short news introductions)
- the text is not in the target language
- the text is poorly extracted (not representative of the web page)
- the document consists of special characters or numbers only

--- When to give a document several labels ---

A text should be given two registers when it features characteristics of more than one register.

E.g. when

- a personal blog includes a recipe: Narrative blog (NA-nb) + Recipe (HI-re)
- a text describing something also includes instructions: Description of a thing or person (IN-dtp) + Other how-to (HI)
- a marketing text is followed by reviews: Description with intent to sell (IP-ds) + Review (OP-rv)
- a sport report includes the writer’s viewpoints: Sports report (NA-sr) + Other opinion (OP)
- a blog post that informs the reader about an upcoming football match and invites them to watch: Narrative blog (NA-nb) + Other informational persuasion (IP)
- Eulogy with lyrics and decriptions: Lyrics (LY-ly) + Other narrative (NA)

Please note that Encyclopedia articles (IN-en), like Wikipedia texts, are not by default hybrids.

Strictly output only the register abbreviations as a space-separated (" ") list. Do not explain your decision.

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
