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
You are a classifier for texts scraped from the unrestricted web in different languages. Given a text, you must label it as one or more of the following categories:

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


PREFIX_IMPROVED = """
Classify the following text into one or more of the following registers:

["Machine-translated", "Lyrical", "Spoken:Interview", "Spoken:Other", "Forum", "Narrative:News", "Narrative:Sports", "Narrative:Blog", "Narrative:Other", "How-to:Recipe", "How-to:Other", "Informational:Encyclopedia", "Informational:Research", "Informational:Description", "Informational:FAQ", "Informational:Legal", "Informational:Other", "Opinion:Review", "Opinion:Blog", "Opinion:Religious", "Opinion:Advice", "Opinion:Other", "Persuasion:Marketing", "Persuasion:Informational", "Persuasion:Other", "Other"]

Here is the text:
```
"""

FULL_PROMPT_IMPROVED = """
You will be given texts randomly scraped from the web. Your task is to classify it into a primary register and, if necessary, a secondary register.

Choose strictly from this list: ["Machine-translated", "Lyrical", "Spoken:Interview", "Spoken:Other", "Forum", "Narrative:News", "Narrative:Sports", "Narrative:Blog", "Narrative:Other", "How-to:Recipe", "How-to:Other", "Informational:Encyclopedia", "Informational:Research", "Informational:Description", "Informational:FAQ", "Informational:Legal", "Informational:Other", "Opinion:Review", "Opinion:Blog", "Opinion:Religious", "Opinion:Advice", "Opinion:Other", "Persuasion:Marketing", "Persuasion:Informational", "Persuasion:Other", "Other"]

You can choose multiple labels if the text contains characteristics from multiple registers. However, choose only one subcategory from the hierarchical categories (Spoken, Narrative, How-to, Informational, Opinion, Persuasion)

Choose the label(s) based on answering these questions:

- Is the web page machine translated or generated from a template? --> "Machine-Translated"
- Is the web page lyrical, such as songs or poems? --> "Lyrical"
- Is the web page an interactive discussion written by multiple participants in a discussion format? --> "Forum"
- Is the web page a spoken interview --> "Spoken:Interview"
- Is the web page some other spoken content --> "Spoken:Other"
- Is the web page a news report, newsletter, weather forecast or similar --> "Narrative:News"
- Is the web page a sports report --> "Narrative:Sports"
- Is the web page a personal blog --> "Narrative:Blog"
- Is the web page some other narrative or report --> "Narrative:Other"
- Is the web page a recipe --> "How-to:Recipe"
- Is the web page some other how-to or instructions --> "How-to:Other"
- Is the web page a description of some topic like a wiki or dictionary entry --> "Informational:Encyclopedia"
- Is the web page a description of a research study --> "Informational:Research"
- Is the web page a description of a thing or a person --> "Informational:Description"
- Is the web page a frequently-asked-questions page --> "Informational:FAQ"
- Is the web page legislational --> "Informational:Legal"
- Is the web page some other description (e.g. course materials, tests, meeting notes) --> "Informational:Other"
- Is the web page an opinionated review --> "Opinion:Review"
- Is the web page an opinionated blog (e.g. politics, society) --> "Opinion:Blog"
- Is the web page denominationally religious --> "Opinion:Religious"
- Is the web page offering personal advice --> "Opinion:Advice"
- Is the web page some other opinion piece --> "Opinion:Other"
- Is the web page intending to sell a product or service --> "Persuasion:Marketing"
- Is the web page intending to persuade with information --> "Persuasion:Informational"
- Is the web page some other informational persuasion (e.g. upcoming event, lifestyle) --> "Persuasion:Other"
- Is the web page mostly list entries or captions, lacking substantial content --> "Other"

Just output the register class(es) from the list above, nothing else. If there are many labels, separate them with a single space (" "). Do not explain your decision in any way. 
"""

FULL_PROMPT = """
You are a web register classifier for texts in different languages scraped from the unrestricted web. Given a text, you must give it one or more of the following labels:

MT-mt, LY-ly, SP-it, SP-os, ID-id, NA-ne, NA-sr, NA-nb, NA-on, HI-re, HI-oh, IN-en, IN-ra, IN-dtp, IN-fi, IN-lt, IN-oi, OP-rv, OP-ob, OP-rs, OP-av, OP-oo, IP-ds, IP-ed, IP-oe, Other

--- Annotation guidelines ---

### MT-mt: Machine translated or generated ###

- Texts that are clearly machine translated or generated from a template
- Such texts can be e.g. from holiday accommodation sites or flight booking sites
- It is not necessary to give another register label to Machine translated or generated text

### LY-ly: Lyrical ###

- Most often song lyrics or poems
- Typically, the texts are originally written by professional songwriters and poets, but they are posted online by fans and online contributors

### SP-it: Interview (Spoken) ###

- Typically one interviewer and one interviewee
- Participants a radio show host / journalist and a famous person or an invited expert
- Most interviews are dialogic and have a question-answer format

### SP-os: Other Spoken ###

- Texts other than Interviews that are composed of more than 50% spoken material
- For example formal speeches, such as ones held by politicians
- TV/movie transcripts or Youtube video transcripts

### ID-id: Interactive discussion ###

- Interactive forum discussions with discussion participants and possibly other readers
- Question-answer forums, where one person asks a question and one or several answer it
- If the text only consists of comments that belong/are related to a blog post or an article, but the blog post or article is missing, then these comments should be annotated as Interactive discussion. However, comments following a visible blog post or article are not separately labelled as Interactive discussion but according to the main body of text.
- Text that consist of a single comment can also be annotated as Interactive discussion (when it’s clear that it is a comment)
- Originally written!

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

### NA-on: Other Narrative ###

- Narrative texts that are not News reports (NA-nr), Sports reports (NA-sr) or Narrative blogs (NA-nb)
- Text purpose is to narrate or report on an event
- Focus on objective, factual, neutral content
- Texts such as short stories, fiction, magazine articles, other online articles

### HI-re: Recipe (How-to / instructional)

- Step-by-step instructions on how to prepare or cook something, typically food
- Should include at least the ingredients and/or the actual instructions

### HI-oh: Other How-to / instructional

- How-to instructions that are not Recipes (HI-re)
- Objective instructions on how to perform a task, often step-by-step
- E.g., rules of a game, tutorials, instructions on how to fill a form
- Subjective instructions should be annotated as Advice (OP-av)
Can be written on a personal, commercial or institutional website

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

### IN-oi: Other Informational description ###
- Texts that describe or explain something but are not Encyclopedia articles (IN-en), Research articles (IN-ra), Descriptions of a thing or a person (IN-dtp), FAQs (IN-fi), or Legal terms or conditions (IN-lt).
- For instance, course materials, test papers, meeting minutes, and descriptive reports
- Also informational blogs informing the reader
- Presented as objective information rather than personal opinion

### OP-rv: Review (Opinion) ###- Texts evaluating the quality of a product or a service
- Can be written on a personal, institutional, or commercial website

### OP-ob: Opinion blog (Opinion) ###

- Blogs written to express the writer’s / writers’ opinion
- Typically written by an amateur writer, such as a politician
- Typical topics include politics, governmental policies and social issues
- The author does not need to have any special expertise or credentials
- Focus on present time orientation
- Expressions of evaluation and stance, overt argumentation### OP-rs: Denominational religious blog / sermon (Opinion) ###

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
- a text describing something also includes instructions: Description of a thing or person (IN-dtp) + Other how-to (HI-oh)
- a marketing text is followed by reviews: Description with intent to sell (IP-ds) + Review (OP-rv)
- a sport report includes the writer’s viewpoints: Sports report (NA-sr) + Other opinion (OP-oo)
- a blog post that informs the reader about an upcoming football match and invites them to watch: Narrative blog (NA-nb) + Other informational persuasion (IP-oe)
- Eulogy with lyrics and decriptions: Lyrics (LY-ly) + Other narrative (NA-on)

Please note that Encyclopedia articles (IN-en), like Wikipedia texts, are not by default hybrids.

--- Output instruction ---

VERY IMPORTANT: Only choose A SINGLE label WITH THE SAME PREFIX!!. For example, if you choose IN-en, that must be the ONLY label starting with IN-. The same goes for all other prefixes. If you choose a label with any of the prefixes (MT-, LY-, SP-, ID-, NA-, HI-, IN-, OP-, IP-), you must not choose any other label with the same prefix!

Just output the label(s), nothing else. Only include the abbreviation, not the full name. If there are many labels, separate them with a single space (" "). Do not explain your decision.
"""


FULL_PROMPT_SIMPLIFIED = """
You are a web register classifier for texts in different languages scraped from the unrestricted web. Given a text, you must label it as one or more of the following:

MT-mt, LY-ly, SP-it, SP-os, ID-id, NA-ne, NA-sr, NA-nb, NA-on, HI-re, HI-oh, IN-en, IN-ra, IN-dtp, IN-fi, IN-lt, IN-oi, OP-rv, OP-ob, OP-rs, OP-av, OP-oo, IP-ds, IP-ed, IP-oe, Other

--- Annotation guidelines ---

MT-mt: Texts that are clearly machine translated or generated from a template
LY-ly: Song lyrics or poems
SP-it: Spoken interviews structured in a dialogic question-answer format
SP-os: Other Spoken texts with more than 50% spoken material
ID-id: Interactive forum discussions with participants and possibly other readers
NA-ne: News reports, releases, newsletters, and weather forecasts
NA-sr: Sports reports written by professional journalists or amateur writers
NA-nb: Personal, travel, lifestyle, or community blogs
NA-on: Other Narrative texts such as short stories, fiction, magazine articles
HI-re: Step-by-step instructions on how to prepare or cook something
HI-oh: Other How-to instructions on how to perform a task
IN-en: Encyclopedia articles or wikis written to objectively describe various topics
IN-ra: Research articles describing a study, methods, and findings
IN-dtp: Descriptions of a thing or person, excluding Encyclopedia articles
IN-fi: FAQs structured as questions-and-answers
IN-lt: Legal terms and conditions, privacy policies, and bills
IN-oi: Other Informational descriptions such as course materials or reports
OP-rv: Reviews evaluating the quality of a product or service
OP-ob: Opinion blogs expressing the writer's opinion on politics, policies, or social issues
OP-rs: Denominational religious blogs or sermons
OP-av: Advice offering solutions to a problem based on personal opinion
OP-oo: Other Opinion pieces expressing the writer's opinion
IP-ds: Descriptions with intent to sell a product or service
IP-ed: News & opinion blogs or editorials persuading the reader with information and facts
IP-oe: Other Informational persuasion texts promoting healthy lifestyles or advertising an event
Other: Rejected text that doesn't fit any of the above categories

--- Instructions ---

A text should be given multiple registers when it features characteristics of more than one register.

Always prefer to use a single label over multiple labels.

Just output the label(s), nothing else. Only include the abbreviation, not the full name. If there are many labels, separate them with a single space (" "). Do not explain your decision.
"""

PREFIX = """
Classify the following text as one or more of the following categories:

MT-mt, LY-ly, SP-it, SP-os, ID-id, NA-ne, NA-sr, NA-nb, NA-on, HI-re, HI-oh, IN-en, IN-ra, IN-dtp, IN-fi, IN-lt, IN-oi, OP-rv, OP-ob, OP-rs, OP-av, OP-oo, IP-ds, IP-ed, IP-oe, Other

VERY IMPORTANT: Make sure to include only A SINGLE label with the same prefix (MT-, LY-, SP-, NA-, HI-, IN-, OP-, IP-). If you are unsure, output "Other". Only output the label(s) abbreviations, nothing else. If there are many labels, separate them with a single space (" ").

Please do the labeling as carefully as possible, this is important scientific work, and you will be rewarded.

Here is the text:
```
"""

PREFIX_SIMPLIFIED = """
Classify the following text as one or more of the following categories:

MT-mt, LY-ly, SP-it, SP-os, ID-id, NA-ne, NA-sr, NA-nb, NA-on, HI-re, HI-oh, IN-en, IN-ra, IN-dtp, IN-fi, IN-lt, IN-oi, OP-rv, OP-ob, OP-rs, OP-av, OP-oo, IP-ds, IP-ed, IP-oe, Other

Only use multiple labels if necessary, otherwise prefer a single label.

Please do the labeling as carefully as possible, this is important scientific work, and you will be rewarded.

Here is the text:
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
        {
            "role": "system",
            "content": FULL_PROMPT_IMPROVED,
        },
        {"role": "user", "content": PREFIX_IMPROVED+ text+'\n```'},
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
    output_file_path = base_path + file_name.replace(".tsv", "_full_gen_simplified_prompt.tsv")

    # Save the new DataFrame to a TSV file
    df.to_csv(output_file_path, sep="\t", index=False, header=False)
