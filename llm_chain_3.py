import os

import torch

os.environ["HF_HOME"] = ".hf/hf_home"

from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "nvidia/Llama3-ChatQA-1.5-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

SYSTEM = "System: This AI assistant works as a multilingual text classifier. The assistant will be given a text in any language scraped from the web, and it will categorize the text into a predefined register. A 'register' is a text variety defined by its situational and communicative characteristics.\n\nThe classification proceeds in three steps: 1) determining if the text was originally spoken or written; 2) assigning a register label to the text; 3) In some cases, assigning a sub-register label to the text.\n\n Here is the text to be classified (enclosed within ``` and ```):"
FIRST_PROMPT = """First, please determine if the text is likely to be originally SPOKEN or WRITTEN. Explain your decision very briefly in one or two sentences.

Next, based on the decision in the first step, please assign the text a register label. If the text was categorized as SPOKEN, determine if the text can be classified into exactly one of the following categories:

"LY": poetic text, such as song or poem
"SP": spoken text with the majority of the text spoken (e.g. interview, speech, tv/movie transcript, etc.)

Alternatively, If the text was categorized as WRITTEN, determine if the text can be classified into exactly one of the following categories:

"ID": interactive discussion written by multiple participants in a discussion format (e.g. forum, comment section, etc.)
"NA": narratives and reports of events (e.g. news report, sports report, narrative blog, fictional story, magazine article, etc.)
"HI": how-to explanations and instructions (e.g. recipe, how-to article, manual, etc.)
"IN": informational description (e.g. wiki, article, FAQ, description of things or persons, legal terms, etc.)
"OP": text expressing opinions (e.g. review, opinion blog, religious text, advice, etc.)
"IP": persuasion, such as marketing or other persuasive writing (e.g. description with intent to sell, news & opinion blog, other marketing texts)

Choose "OTHER" if the text does not clearly belong to any single of the above registers, or is not clearly either spoken or written, or combines characteristics of multiple registers. Explain your decision very briefly in one or two sentences.

Finally, and very importantly, output the chosen register label in a separate line, with JUST the register label in that last line and ABSOLUTELY NOTHING ELSE. The last line should contain exactly one of the following strings: "LY", "SP", "ID", "NA", "HI", "IN", "OP", "IP", "OTHER".
"""

PROMPT_SUFFIX = """ Explain your decision very briefly in one or two sentences.

Finally, and very importantly, output the chosen register label in a separate line, with JUST the register label in that last line and ABSOLUTELY NOTHING ELSE. The last line should contain exactly one of the following strings:"""

PROMPT_SP = f"""
Next, please determine if the text you classified as SP is an interview. An interview typically has one interviewer and one interviewee, such as a radio show host / journalist and a famous person or an invited expert. Most interviews are dialogic and have a question-answer format.

If the text is an interview, choose "it". If it is not an interview, choose "OTHER". {PROMPT_SUFFIX} "it", "OTHER".
"""

PROMPT_NA = """
Next, please determine the subregister of the text you classified as NA. Here are the subregisters for NA, followed by bullet points describing each subregister:

ne: News report

- News reports written by journalists and published by news outlets
- Releases and newsletters published by sports associations, companies, etc.
- Weather forecasts
- Text purpose is to report on recent events
- Typically professionally written and time-sensitive - published and read as fast as possible

sr: Sports report

- Text purpose is to report on a recent sports event
- Typically written by professional journalists, but can also be published by amateur writers, for instance on sports club home pages
- Time-sensitive – published and read as fast as possible
- Note that not all texts on the topic of sport are automatically annotated as Sports report. If the purpose of the text is not to report on a sports event, other register classes should be considered. For example, an article about politics in sports could be annotated as News report (ne)

nb: Narrative blog

- Personal blogs, travel blogs, lifestyle blogs, blogs written by communities
- Purpose to narrate / comment about events experienced by the writer(s)
- Typically amateur writers
- Can include interactive aspects, such as comments following the blog post

If the text is a news report, choose "ne". If the text is a sports report, choose "sr". If the text is a narrative blog, choose "nb". If the text does not belong to any single subregister, choose "OTHER". {PROMPT_SUFFIX} "ne", "sr", "nb", "OTHER".
"""

PROMPT_HI = """
Next, please determine if the text you classified as HI is a recipe. A recipe contains step-by-step instructions on how to prepare or cook something, typically food. It include at least the ingredients and/or the actual instructions.

If the text is a recipe, choose "re". If it is not an interview, choose "OTHER". {PROMPT_SUFFIX} "re", "OTHER".
"""

PROMPT_IN = """
Next, please determine the subregister of the text you classified as IN. Here are the subregisters for IN, followed by bullet points describing each subregister:

en: Encyclopedia article

- Texts that describe or explain a topic
- Objective is to synthesize the current state of knowledge from all available studies
- Typically written by a collaborative group of co-authors
- A “wiki” platform; either Wikipedia or a similar one
- A dictionary entry
- Many of the articles are biographical describing a person’s life and accomplishments

ra: Research article 

- Describes a research study, including the motivation for the study, the methods used, and the major research findings
- Written either by an individual or a collaborative group of co-authors, associated with an academic institution
- Target audience specialists
- Also dissertations and theses are included in this group

dtp: Description of a thing or person

- Texts describing a thing or a person (excluding Encyclopedia articles (IN-en))
- A variety of documents ranging from administrative websites describing taxation to health care officials describing illnessess and associations describing their activities
- This category includes also job descriptions and notices of open tender

fi: Frequently Asked Questions 

- Documents sructured as questions-and-answers
- Text purpose to provide specific information about something
- Websites with procedural information often have special pages with FAQs, anticipating questions that end-users may have
- The author is usually associated with an institutional or commercial site

lt: Legal terms and conditions

- Any document describing legislation
- Texts belonging to Legal terms and conditions are official by nature
- E.g., privacy policies, long cookie descriptions, texts describing terms and conditions, bills, rules of an association

If the text is an encyclopedia article, choose "en". If the text is a research article, choose "ra". If the text is a description of a thing or person, choose "dtp". If the text is a frequently asked questions document, choose "fi". If the text is a legal terms and conditions document, choose "lt". If the text does not belong to any single subregister, choose "OTHER". {PROMPT_SUFFIX} "en", "ra", "dtp", "fi", "lt", "OTHER".
"""

PROMPT_OP = """
Next, please determine the subregister of the text you classified as OP. Here are the subregisters for OP, followed by bullet points describing each subregister:

rv: Review 

- Texts evaluating the quality of a product or a service
- Can be written on a personal, institutional, or commercial website

ob: Opinion blog

- Blogs written to express the writer’s / writers’ opinion
- Typically written by an amateur writer, such as a politician
- Typical topics include politics, governmental policies and social issues
- The author does not need to have any special expertise or credentials
- Focus on present time orientation
- Expressions of evaluation and stance, overt argumentation

rs: Denominational religious blog / sermon

- Denominational religious blog, sermon or basically any other denominational religious text
- Focus on denominational: texts describing a religion can be e.g., Description of a thing or a person (IN-dtp)

av: Advice

- Based on a personal opinion
- Purpose to offer advice that leads to suggested actions and solves a particular problem
- Objective instructions should be annotated as Other how-to (HI-oh)
- Often associated with an institutional or commercial site
- Differs from Opinion blogs (OP-ob) and other Opinions in being directive and suggesting actions for the reader
- Typical topics include healthcare, finding a job, parenting, training for a sport

If the text is a review, choose "rv". If the text is an opinion blog, choose "ob". If the text is a denominational religious blog or sermon, choose "rs". If the text is an advice document, choose "av". If the text does not belong to any single subregister, choose "OTHER". {PROMPT_SUFFIX} "rv", "ob", "rs", "av", "OTHER".
"""

PROMPT_IP = """
Next, please determine the subregister of the text you classified as IP. Here are the subregisters for IP, followed by bullet points describing each subregister:

ds: Description with intent to sell

- Texts describing something with the purpose of selling
- Overtly marketing, but money does not need to be mentioned
- E.g., book blurbs (including library recommendations), product descriptions, marketing a service

ed:  News & opinion blog or editorial

- Purpose to persuade the reader by using information and facts
- Typically written by a professional on a news-related topic
- Can be associated with a newspaper or magazine

If the text is a description with intent to sell, choose "ds". If the text is a news & opinion blog or editorial, choose "ed". If the text does not belong to any single subregister, choose "OTHER". {PROMPT_SUFFIX} "ds", "ed", "OTHER".
"""

document = """Review If you need to change a fan belt on a 1994 Fiat, you buy a Chilton's manual, and not a treatise on the joys of high-speed touring. If you need to make a lemon meringue pie, you get a cookbook, and not a memoir on the joys of great French cuisine. Car manuals and recipes are not always great literature by any means, but they are often necessary in helping to get a job done. Susan Rose Blauner's HOW I STAYED ALIVE WHEN MY BRAIN WAS TRYING TO KILL ME is nobody's idea of great, or even good, literature. From a purely literary standpoint, the book is chatty, tiresome and irritating, filled with sentimental rubbish, New Age nonsense, and ghastly psychological claptrap. It has been edited with an over-gentle hand, preserving every little clich and every annoying scrap of poetry and personal reflection. It is a book that very few people will pick up for pleasurable reading, and rightly so. And yet, it will undoubtedly save lives. HOW I STAYED LIVE WHEN MY BRAIN WAS TRYING TO KILL ME is not, as you might think, merely a personal tale of survival from mental illness. It is primarily a manual, a reference book, a resource for people who have suicidal thoughts. Although the book is guided by the author's own experiences with mental illness and suicide attempts, it is written not to chronicle her life but to provide direction and guidance for others in the same situation. And as such, it is an undeniable success. Blauner's book is guided by several hard-won insights. Suicide begins as a thought, driven by negative feelings, and such feelings are temporary and changeable. "Suicidal," Blauner tells us, "is not a feeling." Suicidal thoughts are paired with feelings of anger, guilt, loneliness, and desperation, and it is necessary to separate those feelings from thoughts of suicide. Suicidal thoughts can be addictive, we learn, with romantic notions of one's death and funeral building upon each other. And these suicidal thoughts from one's brain war with one's spirit, which doesn't want to die, creating the conflict in the title. The heart of the book is the "Tips of the Trade," 25 different ideas, strategies, and plans that people with suicidal thoughts can use to help avoid harming themselves. The most invaluable of these is the "Crisis Plan," which is easily the best thing about the book. Blauner details the plan that she, along with her therapist, worked out to help her deal with suicidal thoughts. It begins with "Take a deep breath," and proceeds from there to prayer, activities, exercise, and phone calls to family, friends, and professionals. Applying the principles of strategic planning and crisis management to one's personal life may seem a little unorthodox, but it is undoubtedly effective, and may prove to be so for people with a variety of different needs. The "Tricks" are extremely varied, and more than a little eclectic. (This is to be expected from an author who describes herself as a "Jewish Unitarian Zen-Quakerish earth-loving type.") Not all of the "Tricks" will help everyone, and more than a few of them may seem a little goofy, if not out-and-out weird. Realistically, though, you never can tell what might help someone set aside a suicidal thought. If throwing eggs at trees, or sitting in a chair with a bucket between your knees helps someone, then it's a trick worth sharing, no matter how odd it sounds. HOW I STAYED ALIVE WHEN MY BRAIN WAS TRYING TO KILL ME is not an incredibly well-written book, but it is brave and courageous and helpful, full of resources and tips and ideas and strength for anyone experiencing suicidal thoughts or anyone with a friend or family member with such experiences. More than that, it is a book that is, quite simply, "normal," if not invaluable, in helping people in this situation finish the job of life. """
document = document[:3000]

main_registers = ["OTHER", "MT", "LY", "ID", "SP", "NA", "HI", "IN", "OP", "IP"]
subregisters = [
    "it",
    "ne",
    "sr",
    "nb",
    "re",
    "en",
    "ra",
    "dtp",
    "fi",
    "lt",
    "rv",
    "ob",
    "rs",
    "av",
    "ds",
    "ed",
    "OTHER",
]


def get_formatted_input(messages, context):
    conversation = (
        "\n\n".join(
            [
                (
                    "User: " + item["content"]
                    if item["role"] == "user"
                    else "Assistant: " + item["content"]
                )
                for item in messages
            ]
        )
        + "\n\nAssistant:"
    )
    formatted_input = SYSTEM + "\n\n```\n" + context + "\n```\n\n" + conversation

    return formatted_input


def converse(messages, document):

    formatted_input = get_formatted_input(messages, document)

    tokenized_prompt = tokenizer(
        tokenizer.bos_token + formatted_input, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=tokenized_prompt.input_ids,
        attention_mask=tokenized_prompt.attention_mask,
        max_new_tokens=512,
        eos_token_id=terminators,
    )

    response = outputs[0][tokenized_prompt.input_ids.shape[-1] :]
    response_str = tokenizer.decode(response, skip_special_tokens=True)
    return response_str


# Stage 1: main register
registers = []
messages = [
    {
        "role": "user",
        "content": FIRST_PROMPT,
    }
]

first_response = converse(messages, document)

messages.append({"role": "assistant", "content": first_response})

main_register = first_response.strip().split("\n")[-1].strip().replace('"', "")
registers.append(main_register)
sub_register = ""
if main_register not in main_registers:
    main_register = "OTHER"
if main_register not in ["OTHER", "MT", "LY", "ID"]:
    subregister_prompt = ""
    if main_register == "SP":
        subregister_prompt = PROMPT_SP
    elif main_register == "NA":
        subregister_prompt = PROMPT_NA
    elif main_register == "HI":
        subregister_prompt = PROMPT_HI
    elif main_register == "IN":
        subregister_prompt = PROMPT_IN
    elif main_register == "OP":
        subregister_prompt = PROMPT_OP
    elif main_register == "IP":
        subregister_prompt = PROMPT_IP

    messages.append({"role": "user", "content": subregister_prompt})

    second_response = converse(messages, document)

    sub_register = second_response.strip().split("\n")[-1].strip()
    registers.append(sub_register)

result = " ".join(registers)
print(result)
