import torch
import os

os.environ["HF_HOME"] = ".hf/hf_home"

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

test_input = "Recently Thomson Reuters celebrated the 20 th anniversary of the publication of the third (looseleaf) edition of Adams on Criminal Law with a luncheon gathering of authors along with consulting editor, Sir Bruce Robertson. 20 years of the third edition of Adams on Criminal Law "On behalf of Thomson Reuters I'd like to welcome everyone to what is an important milestone in the life of Adams on Criminal Law . We are celebrating a 20th anniversary -- the 20th anniversary of publication of the third edition (the looseleaf edition) of Adams. It published in March 1992. But of course Adams is in fact almost 50 years old, with a history dating back to 1964 when the first edition of Criminal Law and Practice in New Zealand (with Sir Francis Adams as the consulting editor) was published by Sweet & Maxwell in response to the Crimes Act 1961. (This was of course, well before Sweet & Maxwell became a sister company of Brookers with the acquisition of Brookers by the Thomson Corporation in 1994). The work sold then for the princely sum of seven pounds and seven shillings. I know this because there is a review of the work that appears in what was just the second issue of the NZ Universities Law Review in 1964. The reviewer was Peter Mahon who from what I can establish was the Crown Solicitor in Christchurch at the time and had been junior counsel for the Crown in the Parker-Hulme murder case in 1954. It's a fascinating review: In it he talks about the greater ease of appeal in criminal cases and he goes on to say that: "The ever expanding body of appellate decisions, replete with refinements and distinctions, accentuates in ratio to its growth the difficulties and responsibilities of the trial judge, and the result has been to convert the ordinary criminal trial from a simple factual inquiry into an elaborate process overlaid with refinements which in many cases are as complex as they are unreal". And he concludes "it is therefore highly desirable, in view of the present state of the criminal law, to have a carefully annotated reference book, and all those involved in this field will welcome the publication of Criminal Law and Practice in New Zealand . Finally he sums up by saying "It is difficult to combine in one volume the necessary attributes of a practice work and a legal textbook, but the authors have unquestionably succeeded in this task, and there can be no doubt that their work represents as a whole, a substantial and valuable contribution to our legal literature." Unquestionably this still holds true today with Adams over the years having clearly established itself as the leading and definitive criminal law work in New Zealand, widely used and relied upon by practitioners and the judiciary (and one we are proud to be associated with). A second edition of Criminal Law and Practice in New Zealand was published in 1971 by Sweet & Maxwell, but Sir Francis died two years later. Stuart Brooker subsequently persuaded Sir Francis Adams widow to give him the rights to publish the text. The business then started an annotations service for the second edition but with the need to update the 2 nd edition becoming critical, Gerald Orchard and Neville Trendle were commissioned to publish a Supplement which covered the years 1982 to 1989, and was published in 1990. In Local to Global , the book published to mark Brookers centenary in 2010 Geoff Adlam relates that the catalyst for the third edition was rumours of the planned publication of a new competing looseleaf edition of Garrow and Turkington and the success of the company's very active looseleaf publishing programme in the 1980s. And it was then that Stuart Brooker contacted Sir Bruce and persuaded him to lead the new edition as consulting editor -- and as I mentioned it published in March 1992. And a new chapter in the history of Adams was opened in late 2005 when a framework for a new 'Super Adams' was discussed in a meeting with Sir Bruce, Warren Young & Neville Trendle. The new blueprint was in anticipation of future legislative developments including the Evidence Bill, the then proposed Criminal Procedure Bill, and sentencing and parole amendments. The blueprint included separate Adams volumes on Evidence, Sentencing, Procedure, Offences and Defences, and later Rights & Powers. And we now of course see the fruit of this planning and all the considerable effort put in by Adams authors -- with new Adams volumes on Evidence, Sentencing, Rights & Powers, and Procedure. Adams is a work of considerable stature, a jewel in Thomson Reuter's crown. I  want to acknowledge the contribution that all of you have made to the quality and reputation of the work. We are lucky to have such a strong and committed author team. I'd also like to acknowledge those authors who couldn't be here today: Warren Brookbanks, Don Mathias,  Neil Cameron, Grant Illingworth and Judge Hikaka. In particular I want to acknowledge Sir Bruce's considerable contribution. The 20 odd years he has been consulting editor is by far the longest for any Thomson Reuters publication. His has been the steady guiding hand that has ensured that the work built on the legacy of the earlier editions and remains the foremost work on criminal law and practice in New Zealand. So thank you Sir Bruce and all the authors assembled here today." Adams on Criminal Law is widely regarded as one of those indispensable foundation texts that no lawyer, member of the judiciary, academic, or student working in the criminal field should be without. Its coverage is broad and uncompromising: the whole field of New Zealand's criminal law from beginning to end prepared by a team of distinguished Judges and lawyers.  From inception the vision was to provide a practical, workable reference on criminal law. "

# model = "meta-llama/Meta-LLAMA-3-8B-Instruct"
model = "lucyknada/microsoft_WizardLM-2-7B"

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
        # tokenizer.convert_tokens_to_ids("<|eot_id|>"),
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
