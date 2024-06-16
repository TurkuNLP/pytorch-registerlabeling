import numpy as np
import os

os.environ["HF_HOME"] = ".hf/hf_home"
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
import logging
import warnings
from trl import SFTTrainer

from ..data import get_dataset
from ..labels import binarize_labels

from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

from dotenv import load_dotenv

load_dotenv()

login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN", ""))


torch_dtype = torch.float16
attn_implementation = "eager"

model_id = "CohereForAI/aya-23-8B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

INSTRUCTION = """Your task is to classify web texts into one or more linguistic register categories. The categories are as follows:

MT: The text is machine translated or generated from a template.
LY: The text is lyrical (e.g. songs or poems).
SP: The text is originally spoken (e.g. interview).
ID: The text is an interactive discussion written by multiple participants in a discussion format (e.g. discussion or Q&A forum).
NA: The text narrates or reports on events (e.g. news report, sports, report, narrative blog).
HI: The text explains how-to or gives instructions (e.g. recipe or other step step-by-step, objective instructions on how to do something).
IN: The text describes or explains information (e.g. encyclopedia article, research article, description of a thing or person, FAQ, Legal terms and conditions, course materials and blogs for informing the reader)
OP: The text expresses opinions (e.g. review, opinion blog typically written by an amateur writer, such as a politician, to express their opinion, denominational religious blog or sermon, advice).
IP: The text describes or explains facts with intent to persuade or market (e.g. description with intent to sell a product or service, a news & opinion blog or editorial typically written by a professional writer on a news-related topic with well-structured argumentation).
    
Your output must strictly be just a space-separated (" ") list of the category abbreviations that match the text (MT, LY, SP, ID, NA, HI, IN, OP, IP). Prefer a single label or at most two. Do not explain your choice in any way.

Here is the text you will classify (enclosed within ``` and ```):
"""


def format_instruct(text):
    return f"{INSTRUCTION}\n```\n{text[:3000]}\n```"


def evaluate(model, tokenizer, dataset):

    dataset = dataset["test"]

    sample = 1000

    predictions = []
    true_labels = [
        binarize_labels(x.split(), "upper")
        for x in list(dataset["label_text"][:sample])
    ]
    for example in tqdm(dataset["text"][:sample]):

        row_json = [
            {"role": "user", "content": format_instruct(example)},
        ]
        example = tokenizer.apply_chat_template(row_json, tokenize=False)

        inputs = tokenizer(
            example,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.01,
        )

        output = tokenizer.decode(outputs[0])

        try:
            result = output.split("<|CHATBOT_TOKEN|>")[-1].split(
                "<|END_OF_TURN_TOKEN|>"
            )[0]
        except:
            result = ""

        print(result)

        predictions.append(binarize_labels(result.split(), "upper"))

    """
    for example in tqdm(dataset["text"][:sample]):

        messages = [
            {"role": "user", "content": format_instruct(example[:3000])},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            batch_size=1,
            model_kwargs={"quantization_config": bnb_config},
        )

        outputs = pipe(
            prompt,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.01,
        )
        pred_label = outputs[0]["generated_text"]
        predictions.append(binarize_labels(pred_label.split(), "upper"))
    """
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average="micro"
    )
    accuracy = accuracy_score(true_labels, predictions)

    metrics = {
        "f1": f1,
        "f1_macro": f1_score(
            true_labels, predictions, average="macro", zero_division=np.nan
        ),
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }

    print(metrics)
    exit()


def run(cfg):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation,
    )

    def format_chat_template(row):
        row_json = [
            {"role": "user", "content": format_instruct(row["text"])},
            {"role": "assistant", "content": row["label_text"]},
        ]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = get_dataset(cfg)

    if cfg.just_evaluate:
        model = PeftModel.from_pretrained(model, f"llm/{model_id}")
        evaluate(model, tokenizer, dataset)
        exit()

    dataset = dataset["train"].map(
        format_chat_template,
        num_proc=4,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "up_proj",
                "down_proj",
                "gate_proj",
                "k_proj",
                "q_proj",
                "v_proj",
                "o_proj",
            ],
        ),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            output_dir=f"llm/{model_id}",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            optim="paged_adamw_32bit",
            num_train_epochs=1,
            logging_steps=1,
            warmup_ratio=0.05,
            learning_rate=2e-4,
            fp16=False,
            bf16=False,
            group_by_length=True,
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # @title Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained(f"llm/{model_id}")  # Local saving
    tokenizer.save_pretrained(f"llm/{model_id}")
