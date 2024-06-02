from unsloth import FastLanguageModel
import torch

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from ..data import get_dataset
from ..labels import binarize_labels

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
"""

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def evaluate(dataset):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth_ft/lora_model",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    dataset = dataset["test"]

    predictions = []
    true_labels = [
        binarize_labels(x.split(), "upper") for x in list(dataset["label_text"])
    ]

    for example in dataset["text"]:

        inputs = tokenizer(
            [prompt_template.format(INSTRUCTION, example[:3000], "")],
            return_tensors="pt",
        ).to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    result = tokenizer.batch_decode(outputs)
    try:
        pred_label = result.split("<|end_of_text|>")[0].split("Response:\n")[1]
    except:
        pred_label = ""
    print(result)
    print(binarize_labels(result.split(), "upper"))
    exit()


def run(cfg):

    # Get dataset
    def formatting_prompts_func(examples):

        inputs = examples["text"]
        outputs = examples["label_text"]
        texts = []
        for input, output in zip(inputs, outputs):
            text = (
                prompt_template.format(INSTRUCTION, input[:3000], output)
                + tokenizer.eos_token
            )
            texts.append(text)
        return {"text": texts, "labels": examples["label_text"]}

    dataset = get_dataset(cfg)

    if cfg.just_evaluate:

        evaluate(dataset)
        exit()

    dataset = dataset["train"].map(
        formatting_prompts_func,
        batched=True,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  #
        random_state=cfg.seed,
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
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=cfg.seed,
            output_dir="unsloth_ft/outputs",
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

    model.save_pretrained("unsloth_ft/lora_model")  # Local saving
    tokenizer.save_pretrained("unsloth_ft/lora_model")
