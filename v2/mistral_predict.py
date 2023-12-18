from .data import get_dataset
from .mistral_prompt import prompt

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, PeftModel


def run(peft_model_path):
    dataset = get_dataset("en-fi-fr-sv", "en-fi-fr-sv", "all", few_shot=10)

    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Mistral, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        cache_dir="cache",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)

    ft_model.eval()

    example = dataset["test"][0]

    model_input = tokenizer(prompt(example), return_tensors="pt")
    labels_true = example["label_text"]
    lang = example["language"]
    with torch.no_grad():
        result = tokenizer.decode(
            ft_model.generate(**model_input, max_new_tokens=100, pad_token_id=2)[0],
            skip_special_tokens=True,
        )
        try:
            labels_pred = result.split("### Labels")[1].strip()
            print(f"True: {labels_true}")
            print(f"Pred: {labels_pred}")
            print(f"Language: {lang}")
        except:
            print("Error: could not parse labels. Here is the full result:")
            print(result)
