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
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, add_bos_token=True, trust_remote_code=True
    )

    ft_model = PeftModel.from_pretrained(base_model, peft_model_path)

    ft_model.eval()

    correct = 0
    incorrect = 0

    with torch.no_grad():
        for i, example in enumerate(dataset["test"]):
            model_input = tokenizer(
                prompt(example, labels=False), return_tensors="pt"
            ).to("cuda")
            labels_true = example["label_text"]
            lang = example["language"]
            result = tokenizer.decode(
                ft_model.generate(**model_input, max_new_tokens=100)[0],
                skip_special_tokens=True,
            )
            try:
                labels_pred = (
                    result.split("### Labels")[1].strip().split("#")[0].strip()
                )
                print(f"True: {labels_true}")
                print(f"Pred: {labels_pred}")
                print(f"Language: {lang}")
                if labels_true == labels_pred:
                    correct += 1
                else:
                    print("Incorrect! Here it is: ")
                    print(result)
                    incorrect += 1

                print()
                print(
                    f"[{i}] Correct: {correct}, Incorrect: {incorrect}, Ratio: {round(correct/((correct+incorrect) or 1), 2)}"
                )
            except:
                print(f"[{i}] Error: could not parse labels. Here is the full result:")
                print(result)
