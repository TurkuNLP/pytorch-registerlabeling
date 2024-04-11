from transformers import (
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

import csv
import os
import sys
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)


class Augment:
    def __init__(self, cfg):
        self.cfg = cfg

        # Run
        getattr(self, cfg.method)()

    def translate(self, text, tokenizer, model):
        # Your input text
        text = " ".join(text.split(" ")[:300])

        # Tokenize input
        input_ids = tokenizer.encode(
            text,
            return_tensors="pt",
        )

        input_ids = input_ids.to("cuda")

        # Generate translation
        outputs = model.generate(input_ids)

        # Decode and print the translation
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text

    def back_translate(self):

        with open(f"data/{self.cfg.source}/train.tsv", "r") as f:
            # Parse the tsv file into a list of lists
            rows = [line.strip().split("\t") for line in f]
            rows = [row for row in rows if len(row) > 1 and row[0] and row[1]]

            for row in tqdm(rows):

                src_lang = self.cfg.source
                tgt_lang = self.cfg.target
                model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
                back_model_name = f"Helsinki-NLP/opus-mt-{tgt_lang}-{src_lang}"

                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

                back_tokenizer = AutoTokenizer.from_pretrained(back_model_name)
                back_model = AutoModelForSeq2SeqLM.from_pretrained(back_model_name).to(
                    "cuda"
                )

                translation = self.translate(row[1], tokenizer, model)
                back_translation = self.translate(
                    translation, back_tokenizer, back_model
                )

                with open(
                    f"data/{self.cfg.source}/train_aug.tsv", "a", encoding="utf-8"
                ) as outfile:
                    outfile.write(f"{row[0]}\t{back_translation}\n")