from transformers import MarianMTModel, MarianTokenizer

import csv
import sys

csv.field_size_limit(sys.maxsize)


class Augment:
    def __init__(self, cfg):
        self.cfg = cfg

        # Run
        getattr(self, cfg.method)()

    def get_model_and_tokenizer(self, model_name):
        target_tokenizer = MarianTokenizer.from_pretrained(model_name)
        target_model = MarianMTModel.from_pretrained(model_name)

        return target_tokenizer, target_model

    def back_translate(self):
        source_tokenizer, source_model = self.get_model_and_tokenizer(
            f"Helsinki-NLP/opus-mt-{self.cfg.source}-{self.cfg.target}"
        )
        target_tokenizer, target_model = self.get_model_and_tokenizer(
            f"Helsinki-NLP/opus-mt-{self.cfg.target}-{self.cfg.source}"
        )

        with open(
            f"data/{self.cfg.source}/train_aug.tsv", "w", encoding="utf-8"
        ) as outfile:

            with open(f"data/{self.cfg.source}/train.tsv", "r") as f:
                # Parse the tsv file into a list of lists
                rows = [line.strip().split("\t") for line in f]

                for row in rows:

                    if not len(row) > 1 and row[0] and row[1]:
                        continue

                    src_text = " ".join(row[1].split()[:300])

                    translated = source_tokenizer.decode(
                        source_model.generate(
                            **source_tokenizer(
                                src_text,
                                return_tensors="pt",
                                padding=True,
                            )
                        )[0],
                        skip_special_tokens=True,
                    )
                    print(translated)
                    back_translated = target_tokenizer.decode(
                        target_model.generate(
                            **target_tokenizer(
                                translated,
                                return_tensors="pt",
                                padding=True,
                            )
                        )[0],
                        skip_special_tokens=True,
                    )

                    outfile.write(f"{row[0]}\t{back_translated}\n")
