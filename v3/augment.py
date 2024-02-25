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

        def iterate_in_batches(lst, batch_size):
            for i in range(0, len(lst), batch_size):
                yield lst[i : i + batch_size]

        # with open(
        #    f"data/{self.cfg.source}/train_aug.tsv", "w", encoding="utf-8"
        # ) as outfile:

        with open(f"data/{self.cfg.source}/train.tsv", "r") as f:
            # Parse the tsv file into a list of lists
            rows = [line.strip().split("\t") for line in f]
            rows = [row for row in rows if len(row) > 1 and row[0] and row[1]]

            for batch in iterate_in_batches(rows, 8):

                batch = [[x[0], " ".join(x[1].split()[:300])] for x in batch]

                translated_tokens = (
                    source_model.generate(
                        **source_tokenizer(
                            batch,
                            return_tensors="pt",
                            padding=True,
                        )
                    ),
                )

                translated = [
                    source_tokenizer.decode(
                        t,
                        skip_special_tokens=True,
                    )
                    for t in translated_tokens
                ]
                print(translated)
                exit()
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
