import deepl

import csv
import os
import sys

csv.field_size_limit(sys.maxsize)

LANG_MAP = {
    "en": "EN-GB",
    "fi": "FI",
    "fr": "FR",
    "sv": "SV",
    "tr": "TR",
}


class Augment:
    def __init__(self, cfg):
        self.cfg = cfg

        # Run
        getattr(self, cfg.method)()

    def back_translate(self):

        with open(f"data/{self.cfg.source}/train.tsv", "r") as f:
            # Parse the tsv file into a list of lists
            rows = [line.strip().split("\t") for line in f]
            rows = [row for row in rows if len(row) > 1 and row[0] and row[1]]

            for row in rows:
                print(row[1])
                auth_key = self.cfg.deepl_auth_key
                translator = deepl.Translator(auth_key)

                translation = translator.translate_text(
                    row[1], target_lang=LANG_MAP[self.cfg.target]
                )
                print(translation)
                back_translation = translator.translate_text(
                    translation, target_lang=LANG_MAP[self.cfg.source]
                )

                print(back_translation)
                """
                with open(
                    f"data/{self.cfg.source}/train_aug.tsv", "a", encoding="utf-8"
                ) as outfile:
                    outfile.write(f"{row[0]}\t{back_translated}\n")
                """
                exit()

            exit()

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
