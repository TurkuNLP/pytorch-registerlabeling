import deepl

import csv
import os
import sys
from tqdm import tqdm

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

            for row in tqdm(rows):

                auth_key = self.cfg.deepl_auth_key
                translator = deepl.Translator(auth_key)

                translation = translator.translate_text(
                    row[1], target_lang=LANG_MAP[self.cfg.target]
                )

                back_translation = translator.translate_text(
                    str(translation), target_lang=LANG_MAP[self.cfg.source]
                )

                with open(
                    f"data/{self.cfg.source}/train_aug.tsv", "a", encoding="utf-8"
                ) as outfile:
                    outfile.write(f"{row[0]}\t{back_translation}\n")
