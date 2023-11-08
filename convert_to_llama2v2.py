import glob, os
import csv, sys

mode = "INST"

labels = [
    "HI",
    "ID",
    "IN",
    "IP",
    "LY",
    "MT",
    "NA",
    "OP",
    "SP",
    "av",
    "ds",
    "dtp",
    "ed",
    "en",
    "fi",
    "it",
    "lt",
    "nb",
    "ne",
    "ob",
    "ra",
    "re",
    "rs",
    "rv",
    "sr",
]

sub_register_map = {
    "NA": "NA",
    "NE": "ne",
    "SR": "sr",
    "PB": "nb",
    "HA": "NA",
    "FC": "NA",
    "TB": "nb",
    "CB": "nb",
    "OA": "NA",
    "OP": "OP",
    "OB": "ob",
    "RV": "rv",
    "RS": "rs",
    "AV": "av",
    "IN": "IN",
    "JD": "IN",
    "FA": "fi",
    "DT": "dtp",
    "IB": "IN",
    "DP": "dtp",
    "RA": "ra",
    "LT": "lt",
    "CM": "IN",
    "EN": "en",
    "RP": "IN",
    "ID": "ID",
    "DF": "ID",
    "QA": "ID",
    "HI": "HI",
    "RE": "re",
    "IP": "IP",
    "DS": "ds",
    "EB": "ed",
    "ED": "ed",
    "LY": "LY",
    "PO": "LY",
    "SO": "LY",
    "SP": "SP",
    "IT": "it",
    "FS": "SP",
    "TV": "SP",
    "OS": "OS",
    "IG": "IP",
    "MT": "MT",
    "HT": "HI",
    "FI": "fi",
    "OI": "IN",
    "TR": "IN",
    "AD": "OP",
    "LE": "OP",
    "OO": "OP",
    "MA": "NA",
    "ON": "NA",
    "SS": "NA",
    "OE": "IP",
    "PA": "IP",
    "OF": "ID",
    "RR": "ID",
    "FH": "HI",
    "OH": "HI",
    "TS": "HI",
    "OL": "LY",
    "PR": "LY",
    "SL": "LY",
    "TA": "SP",
    "OTHER": "OS",
    "": "",
}

inst = "You are a text classification tool. Your task is to categorize the following text (delimited with ```) into one or more of the 25 categories: av, ds, dtp, ed, en, fi, HI, ID, IN, IP, it, lt, LY, MT, NA, nb, ne, ob, OP, ra, re, rs, rv, SP, sr. Your response should include just the category name(s), separated by a single space. Here is the input text:"


# Data preprocessing
def preprocess_data(example):
    text = example[1] or ""
    mapped_labels = " ".join(
        sorted(
            set(
                [
                    sub_register_map[l] if l not in labels else l
                    for l in (example[0] or "NA").split()
                ]
            )
        )
    )
    return mapped_labels, text


path = "data/en"
csv.field_size_limit(sys.maxsize)

with open(path + "/train_instruction_inst.tsv", "w", newline="") as tsvfile:
    writer = csv.writer(tsvfile, delimiter="\t", lineterminator="\n")
    writer.writerow(["text"])

    for filename in glob.glob(os.path.join(path, "*.tsv")):
        with open(os.path.join(os.getcwd(), filename), "r") as f_in:
            text = csv.reader(f_in, delimiter="\t")
            for line in text:
                print(line)
                mapped_labels, raw_text = preprocess_data(line)
                out = f"<s>[INST] {inst} ```{raw_text}``` [/INST] {mapped_labels} </s>"
                writer.writerow([out])
