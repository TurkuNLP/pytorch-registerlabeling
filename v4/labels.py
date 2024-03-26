labels_structure = {
    "MT": [],
    "LY": [],
    "SP": ["it"],
    "ID": [],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}


map_xgenre = {
    ### 1. MACHINE TRANSLATED
    "MT": "Other",
    ### 2. LYRICAL
    "LY": "Prose/Lyrical",
    ### 3. SPOKEN
    "SP": "Other",
    "sp": "Other",
    # Interview
    "it": "Other",
    ### 4. INTERACTIVE DISCUSSION
    "ID": "Forum",
    ### 5. NARRATIVE
    "NA": "Prose/Lyrical",  # Or Opinion/Argumentation
    # News report
    "ne": "News",
    # Sports report
    "sr": "News",
    # Narrative blog
    "nb": "Opinion/Argumentation",
    ### 6. HOW-TO or INSTRUCTIONS
    "HI": "Instruction",
    # Recipe
    "re": "Instruction",
    ### 7. INFORMATIONAL DESCRIPTION
    "IN": "Information/Explanation",
    # Encyclopedia article
    "en": "Information/Explanation",
    # Research article
    "ra": "Information/Explanation",
    # Description of a thing or person
    "dtp": "Information/Explanation",
    # Faq
    "fi": "Instruction",  # ???
    # Legal terms and conditions
    "lt": "Legal",
    ### 8. OPINION
    "OP": "Opinion/Argumentation",
    # Review
    "rv": "Opinion/Argumentation",
    # Opinion blog
    "ob": "Opinion/Argumentation",
    # Denominational religious blog / sermon
    "rs": "Prose/Lyrical",  # ???
    # Advice
    "av": "Opinion/Argumentation",  # ??? Or Instruction?
    ### 9. INFORMATIONAL PERSUASION
    "IP": "Promotion",
    # Description with intent to sell
    "ds": "Promotion",
    # News & opinion blog or editorial
    "ed": "Opinion/Argumentation",  # ???
}

labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]

labels_upper = [x for x in labels_all if x.isupper()]

label_schemes = {
    "all": labels_all,
    "upper": labels_upper,
}

map_normalize = {
    # Our categories, upper
    "MT": "MT",
    "HI": "HI",
    "ID": "ID",
    "IN": "IN",
    "IP": "IP",
    "LY": "LY",
    "NA": "NA",
    "OP": "OP",
    "SP": "SP",
    # Our categories, lower
    "av": "av",
    "ds": "ds",
    "dtp": "dtp",
    "ed": "ed",
    "en": "en",
    "fi": "fi",
    "it": "it",
    "lt": "lt",
    "nb": "nb",
    "ne": "ne",
    "ob": "ob",
    "ra": "ra",
    "re": "re",
    "rs": "rs",
    "rv": "rv",
    "sr": "sr",
    # Converted categories
    "NE": "ne",
    "SR": "sr",
    "PB": "nb",
    "HA": "NA",
    "FC": "NA",
    "TB": "nb",
    "CB": "nb",
    "OA": "NA",
    "OB": "ob",
    "RV": "rv",
    "RS": "rs",
    "AV": "av",
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
    "DF": "ID",
    "QA": "ID",
    "RE": "re",
    "DS": "ds",
    "EB": "ed",
    "ED": "ed",
    "PO": "LY",
    "SO": "LY",
    "IT": "it",
    "FS": "SP",
    "TV": "SP",
    "OS": "SP",
    "IG": "IP",
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
    "OTHER": "",
    "NB": "nb",
    "na": "NA",
    "sp": "SP",
    "": "",
}

map_lower_upper = {
    "it": "SP",
    "os": "SP",
    "ne": "NA",
    "sr": "NA",
    "nb": "NA",
    "on": "NA",
    "re": "HI",
    "oh": "HI",
    "en": "IN",
    "ra": "IN",
    "dtp": "IN",
    "fi": "IN",
    "lt": "IN",
    "oi": "IN",
    "rv": "OP",
    "ob": "OP",
    "rs": "OP",
    "av": "OP",
    "oo": "OP",
    "ds": "IP",
    "ed": "IP",
    "oe": "IP",
}


def normalize_labels(labels, label_scheme_name):
    if type(labels) == str:
        labels = (labels or "").split()

    # Normalize
    labels = [map_normalize[label] for label in labels]

    # Make sure that sublabels have corresponding upper labels
    for label in labels:
        if label in map_lower_upper and map_lower_upper[label] not in labels:
            labels.append(map_lower_upper[label])

    # Upper labels
    if label_scheme_name == "upper":
        labels = [x for x in labels if x.isupper()]

    # XGENRE labels
    if label_scheme_name == "xgenre":

        print(f"orig: {labels}")

        # First, remove upper category if lower present
        mapped_simple = []
        for label in labels:
            if not label in labels_structure:
                if any(x in labels for x in labels_structure[label]):
                    mapped_simple.append(label)

        # Then, map
        labels = [map_xgenre[label] for label in mapped_simple]

        print(f"xgenre: {labels}")

    return sorted(list(set(filter(None, labels))))


def binarize_labels(labels, label_scheme_name):
    label_scheme = label_schemes[label_scheme_name]

    return [1 if scheme_label in labels else 0 for scheme_label in label_scheme]


def decode_binary_labels(data, label_scheme_name):
    label_scheme = label_schemes[label_scheme_name]
    return [
        " ".join([label_scheme[i] for i, bin_val in enumerate(bin) if bin_val == 1])
        for bin in data
    ]
