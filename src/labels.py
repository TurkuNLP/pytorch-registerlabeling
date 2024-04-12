import numpy as np

# The full label hierarchy
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

other_labels = {
    "SP": "os",
    "NA": "on",
    "HI": "oh",
    "IN": "oi",
    "OP": "oo",
    "IP": "oe",
}

# Mapping to XGENRE labels
map_xgenre = {
    ### 1. MACHINE TRANSLATED
    "MT": "Other",
    ### 2. LYRICAL
    "LY": "Prose/Lyrical",
    ### 3. SPOKEN
    "SP": "Other",
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

map_full_names = {
    "MT": "Machine translated (MT)",
    "LY": "Lyrical (LY)",
    "SP": "Spoken (SP)",
    "it": "Interview (it)",
    "os": "Other SP",
    "ID": "Interactive discussion (ID)",
    "NA": "Narrative (NA)",
    "ne": "News report (ne)",
    "sr": "Sports report (sr)",
    "nb": "Narrative blog (nb)",
    "on": "Other NA",
    "HI": "How-to or instructions (HI)",
    "re": "Recipe (re)",
    "oh": "Other HI",
    "IN": "Informational description (IN)",
    "en": "Encyclopedia article (en)",
    "ra": "Research article (ra)",
    "dtp": "Description: thing / person (dtp)",
    "fi": "FAQ (fi)",
    "lt": "Legal (lt)",
    "oi": "Other IN",
    "OP": "Opinion (OP)",
    "rv": "Review (rv)",
    "ob": "Opinion blog (ob)",
    "rs": "Religious blog / sermon (rs)",
    "av": "Advice (av)",
    "oo": "Other OP",
    "IP": "Informational persuasion (IP)",
    "ds": "Description: intent to sell (ds)",
    "ed": "News & opinion blog / editorial (ed)",
    "oe": "Other IP",
}

# Flat list of labels
labels_all = [k for k in labels_structure.keys()] + [
    item for row in labels_structure.values() for item in row
]


# Mapping from subcategory ID to parent ID
subcategory_to_parent_index = {
    labels_all.index(subcategory): labels_all.index(parent)
    for parent, subcategories in labels_structure.items()
    for subcategory in subcategories
}


# XGENRE labels
labels_xgenre = list(sorted(set(map_xgenre.values())))

# Mapping from original binary vector ID to XGENRE ID
category_to_xgenre_index = {
    i: labels_xgenre.index(map_xgenre[label]) for i, label in enumerate(labels_all)
}

# Upper labels
labels_upper = [x for x in labels_all if x.isupper()]

# Upper label indexes in the full taxonomy
upper_all_indexes = [
    labels_all.index(item) for item in labels_upper if item in labels_all
]

label_schemes = {
    "all": labels_all,
    "upper": labels_upper,
    "xgenre": labels_xgenre,
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
        # First, remove upper category if lower present
        mapped_simple = []
        for label in labels:
            if not (
                label in labels_structure
                and any(x in labels for x in labels_structure[label])
            ):
                mapped_simple.append(label)

        # Then, map
        labels = [map_xgenre[label] for label in mapped_simple if label]

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


def map_childless_upper_to_other(doc_labels):
    updated_labels = [
        (
            other_labels.get(label, label)
            if label in labels_structure.keys()
            and not set(labels_structure[label]).intersection(doc_labels)
            else label
        )
        for label in doc_labels
    ]

    labels_with_children = [
        label for label in labels_structure.keys() if labels_structure[label]
    ]

    return [label for label in updated_labels if label not in labels_with_children]


def map_to_xgenre_binary(true_labels, predictions):

    def convert(label_vector, what):
        print(f"Converting {what}\n======")
        print(label_vector)
        print(decode_binary_labels([label_vector], "all"))
        # Initialize XGENRE vector with zeros
        xgenre_vector = [0] * len(labels_xgenre)

        # Determine the effective category probabilities, prioritizing subcategories
        effective_probs = [0] * len(labels_all)
        for parent, subcategories in labels_structure.items():
            parent_index = labels_all.index(parent)
            if subcategories:
                sub_indices = [labels_all.index(sub) for sub in subcategories]
                max_sub_prob = max(label_vector[i] for i in sub_indices)
                # Compare max subcategory probability to parent's probability
                if max_sub_prob >= label_vector[parent_index]:
                    effective_probs[parent_index] = 0
                else:
                    effective_probs[parent_index] = label_vector[parent_index]
                for i in sub_indices:
                    effective_probs[i] = label_vector[i]
            else:
                effective_probs[parent_index] = label_vector[parent_index]

        # Map effective category probabilities to XGENRE
        for i, prob in enumerate(effective_probs):
            xgenre_index = category_to_xgenre_index[i]
            xgenre_vector[xgenre_index] = max(xgenre_vector[xgenre_index], prob)
        print(xgenre_vector)
        print(decode_binary_labels([xgenre_vector], "xgenre"))
        return xgenre_vector

    # Convert labels and predictions
    true_labels_converted = [
        convert(label_vector, "true") for label_vector in true_labels
    ]
    predictions_converted = [
        convert(label_vector, "pred") for label_vector in predictions
    ]

    return np.array(true_labels_converted), np.array(predictions_converted)
