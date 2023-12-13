labels_xgenre = [
    "Information/Explanation",
    "Instruction",
    "Legal",
    "Forum",
    "News",
    "Opinion/Argumentation",
    "Promotion",
    "Prose/Lyrical",
    "Other",
]

labels_all = [
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

labels_all_structure = {
    "SP": ["it"],
    "NA": ["ne", "sr", "nb"],
    "HI": ["re"],
    "IN": ["en", "ra", "dtp", "fi", "lt"],
    "OP": ["rv", "ob", "rs", "av"],
    "IP": ["ds", "ed"],
}

label_to_index = {label: idx for idx, label in enumerate(labels_all)}
label_hierarchy = {}

for parent, children in labels_all_structure.items():
    parent_index = label_to_index[parent]
    for child in children:
        child_index = label_to_index[child]
        label_hierarchy[child_index] = parent_index


labels_upper = [x for x in labels_all if x.isupper()]

labels_all_other = [
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
    "os",
    "on",
    "oh",
    "oi",
    "oo",
    "oe",
]

labels_all_flat = [
    "mt",
    "ly",
    "it",
    "os",
    "id",
    "ne",
    "sr",
    "nb",
    "on",
    "re",
    "oh",
    "en",
    "ra",
    "dtp",
    "fi",
    "lt",
    "oi",
    "rv",
    "ob",
    "rs",
    "av",
    "oo",
    "ds",
    "ed",
    "oe",
]

labels_all_hierarchy = {
    "MT": ["MT"],
    "LY": ["LY"],
    "SP": ["it", "os"],
    "ID": ["ID"],
    "NA": ["ne", "sr", "nb", "on"],
    "HI": ["re", "oh"],
    "IN": ["en", "ra", "dtp", "fi", "lt", "oi"],
    "OP": ["rv", "ob", "rs", "av", "oo"],
    "IP": ["ds", "ed", "oe"],
}

map_full_names = {
    "MT": "Machine translated",
    "LY": "Lyrical",
    "SP": "Spoken",
    "it": "Interview",
    "os": "Other spoken",
    "ID": "Interactive discussion",
    "NA": "Narrative",
    "ne": "News report",
    "sr": "Sports report",
    "nb": "Narrative blog",
    "on": "Other narrative",
    "HI": "How-to or instructions",
    "re": "Recipe",
    "oh": "Other how-to",
    "IN": "Informational description",
    "en": "Encyclopedia article",
    "ra": "Research article",
    "dtp": "Description of a thing or person",
    "fi": "FAQ",
    "lt": "Legal",
    "oi": "Other informational description",
    "OP": "Opinion",
    "rv": "Review",
    "ob": "Opinion blog",
    "rs": "Denominational religious blog / sermon",
    "av": "Advice",
    "oo": "Other opinion",
    "IP": "Informational persuasion",
    "ds": "Description with intent to sell",
    "ed": "News & opinion blog or editorial",
    "oe": "Other informational persuasion",
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

map_optional = {
    "av": "OP",
    "ed": "IP",
    "fi": "IN",
}

map_upper = {
    "HI": "HI",
    "ID": "ID",
    "IN": "IN",
    "IP": "IP",
    "LY": "LY",
    "MT": "MT",
    "NA": "NA",
    "OP": "OP",
    "SP": "SP",
    "av": "OP",
    "ds": "IP",
    "dtp": "IN",
    "ed": "IP",
    "en": "IN",
    "fi": "IN",
    "it": "SP",
    "lt": "IN",
    "nb": "NA",
    "ne": "NA",
    "ob": "OP",
    "ra": "IN",
    "re": "HI",
    "rs": "OP",
    "rv": "OP",
    "sr": "NA",
    "": "",
}

map_xgenre = {
    ### 2. MACHINE TRANSLATED
    "MT": "Other",
    "mt": "Other",
    ### 2. LYRICAL
    "LY": "Prose/Lyrical",
    "ly": "Prose/Lyrical",
    ### 3. SPOKEN
    "SP": "Other",
    "sp": "Other",
    "os": "Other",
    # Interview
    "it": "Other",
    ### 4. INTERACTIVE DISCUSSION
    "ID": "Forum",
    "id": "Forum",
    ### 5. NARRATIVE
    "NA": "Prose/Lyrical",  # Or Opinion/Argumentation
    "na": "Prose/Lyrical",  # Or Opinion/Argumentation
    "on": "Prose/Lyrical",  # Or Opinion/Argumentation
    # News report
    "ne": "News",
    # Sports report
    "sr": "News",
    # Narrative blog
    "nb": "Opinion/Argumentation",
    ### 6. HOW-TO or INSTRUCTIONS
    "HI": "Instruction",
    "hi": "Instruction",
    "oh": "Instruction",
    # Recipe
    "re": "Instruction",
    ### 7. INFORMATIONAL DESCRIPTION
    "IN": "Information/Explanation",
    "id": "Information/Explanation",
    "oi": "Information/Explanation",
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
    "op": "Opinion/Argumentation",
    "oo": "Opinion/Argumentation",
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
    "ip": "Promotion",
    "oe": "Promotion",
    # Description with intent to sell
    "ds": "Promotion",
    # News & opinion blog or editorial
    "ed": "Opinion/Argumentation",  # ???
    "": "Other",
}

map_upper_lower = {
    "SP": ["it", "os"],
    "NA": ["ne", "sr", "nb", "on"],
    "HI": ["re", "oh"],
    "IN": ["en", "ra", "dtp", "fi", "lt", "oi"],
    "OP": ["rv", "ob", "rs", "av", "oo"],
    "IP": ["ds", "ed", "oe"],
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

map_flat = {
    "MT": "mt",
    "LY": "ly",
    "SP": "os",
    "it": "it",
    "os": "os",
    "ID": "id",
    "NA": "on",
    "ne": "ne",
    "sr": "sr",
    "nb": "nb",
    "on": "on",
    "HI": "oh",
    "re": "re",
    "oh": "oh",
    "IN": "oi",
    "en": "en",
    "ra": "ra",
    "dtp": "dtp",
    "fi": "fi",
    "lt": "lt",
    "oi": "oi",
    "OP": "oo",
    "rv": "rv",
    "ob": "ob",
    "rs": "rs",
    "av": "av",
    "oo": "oo",
    "IP": "oe",
    "ds": "ds",
    "ed": "ed",
    "oe": "oe",
    "": "",
}


def get_label_scheme(label_list):
    if label_list in ["all", "all_2"]:
        return labels_all
    elif label_list in ["all_other"]:
        return labels_all_other
    elif label_list == "all_flat":
        return labels_all_flat
    elif label_list == "upper":
        return labels_upper
    elif label_list == "xgenre":
        return labels_xgenre


def normalize_labels(labels, label_config):
    # Normalizer-mapping
    mapping = map_normalize

    # Optionally add some combining to upper categories
    if label_config == "all_2":
        mapping = mapping.update(map_optional)

    # Split labels to a list and map
    if type(labels) == str:
        labels = (labels or "").split()

    mapped = [mapping[label] for label in labels]

    # Make sure that sublabels have corresponding upper labels
    for label in mapped:
        if label in map_lower_upper and map_lower_upper[label] not in mapped:
            mapped.append(map_lower_upper[label])

    # In the "all_other" scheme, we add the "other" labels
    if label_config in ["all_other"]:
        for label in mapped:
            if label in map_upper_lower and not any(
                element in mapped for element in map_upper_lower[label]
            ):
                mapped.append(map_upper_lower[label][-1])

    if label_config in ["all_flat", "xgenre"]:
        # Remove upper category if lower present (needed for XGENRE mapping)
        mapped_simple = []
        for label in mapped:
            if not (
                label in map_upper_lower
                # Check if any of the subcategories of the current label are in the list
                and any(element in mapped for element in map_upper_lower[label])
            ):
                mapped_simple.append(label)

        mapped = mapped_simple

        # flatten
        mapped = [map_flat[label] for label in mapped]

    # Further map to XGENRE
    if label_config == "xgenre":
        mapped = [map_xgenre[label] for label in mapped]

    # Further map to upper
    elif label_config == "upper":
        mapped = [map_upper[label] for label in mapped]

    return sorted(list(set(filter(None, mapped))))


def binarize_labels(labels, label_config):
    label_scheme = get_label_scheme(label_config)

    return [1 if scheme_label in labels else 0 for scheme_label in label_scheme]
