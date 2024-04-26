import csv
import sys

import torch
from captum.attr import LayerIntegratedGradients, visualization
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tqdm import tqdm

sys.path.append(
    f"venv/lib/python{'.'.join(map(str, sys.version_info[:3]))}/site-packages"
)

csv.field_size_limit(sys.maxsize)

from src.data import small_languages
from src.labels import (
    binarize_labels,
    label_schemes,
    labels_structure,
    normalize_labels,
)

label_scheme = label_schemes["all"]


def group_labels_by_parent(example_labels):
    # Create a dictionary to map each child label to its parent
    child_to_parent = {}
    for parent, children in labels_structure.items():
        for child in children:
            child_to_parent[child] = parent

    # Create a dictionary to hold the groups
    grouped_labels = {}

    # Group labels according to their parent
    for label in example_labels:
        if label in labels_structure:
            # This is a parent label
            if label not in grouped_labels:
                grouped_labels[label] = []
            grouped_labels[label].append(label)
        elif label in child_to_parent:
            # This is a child label
            parent = child_to_parent[label]
            if parent not in grouped_labels:
                grouped_labels[parent] = []
            grouped_labels[parent].append(label)

    # Convert the dictionary to a list of lists for the grouped labels
    return list(grouped_labels.values())


def print_aggregated(target, txt, real_label):

    print("<html><body>")

    x = visualization.format_word_importances([t for t, a in txt], [a for t, a in txt])
    print(f"<b>prediction: {target}, real label: {real_label}</b>")
    print(f"""<table style="border:solid;">{x}</table>""")
    print("</body></html>")


def aggregate(scores, tokens, special_tokens):
    scores = scores.cpu().tolist()

    # Initialize variables
    current_word = None
    max_abs_score = float("-inf")
    max_score = None
    word_scores = []

    # Process each token and corresponding score
    for score, token in zip(scores, tokens):
        if token in special_tokens:
            continue

        if token.startswith("▁"):  # This token is the start of a new word
            if current_word is not None:  # Save the max score of the previous word
                word_scores.append((current_word, max_score))
            current_word = token[1:]  # Start a new word (omit the initial "▁")
            max_score = score  # Reset the max score for the new word
            max_abs_score = abs(score)  # Reset the max absolute score for the new word
        else:
            if current_word is not None:
                current_word += token  # Append token to the current word
                if (
                    abs(score) > max_abs_score
                ):  # Check if the absolute value of the score is greater
                    max_score = score  # Update max score
                    max_abs_score = abs(score)  # Update max absolute score

    # Don't forget to save the last word's score
    if current_word is not None:
        word_scores.append((current_word, max_score))

    return word_scores


def perform_ig(inputs, blank_input_ids, idx, model, tokenizer):

    def predict_f(pred_inputs, attention_mask=None):
        return model(pred_inputs, attention_mask=attention_mask).logits

    def custom_f(inputs, attention_mask, target_indices):
        outputs = model(
            inputs, attention_mask
        ).logits  # Assuming the model returns raw logits
        target_outputs = outputs[:, target_indices]  # Select outputs for target classes
        return torch.mean(target_outputs, dim=1)

    lig = LayerIntegratedGradients(custom_f, model.roberta.embeddings)

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

    attrs = lig.attribute(
        inputs=(inputs.input_ids, inputs.attention_mask),
        baselines=(blank_input_ids, inputs.attention_mask),
        additional_forward_args=(idx,),
        target=None,
        internal_batch_size=10,
        n_steps=50,
    )

    attrs_sum = attrs.sum(dim=-1).squeeze(0)
    attrs_sum = attrs_sum / torch.norm(attrs_sum)
    aggregated_tg = aggregate(attrs_sum, tokens, tokenizer.all_special_tokens)
    word_visualizations = visualization.format_word_importances(
        [t for t, _ in aggregated_tg], [a for _, a in aggregated_tg]
    )
    return word_visualizations


def analyse_ig(
    train_languages,
    test_language,
    true_labels_filter,
    predicted_labels_filter,
    data_filename,
    ig_path,
    data,
):

    inner_html = []

    print("Loading model and tokenizer...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = f"models/xlm-roberta-large/labels_all/{train_languages}_{train_languages}/seed_42"

    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    print("Extracting IGs...")
    for row in tqdm(data):

        inputs = tokenizer(
            [row["text"]],
            return_tensors="pt",
            return_special_tokens_mask=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        blank_input_ids = inputs.input_ids.clone().detach()
        blank_input_ids[inputs.special_tokens_mask == 0] = tokenizer.pad_token_id

        true_labels = row["gold"].split()
        true_bin = binarize_labels(true_labels, "all")
        true_indexes = [index for index, value in enumerate(true_bin) if value == 1]

        pred_labels = row["pred"].split()
        pred_bin = binarize_labels(pred_labels, "all")
        pred_indexes = [index for index, value in enumerate(pred_bin) if value == 1]
        pred_attributions = None

        true_attributions = perform_ig(
            inputs, blank_input_ids, true_indexes, model, tokenizer
        )

        if pred_bin != true_bin:
            pred_attributions = perform_ig(
                inputs, blank_input_ids, pred_indexes, model, tokenizer
            )

        true_text = (
            f"Token contributions relative to true labels: {', '.join(true_labels)}"
        )

        if pred_bin == true_bin:
            true_text += " (Correctly predicted)"

        html_block = f"""
            <h3>Row {row['row_idx']}</h3>
            <h4>{true_text}</h4>
            <table style="border:solid;">{true_attributions}</table>
        """

        if pred_bin != true_bin:
            html_block += f"""
                <h4>Token contributions relative to wrongly predicted labels: {', '.join(pred_labels)}</h4>"
                <table style="border:solid;">{pred_attributions}</table>
            """

        inner_html.append(html_block)
    inner_html = "\n".join(inner_html)
    style = """
        body {
            font-family:sans-serif;
            font-family: sans-serif;
            margin: 20px auto;
            max-width: 960px;
            padding:0 20px;
        }
        table {
            border: 1px solid #aaa;
            border-radius: 5px;
            margin: 10px 0 20px;
            padding: 10px;
        }
    """
    html = f"""
        <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>Integrated Gradients</title>
            </head>
            <style>
                {style}

            </style>
                <div style="text-align:center;">
                    <h1>Integrated Gradients</h1>
                    <h2>Model: XLM-RoBERTa-Large, Train: {train_languages}, Test: {test_language}, File: {data_filename}</h2>
                    <h3>True labels filter: {true_labels_filter}, Pred labels filter: {predicted_labels_filter}</h3>
                </div>
                {inner_html}
            </body>
        </html>
    """

    with open(ig_path, "w") as file:
        file.write(html)


def print_with_background(list_items, background_color, text_color=30):
    for i, item in enumerate(list_items):
        list_items[i] = f"\033[{text_color};{background_color}m {item} \033[0m"
    return list_items


def criterion(true_label_test, predicted_label_test, true_label, pred_label):
    return true_label_test in true_label and predicted_label_test in pred_label


def filter_rows(train_languages, test_language, true_label, predicted_label, ig_path):
    lang = test_language
    target = lang if lang in small_languages else "test"

    # Path to the data file
    data_filename = f"data/{lang}/{target}.tsv"

    with open(data_filename, "r", newline="", encoding="utf-8") as data_file:
        data_reader = list(csv.reader(data_file, delimiter="\t"))

    with open(
        f"predictions/xlm-roberta-large/{train_languages}_{train_languages}/seed_42/all_all_{lang}.tsv",
        "r",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="\t")

        results = []

        for row in reader:
            if len(row) < 3:
                print(
                    "Please ensure that the file is in the format [true_label, predicted_label, row_idx]"
                )
                exit()

            if criterion(true_label, predicted_label, row[0], row[1]):
                orig_label = data_reader[int(row[2])][0]
                orig_label_norm = normalize_labels(orig_label, "all")
                gold_norm = normalize_labels(row[0], "all")
                if orig_label_norm != gold_norm:
                    print("There was a label mismatch, exiting...")
                    exit()

                results.append(
                    {
                        "gold": row[0],
                        "pred": row[1],
                        "file": data_filename,
                        "row_idx": row[2],
                        "text": data_reader[int(row[2])][1],
                    }
                )

        for row in results:
            gold = []
            for label in row["gold"].split():
                if label not in row["pred"]:
                    gold.append(print_with_background([label], 43)[0])
                else:
                    gold.append(print_with_background([label], 42)[0])

            pred = []
            for label in row["pred"].split():
                if label in row["gold"]:
                    pred.append(print_with_background([label], 42)[0])
                else:
                    pred.append(print_with_background([label], 41, 37)[0])
            pred = " ".join(pred)
            gold = " ".join(gold)
            print(f"True: {gold}, Pred: {pred}, Text: {row['file']} [{row['row_idx']}]")
            print(row["text"])
            print("-" * 50)

        if ig_path:
            print("")
            print(f"Now saving IG visualisations to {ig_path}")
            analyse_ig(
                train_languages,
                test_language,
                true_label,
                predicted_label,
                data_filename,
                ig_path,
                results,
            )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print(
            "Usage: python extract_predictions.py <train-language(s)> <test-language> <true_label> <predicted_label> <save_ig_path (optional)>"
        )
        sys.exit(1)

    train_languages = sys.argv[1]
    test_language = sys.argv[2]
    true_label = sys.argv[3]
    predicted_label = sys.argv[4]
    ig_path = ""
    if len(sys.argv) == 6:
        ig_path = sys.argv[5]

    filter_rows(train_languages, test_language, true_label, predicted_label, ig_path)
