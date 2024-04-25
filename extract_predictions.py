import csv
import sys

csv.field_size_limit(sys.maxsize)

import captum

from src.data import small_languages
from src.labels import normalize_labels


def print_aggregated(target, txt, real_label):
    """ "
    This requires one target and one agg vector at a time
    Shows agg scores as colors
    """
    print("<html><body>")

    x = captum.attr.visualization.format_word_importances(
        [t for t, a in txt], [a for t, a in txt]
    )
    print(f"<b>prediction: {target}, real label: {real_label}</b>")
    print(f"""<table style="border:solid;">{x}</table>""")
    print("</body></html>")


def print_with_background(list_items, background_color):
    for i, item in enumerate(list_items):
        list_items[i] = f"\033[30;{background_color}m {item} \033[0m"
    return list_items


def filter_rows(filename, true_label, predicted_label, ig):
    lang = filename.split("_")[-1].split(".")[0]
    target = lang if lang in small_languages else "test"

    # Path to the data file
    data_filename = f"data/{lang}/{target}.tsv"

    with open(data_filename, "r", newline="", encoding="utf-8") as data_file:
        data_reader = list(csv.reader(data_file, delimiter="\t"))

    with open(filename, "r", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        results = []

        for row in reader:
            if len(row) < 3:
                print(
                    "Please ensure that the file is in the format [true_label, predicted_label, row_idx]"
                )
                exit()

            if true_label in row[0] and predicted_label in row[1]:
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
            if not ig:
                gold = " ".join(print_with_background(row["gold"].split(), 42))
                pred = []
                for label in row["pred"].split():
                    if label in row["gold"]:
                        pred.append(print_with_background([label], 42)[0])
                    else:
                        pred.append(print_with_background([label], 43)[0])
                pred = " ".join(pred)
                print(
                    f"Gold: {gold}, Pred: {pred}, Text: {row['file']}[{row['row_idx']}]"
                )
                print(row["text"])
                print("-" * 50)


if __name__ == "__main__":
    import sys

    print("\nNote: this only works with the full taxomomy.\n")

    if len(sys.argv) < 4:
        print(
            "Usage: python script.py <filename> <true_label> <predicted_label> <ig (optional)>"
        )
        sys.exit(1)

    filename = sys.argv[1]
    true_label = sys.argv[2]
    predicted_label = sys.argv[3]
    ig = False
    if len(sys.argv) == 5:
        if sys.argv[4] == "ig":
            ig = True

    filter_rows(filename, true_label, predicted_label, ig)
