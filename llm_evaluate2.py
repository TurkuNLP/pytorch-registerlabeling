import pandas as pd
from tqdm import tqdm
from src import labels
from sklearn.metrics import precision_recall_fscore_support, classification_report

# You might need to import tqdm's pandas integration explicitly
tqdm.pandas()

# Define the path to the directories containing the files
base_path = "data/en/"
file_names = ["dev_full_gen_new_prompt.tsv"]


def normalize_gen(label):
    label = list(set(label.split()))

    return label


# Process each file
for file_name in file_names:
    # Construct the file path
    file_path = base_path + file_name

    # Read the TSV file into a DataFrame
    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["true_labels", "gen_labels", "text"],
        na_values="",  # Don't interpret NA as NaN!
        keep_default_na=False,
    )

    # Strip whitespace from strings in the DataFrame
    df["true_labels"] = df["true_labels"].str.strip()
    df["gen_labels"] = df["gen_labels"].str.strip()
    df["text"] = df["text"].str.strip()

    df.dropna(inplace=True)

    # Filter out rows where either 'true_labels' or 'text' are empty
    df = df[(df["true_labels"] != "") & (df["gen_labels"] != "") & (df["text"] != "")]

    true_labels = []
    gen_labels = []
    a = 0
    for row in df.iterrows():
        a += 1
        true_label = sorted(labels.normalize_labels(row[1]["true_labels"], "all"))
        gen_label = sorted(normalize_gen(row[1]["gen_labels"]))

        true_label_bin = labels.binarize_labels(true_label, "all")
        gen_label_bin = labels.binarize_labels(gen_label, "all")

        print(true_label)
        print(gen_label)

        if true_label != gen_label:
            print(row[1]["text"][:3000])

        print(f"-{a}--")
        true_labels.append(true_label_bin)
        gen_labels.append(gen_label_bin)

        if a > 1000:
            break

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, gen_labels, average="micro"
    )

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")

    cl_report = classification_report(
        true_labels,
        gen_labels,
        target_names=labels.label_schemes["all"],
        digits=4,
    )

    print(cl_report)
