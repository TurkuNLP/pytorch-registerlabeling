import csv
from itertools import combinations
import re, ast
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as colors
import seaborn as sns
from datasets import concatenate_datasets
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from scipy.stats import ttest_rel
from scipy.stats import chi2
from sklearn.metrics import f1_score

from .data import get_dataset, language_names, small_languages, preprocess_data
from .labels import (
    binarize_labels,
    get_label_scheme,
    labels_all_hierarchy,
    labels_all_flat,
    map_full_names,
    map_xgenre,
)

from .utils import parse_result_line

pio.kaleido.scope.mathjax = None  # a fix for .pdf files


class Stats:
    def __init__(self, cfg):
        self.cfg = cfg
        self.labels = get_label_scheme(cfg.data.labels)
        self.palette = sns.color_palette("Blues")
        self.palette[0] = (1, 1, 1)
        self.template = "plotly_white"

        # Run
        getattr(self, cfg.method)()

    def get_dataframe(
        self, remove_redundant_parents=False, cols=["label_text", "language"]
    ):
        data = get_dataset(self.cfg)
        data = concatenate_datasets([data["train"], data["dev"], data["test"]])
        df = pd.DataFrame(data)[cols] if cols else pd.DataFrame(data)
        df["label_text"] = df["label_text"].str.split(" ")

        if remove_redundant_parents:

            redundant_parents = [
                x
                for x in labels_all_hierarchy.keys()
                if len(labels_all_hierarchy[x]) > 1
            ]

            # Filtering out redundant parents
            df["label_text"] = df["label_text"].apply(
                lambda labels: [
                    label for label in labels if label not in redundant_parents
                ]
            )
        return df

    def get_ordered_data(self, df, reverse=True):
        # Explode
        df = df.explode("label_text")

        label_to_category = {
            label: category
            for category, labels in labels_all_hierarchy.items()
            for label in labels
        }

        subcategory_counts = (
            df.groupby(["label_text", "language"])
            .size()
            .unstack(fill_value=0)
            .to_dict(orient="index")
        )

        # Adding a "total" key to each dict item
        for label, lang_counts in subcategory_counts.items():
            subcategory_counts[label]["total"] = sum(lang_counts.values())

        category_counts = {
            key: {"total": 0, "subcategories": {}}
            for key in labels_all_hierarchy.keys()
        }

        xgenre_counts = {}

        for key in subcategory_counts.keys():
            total = subcategory_counts[key].pop("total")
            category = label_to_category[key]
            xgenre = map_xgenre[key]
            category_counts[category]["subcategories"][key] = {
                "val": total,
                "xgenre": xgenre,
                "lang": subcategory_counts[key],
            }
            if xgenre not in xgenre_counts:
                xgenre_counts[xgenre] = 0
            xgenre_counts[xgenre] += total
            category_counts[category]["total"] += total

        # Sort categories
        category_counts_sorted = dict(
            sorted(
                category_counts.items(), key=lambda x: x[1]["total"], reverse=reverse
            )
        )

        # Sort labels
        for key in category_counts_sorted.keys():
            sorted_subcategories = dict(
                sorted(
                    category_counts[key]["subcategories"].items(),
                    key=lambda x: x[1]["val"],
                    reverse=reverse,
                )
            )
            category_counts[key]["subcategories"] = sorted_subcategories

        data = category_counts_sorted

        return data

    def heatmap(self, data):
        # Data for the heatmap
        confusion_matrix_data = data["data"]

        # Axis labels
        x_labels = data["x_labels"]
        y_labels = data["y_labels"]

        # Create heatmap
        fig = px.imshow(
            confusion_matrix_data,
            labels=data["labels"],
            x=x_labels,
            y=y_labels,
            aspect="auto",
            color_continuous_scale=[mcolors.to_hex(color) for color in self.palette],
            title=data.get("title", None),
            text_auto=data["text_auto"],
            range_color=data["range_color"],
        )

        fig.update_yaxes(ticksuffix=" ")
        fig.update_layout(
            margin=go.layout.Margin(l=5, r=5, b=5, t=5),
            showlegend=False,
            title=None,
            coloraxis_showscale=False,
            font=data["font"],
        )

        fig.update_xaxes(side="top", tickvals=x_labels)
        fig.update_yaxes(side="left", tickvals=y_labels)

        # Show plot
        fig.write_image(f"output/{data['output']}", scale=3)

    def stacked_bars(self):
        df = self.get_dataframe(True)

        data = self.get_ordered_data(df)

        subcategories = []
        langs = {"en": [], "fi": [], "fr": [], "sv": [], "tr": []}
        lang_full_names = {
            "en": "English",
            "sv": "Swedish",
            "fi": "Finnish",
            "fr": "French",
            "tr": "Turkish",
        }

        shapes = []  # List to store shapes for horizontal lines

        # Counter to keep track of subcategory index
        subcat_index = 0

        for region in data.values():
            print(region)
            for subcat, details in region["subcategories"].items():
                print(map_full_names.get(subcat, subcat))
                subcategories.append(map_full_names.get(subcat, subcat))
                for lang, count in details["lang"].items():
                    langs[lang].append(count)
                subcat_index += 1

            print(subcat_index)
            # Adding a horizontal line at the end of each main category
            shapes.append(
                {
                    "type": "line",
                    "x0": 0,
                    "y0": subcat_index - 0.5,
                    "x1": 1,
                    "y1": subcat_index - 0.5,
                    "xref": "paper",
                    # "yref": "y",
                    "line": {
                        "color": "#aaaaaa",
                        "width": 0.5,
                    },
                }
            )

        # Creating the plot
        fig = go.Figure()
        color_i = 0
        print(subcategories)
        for lang, counts in langs.items():
            fig.add_trace(
                go.Bar(
                    name=lang_full_names[lang],
                    y=subcategories,
                    x=counts,
                    orientation="h",
                    marker_color=sns.color_palette("Blues", n_colors=6).as_hex()[
                        color_i + 1
                    ],
                    # marker_color=colors.qualitative.Plotly[(color_i + 1) - 1],
                )
            )
            color_i += 1

        annotations = []
        current_index = 0
        for region_name, region in data.items():
            region_length = len(region["subcategories"])
            annotations.append(
                go.layout.Annotation(
                    x=0.99,  # Adjust as needed for proper alignment
                    y=current_index + region_length - 1 - ((region_length - 1) / 2),
                    xref="paper",
                    yref="y",
                    text="<b style='font-weight:400;color:black'>"
                    + map_full_names.get(region_name, region_name)
                    + "</b>",
                    showarrow=False,
                    # bgcolor="white",
                    opacity=0.8,
                )
            )
            current_index += region_length

        width = 900
        height = width / 1.618

        fig.update_layout(
            annotations=annotations,
            width=width,
            height=height,
            margin=go.layout.Margin(l=5, r=5, b=5, t=5, pad=4),
            yaxis=dict(autorange="reversed"),
            template=self.template,
            barmode="stack",
            shapes=shapes[:-1],  # Adding shapes to the layout
            legend_traceorder="normal",
        )
        fig.update_yaxes(ticksuffix="  ")

        fig.show()
        fig.write_image("output/stacked.pdf")

    def label_distribution(self):
        df = self.get_dataframe(True)
        # Converting each list in 'label_text' to a set to remove duplicates
        df["label_text"] = df["label_text"].apply(lambda x: [y.lower() for y in x])
        df["label_text"] = df["label_text"].apply(set)
        # Calculate the number of labels per example
        num_labels_per_example = df["label_text"].apply(len)

        # Calculate the frequency of each label count
        label_count_freq = num_labels_per_example.value_counts().sort_index()

        print(label_count_freq)

        # Assuming df is your DataFrame and 'label_text' is the column with label lists
        df_filtered = df[
            num_labels_per_example == 2
        ]  # Filter for examples with exactly 2 labels

        # Generate a Series of sorted tuples for each pair of labels
        label_pairs = df_filtered["label_text"].apply(
            lambda labels: tuple(sorted(labels))
        )

        # Count the frequency of each unique pair
        common_pairs_freq = label_pairs.value_counts()

        # Display the most common combinations
        print(
            common_pairs_freq.head(50)
        )  # Adjust the number in head() to see more or fewer top combinations

    def co_occurrence_heatmap(self):
        df = self.get_dataframe(True)
        df["label_text"] = (
            df["label_text"].apply(lambda x: [y.lower() for y in x]).apply(set)
        )

        total_occurrences = df["label_text"].explode().value_counts()

        label_data = list(
            df["label_text"].apply(
                lambda x: [list(x) for x in list(combinations(x, 2))]
            )
        )

        connections = [list(sorted(item)) for sublist in label_data for item in sublist]
        labels_all_flat = sorted(set(total_occurrences.index))

        # Create a DataFrame to count occurrences
        co_occurrence_matrix = pd.DataFrame(
            0, index=labels_all_flat, columns=labels_all_flat
        )

        # Counting occurrences for each pair
        for pair in connections:
            co_occurrence_matrix.loc[pair[0], pair[1]] += 1
            co_occurrence_matrix.loc[
                pair[1], pair[0]
            ] += 1  # For bidirectional relationship

        # Normalize the counts by the sum of occurrences for the labels in each pair
        for label1 in labels_all_flat:
            for label2 in labels_all_flat:
                total = total_occurrences[label1] + total_occurrences[label2]
                co_occurrence_matrix.loc[label1, label2] = (
                    (co_occurrence_matrix.loc[label1, label2] / total)
                    if total > 0
                    else 0
                )

        # Order labels based on their total occurrences
        ordered_labels = total_occurrences.sort_values(ascending=False).index.tolist()
        print(total_occurrences.sort_values(ascending=False))
        co_occurrence_matrix = co_occurrence_matrix.loc[ordered_labels, ordered_labels]

        # Mask the upper left triangle of the DataFrame for the triangular heatmap
        mask = np.triu(np.ones_like(co_occurrence_matrix, dtype=bool))
        triangular_df = co_occurrence_matrix.mask(mask)

        palette = [mcolors.to_hex(color) for color in self.palette]

        fig = px.imshow(
            triangular_df,
            x=[map_full_names.get(x, x) for x in ordered_labels],
            y=[map_full_names.get(x, x) for x in ordered_labels],
            color_continuous_scale=palette,
            template=self.template,
            text_auto=".2f",
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        fig.update_yaxes(ticksuffix=" ")
        fig.update_layout(
            margin=go.layout.Margin(l=5, r=5, b=5, t=5),
            showlegend=False,
            template=self.template,
            title=None,
            coloraxis_showscale=False,
            font=dict(size=6),
            autosize=False,
            width=500,
            height=500,
        )

        fig.write_image(f"output/heatmap_co_occurrences.png", scale=4)

    def text_lengths(self):

        language_full_names = {
            "en": "English",
            "fi": "Finnish",
            "fr": "French",
            "sv": "Swedish",
            "tr": "Turkish",
        }

        dataset = get_dataset(self.cfg)
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
        dataset = dataset.map(
            lambda example: tokenizer(example["text"]),
            batched=True,
        )

        dataset = concatenate_datasets(
            [dataset["train"], dataset["dev"], dataset["test"]]
        )

        data = [(len(x["input_ids"]), x["language"]) for x in dataset]

        over_8k = sum([1 if x[0] > 8192 else 0 for x in data])
        print(over_8k)

        print(f"Excluded {over_8k/len(data)} texts.")

        data = [x for x in data if x[0] <= 8192]

        # Organize data by language
        text_lengths_by_language = {"en": [], "fi": [], "fr": [], "sv": [], "tr": []}
        for length, language in data:
            text_lengths_by_language[language].append(length)

        # Create a box plot for each language
        fig = go.Figure()
        lang_i = 0
        for lang, lengths in text_lengths_by_language.items():
            full_name = language_full_names[lang]  # Convert code to full name
            fig.add_trace(
                go.Box(
                    y=lengths,
                    name=full_name,
                    # marker_color=sns.color_palette("Blues", n_colors=6).as_hex()[
                    #    lang_i + 1
                    # ],
                )
            )
            lang_i += 1

        fig.update_layout(
            title=None,
            showlegend=False,
            template=self.template,
            margin=go.layout.Margin(l=5, r=5, b=5, t=5),
        )

        fig.write_image(f"output/text_lengths.pdf")

    def prediction_confusion_matrix(self):
        with open(f"{self.cfg.input}", "r") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter="\t")
            data = list(csv_reader)

        # Extract true and predicted labels from the data
        true_labels, predicted_labels = zip(*data)

        true_labels_binary = [
            binarize_labels(label.split(), self.cfg.data.labels)
            for label in true_labels
        ]
        predicted_labels_binary = [
            binarize_labels(label.split(), self.cfg.data.labels)
            for label in predicted_labels
        ]

        confusion_matrix = np.zeros((len(self.labels), len(self.labels)), dtype=float)

        def get_combined(T, P):
            T_ = np.array(T) & (~np.array(P) & 1)  # FN
            P_ = np.array(P) & (~np.array(T) & 1)  # FP

            sumP = np.sum(P) or 1
            sumT = np.sum(T) or 1
            sumP_ = np.sum(P_) or 1

            if all(T_ == P_):
                M = np.diag(T)

            if all(T_ == 0) and any(P_ == 1):
                M = (np.outer(T, P_) + sumT * np.diag(T)) / sumP

            if all(P_ == 0) and any(T_ == 1):
                M = np.outer(T_, P) / sumP + np.diag(P)

            if any(T_ == 1) and any(P_ == 1):
                M = np.outer(T_, P_) / sumP_ + np.diag(T & P)

            return M

        for true_labels, predicted_labels in zip(
            true_labels_binary, predicted_labels_binary
        ):
            T = np.array(true_labels)
            P = np.array(predicted_labels)
            M = get_combined(T, P)

            confusion_matrix += M

        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_confusion_matrix = confusion_matrix / row_sums

        normalized_confusion_matrix_data = np.nan_to_num(normalized_confusion_matrix)

        palette = [mcolors.to_hex(color) for color in self.palette]

        confusion_matrix_fig = px.imshow(
            normalized_confusion_matrix_data,
            x=self.labels,
            y=self.labels,
            color_continuous_scale=palette,
            title="Confusion Matrix",
            color_continuous_midpoint=0,
            zmin=0,
            zmax=1,
            text_auto=".2f",
        )

        confusion_matrix_fig.update_yaxes(ticksuffix=" ")
        confusion_matrix_fig.update_layout(
            margin=go.layout.Margin(l=5, r=5, b=5, t=5),
            showlegend=False,
            title=None,
            coloraxis_showscale=False,
            font=dict(size=9),
            autosize=False,
            width=500,
            height=500,
        )
        name = self.cfg.input.split("/")[-1].split(".")[0]
        confusion_matrix_fig.write_image(f"output/heatmap_{name}.png", scale=4)

    def focal_loss_configuration_heatmap(self):
        self.heatmap(
            {
                "data": np.array(
                    [
                        [0.7557777533604367, 0.752711533565306, 0.7479257914040522],
                        [0.7334222495810677, 0.7582355845349766, 0.75156109860728],
                        [0.7634568756494631, 0.763710514077201, 0.7624217118997911],
                    ]
                ),
                "x_labels": [1, 2, 3],
                "y_labels": [0.25, 0.5, 0.75],
                "labels": {"x": "Gamma", "y": "Alpha", "color": "Value"},
                "text_auto": ".4f",
                "range_color": [0.72, 0.765],
                "font": {"size": 20},
                "output": "focal_heatmap.png",
            }
        )

    def inference_time_heatmap(self):
        self.heatmap(
            {
                "data": np.array(
                    [
                        [17.052708, 0, 22.289936, 0],
                        [18.166394, 0, 0, 0],
                        [0, 0, 0, 0],
                    ]
                ),
                "x_labels": ["xlmr-large", "xlmr-xl", "bge", "fourth model"],
                "y_labels": ["All", "Upper", "XGENRE"],
                "labels": {"x": "Model", "y": "Label taxonomy", "color": "Value"},
                "text_auto": ".4f",
                "range_color": [0, 30],
                "font": {"size": 20},
                "output": "inference_time_heatmap.png",
            }
        )

    def sm_zero_shot(self):
        data = {}
        labels_mapped = [f"{map_full_names[x]} ({x})" for x in self.labels]

        for l in small_languages:
            # if l in ["id"]:
            #    continue
            with open(f"{self.source_path}/test_{l}.csv", "r") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter="\t")
                lang_data = list(csv_reader)

                trues, preds = list(zip(*lang_data))

                true_labels_binary = [
                    binarize_labels(label.split(), self.label_scheme) for label in trues
                ]
                predicted_labels_binary = [
                    binarize_labels(label.split(), self.label_scheme) for label in preds
                ]

                rep = classification_report(
                    true_labels_binary,
                    predicted_labels_binary,
                    target_names=self.labels,
                    output_dict=True,
                    digits=4,
                )
                rep = {k: v for k, v in rep.items() if k in self.labels}

                language_data = {
                    f"{map_full_names[k]} ({k})": (
                        v["f1-score"] if v["f1-score"] != 0 else 0.005
                    )
                    for k, v in rep.items()
                    if v["support"] >= 20
                }
                if language_data:
                    data[language_names[l]] = language_data

        combined_set = set().union(*[list(subdict.keys()) for subdict in data.values()])
        categories = [x for x in labels_mapped if x in combined_set]
        languages = list(data.keys())

        bar_traces = [
            go.Bar(
                orientation="h",
                name=lang,
                y=list(cat.keys()),
                x=list(cat.values()),
            )
            for lang, cat in data.items()
        ]

        category_data = [
            [data[lang][cat] for lang in languages if cat in data[lang]]
            for cat in categories
        ]

        categories = [x for i, x in enumerate(categories) if len(category_data[i]) > 1]

        # categories = [x for x in enumerate(categories)]

        mean_values = [np.mean(x) if len(x) > 1 else None for x in category_data]
        std_dev_values = [np.std(x) if len(x) > 1 else None for x in category_data]

        std_dev_trace = go.Scatter(
            y=categories,
            x=mean_values,
            mode="markers",
            name="Mean and std",
            marker=dict(color="#333", size=5),
            error_x=dict(type="data", array=std_dev_values),
            orientation="h",
        )

        fig = go.Figure(data=bar_traces + [std_dev_trace])

        fig.update_layout(
            barmode="group",
            yaxis={"rangemode": "tozero", "autorange": "reversed"},
            xaxis=dict(automargin=True),
            # yaxis_title="Predicted register",
            xaxis_title="F1-score (micro avg.)",
        )
        fig.update_yaxes(ticksuffix=" ")
        fig.show()

    def sankey_plot(self):
        df = self.get_dataframe(self.cfg)

        data = self.get_ordered_data(df)
        # colors = px.colors.sequential.Agsunset

        # Initialize lists for source, target, value, labels, and positions
        source = []
        target = []
        value = []
        label = []
        x_pos = []
        y_pos = []

        palette = sns.color_palette("Blues", n_colors=12).as_hex()[1:]
        palette2 = sns.color_palette("PuBu", n_colors=12).as_hex()[3:]
        # palette2 = sns.cubehelix_palette(rot=-0.3, n_colors=20).as_hex()[1:]

        # Define a color scheme for main categories
        # Define a color scheme for main categories and rightmost column nodes
        main_category_colors = {
            main_source: palette[i] for i, main_source in enumerate(data.keys())
        }

        rightmost_node_colors = {}
        color_counter = 0
        for main_source, sub_data in data.items():
            for sub_source, sub_details in sub_data["subcategories"].items():
                xgenre = sub_details["xgenre"]
                if xgenre not in rightmost_node_colors:
                    rightmost_node_colors[xgenre] = palette2[color_counter]
                    color_counter += 1

        # Initialize a list for link colors
        link_colors = []

        # Initialize a list for link colors
        link_colors = []

        # Function to incrementally assign x and y positions
        def assign_positions(label_name, x_value, y_increment, current_y):
            if label_name not in label:
                label.append(label_name)
                x_pos.append(x_value)
                y_position = min(
                    current_y * y_increment, 0.95
                )  # Ensure y doesn't exceed 0.95
                y_pos.append(y_position)
                return current_y + 1
            return current_y

        # Initialize current_y counters for main sources and subcategories
        current_y_main = 1
        current_y_sub = 1

        # Populate the source, target, value, and label lists
        y_increment_main = 0.8 / (len(data) + 1)  # Increment for main sources
        y_increment_sub = 0.8 / (
            sum(len(sub_data["subcategories"]) for sub_data in data.values()) + 1
        )  # Increment for subcategories

        for main_source, sub_data in data.items():
            current_y_main = assign_positions(
                main_source, 0.05, y_increment_main, current_y_main
            )

            if (
                len(sub_data["subcategories"]) == 1
                and main_source in sub_data["subcategories"]
            ):
                xgenre = sub_data["subcategories"][main_source]["xgenre"]
                current_y_sub = assign_positions(
                    xgenre, 0.95, y_increment_sub, current_y_sub
                )
                source.append(label.index(main_source))
                target.append(label.index(xgenre))
                value.append(sub_data["subcategories"][main_source]["val"])
                link_colors.append(main_category_colors[main_source])
            else:
                for sub_source, sub_details in sub_data["subcategories"].items():
                    current_y_sub = assign_positions(
                        sub_source, 0.5, y_increment_sub, current_y_sub
                    )
                    source.append(label.index(main_source))
                    target.append(label.index(sub_source))
                    value.append(sub_details["val"])

                    xgenre = sub_details["xgenre"]
                    current_y_sub = assign_positions(
                        xgenre, 0.95, y_increment_sub, current_y_sub
                    )
                    source.append(label.index(sub_source))
                    target.append(label.index(xgenre))
                    value.append(sub_details["val"])
                    link_colors.append(main_category_colors[main_source])

                    # Link from subcategory (middle column) to rightmost column
                    xgenre = sub_details["xgenre"]
                    link_colors.append(rightmost_node_colors[xgenre])

        # Create the Sankey diagram
        fig = go.Figure(
            data=[
                go.Sankey(
                    arrangement="snap",
                    node=dict(
                        label=[
                            '<span style="paint-order:stroke;stroke-width:0px;stroke:white;">'
                            + map_full_names.get(x, x)
                            + "</span>"
                            for x in label
                        ],
                        x=x_pos,
                        y=y_pos,
                        thickness=1,
                        pad=15,
                        color="white",
                        line=dict(color="white", width=0.001),
                    ),
                    link=dict(
                        source=source, target=target, value=value, color=link_colors
                    ),
                )
            ]
        )
        width = 1000
        height = width / 1.618

        # Set the layout
        fig.update_layout(
            template=self.template,
            width=width,
            height=height,
            margin=go.layout.Margin(l=5, r=5, b=15, t=5),
            # paper_bgcolor="rgba(0,0,0,0)",
            # plot_bgcolor="rgba(0,0,0,0)",
        )

        # Show the figure
        fig.show()
        fig.write_image("output/sankey.pdf")

    def compare_zero_shots(self):
        models = ["xlmr-large", "xlmr-xl", "bge-m3-retromae-2048"]
        scores = []

        for model in models:
            scores.append([])
            for lang in small_languages:

                with open(
                    f"v3/configs/{model}/labels_all/small_languages/{lang}.yaml"
                ) as f:
                    for line in f:
                        if line.startswith(f"#z"):
                            dict_str = re.search(r"\{.*\}", line).group()
                            result_dict = ast.literal_eval(dict_str)
                            scores[-1].append(result_dict["f1"])

        data = np.array(scores)

        print(data)

        # Calculate the number of models
        n_models = data.shape[0]

        # Calculate the number of pairwise comparisons
        n_comparisons = (n_models * (n_models - 1)) / 2

        print(n_comparisons)

        # Perform all pairwise Wilcoxon signed-rank tests
        p_values = []
        comparisons = []
        for i, j in combinations(range(n_models), 2):
            print(i, j)
            stat, p = wilcoxon(data[i], data[j])
            comparisons.append(f"{models[i]} vs. {models[j]}")
            p_values.append(p)

        # Apply Holm-Bonferroni correction
        _, result, _, _ = multipletests(p_values, alpha=0.05, method="holm")

        for test, res in zip(comparisons, result):
            print(f"{test}: {res}")

    def compare_class_results(self):

        languages = ["en", "fi", "fr", "sv", "tr"]
        symbols = ["=", "m", "z"]

        lang_scores = {
            lang: {cl: [[], [], []] for cl in self.labels} for lang in languages
        }

        cl_data = {cl: [[], [], []] for cl in self.labels}

        for lang in ["en", "fi", "fr", "sv", "tr"]:

            with open(f"v3/configs/xlmr-large/labels_all/{lang}/{lang}.yaml") as f:
                for line in f:
                    for i, s in enumerate(symbols):
                        if line.startswith(f"#{s}"):
                            result_dict = parse_result_line(line)
                            if "label_scores" not in result_dict:
                                continue
                            for label, value in result_dict["label_scores"].items():
                                lang_scores[lang][label][i].append(value["f1-score"])
                                cl_data[label][i].append(value["f1-score"])

            print(f"\n----- {lang} -----\n")

            for label, label_data in lang_scores[lang].items():
                print(f"\t{label}\n=======")
                data = np.array(label_data)

                t_stat, p_value_t_test = ttest_rel(label_data[0], label_data[1])
                print(f"\t\t {p_value_t_test}")
                try:
                    stat, p = wilcoxon(label_data[0], label_data[1])
                    print(f"\t\t {p}")
                except:
                    print("(no wilcoxon)")
                continue
        print("\n\n\n\n\n")
        for cl, cl_d in cl_data.items():
            data = np.array(cl_d)
            print(
                f"{cl}\n======\nF1-scores: \n Mono: {np.mean(cl_d[0]):.2f}, Multi: {np.mean(cl_d[1]):.2f}, ZS: {np.mean(cl_d[2]):.2f}"
            )
            print()

            # Calculate the number of models
            n_models = data.shape[0]

            # Calculate the number of pairwise comparisons
            n_comparisons = (n_models * (n_models - 1)) / 2

            # Perform all pairwise Wilcoxon signed-rank tests
            p_values = []
            comparisons = []
            full_symbols = ["Mono", "Multi", "ZS"]
            for i, j in combinations(range(n_models), 2):
                if not (np.any(data[i]) or np.any(data[j])):
                    continue
                stat, p = wilcoxon(data[i], data[j])
                comparisons.append(f"{full_symbols[i]} vs. {full_symbols[j]}")
                p_values.append(p)
            if p_values:
                # Apply Holm-Bonferroni correction
                _, result, _, _ = multipletests(p_values, alpha=0.05, method="holm")
                print("\tP-values:\n\t---------")
                for test, res in zip(comparisons, result):
                    print(f"\t\t{test}: {res:.4f}")

    def compare_class_results_mcnemar(self):

        def process_test_and_compare(list1, list2):
            num_labels = len(list1[0][0])
            results = {}

            for label_index in range(num_labels):
                a = b = c = d = 0
                true_labels_list, preds_model_1, preds_model_2 = [], [], []

                for (true_labels, pred1), (_, pred2) in zip(list1, list2):
                    t = true_labels[label_index]
                    p1, p2 = pred1[label_index], pred2[label_index]
                    true_labels_list.append(t)
                    preds_model_1.append(p1)
                    preds_model_2.append(p2)

                    if p1 == t and p2 == t:
                        d += 1
                    elif p1 == t and p2 != t:
                        c += 1
                    elif p1 != t and p2 == t:
                        b += 1
                    else:
                        a += 1

                test_statistic = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
                p_value = 1 - chi2.cdf(test_statistic, 1)

                # Calculate F1 scores for each model
                f1_model_1 = f1_score(true_labels_list, preds_model_1, zero_division=0)
                f1_model_2 = f1_score(true_labels_list, preds_model_2, zero_division=0)
                results[f"{self.labels[label_index]}"] = [
                    p_value,
                    test_statistic,
                    f1_model_1,
                    f1_model_2,
                ]

            return results

        languages = ["en", "fi", "fr", "sv", "tr"]
        bin_lab = lambda x: binarize_labels(x.split(), self.cfg.data.labels)

        get_labels = lambda csvfile: [
            [bin_lab(x[0]), bin_lab(x[1])]
            for x in list(csv.reader(csvfile, delimiter="\t"))
        ]

        for lang in languages:
            print()
            print(f"=== {lang} ===")
            print()
            with open(
                f"output/xlm-roberta-large/labels_all/preds/{lang}_mono.csv", "r"
            ) as csvfile:
                mono = get_labels(csvfile)

            with open(
                f"output/xlm-roberta-large/labels_all/preds/{lang}_multi.csv", "r"
            ) as csvfile:
                multi = get_labels(csvfile)

            with open(
                f"output/xlm-roberta-large/labels_all/preds/{lang}_zero.csv", "r"
            ) as csvfile:
                zero = get_labels(csvfile)

            adjusted_results = process_test_and_compare(mono, multi)

            for key, value in adjusted_results.items():
                print(
                    f"  {key}: Mono: {value[2]:.2f}, Multi: {value[3]:.2f}, P-value: {value[0]:.4f}"
                )

            # for mo, mu in zip(mono, multi):

            """
            # Perform McNemar's test for each label
            results = {}
            for label_index in range(true_labels.shape[1]):
                test_statistic, p_value = mcnemar_test(
                    true_labels[:, label_index],
                    predictions_model_1[:, label_index],
                    predictions_model_2[:, label_index],
                )
                results[f"Label {label_index}"] = {
                    "Test Statistic": test_statistic,
                    "P-value": p_value,
                }
            """
