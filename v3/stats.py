import csv
from itertools import combinations

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from datasets import concatenate_datasets
from sklearn.metrics import classification_report

from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

from .data import get_dataset, language_names, small_languages
from .labels import (
    binarize_labels,
    get_label_scheme,
    labels_all_hierarchy,
    map_full_names,
    map_xgenre,
)

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

    def get_dataframe(self, cfg):
        data = get_dataset(self.cfg)
        data = concatenate_datasets([data["train"], data["dev"], data["test"]])
        df = pd.DataFrame(data)[["label_text", "language"]]
        df["label_text"] = df["label_text"].str.split(" ")

        redundant_parents = [
            x for x in labels_all_hierarchy.keys() if len(labels_all_hierarchy[x]) > 1
        ]

        # Filtering out redundant parents
        df["label_text"] = df["label_text"].apply(
            lambda labels: [label for label in labels if label not in redundant_parents]
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
        for key, val in category_counts_sorted.items():
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
        fig.write_image(f"output/{data['output']}", scale=data.get("scale", 10))

    def stacked_bars(self):
        df = self.get_dataframe(self.cfg)

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
        confusion_matrix_fig.write_image(f"output/heatmap_{name}.png", scale=10)

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

        palette = sns.cubehelix_palette(rot=-0.2, n_colors=20).as_hex()[1:]
        palette2 = sns.cubehelix_palette(rot=-0.3, n_colors=20).as_hex()[1:]

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
                link_colors.append(rightmost_node_colors[xgenre])
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

    def compare_models(self):

        def benjamini_hochberg_correction(p_values, Q=0.05):
            """
            Applies the Benjamini-Hochberg correction for controlling the FDR.

            Parameters:
            - p_values: A list or array of p-values from multiple hypothesis tests.
            - Q: The desired False Discovery Rate level.

            Returns:
            - A boolean array indicating which tests are significant after correction.
            """
            m = len(p_values)  # Total number of tests
            sorted_p_values = np.array(p_values)
            sorted_indices = np.argsort(sorted_p_values)
            sorted_p_values = sorted_p_values[sorted_indices]

            # Calculate BH critical values
            bh_critical_values = (np.arange(1, m + 1) / m) * Q

            # Find the largest p-value where p < BH critical value
            significant_mask = sorted_p_values < bh_critical_values
            max_significant = np.max(np.where(significant_mask, sorted_indices, 0))

            # Mark p-values as significant accordingly
            is_significant = np.arange(m) <= max_significant

            # Return to original ordering
            return is_significant[np.argsort(sorted_indices)]

        def holm_bonferroni_correction(p_values):
            """
            Applies the Holm-Bonferroni correction to a list of p-values.

            Parameters:
            - p_values: A list or array of p-values from multiple hypothesis tests.

            Returns:
            - A boolean array indicating which tests are significant after correction.
            """
            m = len(p_values)  # Total number of tests
            sorted_indices = np.argsort(p_values)  # Sort indices
            sorted_p_values = np.array(p_values)[sorted_indices]  # Sorted p-values

            # Calculate Holm-Bonferroni adjusted p-values
            adjusted_p_values = (m - np.arange(m)) * sorted_p_values

            # Ensure monotonicity of the adjusted p-values
            for i in range(m - 1):
                adjusted_p_values[i + 1] = max(
                    adjusted_p_values[i], adjusted_p_values[i + 1]
                )

            # Reorder to the original order of tests and check against alpha
            reordered_adjusted_p_values = adjusted_p_values[np.argsort(sorted_indices)]
            significant = reordered_adjusted_p_values < 0.05

            return significant

        # F1 scores for each model across three experiments
        # data = np.array(
        #    [
        #        [0.48931116389548696, 0.5089058524173029, 0.5272727272727273],  # xlmr-l
        #        [0.5409429280397021, 0.5206349206349207, 0.56047197640118],  # xlmr-xl
        #        [0.56, 0.5513196480938416, 0.5382830626450116],  # bge-m3-retromae
        #    ]
        # )
        models = ["xlmr-large", "xlmr-xl", "bge-m3-retromae-2048"]
        scores = []
        import re, ast

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

                            f1 = result_dict["f1"]
                            pr_auc = result_dict["pr_auc"]

                            scores[-1].append(f1)

        data = np.array(scores)

        # Calculate the number of models
        n_models = data.shape[0]

        print(n_models)

        # Calculate the number of pairwise comparisons
        n_comparisons = (n_models * (n_models - 1)) / 2

        print(n_comparisons)

        # Adjusted alpha level for Bonferroni correction
        alpha_corrected = 0.05 / n_comparisons

        # Perform all pairwise Wilcoxon signed-rank tests
        p_values = []
        comparisons = []
        for i, j in combinations(range(n_models), 2):
            stat, p = wilcoxon(data[i], data[j])
            comparisons.append((i, j))
            p_values.append(p)

        # Apply Bonferroni correction and determine significance
        significant_differences = np.array(p_values) < alpha_corrected

        # Print results
        print("Bonferroni correction results ")
        for (i, j), p, significant in zip(
            comparisons, p_values, significant_differences
        ):
            print(
                f"Model {i+1} vs Model {j+1}: p-value = {p}, significant: {significant}"
            )

        ben = benjamini_hochberg_correction(p_values)

        print("Benjamini hochberg correction results: ")
        print(ben)

        # significant = holm_bonferroni_correction(p_values)
        # print("Holm-Bonferroni correction results:", significant)

        exit()

        scores = [float(x) for x in self.cfg.data.scores.split(",")]

        print(scores)

        # Perform the Wilcoxon signed-rank test
        stat, p = wilcoxon(scores)
        print(f"Statistics={stat}, p={p}")

        p_vals = [0.01, 0.04, 0.03, 0.05, 0.001]
        alpha = 0.05

        # Apply Bonferroni correction
        reject, pvals_corrected, _, _ = multipletests(
            p_vals, alpha=alpha, method="bonferroni"
        )

        print(f"Adjusted p-values: {pvals_corrected}")
        print(f"Reject null hypothesis: {reject}")
