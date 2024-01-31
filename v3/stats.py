import csv

import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import classification_report

from .data import language_names, small_languages
from .labels import binarize_labels, get_label_scheme, map_full_names


class Stats:
    def __init__(self, cfg):
        self.cfg = cfg
        self.labels = get_label_scheme(cfg.label_scheme)
        self.palette = sns.cubehelix_palette

        # Run
        getattr(self, cfg.method)()

    def prediction_confusion_matrix(self):
        with open(f"{self.cfg.input}", "r") as csvfile:
            csv_reader = csv.reader(csvfile, delimiter="\t")
            data = list(csv_reader)

        # Extract true and predicted labels from the data
        true_labels, predicted_labels = zip(*data)

        true_labels_binary = [
            binarize_labels(label.split(), self.cfg.label_scheme)
            for label in true_labels
        ]
        predicted_labels_binary = [
            binarize_labels(label.split(), self.cfg.label_scheme)
            for label in predicted_labels
        ]

        confusion_matrix_data = np.array(
            [
                [
                    sum(a and b for a, b in zip(true_col, pred_col))
                    for pred_col in zip(*predicted_labels_binary)
                ]
                for true_col in zip(*true_labels_binary)
            ]
        )

        print(confusion_matrix_data)

        normalized_confusion_matrix_data = (
            confusion_matrix_data / confusion_matrix_data.sum(axis=1, keepdims=True)
        )

        cubehelix_palette = self.palette(rot=-0.2)
        palette = [mcolors.to_hex(color) for color in cubehelix_palette]

        # Create confusion matrix using Plotly Express
        confusion_matrix_fig = px.imshow(
            normalized_confusion_matrix_data,
            # labels=dict(x="Predicted Labels", y="True Labels"),
            x=self.labels,
            y=self.labels,
            color_continuous_scale=palette,
            title="Confusion Matrix (Percentages)",
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
        )
        # confusion_matrix_fig.show()
        confusion_matrix_fig.write_image("output/heatmap.png", scale=10)

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
                    f"{map_full_names[k]} ({k})": v["f1-score"]
                    if v["f1-score"] != 0
                    else 0.005
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
