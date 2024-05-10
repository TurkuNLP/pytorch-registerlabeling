import csv

import matplotlib.colors as mcolors
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from ..labels import binarize_labels, label_schemes

labels = label_schemes["all"]

palette = sns.color_palette("Blues")
palette[0] = (1, 1, 1)


def run(cfg):

    with open(f"{cfg.source_data_path}", "r") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter="\t")
        data = list(csv_reader)

    # Extract true and predicted labels from the data
    true_labels, predicted_labels, _ = zip(*data)

    true_labels_binary = [
        binarize_labels(label.split(), "all") for label in true_labels
    ]
    predicted_labels_binary = [
        binarize_labels(label.split(), "all") for label in predicted_labels
    ]

    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=float)

    def get_combined(T, P):
        T_ = np.array(T) & (~np.array(P) & 1)  # FN
        P_ = np.array(P) & (~np.array(T) & 1)  # FP

        sumP = np.sum(P) or 1
        sumT = np.sum(T) or 1
        sumP_ = np.sum(P_) or 1

        if all(T == P):
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

    normalized_confusion_matrix_data = (normalized_confusion_matrix_data * 100).astype(
        int
    )

    confusion_matrix_fig = px.imshow(
        normalized_confusion_matrix_data,
        x=[f"<b>{x}</b>" for x in labels],
        y=[f"<b>{x}</b>" for x in labels],
        color_continuous_scale=[mcolors.to_hex(color) for color in palette],
        title="Confusion Matrix",
        color_continuous_midpoint=0,
        zmin=0,
        zmax=100,
        text_auto=".0f",
    )

    confusion_matrix_fig.update_yaxes(ticksuffix=" ")
    confusion_matrix_fig.update_layout(
        margin=go.layout.Margin(l=5, r=5, b=5, t=5),
        showlegend=False,
        title=None,
        coloraxis_showscale=False,
        font=dict(size=10),
        autosize=False,
        width=500,
        height=500,
    )
    confusion_matrix_fig.update_traces(textfont_size=9)
    name = cfg.source_data_path.split("/")[-1].split(".")[0]
    confusion_matrix_fig.write_image(f"output/heatmap_{name}.png", scale=4)
