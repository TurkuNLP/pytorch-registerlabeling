from itertools import combinations

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

import seaborn as sns
from datasets import concatenate_datasets
from transformers import AutoTokenizer

from ..data import get_dataset
from ..labels import map_childless_upper_to_other, map_full_names, labels_structure

palette = sns.color_palette("Blues")
palette[0] = (1, 1, 1)
template = "plotly_white"

# Identifying keys with children
keys_with_children = [key for key, value in labels_structure.items() if value]

# Filtering map_full_names to remove keys that have children
map_full_names = {
    key: value for key, value in map_full_names.items() if key not in keys_with_children
}


def run(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    data = get_dataset(cfg, tokenizer)
    data = concatenate_datasets([data["train"], data["dev"], data["test"]])
    df = pd.DataFrame(data)[["label_text", "language"]]

    # Convert childless upper categories to "other"
    df["label_text"] = df["label_text"].apply(
        lambda labels: map_childless_upper_to_other(labels.split())
    )

    # Initialize and populate matrix
    keys = map_full_names.keys()
    matrix = pd.DataFrame(0, index=keys, columns=keys)

    # Count connections
    for labels in df["label_text"]:
        if len(labels) == 1:
            # Count solo occurrences on the diagonal
            matrix.loc[labels[0], labels[0]] += 1
        else:
            # Count pairs for co-appearances
            for src, tgt in combinations(labels, 2):
                matrix.loc[src, tgt] += 1 / len(labels)
                matrix.loc[tgt, src] += 1 / len(labels)

    row_totals = matrix.sum(axis=1)
    # matrix["Total"] = row_totals

    for key in keys:
        matrix.loc[key, matrix.columns != "Total"] = (
            matrix.loc[key, matrix.columns != "Total"] / row_totals[key]
        ) * 100

    matrix.index = [
        f"{map_full_names.get(key, key)} <b><span style='font-size:75%;display:inline-block; width:50px;'>({row_totals[key]:.0f})</span></b>"
        for key in matrix.index
    ]
    matrix.columns = [map_full_names.get(key, key) for key in matrix.columns]

    # Create the heatmap
    fig = px.imshow(
        matrix,
        labels=dict(x="Target", y="Source", color="Connection Count"),
        title="Heatmap of Connections",
        color_continuous_scale=[mcolors.to_hex(color) for color in palette],
        template=template,
        text_auto=".0f",
        aspect="auto",
        range_color=[0, 100],
    )

    fig.update_xaxes(side="top", title="")
    fig.update_yaxes(side="left", title="")
    fig.update_yaxes(ticksuffix=" ")
    fig.update_layout(
        margin=go.layout.Margin(l=5, r=5, b=5, t=5),
        showlegend=False,
        title="",
        coloraxis_showscale=False,
        font=dict(size=6),
        autosize=False,
        width=500,
        height=500,
        yaxis=dict(tickangle=0),
        xaxis=dict(tickangle=30),
    )

    # Adding annotations for 'Total' column
    """
    annotations = []
    for idx, key in enumerate(matrix.index):
        annotations.append(
            dict(
                x=len(keys),
                y=idx,
                text=str(matrix.loc[key, "Total"]),
                showarrow=False,
                font=dict(color="black", size=6),
            )
        )

    fig.update_layout(annotations=annotations)
    """

    fig.write_image(f"output/heatmap_co_occurrences_full.png", scale=4)
    exit()

    # Count connections
    count_connections = Counter(tuple(x) for x in connections)

    # Create source, target, and value lists for the Sankey plot
    source = []
    target = []
    value = []

    # Label list duplicated for both source (left) and target (right)
    labels = [map_full_names[k] for k in map_full_names.keys()] * 2

    # Mapping from keys to their position in the first half (sources) and second half (targets)
    half_len = len(map_full_names)
    label_map = {key: idx for idx, key in enumerate(map_full_names.keys())}

    for (src, tgt), val in count_connections.items():
        source.append(label_map[src])
        target.append(label_map[tgt] + half_len)
        value.append(val)

    # Create the Sankey diagram
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                ),
                link=dict(source=source, target=target, value=value),
            )
        ]
    )

    fig.update_layout(title_text="Sankey Diagram of Connections", font_size=10)
    fig.show()

    exit()
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
                # (co_occurrence_matrix.loc[label1, label2] / total * 100)
                # if total > 0
                # else 0
                co_occurrence_matrix.loc[label1, label2]
            )

    # Order labels based on their total occurrences
    ordered_labels = total_occurrences.sort_values(ascending=False).index.tolist()
    print(total_occurrences.sort_values(ascending=False))
    co_occurrence_matrix = co_occurrence_matrix.loc[ordered_labels, ordered_labels]

    # Calculate co-occurrence totals for margins
    co_occurrence_totals = co_occurrence_matrix.sum(axis=1)

    # Enhance label names with co-occurrence totals
    enhanced_labels = [
        f"{map_full_names.get(label, label)} ({co_occurrence_totals[label]:.0f})"
        for label in ordered_labels
    ]

    hex_palette = [mcolors.to_hex(color) for color in palette]

    # Mask the upper left triangle of the DataFrame for the triangular heatmap
    mask = np.triu(np.ones_like(co_occurrence_matrix, dtype=bool))
    triangular_df = co_occurrence_matrix.mask(mask)
    triangular_df = co_occurrence_matrix

    hex_palette = [mcolors.to_hex(color) for color in palette]

    fig = px.imshow(
        triangular_df,
        x=[map_full_names.get(x, x) for x in ordered_labels],
        y=[map_full_names.get(x, x) for x in ordered_labels],
        color_continuous_scale=hex_palette,
        template=template,
        text_auto=".0f",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_yaxes(ticksuffix=" ")
    fig.update_layout(
        margin=go.layout.Margin(l=5, r=5, b=5, t=5),
        showlegend=False,
        template=template,
        title=None,
        coloraxis_showscale=False,
        font=dict(size=6),
        autosize=False,
        width=500,
        height=500,
    )

    fig.write_image(f"output/heatmap_co_occurrences.png", scale=4)
