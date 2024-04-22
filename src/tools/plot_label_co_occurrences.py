import pandas as pd
from itertools import combinations
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import seaborn as sns
import numpy as np
from datasets import concatenate_datasets
import matplotlib.colors as mcolors
from transformers import AutoTokenizer
from ..data import get_dataset, language_names, language_colors
from ..labels import (
    labels_structure,
    map_full_names,
    map_childless_upper_to_other,
    other_labels,
)

pio.kaleido.scope.mathjax = None  # a fix for .pdf files
palette = sns.color_palette("Blues")
palette[0] = (1, 1, 1)
template = "plotly_white"


def run(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    data = get_dataset(cfg, tokenizer)
    data = concatenate_datasets([data["train"], data["dev"], data["test"]])
    df = pd.DataFrame(data)[["label_text", "language"]]

    # Convert childless upper categories to "other"
    df["label_text"] = df["label_text"].apply(
        lambda labels: map_childless_upper_to_other(labels.split())
    )

    print(df.head(100))

    total_occurrences = df["label_text"].explode().value_counts()

    label_data = list(
        df["label_text"].apply(lambda x: [list(x) for x in list(combinations(x, 2))])
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
                # (co_occurrence_matrix.loc[label1, label2] / total) if total > 0 else 0
                co_occurrence_matrix.loc[label1, label2]
            )

    # Order labels based on their total occurrences
    ordered_labels = total_occurrences.sort_values(ascending=False).index.tolist()
    print(total_occurrences.sort_values(ascending=False))
    co_occurrence_matrix = co_occurrence_matrix.loc[ordered_labels, ordered_labels]

    # Mask the upper left triangle of the DataFrame for the triangular heatmap
    mask = np.triu(np.ones_like(co_occurrence_matrix, dtype=bool))
    triangular_df = co_occurrence_matrix.mask(mask)

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
