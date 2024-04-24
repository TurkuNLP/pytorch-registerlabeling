import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from datasets import concatenate_datasets
from transformers import AutoTokenizer
from ..data import get_dataset, language_names, language_colors
from ..labels import (
    labels_structure,
    map_full_names,
    map_childless_upper_to_other,
    other_labels,
)

pio.kaleido.scope.mathjax = None  # a fix for .pdf files


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

    # Explode the labels
    df = df.explode("label_text")

    # Calculate the label distributions
    subcategory_counts = (
        df.groupby(["label_text", "language"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )

    category_counts = {
        key: {"total": 0, "subcategories": {}} for key in labels_structure.keys()
    }

    print(category_counts)

    # Adding a "total" key to each dict item
    for label, lang_counts in subcategory_counts.items():
        subcategory_counts[label]["total"] = sum(lang_counts.values())

    label_to_parent = {
        label: category
        for category, labels in labels_structure.items()
        for label in labels
    }

    inverted_other_labels = {v: k for k, v in other_labels.items()}

    for key in subcategory_counts.keys():
        total = subcategory_counts[key].pop("total")
        category = label_to_parent.get(key, key)
        cat_key = (
            category if category in category_counts else inverted_other_labels[category]
        )
        category_counts[cat_key]["subcategories"][key] = {
            "val": total,
            "lang": subcategory_counts[key],
        }
        category_counts[cat_key]["total"] += total

    # Sort categories
    category_counts_sorted = dict(
        sorted(category_counts.items(), key=lambda x: x[1]["total"], reverse=True)
    )

    # Sort labels
    for key in category_counts_sorted.keys():
        sorted_subcategories = dict(
            sorted(
                category_counts[key]["subcategories"].items(),
                key=lambda x: x[1]["val"],
                reverse=True,
            )
        )
        category_counts[key]["subcategories"] = sorted_subcategories

    data = category_counts_sorted

    subcategories = []
    langs = {lang: [] for lang in cfg.train.split("-")}

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
                "line": {
                    "color": "#aaaaaa",
                    "width": 0.5,
                },
            }
        )

    # Creating the plot
    fig = go.Figure()
    print(subcategories)
    for lang, counts in langs.items():
        fig.add_trace(
            go.Bar(
                name=language_names.get(lang, lang),
                y=subcategories,
                x=counts,
                orientation="h",
                marker_color=language_colors[lang],
            )
        )

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
        template="plotly_white",
        barmode="stack",
        shapes=shapes[:-1],
        legend_traceorder="normal",
    )
    fig.update_yaxes(ticksuffix="  ")
    fig.write_image("output/label_distributions.pdf")
