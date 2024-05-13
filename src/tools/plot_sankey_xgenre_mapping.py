import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from datasets import concatenate_datasets
from transformers import AutoTokenizer
from ..labels import (
    map_full_names,
    map_xgenre,
    map_childless_upper_to_other,
)
from ..data import get_dataset

palette = sns.color_palette("Blues", n_colors=12).as_hex()[1:]
palette2 = sns.color_palette("BuGn", n_colors=12).as_hex()[3:]
template = "plotly_white"

import plotly.io as pio

pio.kaleido.scope.mathjax = None  # a fix for .pdf files

# Label hierarchy with the "other" categories and self-references
# This is needed to map the labels to the correct X-GENRE category
labels_all_hierarchy_with_other = {
    "MT": ["MT"],
    "LY": ["LY"],
    "SP": ["SP", "it", "os"],
    "ID": ["ID"],
    "NA": ["NA", "ne", "sr", "nb", "on"],
    "HI": ["HI", "re", "oh"],
    "IN": ["IN", "en", "ra", "dtp", "fi", "lt", "oi"],
    "OP": ["OP", "rv", "ob", "rs", "av", "oo"],
    "IP": ["IP", "ds", "ed", "oe"],
}


def get_ordered_data(df):
    # Explode labels
    df = df.explode("label_text")

    label_to_category = {
        label: category
        for category, labels in labels_all_hierarchy_with_other.items()
        for label in labels
    }

    # Calculate the label distributions
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
        for key in labels_all_hierarchy_with_other.keys()
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
    return data


def run(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    data = get_dataset(cfg, tokenizer)
    data = concatenate_datasets([data["train"], data["dev"], data["test"]])
    df = pd.DataFrame(data)[["label_text", "language"]]
    # Convert childless upper categories to "other"
    df["label_text"] = df["label_text"].apply(
        lambda labels: map_childless_upper_to_other(labels.split())
    )
    data = get_ordered_data(df)

    # Initialize lists for source, target, value, labels, and positions
    source = []
    target = []
    value = []
    label = []
    x_pos = []
    y_pos = []

    # Define a color scheme for main categories
    main_category_colors = {
        main_source: palette[i] for i, main_source in enumerate(data.keys())
    }

    rightmost_node_colors = {}
    color_counter = 0
    for main_source, sub_data in data.items():
        for sub_source, sub_details in sub_data["subcategories"].items():
            xgenre = sub_details["xgenre"]
            if xgenre not in rightmost_node_colors:
                rightmost_node_colors[xgenre] = palette[color_counter]
                color_counter += 1

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
                link=dict(source=source, target=target, value=value, color=link_colors),
            )
        ]
    )
    width = 1000
    height = width / 1.618

    # Set the layout
    fig.update_layout(
        template=template,
        width=width,
        height=height,
        margin=go.layout.Margin(l=5, r=5, b=15, t=5),
    )

    # Show the figure
    fig.show()
    fig.write_image("output/sankey.pdf")
