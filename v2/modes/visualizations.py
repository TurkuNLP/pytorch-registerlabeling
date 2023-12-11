from pprint import pprint

import pandas as pd

import plotly.io as pio

pio.kaleido.scope.mathjax = None  # a fix for .pdf files
import plotly.graph_objects as go
import plotly.express as px

from ..labels import map_xgenre, map_full_names, labels_all_hierarchy

template = "plotly_white"


def get_ordered_data(dataset, reverse=True):
    # Combine splits and extract label, language and text

    df = pd.concat(
        [pd.DataFrame(dataset[split]) for split in dataset], ignore_index=True
    )[["label_text", "language", "text"]]

    df["label_text"] = df["label_text"].str.split(" ")
    df = df.explode("label_text")

    # Parent and label mappings

    label_to_category = {
        label: category
        for category, labels in labels_all_hierarchy.items()
        for label in labels
    }

    redundant_parents = [
        x for x in labels_all_hierarchy.keys() if len(labels_all_hierarchy[x]) > 1
    ]

    # Filter out parent categories
    df = df[~df["label_text"].isin(redundant_parents)]

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
        key: {"total": 0, "subcategories": {}} for key in labels_all_hierarchy.keys()
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
        sorted(category_counts.items(), key=lambda x: x[1]["total"], reverse=reverse)
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


def stacked_bars(dataset):
    data = get_ordered_data(dataset)
    pprint(data)
    # Preparing data for the stacked bar plot
    subcategories = []
    langs = {"en": [], "fi": [], "fr": [], "sv": []}
    lang_full_names = {
        "en": "English",
        "sv": "Swedish",
        "fi": "Finnish",
        "fr": "French",
    }
    shapes = []  # List to store shapes for horizontal lines

    # Counter to keep track of subcategory index
    subcat_index = 0

    for region in data.values():
        print("REEGION")
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
                    "color": "#555555",
                    "width": 1,
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
                marker_color=px.colors.qualitative.D3[color_i],
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
        template=template,
        barmode="stack",
        shapes=shapes[:-1],  # Adding shapes to the layout
        legend_traceorder="normal",
    )
    fig.update_yaxes(ticksuffix="  ")

    fig.show()
    fig.write_image("output/stacked.pdf")


def sankey_plot(dataset):
    data = get_ordered_data(dataset)
    # colors = px.colors.sequential.Agsunset

    # Initialize lists for source, target, value, labels, and positions
    source = []
    target = []
    value = []
    label = []
    x_pos = []
    y_pos = []

    palette = px.colors.sequential.Reds
    palette2 = px.colors.sequential.Blues

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
                    label=label,
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

    # Set the layout
    fig.update_layout(
        template=template,
        margin=go.layout.Margin(l=5, r=5, b=5, t=5, pad=4),
    )

    # Show the figure
    fig.show()
