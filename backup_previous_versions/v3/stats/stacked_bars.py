from ..labels import labels_all_hierarchy


def get_ordered_data(dataset, reverse=True):
    # Combine splits and extract label, language and text

    df = get_all_data(dataset)

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
    # Preparing data for the stacked bar plot
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
                marker_color=sns.cubehelix_palette(rot=-0.2, n_colors=5).as_hex()[
                    color_i
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
        template=template,
        barmode="stack",
        shapes=shapes[:-1],  # Adding shapes to the layout
        legend_traceorder="normal",
    )
    fig.update_yaxes(ticksuffix="  ")

    fig.show()
    fig.write_image("output/stacked.pdf")
