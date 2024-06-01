import plotly.graph_objects as go
import plotly.io as pio
from datasets import concatenate_datasets
from transformers import AutoTokenizer

from ..data import get_dataset, language_names

template = "plotly_white"
pio.kaleido.scope.mathjax = None  # a fix for .pdf files


def run(cfg):

    language_names["other"] = "All langs (N=16)"
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
    dataset = get_dataset(cfg, tokenizer)

    dataset = dataset.map(
        lambda example: tokenizer(example["text"]),
        batched=True,
    )

    dataset = concatenate_datasets([dataset["train"], dataset["dev"], dataset["test"]])

    data = [(len(x["input_ids"]), x["language"]) for x in dataset]

    over_8k = sum([1 if x[0] > 8192 else 0 for x in data])
    print(over_8k)

    print(f"Excluded {over_8k/len(data)} texts.")

    data = [x for x in data if x[0] <= 8192]

    # Organize data by language
    text_lengths_by_language = {
        "en": [],
        "fi": [],
        "fr": [],
        "sv": [],
        "tr": [],
        "other": [],
    }
    for length, language in data:
        key = language if language in ["en", "fi", "fr", "sv", "tr"] else "other"
        text_lengths_by_language[key].append(length)

    # Create a box plot for each language
    fig = go.Figure()
    lang_i = 0
    for lang, lengths in text_lengths_by_language.items():
        full_name = language_names[lang]  # Convert code to full name
        fig.add_trace(
            go.Box(
                y=lengths,
                name=full_name,
            )
        )
        lang_i += 1

    fig.update_layout(
        title=None,
        showlegend=False,
        template=template,
        margin=go.layout.Margin(l=5, r=5, b=5, t=5),
    )

    # fig.write_image(f"output/text_lengths_all_concat.pdf")
    fig.show()
