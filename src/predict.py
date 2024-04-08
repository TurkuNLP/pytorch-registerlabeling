import json
import os

import requests
import torch
import zstandard as zstd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import sigmoid
from .labels import (
    decode_binary_labels,
    label_schemes,
    subcategory_to_parent_index,
    map_to_xgenre_binary,
    upper_all_indexes,
)


def stream_data(file_url):

    with requests.get(file_url, stream=True) as r:
        dctx = zstd.ZstdDecompressor()

        # Stream and decompress the data
        with dctx.stream_reader(r.raw) as reader:
            buffer = ""  # Initialize a buffer for partial lines
            for chunk in iter(lambda: reader.read(16384), b""):
                # Decode chunk and append to buffer, handle partial multi-byte characters
                buffer += chunk.decode("utf-8", errors="ignore")
                lines = buffer.split("\n")

                # Keep the last, potentially incomplete line in the buffer
                buffer = lines[-1]

                # Process complete lines
                for line in lines[:-1]:  # Exclude the last line because it's incomplete
                    if line:  # Ensure line is not empty
                        yield json.loads(line)


def local_data(data):
    for file_name in os.listdir(data):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                yield file.readlines()


def process_batch(batch_texts, batch_docs, cfg, model, tokenizer, device):
    # Tokenize the batch
    inputs = tokenizer(
        batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = sigmoid(outputs.logits)

    if cfg.train_labels == "all":
        # Ensure that subcategory has corresponding parent category
        for i in range(predictions.shape[0]):
            for (
                subcategory_index,
                parent_index,
            ) in subcategory_to_parent_index.items():
                if predictions[i, parent_index] < predictions[i, subcategory_index]:
                    predictions[i, parent_index] = predictions[i, subcategory_index]

    if cfg.train_labels == "all" and cfg.predict_labels == "upper":
        predictions = predictions[:, upper_all_indexes]

    for i, doc in enumerate(batch_docs):
        binary_predictions = predictions[i] > cfg.threshold
        predicted_labels = decode_binary_labels(
            [binary_predictions.tolist()], cfg.predict_labels
        )
        doc["register_labels"] = predicted_labels[0]

        # Write each document to file
        with open(cfg.output_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(doc, ensure_ascii=False) + "\n")


def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init model
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_path).to(
        device
    )
    model.eval()

    # Get the original model's name and init tokenizer
    with open(f"{cfg.model_path}/config.json", "r") as config_file:
        config = json.load(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config.get("_name_or_path"))

    data_iterator = (
        stream_data(cfg.stream_data) if cfg.stream_data else local_data(cfg.local_data)
    )

    batch_texts = []
    batch_docs = []
    processed_batches = 0

    for doc in data_iterator:
        text = doc["text"]
        batch_texts.append(text)
        batch_docs.append(doc)

        # When the batch is full, process it
        if len(batch_texts) == cfg.batch_size:
            process_batch(batch_texts, batch_docs, cfg, model, tokenizer, device)
            batch_texts, batch_docs = [], []  # Reset for the next batch
            processed_batches += 1
            if cfg.n_batches and processed_batches >= cfg.n_batches:
                print("Done.")
                exit()

    # Don't forget to process the last batch if it's not empty
    if batch_texts:
        process_batch(batch_texts, batch_docs, cfg, model, tokenizer, device)
