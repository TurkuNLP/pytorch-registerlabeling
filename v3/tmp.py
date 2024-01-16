def run(cfg):
    # Set variables

    # Load data

    dataset = get_dataset(cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name, torch_dtype=cfg.torch_dtype
    )

    dataset = preprocess_data(dataset, tokenizer, cfg.seed, cfg.data.max_length)

    # Get model

    model = RegisterModel(cfg)
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

    from tqdm.auto import tqdm

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
