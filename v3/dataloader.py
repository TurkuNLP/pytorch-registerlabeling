import torch
from torch.utils.data import DataLoader
from .sampler import BalancedLanguageSampler

SAMPLER_CNF = {
    "train": {
        "size": "smallest",
        "lang_cycle": "random",
    },
    "test": {
        "size": "mean",
        "lang_cycle": "cycle",
    },
    "dev": {
        "size": "mean",
        "lang_cycle": "cycle",
    },
}


def init_split_dataloader(
    dataset, split, batch_size, tokenizer_pad_token_id, balance_languages
):
    def collate_fn(batch):
        max_length = max(len(example["input_ids"]) for example in batch)
        # Pad sequences dynamically to the maximum length in the batch
        for example in batch:
            pad_length = max_length - len(example["input_ids"])
            for key in example:
                if key == "input_ids":
                    # Use tokenizer.pad_token_id as the padding value for input_ids
                    example[key] = torch.nn.functional.pad(
                        example[key], (0, pad_length), value=tokenizer_pad_token_id
                    )
                elif key == "attention_mask":
                    # Use 0 as the padding value for attention_mask
                    example[key] = torch.nn.functional.pad(
                        example[key], (0, pad_length), value=0
                    )

        return {
            key: torch.stack([example[key] for example in batch]) for key in batch[0]
        }

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=BalancedLanguageSampler(SAMPLER_CNF[split])
        if balance_languages
        else None,
    )
    print(f"{split} dataloader size: {len(dataloader)}")

    return dataloader


def init_dataloaders(dataset, cfg, tokenizer_pad_token_id):
    return {
        split: init_split_dataloader(
            ds,
            split,
            cfg[f"{split}_batch_size"],
            tokenizer_pad_token_id,
            cfg.balance_languages,
        )
        for split, ds in dataset.items()
    }
