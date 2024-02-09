import torch
from torch.utils.data import DataLoader

from .sampler import BalancedLanguageSampler

SAMPLER_CNF = {
    "train": {
        "size": "smallest",
        "lang_cycle": "random",
    },
    "dev": {
        "size": "mean",
        "lang_cycle": "cycle",
    },
    "test": {
        "size": "mean",
        "lang_cycle": "cycle",
    },
}


def init_split_dataloader(
    dataset,
    split,
    batch_size,
    tokenizer_pad_token_id,
    balance_languages,
    device,
):
    def collate_fn(batch):
        if "input_ids" in batch[0]:
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
                    elif key in ["attention_mask", "token_type_ids"]:
                        # Use 0 as the padding value for attention_mask
                        example[key] = torch.nn.functional.pad(
                            example[key], (0, pad_length), value=0
                        )

        return {
            key: torch.stack([example[key] for example in batch]) for key in batch[0]
        }

    language_data = [sample["language"] for sample in dataset]
    dataset = dataset.remove_columns(["language"])
    use_balancer = balance_languages and len(set(language_data)) > 1
    print(f"Languages: {set(language_data)}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **(
            {"sampler": BalancedLanguageSampler(language_data, **SAMPLER_CNF[split])}
            if use_balancer
            else {"shuffle": True}
        ),
        #generator=torch.Generator(device=device),
    )
    print(f"{split} dataloader size: {len(dataloader)} (balancer: {use_balancer})")

    return dataloader


def init_dataloaders(dataset, cfg, tokenizer_pad_token_id, device):
    return {
        split: init_split_dataloader(
            ds,
            split,
            cfg[f"{split}_batch_size"],
            tokenizer_pad_token_id,
            cfg.balancing_sampler,
            device,
        )
        for split, ds in dataset.items()
    }
