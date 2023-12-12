import numpy as np
from torch.utils.data import Sampler
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader


class BalancedMixedLanguageSampler(Sampler):
    def __init__(self, language_data):
        self.language_data = language_data
        self.indices_per_language = self._create_indices_per_language()
        self.num_languages = len(self.indices_per_language)
        self.smallest_dataset_size = min(
            len(indices) for indices in self.indices_per_language.values()
        )
        # Define the epoch size as the size of the smallest dataset times the number of languages
        self.epoch_size = int(self.num_languages * self.smallest_dataset_size / 100)

        print(f"Sampler epoch size: {self.epoch_size}")

    def _create_indices_per_language(self):
        indices_per_language = {}
        for idx, language in enumerate(self.language_data):
            if language not in indices_per_language:
                indices_per_language[language] = []
            indices_per_language[language].append(idx)
        return indices_per_language

    def __len__(self):
        # The total number of samples per epoch is the size of the smallest dataset times the number of languages
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.epoch_size):
            # Randomly select a language
            language = np.random.choice(list(self.indices_per_language.keys()))

            # Replenish the indices for the language if necessary
            if not self.indices_per_language[language]:
                self.indices_per_language[language] = [
                    idx
                    for idx, lang in enumerate(self.language_data)
                    if lang == language
                ]

            # Randomly select one index from the language's indices
            idx = np.random.choice(self.indices_per_language[language], replace=False)
            yield idx

            # Remove the selected index
            self.indices_per_language[language].remove(idx)


class BalancedFocusedLanguageSampler(Sampler):
    def __init__(self, language_data):
        self.language_data = language_data
        self.indices_per_language = self._create_indices_per_language()
        self.num_languages = len(self.indices_per_language)
        self.smallest_dataset_size = min(
            len(indices) for indices in self.indices_per_language.values()
        )
        # Define the epoch size as the size of the smallest dataset times the number of languages
        self.epoch_size = self.num_languages * self.smallest_dataset_size

    def _create_indices_per_language(self):
        indices_per_language = {}
        for idx, language in enumerate(self.language_data):
            if language not in indices_per_language:
                indices_per_language[language] = []
            indices_per_language[language].append(idx)
        return indices_per_language

    def __len__(self):
        # The total number of samples per epoch is the size of the smallest dataset times the number of languages
        return self.epoch_size

    def __iter__(self):
        for _ in range(self.smallest_dataset_size):
            for language in self.indices_per_language.keys():
                # Replenish the indices for the language if necessary
                if not self.indices_per_language[language]:
                    self.indices_per_language[language] = [
                        idx
                        for idx, lang in enumerate(self.language_data)
                        if lang == language
                    ]

                # Randomly select one index from the language's indices
                idx = np.random.choice(
                    self.indices_per_language[language], replace=False
                )
                yield idx

                # Remove the selected index
                self.indices_per_language[language].remove(idx)


def custom_train_dataloader(self, custom_sampler) -> DataLoader:
    language_data = [sample["language"] for sample in self.train_dataset]
    train_dataset = self._remove_unused_columns(
        self.train_dataset, description="training"
    )
    if custom_sampler == "mixed":
        sampler = BalancedMixedLanguageSampler
    elif custom_sampler == "focused":
        sampler = BalancedFocusedLanguageSampler
    batch_size = self._train_batch_size
    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": self.data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "sampler": sampler(language_data),
        "drop_last": self.args.dataloader_drop_last,
        "worker_init_fn": seed_worker,
    }

    return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))