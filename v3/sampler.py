from itertools import cycle

import numpy as np
from torch.utils.data import Sampler


class BalancedLanguageSampler(Sampler):
    def __init__(self, language_data, size="smallest", lang_cycle="random"):
        self.language_data = language_data
        self.indices_per_language = self._create_indices_per_language()
        language_indice_lengths = [
            len(indices) for indices in self.indices_per_language.values()
        ]
        if size == "smallest":
            dataset_size = min(language_indice_lengths)
        elif size == "mean":
            dataset_size = int(np.mean(language_indice_lengths))

        self.epoch_size = dataset_size * len(self.indices_per_language)
        self.cycle = lang_cycle

        print(
            f"Sampler epoch size [criterion: {size}, cycle: {lang_cycle}]: {self.epoch_size}"
        )

    def _create_indices_per_language(self):
        indices_per_language = {lang: [] for lang in set(self.language_data)}
        for idx, lang in enumerate(self.language_data):
            indices_per_language[lang].append(idx)
        return indices_per_language

    def __len__(self):
        return self.epoch_size

    def __iter__(self):
        language_cycle = cycle(self.indices_per_language.keys())
        for _ in range(self.epoch_size):
            if self.cycle == "random":
                language = np.random.choice(list(self.indices_per_language.keys()))
            elif self.cycle == "cycle":
                language = next(language_cycle)

            # Replenish the indices for the language if necessary
            if not self.indices_per_language[language]:
                self.indices_per_language[language] = [
                    idx
                    for idx, lang in enumerate(self.language_data)
                    if lang == language
                ]

            idx = np.random.choice(self.indices_per_language[language], replace=False)
            self.indices_per_language[language].remove(idx)
            yield idx
