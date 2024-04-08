from dataclasses import dataclass

from jsonargparse import ArgumentParser


@dataclass
class Data:
    train: str = None
    dev: str = None
    test: str = None
    labels: str = "all"
    concat_small: bool = False
    text_prefix: str = ""
    test_all_data: bool = False
    scores: str = ""
    use_fold: bool = False
    use_augmented_data: bool = False


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input", "-i")
    parser.add_argument("--method", "-m")
    parser.add_argument("--working_dir_root", default=".")
    parser.add_argument("--data", type=Data, default=Data())

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.stats import Stats

    Stats(cfg)
