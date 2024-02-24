from dataclasses import dataclass

from jsonargparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", "-m")
    parser.add_argument("--source", "-s")
    parser.add_argument("--target", "-t")

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.augment import Augment

    Augment(cfg)
