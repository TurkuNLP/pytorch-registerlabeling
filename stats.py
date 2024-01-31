from jsonargparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input", "-i")
    parser.add_argument("--method", "-m")
    parser.add_argument("--label_scheme", "-l", default="all")

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.stats import Stats

    Stats(cfg)
