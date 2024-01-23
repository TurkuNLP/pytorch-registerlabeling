from jsonargparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--source_path", "-p")
    parser.add_argument("--source_file", "-f")
    parser.add_argument("--method", "-m")
    parser.add_argument("--labels", "-l", default="all")

    cfg = parser.parse_args()

    print(parser.dump(cfg))

    from v3.stats import Stats

    Stats(cfg)
