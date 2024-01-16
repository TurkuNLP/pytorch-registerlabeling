from jsonargparse import ArgumentParser


def main_cli():
    parser = ArgumentParser()
    parser.add_argument("--opt2", type=float, default=1.0, help="Help for option 2.")
    cfg = parser.parse_args()

    return cfg
