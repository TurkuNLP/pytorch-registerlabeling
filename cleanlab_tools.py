from pydoc import locate

from jsonargparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", "-t")

    cfg = parser.parse_args()

    print(parser.dump(cfg))
    locate(f"src.cleanlab_tools.{cfg.task}")
