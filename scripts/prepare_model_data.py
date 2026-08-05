from __future__ import annotations
import argparse
import subprocess
import sys

from scripts._common import add_config_arguments


def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument(
        "--dataset",
        choices=["m5", "favorita", "synthetic"],
        required=True,
    )
    args = parser.parse_args()

    common = ["--config", args.config]
    for override in args.set:
        common.extend(["--set", override])

    commands = [
        [sys.executable, "-m", "scripts.preprocess", "--dataset", args.dataset, *common],
        [sys.executable, "-m", "scripts.build_features", "--dataset", args.dataset, *common],
        [sys.executable, "-m", "scripts.create_windows", "--dataset", args.dataset, *common],
        [sys.executable, "-m", "scripts.build_tensors", "--dataset", args.dataset, *common],
    ]

    for command in commands:
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
