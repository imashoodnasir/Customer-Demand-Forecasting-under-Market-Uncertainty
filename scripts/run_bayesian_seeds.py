from __future__ import annotations

import argparse
import subprocess
import sys

from bayesian_retail.config import load_config
from scripts._common import add_config_arguments


def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    for seed in config.experiment.seeds:
        command = [
            sys.executable,
            "-m",
            "scripts.train_bayesian",
            "--dataset",
            args.dataset,
            "--split",
            args.split,
            "--seed",
            str(seed),
            "--config",
            args.config,
        ]
        for override in args.set:
            command.extend(["--set", override])
        print("Running:", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
