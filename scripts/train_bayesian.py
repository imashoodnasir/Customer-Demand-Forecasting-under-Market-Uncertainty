from __future__ import annotations

import argparse

from bayesian_retail.config import load_config
from bayesian_retail.experiments.bayesian import (
    train_bayesian_experiment,
)
from scripts._common import add_config_arguments


def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument(
        "--dataset",
        choices=["m5", "favorita", "synthetic"],
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--transfer-prior", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    context = train_bayesian_experiment(
        config=config,
        dataset=args.dataset,
        split=args.split,
        seed=args.seed,
        transfer_prior_path=args.transfer_prior,
    )
    print(context.run_dir)


if __name__ == "__main__":
    main()
