from __future__ import annotations

import argparse
import json

from bayesian_retail.config import load_config
from bayesian_retail.data.pipeline import preprocess_dataset
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

    config = load_config(args.config, args.set)
    bundle = preprocess_dataset(config, args.dataset)
    print(json.dumps(bundle.metadata, indent=2))


if __name__ == "__main__":
    main()
