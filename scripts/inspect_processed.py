from __future__ import annotations

import argparse
import json

from bayesian_retail.config import load_config
from bayesian_retail.paths import ProjectPaths
from bayesian_retail.data.bundle import ProcessedDataset
from scripts._common import add_config_arguments


def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument(
        "--dataset",
        choices=["m5", "favorita", "synthetic"],
        required=True,
    )
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ProjectPaths.from_config(config)
    bundle = ProcessedDataset.load(
        paths.processed_root / args.dataset
    )

    print(json.dumps(bundle.metadata, indent=2))
    print("\nSplit summary:")
    print(bundle.split_summary.to_string(index=False))
    print("\nHierarchy:")
    print(bundle.hierarchy.head(args.head).to_string(index=False))
    print("\nObservations:")
    print(bundle.observations.head(args.head).to_string(index=False))


if __name__ == "__main__":
    main()
