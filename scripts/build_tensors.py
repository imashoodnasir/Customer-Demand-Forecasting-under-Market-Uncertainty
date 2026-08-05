from __future__ import annotations
import argparse
import json

from bayesian_retail.config import load_config
from bayesian_retail.features.tensors import build_all_tensor_bundles
from bayesian_retail.paths import ProjectPaths
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
    paths = ProjectPaths.from_config(config)
    base = paths.processed_root / args.dataset

    bundles = build_all_tensor_bundles(
        windows_directory=base / "windows",
        tensors_directory=base / "tensors",
    )
    print(json.dumps(
        {key: value.metadata for key, value in bundles.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
