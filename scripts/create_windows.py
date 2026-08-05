from __future__ import annotations
import argparse
import json

from bayesian_retail.config import load_config
from bayesian_retail.features.bundle import FeatureBundle
from bayesian_retail.features.windows import create_all_window_bundles
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
    directory = paths.processed_root / args.dataset
    feature_bundle = FeatureBundle.load(directory)

    bundles = create_all_window_bundles(
        feature_bundle,
        directory=directory / "windows",
        history_length=config.data.history_length[args.dataset],
        forecast_horizon=config.data.forecast_horizon,
        stride=config.features.window_stride,
    )
    print(json.dumps(
        {key: value.metadata for key, value in bundles.items()},
        indent=2,
    ))


if __name__ == "__main__":
    main()
