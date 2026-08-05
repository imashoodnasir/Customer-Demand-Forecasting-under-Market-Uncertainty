from __future__ import annotations

import argparse

from bayesian_retail.config import load_config
from bayesian_retail.paths import ProjectPaths
from bayesian_retail.data.synthetic import save_synthetic_raw
from scripts._common import add_config_arguments


def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument("--series", type=int, default=24)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ProjectPaths.from_config(config)
    paths.create()
    path = save_synthetic_raw(
        paths.raw_root / "synthetic",
        n_series=args.series,
        n_days=args.days,
        seed=args.seed,
    )
    print(path)


if __name__ == "__main__":
    main()
