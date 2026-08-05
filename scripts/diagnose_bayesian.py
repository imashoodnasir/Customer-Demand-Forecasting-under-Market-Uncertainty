from __future__ import annotations

import argparse
import json
from pathlib import Path

from bayesian_retail.config import load_config
from bayesian_retail.inference.diagnostics import (
    save_diagnostic_figures,
    summarize_inference,
)
from bayesian_retail.inference.engine import load_inference_data
from bayesian_retail.io_utils import write_json
from bayesian_retail.paths import ProjectPaths
from scripts._common import add_config_arguments


def resolve_run(paths, dataset, seed, run_dir):
    if run_dir:
        return Path(run_dir)
    matches = sorted(
        paths.runs_root.glob(
            f"*_{dataset}_bayesian-hierarchical_seed{seed}*"
        )
    )
    if not matches:
        raise FileNotFoundError("No matching Bayesian run found.")
    return matches[-1]


def main():
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ProjectPaths.from_config(config)
    run_dir = resolve_run(paths, args.dataset, args.seed, args.run_dir)
    idata = load_inference_data(run_dir / "inference_data.nc")
    summary, diagnostics = summarize_inference(
        idata,
        config.inference.monitored_variables,
    )
    summary.to_csv(run_dir / "posterior_summary.csv")
    write_json(diagnostics, run_dir / "diagnostics.json")
    figures = save_diagnostic_figures(
        idata,
        run_dir / "figures",
        config.inference.monitored_variables,
    )
    print(json.dumps({"diagnostics": diagnostics, "figures": figures}, indent=2))


if __name__ == "__main__":
    main()
