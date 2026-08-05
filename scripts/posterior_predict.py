from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_retail.config import load_config
from bayesian_retail.experiments.bayesian import load_tensor_bundle
from bayesian_retail.features.tensors import TensorBundle
from bayesian_retail.inference.engine import load_inference_data
from bayesian_retail.inference.posterior import (
    extract_predictive_samples,
    sample_posterior_predictive,
    save_predictive_npz,
    summarize_posterior_predictive,
    summaries_to_long_frame,
)
from bayesian_retail.models.hierarchical import (
    build_bayesian_hierarchical_model,
)
from bayesian_retail.models.spec import build_model_spec
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
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="test",
    )
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ProjectPaths.from_config(config)
    run_dir = resolve_run(paths, args.dataset, args.seed, args.run_dir)
    processed = paths.processed_root / args.dataset
    tensors = load_tensor_bundle(processed, args.split)
    spec = build_model_spec(
        args.dataset,
        args.split,
        processed,
        tensors,
    )
    model = build_bayesian_hierarchical_model(
        tensors=tensors,
        spec=spec,
        config=config,
    )
    idata = load_inference_data(run_dir / "inference_data.nc")
    predictive = sample_posterior_predictive(
        model,
        idata,
        seed=args.seed,
        predictions=True,
    )
    samples = extract_predictive_samples(predictive)
    summaries = summarize_posterior_predictive(
        samples,
        config.experiment.interval_levels,
    )
    save_predictive_npz(
        samples,
        summaries,
        run_dir / f"posterior_predictive_{args.split}.npz",
    )
    frame = summaries_to_long_frame(
        summaries,
        tensors.arrays["series_index"],
        tensors.arrays["forecast_start_time"],
    )
    frame.to_csv(
        run_dir / f"posterior_predictive_{args.split}_summary.csv",
        index=False,
    )
    print(run_dir / f"posterior_predictive_{args.split}.npz")


if __name__ == "__main__":
    main()
