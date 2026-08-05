from __future__ import annotations

import argparse
from pathlib import Path

from bayesian_retail.inference.engine import load_inference_data
from bayesian_retail.inference.transfer import (
    posterior_moments,
    save_posterior_moments,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    idata = load_inference_data(run_dir / "inference_data.nc")
    moments = posterior_moments(idata)
    save_posterior_moments(moments, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
