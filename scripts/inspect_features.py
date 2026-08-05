from __future__ import annotations
import argparse
import json
import numpy as np

from bayesian_retail.config import load_config
from bayesian_retail.features.bundle import FeatureBundle
from bayesian_retail.io_utils import read_json
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
    parser.add_argument("--head", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    paths = ProjectPaths.from_config(config)
    base = paths.processed_root / args.dataset

    bundle = FeatureBundle.load(base)
    print(json.dumps(bundle.metadata, indent=2))
    columns = [
        "series_id",
        "date",
        "demand",
        "split",
    ] + bundle.metadata["past_features"][:10]
    print("\nFeature table:")
    print(bundle.frame[columns].head(args.head).to_string(index=False))

    metadata_path = base / "tensors" / "tensor_metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        print("\nTensor metadata:")
        print(json.dumps(metadata, indent=2))
        for split in ["train", "validation", "test"]:
            path = base / "tensors" / f"{split}.npz"
            if path.exists():
                data = np.load(path)
                print(f"\n{split}:")
                for key in data.files:
                    print(f"  {key}: {data[key].shape} {data[key].dtype}")


if __name__ == "__main__":
    main()
