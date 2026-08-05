from pathlib import Path

from bayesian_retail.config import load_config
from bayesian_retail.data.synthetic import save_synthetic_raw
from bayesian_retail.data.pipeline import preprocess_dataset


def test_synthetic_pipeline(tmp_path):
    overrides = [
        f"project.output_root={repr(str(tmp_path / 'outputs'))}",
        f"project.data_root={repr(str(tmp_path / 'data'))}",
        f"data.raw_root={repr(str(tmp_path / 'data/raw'))}",
        f"data.interim_root={repr(str(tmp_path / 'data/interim'))}",
        f"data.processed_root={repr(str(tmp_path / 'data/processed'))}",
        "data.max_series=6",
    ]
    config = load_config("configs/default.yaml", overrides)
    save_synthetic_raw(
        tmp_path / "data/raw/synthetic",
        n_series=6,
        n_days=120,
        seed=11,
    )
    bundle = preprocess_dataset(config, "synthetic", root=tmp_path)

    assert bundle.metadata["hierarchy_counts"]["series"] == 6
    assert "z_price" in bundle.observations
    assert set(bundle.observations["split"].unique()) == {
        "train",
        "validation",
        "test",
    }
