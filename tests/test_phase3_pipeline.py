from bayesian_retail.config import load_config
from bayesian_retail.data.synthetic import save_synthetic_raw
from bayesian_retail.data.pipeline import preprocess_dataset
from bayesian_retail.features.pipeline import build_feature_table
from bayesian_retail.features.windows import create_all_window_bundles
from bayesian_retail.features.tensors import build_all_tensor_bundles
from bayesian_retail.paths import ProjectPaths


def test_phase3_pipeline(tmp_path):
    overrides = [
        f"project.output_root={repr(str(tmp_path / 'outputs'))}",
        f"project.data_root={repr(str(tmp_path / 'data'))}",
        f"data.raw_root={repr(str(tmp_path / 'data/raw'))}",
        f"data.interim_root={repr(str(tmp_path / 'data/interim'))}",
        f"data.processed_root={repr(str(tmp_path / 'data/processed'))}",
        "data.max_series=4",
        "data.history_length.synthetic=30",
        "data.forecast_horizon=7",
        "features.explicit_lags=[1,7,14]",
        "features.rolling_windows=[7,14]",
        "features.ewm_spans=[7]",
    ]
    config = load_config("configs/default.yaml", overrides)
    paths = ProjectPaths.from_config(config, root=tmp_path)
    save_synthetic_raw(
        paths.raw_root / "synthetic",
        n_series=4,
        n_days=180,
        seed=11,
    )

    preprocess_dataset(config, "synthetic", root=tmp_path)
    features = build_feature_table(config, "synthetic", root=tmp_path)

    assert features.metadata["past_features"]
    assert "lag_7" in features.frame

    base = paths.processed_root / "synthetic"
    windows = create_all_window_bundles(
        features,
        base / "windows",
        history_length=30,
        forecast_horizon=7,
        stride=1,
    )
    tensors = build_all_tensor_bundles(
        base / "windows",
        base / "tensors",
    )

    assert windows["train"].arrays["targets"].shape[1] == 7
    assert tensors["test"].arrays["history_target"].shape[1] == 30
