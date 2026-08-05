import numpy as np
import pandas as pd

from bayesian_retail.features.bundle import FeatureBundle
from bayesian_retail.features.windows import create_window_bundle


def test_window_shapes():
    n = 30
    frame = pd.DataFrame({
        "series_id_idx": [0] * n,
        "series_id": ["s1"] * n,
        "date": pd.date_range("2024-01-01", periods=n),
        "demand": np.arange(n, dtype=float),
        "split": ["train"] * 20 + ["test"] * 10,
        "time_idx": np.arange(n),
        "item_id_idx": [0] * n,
        "department_id_idx": [0] * n,
        "category_id_idx": [0] * n,
        "store_id_idx": [0] * n,
        "region_id_idx": [0] * n,
        "dataset_idx": [0] * n,
        "category_store_idx": [0] * n,
        "x": np.arange(n, dtype=float),
        "f": np.arange(n, dtype=float),
    })
    bundle = FeatureBundle(
        frame=frame,
        metadata={
            "past_features": ["x"],
            "future_features": ["f"],
        },
        state={},
    )
    windows = create_window_bundle(
        bundle,
        split="test",
        history_length=5,
        forecast_horizon=3,
    )
    assert windows.arrays["history_target"].shape[1] == 5
    assert windows.arrays["targets"].shape[1] == 3
    assert windows.arrays["past_covariates"].shape[2] == 1
