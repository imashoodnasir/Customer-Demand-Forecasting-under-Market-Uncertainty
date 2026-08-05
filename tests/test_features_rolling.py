import pandas as pd

from bayesian_retail.features.rolling import (
    add_ewm_features,
    add_rolling_features,
)


def frame():
    return pd.DataFrame({
        "series_id_idx": [0] * 10,
        "date": pd.date_range("2024-01-01", periods=10),
        "demand": range(10),
    })


def test_rolling_is_shifted():
    out, _ = add_rolling_features(frame(), [3])
    assert out.loc[3, "rolling_mean_3"] == 1.0


def test_ewm():
    out, names = add_ewm_features(frame(), [3])
    assert names == ["ewm_mean_3", "ewm_std_3"]
    assert out["ewm_mean_3"].notna().sum() > 0
