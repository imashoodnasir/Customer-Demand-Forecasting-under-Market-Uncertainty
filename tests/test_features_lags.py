import pandas as pd

from bayesian_retail.features.lags import (
    add_lag_features,
    resolved_lags,
)


def test_resolved_lags():
    assert resolved_lags(3, [1, 7]) == [1, 2, 3, 7]


def test_lag_features():
    frame = pd.DataFrame({
        "series_id_idx": [0] * 5,
        "date": pd.date_range("2024-01-01", periods=5),
        "demand": [1, 2, 3, 4, 5],
    })
    out, names = add_lag_features(frame, [1, 2])
    assert names == ["lag_1", "lag_2"]
    assert out.loc[2, "lag_1"] == 2
    assert out.loc[2, "lag_2"] == 1
