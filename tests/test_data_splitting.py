import pandas as pd

from bayesian_retail.data.splitting import chronological_split


def test_chronological_split():
    frame = pd.DataFrame({
        "series_id": ["s1"] * 10,
        "date": pd.date_range("2024-01-01", periods=10),
        "demand": range(10),
    })
    out = chronological_split(frame, 0.7, 0.1)
    assert (out.iloc[:7]["split"] == "train").all()
    assert out.iloc[7]["split"] == "validation"
    assert (out.iloc[8:]["split"] == "test").all()
