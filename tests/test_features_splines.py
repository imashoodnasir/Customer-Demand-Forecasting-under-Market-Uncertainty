import pandas as pd

from bayesian_retail.features.splines import fit_transform_splines


def test_spline_fit_transform():
    frame = pd.DataFrame({
        "split": ["train"] * 8 + ["test"] * 2,
        "z_price": [float(i) for i in range(10)],
    })
    out, names, state = fit_transform_splines(
        frame,
        columns=["price"],
        degree=3,
        knots=4,
        fit=True,
    )
    assert names
    assert "price" in state
    assert out[names].shape[0] == 10
