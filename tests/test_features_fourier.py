import numpy as np
import pandas as pd

from bayesian_retail.features.fourier import add_fourier_features


def test_fourier_features():
    frame = pd.DataFrame({"time_idx": np.arange(10)})
    out, names = add_fourier_features(frame, [7], 2)
    assert len(names) == 4
    assert all(name in out for name in names)
    assert np.isfinite(out[names].to_numpy()).all()
