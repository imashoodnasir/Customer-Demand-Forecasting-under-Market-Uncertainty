import numpy as np

from bayesian_retail.inference.posterior import (
    summarize_posterior_predictive,
)


def test_predictive_summary_shapes():
    samples = np.arange(5 * 3 * 2).reshape(5, 3, 2)
    summary = summarize_posterior_predictive(
        samples,
        [0.5, 0.8, 0.95],
    )
    assert summary["mean"].shape == (3, 2)
    assert summary["lower_0.95"].shape == (3, 2)
