import numpy as np

from bayesian_retail.models.priors import TransferPrior


def test_transfer_prior_defaults_and_values():
    prior = TransferPrior({
        "beta": {
            "mean": np.array([1.0, 2.0]),
            "std": np.array([0.1, 0.2]),
        }
    })
    mean, std = prior.get("beta", 0.0, 1.0)
    assert mean.tolist() == [1.0, 2.0]
    assert std.tolist() == [0.1, 0.2]

    mean, std = prior.get("missing", 0.0, 1.0)
    assert float(mean) == 0.0
    assert float(std) == 1.0
