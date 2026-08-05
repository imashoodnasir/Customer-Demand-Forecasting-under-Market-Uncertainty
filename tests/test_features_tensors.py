import numpy as np
from bayesian_retail.features.tensors import validate_tensor_arrays


def test_tensor_validation():
    arrays = {
        "history_target": np.zeros((2, 5), dtype=np.float32),
        "targets": np.zeros((2, 3), dtype=np.float32),
    }
    metadata = {"history_length": 5, "forecast_horizon": 3}
    validate_tensor_arrays(arrays, metadata)
