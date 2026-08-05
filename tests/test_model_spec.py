import numpy as np

from bayesian_retail.features.tensors import TensorBundle
from bayesian_retail.models.spec import BayesianModelSpec


def test_spec_dataclass():
    spec = BayesianModelSpec(
        dataset="synthetic",
        split="train",
        sample_count=10,
        history_length=30,
        forecast_horizon=7,
        past_feature_count=4,
        future_feature_count=3,
        hierarchy_columns=["item_id_idx"],
        hierarchy_cardinalities={"item_id_idx": 2, "series_index": 2},
        past_features=["a", "b", "c", "d"],
        future_features=["f1", "f2", "f3"],
    )
    assert spec.to_dict()["forecast_horizon"] == 7
