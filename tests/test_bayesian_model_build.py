import numpy as np

from bayesian_retail.config import load_config
from bayesian_retail.features.tensors import TensorBundle
from bayesian_retail.models.hierarchical import (
    build_bayesian_hierarchical_model,
)
from bayesian_retail.models.spec import BayesianModelSpec


def test_model_builds():
    cfg = load_config(
        "configs/default.yaml",
        overrides=[
            "data.forecast_horizon=3",
            "data.history_length.synthetic=5",
        ],
    )
    arrays = {
        "history_target": np.ones((4, 5), dtype=np.float32),
        "history_observed_mask": np.ones((4, 5), dtype=np.float32),
        "past_covariates": np.ones((4, 5, 2), dtype=np.float32),
        "future_covariates": np.ones((4, 3, 2), dtype=np.float32),
        "targets": np.ones((4, 3), dtype=np.float32),
        "target_observed_mask": np.ones((4, 3), dtype=np.float32),
        "hierarchy_indices": np.zeros((4, 7), dtype=np.int32),
        "series_index": np.zeros((4,), dtype=np.int32),
        "forecast_start_time": np.zeros((4,), dtype=np.int32),
        "forecast_start_date_ordinal": np.zeros((4,), dtype=np.int32),
    }
    bundle = TensorBundle(arrays=arrays, metadata={})
    spec = BayesianModelSpec(
        dataset="synthetic",
        split="train",
        sample_count=4,
        history_length=5,
        forecast_horizon=3,
        past_feature_count=2,
        future_feature_count=2,
        hierarchy_columns=[
            "item_id_idx",
            "department_id_idx",
            "category_id_idx",
            "store_id_idx",
            "region_id_idx",
            "dataset_idx",
            "category_store_idx",
        ],
        hierarchy_cardinalities={
            "item_id_idx": 1,
            "department_id_idx": 1,
            "category_id_idx": 1,
            "store_id_idx": 1,
            "region_id_idx": 1,
            "dataset_idx": 1,
            "category_store_idx": 1,
            "series_index": 1,
        },
        past_features=["p1", "p2"],
        future_features=["f1", "f2"],
    )
    model = build_bayesian_hierarchical_model(
        bundle,
        spec,
        cfg,
    )
    assert "y_obs" in model.named_vars
    assert "mu" in model.named_vars
