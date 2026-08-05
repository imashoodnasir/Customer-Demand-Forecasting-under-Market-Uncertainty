import pytest
from bayesian_retail.config import load_config
from bayesian_retail.exceptions import ConfigurationError

def test_default():
    assert load_config("configs/default.yaml").data.forecast_horizon == 28

def test_override():
    cfg = load_config("configs/default.yaml", ["inference.draws=3000", "data.max_series=null"])
    assert cfg.inference.draws == 3000 and cfg.data.max_series is None

def test_bad_override():
    with pytest.raises(ConfigurationError):
        load_config("configs/default.yaml", ["missing.value=1"])

def test_inheritance():
    assert load_config("configs/manuscript.yaml").inference.draws == 3000
