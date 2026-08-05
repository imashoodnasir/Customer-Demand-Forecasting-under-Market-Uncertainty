import torch
from bayesian_retail.baselines.tft.model import TemporalFusionTransformer

def test_forward():
    model=TemporalFusionTransformer(8)
    x=torch.randn(4,20,8)
    y,info=model(x)
    assert y.shape==(4,3)
    assert "attention" in info
