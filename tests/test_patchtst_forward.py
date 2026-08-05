import torch
from bayesian_retail.baselines.patchtst.model import PatchTST

def test_patchtst_forward():
    model=PatchTST(channels=4,horizon=7)
    x=torch.randn(2,64,4)
    y=model(x)
    assert y.shape==(2,7)
