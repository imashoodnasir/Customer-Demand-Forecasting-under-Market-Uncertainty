import torch
from bayesian_retail.baselines.losses import negative_binomial_nll, masked_mse

def test_losses():
    y=torch.ones(2,3); mu=torch.ones(2,3)*2; a=torch.ones(2,3)
    assert torch.isfinite(negative_binomial_nll(y,mu,a)); assert masked_mse(mu,y)>0
