import torch
from bayesian_retail.baselines.nbeats import NBeatsNetwork

def test_nbeats_shapes():
    m=NBeatsNetwork(10,3,['trend','seasonality','generic'],1,16,2,8,0)
    y=m(torch.ones(4,10)); assert y.shape==(4,3) and (y>=0).all()
