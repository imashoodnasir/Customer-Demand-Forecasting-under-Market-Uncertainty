import torch
from bayesian_retail.baselines.deepar import DeepARNetwork

def test_deepar_shapes():
    m=DeepARNetwork(10,3,2,2,5,hidden_size=8,num_layers=1,dropout=0,embedding_size=4)
    mu,a=m(torch.ones(4,10),torch.ones(4,10,2),torch.ones(4,3,2),torch.zeros(4,dtype=torch.long))
    assert mu.shape==(4,3) and a.shape==(4,3) and (mu>0).all()
