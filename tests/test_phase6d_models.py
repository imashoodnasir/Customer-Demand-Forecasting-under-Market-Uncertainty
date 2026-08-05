import torch
from bayesian_retail.baselines.itransformer.model import iTransformer
from bayesian_retail.baselines.hint.model import HINT
from bayesian_retail.baselines.gbpf.model import GBPF
from bayesian_retail.baselines.timemoe.model import TimeMoE

def test_models():
    x=torch.randn(2,32,5)
    assert iTransformer(5)(x).shape[0]==2
    assert HINT(5)(x).shape[0]==2
    m,s=GBPF(5)(x)
    assert m.shape[0]==2
    assert TimeMoE(5)(x).shape[0]==2
